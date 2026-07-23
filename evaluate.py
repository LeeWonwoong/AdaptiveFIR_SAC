"""
evaluate.py — online-play validation for the paper
====================================================
Runs every method over FULL held-out trajectories with a FIXED measurement-
noise seed (all methods see identical measurement sequences — fair comparison):

  EKF | UKF | FME(N grid, lam=1) | FIR(N=14, gamma=0.3) | Rule-FME
  | SAC-AFME (trained actor, deterministic — GT-free at deployment)
  | Greedy-GT (per-step argmin of GT error over a (N,lam) grid — a near-
    upper-bound reference when the closed-loop coupling is benign (fully
    observable sensor suite), but NOT a guaranteed bound: the myopically-best
    commit feeds the next linearization. The gap between the best fixed
    setting and Greedy-GT is the headroom adaptive policies can claim.)

Outputs (results/<outdir>/eval/):
  summary.csv           overall + disturbance-window RMSE, max err, recovery
  money_fig.png         disturbance shading + error curves + N_t, lam_t
  per_method curves .npz
Usage:
  python evaluate.py --outdir results/run0 [--ckpt results/run0/ckpt.pt]
"""
import argparse
import json
import os
import numpy as np
import torch

from config import Config, parse_cli
from rlenv.dataset import TrajDataset
from filter.wfme import WeightedFME
from filter.baselines import EKF, UKF, FixedFME, DIFME, RuleFME
from rl.sac import SACAgent
from datagen.scenario import disturbance_intervals
# tools/_common.py is the SINGLE SOURCE OF TRUTH for baseline (EKF/UKF/FME)
# tuning; import the builders so this online-play run uses exactly the same
# constants as every tools/ analysis script.
from tools._common import make_ekf, make_ukf, make_fme


# ────────────────────────────────────────────── rollout engine
class Runner:
    """Streams a whole held-out split (M = n_heldout trajectories in batch)."""

    def __init__(self, cfg, ds, device, noise_seed=1234):
        self.cfg, self.ds, self.dev = cfg, ds, device
        self.M = ds.n
        g = torch.Generator(device=device)
        g.manual_seed(noise_seed)
        lo, hi = cfg.uwb_sigma_range
        sig_u = lo + (hi - lo) * torch.rand(self.M, 1, 1, generator=g, device=device)
        ms = torch.tensor(cfg.meas_sigma, device=device)
        n_rng = 4
        n_rng = ds.range_clean.shape[2]
        # pre-draw the ENTIRE noise tensor once → identical for every method
        nscale = ds.noise_scale if hasattr(ds, "noise_scale") else 1.0    # [M,T,4]
        rbias = ds.range_bias if hasattr(ds, "range_bias") else 0.0       # [M,T,4]
        z_rng = ds.range_clean + rbias + sig_u * nscale * torch.randn(
            self.M, ds.T, n_rng, generator=g, device=device)
        # IMU rows (attitude + gyro) — the 10-D FUSED measurement, identical
        # synthesis to rlenv.replay_env._measure (was UWB-only legacy: the
        # 4-D z crashed every 10-D filter with a dim-mismatch).
        z_att = ds.gt[:, :, 6:9] + ms[n_rng] * torch.randn(
            self.M, ds.T, 3, generator=g, device=device)
        z_gyr = ds.gt[:, :, 9:12] + ms[n_rng + 3] * torch.randn(
            self.M, ds.T, 3, generator=g, device=device)
        self.z_noisy = torch.cat([z_rng, z_att, z_gyr], dim=2)     # [M,T,10]
        self.meas_sig = ms

    def run(self, flt, policy=None, oracle=False, agent_env_stats=None):
        cfg, ds, M = self.cfg, self.ds, self.M
        T = ds.T
        s0 = ds.gt[:, 0] + cfg.init_pos_noise * torch.randn(
            M, cfg.state_dim, device=self.dev)
        flt.reset(torch.arange(M, device=self.dev), s0)
        errs = torch.zeros(M, T, device=self.dev)
        evec = torch.zeros(M, T, 3, device=self.dev)   # per-axis error
        Ns = torch.zeros(M, T, device=self.dev)
        Ls = torch.zeros(M, T, device=self.dev)
        feat = (cfg.n_obs_groups + cfg._n_par_feat) if cfg.obs_channel_norms \
            else (cfg.meas_dim + cfg._n_par_feat)
        stack = torch.zeros(M, cfg.L_obs, feat, device=self.dev)   # [M,L,feat]
        grp_scale = torch.tensor(cfg.obs_group_scale[:cfg.n_obs_groups],
                                 device=self.dev).view(1, -1)
        defN = torch.full((M,), float(cfg.N_default), device=self.dev)
        defL = torch.full((M,), float(cfg.lam_default), device=self.dev)

        def _nN(N):
            return 2.0 * (N - cfg.N_min) / max(cfg.N_max - cfg.N_min, 1e-6) - 1.0

        def _nL(lm):
            if getattr(cfg, "lam_fixed", -1.0) > 0:
                return torch.zeros_like(lm) if hasattr(lm, "shape") else 0.0
            return 2.0 * (lm - cfg.lam_min) / max(1.0 - cfg.lam_min, 1e-6) - 1.0

        def _push(nu, N, lm):
            # MUST mirror rlenv.replay_env._push_feature EXACTLY — the policy
            # was trained on channel-GROUP whitened norms / nominal scale.
            r = torch.nan_to_num(nu / self.meas_sig, nan=0.0)
            if cfg.obs_channel_norms:
                nr = len(cfg.anchors)
                cols = [r[:, 0:nr].norm(dim=1, keepdim=True) / nr ** 0.5,
                        r[:, nr:nr+3].norm(dim=1, keepdim=True) / 3 ** 0.5]
                if not getattr(cfg, "obs_drop_gyro", False):
                    cols.append(r[:, nr+3:nr+6].norm(dim=1, keepdim=True) / 3 ** 0.5)
                g = torch.cat(cols, dim=1)
                head = (g / grp_scale).clamp(max=cfg.resid_clip)
            else:
                head = r.clamp(-cfg.resid_clip, cfg.resid_clip)
            cols = [head, _nN(N).unsqueeze(1)]
            if getattr(cfg, "lam_fixed", -1.0) <= 0:
                cols.append(_nL(lm).unsqueeze(1))
            return torch.cat(cols, dim=1)
        combos = None
        if oracle:
            Ng = torch.tensor(getattr(cfg, "oracle_N_grid",
                                       [8., 12., 16., 20.]), device=self.dev)
            Lg = torch.tensor(getattr(cfg, "oracle_lam_grid",
                                       [0.7, 0.85, 1.0]), device=self.dev)
            combos = [(n.item(), l.item()) for n in Ng for l in Lg]

        stride = max(1, cfg.uwb_stride)
        warm_t = cfg.warmup_steps * stride               # warmup epochs -> base steps
        N, lam = defN.clone(), defL.clone()
        for t in range(1, T):
            up = ds.u[:, t - 1]
            is_epoch = (t % stride == 0)
            z = self.z_noisy[:, t] if is_epoch else None
            if policy is not None and is_epoch and t > warm_t:
                obs = stack.reshape(M, cfg.L_obs * feat)
                a = policy(obs)
                N = torch.round(0.5 * (cfg.N_max + cfg.N_min) +
                                0.5 * (cfg.N_max - cfg.N_min) * a[:, 0]
                                ).clamp(cfg.N_min, cfg.N_max)
                if getattr(cfg, "lam_fixed", -1.0) > 0:
                    lam = torch.full((a.shape[0],), float(cfg.lam_fixed),
                                     device=a.device)
                else:
                    lam = (0.5 * (1 + cfg.lam_min) +
                           0.5 * (1 - cfg.lam_min) * a[:, 1]).clamp(
                               cfg.lam_min, 1.0)
                s_hat, nu, _ = flt.step(up, z, N, lam)
            elif oracle and is_epoch and t > warm_t:
                # buffers are (N,lam)-independent → push once via a probe step
                # then re-solve per combo and commit the best (greedy, GT access)
                s_hat, nu, s_pred = flt.step(up, z, defN, defL)
                best_err = (ds.gt[:, t, 0:3] - s_hat[:, 0:3]).pow(2).sum(1)
                best_s = s_hat.clone()
                bN = defN.clone(); bL = defL.clone()
                for (n, l) in combos:
                    Nc = torch.full((M,), n, device=self.dev)
                    Lc = torch.full((M,), l, device=self.dev)
                    sc = flt._solve(Nc, Lc)
                    e = (ds.gt[:, t, 0:3] - sc[:, 0:3]).pow(2).sum(1)
                    better = e < best_err
                    best_err = torch.where(better, e, best_err)
                    best_s[better] = sc[better]
                    bN = torch.where(better, Nc, bN)
                    bL = torch.where(better, Lc, bL)
                flt.s_hat = best_s                     # commit greedy choice
                s_hat, N, lam = best_s, bN, bL
            else:
                s_hat, nu, _ = flt.step(up, z, defN, defL)
                if is_epoch:
                    N = getattr(flt, "last_N", defN)  # rule-based exposes its choice
                    lam = getattr(flt, "last_lam", defL)
            if nu is not None:                       # innovation exists on epochs only
                stack = torch.roll(stack, 1, dims=1)
                stack[:, 0] = _push(nu, N, lam)
            evec[:, t] = ds.gt[:, t, 0:3] - s_hat[:, 0:3]
            errs[:, t] = torch.linalg.vector_norm(evec[:, t], dim=1)
            Ns[:, t] = N
            Ls[:, t] = lam
        return errs.cpu().numpy(), Ns.cpu().numpy(), Ls.cpu().numpy(), evec.cpu().numpy()


# ────────────────────────────────────────────── metrics
def metrics(cfg, ds, errs, Ns, evec=None):
    dt = cfg.dt
    # skip the growing-window ramp of ALL methods (handover starts at N_min,
    # but fixed-N baselines only reach their full window at N_max epochs)
    W = max(cfg.warmup_steps, cfg.N_max) * max(1, cfg.uwb_stride)
    # v12 FINAL: close the window at 40 s so the reported RMSE matches exactly
    # what the time-series figures show (2-40 s). The window covers both gust
    # windows (6-16 s, 26-36 s) and the whole payload window (15-33 s) plus the
    # recovery; the remaining 10 s is undisturbed cruise that only dilutes the
    # disturbance statistics.
    W_END = min(int(40.0 / cfg.dt), errs.shape[1])
    S = slice(W, W_END)
    out = {"rmse": float(np.sqrt((errs[:, S] ** 2).mean())),
           "max_err": float(errs[:, S].max()),
           "mean_N": float(Ns[:, S].mean())}
    if evec is not None:            # per-axis (paper-table convention)
        for j, axn in enumerate("xyz"):
            out[f"rmse_{axn}"] = float(np.sqrt((evec[:, S, j] ** 2).mean()))
    # disturbance-window rmse + recovery time
    d_err, rec = [], []
    for i, meta in enumerate(ds.metas):
        sc = meta.get("scenario", {})
        for (t0, t1, lab) in disturbance_intervals(sc):
            k0, k1 = int(t0 / dt), min(int(t1 / dt), ds.T - 1)
            if k1 <= k0:
                continue
            d_err.append(errs[i, k0:k1] ** 2)
            # recovery: first time err < 1.5x nominal after disturbance end
            nom = np.median(errs[i, W:max(k0, W + 1)]) if k0 > W else np.median(errs[i, W:])
            post = errs[i, k1:]
            idx = np.where(post < 1.5 * max(nom, 1e-3))[0]
            rec.append(idx[0] * dt if len(idx) else (ds.T - k1) * dt)
    if d_err:
        out["rmse_disturb"] = float(np.sqrt(np.concatenate(d_err).mean()))
        out["recovery_s"] = float(np.mean(rec))
    else:
        out["rmse_disturb"] = float("nan")
        out["recovery_s"] = float("nan")
    return out


# ────────────────────────────────────────────── main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--skip", default="", help="comma list: ukf,difme,oracle,...")
    a, _ = ap.parse_known_args()
    cfg = parse_cli(Config())
    dev = cfg.resolve_device()
    torch.manual_seed(cfg.seed)
    ds = TrajDataset(cfg, "heldout", dev)
    M = ds.n
    run = Runner(cfg, ds, dev, noise_seed=1234 + int(cfg.seed) * 101)
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())
    ev_dir = os.path.join(cfg.outdir, "eval")
    os.makedirs(ev_dir, exist_ok=True)

    methods = {}
    # Baseline (EKF/UKF/FME) tuning is defined ONCE in tools/_common.py and
    # imported via make_ekf / make_ukf / make_fme, so this online-play run and
    # every tools/ analysis script build the baselines from identical constants.
    # See tools/_common.py for the grid-search provenance of each value (Q_EKF,
    # Q_UKF, R_EKF, R_UKF, UKF_ALPHA/BETA/KAPPA, UKF_R_UWB_SCALE, FME_N/LAM).
    #
    # NOTE: the make_* builders use the NOMINAL levels. tools/_common also
    # defines disturbance-regime (*_DIST) levels, but that nominal/disturbance
    # split is only meaningful for the scenario-separated tools (which run the
    # KFs twice); this batched online-play run scores all held-out trajectories
    # together and therefore uses a single (nominal) regime per filter.
    if "ekf" not in skip:
        methods["EKF"] = dict(flt=make_ekf(cfg, dev, M))
    if "ukf" not in skip:
        methods["UKF"] = dict(flt=make_ukf(cfg, dev, M))
    if "fme" not in skip:
        # primary fixed-horizon FME baseline, tuned in tools/_common.py
        # (FME_N, FME_LAM). The FIR-N6 / FIR-N14 variants below form the
        # N-sensitivity column, holding lambda at the same tuned value.
        methods["FIR"] = dict(flt=make_fme(cfg, dev, M))
        for N in (6, 14):
            methods[f"FIR-N{N}"] = dict(flt=make_fme(cfg, dev, M, N=N))
    if "rule" not in skip:
        methods["Rule-FME"] = dict(flt=RuleFME(cfg, dev, M))
    ckpt = a.ckpt or os.path.join(cfg.outdir, "ckpt.pt")
    if os.path.exists(ckpt) and "sac" not in skip:
        agent = SACAgent(cfg, obs_dim=cfg.obs_dim, device=dev)
        agent.load(ckpt)
        methods["SAC-AFME"] = dict(flt=WeightedFME(cfg, dev, M),
                                   policy=lambda o: agent.act(o, deterministic=True))
    if "oracle" not in skip:
        methods["Greedy-GT"] = dict(flt=WeightedFME(cfg, dev, M), oracle=True)

    rows, curves = [], {}
    for name, kw in methods.items():
        errs, Ns, Ls, evec = run.run(kw["flt"], kw.get("policy"),
                                     kw.get("oracle", False))
        mt = metrics(cfg, ds, errs, Ns, evec)
        rows.append((name, mt))
        curves[name] = (errs, Ns, Ls)
        print(f"{name:14s} rmse {mt['rmse']:.4f} "
              f"(x {mt.get('rmse_x', 0):.3f} y {mt.get('rmse_y', 0):.3f} "
              f"z {mt.get('rmse_z', 0):.3f}) | disturb {mt['rmse_disturb']:.4f} | "
              f"max {mt['max_err']:.3f} | rec {mt['recovery_s']:.2f}s | "
              f"N {mt['mean_N']:.1f}", flush=True)
        np.savez_compressed(os.path.join(ev_dir, f"curve_{name}.npz"),
                            errs=errs, N=Ns, lam=Ls, evec=evec)

    with open(os.path.join(ev_dir, "summary.csv"), "w") as f:
        f.write("method,rmse,rmse_x,rmse_y,rmse_z,rmse_disturb,max_err,recovery_s,mean_N\n")
        for name, mt in rows:
            f.write(f"{name},{mt['rmse']:.5f},{mt.get('rmse_x', 0):.5f},"
                    f"{mt.get('rmse_y', 0):.5f},{mt.get('rmse_z', 0):.5f},"
                    f"{mt['rmse_disturb']:.5f},"
                    f"{mt['max_err']:.4f},{mt['recovery_s']:.3f},{mt['mean_N']:.2f}\n")
    print("wrote", os.path.join(ev_dir, "summary.csv"))

    # ── money figure: pick the held-out traj with the largest disturbance ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = 0
        for i, m in enumerate(ds.metas):
            if m.get("scenario", {}).get("type") in ("sustained_wind", "mass_step", "mixed", "gust"):
                pick = i
                break
        t = np.arange(ds.T) * cfg.dt
        sc = ds.metas[pick].get("scenario", {})
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1, 1]})
        for (t0, t1, lab) in disturbance_intervals(sc):
            for A in ax:
                A.axvspan(t0, min(t1, t[-1]), color="orange", alpha=0.15)
        show = [k for k in ("FIR", "EKF", "SAC-AFME", "Greedy-GT")
                if k in curves]
        for k in show:
            ax[0].plot(t, curves[k][0][pick], label=k, lw=1.1)
        ax[0].set_ylabel("pos. error [m]")
        ax[0].legend(ncol=3, fontsize=8)
        ax[0].grid(alpha=.3)
        if "SAC-AFME" in curves:
            ax[1].plot(t, curves["SAC-AFME"][1][pick], color="tab:red")
            ax[2].plot(t, curves["SAC-AFME"][2][pick], color="tab:purple")
        ax[1].set_ylabel("$N_t$"); ax[1].grid(alpha=.3)
        ax[2].set_ylabel("$\\lambda_t$"); ax[2].set_xlabel("time [s]"); ax[2].grid(alpha=.3)
        ax[0].set_title(f"held-out traj #{pick} ({sc.get('type')}, {sc.get('pattern')})")
        fig.tight_layout()
        fig.savefig(os.path.join(ev_dir, "money_fig.png"), dpi=150)
        print("wrote", os.path.join(ev_dir, "money_fig.png"))
    except Exception as e:
        print("figure skipped:", e)


if __name__ == "__main__":
    main()
