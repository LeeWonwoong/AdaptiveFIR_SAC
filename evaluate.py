"""
evaluate.py — online-play validation for the paper
====================================================
Runs every method over FULL held-out trajectories with a FIXED measurement-
noise seed (all methods see identical measurement sequences — fair comparison):

  EKF | UKF | FME(N grid, lam=1) | DI-FME(N=14, gamma=0.3) | Rule-FME
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
        self.z_noisy = z_rng
        self.meas_sig = ms

    def run(self, flt, policy=None, oracle=False, agent_env_stats=None):
        cfg, ds, M = self.cfg, self.ds, self.M
        T = ds.T
        s0 = ds.gt[:, 0] + cfg.init_pos_noise * torch.randn(
            M, cfg.state_dim, device=self.dev)
        flt.reset(torch.arange(M, device=self.dev), s0)
        errs = torch.zeros(M, T, device=self.dev)
        Ns = torch.zeros(M, T, device=self.dev)
        Ls = torch.zeros(M, T, device=self.dev)
        feat = cfg.meas_dim + 2
        stack = torch.zeros(M, cfg.L_obs, feat, device=self.dev)   # [M,L,feat]
        defN = torch.full((M,), float(cfg.N_default), device=self.dev)
        defL = torch.full((M,), float(cfg.lam_default), device=self.dev)

        def _nN(N):
            return 2.0 * (N - cfg.N_min) / max(cfg.N_max - cfg.N_min, 1e-6) - 1.0

        def _nL(lm):
            return 2.0 * (lm - cfg.lam_min) / max(1.0 - cfg.lam_min, 1e-6) - 1.0

        def _push(nu, N, lm):
            r = torch.nan_to_num((nu / self.meas_sig).clamp(
                -cfg.resid_clip, cfg.resid_clip), nan=0.0)
            return torch.cat([r, _nN(N).unsqueeze(1), _nL(lm).unsqueeze(1)], dim=1)
        combos = None
        if oracle:
            Ng = torch.tensor([8., 12., 16., 20.], device=self.dev)
            Lg = torch.tensor([0.7, 0.85, 1.0], device=self.dev)
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
                lam = (0.5 * (1 + cfg.lam_min) +
                       0.5 * (1 - cfg.lam_min) * a[:, 1]).clamp(cfg.lam_min, 1.0)
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
            errs[:, t] = torch.linalg.vector_norm(
                ds.gt[:, t, 0:3] - s_hat[:, 0:3], dim=1)
            Ns[:, t] = N
            Ls[:, t] = lam
        return errs.cpu().numpy(), Ns.cpu().numpy(), Ls.cpu().numpy()


# ────────────────────────────────────────────── metrics
def metrics(cfg, ds, errs, Ns):
    dt = cfg.dt
    # skip the growing-window ramp of ALL methods (handover starts at N_min,
    # but fixed-N baselines only reach their full window at N_max epochs)
    W = max(cfg.warmup_steps, cfg.N_max) * max(1, cfg.uwb_stride)
    out = {"rmse": float(np.sqrt((errs[:, W:] ** 2).mean())),
           "max_err": float(errs[:, W:].max()),
           "mean_N": float(Ns[:, W:].mean())}
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
    run = Runner(cfg, ds, dev)
    skip = set(s.strip() for s in a.skip.split(",") if s.strip())
    ev_dir = os.path.join(cfg.outdir, "eval")
    os.makedirs(ev_dir, exist_ok=True)

    methods = {}
    if "ekf" not in skip:
        methods["EKF"] = dict(flt=EKF(cfg, dev, M))
    if "ukf" not in skip:
        methods["UKF"] = dict(flt=UKF(cfg, dev, M))
    if "fme" not in skip:
        for N in (8, 14, 20):
            methods[f"FME-N{N}"] = dict(flt=FixedFME(cfg, dev, M, N=N, lam=1.0))
    if "difme" not in skip:
        methods["DI-FME"] = dict(flt=DIFME(cfg, dev, M))
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
        errs, Ns, Ls = run.run(kw["flt"], kw.get("policy"), kw.get("oracle", False))
        mt = metrics(cfg, ds, errs, Ns)
        rows.append((name, mt))
        curves[name] = (errs, Ns, Ls)
        print(f"{name:14s} rmse {mt['rmse']:.4f} | disturb {mt['rmse_disturb']:.4f} | "
              f"max {mt['max_err']:.3f} | rec {mt['recovery_s']:.2f}s | "
              f"N {mt['mean_N']:.1f}", flush=True)
        np.savez_compressed(os.path.join(ev_dir, f"curve_{name}.npz"),
                            errs=errs, N=Ns, lam=Ls)

    with open(os.path.join(ev_dir, "summary.csv"), "w") as f:
        f.write("method,rmse,rmse_disturb,max_err,recovery_s,mean_N\n")
        for name, mt in rows:
            f.write(f"{name},{mt['rmse']:.5f},{mt['rmse_disturb']:.5f},"
                    f"{mt['max_err']:.4f},{mt['recovery_s']:.3f},{mt['mean_N']:.2f}\n")
    print("wrote", os.path.join(ev_dir, "summary.csv"))

    # ── money figure: pick the held-out traj with the largest disturbance ──
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pick = 0
        for i, m in enumerate(ds.metas):
            if m.get("scenario", {}).get("type") in ("mass_step", "mixed", "gust"):
                pick = i
                break
        t = np.arange(ds.T) * cfg.dt
        sc = ds.metas[pick].get("scenario", {})
        fig, ax = plt.subplots(3, 1, figsize=(9, 7), sharex=True,
                               gridspec_kw={"height_ratios": [2, 1, 1]})
        for (t0, t1, lab) in disturbance_intervals(sc):
            for A in ax:
                A.axvspan(t0, min(t1, t[-1]), color="orange", alpha=0.15)
        show = [k for k in ("DI-FME", "FME-N14", "EKF", "SAC-AFME", "Greedy-GT")
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
