"""
tools/adapt_signal.py — IIR < FIR < AFIR measurement (DI-FME Table I pattern)
=============================================================================
Question: does nominal give IIR≈FIR (tied) while each disturbance WINDOW gives
IIR < FIR < AFIR (FIR beats the recursive filter, AFIR beats fixed FIR)?

Tiers (measured before SAC is trained):
  IIR  = EKF, UKF          (recursive; fixed R = LoS σ, [수정C] — over-trusts NLoS)
  FIR  = best fixed-N FME   (finite window, single N)
  AFIR = Greedy-GT          (per-step optimal (N,λ) on a grid; SAC replaces later)

Scenarios (measurement-side, [수정D]):
  nominal        LoS σ=0.12, no fault
  anchor_dropout one anchor out 3~5 s (range→NaN; EKF/UKF prediction-only rows)
  nlos_burst     one anchor σ↑0.45 + bias, intermittent 2~4 s

Reports, per scenario:
  (1) tiers figure: nominal flat/tied, disturbance window (shaded) IIR spike?
  (2) segment table: nominal-segment vs disturbance-window RMSE per tier
  (3) optimal N split: nominal vs window (N grid widened to N_max=32)
  (4) learnability: per-ANCHOR vector innovation [ν1..ν4] vs optimal N/λ
                    (per-anchor corr + multiple-regression R)

Outputs → results/adapt_signal/.
Usage: python -m tools.adapt_signal [--data_dir data_diag] [--split train]
                                    [--n 24] [--T 1600]
"""
import argparse
import os
import sys
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.dataset import TrajDataset
from filter.baselines import EKF, UKF, FixedFME, DIFME
from filter.wfme import WeightedFME
from datagen.scenario import disturbance_intervals

NG = [8, 11, 14, 17, 20]
LG = [0.70, 0.85, 1.00]
# (scenario type, window label or None, post-window margin [s])
SCEN = [
    ("nominal",          None,               0.0),
    ("turbulence_burst", "turbulence_burst", 0.3),
    ("nlos_burst",       "nlos_burst",       0.3),
    ("anchor_dropout",   "anchor_dropout",   0.3),
]


# ───────────────────────────── masks
def type_trajs(ds, stype):
    return [i for i, m in enumerate(ds.metas)
            if m.get("scenario", {}).get("type") == stype]


def window_mask(cfg, ds, label, margin_s):
    dt, T = cfg.dt, ds.T
    mg = int(margin_s / dt)
    m = np.zeros((ds.n, T), bool)
    for i, meta in enumerate(ds.metas):
        for (t0, t1, lab) in disturbance_intervals(meta.get("scenario", {})):
            if lab == label:
                m[i, int(t0 / dt):min(int(t1 / dt) + mg, T)] = True
    m[:, :cfg.N_max] = False
    return m


def nominal_mask(cfg, ds):
    dt, T = cfg.dt, ds.T
    nom = np.ones((ds.n, T), bool)
    for i, meta in enumerate(ds.metas):
        for (t0, t1, _) in disturbance_intervals(meta.get("scenario", {})):
            nom[i, int(t0 / dt):min(int(t1 / dt), T)] = False
    nom[:, :cfg.N_max] = False
    return nom


# ───────────────────────────── filters
def _reset(flt, cfg, ds, dev):
    flt.reset(torch.arange(ds.n, device=dev),
              ds.gt[:, 0] + cfg.init_pos_noise * torch.randn(ds.n, 12, device=dev))


def run_recursive(cfg, ds, z, dev, cls, **kw):
    flt = cls(cfg, dev, ds.n, **kw); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def run_fixed(cfg, ds, N, lam, z, dev):
    flt = FixedFME(cfg, dev, ds.n, N=N, lam=lam); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def run_seq(cfg, ds, z, Nseq, Lseq, dev):
    """WFME with per-(traj,step) (N,λ) — used for the regime-oracle (switch the
    fixed per-regime optimum by GT label)."""
    flt = WeightedFME(cfg, dev, ds.n); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t], Nseq[:, t], Lseq[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def run_greedy(cfg, ds, z, dev):
    M = ds.n
    flt = WeightedFME(cfg, dev, M); _reset(flt, cfg, ds, dev)
    errs = np.zeros((M, ds.T)); chN = np.full((M, ds.T), np.nan)
    chL = np.full((M, ds.T), np.nan); nu_all = np.full((M, ds.T, 4), np.nan)
    defN = torch.full((M,), 20.0, device=dev); defL = torch.full((M,), 1.0, device=dev)
    warm = cfg.warmup_steps
    for t in range(1, ds.T):
        s, nu, _ = flt.step(ds.u[:, t - 1], z[:, t], defN, defL)
        if nu is not None:
            nu_all[:, t] = nu.cpu().numpy()
        if t > warm:
            best = (ds.gt[:, t, 0:3] - s[:, 0:3]).pow(2).sum(1)
            bs = s.clone(); bN = defN.clone(); bL = defL.clone()
            for n in NG:
                for l in LG:
                    sc = flt._solve(torch.full((M,), float(n), device=dev),
                                    torch.full((M,), float(l), device=dev))
                    e = (ds.gt[:, t, 0:3] - sc[:, 0:3]).pow(2).sum(1)
                    b = e < best
                    best = torch.where(b, e, best); bs[b] = sc[b]
                    bN = torch.where(b, torch.full_like(bN, float(n)), bN)
                    bL = torch.where(b, torch.full_like(bL, float(l)), bL)
            flt.s_hat = bs
            errs[:, t] = best.sqrt().cpu().numpy()
            chN[:, t] = bN.cpu().numpy(); chL[:, t] = bL.cpu().numpy()
        else:
            errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs, chN, chL, nu_all


# ───────────────────────────── metrics
def rmse(errs, mask):
    v = errs[mask]; v = v[np.isfinite(v)]
    return float(np.sqrt((v ** 2).mean())) if v.size else float("nan")


def pearson(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    a, b = a[m], b[m]
    if a.size < 3 or a.std() < 1e-9 or b.std() < 1e-9:
        return float("nan"), int(a.size)
    return float(np.corrcoef(a, b)[0, 1]), int(a.size)


def multi_R(X, y):
    """multiple-correlation R between y and its best linear fit on columns of X."""
    m = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[m], y[m]
    if X.shape[0] < 6 or y.std() < 1e-9:
        return float("nan")
    A = np.concatenate([np.ones((X.shape[0], 1)), X], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    yhat = A @ coef
    if yhat.std() < 1e-12:
        return 0.0
    return float(np.corrcoef(yhat, y)[0, 1])


# ───────────────────────────── figures
def tiers_figure(cfg, ds, name, label, rows, ekf, fir, grd, bestN, outdir):
    if not rows:
        return
    ex = rows[0]; dt = cfg.dt; tt = np.arange(ds.T) * dt
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(tt, ekf[ex], color="tab:green", lw=1.0, label="IIR (EKF)")
    ax.plot(tt, fir[ex], color="tab:blue", lw=1.0, label=f"FIR (fixed N={bestN})")
    ax.plot(tt, grd[ex], color="tab:red", lw=1.1, label="AFIR (Greedy)")
    if label:
        for (t0, t1, l) in disturbance_intervals(ds.metas[ex].get("scenario", {})):
            if l == label:
                ax.axvspan(t0, t1, color="orange", alpha=0.18)
    ax.set_ylabel("pos. error [m]"); ax.set_xlabel("t [s]")
    ax.set_title(f"{name}: IIR / FIR / AFIR (traj {ex})")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"tiers_{name}.png"), dpi=110); plt.close(fig)


def signal_figures(cfg, ds, name, label, mwin, mnom, chN, chL, nu_all, outdir):
    nu_abs = np.abs(nu_all)                       # [n,T,4] per-anchor |ν|
    nu_max = np.nanmax(nu_abs, axis=2)
    # hist
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for j, (arr, nm, rg) in enumerate([(chN, "optimal N", (cfg.N_min, cfg.N_max)),
                                       (chL, "optimal λ", (cfg.lam_min, 1.0))]):
        an = arr[mnom]; ad = arr[mwin]
        an = an[np.isfinite(an)]; ad = ad[np.isfinite(ad)]
        bins = np.linspace(rg[0], rg[1], 14)
        if an.size:
            ax[j].hist(an, bins=bins, density=True, alpha=0.5, color="gray",
                       label=f"nominal (μ={an.mean():.1f})")
        if ad.size:
            ax[j].hist(ad, bins=bins, density=True, alpha=0.5, color="tab:red",
                       label=f"window (μ={ad.mean():.1f})")
        ax[j].set_xlabel(nm); ax[j].set_ylabel("density"); ax[j].legend(fontsize=8)
    fig.suptitle(f"{name}: per-step optimal (N, λ) — nominal vs window")
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_hist.png"), dpi=110); plt.close(fig)
    # innovation(max anchor) vs optimal N
    fig, ax = plt.subplots(figsize=(5.4, 4.2))
    ax.scatter(nu_max[mnom], chN[mnom], s=4, alpha=0.15, color="gray", label="nominal")
    ax.scatter(nu_max[mwin], chN[mwin], s=8, alpha=0.35, color="tab:red", label=name)
    r, npt = pearson(nu_max[mwin], chN[mwin])
    ax.set_xlabel("max-anchor |ν| [m]"); ax.set_ylabel("optimal N")
    ax.set_title(f"{name}: max|ν_a| vs N*  (window r={r:+.2f}, n={npt})")
    ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(os.path.join(outdir, f"{name}_innov_vs_N.png"), dpi=110); plt.close(fig)


# ───────────────────────────── main
def _crop(ds, keep, T):
    idx = torch.as_tensor(keep, device=ds.dev)
    for a in ("gt", "u", "m_true", "wind", "range_clean", "noise_scale", "range_bias"):
        setattr(ds, a, getattr(ds, a)[idx, :T])
    ds.metas = [ds.metas[i] for i in keep]
    ds.n, ds.T = len(keep), T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_diag")
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--T", type=int, default=1600)
    ap.add_argument("--outdir", default="results/adapt_signal")
    args = ap.parse_args()

    cfg = Config(); cfg.data_dir = args.data_dir
    dev = "cpu"; torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    keep = []
    for stype, _, _ in SCEN:
        keep += type_trajs(ds, stype)[: max(6, args.n // 3)]
    keep = sorted(set(keep))[:args.n]
    _crop(ds, keep, min(args.T, ds.T))
    print(f"[adapt_signal] split={args.split} {ds.n} trajs x {ds.T} steps  "
          f"counts={ {t: len(type_trajs(ds, t)) for t, _, _ in SCEN} }")

    base = torch.tensor(cfg.meas_sigma, device=dev).view(1, 1, 4)     # LoS σ
    g = torch.Generator(device=dev).manual_seed(1234)
    z = ds.range_clean + ds.range_bias + base * ds.noise_scale * torch.randn(
        ds.n, ds.T, 4, generator=g, device=dev)

    print("  IIR: EKF/UKF (practitioner) + oracle-KF ...")
    ekf = run_recursive(cfg, ds, z, dev, EKF)             # practitioner (mis-tuned)
    ukf = run_recursive(cfg, ds, z, dev, UKF)
    ekf_or = run_recursive(cfg, ds, z, dev, EKF, oracle=True)   # correct stats
    difme = run_recursive(cfg, ds, z, dev, DIFME)         # info-form FIR (prior)
    print("  FIR: fixed (N,λ) grid ...")
    combos = [(N, l) for N in NG for l in LG]
    fixedc = {c: run_fixed(cfg, ds, c[0], c[1], z, dev) for c in combos}
    fir14 = fixedc[(cfg.N_default, 1.0)]                  # DI-FME N=14, λ=1 baseline
    print("  AFIR: greedy oracle (N×λ grid) ...")
    grd, chN, chL, nu_all = run_greedy(cfg, ds, z, dev)
    L = cfg.L_obs
    ne = np.nansum(nu_all ** 2, axis=2)
    nu_var = np.full_like(ne, np.nan)
    for t in range(ne.shape[1]):
        nu_var[:, t] = np.nanmean(ne[:, max(0, t - L + 1):t + 1], axis=1)
    os.makedirs(args.outdir, exist_ok=True)

    nom_all = nominal_mask(cfg, ds)
    # ── per-scenario masks + per-region best fixed combos ──
    rows_table = []
    for name, label, margin in SCEN:
        rows = type_trajs(ds, name)
        if not rows:
            continue
        rset = np.zeros(ds.n, bool); rset[rows] = True
        mnom = nom_all & rset[:, None]
        mwin = (window_mask(cfg, ds, label, margin) & rset[:, None]) if label else None
        r = dict(name=name, label=label, rows=rows, mnom=mnom, mwin=mwin,
                 winmask=(window_mask(cfg, ds, label, margin) if label else None))
        r["combo_nom"] = min(combos, key=lambda c: rmse(fixedc[c], mnom))
        if mwin is not None and mwin.any():
            mall = mnom | mwin
            r["combo_win"] = min(combos, key=lambda c: rmse(fixedc[c], mwin))
            r["combo_all"] = min(combos, key=lambda c: rmse(fixedc[c], mall))  # best global fixed
        rows_table.append(r)

    # ── regime-oracle: switch each scenario's per-region best fixed by GT label ──
    Nseq = torch.full((ds.n, ds.T), float(cfg.N_default), device=dev)
    Lseq = torch.full((ds.n, ds.T), 1.0, device=dev)
    for r in rows_table:
        cn = r["combo_nom"]; cw = r.get("combo_win", cn)
        for i in r["rows"]:
            Nseq[i, :] = cn[0]; Lseq[i, :] = cn[1]
            if r["winmask"] is not None:
                wm = torch.tensor(r["winmask"][i], device=dev)
                Nseq[i, wm] = cw[0]; Lseq[i, wm] = cw[1]
    print("  regime-oracle: GT-label (N,λ) switching ...")
    regime = run_seq(cfg, ds, z, Nseq, Lseq, dev)

    lines = []
    def P(s=""): lines.append(s); print(s)
    P("\n" + "=" * 82)
    P("IIR<FIR<AFIR + learnability decomposition (data=%s/%s, σ_LoS=%.2f, N∈[%d,%d])"
      % (args.data_dir, args.split, cfg.meas_sigma[0], cfg.N_min, cfg.N_max))
    P("  EKF/UKF = practitioner (R σ=%.2f, Q×%.2f);  EKF-or = oracle stats"
      % (cfg.ekf_R_sigma, cfg.ekf_Q_scale))
    P("=" * 82)

    # ── (2) segment RMSE table ──
    P("\n(2) SEGMENT RMSE [m]")
    P(f"  {'scenario':16s} {'region':7s} {'EKF':>6s} {'EKF-or':>6s} {'UKF':>6s} "
      f"{'FIR14':>6s} {'FIRbest':>7s} {'DI-FME':>6s} {'AFIR':>6s}")
    for r in rows_table:
        cb = r.get("combo_all", r["combo_nom"])
        def row(reg, m):
            P(f"  {r['name'] if reg=='nominal' else '':16s} {reg:7s} "
              f"{rmse(ekf,m):6.3f} {rmse(ekf_or,m):6.3f} {rmse(ukf,m):6.3f} "
              f"{rmse(fir14,m):6.3f} {rmse(fixedc[cb],m):7.3f} {rmse(difme,m):6.3f} "
              f"{rmse(grd,m):6.3f}")
        row("nominal", r["mnom"])
        if r.get("mwin") is not None and r["mwin"].any():
            row("WINDOW", r["mwin"])
    P("  (FIRbest = best global fixed (N,λ) for that scenario: %s)"
      % ", ".join(f"{r['name']}={r.get('combo_all')}" for r in rows_table if 'combo_all' in r))

    # ── (3) LEARNABILITY DECOMPOSITION (window) ──
    P("\n(3) LEARNABILITY DECOMPOSITION — window RMSE [m]  (lower better)")
    P("    best-fixed = one global (N,λ);  regime-oracle = GT-label switch;  greedy = per-step")
    P(f"  {'scenario':16s} {'bestfix':>7s} {'regime':>7s} {'greedy':>7s} | "
      f"{'LEARNABLE%':>10s} {'luck%':>6s}   combos(nom→win)")
    for r in rows_table:
        if r.get("mwin") is None or not r["mwin"].any():
            continue
        m = r["mwin"]
        bf = rmse(fixedc[r["combo_all"]], m)
        ro = rmse(regime, m)
        gr = rmse(grd, m)
        learn = 100 * (bf - ro) / bf if bf > 0 else 0.0
        luck = 100 * (ro - gr) / ro if ro > 0 else 0.0
        P(f"  {r['name']:16s} {bf:7.3f} {ro:7.3f} {gr:7.3f} | {learn:9.1f}% {luck:5.1f}%   "
          f"{r['combo_nom']}→{r['combo_win']}  "
          f"{'← SAC TARGET ≥10%' if learn >= 10 else ''}")

    # ── (4) optimal-N split + learnability corr ──
    P("\n(4) OPTIMAL N split & corr(innovation,N*)")
    for r in rows_table:
        if r.get("mwin") is None or not r["mwin"].any():
            continue
        mnom, mwin = r["mnom"], r["mwin"]
        Nn = chN[mnom]; Nn = Nn[np.isfinite(Nn)]
        Nd = chN[mwin]; Nd = Nd[np.isfinite(Nd)]
        Xw = np.stack([np.abs(nu_all[..., a])[mwin] for a in range(4)], axis=1)
        mR = multi_R(Xw, chN[mwin])
        rVar = pearson(nu_var[mwin | mnom], chN[mwin | mnom])[0]
        P(f"  {r['name']:16s} N_nom={Nn.mean():4.1f}→N_win={Nd.mean():4.1f} "
          f"(ΔN={Nd.mean()-Nn.mean():+4.1f})  multiR(ν→N*)={mR:+.2f} "
          f"corr(energy,N*)={rVar:+.2f}")
        signal_figures(cfg, ds, r["name"], r["label"], mwin, mnom, chN, chL, nu_all, args.outdir)
        fir_fig = difme if r["name"] == "anchor_dropout" else fir14
        tiers_figure(cfg, ds, r["name"], r["label"], r["rows"], ekf, fir_fig, grd,
                     "DI" if r["name"] == "anchor_dropout" else cfg.N_default, args.outdir)

    # ── (5) verdict ──
    P("\n(5) VERDICT  (nominal EKF≈FIR?  window IIR<FIR<AFIR?)")
    for r in rows_table:
        n_ekf = rmse(ekf, r["mnom"]); n_fir = min(rmse(fir14, r["mnom"]), rmse(difme, r["mnom"]))
        nom_tied = abs(n_ekf - n_fir) <= 0.15 * max(n_fir, 1e-6)
        line = f"  {r['name']:16s} nominal EKF={n_ekf:.3f} vs FIR={n_fir:.3f} " \
               f"→ {'TIED' if nom_tied else 'EKF leads'}"
        if r.get("mwin") is not None and r["mwin"].any():
            m = r["mwin"]
            w_ekf = rmse(ekf, m); w_fir = min(rmse(fir14, m), rmse(difme, m)); w_afir = rmse(grd, m)
            chain = (w_ekf > w_fir) and (w_afir <= w_fir + 1e-6)
            line += f" | window EKF={w_ekf:.3f}>FIR={w_fir:.3f}>AFIR={w_afir:.3f}? " \
                    f"{'IIR<FIR<AFIR' if chain else ('IIR<FIR' if w_ekf>w_fir else 'EKF leads')}"
        P(line)
    P("\nfigures + table → %s/" % args.outdir)
    with open(os.path.join(args.outdir, "summary_table.txt"), "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()
