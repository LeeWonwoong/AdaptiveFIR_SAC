"""
tools/cm_test.py — Phase-0 FINAL: tag-side common-mode world
============================================================
Pre-registered last test. Tag-side UWB errors (DW1000 RX-power-dependent
timestamp shift, tag antenna radiation pattern, clock drift) are COMMON to all
anchors and vary time-correlated with attitude. Measurement layer (dataset):
    range[a] += b_common(k)·s_a + b_a(k)
b_common = tag OU switched calm↔dynamic by attitude activity (cm_regime); s_a =
per-anchor sensitivity; b_a = per-anchor NLoS OU. Because it hits EVERY anchor it
is NOT geometrically rejected (the wall that killed single-anchor NLoS).

Four measurements over the tag_commonmode scenario (labels calm / dynamic /
recovery-tail = first --tail steps after each dynamic segment):
  (a) regime-wise fixed-N U-curve  — does the dynamic floor SHIFT left of calm?
  (b) 3-label learnability decomposition — LEARNABLE% (best-fixed→regime-oracle)
  (c) tiers EKF / EKF-oracle / FIR14 / FIRbest / AFIR — IIR<FIR in dynamic?
  (d) corr — COMMON-MODE feature (rolling mean/std of the anchor-MEAN innovation)
      vs N*  (the common component shows up directly here)

PRE-REGISTERED GATE:  GO = LEARNABLE% ≥ 10% AND dynamic IIR<FIR.

Usage: python -m tools.cm_test [--data_dir data_cm] [--split train]
                               [--T 2000] [--tail 25]
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
from filter.baselines import EKF, UKF, FixedFME
from datagen.scenario import disturbance_intervals
from tools.adapt_signal import (nominal_mask, rmse, pearson,
                                run_recursive, run_fixed, run_seq, run_greedy,
                                NG, LG)

LABEL = "tag_commonmode"
NGRID = [8, 10, 12, 14, 16, 18, 20]


def dyn_intervals_steps(cfg, ds):
    dt = cfg.dt
    out = []
    for meta in ds.metas:
        ks = [(int(t0 / dt), min(int(t1 / dt), ds.T))
              for (t0, t1, lab) in disturbance_intervals(meta.get("scenario", {}))
              if lab == LABEL]
        out.append(sorted(ks))
    return out


def make_masks(cfg, ds, margin_s, tail_steps):
    T = ds.T; mg = int(margin_s / cfg.dt)
    dyn = np.zeros((ds.n, T), bool); tail = np.zeros((ds.n, T), bool)
    ivs = dyn_intervals_steps(cfg, ds)
    for i in range(ds.n):
        for (k0, k1) in ivs[i]:
            dyn[i, k0:min(k1 + mg, T)] = True
        for (k0, k1) in ivs[i]:
            t0 = min(k1 + mg, T); tail[i, t0:min(t0 + tail_steps, T)] = True
    tail &= ~dyn
    calm = ~dyn & ~tail
    for m in (calm, dyn, tail):
        m[:, :cfg.N_max] = False
    return {"calm": calm, "dynamic": dyn, "recovery-tail": tail}


def run_fixed_err(cfg, ds, N, z, dev):
    from tools.adapt_signal import _reset
    flt = FixedFME(cfg, dev, ds.n, N=N, lam=1.0); _reset(flt, cfg, ds, dev)
    errs = np.zeros((ds.n, ds.T))
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_cm")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--margin_s", type=float, default=0.3)
    ap.add_argument("--outdir", default="results/cm_test")
    ap.add_argument("--cm_mode", default=None, choices=[None, "common", "independent"])
    args = ap.parse_args()
    cfg = Config(); cfg.data_dir = args.data_dir
    if args.cm_mode:
        cfg.cm_mode = args.cm_mode
    dev = "cpu"; torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    if args.T < ds.T:
        for a in ("gt", "u", "m_true", "wind", "range_clean", "noise_scale", "range_bias"):
            setattr(ds, a, getattr(ds, a)[:, :args.T])
        ds.T = args.T
    os.makedirs(args.outdir, exist_ok=True)
    lines = []
    def P(s=""): lines.append(s); print(s)

    P(f"[cm_test] {ds.n} trajs x {ds.T} steps  cm_bias={cfg.cm_bias} "
      f"(calm σ_b={cfg.cm_calm_std}/τ={cfg.cm_calm_tau}; dyn σ_b={cfg.cm_dyn_std}/"
      f"τ={cfg.cm_dyn_tau}; s_a∈{cfg.cm_sens_range})")

    base = torch.tensor(cfg.meas_sigma, device=dev).view(1, 1, 4)
    g = torch.Generator(device=dev).manual_seed(1234)
    z = ds.range_clean + ds.range_bias + base * ds.noise_scale * torch.randn(
        ds.n, ds.T, 4, generator=g, device=dev)

    masks = make_masks(cfg, ds, args.margin_s, args.tail)
    order = ["calm", "dynamic", "recovery-tail"]
    combined = np.zeros((ds.n, ds.T), bool)
    for k in order:
        combined |= masks[k]

    P("  running EKF / EKF-oracle / UKF / fixed grid / greedy ...")
    ekf = run_recursive(cfg, ds, z, dev, EKF)
    ekf_or = run_recursive(cfg, ds, z, dev, EKF, oracle=True)
    ukf = run_recursive(cfg, ds, z, dev, UKF)
    combos = [(N, l) for N in NG for l in LG]
    fixedc = {c: run_fixed(cfg, ds, c[0], c[1], z, dev) for c in combos}
    fir14 = fixedc[(cfg.N_default, 1.0)]
    # dense N grid for the U-curve (λ=1)
    errN = {N: run_fixed_err(cfg, ds, N, z, dev) for N in NGRID}
    grd, chN, chL, nu_all = run_greedy(cfg, ds, z, dev)

    # ── (a) regime-wise fixed-N U-curve ──
    P("\n(a) REGIME fixed-N U-curve RMSE [m] (λ=1)  — dynamic floor shift?")
    P("  " + f"{'regime':14s}" + " ".join(f"N{n:>2d}" for n in NGRID) + "  N_opt")
    ucurve = {}
    for k in order:
        c = np.array([rmse(errN[N], masks[k]) for N in NGRID]); ucurve[k] = c
        nopt = NGRID[int(np.argmin(c))]
        P(f"  {k:14s}" + " ".join(f"{v:.3f}" for v in c) + f"  {nopt:>4d}")
    ncalm = NGRID[int(np.argmin(ucurve['calm']))]
    ndyn = NGRID[int(np.argmin(ucurve['dynamic']))]
    P(f"  → calm N*={ncalm}  dynamic N*={ndyn}  (ΔN={ndyn-ncalm:+d}; expect leftward)")

    # ── (b) 3-label decomposition ──
    combo_lab = {k: min(combos, key=lambda c: rmse(fixedc[c], masks[k])) for k in order}
    combo_glob = min(combos, key=lambda c: rmse(fixedc[c], combined))
    Nseq = torch.full((ds.n, ds.T), float(cfg.N_default), device=dev)
    Lseq = torch.full((ds.n, ds.T), 1.0, device=dev)
    for k in order:
        cN, cL = combo_lab[k]; m = torch.tensor(masks[k], device=dev)
        Nseq[m] = cN; Lseq[m] = cL
    regime = run_seq(cfg, ds, z, Nseq, Lseq, dev)

    P("\n(b) LEARNABILITY DECOMPOSITION  (global best-fixed=%s)" % str(combo_glob))
    P(f"  {'label':14s} {'n':>6s} {'bestfix':>7s} {'regime':>7s} {'greedy':>7s} | "
      f"{'LEARN%':>7s} {'luck%':>6s}  combo(label)")
    learn = {}
    for k in order + ["COMBINED"]:
        m = combined if k == "COMBINED" else masks[k]
        bf = rmse(fixedc[combo_glob], m); ro = rmse(regime, m); gr = rmse(grd, m)
        lp = 100 * (bf - ro) / bf if bf > 0 else 0.0
        lk = 100 * (ro - gr) / ro if ro > 0 else 0.0
        learn[k] = lp
        cstr = str(combo_glob) if k == "COMBINED" else str(combo_lab[k])
        P(f"  {k:14s} {int(m.sum()):6d} {bf:7.3f} {ro:7.3f} {gr:7.3f} | "
          f"{lp:6.1f}% {lk:5.1f}%  {cstr}")

    # ── (c) tiers ──
    P("\n(c) TIERS RMSE [m]  (IIR=EKF/UKF, FIR14/FIRbest, AFIR=greedy)")
    P(f"  {'region':14s} {'EKF':>6s} {'EKF-or':>6s} {'UKF':>6s} {'FIR14':>6s} "
      f"{'FIRbest':>7s} {'AFIR':>6s}   chain")
    iir_fir = {}
    for k in order + ["COMBINED"]:
        m = combined if k == "COMBINED" else masks[k]
        cb = combo_lab[k] if k != "COMBINED" else combo_glob
        e_ekf = rmse(ekf, m); e_fir = min(rmse(fir14, m), rmse(fixedc[cb], m))
        e_afir = rmse(grd, m)
        chain = "IIR<FIR<AFIR" if (e_ekf > e_fir and e_afir <= e_fir + 1e-6) else \
                ("IIR<FIR" if e_ekf > e_fir else "EKF leads")
        iir_fir[k] = e_ekf > e_fir
        P(f"  {k:14s} {e_ekf:6.3f} {rmse(ekf_or,m):6.3f} {rmse(ukf,m):6.3f} "
          f"{rmse(fir14,m):6.3f} {rmse(fixedc[cb],m):7.3f} {e_afir:6.3f}   {chain}")

    # ── (d) common-mode corr feature ──
    anchor_mean = np.nanmean(nu_all, axis=2)          # [n,T] common-mode proxy
    L = cfg.L_obs
    roll_mean = np.full_like(anchor_mean, np.nan)
    roll_std = np.full_like(anchor_mean, np.nan)
    with np.errstate(invalid="ignore"):
        for t in range(anchor_mean.shape[1]):
            w = anchor_mean[:, max(0, t - L + 1):t + 1]
            roll_mean[:, t] = np.nanmean(w, axis=1)
            roll_std[:, t] = np.nanstd(w, axis=1)
    P("\n(d) COMMON-MODE feature vs N*  (anchor-MEAN innovation, window L=%d)" % L)
    P(f"  {'label':14s} {'corr(|rollMean|,N*)':>20s} {'corr(rollStd,N*)':>17s} {'N*mean':>7s}")
    for k in order + ["COMBINED"]:
        m = combined if k == "COMBINED" else masks[k]
        r1, _ = pearson(np.abs(roll_mean)[m], chN[m])
        r2, _ = pearson(roll_std[m], chN[m])
        Nm = chN[m]; Nm = Nm[np.isfinite(Nm)]
        P(f"  {k:14s} {r1:+20.2f} {r2:+17.2f} {Nm.mean() if Nm.size else float('nan'):7.1f}")

    # ── verdict ──
    learn_ok = (learn["COMBINED"] >= 10.0) or (learn["dynamic"] >= 10.0)
    fir_ok = iir_fir["dynamic"]
    go = learn_ok and fir_ok
    P("\n" + "=" * 74)
    P("PRE-REGISTERED GATE: LEARNABLE% ≥ 10% AND dynamic IIR<FIR")
    P("  LEARNABLE%%: combined=%.1f%%  dynamic=%.1f%%  → %s" %
      (learn["COMBINED"], learn["dynamic"], "PASS" if learn_ok else "FAIL"))
    P("  dynamic IIR<FIR: %s" % ("PASS" if fir_ok else "FAIL"))
    P("  VERDICT: %s" % ("GO — tag-common-mode is the paper world" if go
                          else "NO-GO — end noise-model hunt"))
    P("=" * 74)

    # figure (a)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k, col in [("calm", "tab:gray"), ("dynamic", "tab:red"),
                   ("recovery-tail", "tab:green")]:
        c = ucurve[k]; nopt = NGRID[int(np.argmin(c))]
        ax.plot(NGRID, c, marker="o", color=col, label=f"{k} (N*={nopt})")
        ax.scatter([nopt], [c.min()], color=col, s=80, zorder=5, edgecolor="k", lw=.6)
    ax.set_xlabel("fixed horizon N"); ax.set_ylabel("regime RMSE [m]")
    ax.set_title("cm_test (a) regime U-curve — dynamic floor shift?")
    ax.legend(fontsize=8); ax.grid(alpha=.3); fig.tight_layout()
    fig.savefig(os.path.join(args.outdir, "cm_ucurve.png"), dpi=120); plt.close(fig)
    with open(os.path.join(args.outdir, "cm_test.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nfigure+summary → {args.outdir}/")


if __name__ == "__main__":
    main()
