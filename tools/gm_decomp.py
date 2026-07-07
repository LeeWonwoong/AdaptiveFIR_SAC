"""
tools/gm_decomp.py — A-3: 3-label learnability decomposition under OU bias
=========================================================================
Phase-0 last-mechanism verdict tool. With the time-correlated (OU) measurement
bias enabled (config.gm_bias), decompose the nlos_burst scenario into THREE GT
regimes so the recovery tail is not diluted into nominal:

    nominal        clean steps of nlos trajs (LoS OU only)
    burst-window   the NLoS burst interval (+margin) — fast-τ OU + σ↑
    recovery-tail  first --tail steps AFTER each burst ends (still polluted;
                   OU bias relaxes over τ_LoS, so N-flush may still pay)

For each label and COMBINED:
    best-fixed    = ONE global (N,λ)   (the practitioner's single choice)
    regime-oracle = GT-label (N,λ) switch  (what a perfect regime detector buys)
    greedy        = per-step oracle    (unrealizable upper bound)
    LEARNABLE% = 100·(best-fixed − regime-oracle)/best-fixed
    luck%      = 100·(regime-oracle − greedy)/regime-oracle

corr feature (A-3, replaces innovation ENERGY): per-anchor innovation ROLLING
MEAN (low-frequency component — matched to a slowly-wandering bias). Reports
corr(max|rolling-mean ν|, N*) and multi-R of the 4 rolling means → N*.

Also reports RECOVERY TIME (steps for pos-error to fall back below the nominal
band) for IIR(EKF)/FIR(fixed)/AFIR(greedy) — the path-B (framing-switch) evidence.

VERDICT: LEARNABLE% ≥ 10% (combined) OR recovery-tail alone ≥ 20% → GO.

Usage: python -m tools.gm_decomp [--data_dir data_diag] [--split train]
                                 [--T 2000] [--tail 25]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.dataset import TrajDataset
from filter.baselines import EKF, FixedFME
from datagen.scenario import disturbance_intervals
from tools.adapt_signal import (
    type_trajs, nominal_mask, rmse, pearson, multi_R,
    run_recursive, run_fixed, run_seq, run_greedy, NG, LG, _reset)

LABEL = "nlos_burst"


def burst_intervals_steps(cfg, ds):
    """per-traj list of (k0,k1) NLoS burst step intervals."""
    dt = cfg.dt
    out = []
    for meta in ds.metas:
        ks = []
        for (t0, t1, lab) in disturbance_intervals(meta.get("scenario", {})):
            if lab == LABEL:
                ks.append((int(t0 / dt), min(int(t1 / dt), ds.T)))
        out.append(sorted(ks))
    return out


def make_masks(cfg, ds, rows, margin_s, tail_steps):
    """nominal / burst(+margin) / recovery-tail boolean masks [n,T],
    restricted to `rows`. Precedence burst > tail > nominal; first N_max off."""
    T = ds.T; mg = int(margin_s / cfg.dt)
    rset = np.zeros(ds.n, bool); rset[rows] = True
    burst = np.zeros((ds.n, T), bool)
    tail = np.zeros((ds.n, T), bool)
    ivs = burst_intervals_steps(cfg, ds)
    for i in rows:
        for (k0, k1) in ivs[i]:
            burst[i, k0:min(k1 + mg, T)] = True
    for i in rows:
        for (k0, k1) in ivs[i]:
            t0 = min(k1 + mg, T); t1 = min(t0 + tail_steps, T)
            tail[i, t0:t1] = True
    tail &= ~burst                       # burst wins overlaps
    nom = rset[:, None] & ~burst & ~tail
    nom[:, :cfg.N_max] = False; burst[:, :cfg.N_max] = False; tail[:, :cfg.N_max] = False
    burst &= rset[:, None]; tail &= rset[:, None]
    return {"nominal": nom, "burst-window": burst, "recovery-tail": tail}


def _smooth(x, w=5):
    """causal rolling mean over axis-1 (time)."""
    out = np.full_like(x, np.nan)
    for t in range(x.shape[1]):
        out[:, t] = np.nanmean(x[:, max(0, t - w + 1):t + 1], axis=1)
    return out


def recovery_times(cfg, ds, rows, err_series, band, margin_s, max_tail_s=1.5):
    """mean seconds after each burst END for the (5-step SMOOTHED) pos-error to
    fall below `band` [m]. err_series [n,T]. returns (mean_s, n_events, %reached)."""
    dt = cfg.dt; mg = int(margin_s / cfg.dt)
    lim = int(max_tail_s / dt)
    sm = _smooth(err_series, 5)
    ivs = burst_intervals_steps(cfg, ds)
    times, reached = [], 0
    for i in rows:
        for (k0, k1) in ivs[i]:
            s0 = min(k1 + mg, ds.T)
            rec = None
            for k in range(s0, min(s0 + lim, ds.T)):
                if np.isfinite(sm[i, k]) and sm[i, k] < band:
                    rec = k - s0; break
            if rec is not None:
                reached += 1
            times.append(rec if rec is not None else lim)
    if not times:
        return float("nan"), 0, 0.0
    return float(np.mean(times)) * dt, len(times), 100.0 * reached / len(times)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="data_diag")
    ap.add_argument("--split", default="train")
    ap.add_argument("--T", type=int, default=2000)
    ap.add_argument("--tail", type=int, default=25)
    ap.add_argument("--margin_s", type=float, default=0.3)
    ap.add_argument("--outdir", default="results/gm_decomp")
    args = ap.parse_args()
    cfg = Config(); cfg.data_dir = args.data_dir
    dev = "cpu"; torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    if args.T < ds.T:
        for a in ("gt", "u", "m_true", "wind", "range_clean", "noise_scale", "range_bias"):
            setattr(ds, a, getattr(ds, a)[:, :args.T])
        ds.T = args.T
    rows = type_trajs(ds, LABEL)
    assert rows, "no nlos_burst trajectories in split"
    os.makedirs(args.outdir, exist_ok=True)
    print(f"[gm_decomp] {ds.n} trajs x {ds.T} steps  nlos_burst rows={len(rows)}  "
          f"gm_bias={cfg.gm_bias} (LoS σ_b={cfg.los_bias_std} τ={cfg.los_bias_tau}; "
          f"NLoS σ_b={cfg.nlos_bias_std} τ={cfg.nlos_bias_tau})")

    # measurement draw (same construction as adapt_signal / ucurve_shift)
    base = torch.tensor(cfg.meas_sigma, device=dev).view(1, 1, 4)
    g = torch.Generator(device=dev).manual_seed(1234)
    z = ds.range_clean + ds.range_bias + base * ds.noise_scale * torch.randn(
        ds.n, ds.T, 4, generator=g, device=dev)

    print("  running EKF / EKF-oracle / fixed grid / greedy ...")
    ekf = run_recursive(cfg, ds, z, dev, EKF)
    ekf_or = run_recursive(cfg, ds, z, dev, EKF, oracle=True)
    combos = [(N, l) for N in NG for l in LG]
    fixedc = {c: run_fixed(cfg, ds, c[0], c[1], z, dev) for c in combos}
    grd, chN, chL, nu_all = run_greedy(cfg, ds, z, dev)

    masks = make_masks(cfg, ds, rows, args.margin_s, args.tail)
    order = ["nominal", "burst-window", "recovery-tail"]
    combined = np.zeros((ds.n, ds.T), bool)
    for k in order:
        combined |= masks[k]

    lines = []
    def P(s=""): lines.append(s); print(s)

    # per-label best fixed combo + global best fixed over combined
    combo_lab = {k: min(combos, key=lambda c: rmse(fixedc[c], masks[k])) for k in order}
    combo_glob = min(combos, key=lambda c: rmse(fixedc[c], combined))

    # regime-oracle: switch per-label best fixed by GT label
    Nseq = torch.full((ds.n, ds.T), float(cfg.N_default), device=dev)
    Lseq = torch.full((ds.n, ds.T), 1.0, device=dev)
    for k in order:
        cN, cL = combo_lab[k]
        m = torch.tensor(masks[k], device=dev)
        Nseq[m] = cN; Lseq[m] = cL
    regime = run_seq(cfg, ds, z, Nseq, Lseq, dev)

    P("\n" + "=" * 78)
    P("A-3  3-LABEL LEARNABILITY DECOMPOSITION under OU bias  (nlos_burst)")
    P(f"  global best-fixed (one (N,λ) over all 3 labels) = {combo_glob}")
    P("=" * 78)
    P(f"  {'label':14s} {'n':>6s} {'bestfix':>7s} {'regime':>7s} {'greedy':>7s} | "
      f"{'LEARN%':>7s} {'luck%':>6s}  combo(label)")
    learn = {}
    for k in order + ["COMBINED"]:
        m = combined if k == "COMBINED" else masks[k]
        npts = int(m.sum())
        bf = rmse(fixedc[combo_glob], m)      # ONE global fixed choice
        ro = rmse(regime, m)
        gr = rmse(grd, m)
        lp = 100 * (bf - ro) / bf if bf > 0 else 0.0
        lk = 100 * (ro - gr) / ro if ro > 0 else 0.0
        learn[k] = lp
        cstr = str(combo_glob) if k == "COMBINED" else str(combo_lab[k])
        P(f"  {k:14s} {npts:6d} {bf:7.3f} {ro:7.3f} {gr:7.3f} | {lp:6.1f}% {lk:5.1f}%  {cstr}")

    # ── new corr feature: per-anchor innovation ROLLING MEAN (low-freq) ──
    L = cfg.L_obs
    roll = np.full_like(nu_all, np.nan)      # [n,T,4] causal rolling mean
    with np.errstate(invalid="ignore"):
        for t in range(nu_all.shape[1]):
            roll[:, t] = np.nanmean(nu_all[:, max(0, t - L + 1):t + 1], axis=1)
        roll_absmax = np.nanmax(np.abs(roll), axis=2)     # [n,T]
    P("\n(corr) per-anchor innovation ROLLING-MEAN (window L=%d) vs N*  "
      "[replaces energy]" % L)
    P(f"  {'label':14s} {'corr(max|rollν|,N*)':>20s} {'multiR(rollν→N*)':>17s} "
      f"{'N*mean':>7s}")
    for k in order + ["COMBINED"]:
        m = combined if k == "COMBINED" else masks[k]
        r, npt = pearson(roll_absmax[m], chN[m])
        X = np.stack([roll[..., a][m] for a in range(4)], axis=1)
        mR = multi_R(X, chN[m])
        Nm = chN[m]; Nm = Nm[np.isfinite(Nm)]
        P(f"  {k:14s} {r:+20.2f} {mR:+17.2f} {Nm.mean() if Nm.size else float('nan'):7.1f}")

    # ── recovery (IIR/FIR/AFIR) — path-B (framing-switch) evidence ──
    nom_band = rmse(fixedc[combo_glob], masks["nominal"])
    band = 1.5 * nom_band
    fir_best = fixedc[combo_glob]
    tiers = [("IIR (EKF)", ekf), ("IIR (EKF-oracle)", ekf_or),
             ("FIR (best-fixed)", fir_best), ("AFIR (greedy)", grd)]
    P("\n(recovery) per-tier — burst & tail RMSE [m] and time→below %.2fm "
      "(5-step smoothed) after each burst end:" % band)
    P(f"  {'tier':18s} {'burstRMSE':>9s} {'tailRMSE':>8s} {'recovery':>9s} {'%reached':>8s}")
    for nm, series in tiers:
        wr = rmse(series, masks["burst-window"]); tr = rmse(series, masks["recovery-tail"])
        rt, ne, pr = recovery_times(cfg, ds, rows, series, band, args.margin_s)
        P(f"  {nm:18s} {wr:9.3f} {tr:8.3f} {rt*1000:7.0f}ms {pr:7.0f}%")

    # ── verdict ──
    go = (learn["COMBINED"] >= 10.0) or (learn["recovery-tail"] >= 20.0)
    P("\n" + "=" * 78)
    P("VERDICT: LEARNABLE%% combined=%.1f%%  recovery-tail=%.1f%%  →  %s" %
      (learn["COMBINED"], learn["recovery-tail"], "GO" if go else "NO-GO"))
    P("  (GO gate: combined ≥ 10%% OR recovery-tail ≥ 20%%)")
    P("=" * 78)
    with open(os.path.join(args.outdir, "gm_decomp.txt"), "w") as f:
        f.write("\n".join(lines))
    print(f"\nsummary → {args.outdir}/gm_decomp.txt")


if __name__ == "__main__":
    main()
