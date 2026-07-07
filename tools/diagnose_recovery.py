"""
tools/diagnose_recovery.py — is there ADAPTATION headroom in the recovery
regime? (run BEFORE committing to Isaac datagen / full training)

The paper's claim is NOT "lower nominal RMSE" but "faster RMSE recovery in
disturbance / dropout windows" (cf. DI-FME vs EKF/UKF). This script measures,
on synthetic held-out data, whether that regime actually rewards adaptation:

  (1) Per-regime best FIXED N: for each regime (nominal / mass / gust /
      dropout) find which single N minimizes RMSE. If the argmin N DIFFERS
      across regimes, a fixed filter is structurally suboptimal -> adaptation
      can win.
  (2) Oracle gap in the DISTURBANCE window only: Greedy-GT vs best fixed FME,
      restricted to disturbance samples. This is the ceiling the SAC agent
      can claim in the regime that matters.
  (3) Recovery time: after each disturbance ends, steps until err < 1.5x the
      pre-disturbance median, for each fixed N and for Greedy-GT. Shows
      whether small-N recovers faster (the mechanism the agent would exploit).

Usage: python -m tools.diagnose_recovery [--n 8]
"""
import argparse
import os
import sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import Config
from rlenv.dataset import TrajDataset
from filter.baselines import FixedFME
from filter.wfme import WeightedFME
from datagen.scenario import disturbance_intervals


def regime_masks(cfg, ds):
    """per-trajectory boolean masks: nominal vs each disturbance label."""
    dt = cfg.dt
    T = ds.T
    labels = ("mass_step", "gust", "sustained_wind", "anchor_dropout")
    masks = {lab: np.zeros((ds.n, T), bool) for lab in labels}
    nominal = np.ones((ds.n, T), bool)
    for i, meta in enumerate(ds.metas):
        sc = meta.get("scenario", {})
        for (t0, t1, lab) in disturbance_intervals(sc):
            k0, k1 = int(t0 / dt), min(int(t1 / dt), T)
            if lab in masks:
                masks[lab][i, k0:k1] = True
            nominal[i, k0:k1] = False
    masks["nominal"] = nominal
    return masks


def run_fixed(cfg, ds, N, z_noisy, dev):
    M = ds.n
    flt = FixedFME(cfg, dev, M, N=N, lam=1.0)
    flt.reset(torch.arange(M, device=dev),
              ds.gt[:, 0] + cfg.init_pos_noise * torch.randn(M, 12, device=dev))
    errs = torch.zeros(M, ds.T, device=dev)
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z_noisy[:, t])
        errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1)
    return errs.cpu().numpy()


def run_greedy(cfg, ds, z_noisy, dev):
    M = ds.n
    flt = WeightedFME(cfg, dev, M)
    flt.reset(torch.arange(M, device=dev),
              ds.gt[:, 0] + cfg.init_pos_noise * torch.randn(M, 12, device=dev))
    Ng = [8., 12., 16., 20.]
    Lg = [0.7, 1.0]
    errs = np.zeros((M, ds.T))
    chosenN = np.full((M, ds.T), np.nan)
    defN = torch.full((M,), 14.0, device=dev)
    defL = torch.full((M,), 1.0, device=dev)
    warm = cfg.warmup_steps
    for t in range(1, ds.T):
        s, _, _ = flt.step(ds.u[:, t - 1], z_noisy[:, t], defN, defL)
        if t > warm:
            best = (ds.gt[:, t, 0:3] - s[:, 0:3]).pow(2).sum(1)
            bs, bN = s.clone(), defN.clone()
            for n in Ng:
                for l in Lg:
                    sc = flt._solve(torch.full((M,), n, device=dev),
                                    torch.full((M,), l, device=dev))
                    e = (ds.gt[:, t, 0:3] - sc[:, 0:3]).pow(2).sum(1)
                    better = e < best
                    best = torch.where(better, e, best)
                    bs[better] = sc[better]
                    bN = torch.where(better, torch.full((M,), n, device=dev), bN)
            flt.s_hat = bs
            errs[:, t] = best.sqrt().cpu().numpy()
            chosenN[:, t] = bN.cpu().numpy()
        else:
            errs[:, t] = (ds.gt[:, t, 0:3] - s[:, 0:3]).norm(dim=1).cpu().numpy()
    return errs, chosenN


def rmse(errs, mask):
    v = errs[mask]
    v = v[np.isfinite(v)]
    return float(np.sqrt((v ** 2).mean())) if v.size else float("nan")


def recovery_times(cfg, ds, errs):
    """per-disturbance: steps until err<1.5x pre-disturbance median."""
    dt, W = cfg.dt, cfg.warmup_steps
    rec = []
    for i, meta in enumerate(ds.metas):
        sc = meta.get("scenario", {})
        for (t0, t1, lab) in disturbance_intervals(sc):
            k0, k1 = int(t0 / dt), min(int(t1 / dt), ds.T - 1)
            if k1 <= k0 or k0 <= W:
                continue
            nom = np.nanmedian(errs[i, W:k0])
            post = errs[i, k1:]
            idx = np.where(post < 1.5 * max(nom, 1e-3))[0]
            rec.append(idx[0] * dt if len(idx) else (ds.T - k1) * dt)
    return float(np.mean(rec)) if rec else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--T", type=int, default=1200)
    ap.add_argument("--split", default="heldout")
    args = ap.parse_args()
    cfg = Config()
    dev = "cpu"
    torch.manual_seed(0)
    ds = TrajDataset(cfg, args.split, dev)
    g = torch.Generator(device=dev).manual_seed(1234)
    sig = 0.06
    z = ds.range_clean + sig * torch.randn(ds.n, ds.T, 4, generator=g, device=dev)

    masks = regime_masks(cfg, ds)
    Ns = [8, 10, 12, 14, 16, 18, 20]
    print(f"\n=== per-regime RMSE by fixed N (split={args.split}, {ds.n} trajs) ===")
    fixed_errs = {N: run_fixed(cfg, ds, N, z, dev) for N in Ns}
    header = "regime        " + "".join(f"  N={N:<2d}" for N in Ns) + "   argmin"
    print(header)
    regime_best = {}
    for lab in ("nominal", "mass_step", "gust", "anchor_dropout"):
        if lab not in masks or not masks[lab].any():
            continue
        row = {N: rmse(fixed_errs[N], masks[lab]) for N in Ns}
        bestN = min(row, key=row.get)
        regime_best[lab] = bestN
        cells = "".join(f" {row[N]:.3f}" for N in Ns)
        print(f"{lab:13s}" + cells + f"   N={bestN}")

    print("\n=== argmin N per regime ===")
    print("  " + " | ".join(f"{k}: N={v}" for k, v in regime_best.items()))
    spread = len(set(regime_best.values()))
    print(f"  distinct optimal N across regimes: {spread}  "
          f"({'ADAPTATION HELPS' if spread > 1 else 'fixed N already optimal — weak case'})")

    print("\n=== Greedy-GT (upper bound) vs best fixed, DISTURBANCE window only ===")
    g_err, g_N = run_greedy(cfg, ds, z, dev)
    dist_mask = np.zeros((ds.n, ds.T), bool)
    for lab in ("mass_step", "gust", "sustained_wind", "anchor_dropout"):
        if lab in masks:
            dist_mask |= masks[lab]
    best_fixed_overall = min(Ns, key=lambda N: rmse(fixed_errs[N], dist_mask))
    r_fixed = rmse(fixed_errs[best_fixed_overall], dist_mask)
    r_greedy = rmse(g_err, dist_mask)
    gap = 100 * (1 - r_greedy / r_fixed) if r_fixed > 0 else float("nan")
    print(f"  best fixed (N={best_fixed_overall}) disturb-RMSE : {r_fixed:.4f} m")
    print(f"  Greedy-GT              disturb-RMSE : {r_greedy:.4f} m")
    print(f"  >>> adaptation headroom in disturbance regime: {gap:.1f}%  "
          f"({'PROMISING (>15%)' if gap > 15 else 'THIN — reconsider scenarios/obs'})")

    # Greedy chosen-N distribution per regime
    print("\n=== Greedy-GT chosen N distribution (does it differ by regime?) ===")
    for lab in ("nominal", "mass_step", "gust", "anchor_dropout"):
        if lab not in masks or not masks[lab].any():
            continue
        sel = g_N[masks[lab] & np.isfinite(g_N)]
        if sel.size:
            print(f"  {lab:14s} meanN={sel.mean():4.1f}  "
                  f"[p10={np.percentile(sel,10):.0f} p50={np.percentile(sel,50):.0f} "
                  f"p90={np.percentile(sel,90):.0f}]")

    print("\n=== recovery time (s) after disturbance ends (lower = better) ===")
    for N in (8, 14, 20):
        print(f"  fixed N={N:2d} : {recovery_times(cfg, ds, fixed_errs[N]):.3f} s")
    print(f"  Greedy-GT  : {recovery_times(cfg, ds, g_err):.3f} s")

    print("\n=== VERDICT ===")
    ok_spread = spread > 1
    ok_gap = gap > 15
    if ok_spread and ok_gap:
        print("  ✅ adaptation regime is real: optimal N shifts AND disturbance")
        print("     headroom > 15%. Proceed to full training / Isaac datagen.")
    else:
        print("  ⚠️  weak adaptation signal. Before Isaac datagen, make scenarios")
        print("     harsher (overlap mass+gust+dropout, sharper onsets, faster")
        print("     maneuvers) and/or widen N range, until spread>1 & gap>15%.")


if __name__ == "__main__":
    main()
