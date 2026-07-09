"""
rlenv/dataset.py — trajectory log storage for log-replay training
==================================================================
Loads all .npz trajectories of a split into device-resident tensors
(decision #3: UWB clean ranges are COMPUTED here from p_gt + anchors, never
stored — measurement noise is injected online by the env with per-episode
sigma randomization → free data augmentation, exogenous w.r.t. actions).
"""
import glob
import json
import os
import numpy as np
import torch


def gauss_markov_bias(std_field, tau_field, dt, rng, clip=None):
    """Time-correlated (Gauss-Markov / discrete OU) per-anchor measurement bias.

    Non-stationary AR(1) with per-step target std/τ so the parameters can switch
    inside a burst window (and relax afterwards, giving a natural recovery tail):
        a_k = 1 - dt/τ_k ,  σ_w,k = σ_b,k·√(1-a_k²)
        b_k = a_k·b_{k-1} + σ_w,k·N(0,1)
    b_0 drawn from the stationary N(0, σ_b,0²). Zero-mean; stationary std → σ_b.

    std_field, tau_field : [n, T, n_a] float arrays (meters, seconds).
    returns bias : [n, T, n_a] float32.
    """
    std = np.asarray(std_field, np.float64)
    tau = np.asarray(tau_field, np.float64)
    n, T, n_a = std.shape
    a = np.clip(1.0 - dt / np.maximum(tau, dt), 0.0, 0.9999)      # [n,T,n_a]
    sw = std * np.sqrt(np.maximum(1.0 - a * a, 1e-12))
    b = np.empty((n, T, n_a), np.float64)
    b[:, 0] = std[:, 0] * rng.standard_normal((n, n_a))
    for k in range(1, T):
        b[:, k] = a[:, k] * b[:, k - 1] + sw[:, k] * rng.standard_normal((n, n_a))
    if clip is not None:
        np.clip(b, -clip, clip, out=b)
    return b.astype(np.float32)


class TrajDataset:
    def __init__(self, cfg, split, device):
        self.cfg, self.dev = cfg, device
        d = os.path.join(cfg.data_dir, split)
        files = sorted(glob.glob(os.path.join(d, "traj_*.npz")))
        if not files:
            raise FileNotFoundError(
                f"no trajectories in {d} — run `python -m rlenv.synth` (Tier-0) "
                f"or datagen/run_datagen.py (Isaac Sim) first")
        self.metas = []
        gts, us, mts, wds = [], [], [], []
        Tmin = None
        wl = tuple(getattr(cfg, "train_scenario_types", ()) or ())
        skipped = 0
        for f in files:
            if wl:
                mf0 = f.replace("traj_", "meta_").replace(".npz", ".json")
                sc0 = (json.load(open(mf0)).get("scenario", {})
                       if os.path.exists(mf0) else {})
                stype = sc0.get("type") or "nominal"   # empty scenario == clean flight
                if stype not in wl:
                    skipped += 1
                    continue
            z = np.load(f)
            T = z["gt"].shape[0]
            Tmin = T if Tmin is None else min(Tmin, T)
            gts.append(z["gt"]); us.append(z["u"])
            mts.append(z["m_true"]); wds.append(z["wind"])
            mf = f.replace("traj_", "meta_").replace(".npz", ".json")
            self.metas.append(json.load(open(mf)) if os.path.exists(mf) else {})
        # crop to common length → stack
        self.T = Tmin
        self.gt = torch.tensor(np.stack([g[:Tmin] for g in gts]), device=device)   # [n,T,12]
        self.u = torch.tensor(np.stack([u[:Tmin] for u in us]), device=device)     # [n,T,4]
        self.m_true = torch.tensor(np.stack([m[:Tmin] for m in mts]), device=device)
        self.wind = torch.tensor(np.stack([w[:Tmin] for w in wds]), device=device)
        self.n = self.gt.shape[0]
        if wl and skipped:
            print(f"[dataset:{split}] scenario whitelist {wl}: "
                  f"kept {self.n}, skipped {skipped}")
        anch = torch.tensor(cfg.anchors, dtype=torch.float32, device=device)       # [4,3]
        # clean UWB ranges  [n,T,4]
        self.range_clean = torch.linalg.vector_norm(
            self.gt[:, :, None, 0:3] - anch[None, None], dim=3)
        # anchor-dropout scenario: NaN out the dropped anchor(s) during their
        # intervals (DI-FME intermittent loss). The env passes NaN through and
        # the filter excludes that anchor row (see wfme.step).
        dt = cfg.dt
        for i, meta in enumerate(self.metas):
            for dp in meta.get("scenario", {}).get("dropouts", []):
                k0 = int(dp["start_s"] / dt)
                k1 = min(int((dp["start_s"] + dp["duration_s"]) / dt), self.T)
                for aidx in dp.get("anchors", []):
                    if 0 <= aidx < self.range_clean.shape[2] and k1 > k0:
                        self.range_clean[i, k0:k1, aidx] = float("nan")
        # NLoS burst: PER-ANCHOR measurement corruption (exogenous w.r.t.
        # actions). The actual draw is  z = range_clean + range_bias
        # + (σ_ep * noise_scale) * randn  (see replay_env._measure /
        # evaluate.Runner / tools.adapt_signal). During a burst the affected
        # anchor's σ jumps LoS(0.12)→NLoS(0.45) and gains a positive multipath
        # bias; the model-based filters keep believing R=LoS.
        n_a = self.range_clean.shape[2]
        nom_sig = float(cfg.meas_sigma[0])
        self.noise_scale = torch.ones(self.n, self.T, n_a, device=device)
        # per-(traj,step,anchor) OU parameter fields: LoS baseline everywhere,
        # NLoS burst windows switch to the fast/large regime (see gauss_markov_bias).
        gm = bool(getattr(cfg, "gm_bias", False))
        std_f = np.full((self.n, self.T, n_a), float(getattr(cfg, "los_bias_std", 0.0)), np.float64)
        tau_f = np.full((self.n, self.T, n_a), float(getattr(cfg, "los_bias_tau", 1.0)), np.float64)
        for i, meta in enumerate(self.metas):
            for nb in meta.get("scenario", {}).get("nlos_burst", []):
                a = int(nb.get("anchor", -1))
                if not (0 <= a < n_a):
                    continue
                k0 = int(nb["start_s"] / dt)
                k1 = min(int((nb["start_s"] + nb["duration_s"]) / dt), self.T)
                if k1 <= k0:
                    continue
                self.noise_scale[i, k0:k1, a] = float(nb.get("sigma", nom_sig)) / max(nom_sig, 1e-9)
                # [Phase-0 last mechanism] REPLACE the old within-burst CONSTANT
                # bias with a fast time-correlated (OU) bias — the wandering
                # within the window is what breaks √N averaging and gives finite N_opt.
                std_f[i, k0:k1, a] = float(getattr(cfg, "nlos_bias_std", nb.get("bias_m", 0.0)))
                tau_f[i, k0:k1, a] = float(getattr(cfg, "nlos_bias_tau", 1.0))
        # [Phase-0 FINAL, cm_mode="independent"] antenna-pattern model: each anchor
        # gets its OWN attitude-driven OU (dynamic params) in dynamic segments —
        # the per-anchor-INDEPENDENT world that gave A-1 its finite N_opt.
        cm_indep = (bool(getattr(cfg, "cm_bias", False))
                    and getattr(cfg, "cm_mode", "common") == "independent")
        if cm_indep:
            for i, meta in enumerate(self.metas):
                for seg in meta.get("scenario", {}).get("cm_regime", []):
                    k0 = int(seg["start_s"] / dt)
                    k1 = min(int((seg["start_s"] + seg["duration_s"]) / dt), self.T)
                    if seg.get("mode") == "dynamic":
                        std_f[i, k0:k1, :] = float(cfg.cm_dyn_std)
                        tau_f[i, k0:k1, :] = float(cfg.cm_dyn_tau)
                    else:
                        std_f[i, k0:k1, :] = float(cfg.cm_calm_std)
                        tau_f[i, k0:k1, :] = float(cfg.cm_calm_tau)
        if gm:
            rng = np.random.default_rng(int(getattr(cfg, "gm_bias_seed", 0)))
            bias = gauss_markov_bias(std_f, tau_f, dt, rng,
                                     clip=float(getattr(cfg, "gm_bias_clip", 1.5)))
            # ── TAG-SIDE COMMON-MODE component [Phase-0 FINAL]: one OU per traj
            #    (common to all anchors), scaled by per-anchor sensitivity s_a,
            #    with (σ_b,τ) switched calm↔dynamic by the cm_regime segments.
            #    range[a] += b_common(k)·s_a + b_a(k).  b_common hits every anchor
            #    so it is NOT geometrically rejected (unlike single-anchor NLoS).
            if (bool(getattr(cfg, "cm_bias", False))
                    and getattr(cfg, "cm_mode", "common") == "common"
                    and any(m.get("scenario", {}).get("cm_regime") for m in self.metas)):
                c_std = np.full((self.n, self.T, 1), float(cfg.cm_calm_std), np.float64)
                c_tau = np.full((self.n, self.T, 1), float(cfg.cm_calm_tau), np.float64)
                for i, meta in enumerate(self.metas):
                    for seg in meta.get("scenario", {}).get("cm_regime", []):
                        if seg.get("mode") != "dynamic":
                            continue
                        k0 = int(seg["start_s"] / dt)
                        k1 = min(int((seg["start_s"] + seg["duration_s"]) / dt), self.T)
                        c_std[i, k0:k1, 0] = float(cfg.cm_dyn_std)
                        c_tau[i, k0:k1, 0] = float(cfg.cm_dyn_tau)
                crng = np.random.default_rng(int(getattr(cfg, "gm_bias_seed", 0)) + 991)
                b_common = gauss_markov_bias(c_std, c_tau, dt, crng,
                                             clip=float(getattr(cfg, "gm_bias_clip", 1.5)))  # [n,T,1]
                lo, hi = cfg.cm_sens_range
                s_a = crng.uniform(lo, hi, size=(self.n, 1, n_a)).astype(np.float64)
                bias = bias + (b_common * s_a).astype(np.float32)
                np.clip(bias, -float(getattr(cfg, "gm_bias_clip", 1.5)),
                        float(getattr(cfg, "gm_bias_clip", 1.5)), out=bias)
            self.range_bias = torch.tensor(bias, device=device)
        else:
            # legacy path: within-burst constant multipath bias
            self.range_bias = torch.zeros(self.n, self.T, n_a, device=device)
            for i, meta in enumerate(self.metas):
                for nb in meta.get("scenario", {}).get("nlos_burst", []):
                    a = int(nb.get("anchor", -1))
                    if not (0 <= a < n_a):
                        continue
                    k0 = int(nb["start_s"] / dt)
                    k1 = min(int((nb["start_s"] + nb["duration_s"]) / dt), self.T)
                    if k1 > k0:
                        self.range_bias[i, k0:k1, a] = float(nb.get("bias_m", 0.0))
        print(f"[dataset:{split}] {self.n} trajs x {self.T} steps "
              f"({self.gt.element_size()*self.gt.nelement()/1e6:.1f} MB gt)")
        self._build_disturb_onsets(cfg)

    # ── segment sampling for episodes ──
    def _build_disturb_onsets(self, cfg):
        """per-trajectory list of disturbance onset step indices (for biased
        segment sampling so episodes actually contain the events that reward
        adaptation)."""
        from datagen.scenario import disturbance_intervals
        self.onsets = []
        for meta in self.metas:
            sc = meta.get("scenario", {})
            ks = [int(t0 / cfg.dt) for (t0, t1, _) in disturbance_intervals(sc)]
            self.onsets.append(ks)

    def sample_segments(self, M, seg_len, rng: torch.Generator, disturb_frac=0.7):
        """returns traj_idx [M], t0 [M] with t0+seg_len+1 <= T.
        A fraction `disturb_frac` of segments are ANCHORED so a disturbance
        onset falls early in the window (the learning signal lives there;
        purely-random segments are mostly nominal where N is irrelevant)."""
        if not hasattr(self, "onsets"):
            ti = torch.randint(0, self.n, (M,), generator=rng, device=self.dev)
            t0 = torch.randint(0, self.T - seg_len - 1, (M,), generator=rng,
                               device=self.dev)
            return ti, t0
        hi = self.T - seg_len - 1
        ti = torch.randint(0, self.n, (M,), generator=rng, device=self.dev)
        t0 = torch.randint(0, hi, (M,), generator=rng, device=self.dev)
        want = torch.rand(M, generator=rng, device=self.dev) < disturb_frac
        for j in range(M):
            if not want[j]:
                continue
            ks = self.onsets[int(ti[j])]
            if not ks:
                # this traj has no disturbance → repick a traj that does
                cand = [i for i in range(self.n) if self.onsets[i]]
                if not cand:
                    continue
                ti[j] = cand[int(torch.randint(0, len(cand), (1,),
                              generator=rng, device=self.dev))]
                ks = self.onsets[int(ti[j])]
            k = ks[int(torch.randint(0, len(ks), (1,), generator=rng,
                                     device=self.dev))]
            # place onset ~1/4 into the window so pre/post are both seen
            start = max(0, min(hi - 1, k - seg_len // 4))
            t0[j] = start
        return ti, t0

    # ── batched per-step gather ──
    def get(self, ti, t):
        """ti [M] traj indices, t [M] time indices →
           u_prev [M,4] (control acting t-1→t), range_clean [M,4], p_gt [M,3], gt [M,12]"""
        up = self.u[ti, torch.clamp(t - 1, min=0)]
        return up, self.range_clean[ti, t], self.gt[ti, t, 0:3], self.gt[ti, t]
