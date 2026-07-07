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
        for f in files:
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
        self.range_bias = torch.zeros(self.n, self.T, n_a, device=device)
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
