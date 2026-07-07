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
        print(f"[dataset:{split}] {self.n} trajs x {self.T} steps "
              f"({self.gt.element_size()*self.gt.nelement()/1e6:.1f} MB gt)")

    # ── segment sampling for episodes ──
    def sample_segments(self, M, seg_len, rng: torch.Generator):
        """returns traj_idx [M], t0 [M] such that t0+seg_len+1 <= T."""
        ti = torch.randint(0, self.n, (M,), generator=rng, device=self.dev)
        t0 = torch.randint(0, self.T - seg_len - 1, (M,), generator=rng,
                           device=self.dev)
        return ti, t0

    # ── batched per-step gather ──
    def get(self, ti, t):
        """ti [M] traj indices, t [M] time indices →
           u_prev [M,4] (control acting t-1→t), range_clean [M,4], p_gt [M,3], gt [M,12]"""
        up = self.u[ti, torch.clamp(t - 1, min=0)]
        return up, self.range_clean[ti, t], self.gt[ti, t, 0:3], self.gt[ti, t]
