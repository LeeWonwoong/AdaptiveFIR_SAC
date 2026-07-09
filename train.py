"""
train.py — SAC-AFME training entry (log-replay online SAC)
============================================================
  python train.py --outdir results/run0 [--synth N]  [--fix_lambda|--fix_N]

Precision policy (rhukf): NN forward under TF32 (if CUDA), filter math FP32.
"""
import argparse
import os
import sys
import time
import numpy as np
import torch

from config import Config, parse_cli
from rlenv.dataset import TrajDataset
from rlenv.replay_env import VectorReplayEnv
from rl.sac import SACAgent
from rl.buffer import TensorReplayBuffer


def apply_tf32_policy():
    # NN matmuls may use TF32; filter code calls linalg in FP32 explicitly.
    # TF32 DISABLED (2026-07-09): TF32's 10-bit mantissa on the aux-EKF
    # covariance chain (P=APA^T+Q, (I-KC)P) on CUDA intermittently breaks P's
    # positive-definiteness -> garbage K -> bad handover states; observed as a
    # linearly climbing DI-FME monitor on GPU while the identical run on CPU
    # stayed flat at 0.32. The SAC nets are tiny MLPs, so TF32 bought nothing.
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False


def main():
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--synth", type=int, default=0,
                     help="generate a synthetic dataset with N train trajs first")
    pre.add_argument("--synth_heldout", type=int, default=None)
    a, _ = pre.parse_known_args()
    cfg = parse_cli(Config())
    dev = cfg.resolve_device()
    apply_tf32_policy()
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    os.makedirs(cfg.outdir, exist_ok=True)
    cfg.save(os.path.join(cfg.outdir, "config.json"))

    if a.synth > 0:
        from rlenv.synth import generate_dataset
        nh = a.synth_heldout if a.synth_heldout is not None else max(2, a.synth // 4)
        generate_dataset(cfg, cfg.data_dir, n_train=a.synth, n_heldout=nh)

    ds = TrajDataset(cfg, "train", dev)
    env = VectorReplayEnv(cfg, ds, dev, seed=cfg.seed)
    agent = SACAgent(cfg, obs_dim=cfg.obs_dim, device=dev)
    buf = TensorReplayBuffer(cfg.buffer_size, cfg.obs_dim, cfg.act_dim, dev)
    gen = torch.Generator(device=dev)
    gen.manual_seed(cfg.seed + 1)

    log_path = os.path.join(cfg.outdir, "train_log.csv")
    with open(log_path, "w") as f:
        f.write("step,transitions,ep_return,rmse,mean_N,mean_lam,loss_q,loss_pi,alpha,sps\n")

    obs = env.reset()
    ep_ret = torch.zeros(cfg.n_envs, device=dev)
    hist_ret, hist_rmse = [], []
    err_acc, ref_acc, n_acc, l_acc, cnt = 0.0, 0.0, 0.0, 0.0, 0
    nsd_acc = 0.0
    refbig = 0
    nhi_acc = nlo_acc = 0.0
    nhi_cnt = nlo_cnt = 0
    losses = {"loss_q": 0.0, "loss_pi": 0.0, "alpha": 0.0}
    t0 = time.time()
    transitions = 0

    for step in range(1, cfg.total_steps + 1):
        if step <= cfg.start_random_steps:
            act = 2 * torch.rand(cfg.n_envs, cfg.act_dim, device=dev) - 1
        else:
            act = agent.act(obs)
        obs2, rew, done, info = env.step(act)
        buf.push_batch(obs, act, rew, obs2, done)
        transitions += cfg.n_envs
        ep_ret += rew
        err_acc += float(info["err"].mean())
        ref_acc += float(info.get("err_ref", info["err"]).mean())
        refbig += int((info.get("err_ref", info["err"]) > 2.0).sum())
        n_acc += float(info["N"].mean())
        nsd_acc += float(info["N"].float().std())
        l_acc += float(info["lam"].mean())
        cnt += 1
        # ── adaptation diagnostic: N conditioned on the UWB innovation level
        #    the policy just saw (obs[:,0] = newest g_uwb, units of "x nominal").
        #    A CONSTANT policy shows gap ~= 0; adaptation shows N|hi < N|lo.
        _g = obs[:, 0]
        _hi = _g > 1.5
        _lo = _g < 1.2
        if _hi.any():
            nhi_acc += float(info["N"][_hi].sum()); nhi_cnt += int(_hi.sum())
        if _lo.any():
            nlo_acc += float(info["N"][_lo].sum()); nlo_cnt += int(_lo.sum())
        obs = obs2

        if info["ep_end"]:
            hist_ret.append(float(ep_ret.mean()))
            ep_ret.zero_()
            obs = env.reset()

        if transitions >= cfg.learning_starts:
            for _ in range(cfg.updates_per_step):
                losses = agent.update(buf.sample(cfg.batch_size, gen))

        if step % 500 == 0:
            sps = step / (time.time() - t0)
            rmse = err_acc / max(cnt, 1)
            hist_rmse.append(rmse)
            ret = hist_ret[-1] if hist_ret else float("nan")
            nhi = nhi_acc / max(nhi_cnt, 1)
            nlo = nlo_acc / max(nlo_cnt, 1)
            rref = ref_acc / max(cnt, 1)
            print(f"[{step:7d}] ret/ep {ret:9.1f} | rmse {rmse:.4f} (DI-FME {rref:.4f}, div {refbig}) | "
                  f"N {n_acc/max(cnt,1):5.1f}±{nsd_acc/max(cnt,1):3.1f} lam {l_acc/max(cnt,1):.3f} | "
                  f"N|dist {nhi:4.1f} vs N|calm {nlo:4.1f} (gap {nlo-nhi:+4.1f}) | "
                  f"q {losses['loss_q']:.3f} pi {losses['loss_pi']:.3f} "
                  f"a {losses['alpha']:.3f} | {sps:.1f} vsteps/s", flush=True)
            with open(log_path, "a") as f:
                f.write(f"{step},{transitions},{ret:.2f},{rmse:.5f},"
                        f"{n_acc/max(cnt,1):.2f},{l_acc/max(cnt,1):.4f},"
                        f"{losses['loss_q']:.4f},{losses['loss_pi']:.4f},"
                        f"{losses['alpha']:.4f},{sps:.1f}\n")
            err_acc = n_acc = l_acc = 0.0
            cnt = 0

        if step % cfg.ckpt_every == 0 or step == cfg.total_steps:
            agent.save(os.path.join(cfg.outdir, "ckpt.pt"))

    agent.save(os.path.join(cfg.outdir, "ckpt.pt"))
    # learning curve
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.2))
        ax[0].plot(hist_ret); ax[0].set_title("episode return"); ax[0].grid(alpha=.3)
        ax[1].plot(hist_rmse); ax[1].set_title("running RMSE [m]"); ax[1].grid(alpha=.3)
        fig.tight_layout()
        fig.savefig(os.path.join(cfg.outdir, "learning_curve.png"), dpi=140)
        print("saved", os.path.join(cfg.outdir, "learning_curve.png"))
    except Exception as e:
        print("plot skipped:", e)


if __name__ == "__main__":
    main()
