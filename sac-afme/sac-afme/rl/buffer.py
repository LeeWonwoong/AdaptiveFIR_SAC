"""rl/buffer.py — device-resident replay buffer (rhukf TensorReplayBuffer
pattern: preallocated tensors, no host<->device round-trips; PER/N-step off,
continuous action [cap,2])."""
import torch


class TensorReplayBuffer:
    def __init__(self, cap, obs_dim, act_dim, device):
        self.cap, self.dev = cap, device
        self.o = torch.zeros(cap, obs_dim, device=device)
        self.a = torch.zeros(cap, act_dim, device=device)
        self.r = torch.zeros(cap, device=device)
        self.o2 = torch.zeros(cap, obs_dim, device=device)
        self.d = torch.zeros(cap, device=device)
        self.ptr = 0
        self.size = 0

    def push_batch(self, o, a, r, o2, d):
        n = o.shape[0]
        idx = (self.ptr + torch.arange(n, device=self.dev)) % self.cap
        self.o[idx], self.a[idx], self.r[idx] = o, a, r
        self.o2[idx], self.d[idx] = o2, d
        self.ptr = int((self.ptr + n) % self.cap)
        self.size = min(self.size + n, self.cap)

    def sample(self, batch, gen):
        idx = torch.randint(0, self.size, (batch,), generator=gen, device=self.dev)
        return self.o[idx], self.a[idx], self.r[idx], self.o2[idx], self.d[idx]
