"""rl/networks.py — SAC actor / critic (CleanRL sac_continuous_action style)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(inp, hid, out):
    return nn.Sequential(nn.Linear(inp, hid), nn.ReLU(),
                         nn.Linear(hid, hid), nn.ReLU(),
                         nn.Linear(hid, out))


class QNetwork(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden):
        super().__init__()
        self.net = mlp(obs_dim + act_dim, hidden, 1)

    def forward(self, o, a):
        return self.net(torch.cat([o, a], dim=1))


class Actor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden, log_std_min=-5.0, log_std_max=2.0):
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(obs_dim, hidden), nn.ReLU(),
                                   nn.Linear(hidden, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.lo, self.hi = log_std_min, log_std_max

    def forward(self, o):
        h = self.trunk(o)
        mu = self.mu(h)
        log_std = self.lo + 0.5 * (self.hi - self.lo) * (torch.tanh(self.log_std(h)) + 1)
        return mu, log_std

    def sample(self, o):
        mu, log_std = self(o)
        std = log_std.exp()
        dist = torch.distributions.Normal(mu, std)
        x = dist.rsample()
        a = torch.tanh(x)
        logp = dist.log_prob(x) - torch.log((1 - a.pow(2)) + 1e-6)
        return a, logp.sum(1, keepdim=True), torch.tanh(mu)
