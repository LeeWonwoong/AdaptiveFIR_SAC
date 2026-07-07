"""rl/sac.py — Soft Actor-Critic (Haarnoja et al. 2018), CleanRL-style,
adapted for the log-replay vector env. NN forward may run under TF32; the
filter path stays FP32 (rhukf precision policy — set in train.py)."""
import torch
import torch.nn.functional as F
from .networks import Actor, QNetwork


class SACAgent:
    def __init__(self, cfg, obs_dim, device):
        self.cfg, self.dev = cfg, device
        ad = cfg.act_dim
        self.actor = Actor(obs_dim, ad, cfg.hidden, cfg.log_std_min,
                           cfg.log_std_max).to(device)
        self.q1 = QNetwork(obs_dim, ad, cfg.hidden).to(device)
        self.q2 = QNetwork(obs_dim, ad, cfg.hidden).to(device)
        self.q1t = QNetwork(obs_dim, ad, cfg.hidden).to(device)
        self.q2t = QNetwork(obs_dim, ad, cfg.hidden).to(device)
        self.q1t.load_state_dict(self.q1.state_dict())
        self.q2t.load_state_dict(self.q2.state_dict())
        # AdamW (decoupled weight decay; wd=0이면 Adam과 동일 업데이트)
        self.opt_actor = torch.optim.AdamW(self.actor.parameters(), lr=cfg.lr,
                                           weight_decay=cfg.weight_decay)
        self.opt_q = torch.optim.AdamW(list(self.q1.parameters()) +
                                       list(self.q2.parameters()), lr=cfg.lr,
                                       weight_decay=cfg.weight_decay)
        if cfg.autotune_alpha:
            self.target_entropy = -cfg.target_entropy_scale * ad
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            # alpha(스칼라 온도)에는 decay를 걸지 않음
            self.opt_alpha = torch.optim.AdamW([self.log_alpha], lr=cfg.lr,
                                               weight_decay=0.0)
        else:
            self.log_alpha = torch.log(torch.tensor([0.2], device=device))

    @property
    def alpha(self):
        return self.log_alpha.exp().detach()

    @torch.no_grad()
    def act(self, obs, deterministic=False):
        a, _, mu = self.actor.sample(obs)
        return mu if deterministic else a

    def update(self, batch):
        cfg = self.cfg
        o, a, r, o2, d = batch
        with torch.no_grad():
            a2, logp2, _ = self.actor.sample(o2)
            qt = torch.min(self.q1t(o2, a2), self.q2t(o2, a2)) - self.alpha * logp2
            y = r.unsqueeze(1) + cfg.gamma * (1 - d.unsqueeze(1)) * qt
        q1 = self.q1(o, a)
        q2 = self.q2(o, a)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        self.opt_q.zero_grad(set_to_none=True)
        loss_q.backward()
        self.opt_q.step()

        ap, logp, _ = self.actor.sample(o)
        qpi = torch.min(self.q1(o, ap), self.q2(o, ap))
        loss_pi = (self.alpha * logp - qpi).mean()
        self.opt_actor.zero_grad(set_to_none=True)
        loss_pi.backward()
        self.opt_actor.step()

        if cfg.autotune_alpha:
            loss_a = (-self.log_alpha.exp() * (logp.detach() + self.target_entropy)).mean()
            self.opt_alpha.zero_grad(set_to_none=True)
            loss_a.backward()
            self.opt_alpha.step()

        with torch.no_grad():
            for p, pt in zip(self.q1.parameters(), self.q1t.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
            for p, pt in zip(self.q2.parameters(), self.q2t.parameters()):
                pt.mul_(1 - cfg.tau).add_(cfg.tau * p)
        return {"loss_q": loss_q.item(), "loss_pi": loss_pi.item(),
                "alpha": float(self.alpha)}

    # ── checkpoint ──
    def save(self, path):
        torch.save({"actor": self.actor.state_dict(),
                    "q1": self.q1.state_dict(), "q2": self.q2.state_dict(),
                    "log_alpha": self.log_alpha}, path)

    def load(self, path):
        ck = torch.load(path, map_location=self.dev, weights_only=False)
        self.actor.load_state_dict(ck["actor"])
        self.q1.load_state_dict(ck["q1"])
        self.q2.load_state_dict(ck["q2"])
