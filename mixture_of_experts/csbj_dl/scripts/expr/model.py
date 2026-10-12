"""Mixture-of-experts variational model of proximal tubule cells.

x (raw counts) -> log1p(CP10k) -> shared MLP backbone h
  -> gating network: softmax(W_g h / T) = p (dense routing probabilities); top-k renormalised = g
  -> K experts e_k(h); m = sum_k g_k e_k(h)
  -> variational z ~ N(mu(m), sigma(m))
  -> negative-binomial decoder conditioned on technology; mean = softmax(f(z, tech)) * library
Auxiliary heads
  condition head: linear on the routing probabilities p only (Reference / HKD / DKD)
  technology discriminator: on [z, p] through a gradient-reversal layer
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, lam):
        ctx.lam = lam
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.lam * g, None


def mlp(d_in, d_hidden, n_layers, dropout):
    layers, d = [], d_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout)]
        d = d_hidden
    return nn.Sequential(*layers)


def log_norm(x: torch.Tensor) -> torch.Tensor:
    lib = x.sum(1, keepdim=True).clamp_min(1.0)
    return torch.log1p(x / lib * 1e4)


def nb_nll(x, mu, theta, eps=1e-8):
    """Negative-binomial negative log-likelihood (mean mu, inverse dispersion theta), summed over genes."""
    lt = torch.log(theta + eps); lm = torch.log(mu + theta + eps)
    ll = (theta * (lt - lm) + x * (torch.log(mu + eps) - lm)
          + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))
    return -ll.sum(-1)


class MoEVAE(nn.Module):
    def __init__(self, n_genes, n_tech, n_cond, K, n_hidden=256, n_backbone_layers=2, expert_hidden=128,
                 n_latent=16, top_k=2, gate_temperature=1.0, dropout=0.1, gate_noise=1.0):
        super().__init__()
        self.K, self.top_k, self.T, self.n_tech = K, min(top_k, K), gate_temperature, n_tech
        self.gate_noise = gate_noise
        self.noise_mult = 1.0          # annealed by the training loop when requested
        self.backbone = mlp(n_genes, n_hidden, n_backbone_layers, dropout)
        self.gate = nn.Linear(n_hidden, K)
        self.experts = nn.ModuleList([nn.Sequential(nn.Linear(n_hidden, expert_hidden), nn.LayerNorm(expert_hidden),
                                                    nn.GELU(), nn.Linear(expert_hidden, expert_hidden))
                                      for _ in range(K)])
        self.z_mu = nn.Linear(expert_hidden, n_latent)
        self.z_lv = nn.Linear(expert_hidden, n_latent)
        self.dec = nn.Sequential(nn.Linear(n_latent + n_tech, n_hidden), nn.LayerNorm(n_hidden), nn.GELU(),
                                 nn.Linear(n_hidden, n_hidden), nn.LayerNorm(n_hidden), nn.GELU())
        self.dec_out = nn.Linear(n_hidden, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_tech, n_genes) + 1.0)
        self.cond_head = nn.Linear(K, n_cond)
        self.disc = nn.Sequential(nn.Linear(n_latent + K, 64), nn.GELU(), nn.Linear(64, n_tech))

    # ---- encoder pieces (exposed for interpretation) ----
    def encode_h(self, xlog):
        return self.backbone(xlog)

    def gate_probs(self, h, noisy=False):
        logits = self.gate(h)
        if noisy and self.gate_noise > 0:
            logits = logits + torch.randn_like(logits) * self.gate_noise * self.noise_mult / max(self.K, 1) ** 0.5
        return F.softmax(logits / self.T, -1), logits

    def sparse_gate(self, p):
        if self.top_k >= self.K:
            return p
        v, i = p.topk(self.top_k, -1)
        g = torch.zeros_like(p).scatter(-1, i, v)
        return g / g.sum(-1, keepdim=True).clamp_min(1e-8)

    def expert_out(self, h):
        return torch.stack([e(h) for e in self.experts], 1)        # B x K x d

    def mix(self, g, E):
        return (g.unsqueeze(-1) * E).sum(1)

    def decode(self, z, tech_onehot, lib):
        d = self.dec(torch.cat([z, tech_onehot], -1))
        rho = F.softmax(self.dec_out(d), -1)
        return rho, rho * lib

    def forward(self, x, tech, sample=True, grl_lambda=1.0):
        xlog = log_norm(x)
        h = self.encode_h(xlog)
        p, logits = self.gate_probs(h, noisy=self.training)
        g = self.sparse_gate(p)
        m = self.mix(g, self.expert_out(h))
        mu, lv = self.z_mu(m), self.z_lv(m).clamp(-10, 6)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        toh = F.one_hot(tech, self.n_tech).float()
        lib = x.sum(1, keepdim=True).clamp_min(1.0)
        rho, mean = self.decode(z, toh, lib)
        theta = torch.exp(self.log_theta[tech])
        cond_logits = self.cond_head(p)
        disc_logits = self.disc(GradReverse.apply(torch.cat([z, p], -1), grl_lambda))
        return dict(p=p, g=g, logits=logits, mu=mu, lv=lv, z=z, rho=rho, mean=mean, theta=theta,
                    cond_logits=cond_logits, disc_logits=disc_logits)


def cond_logodds(model, p: torch.Tensor) -> torch.Tensor:
    """DKD-vs-HKD log-odds of the condition head (classes [HKD, DKD]); linear in routing probabilities."""
    W, b = model.cond_head.weight, model.cond_head.bias
    return p @ (W[1] - W[0]) + (b[1] - b[0])


def load_balance_loss(p: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
    """Switch-style auxiliary loss K * sum_k f_k P_k (f_k = share of routed mass, P_k = mean probability);
    equals 1 at perfectly uniform routing."""
    K = p.shape[1]
    f = (g > 0).float().mean(0); f = f / f.sum().clamp_min(1e-8)
    P = p.mean(0)
    return K * (f * P).sum()


def losses(out, x, tech, cond, w, cond_weight=None):
    rec = nb_nll(x, out["mean"], out["theta"]).mean()
    kl = (-0.5 * (1 + out["lv"] - out["mu"] ** 2 - out["lv"].exp()).sum(-1)).mean()
    bal = load_balance_loss(out["p"], out["g"])
    m = cond >= 0
    ce = F.cross_entropy(out["cond_logits"][m], cond[m], weight=cond_weight) if m.any() else rec * 0
    adv = F.cross_entropy(out["disc_logits"], tech)
    total = rec + w["kl"] * kl + w["balance"] * bal + w["cond"] * ce + w["adv"] * adv
    return total, dict(rec=rec.item(), kl=kl.item(), bal=bal.item(), ce=ce.item(), adv=adv.item())
