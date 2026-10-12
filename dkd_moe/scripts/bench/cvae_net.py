"""CPA-like conditional VAE (own torch implementation; CPA/scGen packages pin scvi-tools versions that
conflict with the environment). A basal latent z is made condition-free by an adversarial classifier
behind a gradient-reversal layer; the decoder sees z plus a learned condition embedding and the
technology, and models counts with a negative binomial likelihood."""
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


def mlp(n_in, n_hidden, n_layers, dropout):
    layers, d = [], n_in
    for _ in range(n_layers):
        layers += [nn.Linear(d, n_hidden), nn.LayerNorm(n_hidden), nn.ReLU(), nn.Dropout(dropout)]
        d = n_hidden
    return nn.Sequential(*layers)


def nb_log_prob(x, mu, theta, eps=1e-8):
    """log NB(x | mean mu, inverse dispersion theta), per element."""
    lt = torch.log(theta + eps); lm = torch.log(theta + mu + eps)
    return (theta * (lt - lm) + x * (torch.log(mu + eps) - lm)
            + torch.lgamma(x + theta) - torch.lgamma(theta) - torch.lgamma(x + 1))


class CondVAE(nn.Module):
    def __init__(self, n_genes, n_tech, n_cond, n_latent=16, n_hidden=256, n_layers=2, dropout=0.1,
                 adv_hidden=128):
        super().__init__()
        self.n_tech = n_tech
        self.enc = mlp(n_genes + n_tech, n_hidden, n_layers, dropout)
        self.mu = nn.Linear(n_hidden, n_latent); self.lv = nn.Linear(n_hidden, n_latent)
        self.cond_emb = nn.Embedding(n_cond, n_latent)
        self.dec = mlp(n_latent + n_tech, n_hidden, n_layers, dropout)
        self.px = nn.Linear(n_hidden, n_genes)
        self.log_theta = nn.Parameter(torch.zeros(n_genes))
        self.adv = nn.Sequential(nn.Linear(n_latent, adv_hidden), nn.ReLU(), nn.Linear(adv_hidden, n_cond))

    def encode(self, x, tech):
        xl = torch.log1p(x / x.sum(1, keepdim=True).clamp(min=1) * 1e4)
        h = self.enc(torch.cat([xl, F.one_hot(tech, self.n_tech).float()], 1))
        return self.mu(h), self.lv(h).clamp(-10, 10)

    def decode(self, z, cond, tech, lib):
        h = self.dec(torch.cat([z + self.cond_emb(cond), F.one_hot(tech, self.n_tech).float()], 1))
        return torch.softmax(self.px(h), 1) * lib, torch.exp(self.log_theta)

    def forward(self, x, tech, cond, lam):
        mu, lv = self.encode(x, tech)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv)
        lib = x.sum(1, keepdim=True)
        m, th = self.decode(z, cond, tech, lib)
        rec = -nb_log_prob(x, m, th).sum(1).mean()
        kl = (-0.5 * (1 + lv - mu ** 2 - lv.exp()).sum(1)).mean()
        adv = F.cross_entropy(self.adv(GradReverse.apply(z, lam)), cond)
        return rec, kl, adv

    @torch.no_grad()
    def counterfactual_llr(self, x, tech, c_case, c_ctrl):
        """Per-cell log-likelihood ratio of the observed counts under case vs control condition, per count."""
        mu, _ = self.encode(x, tech)
        lib = x.sum(1, keepdim=True)
        out = []
        for c in (c_case, c_ctrl):
            m, th = self.decode(mu, torch.full_like(tech, c), tech, lib)
            out.append(nb_log_prob(x, m, th).sum(1))
        return (out[0] - out[1]) / lib.squeeze(1).clamp(min=1)

    @torch.no_grad()
    def decoded_shift(self, x, tech, c_case, c_ctrl):
        """Mean difference of decoded log1p(CP10k) expression (case vs control embedding) for the same basal
        states; the pseudocount keeps rarely expressed genes from dominating the ranking."""
        mu, _ = self.encode(x, tech)
        lib = torch.full((x.shape[0], 1), 1e4, device=x.device)
        a, _ = self.decode(mu, torch.full_like(tech, c_case), tech, lib)
        b, _ = self.decode(mu, torch.full_like(tech, c_ctrl), tech, lib)
        return (torch.log1p(a) - torch.log1p(b)).mean(0)
