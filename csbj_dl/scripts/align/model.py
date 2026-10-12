"""Contrastive image-expression alignment: projection heads, symmetric InfoNCE, batch samplers."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def mlp(dims, dropout):
    layers = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(dims[i], dims[i + 1]))
        if i < len(dims) - 2:
            layers += [nn.GELU(), nn.Dropout(dropout)]
    return nn.Sequential(*layers)


class AlignModel(nn.Module):
    """Fixed-encoder image embedding -> image head; spot expression -> expression encoder.
    Standardisation constants (training spots) are buffers so a saved model is self-contained."""

    def __init__(self, n_genes: int, mcfg: dict, img_mu, img_sd, expr_mu, expr_sd):
        super().__init__()
        self.register_buffer("img_mu", torch.as_tensor(img_mu)); self.register_buffer("img_sd", torch.as_tensor(img_sd))
        self.register_buffer("expr_mu", torch.as_tensor(expr_mu)); self.register_buffer("expr_sd", torch.as_tensor(expr_sd))
        self.img = mlp([mcfg["emb_dim"], *mcfg["img_hidden"], mcfg["proj_dim"]], mcfg["dropout"])
        self.expr = mlp([n_genes, *mcfg["expr_hidden"], mcfg["proj_dim"]], mcfg["dropout"])
        self.img_in = nn.Dropout(mcfg.get("img_in_dropout", 0.0))      # input dropout (training only)
        self.expr_in = nn.Dropout(mcfg.get("expr_in_dropout", 0.0))
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / mcfg["init_temperature"])))
        self.max_scale = math.log(mcfg["max_logit_scale"])

    def encode_img(self, e):
        return F.normalize(self.img(self.img_in((e - self.img_mu) / self.img_sd)), dim=-1)

    def encode_expr(self, x):
        return F.normalize(self.expr(self.expr_in((x - self.expr_mu) / self.expr_sd)), dim=-1)

    def scale(self):
        return self.logit_scale.clamp(max=self.max_scale).exp()


def info_nce(zi, ze, scale, block=None):
    """Symmetric InfoNCE. block: section index per row; when given, negatives are restricted to
    rows of the same section (within-section contrast)."""
    logits = scale * zi @ ze.T
    if block is not None:
        same = block[:, None] == block[None, :]
        logits = logits.masked_fill(~same, float("-inf"))
    t = torch.arange(len(zi), device=zi.device)
    return 0.5 * (F.cross_entropy(logits, t) + F.cross_entropy(logits.T, t))


class Sampler:
    """within_section: batch = n_blocks sections x (batch/n_blocks) spots, negatives within section;
    across_sections: batch drawn uniformly from all training spots, negatives from any section."""

    def __init__(self, sec_idx: np.ndarray, rows: np.ndarray, mode: str, batch: int, rng, n_blocks: int = 4):
        self.mode, self.batch, self.rng, self.n_blocks = mode, batch, rng, n_blocks
        self.rows = rows
        self.by_sec = {s: rows[sec_idx[rows] == s] for s in np.unique(sec_idx[rows])}
        self.secs = np.array(list(self.by_sec))
        w = np.array([len(self.by_sec[s]) for s in self.secs], float)
        self.w = w / w.sum()

    def __call__(self):
        if self.mode == "across_sections":
            r = self.rng.choice(self.rows, self.batch, replace=False)
            return r, None
        per = self.batch // self.n_blocks
        secs = self.rng.choice(self.secs, self.n_blocks, replace=False, p=self.w)
        out, blk = [], []
        for j, s in enumerate(secs):
            pool = self.by_sec[s]
            r = self.rng.choice(pool, min(per, len(pool)), replace=False)
            out.append(r); blk.append(np.full(len(r), j))
        return np.concatenate(out), np.concatenate(blk)


@torch.no_grad()
def embed_all(model, E, Xexpr, rows, device, bs=4096):
    model.eval()
    zi, ze = [], []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        zi.append(model.encode_img(torch.from_numpy(E[r]).to(device)).cpu())
        ze.append(model.encode_expr(torch.from_numpy(Xexpr[r]).to(device)).cpu())
    return torch.cat(zi).numpy(), torch.cat(ze).numpy()
