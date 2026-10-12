"""Training loop for the mixture-of-experts model (one model = one participant subset, K, weights, seed)."""
from __future__ import annotations
import time
import numpy as np
import pandas as pd
import torch

from model import MoEVAE, losses
from data import CellData, TECHS, capped_epoch_indices, class_weights


def build_model(cfg: dict, n_genes: int, K: int) -> MoEVAE:
    m = cfg["model"]
    return MoEVAE(n_genes, len(TECHS), len(cfg["condition"]["classes"]), K, n_hidden=m["n_hidden"],
                  n_backbone_layers=m["n_backbone_layers"], expert_hidden=m["expert_hidden"],
                  n_latent=m["n_latent"], top_k=m["top_k"], gate_temperature=m["gate_temperature"],
                  dropout=m["dropout"], gate_noise=m.get("gate_noise", 1.0))


def warm_init(model: MoEVAE, single: dict, noise: float, seed: int) -> None:
    """Initialise a K-expert model from a trained K=1 model: shared backbone, decoder, latent heads and
    dispersion copied; every expert copied from the single expert plus small Gaussian noise."""
    g = torch.Generator().manual_seed(seed)
    own = model.state_dict()
    for k, v in single.items():
        if k.startswith(("backbone.", "dec.", "dec_out.", "z_mu.", "z_lv.", "log_theta")):
            own[k] = v.clone()
        elif k.startswith("experts.0."):
            for e in range(model.K):
                kk = k.replace("experts.0.", f"experts.{e}.", 1)
                own[kk] = v.clone() + noise * v.std().clamp_min(1e-3) * torch.randn(v.shape, generator=g).to(v.device)
    model.load_state_dict(own)


def train_model(cfg: dict, data: CellData, train_idx: np.ndarray, K: int, seed: int, w_over: dict | None = None,
                epochs: int | None = None, log_every: int = 1, tag: str = "", init_from: dict | None = None,
                anneal_noise: bool = False, init_noise: float = 0.02) -> tuple[MoEVAE, pd.DataFrame]:
    tr = cfg["train"]
    torch.manual_seed(seed); np.random.seed(seed)
    rng = np.random.default_rng(seed)
    w = {"kl": tr["w_kl"], "balance": tr["w_balance"], "cond": tr["w_cond"], "adv": tr["w_adv"]}
    w.update(w_over or {})
    epochs = epochs or tr["epochs"]
    model = build_model(cfg, data.X.shape[1], K).to(data.device)
    if init_from is not None:
        warm_init(model, init_from, init_noise, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["lr"] * 0.05)
    cw = class_weights(data.cond, data.participant, train_idx, len(cfg["condition"]["classes"]), data.device)
    rows, bs = [], tr["batch_size"]
    for ep in range(epochs):
        model.train(); t0 = time.time()
        if anneal_noise:
            model.noise_mult = max(0.0, 1.0 - ep / max(1, epochs - 1))
        kl_w = w["kl"] * min(1.0, (ep + 1) / max(1, tr["kl_warmup_epochs"]))
        ww = dict(w, kl=kl_w)
        idx = capped_epoch_indices(data.participant, train_idx, tr["cells_per_participant_cap"], rng)
        acc = {}
        nb = 0
        for s in range(0, len(idx) - bs + 1, bs):
            bn = idx[s:s + bs]; b = torch.as_tensor(bn, device=data.device); xb = data.rows(bn)
            out = model(xb, data.tech[b], sample=True, grl_lambda=tr["grl_lambda"])
            loss, parts = losses(out, xb, data.tech[b], data.cond[b], ww, cond_weight=cw)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
            with torch.no_grad():
                share = out["g"].gt(0).float().mean(0)
                acc["min_expert_share"] = acc.get("min_expert_share", 0.0) + share.min().item() / model.top_k
        sched.step()
        row = {"epoch": ep + 1, **{k: v / max(nb, 1) for k, v in acc.items()}, "kl_weight": kl_w,
               "sec": time.time() - t0, "n_cells": len(idx)}
        rows.append(row)
        if (ep + 1) % log_every == 0:
            print(tag, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()),
                  flush=True)
    return model, pd.DataFrame(rows)
