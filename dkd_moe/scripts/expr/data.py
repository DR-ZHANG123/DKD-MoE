"""Data access for the expression model: dense GPU count tensors, encodings, participant-capped sampling."""
from __future__ import annotations
import os
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch

TECHS = ["sc", "sn"]


class CellData:
    """Holds counts (on device) and per-cell metadata for one cohort object."""

    def __init__(self, h5ad: str | Path, cfg: dict, device: str, genes: list[str] | None = None):
        a = ad.read_h5ad(h5ad)
        if genes is not None:
            a = a[:, genes]
        self.obs = a.obs.copy()
        self.genes = list(a.var_names)
        X = a.X.toarray() if hasattr(a.X, "toarray") else np.asarray(a.X)
        # counts stay in host memory; batches are moved to the device on demand
        self.X = torch.from_numpy(X.astype(np.float32))
        if os.environ.get("DKDMOE_DATA_ON_DEVICE") == "1":      # faster when the device has room for the counts
            self.X = self.X.to(device)
        self.device = device
        self.tech = torch.tensor(self.obs.tech.map({t: i for i, t in enumerate(TECHS)}).values,
                                 dtype=torch.long, device=device)
        classes = cfg["condition"]["classes"]
        # condition-head label from group via the config map (unmapped groups = -1, unsupervised)
        cmap = {g: classes.index(c) for g, c in cfg["condition"]["map"].items()}
        c = self.obs.group.astype(str).map(cmap).fillna(-1).astype(int).values
        self.cond = torch.tensor(c, dtype=torch.long, device=device)
        self.participant = self.obs.participant.astype(str).values

    def __len__(self):
        return self.X.shape[0]

    def rows(self, idx) -> torch.Tensor:
        idx = torch.as_tensor(np.asarray(idx), dtype=torch.long, device=self.X.device)
        return self.X[idx].to(self.device)


def capped_epoch_indices(participant: np.ndarray, idx: np.ndarray, cap: int, rng) -> np.ndarray:
    """Shuffle `idx`, keeping at most `cap` cells per participant (fresh subsample every epoch)."""
    order = rng.permutation(len(idx))
    rank = pd.Series(participant[idx][order]).groupby(participant[idx][order]).cumcount().values
    return idx[order[rank < cap]]


def class_weights(cond: torch.Tensor, participant: np.ndarray, idx: np.ndarray, n_cls: int, device) -> torch.Tensor:
    """Inverse participant-count weights so each condition contributes equally to the head."""
    df = pd.DataFrame({"p": participant[idx], "c": cond[idx].cpu().numpy()})
    df = df[df.c >= 0].drop_duplicates("p")
    n = df.c.value_counts().reindex(range(n_cls), fill_value=0).values.astype(float)
    w = np.where(n > 0, n.sum() / (n_cls * np.maximum(n, 1)), 0.0)
    return torch.tensor(w, dtype=torch.float32, device=device)


@torch.no_grad()
def forward_all(model, data: CellData, idx: np.ndarray | None = None, bs: int = 4096) -> dict:
    """Deterministic pass (z = mu, no gate noise): routing probabilities, sparse gates, latent, NLL/cell."""
    from model import nb_nll
    model.eval()
    idx = np.arange(len(data)) if idx is None else idx
    P, G, Z, NLL = [], [], [], []
    for s in range(0, len(idx), bs):
        bn = idx[s:s + bs]; b = torch.as_tensor(bn, device=data.device); xb = data.rows(bn)
        out = model(xb, data.tech[b], sample=False)
        P.append(out["p"].cpu()); G.append(out["g"].cpu()); Z.append(out["mu"].cpu())
        NLL.append((nb_nll(xb, out["mean"], out["theta"]) / xb.sum(1).clamp_min(1)).cpu())
    return {k: torch.cat(v).numpy() for k, v in dict(p=P, g=G, z=Z, nll=NLL).items()}
