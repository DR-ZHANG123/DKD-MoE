"""Helpers for the in-silico expert intervention: cell selection, gate re-assignment inside the model's own
forward pass, decoded log expression, and the participant-level readout-gene correlation of the signature.

Intervention: in the top-2 renormalised routing weights g, the weight of a source expert is moved onto a
destination expert (g'_dst = g_dst + g_src, g'_src = 0; total weight unchanged). Everything else in the
forward pass is unchanged (same h, same dense routing probabilities p, same technology, z = mu).
Decoded expression is log rho (natural log of the decoder's gene proportions); with the cell's own library
size, log expected counts differ from log rho by a per-cell constant, so per-gene changes are identical.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import cond_logodds  # noqa: E402
from transfer_common import kpmp_pseudobulk, expressed, fisher_combined_spearman, cell_logodds  # noqa: E402


def select_cells(obs: pd.DataFrame, group: str, cap: int, min_cells: int, rng) -> pd.DataFrame:
    """Injured-PT cells of one diagnosis; participant x technology units with >= min_cells cells, at most
    `cap` cells sampled per unit. Returns row index + participant + tech."""
    o = obs[["participant", "group", "tech", "pt_state"]].astype(str).reset_index(drop=True)
    o = o[(o.pt_state == "injured_PT") & (o.group == group)]
    keep = []
    for _, d in o.groupby(["participant", "tech"]):
        if len(d) < min_cells:
            continue
        keep.append(np.sort(rng.choice(d.index.values, min(cap, len(d)), replace=False)))
    idx = np.concatenate(keep)
    return pd.DataFrame({"row": idx, "participant": o.loc[idx, "participant"].values, "tech": o.loc[idx, "tech"].values})


def moved_gate(src: int | None, dst: int | None):
    """Function g -> g' moving the weight of expert `src` onto expert `dst` (identity when src is None)."""
    def f(g):
        if src is None:
            return g
        g2 = g.clone()
        g2[:, dst] = g2[:, dst] + g2[:, src]
        g2[:, src] = 0.0
        return g2
    return f


@torch.no_grad()
def forward_with_gate(model, x, tech, gate_fn) -> dict:
    """Model forward (deterministic, z = mu) with the sparse gate post-processed by `gate_fn`."""
    orig = model.sparse_gate
    model.sparse_gate = lambda p: gate_fn(orig(p))
    try:
        out = model(x, tech, sample=False)
    finally:
        model.sparse_gate = orig
    return {"logrho": torch.log(out["rho"].clamp_min(1e-30)), "p": out["p"], "g": out["g"],
            "logodds": cond_logodds(model, out["p"])}


@torch.no_grad()
def run_intervention(model, data, rows: np.ndarray, src: int, dst: int, bs: int = 2048) -> dict:
    """Per-cell decoded log rho before and after, moved weight, and the p / readout sanity differences."""
    D, L0, W, dp, dlo = [], [], [], 0.0, 0.0
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]
        x = data.rows(r); t = data.tech[torch.as_tensor(r, device=data.device)]
        a = forward_with_gate(model, x, t, moved_gate(None, None))
        b = forward_with_gate(model, x, t, moved_gate(src, dst))
        D.append((b["logrho"] - a["logrho"]).cpu()); L0.append(a["logrho"].cpu()); W.append(a["g"][:, src].cpu())
        dp = max(dp, float((b["p"] - a["p"]).abs().max())); dlo = max(dlo, float((b["logodds"] - a["logodds"]).abs().max()))
    return {"delta": torch.cat(D).numpy(), "logrho0": torch.cat(L0).numpy(), "moved": torch.cat(W).numpy(),
            "max_abs_dp": dp, "max_abs_dlogodds": dlo}


def participant_means(cells: pd.DataFrame, M: np.ndarray) -> pd.DataFrame:
    """Participant x technology means of per-cell rows of M, then mean over technologies (readout convention)."""
    df = pd.DataFrame(M); df["participant"] = cells.participant.values; df["tech"] = cells.tech.values
    return df.groupby(["participant", "tech"]).mean().groupby("participant").mean()


def readout_gene_correlation(data, models, rc) -> tuple[pd.Series, pd.Series, list[str]]:
    """Participant-level Spearman(readout, injured-PT pseudobulk log2 CPM) within sc and sn combined by Fisher
    z, readout from the final models on all KPMP participants (the construction of the transferable
    signature). Returns (r over expressed genes, r over all model genes, expressed genes)."""
    Lp, Mp = kpmp_pseudobulk(data, rc); genes = expressed(Lp, Mp, rc)
    st = data.obs.pt_state.astype(str).values
    idx = np.where(st == "injured_PT")[0]
    lo = np.mean([cell_logodds(m, data, idx) for m in models], 0)
    o = data.obs.iloc[idx]
    r = pd.Series(lo, index=(o.participant.astype(str) + "|" + o.tech.astype(str)).values).groupby(level=0).mean()
    r = r[r.index.isin(Mp.index)]
    return (fisher_combined_spearman(Lp, Mp, r, genes), fisher_combined_spearman(Lp, Mp, r, data.genes), genes)
