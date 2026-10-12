"""Transfer-signature helpers: KPMP injured-PT pseudobulk, signature scoring, fold-model readout and IG rankings."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from model import log_norm, cond_logodds  # noqa: E402
from train import build_model  # noqa: E402
from interpret_common import integrated_gradients  # noqa: E402


def kpmp_pseudobulk(data, rc) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Injured-PT pseudobulk per participant x tech: log2 CPM (over the HVGs) and a meta table."""
    o = data.obs[["participant", "group", "tech", "pt_state", "fold"]].astype(str).reset_index(drop=True)
    o = o[o.pt_state == rc["cells"]]
    rows, meta = [], []
    for (p, t), d in o.groupby(["participant", "tech"]):
        if len(d) < rc["min_cells"]:
            continue
        rows.append(data.rows(d.index.values).sum(0).cpu().numpy())
        meta.append({"participant": p, "tech": t, "group": d.group.iloc[0], "fold": int(d.fold.iloc[0]), "n_cells": len(d)})
    M = pd.DataFrame(meta); M.index = M.participant + "|" + M.tech
    C = pd.DataFrame(np.array(rows), index=M.index, columns=data.genes)
    L = np.log2(C.div(C.sum(1), axis=0) * 1e6 + 1)
    return L, M


def expressed(L: pd.DataFrame, M: pd.DataFrame, rc) -> list[str]:
    ok = None
    for t in M.tech.unique():
        e = ((L[M.tech == t] > np.log2(rc["min_cpm"] + 1)).mean() >= rc["min_frac_expressed"])
        ok = e if ok is None else ok & e
    return list(L.columns[ok.values])


def score_kpmp(L, M, up, down) -> pd.Series:
    """z across all KPMP participants within technology (label-free); participant x tech score."""
    out = []
    for t in M.tech.unique():
        x = L[M.tech == t]; z = (x - x.mean()) / x.std(ddof=1).replace(0, np.nan)
        u = [g for g in up if g in z.columns]; d = [g for g in down if g in z.columns]
        out.append(z[u].mean(1) - z[d].mean(1))
    return pd.concat(out)


def top_bottom(s: pd.Series, n_up, n_down):
    s = s.dropna().sort_values(ascending=False)
    return s.index[:n_up].tolist(), s.index[::-1][:n_down].tolist()


def fisher_combined_spearman(L, M, readout: pd.Series, genes) -> pd.Series:
    """Spearman(readout, gene) within each technology, combined by Fisher z weighted by n - 3."""
    num, den = 0, 0
    for t in M.tech.unique():
        idx = M.index[(M.tech == t) & M.index.isin(readout.index)]
        if len(idx) < 8:
            continue
        r = L.loc[idx, genes].rank().corrwith(readout[idx].rank())
        num = num + np.arctanh(r.clip(-0.999, 0.999)) * (len(idx) - 3); den += len(idx) - 3
    return np.tanh(num / den)


def load_state(cfg, path, K, device):
    m = build_model(cfg, 3000, K).to(device)
    sd = torch.load(path, map_location=device)
    m.load_state_dict(sd["state_dict"] if "state_dict" in sd else sd); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


@torch.no_grad()
def cell_logodds(model, data, idx, bs=8192):
    out = []
    for s in range(0, len(idx), bs):
        out.append(cond_logodds(model, model.gate_probs(model.encode_h(log_norm(data.rows(idx[s:s + bs]))))[0]).cpu())
    return torch.cat(out).numpy()


def ig_logodds(model, xl, x0, steps):
    fn = lambda z: cond_logodds(model, model.gate_probs(model.encode_h(z))[0])
    a = integrated_gradients(fn, xl, x0, steps).mean(0).cpu().numpy()
    return (a - a.mean()) / a.std()
