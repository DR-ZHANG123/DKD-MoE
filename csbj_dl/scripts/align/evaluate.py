"""Cross-modal retrieval metrics (section and participant are the units; spots are never pooled as
independent samples for inference)."""
from __future__ import annotations
import numpy as np
import pandas as pd


def _ranks(sim: np.ndarray) -> np.ndarray:
    d = np.diag(sim)
    return (sim > d[:, None]).sum(1)


def within_section(zi, ze, meta: pd.DataFrame, topk, radius_um: float = 150.0) -> list[dict]:
    """meta rows aligned with zi/ze. For each section: image->expression and expression->image
    ranks of the true partner among all spots of that section; top-k accuracy, chance = k/n;
    top-1 hit within radius_um of the true spot (neighbouring spots share tissue context)."""
    out = []
    for sec, ix in meta.groupby("section").indices.items():
        n = len(ix)
        if n < 2:
            continue
        sim = zi[ix] @ ze[ix].T
        r_ie, r_ei = _ranks(sim), _ranks(sim.T)
        m = meta.iloc[ix]
        um = m.um_per_px.values[0]
        xy = np.c_[m.px_row.values, m.px_col.values] * um
        j = sim.argmax(1)
        near = np.sqrt(((xy[j] - xy) ** 2).sum(1)) <= radius_um
        dd = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)) <= radius_um
        row = {"section": sec, "participant": m.participant.values[0], "cohort": m.cohort.values[0],
               "group": m.group.values[0], "n_spots": n,
               "median_rank_pct_img2expr": float(np.median(r_ie) / (n - 1)),
               "median_rank_pct_expr2img": float(np.median(r_ei) / (n - 1)),
               f"top1_within{int(radius_um)}um_img2expr": float(near.mean()),
               f"chance_top1_within{int(radius_um)}um": float(dd.sum(1).mean() / n)}
        for k in topk:
            row[f"top{k}_img2expr"] = float((r_ie < k).mean())
            row[f"top{k}_expr2img"] = float((r_ei < k).mean())
            row[f"chance_top{k}"] = min(k, n) / n
        out.append(row)
    return out


def global_pool(zi, ze, meta: pd.DataFrame, topk, pool: int, n_queries: int, rng) -> dict:
    """Queries drawn from all evaluation spots; candidates = true partner + (pool-1) random spots
    from any evaluation section (so section identity can help, unlike within_section)."""
    n = len(zi)
    pool = min(pool, n)
    q = rng.choice(n, min(n_queries, n), replace=False)
    hits = {k: [] for k in topk}; same_sec = []
    sec = meta.section.values
    for i in q:
        cand = rng.choice(np.delete(np.arange(n), i), pool - 1, replace=False)
        c = np.r_[i, cand]
        s = ze[c] @ zi[i]
        r = int((s[1:] > s[0]).sum())
        for k in topk:
            hits[k].append(r < k)
        same_sec.append(sec[c[np.argmax(s)]] == sec[i])
    out = {f"top{k}_img2expr": float(np.mean(v)) for k, v in hits.items()}
    out.update({f"chance_top{k}": k / pool for k in topk})
    out.update({"pool": pool, "n_queries": len(q), "top1_same_section": float(np.mean(same_sec))})
    return out


def by_participant(rows: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """Section metrics -> participant means (participants with several sections are averaged)."""
    keys = [c for c in ["fold", "seed", "mode", "model", "eval_set", "participant", "cohort", "group"] if c in rows]
    return rows.groupby(keys, dropna=False)[cols].mean().reset_index()
