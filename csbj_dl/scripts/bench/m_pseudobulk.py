"""Method 1: participant-level pseudobulk DE (pyDESeq2, ~tech + condition) on KPMP training folds.

Score = signed mean z of the top up- and down-regulated genes (DKD vs HKD). The cell embedding returned
for the integration metric is an uncorrected PCA of log-normalised expression (reference for iLISI).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from bench_score import geneset_score


def deseq_training(ctx) -> pd.DataFrame:
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    cfg = ctx["cfg"]; p = cfg["pseudobulk"]; case, ctrl = cfg["contrast"]["case"], cfg["contrast"]["control"]
    C, meta, ref = ctx["pb_counts"], ctx["meta"], ctx["ref"]
    counts = C.iloc[ref].round().astype(int)
    md = meta.iloc[ref][["tech", "condition"]].copy()
    counts = counts.loc[:, counts.sum(0) > 0]
    factors = ["tech", "condition"] if md.tech.nunique() > 1 else ["condition"]
    dds = DeseqDataSet(counts=counts, metadata=md, design_factors=factors, ref_level=["condition", ctrl],
                       refit_cooks=True, inference=DefaultInference(n_cpus=p["n_cpus"]), quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", case, ctrl], inference=DefaultInference(n_cpus=p["n_cpus"]),
                    quiet=True)
    st.summary()
    return st.results_df


def top_genes(res: pd.DataFrame, n: int, padj_max: float) -> tuple[list, list]:
    r = res.dropna(subset=["stat"])
    sig = r[r.padj.fillna(1) <= padj_max]
    pool = sig if (sig.stat > 0).sum() >= n and (sig.stat < 0).sum() >= n else r
    up = pool[pool.stat > 0].sort_values("stat", ascending=False).index[:n].tolist()
    down = pool[pool.stat < 0].sort_values("stat").index[:n].tolist()
    return up, down


def uncorrected_pca(ctx, n_comp: int = 16) -> np.ndarray:
    import scanpy as sc
    tr = ctx["train"].copy(); cells = ctx["cells"].copy()
    for a in (tr, cells):
        sc.pp.normalize_total(a, target_sum=1e4); sc.pp.log1p(a)
    pca = PCA(n_components=n_comp, random_state=ctx["seed"])
    from bench_data import dense
    pca.fit(dense(tr.X))
    return np.vstack([pca.transform(dense(cells.X[i:i + 50000])) for i in range(0, cells.n_obs, 50000)])


def run(ctx) -> dict:
    p = ctx["cfg"]["pseudobulk"]
    res = deseq_training(ctx)
    up, down = top_genes(res, p["n_top"], p["padj_max"])
    s = geneset_score(ctx["pb_logcpm"], ctx["meta"], up, down, ctx["ref"])
    res = res.assign(selected=np.where(res.index.isin(up), "up", np.where(res.index.isin(down), "down", "")))
    return {"scores": {"pseudobulk_DE_set": s}, "emb": uncorrected_pca(ctx), "emb_name": "PCA_uncorrected",
            "tables": {"de_results": res}}
