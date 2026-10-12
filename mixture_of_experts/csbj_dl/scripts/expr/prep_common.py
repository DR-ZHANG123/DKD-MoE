"""Shared preparation helpers: label-source resolution, PT-state mapping, gene universe, HVG selection."""
from __future__ import annotations
import re, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import ROOT, path  # noqa: E402


def resolve_label_source(cfg: dict) -> dict:
    """First configured source whose sc and sn files both exist (preferred order in expr.yaml)."""
    for s in cfg["labels"]["sources"]:
        if (ROOT / s["sc"]).exists() and (ROOT / s["sn"]).exists():
            return {"name": s["name"], "sc": str(ROOT / s["sc"]), "sn": str(ROOT / s["sn"])}
    raise SystemExit("no label source available")


def pt_state(labels: pd.Series, cfg: dict) -> pd.Series:
    """Map fine labels to healthy_PT / injured_PT / NaN (non-PT)."""
    L = cfg["labels"]
    healthy = set(L["healthy_pt"]) | set(L["healthy_pt_alias"])
    injured = set(L["injured_pt"]) | set(L["injured_pt_alias"])
    lab = labels.astype(str)
    out = pd.Series(np.nan, index=labels.index, dtype=object)
    out[lab.isin(healthy)] = "healthy_PT"
    out[lab.isin(injured)] = "injured_PT"
    return out


def read_labels(fn: str, cfg: dict) -> pd.DataFrame:
    lab = pd.read_csv(fn, sep="\t", index_col=0)
    col = cfg["labels"]["label_column"]
    if col not in lab.columns:
        cand = [c for c in lab.columns if c.lower() in ("label", "celltype", "cell_type", "subclass_l2", "subclass.l2")]
        if not cand:
            raise SystemExit(f"no label column in {fn}: {list(lab.columns)}")
        col = cand[0]
    out = pd.DataFrame({"label": lab[col].astype(str)}, index=lab.index.astype(str))
    pcol = [c for c in lab.columns if c in ("max_prob", "prob", "score")]
    out["label_prob"] = lab[pcol[0]].astype(float) if pcol else np.nan
    return out


def gene_universe(cfg: dict) -> list[str]:
    """Genes measured (by symbol) in KPMP sc, KPMP sn and both external cohorts, HGNC-approved,
    minus a-priori technical genes. Uses gene names only, never expression values of external data."""
    sets = [set(ad.read_h5ad(path(cfg, k), backed="r").var_names)
            for k in ("kpmp_sc", "kpmp_sn", "gse211785", "gse195460")]
    uni = set.intersection(*sets)
    hg = pd.read_csv(ROOT / "data/raw/reference/hgnc_complete_set.txt", sep="\t", usecols=["symbol"], low_memory=False)
    uni &= set(hg.symbol.astype(str))
    rx = re.compile(cfg["prep"]["exclude_regex"])
    return sorted(g for g in uni if not rx.search(g))


def total_counts(X) -> np.ndarray:
    return np.asarray(X.sum(1)).ravel().astype(np.float32)


def select_hvg(adata: ad.AnnData, cfg: dict, seed: int) -> list[str]:
    """seurat_v3 HVGs on raw counts, batch = technology, within KPMP PT cells only.
    A participant-stratified subsample keeps large participants from dominating."""
    import scanpy as sc
    n = cfg["prep"]["hvg_sample_cells"]
    rng = np.random.default_rng(seed)
    obs = adata.obs
    if adata.n_obs > n:
        per = max(1, n // obs.participant.nunique())
        idx = np.concatenate([rng.permutation(np.where(obs.participant.values == p)[0])[:per]
                              for p in obs.participant.unique()])
        sub = adata[np.sort(idx)].copy()
    else:
        sub = adata.copy()
    sc.pp.highly_variable_genes(sub, n_top_genes=cfg["prep"]["n_hvg"], flavor="seurat_v3", batch_key="tech",
                                subset=False)
    return sorted(sub.var_names[sub.var.highly_variable.values].tolist())


def subset_genes(adata: ad.AnnData, genes: list[str]) -> ad.AnnData:
    a = adata[:, genes].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    return a
