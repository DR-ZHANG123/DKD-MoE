"""Leukocyte / ambient-RNA contamination helpers for iPT pseudobulk (stage 14d).

All quantities are computed on participant (or library) level pseudobulk counts; cells are never
treated as replicates."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp


def compartment_map(cfg) -> dict[str, str]:
    """fine cell-type label -> compartment (iPT, PT_healthy, immune, stroma; others -> 'other')."""
    c = cfg["contamination_controls"]; lm = cfg["lineage_map"]
    m = {"iPT": "iPT", **{x: "PT_healthy" for x in c["healthy_pt"]}}
    for lin in c["immune_lineages"]:
        m.update({x: "immune" for x in lm[lin]})
    for lin in c["stroma_lineages"]:
        m.update({x: "stroma" for x in lm[lin]})
    return m


def grouped_pseudobulk(X, keys: np.ndarray, genes) -> tuple[pd.DataFrame, pd.Series]:
    """Sum raw counts of cells sharing a key -> (keys x genes counts, cells per key)."""
    uk, inv = np.unique(keys.astype(str), return_inverse=True)
    ind = sp.csr_matrix((np.ones(len(inv)), (inv, np.arange(len(inv)))), shape=(len(uk), len(inv)))
    S = ind @ (X.tocsr() if sp.issparse(X) else sp.csr_matrix(X))
    S = S.toarray() if sp.issparse(S) else np.asarray(S)
    return pd.DataFrame(S, index=uk, columns=genes), pd.Series(np.bincount(inv), index=uk)


def cpm(pb: pd.DataFrame) -> pd.DataFrame:
    return pb.div(pb.sum(1).replace(0, np.nan), axis=0) * 1e6


def nonpt_markers(pooled: pd.DataFrame, exclude: set, ratio: float, min_cpm: float) -> list[str]:
    """Genes enriched in pooled immune+stroma over pooled PT (iPT + healthy PT) CPM."""
    nonpt = cpm(pooled.loc[["immune", "stroma"]].sum(0).to_frame().T).iloc[0]
    pt = cpm(pooled.loc[["iPT", "PT_healthy"]].sum(0).to_frame().T).iloc[0]
    ok = (nonpt >= min_cpm) & (nonpt >= ratio * (pt + 1))
    return [g for g in nonpt.index[ok] if g not in exclude]


def contamination_indices(ipt: pd.DataFrame, nonpt: pd.DataFrame, n_nonpt: pd.Series, pooled_cpm: pd.DataFrame,
                          genes: dict[str, list[str]], markers: list[str], min_cells: int) -> pd.DataFrame:
    """Per-sample indices on the iPT pseudobulk.

    frac_*       : fraction of iPT counts from a gene list
    c_immune / c_stroma : ratio estimate of the contaminating share, sum iPT CPM / sum pooled compartment
                   CPM over lineage-restricted genes (upper bound: assumes iPT never expresses them)
    r_ambient_all / r_ambient_markers : Pearson r between iPT log2CPM and the same sample's non-PT
                   (immune + stroma) log2CPM across expressed genes / across non-PT marker genes."""
    tot = ipt.sum(1)
    icpm = cpm(ipt)
    out = pd.DataFrame(index=ipt.index)
    for k in ["leukocyte_genes", "leukocyte_strict", "stroma_strict", "ig_genes"]:
        out[f"frac_{k}"] = ipt[[g for g in genes[k] if g in ipt.columns]].sum(1) / tot
    out["frac_nonpt_markers"] = ipt[[g for g in markers if g in ipt.columns]].sum(1) / tot
    for comp, key in [("immune", "leukocyte_strict"), ("stroma", "stroma_strict")]:
        gg = [g for g in genes[key] if g in ipt.columns]
        out[f"c_{comp}"] = icpm[gg].sum(1) / pooled_cpm.loc[comp, gg].sum()
    lc_i = np.log2(icpm + 1); lc_n = np.log2(cpm(nonpt) + 1)
    expressed = lc_i.columns[(lc_i.mean(0) > 1)]
    mk = [g for g in markers if g in lc_i.columns]
    for s in ipt.index:
        ok = s in nonpt.index and n_nonpt.get(s, 0) >= min_cells
        out.loc[s, "n_nonpt_cells"] = n_nonpt.get(s, 0)
        out.loc[s, "r_ambient_all"] = np.corrcoef(lc_i.loc[s, expressed], lc_n.loc[s, expressed])[0, 1] if ok else np.nan
        out.loc[s, "r_ambient_markers"] = np.corrcoef(lc_i.loc[s, mk], lc_n.loc[s, mk])[0, 1] if ok else np.nan
    return out


def expected_contamination_share(ipt: pd.DataFrame, idx: pd.DataFrame, pooled_cpm: pd.DataFrame,
                                 genes: list[str]) -> pd.DataFrame:
    """Per sample x gene: CPM expected in iPT from immune / stromal contamination (c x compartment CPM)
    and its share of the observed iPT CPM."""
    icpm = cpm(ipt)
    rows = []
    for g in [x for x in genes if x in ipt.columns]:
        for s in ipt.index:
            e_i = idx.loc[s, "c_immune"] * pooled_cpm.loc["immune", g]
            e_s = idx.loc[s, "c_stroma"] * pooled_cpm.loc["stroma", g]
            o = icpm.loc[s, g]
            rows.append({"sample": s, "gene": g, "obs_cpm": o, "exp_immune_cpm": e_i, "exp_stroma_cpm": e_s,
                         "share_immune": e_i / o if o > 0 else np.nan, "share_stroma": e_s / o if o > 0 else np.nan})
    return pd.DataFrame(rows)


def residualize(z: pd.DataFrame, cov: pd.Series) -> pd.DataFrame:
    """Residuals of each column of z on an intercept + cov (samples in common)."""
    X = np.column_stack([np.ones(len(cov)), (cov - cov.mean()) / cov.std(ddof=1)])
    B, *_ = np.linalg.lstsq(X, z.loc[cov.index].values, rcond=None)
    return pd.DataFrame(z.loc[cov.index].values - X @ B, index=cov.index, columns=z.columns)
