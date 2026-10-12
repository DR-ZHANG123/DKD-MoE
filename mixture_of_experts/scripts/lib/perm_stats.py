"""Vectorised statistics for participant-label permutation of a two-group pseudobulk contrast.

nb_glm_lfc: negative-binomial GLM group coefficient with DESeq2 size factors and dispersions held
fixed at the values DESeq2 estimated on the observed labels (the same IRLS model DESeq2 fits:
log mu = log sf + X beta), solved for many label vectors at once."""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats


def median_ratio_sf(counts: np.ndarray) -> np.ndarray:
    """DESeq2 median-of-ratios size factors (genes with no zero count)."""
    x = counts.astype(float)
    lx = np.log(x[:, (x > 0).all(0)])
    return np.exp(np.median(lx - lx.mean(0), axis=1))


def mean_ratio_lfc(norm: np.ndarray, lab: np.ndarray, pc: float) -> np.ndarray:
    """lab (P, n) bool (True = group a) -> (P, G) log2 ratio of group means of normalised counts."""
    la = lab.astype(float); lb = 1 - la
    ma = la @ norm / la.sum(1, keepdims=True); mb = lb @ norm / lb.sum(1, keepdims=True)
    return np.log2(ma + pc) - np.log2(mb + pc)


def deseq_fit(counts: pd.DataFrame, md: pd.DataFrame, covars: list[str], a: str, b: str):
    """pydeseq2 fit with design ~ sex + covars + group (as 07c) -> size factors, dispersions, log2FC."""
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    m = md[["group", "sex", *covars]].copy()
    m["sex"] = m.sex.where(m.sex.isin(["Male", "Female"]), "Unknown")
    m["group"] = m.group.str.replace("_", "")
    for c in covars:
        m[c] = (m[c] - m[c].mean()) / m[c].std(ddof=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        dds = DeseqDataSet(counts=counts.astype(int), metadata=m, design_factors=["sex", *covars, "group"],
                           continuous_factors=covars or None, ref_level=["group", b.replace("_", "")],
                           n_cpus=16, quiet=True)
        dds.deseq2()
        st = DeseqStats(dds, contrast=["group", a.replace("_", ""), b.replace("_", "")], quiet=True)
        st.summary()
    return (np.asarray(dds.obsm["size_factors"]), pd.Series(np.asarray(dds.varm["dispersions"]), index=counts.columns),
            st.results_df.log2FoldChange)


def nb_glm_lfc(y: np.ndarray, sf: np.ndarray, disp: np.ndarray, X0: np.ndarray, lab: np.ndarray,
               n_iter: int = 30, ridge: float = 1e-6, chunk: int = 100) -> np.ndarray:
    """log2 group coefficient of NB GLM y ~ X0 + group, offset log sf, fixed dispersion.
    y (n, G) counts; X0 (n, k0) covariates incl. intercept; lab (P, n) bool -> (P, G)."""
    n, G = y.shape
    off = np.log(sf)[:, None]                                      # (n, 1)
    yb = y[None]                                                   # (1, n, G)
    out = np.empty((lab.shape[0], G))
    for s in range(0, lab.shape[0], chunk):
        L = lab[s:s + chunk].astype(float); P = L.shape[0]
        X = np.concatenate([np.broadcast_to(X0, (P, n, X0.shape[1])), L[:, :, None]], 2)   # (P, n, k)
        k = X.shape[2]
        beta = np.zeros((P, G, k)); beta[:, :, 0] = np.log(np.maximum((y / sf[:, None]).mean(0), 1e-8))[None]
        for _ in range(n_iter):
            eta = np.clip(np.einsum("pnk,pgk->png", X, beta) + off[None], -30, 30)
            mu = np.exp(eta)
            w = mu / (1 + disp[None, None] * mu)
            z = eta - off[None] + (yb - mu) / mu
            A = np.einsum("pnk,png,pnj->pgkj", X, w, X) + ridge * np.eye(k)
            rhs = np.einsum("pnk,png->pgk", X, w * z)
            new = np.linalg.solve(A, rhs[..., None])[..., 0]
            done = np.abs(new - beta).max() < 1e-6
            beta = new
            if done:
                break
        out[s:s + P] = beta[:, :, -1] / np.log(2)
    return out


def permute_labels(lab: np.ndarray, strata: np.ndarray, n: int, rng) -> np.ndarray:
    """n permutations of lab, shuffled within strata (group sizes per stratum kept)."""
    out = np.empty((n, len(lab)), bool)
    groups = [np.where(strata == s)[0] for s in np.unique(strata)]
    for i in range(n):
        new = lab.copy()
        for idx in groups:
            new[idx] = lab[rng.permutation(idx)]
        out[i] = new
    return out


def row_spearman(ref: np.ndarray, M: np.ndarray) -> np.ndarray:
    r0 = stats.rankdata(ref); r0 = (r0 - r0.mean()) / r0.std()
    R = stats.rankdata(M, axis=1); R = (R - R.mean(1, keepdims=True)) / R.std(1, keepdims=True)
    return (R * r0).mean(1)


def emp_p(null: np.ndarray, obs: float) -> float:
    return float((np.sum(null >= obs) + 1) / (len(null) + 1))
