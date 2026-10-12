"""Donor-level group comparison helpers."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def hedges_g(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = np.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    j = 1 - 3 / (4 * (na + nb) - 9)
    return float((a.mean() - b.mean()) / sp * j) if sp > 0 else np.nan


def compare(x: pd.Series, groups: pd.Series, a: str, b: str) -> dict:
    xa, xb = x[groups == a].dropna(), x[groups == b].dropna()
    if len(xa) < 2 or len(xb) < 2:
        return {"n_a": len(xa), "n_b": len(xb)}
    return {"n_a": len(xa), "n_b": len(xb), "mean_a": xa.mean(), "mean_b": xb.mean(),
            "hedges_g": hedges_g(xa, xb), "welch_p": stats.ttest_ind(xa, xb, equal_var=False).pvalue,
            "mwu_p": stats.mannwhitneyu(xa, xb, alternative="two-sided").pvalue}


def matched_random_null(z: pd.DataFrame, genes: list[str], groups: pd.Series, a: str, b: str,
                        mean_expr: pd.Series, n_perm: int = 1000, n_bins: int = 20, seed: int = 0) -> dict:
    """Empirical position of the observed Hedges g among random gene sets matched on mean expression."""
    rng = np.random.default_rng(seed)
    genes = [g for g in genes if g in z.columns]
    bins = pd.qcut(mean_expr[z.columns].rank(method="first"), n_bins, labels=False)
    pool = {k: list(v.index) for k, v in bins.groupby(bins)}
    obs = hedges_g(z.loc[groups == a, genes].mean(1), z.loc[groups == b, genes].mean(1))
    null = []
    for _ in range(n_perm):
        rs = [rng.choice(pool[bins[g]]) for g in genes]
        s = z[rs].mean(1)
        null.append(hedges_g(s[groups == a], s[groups == b]))
    null = np.asarray(null)
    return {"g_obs": obs, "null_mean": float(np.nanmean(null)), "null_sd": float(np.nanstd(null)),
            "p_emp_two_sided": float((np.sum(np.abs(null) >= abs(obs)) + 1) / (n_perm + 1)),
            "p_emp_greater": float((np.sum(null >= obs) + 1) / (n_perm + 1))}
