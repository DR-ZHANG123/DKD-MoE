"""Participant-level statistics and expert matching."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import linear_sum_assignment


def hedges_g(a, b) -> dict:
    """Hedges g (a minus b) with normal-approximation 95% CI and two-sided Wilcoxon rank-sum P."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return {"n1": n1, "n2": n2, "g": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_wilcoxon": np.nan}
    sp = np.sqrt(((n1 - 1) * a.var(ddof=1) + (n2 - 1) * b.var(ddof=1)) / (n1 + n2 - 2))
    d = (a.mean() - b.mean()) / sp if sp > 0 else np.nan
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = d * J
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    p = stats.mannwhitneyu(a, b, alternative="two-sided").pvalue
    return {"n1": n1, "n2": n2, "mean1": a.mean(), "mean2": b.mean(), "g": g,
            "ci_low": g - 1.96 * se, "ci_high": g + 1.96 * se, "p_wilcoxon": p}


def auc(pos, neg) -> float:
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    u = stats.mannwhitneyu(pos, neg, alternative="two-sided").statistic
    return u / (len(pos) * len(neg))


def participant_means(obs: pd.DataFrame, W: np.ndarray, by=("participant",), min_cells: int = 20,
                      prefix: str = "expert_") -> pd.DataFrame:
    """Mean of per-cell weights within participant (x other keys); groups below min_cells dropped."""
    cols = [f"{prefix}{k}" for k in range(W.shape[1])]
    df = pd.concat([obs[list(by)].reset_index(drop=True), pd.DataFrame(W, columns=cols)], axis=1)
    g = df.groupby(list(by), observed=True)
    out = g[cols].mean()
    out["n_cells"] = g.size()
    return out[out.n_cells >= min_cells].reset_index()


def match_experts(ref: np.ndarray, other: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Hungarian matching of expert columns of `other` onto `ref` by correlation of per-cell routing
    probabilities over the same cells. Returns perm (other column for each ref column) and matched r."""
    K = ref.shape[1]
    C = np.corrcoef(ref.T, other.T)[:K, K:]
    C = np.nan_to_num(C, nan=0.0)
    r, c = linear_sum_assignment(-C)
    perm = np.empty(K, dtype=int); perm[r] = c
    return perm, C[np.arange(K), perm]


def group_tests(pm: pd.DataFrame, group_col: str, comps: list[tuple[str, str]], cols: list[str],
                extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for a, b in comps:
        for c in cols:
            r = hedges_g(pm.loc[pm[group_col] == a, c], pm.loc[pm[group_col] == b, c])
            rows.append({**(extra or {}), "comparison": f"{a}_vs_{b}", "feature": c, **r})
    return pd.DataFrame(rows)
