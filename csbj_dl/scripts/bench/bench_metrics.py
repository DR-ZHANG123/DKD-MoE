"""Donor-level effect sizes (Hedges g, AUC with CIs) and the integration LISI used by the bench."""
from __future__ import annotations
import numpy as np
from scipy import stats
from sklearn.neighbors import NearestNeighbors


def hedges_g(x: np.ndarray, y: np.ndarray, ci: float = 0.95) -> tuple[float, float, float]:
    """Hedges g (x minus y) with the large-sample normal CI (Hedges & Olkin)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 2 or n2 < 2:
        return np.nan, np.nan, np.nan
    sp = np.sqrt(((n1 - 1) * x.var(ddof=1) + (n2 - 1) * y.var(ddof=1)) / (n1 + n2 - 2))
    if sp == 0:
        return np.nan, np.nan, np.nan
    J = 1 - 3 / (4 * (n1 + n2) - 9)
    g = J * (x.mean() - y.mean()) / sp
    se = np.sqrt((n1 + n2) / (n1 * n2) + g ** 2 / (2 * (n1 + n2)))
    zc = stats.norm.ppf(0.5 + ci / 2)
    return g, g - zc * se, g + zc * se


def auc_ci(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int, ci: float = 0.95) -> tuple[float, float, float]:
    """AUC of x (cases) vs y (controls); stratified donor-level bootstrap percentile CI.
    Vectorised: with the pairwise win matrix C, a bootstrap AUC is wx' C wy / (n1 n2) for resampling counts."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n1, n2 = len(x), len(y)
    if n1 < 1 or n2 < 1:
        return np.nan, np.nan, np.nan
    C = (x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :])
    auc = float(C.mean())
    rng = np.random.default_rng(seed)
    wx = rng.multinomial(n1, np.full(n1, 1 / n1), size=n_boot)
    wy = rng.multinomial(n2, np.full(n2, 1 / n2), size=n_boot)
    bs = ((wx @ C) * wy).sum(1) / (n1 * n2)
    a = (1 - ci) / 2
    return auc, float(np.quantile(bs, a)), float(np.quantile(bs, 1 - a))


def _beta_search(d: np.ndarray, perplexity: float, n_iter: int = 50) -> np.ndarray:
    """Per-row Gaussian precision so that the neighbour distribution has the target perplexity."""
    n = d.shape[0]
    beta = np.ones(n); lo = np.full(n, -np.inf); hi = np.full(n, np.inf)
    logU = np.log(perplexity)
    for _ in range(n_iter):
        P = np.exp(-d * beta[:, None]); sP = P.sum(1) + 1e-12
        H = np.log(sP) + beta * (d * P).sum(1) / sP
        diff = H - logU
        up = diff > 0
        lo[up] = beta[up]; beta[up] = np.where(np.isinf(hi[up]), beta[up] * 2, (beta[up] + hi[up]) / 2)
        hi[~up] = beta[~up]; beta[~up] = np.where(np.isinf(lo[~up]), beta[~up] / 2, (beta[~up] + lo[~up]) / 2)
    return beta


def ilisi(emb: np.ndarray, labels: np.ndarray, k: int = 90, perplexity: float = 30) -> tuple[float, float]:
    """Integration LISI rescaled to [0, 1] ((LISI - 1) / (B - 1)); 1 = perfectly mixed.
    Returns (median, mean) over cells; the median is the scIB convention."""
    labels = np.asarray(labels); cats, codes = np.unique(labels, return_inverse=True)
    B = len(cats)
    if B < 2:
        return np.nan, np.nan
    k = min(k, len(emb) - 1)
    dist, idx = NearestNeighbors(n_neighbors=k + 1).fit(emb).kneighbors(emb)
    d, idx = dist[:, 1:] ** 2, idx[:, 1:]
    d = d / (d.mean(1, keepdims=True) + 1e-12)
    P = np.exp(-d * _beta_search(d, perplexity)[:, None]); P /= P.sum(1, keepdims=True)
    lab = codes[idx]
    simpson = np.zeros(len(emb))
    for b in range(B):
        simpson += (P * (lab == b)).sum(1) ** 2
    lisi = 1 / simpson
    return float((np.median(lisi) - 1) / (B - 1)), float((np.mean(lisi) - 1) / (B - 1))
