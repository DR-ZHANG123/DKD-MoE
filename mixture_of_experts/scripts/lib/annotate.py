"""Reference-based cell-type label transfer and donor-level programme scoring.

Reference: discovery SN libraries (fine cell types). Classifier: multinomial logistic regression
on the top PCs of log-normalised shared HVGs, trained with class balancing. Accuracy is
estimated by donor-grouped cross-validation on the reference itself."""
from __future__ import annotations
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler


def _lognorm(a: ad.AnnData, genes) -> np.ndarray:
    b = a[:, genes].copy()
    sc.pp.normalize_total(b, target_sum=1e4, exclude_highly_expressed=False)
    sc.pp.log1p(b)
    X = b.X.toarray() if hasattr(b.X, "toarray") else np.asarray(b.X)
    return X


def fit_reference(ref: ad.AnnData, query_genes, label_key: str, n_hvg: int = 3000, n_pcs: int = 50,
                  max_per_class: int = 3000, seed: int = 0):
    genes = [g for g in ref.var_names if g in set(query_genes)]
    r = ref[:, genes].copy()
    sc.pp.highly_variable_genes(r, n_top_genes=n_hvg, flavor="seurat_v3", subset=False)
    hvg = list(r.var_names[r.var.highly_variable])
    rng = np.random.default_rng(seed)
    idx = np.concatenate([rng.choice(np.where(ref.obs[label_key].values == c)[0],
                                     min(max_per_class, (ref.obs[label_key] == c).sum()), replace=False)
                          for c in ref.obs[label_key].unique()])
    X = _lognorm(ref[idx], hvg)
    y = ref.obs[label_key].values[idx]; groups = ref.obs["donor"].values[idx]
    scaler = StandardScaler().fit(X); pca = PCA(n_pcs, random_state=seed).fit(scaler.transform(X))
    Z = pca.transform(scaler.transform(X))
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    # donor-grouped CV accuracy on the reference
    cv = []
    for tr, te in GroupKFold(n_splits=min(5, len(set(groups)))).split(Z, y, groups):
        m = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0).fit(Z[tr], y[tr])
        p = m.predict(Z[te]); cv.append(pd.DataFrame({"true": y[te], "pred": p}))
    cv = pd.concat(cv)
    clf.fit(Z, y)
    return {"genes": hvg, "scaler": scaler, "pca": pca, "clf": clf, "cv": cv}


def predict(model: dict, q: ad.AnnData, chunk: int = 20000) -> pd.DataFrame:
    out = []
    for s in range(0, q.n_obs, chunk):
        X = _lognorm(q[s:s + chunk], model["genes"])
        Z = model["pca"].transform(model["scaler"].transform(X))
        P = model["clf"].predict_proba(Z)
        out.append(pd.DataFrame({"label": model["clf"].classes_[P.argmax(1)], "max_prob": P.max(1)},
                                index=q.obs_names[s:s + chunk]))
    return pd.concat(out)


def donor_pseudobulk(a: ad.AnnData, mask, donor_key: str = "donor") -> pd.DataFrame:
    """Sum raw counts of masked cells per donor -> donors x genes."""
    b = a[np.asarray(mask)]
    d = b.obs[donor_key].astype(str).values
    X = b.X.tocsr() if hasattr(b.X, "tocsr") else b.X
    rows = {k: np.asarray(X[d == k].sum(0)).ravel() for k in np.unique(d)}
    return pd.DataFrame(rows, index=b.var_names).T


def zmatrix(pb: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """log2 CPM, genes expressed (>1 log2CPM) in >=50% donors, z-scored across donors."""
    cpm = np.log2(pb.div(pb.sum(1), axis=0) * 1e6 + 1)
    expressed = cpm.columns[(cpm > 1).mean(0) >= 0.5]
    z = (cpm[expressed] - cpm[expressed].mean()) / cpm[expressed].std(ddof=1).replace(0, np.nan)
    z = z.loc[:, z.notna().all()]
    return z, cpm[z.columns].mean()


def programme_scores(pb: pd.DataFrame, gene_sets: dict[str, list[str]]) -> tuple[pd.DataFrame, dict]:
    """mean z over the set's measured genes (see zmatrix)."""
    z, _ = zmatrix(pb)
    sc_, cov = {}, {}
    for k, g in gene_sets.items():
        gg = [x for x in g if x in z.columns]
        cov[k] = {"n_set": len(g), "n_measured": len(gg)}
        sc_[k] = z[gg].mean(1) if gg else pd.Series(np.nan, index=z.index)
    return pd.DataFrame(sc_), cov
