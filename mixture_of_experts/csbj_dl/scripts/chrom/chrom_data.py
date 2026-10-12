"""Data access and chromatin feature transforms for the chrom module.

Peaks: binarised -> TF-IDF (log1p(tf * 1e4 * idf), Signac-style) -> truncated SVD (LSI) fitted on training
nuclei only; the first LSI component is dropped when it tracks sequencing depth; components standardised
with training statistics. Gene activity: log1p(CP10k over the encoder genes), standardised with training
statistics. Every transform is fitted on training-fold KPMP nuclei and applied unchanged elsewhere."""
from __future__ import annotations
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.utils.extmath import randomized_svd

from chrom_common import P, result_dir


def binarize(X) -> sp.csr_matrix:
    X = sp.csr_matrix(X, dtype=np.float32)
    X.data[:] = 1.0
    return X


def labelled_obs(c: dict) -> pd.DataFrame:
    lab = pd.read_csv(result_dir("chrom_labels") / "multiome_labels.tsv.gz", sep="\t", index_col=0,
                      dtype={"participant": str})
    return lab


def load_multiome(c: dict, states=("healthy_PT", "injured_PT"), rna_genes: list[str] | None = None,
                  peaks: bool = True):
    """PT nuclei of the merged Multiome objects: obs (with labels and fold), RNA counts (HVG order), peak CSR."""
    lab = labelled_obs(c)
    keep = lab.index[lab.pt_state.isin(states)]
    r = ad.read_h5ad(P(c, "multiome_rna"), backed="r")
    pos = np.where(r.obs_names.isin(keep))[0]
    obs = lab.loc[r.obs_names[pos]].copy()
    for k in ("n_fragments", "tss_enrichment", "frip"):
        obs[k] = r.obs[k].values[pos]
    R = None
    if rna_genes is not None:
        m = ad.read_h5ad(P(c, "multiome_rna"))
        have = [g for g in rna_genes if g in set(m.var_names)]
        R = sp.csr_matrix(m[pos][:, have].X, dtype=np.float32)
        del m
        miss = [g for g in rna_genes if g not in set(have)]
        if miss:                      # measured in every library but under reference-specific Ensembl ids
            R = sp.hstack([R, rna_from_libraries(c, obs.index, miss)]).tocsr()
            order = pd.Index(have + miss).get_indexer(rna_genes)
            R = R[:, order]
    A = None
    if peaks:
        a = ad.read_h5ad(P(c, "multiome_atac"))
        assert (a.obs_names[pos] == obs.index).all()
        A = binarize(a.X[pos]); peak_names = a.var_names.copy(); del a
        return obs, R, A, peak_names
    return obs, R, None, None


def rna_from_libraries(c: dict, obs_names: pd.Index, genes: list[str]) -> sp.csr_matrix:
    """RNA counts of genes absent from the merged object, read by gene symbol from the per-library files
    (each symbol must map to exactly one gene in every library; no gene is filled in)."""
    libdir = P(c, "multiome_lib")
    libs = pd.Index(obs_names).str.rsplit("_", n=1).str[0]
    out = pd.DataFrame(np.nan, index=obs_names, columns=genes, dtype=np.float32)
    for lib in libs.unique():
        r = ad.read_h5ad(libdir / f"{lib}_rna.h5ad")
        sym = r.var.symbol.astype(str) if "symbol" in r.var else pd.Series(r.var_names, index=r.var_names)
        cols = []
        for g in genes:
            hit = np.where(sym.values == g)[0]
            if len(hit) != 1:
                raise SystemExit(f"{g}: {len(hit)} matches in {lib}; gene not measured consistently")
            cols.append(hit[0])
        names = obs_names[libs == lib]
        rows = r.obs_names.get_indexer(names)
        if (rows < 0).any():
            raise SystemExit(f"{(rows < 0).sum()} nuclei of {lib} missing from its RNA file")
        X = sp.csr_matrix(r.X)[np.sort(rows)][:, cols].toarray()
        out.loc[r.obs_names[np.sort(rows)], genes] = X
    if out.isna().any().any():
        raise SystemExit("unfilled nuclei in per-library RNA lookup")
    return sp.csr_matrix(out.values.astype(np.float32))


def load_multiome_gact(c: dict, obs_names: pd.Index, genes: list[str] | None = None) -> ad.AnnData:
    """Gene-activity counts of the given Multiome nuclei, from the per-library files of the data module."""
    libdir = P(c, "multiome_lib")
    libs = pd.Index(obs_names).str.rsplit("_", n=1).str[0].unique()
    parts = []
    for lib in libs:
        g = ad.read_h5ad(libdir / f"{lib}_gact.h5ad")
        g = g[g.obs_names.isin(obs_names)]
        if genes is not None:
            g = g[:, genes]
        parts.append(g.to_memory() if g.is_view else g)
    G = ad.concat(parts, join="inner")
    G = G[pd.Index(obs_names)].copy()
    G.X = sp.csr_matrix(G.X, dtype=np.float32)
    return G


# ---------------- peak TF-IDF + LSI ----------------
def fit_lsi(Xb: sp.csr_matrix, logfrag: np.ndarray, c: dict, seed: int) -> dict:
    A = c["atac"]
    frac = np.asarray(Xb.mean(0)).ravel()
    keep = np.where(frac >= A["min_frac_peak"])[0]
    X = Xb[:, keep]
    idf = (X.shape[0] / np.maximum(np.asarray(X.sum(0)).ravel(), 1.0)).astype(np.float32)
    rng = np.random.default_rng(seed)
    sub = rng.choice(X.shape[0], min(A["lsi_fit_cells"], X.shape[0]), replace=False)
    T = tfidf(X[sub], idf)
    U, S, VT = randomized_svd(T, n_components=A["n_lsi"] + 1, random_state=seed, n_iter=5)
    Z = T @ VT.T
    r1 = np.corrcoef(Z[:, 0], logfrag[sub])[0, 1]
    first = 1 if abs(r1) > A["depth_r_cut"] else 0
    V = VT[first:first + A["n_lsi"]].T.astype(np.float32)              # kept peaks x n_lsi
    Zf = T @ V
    return {"keep": keep, "idf": idf, "V": V, "mu": Zf.mean(0).astype(np.float32),
            "sd": (Zf.std(0) + 1e-6).astype(np.float32), "dropped_first": bool(first), "r_depth_first": float(r1),
            "n_peaks_kept": int(len(keep))}


def tfidf(Xk: sp.csr_matrix, idf: np.ndarray) -> sp.csr_matrix:
    """log1p(tf * 1e4 * idf) on binarised counts restricted to kept peaks (sparse)."""
    n = np.asarray(Xk.sum(1)).ravel()
    T = sp.csr_matrix(Xk, dtype=np.float32, copy=True)
    T = sp.diags((1.0 / np.maximum(n, 1)).astype(np.float32)) @ T
    T = T @ sp.diags(idf * 1e4)
    T.data = np.log1p(T.data)
    return sp.csr_matrix(T)


def lsi_transform(Xb: sp.csr_matrix, L: dict) -> np.ndarray:
    Z = tfidf(Xb[:, L["keep"]], L["idf"]) @ L["V"]
    return ((Z - L["mu"]) / L["sd"]).astype(np.float32)


def align_peaks(X: sp.csr_matrix, src: pd.Index, dst: pd.Index) -> sp.csr_matrix:
    """Reorder peak columns of an external matrix to the Multiome peak order (absent peaks = not measured)."""
    if src.equals(dst):
        return X
    pos = pd.Series(np.arange(len(src)), index=src)
    j = pos.reindex(dst)
    miss = j.isna().sum()
    if miss:
        raise SystemExit(f"{miss} Multiome peaks absent from the external matrix; recount it on the current consensus")
    return X[:, j.values.astype(int)]


# ---------------- gene activity ----------------
def ga_lognorm(X: sp.csr_matrix, scale: float) -> np.ndarray:
    n = np.asarray(X.sum(1)).ravel()
    Y = sp.diags((scale / np.maximum(n, 1)).astype(np.float32)) @ sp.csr_matrix(X, dtype=np.float32)
    return np.log1p(Y.toarray()).astype(np.float32)


def fit_ga(X: sp.csr_matrix, c: dict) -> dict:
    Y = ga_lognorm(X, c["atac"]["ga_scale"])
    return {"mu": Y.mean(0), "sd": Y.std(0) + 1e-3}


def ga_transform(X: sp.csr_matrix, G: dict, c: dict) -> np.ndarray:
    return ((ga_lognorm(X, c["atac"]["ga_scale"]) - G["mu"]) / G["sd"]).astype(np.float32)


def ga_genes(c: dict) -> list[str]:
    """Expression HVGs that have gene activity in KPMP Multiome, GSE211785 and GSE195460."""
    hvg = pd.read_csv(P(c, "hvg"), sep="\t").gene.tolist()
    libdir = P(c, "multiome_lib")
    f = sorted(libdir.glob("*_gact.h5ad"))[0]
    k = set(ad.read_h5ad(f, backed="r").var_names)
    g1 = set(ad.read_h5ad(P(c, "gse211785_atac"), backed="r").var_names)
    g2 = set(ad.read_h5ad(P(c, "gse195460_gact"), backed="r").var_names)
    return [g for g in hvg if g in k and g in g1 and g in g2]
