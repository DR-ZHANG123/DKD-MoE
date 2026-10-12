"""Shared helpers for the Visium stages (10a GSE211785, 10b KPMP).

- ipt_markers(): iPT-state and healthy-PT marker genes from the discovery atlas, defined once and
  used only to locate injured-PT-rich spots. Rule:
  SN-stratum, QC-passing libraries with >= min_cells iPT and >= min_cells PT_S1-3 cells; per library
  log2 CPM of the iPT pseudobulk minus that of the pooled PT_S1-3 pseudobulk; HGNC-approved
  protein-coding genes with paired |t| > marker_min_t and mean log2 CPM > min_log2cpm in the
  enriched state, ranked by mean difference; top n_markers each way. Genes of any marker programme set
  are removed so that spot selection is not circular with the programme scores.
- barcode_array_map(): Visium v1 barcode -> (array_row, array_col), read from the KPMP
  tissue_positions files (every file lists all 4,992 spots; the map is identical across slides).
- spot_scores(): per-gene z of log1p(CP10k) across spots, mean over measured genes of each set.
"""
from __future__ import annotations
import glob, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT

SETS_USED = ["DKDiPT_up", "DKDiPT_down", "DEonly_up", "DEonly_down", "SharedInjury_up", "SharedInjury_down"]


def marker_sets() -> dict[str, list[str]]:
    gs = pd.read_csv(ROOT / "results/05_lock_programme/gene_sets.tsv", sep="\t")
    return {k: v.gene.tolist() for k, v in gs.groupby("set_id")}


def ipt_markers(cfg: dict) -> pd.DataFrame:
    v = cfg["visium"]
    pb = ad.read_h5ad(ROOT / cfg["paths"]["processed"] / "pb_fine.h5ad")
    q = pd.read_csv(ROOT / "results/02c_library_qc/library_qc.tsv", sep="\t").set_index("library")
    o = pb.obs
    ok = o.library.map(q.batch).eq("SN") & o.library.map(q.qc_pass).astype(bool)
    X = pb.X.toarray() if hasattr(pb.X, "toarray") else np.asarray(pb.X)
    diffs, ipt_mean, pt_mean = [], [], []
    for lib in sorted(o.library[ok].unique()):
        i = np.where(ok & (o.library == lib) & (o.celltype == "iPT") & (o.n_cells >= v["marker_min_cells"]))[0]
        h = np.where(ok & (o.library == lib) & o.celltype.isin(["PT_S1", "PT_S2", "PT_S3"]))[0]
        if len(i) != 1 or o.n_cells.values[h].sum() < v["marker_min_cells"]:
            continue
        a, b = X[i[0]], X[h].sum(0)
        la, lb = np.log2(a / a.sum() * 1e6 + 1), np.log2(b / b.sum() * 1e6 + 1)
        diffs.append(la - lb); ipt_mean.append(la); pt_mean.append(lb)
    D = np.vstack(diffs)
    t = D.mean(0) / (D.std(0, ddof=1) / np.sqrt(len(D)) + 1e-9)
    res = pd.DataFrame({"gene": pb.var_names, "mean_diff": D.mean(0), "t": t,
                        "ipt_log2cpm": np.vstack(ipt_mean).mean(0), "pt_log2cpm": np.vstack(pt_mean).mean(0)})
    res["n_libraries"] = len(D)
    excl = set(g for s in marker_sets().values() for g in s)
    hg = pd.read_csv(ROOT / cfg["paths"]["raw"] / "reference/hgnc_complete_set.txt", sep="\t",
                     usecols=["symbol", "locus_group", "status"], low_memory=False)
    pc = set(hg.symbol[(hg.status == "Approved") & (hg.locus_group == "protein-coding gene")])
    res = res[~res.gene.isin(excl) & res.gene.isin(pc) & ~res.gene.str.match(r"^(MT-|RP[SL]\d|IG[HKL][VDJC]|HB[AB])")]
    up = res[(res.t > v["marker_min_t"]) & (res.ipt_log2cpm > v["marker_min_log2cpm"])].nlargest(v["n_markers"], "mean_diff")
    dn = res[(res.t < -v["marker_min_t"]) & (res.pt_log2cpm > v["marker_min_log2cpm"])].nsmallest(v["n_markers"], "mean_diff")
    return pd.concat([up.assign(set="iPT_marker"), dn.assign(set="healthyPT_marker")])


def barcode_array_map() -> pd.DataFrame:
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data/processed/kpmp_visium/*/*/**/tissue_positions*.csv"), recursive=True))[:5]:
        d = pd.read_csv(f, header=0 if f.endswith("tissue_positions.csv") else None)
        d.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        rows.append(d[["barcode", "array_row", "array_col"]])
    m = pd.concat(rows).drop_duplicates()
    assert not m.barcode.duplicated().any(), "barcode map differs between slides"
    return m.set_index("barcode")


def lognorm_z(X, genes: pd.Index) -> tuple[pd.DataFrame, pd.Series]:
    """X spots x genes (dense or sparse counts) -> z of log1p(CP10k) for genes detected in >=1% spots."""
    import scipy.sparse as sp
    X = sp.csr_matrix(X)
    lib = np.asarray(X.sum(1)).ravel()
    det = np.asarray((X > 0).mean(0)).ravel() >= 0.01
    Y = sp.diags(1e4 / np.maximum(lib, 1)) @ X[:, det]
    Y = Y.log1p().toarray()
    mu, sd = Y.mean(0), Y.std(0, ddof=1)
    Z = (Y - mu) / np.where(sd > 0, sd, np.nan)
    return pd.DataFrame(Z, columns=genes[det]), pd.Series(mu, index=genes[det])


def set_scores(Z: pd.DataFrame, sets: dict[str, list[str]], min_genes: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    out, cov = {}, []
    for k, g in sets.items():
        gg = [x for x in g if x in Z.columns]
        cov.append({"set": k, "n_set": len(g), "n_measured": len(gg), "evaluable": len(gg) >= min_genes,
                    "genes_measured": ",".join(gg)})
        out[k] = Z[gg].mean(1) if gg else np.nan
    return pd.DataFrame(out, index=Z.index), pd.DataFrame(cov)


def partial_r(x, y, z) -> float:
    """Pearson correlation of x and y after regressing both on z (1-D)."""
    x, y, z = map(np.asarray, (x, y, z))
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    A = np.c_[np.ones_like(z), z]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    return float(stats.pearsonr(rx, ry)[0])


# ---------------------------------------------------------------- deconvolution
DECONV_TYPES = {"PT_healthy": ["PT_S1", "PT_S2", "PT_S3"], "iPT": ["iPT"], "TAL": ["M_TAL", "C_TAL", "Macula_Densa"],
                "ThinLimb": ["Des-Thin_Limb", "Ascending_Thin_LOH"], "DCT_CNT": ["DCT1", "DCT2", "CNT"], "PC": ["PC"],
                "IC": ["IC_A", "IC_B"], "Podo": ["Podo"], "PEC": ["PEC"],
                "Endo": ["Endo_GC", "Endo_Peritubular", "Endo_Lymphatic"],
                "Stroma": ["Fibroblast_1", "Fibroblast_2", "MyoFib/VSMC", "GS_Stromal", "Mes"],
                "Immune": ["CD14_Mono", "CD16_Mono", "Mac", "cDC", "pDC", "Neutrophil", "Baso/Mast", "CD4T", "CD8T",
                           "NK", "B_Naive", "B_memory", "Plasma_Cells", "Prolif_Lym"]}


def reference_signatures(cfg: dict, n_markers: int, min_cp10k: float = 2.0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Discovery SN (QC-pass) pseudobulk -> CP10k profile per deconvolution type and marker genes
    (top n_markers by log2 ratio of own profile over the highest other type among genes with
    >= min_cp10k in the own type; HGNC protein-coding, marker-programme genes excluded).
    Positive control on GSE211785 author spot labels (10a): PT-lineage share separates PT/iPT spots
    from all others (AUC ~0.86), but the iPT-vs-healthy-PT split is not resolvable at spot level
    (AUC ~0.57); spot analyses therefore use PT-rich spots, with injury indices as covariates."""
    pb = ad.read_h5ad(ROOT / cfg["paths"]["processed"] / "pb_fine.h5ad")
    q = pd.read_csv(ROOT / "results/02c_library_qc/library_qc.tsv", sep="\t").set_index("library")
    o = pb.obs
    ok = (o.library.map(q.batch).eq("SN") & o.library.map(q.qc_pass).astype(bool)).values
    X = pb.X.toarray() if hasattr(pb.X, "toarray") else np.asarray(pb.X)
    prof = {}
    for t, cts in DECONV_TYPES.items():
        m = ok & o.celltype.isin(cts).values
        v = X[m].sum(0); prof[t] = v / v.sum() * 1e4
    P = pd.DataFrame(prof, index=pb.var_names)
    hg = pd.read_csv(ROOT / cfg["paths"]["raw"] / "reference/hgnc_complete_set.txt", sep="\t",
                     usecols=["symbol", "locus_group", "status"], low_memory=False)
    pc = set(hg.symbol[(hg.status == "Approved") & (hg.locus_group == "protein-coding gene")])
    excl = set(g for s in marker_sets().values() for g in s)
    P = P[P.index.isin(pc) & ~P.index.isin(excl) & ~P.index.str.match(r"^(MT-|RP[SL]\d)")]
    L = np.log2(P + 0.01)
    rows = []
    for t in P.columns:
        ratio = L[t] - L.drop(columns=t).max(1)
        top = ratio[(P[t] > min_cp10k)].nlargest(n_markers)
        rows += [{"type": t, "gene": g, "log2_ratio": r} for g, r in top.items()]
    return P, pd.DataFrame(rows)


def nnls_deconvolve(X, genes: pd.Index, P: pd.DataFrame, markers: pd.DataFrame) -> tuple[pd.DataFrame, list]:
    """markers must be selected on P restricted to the query genes."""
    """Per-spot NNLS of CP10k over marker genes on the reference profiles; rows normalised to 1."""
    import scipy.sparse as sp
    from scipy.optimize import nnls
    X = sp.csr_matrix(X)
    g = [x for x in markers.gene.unique() if x in set(genes)]
    gi = pd.Index(genes).get_indexer(g)
    lib = np.asarray(X.sum(1)).ravel()
    Y = (sp.diags(1e4 / np.maximum(lib, 1)) @ X[:, gi]).toarray()
    A = P.loc[g].values
    W = np.vstack([nnls(A, y)[0] for y in Y])
    W = W / np.maximum(W.sum(1, keepdims=True), 1e-12)
    return pd.DataFrame(W, columns=P.columns), g


def reference_signatures_from(P: pd.DataFrame, n_markers: int, min_cp10k: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Marker selection (as in reference_signatures) on a profile table restricted to query genes."""
    L = np.log2(P + 0.01)
    rows = []
    for t in P.columns:
        ratio = L[t] - L.drop(columns=t).max(1)
        rows += [{"type": t, "gene": g, "log2_ratio": r} for g, r in ratio[P[t] > min_cp10k].nlargest(n_markers).items()]
    return P, pd.DataFrame(rows)


def pt_lineage_markers(P: pd.DataFrame, n: int, min_cp10k: float) -> list[str]:
    """Genes specific to the PT lineage (mean of PT_healthy and iPT profiles) over every other
    reference type; used for a calibration-free, within-section ranking of PT-rich spots.
    Rationale: absolute NNLS proportions were mis-calibrated between the nuclear reference and
    the spatial platforms (KPMP mean PT share 0.11, stroma 0.37; a gene-wise platform correction
    shifted mass to thin limb), whereas within-section ranks need no calibration. Positive control
    on GSE211785 author labels: AUC 0.85; top-20% spots per section 62% PT/iPT vs 25% baseline."""
    L = np.log2(P + 0.01)
    pt = P[["PT_healthy", "iPT"]].mean(1)
    ratio = np.log2(pt + 0.01) - L.drop(columns=["PT_healthy", "iPT"]).max(1)
    return ratio[pt > min_cp10k].nlargest(n).index.tolist()
