"""In-memory spot table for alignment: image embeddings, log-normalised expression on HVGs, labels.

Expression: log1p(CP10k) with the library size taken over the gene set measured on both platforms;
genes = HVGs (expr_prep list if the expr module wrote one, else seurat_v3 HVGs selected on
KPMP training spots only). Standardisation statistics (genes and image dims) come from training
spots only and are stored with each model.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp

import common as C


class SpotData:
    def __init__(self, sections: pd.DataFrame):
        self.sections = sections.reset_index(drop=True)
        embs, mats, meta = [], [], []
        for i, s in enumerate(self.sections.itertuples()):
            z = np.load(s.cache)
            embs.append(z["emb"].astype(np.float32))
            X = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
            mats.append(X)
            n = len(z["spot_id"])
            meta.append(pd.DataFrame({"sec_idx": i, "section": s.section, "participant": s.participant,
                                      "cohort": s.cohort, "group": s.group, "spot_id": z["spot_id"],
                                      "lib": z["lib"], "pt_rich": z["pt_rich"], "pt_score": z["pt_score"],
                                      "px_row": z["px_row"], "px_col": z["px_col"],
                                      "um_per_px": s.um_per_px}))
            if i == 0:
                self.feat_names = list(z["feat_names"])
            meta[-1][self.feat_names] = z["feats"]
        self.E = np.vstack(embs)
        self.X = sp.vstack(mats).tocsr()
        self.meta = pd.concat(meta, ignore_index=True)
        self.genes = pd.Index((C.p(C.load_cfg()["paths"]["cache"]) / "genes.txt").read_text().split())

    def lognorm(self, gene_idx: np.ndarray) -> np.ndarray:
        lib = np.maximum(self.meta.lib.values, 1.0)
        Y = sp.diags(C.load_cfg()["data"]["target_sum"] / lib) @ self.X[:, gene_idx]
        return np.log1p(Y.toarray()).astype(np.float32)


def select_hvgs(data: SpotData, train_mask: np.ndarray, n: int, source: str = "expr_prep") -> tuple[np.ndarray, str]:
    """source 'expr_prep': the expr module's KPMP PT-cell HVGs restricted to genes measured on
    both Visium platforms; 'visium': seurat_v3 HVGs of KPMP training spots."""
    cfg = C.load_cfg()
    d = C.p(cfg["paths"]["expr_prep"])
    files = sorted(list(d.glob("hvg*.tsv")) + list(d.glob("hvg*.txt"))) if d.exists() else []
    if files and source == "expr_prep":
        t = pd.read_csv(files[0], sep="\t")
        col = "gene" if "gene" in t.columns else t.columns[0]
        idx = data.genes.get_indexer(t[col].astype(str))
        idx = idx[idx >= 0]
        if len(idx) >= 500:
            return np.sort(idx), f"expr_prep:{files[0].name} ({len(idx)} of {len(t)} measured on both platforms)"
    import anndata as ad, scanpy as sc
    a = ad.AnnData(data.X[train_mask].astype(np.float32))
    a.var_names = data.genes
    sc.pp.highly_variable_genes(a, n_top_genes=n, flavor="seurat_v3", batch_key=None)
    return np.sort(np.where(a.var.highly_variable.values)[0]), f"seurat_v3 on {int(train_mask.sum())} KPMP training spots"


def standardiser(A: np.ndarray, mask: np.ndarray):
    mu = A[mask].mean(0)
    sd = A[mask].std(0)
    return mu.astype(np.float32), np.where(sd > 1e-6, sd, 1.0).astype(np.float32)
