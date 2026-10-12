"""Data access for the bench module: shared expr_prep PT-cell objects, folds, cell subsampling,
participant x technology pseudobulk."""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import ROOT, CDL, load_cfg, path  # noqa: E402

EXT_NAMES = ("GSE211785", "GSE195460")


def expr_bulk_dir() -> Path:
    return Path(load_cfg("expr")["paths"]["bulk_dir"])


def _clean_obs(a: ad.AnnData, dataset: str, cfg: dict) -> ad.AnnData:
    o = a.obs
    o["participant"] = o.participant.astype(str)
    o["tech"] = o.tech.astype(str)
    o["dataset"] = dataset
    if dataset == "KPMP":
        o["condition"] = o.group.astype(str).map(cfg["contrast"]["map"]).fillna("excluded")
    else:
        o["condition"] = o.group.astype(str)
    if "library" not in o:
        o["library"] = o.participant
    o["library"] = o.library.astype(str)
    o["pt_state"] = o.pt_state.astype(str) if "pt_state" in o else "unknown"
    keep = ~o.participant.isin(cfg["contrast"]["exclude_donors"])
    a = a[keep.values].copy()
    if not sp.issparse(a.X):
        a.X = sp.csr_matrix(a.X, dtype=np.float32)
    return a


def load_bundle(cfg: dict, dev: bool = False) -> tuple[ad.AnnData, dict]:
    """KPMP PT cells (training/held-out folds) and external cohorts (forward inference only)."""
    if dev:
        k = ad.read_h5ad(path(cfg, "dev_subsample"))
        e = ad.read_h5ad(str(path(cfg, "dev_subsample")).replace(".h5ad", "_ext.h5ad"))
        return _clean_obs(k, "KPMP", cfg), {"GSE211785": _clean_obs(e, "GSE211785", cfg)}
    b = expr_bulk_dir()
    k = _clean_obs(ad.read_h5ad(b / "kpmp_pt_hvg.h5ad"), "KPMP", cfg)
    splits = pd.read_csv(path(cfg, "splits"), sep="\t", dtype={"participant": str}).set_index("participant")
    k.obs["fold"] = k.obs.participant.map(splits.fold).astype(int)   # the shared split is authoritative
    ext = {}
    for name in EXT_NAMES:
        f = b / f"{name.lower()}_pt_hvg.h5ad"
        e = ad.read_h5ad(f)
        ext[name] = _clean_obs(e[:, k.var_names].copy(), name, cfg)
    return k, ext


def bundle_inputs(dev: bool, cfg: dict) -> list:
    if dev:
        return [path(cfg, "dev_subsample")]
    b = expr_bulk_dir()
    return [b / "kpmp_pt_hvg.h5ad"] + [b / f"{n.lower()}_pt_hvg.h5ad" for n in EXT_NAMES] + [path(cfg, "splits")]


def subsample_cells(a: ad.AnnData, cap: int, seed: int) -> ad.AnnData:
    """At most `cap` cells per participant x technology (fixed by seed)."""
    rng = np.random.default_rng(seed)
    key = (a.obs.participant + "|" + a.obs.tech).values
    idx = []
    for u in pd.unique(key):
        w = np.where(key == u)[0]
        idx.append(w if len(w) <= cap else rng.choice(w, cap, replace=False))
    return a[np.sort(np.concatenate(idx))].copy()


def pseudobulk(a: ad.AnnData, min_cells: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summed counts per participant x technology; returns (counts: samples x genes, meta)."""
    key = (a.obs.participant + "|" + a.obs.tech).values
    codes, uniq = pd.factorize(key)
    M = sp.csr_matrix((np.ones(len(codes), np.float32), (codes, np.arange(len(codes)))),
                      shape=(len(uniq), len(codes)))
    C = np.asarray((M @ a.X).todense())
    ncell = np.bincount(codes)
    first = pd.Series(np.arange(len(codes))).groupby(codes).first().values
    meta = a.obs.iloc[first][["participant", "tech", "condition", "dataset"]].copy()
    meta.index = uniq
    meta["n_cells"] = ncell
    if "fold" in a.obs:
        meta["fold"] = a.obs.fold.values[first]
    keep = ncell >= min_cells
    return pd.DataFrame(C[keep], index=uniq[keep], columns=a.var_names), meta[keep]


def log_cpm(C: pd.DataFrame) -> pd.DataFrame:
    lib = C.sum(1).values[:, None]
    return np.log1p(C / lib * 1e6)


def dense(X) -> np.ndarray:
    return np.asarray(X.todense() if sp.issparse(X) else X, dtype=np.float32)


def write_json(p: Path, obj) -> None:
    p.write_text(json.dumps(obj, indent=2, default=str))
