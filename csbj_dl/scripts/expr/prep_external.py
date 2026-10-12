"""Harmonised PT-cell objects for the external test cohorts (forward inference only, never training).

GSE211785: author cell types (lineage PT: PT_S1-3 healthy, iPT injured), libraries passing the main
pipeline's library QC; tech SC/SN kept with the scRNA protocol batch (SC_A/SC_B/SN).
GSE195460: PT labels from the main pipeline's label transfer (results/06_validate_gse195460).
Genes restricted to the KPMP HVG list (all HVGs are measured in both cohorts by construction).
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, path, result_dir, write_provenance, set_global_seed  # noqa: E402
from prep_common import pt_state, total_counts  # noqa: E402

COLS = ["participant", "library", "group", "tech", "batch", "pt_state", "label", "label_prob",
        "total_counts", "n_genes", "dataset"]


def finish(a: ad.AnnData, hvg: list[str], cfg) -> ad.AnnData:
    a.obs["total_counts"] = total_counts(a.X)
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    a = a[a.obs.pt_state.notna().values][:, hvg].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    n = a.obs.groupby(["participant", "tech"]).size()
    ok = n[n >= cfg["prep"]["min_cells_per_participant"]].index
    a = a[pd.MultiIndex.from_frame(a.obs[["participant", "tech"]]).isin(ok)].copy()
    for c in COLS:
        if c not in ("total_counts", "n_genes", "label_prob"):
            a.obs[c] = a.obs[c].astype(str)
    a.obs = a.obs[COLS + [c for c in a.obs.columns if c not in COLS]]
    return a


def gse211785(cfg, hvg):
    a = ad.read_h5ad(path(cfg, "gse211785"))
    q = pd.read_csv(path(cfg, "gse211785_library_qc"), sep="\t").set_index("library")
    a = a[a.obs.library.map(q.qc_pass).astype(bool).values].copy()
    a.obs["participant"] = a.obs.donor.astype(str)
    a.obs["tech"] = a.obs.tech.astype(str).map({"SC_RNA": "sc", "SN_RNA": "sn"})
    a.obs["batch"] = a.obs.library.map(q.batch).astype(str)
    a.obs["label"] = a.obs.celltype.astype(str)
    a.obs["label_prob"] = np.nan
    a.obs["pt_state"] = pt_state(a.obs.label, cfg).values
    a.obs.loc[a.obs.lineage.astype(str) != "PT", "pt_state"] = np.nan
    a.obs["dataset"] = "GSE211785"
    return finish(a, hvg, cfg)


def gse195460(cfg, hvg):
    a = ad.read_h5ad(path(cfg, "gse195460"))
    lab = pd.read_csv(path(cfg, "gse195460_labels"), sep="\t", index_col=0)
    a.obs["label"] = lab.label.reindex(a.obs_names).astype(str).values
    a.obs["label_prob"] = lab.max_prob.reindex(a.obs_names).values
    a.obs["participant"] = a.obs.donor.astype(str)
    a.obs["group"] = a.obs.diagnosis.astype(str)
    a.obs["tech"] = "sn"; a.obs["batch"] = "SN"
    a.obs["pt_state"] = pt_state(a.obs.label, cfg).values
    a.obs["dataset"] = "GSE195460"
    return finish(a, hvg, cfg)


def main():
    cfg = load_cfg("expr"); seed = cfg["seed"]; set_global_seed(seed)
    out = result_dir("expr_prep"); bulk = Path(cfg["paths"]["bulk_dir"])
    hvg = pd.read_csv(out / "hvg_genes.tsv", sep="\t").gene.tolist()
    summ, outs = {}, []
    for name, fn in [("gse211785", gse211785), ("gse195460", gse195460)]:
        a = fn(cfg, hvg)
        f = bulk / f"{name}_pt_hvg.h5ad"
        a.write_h5ad(f, compression="gzip"); outs.append(f)
        t = (a.obs.groupby(["participant", "group", "tech", "batch", "pt_state"]).size()
             .unstack("pt_state", fill_value=0).reset_index())
        t.to_csv(out / f"{name}_pt_cells_by_participant.tsv", sep="\t", index=False)
        outs.append(out / f"{name}_pt_cells_by_participant.tsv")
        summ[name] = {"n_cells": int(a.n_obs), "n_genes": int(a.n_vars),
                      "participants_by_group_tech": repr(a.obs.groupby(["group", "tech"]).participant.nunique().to_dict())}
        print(name, json.dumps(summ[name], indent=1), flush=True)
    write_provenance("expr_prep", [path(cfg, "gse211785"), path(cfg, "gse195460"), path(cfg, "gse195460_labels"),
                                   out / "hvg_genes.tsv"], outs, seed, cfg,
                     extra={"step": "prep_external", **summ}, tag="external")


if __name__ == "__main__":
    main()
