"""Prepare KPMP proximal-tubule cells (healthy and injured PT) from sc and sn, select 3,000 HVGs.

Outputs
  bulk : <bulk_dir>/kpmp_pt_hvg.h5ad   raw counts, HVG genes, obs documented in expr_prep/README.md
  table: csbj_dl/results/expr_prep/{hvg_genes.tsv, gene_universe.tsv, label_source.json,
         kpmp_pt_cells_by_participant.tsv}
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
from cdl import ROOT, load_cfg, path, result_dir, write_provenance, set_global_seed  # noqa: E402
from prep_common import (resolve_label_source, read_labels, pt_state, gene_universe,  # noqa: E402
                         total_counts, select_hvg)

OBS_KEEP = ["participant", "library", "group", "egfr_band", "proteinuria_band", "albuminuria_band",
            "sex", "age_band", "dm", "htn", "tissue_source"]


def load_tech(cfg, tech: str, labfile: str, genes: list[str]) -> ad.AnnData:
    a = ad.read_h5ad(path(cfg, f"kpmp_{tech}"))
    a.obs_names = a.obs_names.astype(str)
    lab = read_labels(labfile, cfg)
    matched = lab.index.isin(a.obs_names).mean()
    print(tech, a.shape, "label rows matched:", round(float(matched), 4), flush=True)
    a.obs = a.obs[OBS_KEEP].join(lab, how="left")
    a.obs["pt_state"] = pt_state(a.obs.label.fillna(""), cfg).values
    a.obs["total_counts"] = total_counts(a.X)
    a.obs["n_genes"] = np.asarray((a.X > 0).sum(1)).ravel()
    keep = a.obs.pt_state.notna().values
    if cfg["labels"]["min_max_prob"] > 0:
        keep &= (a.obs.label_prob.fillna(1).values >= cfg["labels"]["min_max_prob"])
    a = a[keep][:, genes].copy()
    a.X = sp.csr_matrix(a.X, dtype=np.float32)
    a.obs["tech"] = tech
    return a


def main():
    cfg = load_cfg("expr"); seed = cfg["seed"]; set_global_seed(seed)
    out = result_dir("expr_prep"); bulk = Path(cfg["paths"]["bulk_dir"]); bulk.mkdir(parents=True, exist_ok=True)
    src = resolve_label_source(cfg)
    genes = gene_universe(cfg)
    print("gene universe:", len(genes), flush=True)
    parts = [load_tech(cfg, t, src[t], genes) for t in ("sc", "sn")]
    A = ad.concat(parts, join="inner", merge="same")
    A.obs_names_make_unique()
    for c in OBS_KEEP + ["label", "pt_state", "tech"]:
        A.obs[c] = A.obs[c].astype(str)
    splits = pd.read_csv(path(cfg, "splits"), sep="\t", dtype={"participant": str}).set_index("participant")
    A.obs["fold"] = A.obs.participant.map(splits.fold).astype(int)
    A.obs["dataset"] = "KPMP"
    # minimum PT cells per participant x technology for inclusion
    n = A.obs.groupby(["participant", "tech"]).size()
    ok = n[n >= cfg["prep"]["min_cells_per_participant"]].index
    keep = pd.MultiIndex.from_frame(A.obs[["participant", "tech"]]).isin(ok)
    A = A[keep].copy()
    hvg = select_hvg(A, cfg, seed)
    A = A[:, hvg].copy()
    A.var["gene"] = A.var_names
    fn = bulk / "kpmp_pt_hvg.h5ad"
    A.write_h5ad(fn, compression="gzip")
    pd.Series(genes, name="gene").to_csv(out / "gene_universe.tsv", sep="\t", index=False)
    pd.Series(hvg, name="gene").to_csv(out / "hvg_genes.tsv", sep="\t", index=False)
    tab = (A.obs.groupby(["participant", "group", "tech", "fold", "pt_state"]).size()
           .unstack("pt_state", fill_value=0).reset_index())
    tab = tab[(tab.get("healthy_PT", 0) + tab.get("injured_PT", 0)) > 0]
    tab.to_csv(out / "kpmp_pt_cells_by_participant.tsv", sep="\t", index=False)
    info = {"label_source": src["name"], "sc_labels": src["sc"], "sn_labels": src["sn"],
            "depends_on_external_reference": src["name"].startswith("label_transfer"),
            "n_cells": int(A.n_obs), "n_genes": int(A.n_vars), "gene_universe": len(genes),
            "cells_by_tech_state": A.obs.groupby(["tech", "pt_state"]).size().to_dict().__repr__(),
            "participants_by_tech_group": A.obs.groupby(["tech", "group"]).participant.nunique().to_dict().__repr__()}
    (out / "label_source.json").write_text(json.dumps(info, indent=2))
    print(json.dumps(info, indent=2))
    write_provenance("expr_prep", [path(cfg, "kpmp_sc"), path(cfg, "kpmp_sn"), src["sc"], src["sn"],
                                   path(cfg, "splits")],
                     [fn, out / "hvg_genes.tsv", out / "gene_universe.tsv", out / "label_source.json",
                      out / "kpmp_pt_cells_by_participant.tsv"], seed, cfg, extra={"step": "prep_kpmp"})


if __name__ == "__main__":
    main()
