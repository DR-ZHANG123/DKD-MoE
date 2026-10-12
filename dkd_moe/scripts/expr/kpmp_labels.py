"""Cell labels built inside KPMP only (independent of the external GSE211785 reference).

Per technology: cells carrying KPMP's own atlas annotation (dkd_moe/results/data_annotation/) train a
multinomial logistic regression on the top PCs of log-normalised HVGs with balanced classes (the main
pipeline's recipe, scripts/lib/annotate.py); accuracy by participant-grouped cross-validation. Cells
with an atlas label keep it; the remaining cells receive the classifier label.
Target label: PT-S1 / PT-S2 / PT-S3 (subclass.l2) and aPT / frPT / dPT / cycPT (subclass.l1) for the
proximal tubule; subclass.l1 for every other cell type.
Usage: python kpmp_labels.py sn|sc
Outputs: dkd_moe/results/expr_labels/{kpmp_cell_labels_<t>.tsv.gz, cv_accuracy_<t>.tsv,
         agreement_transfer_vs_kpmp_<t>.tsv, summary_<t>.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, path, result_dir, write_provenance, set_global_seed  # noqa: E402
from prep_common import pt_state  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from annotate import fit_reference, predict  # noqa: E402

PT_L1 = {"aPT", "frPT", "dPT", "cycPT"}


def target_label(df: pd.DataFrame) -> pd.Series:
    l1, l2 = df.kpmp_subclass_l1.astype(str), df.kpmp_subclass_l2.astype(str)
    lab = l1.copy()
    lab[l1 == "PT"] = l2[l1 == "PT"]
    lab[l1.isin(PT_L1)] = l1[l1.isin(PT_L1)]
    return lab


def main():
    tech = sys.argv[1]
    cfg = load_cfg("expr"); seed = cfg["seed"]; set_global_seed(seed)
    out = result_dir("expr_labels")
    src = ROOT / f"dkd_moe/results/data_annotation/kpmp_cell_labels_{tech}.tsv.gz"
    a = ad.read_h5ad(path(cfg, f"kpmp_{tech}"))
    a.obs_names = a.obs_names.astype(str)
    atlas = pd.read_csv(src, sep="\t", index_col=0)
    atlas = atlas[~atlas.index.duplicated(keep=False)]
    lab = target_label(atlas).reindex(a.obs_names)
    a.obs["kpmp_label"] = lab.values
    a.obs["donor"] = a.obs.participant.astype(str)
    has = a.obs.kpmp_label.notna().values & (a.obs.kpmp_label.astype(str) != "nan").values
    ref = a[has].copy()
    # drop very rare classes from the classifier (kept as atlas labels on their own cells)
    vc = ref.obs.kpmp_label.value_counts()
    ref = ref[ref.obs.kpmp_label.isin(vc[vc >= 50].index).values].copy()
    print(tech, "atlas-labelled cells", int(has.sum()), "of", a.n_obs, flush=True)
    model = fit_reference(ref, a.var_names, "kpmp_label", seed=seed)
    cv = model["cv"]
    acc = cv.assign(ok=cv.true == cv.pred).groupby("true").ok.agg(["mean", "size"]).rename(
        columns={"mean": "accuracy", "size": "n_cv_cells"})
    acc.loc["__overall__"] = [float((cv.true == cv.pred).mean()), len(cv)]
    acc.to_csv(out / f"cv_accuracy_{tech}.tsv", sep="\t")
    pred = predict(model, a)
    res = pd.DataFrame({"participant": a.obs.participant.astype(str).values}, index=a.obs_names)
    res["label"] = np.where(has, a.obs.kpmp_label.astype(str).values, pred.label.values)
    res["max_prob"] = np.where(has, 1.0, pred.max_prob.values)
    res["classifier_label"] = pred.label.values
    res["label_origin"] = np.where(has, "kpmp_atlas", "classifier")
    res.to_csv(out / f"kpmp_cell_labels_{tech}.tsv.gz", sep="\t")
    # agreement with the GSE211785-reference label transfer
    old = pd.read_csv(ROOT / f"results/07b_validate_kpmp/cell_labels_{tech}.tsv.gz", sep="\t", index_col=0)
    so = pt_state(old.label.reindex(res.index).fillna(""), cfg).fillna("not_PT_state")
    sn = pt_state(res.label, cfg).fillna("not_PT_state")
    xt = pd.crosstab(so.rename("gse211785_transfer"), sn.rename("kpmp_annotation"))
    xt.to_csv(out / f"agreement_transfer_vs_kpmp_{tech}.tsv", sep="\t")
    io, inn = set(so.index[so == "injured_PT"]), set(sn.index[sn == "injured_PT"])
    po, pn = set(so.index[so != "not_PT_state"]), set(sn.index[sn != "not_PT_state"])
    # classifier agreement on atlas-labelled cells at the PT-state level (resubstitution excluded: CV)
    cvs = pd.DataFrame({"t": pt_state(cv.true.reset_index(drop=True), cfg).fillna("x"),
                        "p": pt_state(cv.pred.reset_index(drop=True), cfg).fillna("x")})
    summ = {"tech": tech, "n_cells": int(a.n_obs), "n_atlas_labelled": int(has.sum()),
            "cv_accuracy_overall": float((cv.true == cv.pred).mean()),
            "cv_pt_state_accuracy_injured": float((cvs[cvs.t == "injured_PT"].p == "injured_PT").mean()),
            "cv_pt_state_accuracy_healthy": float((cvs[cvs.t == "healthy_PT"].p == "healthy_PT").mean()),
            "jaccard_injured_PT_old_vs_new": len(io & inn) / max(1, len(io | inn)),
            "jaccard_PT_old_vs_new": len(po & pn) / max(1, len(po | pn)),
            "n_injured_old": len(io), "n_injured_new": len(inn),
            "injured_definition": "aPT + frPT (KPMP adaptive / failed-repair PT); dPT and cycPT not used"}
    (out / f"summary_{tech}.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1)); print(xt)
    write_provenance("expr_labels", [path(cfg, f"kpmp_{tech}"), src],
                     [out / f"kpmp_cell_labels_{tech}.tsv.gz", out / f"cv_accuracy_{tech}.tsv",
                      out / f"agreement_transfer_vs_kpmp_{tech}.tsv", out / f"summary_{tech}.json"],
                     seed, cfg, extra=summ, tag=tech)


if __name__ == "__main__":
    main()
