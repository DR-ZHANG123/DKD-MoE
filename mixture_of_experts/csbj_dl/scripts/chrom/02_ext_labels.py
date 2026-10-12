"""02_ext_labels: proximal-tubule states of external snATAC nuclei.

GSE211785: the authors' cell annotation (Cluster_Idents of the merged object; PT_S1-3 healthy, iPT injured).
GSE195460: label transfer on gene activity -- the KPMP-internal recipe (scripts/lib/annotate.py: HVGs, PCA,
balanced multinomial logistic regression) fitted on KPMP Multiome gene activity with the RNA-side labels of
the same nuclei (01_labels). The transfer is also applied to GSE211785 gene activity and compared with the
authors' annotation as an external check of the transfer.
Outputs: csbj_dl/results/chrom_ext_labels/{gse211785_labels.tsv.gz, gse195460_labels.tsv.gz,
         transfer_vs_author_gse211785.tsv, counts_by_donor.tsv, summary.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import ROOT, cfg, P, result_dir, write_provenance, set_global_seed, pt_state  # noqa: E402
from chrom_data import labelled_obs, load_multiome_gact  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from annotate import fit_reference, predict  # noqa: E402


def author_labels_211785(c: dict, names: pd.Index) -> pd.Series:
    a = ad.read_h5ad(P(c, "gse211785_meta"), backed="r")
    return a.obs.Cluster_Idents.astype(str).reindex(names)


def main():
    c = cfg(); seed = c["seed"]; set_global_seed(seed); out = result_dir("chrom_ext_labels"); L = c["labels"]
    g211 = ad.read_h5ad(P(c, "gse211785_atac")); g195 = ad.read_h5ad(P(c, "gse195460_gact"))
    genes = [g for g in g195.var_names if g in set(g211.var_names)]
    lab = labelled_obs(c)
    lab = lab[lab.label.astype(str) != "nan"]
    ref = load_multiome_gact(c, lab.index, None)
    ref.obs["label"] = lab.label.loc[ref.obs_names].astype(str).values
    ref.obs["donor"] = lab.participant.loc[ref.obs_names].astype(str).values
    vc = ref.obs.label.value_counts()
    ref = ref[ref.obs.label.isin(vc[vc >= L["classifier_min_class"]].index).values].copy()
    model = fit_reference(ref, genes, "label", n_hvg=L["ga_n_hvg"], n_pcs=L["ga_n_pcs"],
                          max_per_class=L["ga_max_per_class"], seed=seed)
    cv = model["cv"]
    cvs = pd.crosstab(pt_state(cv.true.reset_index(drop=True), c).fillna("other").rename("rna_label"),
                      pt_state(cv.pred.reset_index(drop=True), c).fillna("other").rename("ga_transfer"))
    del ref
    # GSE211785: authors' annotation (primary) + transfer (check)
    auth = author_labels_211785(c, g211.obs_names)
    p211 = predict(model, g211)
    o211 = g211.obs[["library", "donor", "group"]].astype(str).copy()
    o211["author_label"] = auth.values; o211["transfer_label"] = p211.label.values; o211["transfer_prob"] = p211.max_prob.values
    o211["label"] = o211.author_label; o211["pt_state"] = pt_state(o211.label, c).values
    o211["transfer_pt_state"] = pt_state(o211.transfer_label, c).values
    o211.to_csv(out / "gse211785_labels.tsv.gz", sep="\t")
    xt = pd.crosstab(o211.pt_state.fillna("other").rename("author"), o211.transfer_pt_state.fillna("other").rename("transfer"))
    xt.to_csv(out / "transfer_vs_author_gse211785.tsv", sep="\t")
    # GSE195460: transfer (primary)
    p195 = predict(model, g195)
    o195 = g195.obs[["library", "donor", "group"]].astype(str).copy()
    o195["label"] = p195.label.values; o195["label_prob"] = p195.max_prob.values
    o195["pt_state"] = pt_state(o195.label, c).values
    o195.to_csv(out / "gse195460_labels.tsv.gz", sep="\t")
    cnt = []
    for name, o in (("GSE211785", o211), ("GSE195460", o195)):
        t = o[o.pt_state.notna()].groupby(["donor", "group", "pt_state"]).size().unstack("pt_state", fill_value=0).reset_index()
        t["n_nuclei_all"] = t.donor.map(o.donor.value_counts()); t.insert(0, "cohort", name); cnt.append(t)
    cnt = pd.concat(cnt, ignore_index=True); cnt.to_csv(out / "counts_by_donor.tsv", sep="\t", index=False)
    a_inj = (o211.pt_state == "injured_PT"); a_pt = o211.pt_state.notna()
    summ = {"ga_genes_classifier_pool": len(genes), "ga_classifier_genes": len(model["genes"]),
            "kpmp_ga_cv_accuracy": float((cv.true == cv.pred).mean()),
            "kpmp_ga_cv_pt_state": cvs.to_dict(),
            "gse211785_transfer_pt_recall": float(o211.transfer_pt_state[a_pt].notna().mean()) if a_pt.any() else None,
            "gse211785_transfer_injured_recall": float((o211.transfer_pt_state[a_inj] == "injured_PT").mean()) if a_inj.any() else None,
            "gse211785_transfer_injured_precision": float((o211.pt_state[o211.transfer_pt_state == "injured_PT"] == "injured_PT").mean())
            if (o211.transfer_pt_state == "injured_PT").any() else None}
    (out / "summary.json").write_text(json.dumps(summ, indent=2, default=str))
    print(json.dumps(summ, indent=1, default=str)); print(xt); print(cnt.to_string(index=False))
    write_provenance("chrom_ext_labels", [P(c, "gse211785_atac"), P(c, "gse195460_gact"), P(c, "gse211785_meta"),
                                          result_dir("chrom_labels") / "multiome_labels.tsv.gz"],
                     [out / f for f in ("gse211785_labels.tsv.gz", "gse195460_labels.tsv.gz",
                                        "transfer_vs_author_gse211785.tsv", "counts_by_donor.tsv", "summary.json")],
                     seed, c, extra=summ)


if __name__ == "__main__":
    main()
