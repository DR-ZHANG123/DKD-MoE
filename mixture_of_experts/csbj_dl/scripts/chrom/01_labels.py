"""01_labels: healthy / injured proximal-tubule labels for KPMP Multiome nuclei (KPMP-internal recipe).

The KPMP snRNA release contains the Multiome GEX libraries, so most Multiome nuclei are present in the KPMP
snRNA object under the same participant and barcode. Such nuclei inherit their KPMP-internal label
(csbj_dl/results/expr_labels/kpmp_cell_labels_sn.tsv.gz: KPMP atlas label, else the KPMP-trained classifier).
A match is accepted when total RNA counts over shared genes agree within a factor of 2. All Multiome nuclei
are also labelled by the same recipe's classifier (scripts/lib/annotate.py fitted on atlas-labelled KPMP sn
cells); unmatched nuclei take that label, and classifier-vs-inherited agreement is reported.
Outputs: csbj_dl/results/chrom_labels/{multiome_labels.tsv.gz, counts_by_participant.tsv, group_summary.tsv,
         classifier_agreement.tsv, match_summary.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import ROOT, cfg, P, result_dir, write_provenance, set_global_seed, pt_state, splits  # noqa: E402
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
from annotate import fit_reference, predict  # noqa: E402
from kpmp_labels import target_label  # noqa: E402  (expr module, read-only import)


def match_sn(m: ad.AnnData, sn: ad.AnnData, lab: pd.DataFrame) -> pd.DataFrame:
    o = sn.obs
    bc = pd.Index(o.index.astype(str)).str.extract(r"([ACGT]{16}-\d+)$")[0].values
    key = pd.Series(o.index.astype(str), index=o.participant.astype(str).values + "|" + bc)
    key = key[~key.index.duplicated(keep=False)]
    mk = m.obs.participant.astype(str) + "|" + m.obs.barcode.astype(str)
    cell = mk.map(key)
    res = pd.DataFrame({"sn_cell": cell.values}, index=m.obs_names)
    ok = res.sn_cell.notna().values
    genes = sn.var_names.intersection(m.var_names)
    i_sn = sn.obs_names.get_indexer(res.sn_cell[ok].values)
    t_sn = np.asarray(sn.X[i_sn][:, sn.var_names.get_indexer(genes)].sum(1)).ravel()
    t_m = np.asarray(m.X[np.where(ok)[0]][:, m.var_names.get_indexer(genes)].sum(1)).ravel()
    ratio = np.full(len(res), np.nan); ratio[ok] = t_m / np.maximum(t_sn, 1)
    res["count_ratio"] = ratio
    res["matched"] = ok & (np.abs(np.log2(np.where(ok, ratio, 1.0))) < 1.0)
    res["sn_label"] = lab.label.reindex(res.sn_cell.where(res.matched)).values
    res["sn_label_origin"] = lab.label_origin.reindex(res.sn_cell.where(res.matched)).values
    res.attrs["r_log_counts"] = float(np.corrcoef(np.log1p(t_sn), np.log1p(t_m))[0, 1]) if ok.sum() > 2 else np.nan
    return res


def main():
    c = cfg(); seed = c["seed"]; set_global_seed(seed); out = result_dir("chrom_labels")
    m = ad.read_h5ad(P(c, "multiome_rna"))
    sn = ad.read_h5ad(P(c, "kpmp_sn")); sn.obs_names = sn.obs_names.astype(str)
    lab = pd.read_csv(P(c, "kpmp_sn_labels"), sep="\t", index_col=0)
    res = match_sn(m, sn, lab)
    # classifier of the KPMP-internal recipe, fitted on atlas-labelled KPMP sn cells
    atlas = pd.read_csv(P(c, "kpmp_sn_atlas"), sep="\t", index_col=0)
    atlas = atlas[~atlas.index.duplicated(keep=False)]
    tl = target_label(atlas).reindex(sn.obs_names)
    has = tl.notna().values & (tl.astype(str) != "nan").values
    ref = sn[has].copy(); ref.obs["kpmp_label"] = tl[has].values; ref.obs["donor"] = ref.obs.participant.astype(str)
    del sn
    vc = ref.obs.kpmp_label.value_counts()
    ref = ref[ref.obs.kpmp_label.isin(vc[vc >= c["labels"]["classifier_min_class"]].index).values].copy()
    model = fit_reference(ref, m.var_names, "kpmp_label", seed=seed)
    del ref
    pred = predict(model, m)
    res["classifier_label"] = pred.label.values; res["classifier_prob"] = pred.max_prob.values
    res["label"] = np.where(res.matched, res.sn_label.astype(str), res.classifier_label.astype(str))
    res["label_origin"] = np.where(res.matched, "kpmp_sn_" + res.sn_label_origin.astype(str), "classifier_multiome")
    res["pt_state"] = pt_state(res.label, c).values
    S = splits(c)
    o = m.obs
    res.insert(0, "participant", o.participant.astype(str).values)
    res.insert(1, "library", o.library.astype(str).values)
    res.insert(2, "group", o.group.astype(str).values)
    res.insert(3, "fold", S.fold.reindex(res.participant).values)
    for k in ("in_sc", "in_sn"):
        res[k] = S[k].reindex(res.participant).fillna(0).astype(int).values
    res.to_csv(out / "multiome_labels.tsv.gz", sep="\t")
    # classifier vs inherited label at PT-state level on matched nuclei
    mm = res[res.matched]
    a = pd.crosstab(pt_state(mm.sn_label, c).fillna("other").rename("inherited"),
                    pt_state(mm.classifier_label, c).fillna("other").rename("classifier"))
    a.to_csv(out / "classifier_agreement.tsv", sep="\t")
    cnt = (res[res.pt_state.notna()].groupby(["participant", "group", "fold", "pt_state"]).size()
           .unstack("pt_state", fill_value=0).reset_index())
    cnt["n_nuclei_all"] = cnt.participant.map(res.participant.value_counts())
    cnt.to_csv(out / "counts_by_participant.tsv", sep="\t", index=False)
    gs = cnt.groupby("group").agg(participants=("participant", "nunique"), healthy_PT=("healthy_PT", "sum"),
                                  injured_PT=("injured_PT", "sum"), nuclei_all=("n_nuclei_all", "sum"),
                                  participants_ge20_injured=("injured_PT", lambda x: int((x >= 20).sum()))).reset_index()
    gs.to_csv(out / "group_summary.tsv", sep="\t", index=False)
    inj = (mm.pipe(lambda d: pt_state(d.sn_label, c)) == "injured_PT")
    summ = {"n_nuclei": int(len(res)), "n_matched_sn": int(res.matched.sum()),
            "n_barcode_hits": int(res.sn_cell.notna().sum()), "r_log_total_counts_matched": res.attrs["r_log_counts"],
            "median_count_ratio": float(np.nanmedian(res.count_ratio)),
            "classifier_injured_recall_on_matched": float((pt_state(mm.classifier_label[inj], c) == "injured_PT").mean())
            if inj.any() else None,
            "classifier_cv_accuracy_overall": float((model["cv"].true == model["cv"].pred).mean()),
            "participants": int(res.participant.nunique()), "libraries": int(res.library.nunique()),
            "multiome_h5ad_md5_note": "see PROVENANCE"}
    (out / "match_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1)); print(gs.to_string(index=False)); print(a)
    write_provenance("chrom_labels", [P(c, "multiome_rna"), P(c, "kpmp_sn_labels"), P(c, "kpmp_sn_atlas")],
                     [out / f for f in ("multiome_labels.tsv.gz", "counts_by_participant.tsv", "group_summary.tsv",
                                        "classifier_agreement.tsv", "match_summary.json")], seed, c, extra=summ)


if __name__ == "__main__":
    main()
