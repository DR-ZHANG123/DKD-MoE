"""Participant-level 5-fold split covering every KPMP participant across sc, sn, Visium and Multiome.

All modalities of one participant fall in the same fold. Folds are stratified by group with a fixed
seed. Group follows the main pipeline's definition (00b_download_kpmp.assign_group on the public
clinical table). Output: csbj_dl/config/splits.tsv
"""
from __future__ import annotations
import importlib, json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, path, write_provenance  # noqa: E402

sys.path.insert(0, str(ROOT / "scripts" / "stages"))


def main():
    cfg = load_cfg("expr"); seed = cfg["seed"]
    sc = set(ad.read_h5ad(path(cfg, "kpmp_sc"), backed="r").obs.participant.astype(str))
    sn = set(ad.read_h5ad(path(cfg, "kpmp_sn"), backed="r").obs.participant.astype(str))
    vis = set(pd.read_csv(path(cfg, "visium_samples"), sep="\t").participant.astype(str))
    files = pd.read_csv(path(cfg, "kpmp_files"), sep="\t", low_memory=False)
    mo = files[(files.access == "open") & files.experimental_strategy.astype(str).str.startswith("10x Multiome")]
    multi = set(mo.redcap_id.astype(str))
    allp = sorted(sc | sn | vis | multi)
    clin = importlib.import_module("00b_download_kpmp").participant_groups()
    clin["Participant ID"] = clin["Participant ID"].astype(str)
    grp = clin.set_index("Participant ID").group.reindex(allp)
    # sc/sn objects carry their own group (same definition); use it where clinical row is missing
    missing = grp[grp.isna()].index.tolist()
    for f in ("kpmp_sc", "kpmp_sn"):
        o = ad.read_h5ad(path(cfg, f), backed="r").obs[["participant", "group"]].drop_duplicates()
        o = o.set_index(o.participant.astype(str)).group.astype(str)
        grp = grp.fillna(o.reindex(grp.index))
    grp = grp.fillna("unknown")
    df = pd.DataFrame({"participant": allp, "group": grp.values})
    df["in_sc"] = df.participant.isin(sc).astype(int)
    df["in_sn"] = df.participant.isin(sn).astype(int)
    df["in_visium"] = df.participant.isin(vis).astype(int)
    df["in_multiome"] = df.participant.isin(multi).astype(int)
    # stratify by group; rare groups (< n_folds) pooled into one stratum for the splitter only
    strat = df.group.copy()
    vc = strat.value_counts()
    strat[strat.isin(vc[vc < cfg["n_folds"]].index)] = "rare"
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=seed)
    df["fold"] = -1
    for k, (_, te) in enumerate(skf.split(df, strat)):
        df.loc[df.index[te], "fold"] = k
    df = df[["participant", "group", "fold", "in_sc", "in_sn", "in_visium", "in_multiome"]]
    out = path(cfg, "splits")
    df.to_csv(out, sep="\t", index=False)
    summ = {"n_participants": len(df), "missing_clinical_row": missing,
            "group_by_fold": pd.crosstab(df.group, df.fold).to_dict(),
            "modality_counts": {c: int(df[c].sum()) for c in ["in_sc", "in_sn", "in_visium", "in_multiome"]},
            "id_format_check": {"multiome_not_in_clinical": sorted(multi - set(clin["Participant ID"]))[:20]}}
    print(pd.crosstab(df.group, df.fold, margins=True)); print(json.dumps(summ["modality_counts"]))
    print("missing clinical row:", missing)
    write_provenance("expr_splits", [path(cfg, "kpmp_sc"), path(cfg, "kpmp_sn"), path(cfg, "visium_samples"),
                                     path(cfg, "kpmp_files")], [out], seed, cfg, extra=summ)


if __name__ == "__main__":
    main()
