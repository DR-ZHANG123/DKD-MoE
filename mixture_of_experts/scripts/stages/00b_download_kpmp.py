"""00b_download_kpmp: select and fetch KPMP open-access files for the study groups.

Group definition uses the public clinical table (adjudicated category first, then diabetes /
hypertension history), see assign_group(). Files fetched per tier:
  expr    : sc/snRNA expression-matrix zips + their metadata xlsx; Visium matrices, spatial
            bundles (scalefactors, low/high-res H&E) and metadata xlsx
  visium_tif : full-resolution Visium H&E tif
  wsi     : one H&E and one PAS whole-slide image per participant with descriptor scores
Usage: python 00b_download_kpmp.py [expr|visium_tif|wsi] [--priority] [--dry]
"""
from __future__ import annotations
import subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config
from kpmp import read_descriptors

RAW = ROOT / "data/raw/kpmp"


def assign_group(r) -> str:
    adj = str(r["Primary Adjudicated Category"])
    enr = str(r["Enrollment Category"])
    dm, htn = str(r["Diabetes History"]), str(r["Hypertension History"])
    if enr == "Healthy Reference":
        return "Reference"
    if enr == "AKI":
        return "AKI"
    if adj.startswith("Diabetic Kidney Disease"):
        return "DKD"
    if adj.startswith("Hypertensive Kidney Disease"):
        return "HKD" if dm != "Yes" else "HKD_withDM"
    if enr == "DM-R":
        return "DM_noCKD"
    if adj in ("", "nan") and enr == "CKD":
        return "CKD_unadjudicated"
    return "CKD_other"


def participant_groups() -> pd.DataFrame:
    clin = pd.read_csv(RAW / load_config()["kpmp"]["clinical_table"])
    clin["group"] = clin.apply(assign_group, axis=1)
    return clin


def select(tier: str, cfg: dict) -> pd.DataFrame:
    files = pd.read_csv(RAW / "kpmp_repository_all_files.tsv", sep="\t", low_memory=False)
    files = files[files.access == "open"]
    clin = participant_groups()
    keep = set(clin.loc[clin.group.isin(cfg["kpmp"]["download_groups"]), "Participant ID"])
    f = files[files.redcap_id.isin(keep)].copy()
    es = f.experimental_strategy
    if tier == "expr":
        sc = es.isin(["Single-nucleus RNA-Seq", "Single-cell RNA-Seq"]) & ~f.data_format.eq("h5Seurat")
        vis = es.eq("Spatial Transcriptomics") & ~f.data_format.isin(["cloupe", "tif"])
        sel = f[sc | vis]
    elif tier == "visium_tif":
        sel = f[es.eq("Spatial Transcriptomics") & f.data_format.eq("tif")]
    elif tier == "wsi":
        scored = set(read_descriptors(RAW / cfg["kpmp"]["descriptor_table"])["Participant ID"])
        w = f[es.eq("Light Microscopic Whole Slide Images") & f.redcap_id.isin(scored)]
        w = w[w.workflow_type.isin(cfg["kpmp"]["wsi_stains"])]
        sel = (w.sort_values(["redcap_id", "workflow_type", "file_size"])
                .groupby(["redcap_id", "workflow_type"]).head(1))       # smallest file per stain
    else:
        raise SystemExit(f"unknown tier {tier}")
    return sel


def fetch(row) -> str:
    sub = RAW / row.experimental_strategy.replace(" ", "_").replace("/", "_") / str(row.redcap_id)
    sub.mkdir(parents=True, exist_ok=True)
    dst = sub / row.file_name
    if dst.exists() and dst.stat().st_size == int(float(row.file_size)):
        return f"skip {dst.name}"
    r = subprocess.run(["wget", "-q", "--tries=10", "--waitretry=20", "-O", str(dst), row.download_url])
    ok = r.returncode == 0 and dst.stat().st_size == int(float(row.file_size))
    if not ok:
        dst.unlink(missing_ok=True)
    return f"{'ok' if ok else 'FAIL'} {dst.name}"


def main():
    cfg = load_config()
    tier = sys.argv[1] if len(sys.argv) > 1 else "expr"
    sel = select(tier, cfg)
    if tier == "wsi" and "--priority" in sys.argv:
        # participants that also have sc/snRNA programme scores (07b) are fetched first
        ids = set()
        for t in ["sn", "sc"]:
            f = ROOT / f"results/07b_validate_kpmp/scores_{t}.tsv"
            if f.exists():
                ids |= set(pd.read_csv(f, sep="\t", index_col=0).index.astype(str))
        sel = sel[sel.redcap_id.isin(ids)]
        # rarest disease groups first so the DKD-vs-HKD contrast fills early
        order = {"HKD": 0, "HKD_withDM": 1, "DM_noCKD": 2, "DKD": 3, "Reference": 4}
        g = participant_groups().set_index("Participant ID").group
        sel = sel.assign(_o=sel.redcap_id.map(g).map(order).fillna(9)).sort_values(["_o", "redcap_id"]).drop(columns="_o")
    sel.to_csv(RAW / f"download_manifest_{tier}.tsv", sep="\t", index=False)
    print(tier, len(sel), "files", round(sel.file_size.astype(float).sum() / 1e9, 1), "GB",
          sel.redcap_id.nunique(), "participants")
    if "--dry" in sys.argv:
        return
    with ThreadPoolExecutor(cfg["kpmp"]["parallel"]) as ex:
        for msg in ex.map(fetch, [r for r in sel.itertuples()]):
            print(msg, flush=True)


if __name__ == "__main__":
    main()
