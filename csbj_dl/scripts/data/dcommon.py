"""Shared helpers for the csbj_dl data module: config, KPMP file selection, participant groups."""
from __future__ import annotations
import importlib, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import ROOT, CDL, load_cfg, result_dir, write_provenance, set_global_seed, md5  # noqa: E402,F401

sys.path.insert(0, str(ROOT / "scripts" / "stages"))
sys.path.insert(0, str(ROOT / "scripts" / "lib"))

CLIN_COLS = [("Baseline eGFR (ml/min/1.73m2) (Binned)", "egfr_band"), ("Proteinuria (mg) (Binned)", "proteinuria_band"),
             ("Albuminuria (mg) (Binned)", "albuminuria_band"), ("Sex", "sex"), ("Age (Years) (Binned)", "age_band"),
             ("Diabetes History", "dm"), ("Hypertension History", "htn"), ("Tissue Source", "tissue_source")]
CATEGORIES = ["multiome", "visium", "pathomics", "annotation"]


def cfg() -> dict:
    return load_cfg("data")


def rpath(p: str | Path) -> Path:
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def groups() -> pd.DataFrame:
    """Participant table with the same group definition as the main pipeline (00b assign_group)."""
    return importlib.import_module("00b_download_kpmp").participant_groups().set_index("Participant ID")


def clinical_obs(pids, G: pd.DataFrame | None = None) -> pd.DataFrame:
    G = groups() if G is None else G
    g = G.reindex(list(pids))
    out = pd.DataFrame({"group": g.group.values}, index=g.index)
    for col, new in CLIN_COLS:
        out[new] = g[col].astype(str).values
    return out


def file_table(c: dict) -> pd.DataFrame:
    f = pd.read_csv(rpath(c["file_table"]), sep="\t", low_memory=False)
    f = f[f.access == "open"].copy()
    f["file_size"] = f.file_size.astype(float)
    return f


def existing_raw_names(c: dict) -> dict[str, Path]:
    """File names already present (complete) under the main pipeline's raw KPMP dir."""
    base = rpath(c["existing_raw"])
    return {p.name: p for p in base.glob("*/*/*") if p.is_file()}


def section_of(fname: str) -> str:
    """Visium section name from a repository file name (strip uuid_ prefix, spatial_ and extension)."""
    s = fname.split("_", 1)[1]
    s = s[len("spatial_"):] if s.startswith("spatial_") else s
    for ext in (".tar.gz", ".tif", ".zip"):
        if s.endswith(ext):
            return s[: -len(ext)]
    return s


def select(category: str, c: dict, F: pd.DataFrame | None = None) -> pd.DataFrame:
    """All open repository files that belong to a csbj_dl category (before de-duplication with local copies)."""
    F = file_table(c) if F is None else F
    es, wt = F.experimental_strategy, F.workflow_type
    if category == "multiome":
        m = (es.eq("10x Multiome") & wt.eq("Expression Matrix")) | es.eq("10x Multiome|ATAC-Seq|Single-nucleus RNA-Seq")
    elif category == "atac_seq":
        m = es.eq("ATAC-Seq")
    elif category == "visium":
        m = es.eq("Spatial Transcriptomics") & (wt.isin(["Spatial Imaging Data", "Experimental Metadata"]) | F.data_format.eq("tif"))
    elif category == "pathomics":
        m = es.eq("Segmentation Masks & Pathomics Vectors")
    elif category == "annotation":
        names = list(c["annotation_files"].values())
        m = wt.eq("Aggregated Clustered Data") & F.file_name.apply(lambda n: any(n.endswith(x) for x in names))
    else:
        raise SystemExit(f"unknown category {category}")
    return F[m].copy()


def dest(row, c: dict, category: str) -> Path:
    sub = "_atlas" if category == "annotation" else str(row.redcap_id)
    return Path(c["raw_dir"]) / category / sub / row.file_name
