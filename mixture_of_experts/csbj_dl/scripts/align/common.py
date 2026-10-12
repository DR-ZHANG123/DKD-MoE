"""Shared helpers of the align module: config, section registry, provenance.

Section registry rows: cohort, section, participant, group, emb (npz of spot_id/emb), feats
(morphology tsv), and the expression source (KPMP 10x h5 + positions, or GSE211785 author matrix).
Existing embeddings come from stage 11a (read-only); new KPMP sections embedded by
01_embed_new.py live under paths.emb_new with the same file layout.
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CSBJ = ROOT / "csbj_dl"
sys.path.insert(0, str(ROOT / "scripts/lib"))
sys.path.insert(0, str(ROOT / "scripts/stages"))
import repro  # noqa: E402

CFG_PATH = CSBJ / "config/align.yaml"


TRACK = os.environ.get("ALIGN_TRACK", "")     # "" = primary (full-resolution patches); "lowres" = matched resolution


def load_cfg() -> dict:
    """align.yaml; with ALIGN_TRACK set, the tracks.<name> block overrides paths/data/histology keys."""
    cfg = yaml.safe_load(CFG_PATH.read_text())
    if TRACK:
        for k, v in cfg["tracks"][TRACK].items():
            if isinstance(v, dict):
                cfg.setdefault(k, {}).update(v)
            else:
                cfg[k] = v
    if os.environ.get("ALIGN_DEVICE"):        # shared host: run on the GPU with most free memory
        cfg["device"] = os.environ["ALIGN_DEVICE"]
    return cfg


def run_cfg() -> dict:
    return repro.load_config()


def p(rel: str) -> Path:
    q = Path(rel)
    return q if q.is_absolute() else ROOT / q


def tname(name: str) -> str:
    """result / source-data name for the active track."""
    return f"{name}_{TRACK}" if TRACK else name


def result_dir(name: str) -> Path:
    d = CSBJ / "results" / tname(name)
    d.mkdir(parents=True, exist_ok=True)
    return d


def provenance(name: str, inputs, outputs, seed, extra=None):
    """repro.write_provenance into csbj_dl/results/<name>/ (stage path relative to results/)."""
    extra = dict(extra or {}); extra["align_config_sha"] = repro.config_sha(CFG_PATH)
    extra["track"] = TRACK or "primary"
    return repro.write_provenance(f"../csbj_dl/results/{tname(name)}", inputs, outputs, seed, extra=extra)


def kpmp_groups() -> pd.Series:
    g = importlib.import_module("00b_download_kpmp").participant_groups()
    return g.set_index("Participant ID").group


def gse_groups() -> pd.Series:
    t = pd.read_csv(ROOT / "results/01_cohort_audit/sample_table_GSE211785_visium.tsv", sep="\t")
    return t.set_index("donor").diagnosis_geo


def splits() -> pd.DataFrame:
    cfg = load_cfg()
    f = p(cfg["paths"]["splits"])
    if not f.exists():
        raise SystemExit(f"{f} not written yet (expr module)")
    sp = pd.read_csv(f, sep="\t", dtype={"participant": str}).assign(source="splits")
    x = CSBJ / "config/align_extra_partitions.tsv"     # participants absent from splits.tsv (alignment only)
    if x.exists():
        e = pd.read_csv(x, sep="\t", dtype={"participant": str})
        sp = pd.concat([sp, e[~e.participant.isin(sp.participant)]], ignore_index=True)
    return sp.set_index("participant")


def _kpmp_expr_paths() -> pd.DataFrame:
    """participant, sample, h5, positions for every KPMP Visium section (existing + new)."""
    cfg = load_cfg()
    t = pd.read_csv(ROOT / "results/10b_visium_kpmp/sample_table.tsv", sep="\t", dtype={"participant": str})
    t = t[["participant", "sample", "h5", "positions"]].assign(source="10b")
    nf = p(cfg["paths"]["new_sections"])
    if nf.exists():
        n = new_section_table()
        if len(n):
            t = pd.concat([t, n[["participant", "sample", "h5", "positions"]].assign(source="new")])
    t["section"] = t.participant + "__" + t["sample"]
    return t.drop_duplicates("section", keep="first").set_index("section")


def new_section_table() -> pd.DataFrame:
    """Sections downloaded by the data module; resolves h5/positions/tif/scalefactors from the
    spatial output folder of each section (layout as KPMP 'outs/')."""
    cfg = load_cfg()
    n = pd.read_csv(p(cfg["paths"]["new_sections"]), sep="\t", dtype=str)
    if "status" in n.columns:
        n = n[n.status == "ok"]
    rows = []
    for r in n.to_dict("records"):
        part = r.get("participant") or r.get("redcap_id")
        samp = r.get("sample") or r.get("section")
        dd = r.get("dir") or r.get("processed_dir")
        d = Path(dd) if isinstance(dd, str) and dd else None
        h5 = r.get("h5") or (next(iter(sorted(d.rglob("filtered_feature_bc_matrix.h5"))), None) if d else None)
        if not part or not samp or not h5:
            continue
        sp = Path(h5).parent / "spatial"
        pos = r.get("positions") or next(iter(sorted(sp.glob("tissue_positions*.csv"))), None)
        sf = r.get("scalefactors") or (sp / "scalefactors_json.json")
        tif = r.get("tif") or r.get("fullres_tif") or r.get("tif_path")
        tif = tif if isinstance(tif, str) and tif else None
        if tif is None and d is not None:
            tif = next(iter(sorted(d.rglob("*.tif"))), None)
        rows.append({"participant": part, "sample": samp, "h5": str(h5), "positions": str(pos) if pos else None,
                     "scalefactors": str(sf), "tif": str(tif) if tif else None})
    return pd.DataFrame(rows)


def registry(primary: bool = False) -> pd.DataFrame:
    """All embedded sections usable for alignment (primary track, or the active track's own index)."""
    cfg = load_cfg()
    if TRACK and not primary:
        return _track_registry(cfg)
    idx = pd.read_csv(ROOT / "results/11a_histology_embed/section_index.tsv", sep="\t", dtype={"participant": str})
    idx = idx[idx.status == "done"].copy()
    idx["emb_dir"] = str(p(cfg["paths"]["emb_existing"]))
    newi = p(cfg["paths"]["emb_new"]) / "section_index.tsv"
    if newi.exists():
        ni = pd.read_csv(newi, sep="\t", dtype={"participant": str})
        ni = ni[(ni.status == "done") & ~ni.section.isin(idx.section)].copy()
        ni["emb_dir"] = str(p(cfg["paths"]["emb_new"]))
        idx = pd.concat([idx, ni])
    idx["emb"] = idx.emb_dir + "/" + idx.cohort + "__" + idx.section + ".npz"
    idx["feats"] = idx.emb_dir + "/" + idx.cohort + "__" + idx.section + ".features.tsv"
    kp = _kpmp_expr_paths()
    kg, gg = kpmp_groups(), gse_groups()
    idx["group"] = np.where(idx.cohort == "kpmp", idx.participant.map(kg), idx.participant.map(gg))
    idx["h5"] = idx.section.map(kp.h5); idx["positions"] = idx.section.map(kp.positions)
    idx = idx[(idx.cohort == "gse211785") | idx.h5.notna()]
    idx = idx[~idx.participant.isin(cfg["data"]["exclude_participants"])]
    return idx[["cohort", "section", "participant", "group", "emb", "feats", "h5", "positions",
                "um_per_px", "field_px_native", "tif"]].reset_index(drop=True)


def set_seed(seed: int):
    repro.set_global_seed(seed)


def tif_path(cohort: str, section: str, tif_name: str) -> Path | None:
    """Full-resolution H&E image of a section (stage 11a / data module locations)."""
    if Path(str(tif_name)).is_absolute() and Path(str(tif_name)).exists():
        return Path(str(tif_name))
    if cohort == "gse211785":
        q = ROOT / "data/processed/gse211785_he" / str(tif_name)
        return q if q.exists() else None
    part = section.split("__")[0]
    q = ROOT / "data/raw/kpmp/Spatial_Transcriptomics" / part / str(tif_name)
    if q.exists():
        return q
    cfg = load_cfg()
    if p(cfg["paths"]["new_sections"]).exists():
        n = new_section_table()
        n = n[(n.participant + "__" + n["sample"]) == section]
        if len(n) and n.tif.iloc[0] and Path(n.tif.iloc[0]).exists():
            return Path(n.tif.iloc[0])
    return None


def rpath(name: str) -> Path:
    """results/<name> of the active track (read side)."""
    return CSBJ / "results" / tname(name)


def src(fname: str) -> Path:
    """figures/source_data/<fname>, with the track inserted after the 'align_' prefix."""
    d = CSBJ / "figures/source_data"; d.mkdir(parents=True, exist_ok=True)
    return d / (fname.replace("align_", f"align_{TRACK}_", 1) if TRACK else fname)


def _track_registry(cfg) -> pd.DataFrame:
    """Sections embedded by the active track (01b_embed_lowres): same columns as registry()."""
    d = p(cfg["paths"]["emb_dir"])
    idx = pd.read_csv(d / "section_index.tsv", sep="\t", dtype={"participant": str})
    idx = idx[idx.status == "done"].copy()
    idx["emb"] = str(d) + "/" + idx.cohort + "__" + idx.section + ".npz"
    idx["feats"] = str(d) + "/" + idx.cohort + "__" + idx.section + ".features.tsv"
    kp = _kpmp_expr_paths()
    ext = pd.read_csv(p(cfg["paths"]["external_sections"]), sep="\t")
    ext["sec"] = ext.section + "_r" + ext.region.astype(str)
    eh = dict(zip(ext.sec, ext.processed_dir + "/outs/filtered_feature_bc_matrix.h5"))
    ep = dict(zip(ext.sec, ext.processed_dir + "/outs/spatial/tissue_positions.csv"))
    idx["h5"] = np.where(idx.cohort == "kpmp", idx.section.map(kp.h5), idx.section.map(eh))
    idx["positions"] = np.where(idx.cohort == "kpmp", idx.section.map(kp.positions), idx.section.map(ep))
    idx["tif"] = idx["image"]
    idx = idx[~idx.participant.isin(cfg["data"]["exclude_participants"])]
    return idx[["cohort", "section", "participant", "group", "emb", "feats", "h5", "positions",
                "um_per_px", "field_px_native", "tif"]].reset_index(drop=True)
