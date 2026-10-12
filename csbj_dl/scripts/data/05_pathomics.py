"""05_pathomics: participant-level table of KPMP segmentation-derived (pathomics) features.

KPMP packages come in two layouts:
 A  <slide>_features[_v2].xlsx with sheets gloms / s_gloms / tubs / arts (v2: pixels; without _v2: microns);
 B  final_combined_features_<compartment>.xlsx (one workbook per structure type, pixels).
Pixel measures are converted with the slide's microns-per-pixel read from its svs header (02_download).
Core features available in both layouts are pooled over all structures of all slides of a participant
(counts summed, per-structure measures summarised by the median). Layout-B-only features (nuclear,
texture, colour) are summarised separately for participants with B packages.
Outputs csbj_dl/results/data_pathomics/{pathomics_features.tsv, pathomics_features_extended.tsv,
slide_table.tsv, DATA_DICTIONARY.md}."""
from __future__ import annotations
import json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from dcommon import cfg, result_dir, write_provenance, set_global_seed, groups

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

# core per-structure measures: (structure, output name, source column, unit kind)
CORE = [("glom", "glom_area_um2", "Area", "area"), ("glom", "glom_mesangial_fraction", "Mesangial Fraction", "unitless"),
        ("tubule", "tubule_tbm_thickness_um", "Average TBM Thickness", "length"),
        ("tubule", "tubule_cell_thickness_um", "Average Cell Thickness", "length"),
        ("tubule", "tubule_luminal_fraction", "Luminal Fraction", "unitless"), ("artery", "artery_area_um2", "Arterial Area", "area")]
SHEET_A = {"gloms": "glom", "s_gloms": "sglom", "tubs": "tubule", "arts": "artery"}
FILE_B = {"non_globally_sclerotic_gloms": "glom", "globally_sclerotic_gloms": "sglom", "tubules": "tubule", "muscular_vessels": "artery"}


def unit_kind(col: str) -> str:
    c = col.lower()
    if "by object area" in c or "fraction" in c or "ratio" in c or c.startswith(("mean red", "mean green", "mean blue", "standard deviation")) \
            or c.split()[0] in ("contrast", "homogeneity", "correlation", "energy") or "number" in c or c == "in medulla":
        return "unitless"
    if "distance transform" in c:
        return "pixel_scale"            # kept as reported; depends on resolution
    if "area" in c:
        return "area"
    if any(k in c for k in ("perimeter", "length", "radius", "thickness")):
        return "length"
    return "unitless"


def convert(df: pd.DataFrame, mpp: float | None, in_pixels: bool) -> pd.DataFrame:
    if not in_pixels:
        return df
    out = df.copy()
    for c in out.columns:
        k = unit_kind(c)
        if k in ("area", "length"):
            out[c] = out[c] * (mpp ** 2 if k == "area" else mpp) if mpp else np.nan
    return out


def read_package(d: Path) -> tuple[dict, list[dict]]:
    mpp_map = json.loads((d / "slide_mpp.json").read_text()) if (d / "slide_mpp.json").exists() else {}
    mpp = next((v for v in mpp_map.values() if v), None)
    tables, layout = {}, None
    for f in d.glob("*.xlsx"):
        if f.name.startswith("final_combined_features_"):
            layout = "B"
            key = FILE_B.get(f.stem.replace("final_combined_features_", ""))
            if key:
                tables[key] = convert(pd.read_excel(f, sheet_name="Features"), mpp, True)
        else:
            layout = "A"
            x = pd.read_excel(f, sheet_name=None)
            px = "_v2" in f.name
            for s, key in SHEET_A.items():
                if s in x:
                    tables[key] = convert(x[s], mpp, px)
    slide = {"package": d.name, "layout": layout, "mpp": mpp, "stain": "PAS" if "_PAS_" in " ".join(list(mpp_map) + [f.name for f in d.glob("*.ome.xml")]) else "other",
             **{f"n_{k}": len(v) for k, v in tables.items()}}
    return slide, tables


def summarise(pid: str, packs: list[tuple[dict, dict]]) -> tuple[dict, dict]:
    pool = {}
    for _, t in packs:
        for k, v in t.items():
            pool.setdefault(k, []).append(v)
    pool = {k: pd.concat(v, ignore_index=True) for k, v in pool.items()}
    n = lambda k: len(pool.get(k, []))
    core = {"participant": pid, "n_slides": len(packs), "n_glomeruli": n("glom"), "n_sclerotic_glomeruli": n("sglom"),
            "pct_globally_sclerotic": 100 * n("sglom") / max(1, n("glom") + n("sglom")) if n("glom") + n("sglom") else np.nan,
            "n_tubules": n("tubule"), "n_arteries": n("artery")}
    for struct, name, col, _ in CORE:
        t = pool.get(struct)
        core[name] = float(t[col].median()) if t is not None and col in t and t[col].notna().any() else np.nan
    ext = {"participant": pid}
    for struct in ("glom", "tubule", "artery"):
        t = pool.get(struct)
        if t is None or "Nuclei Number" not in t:
            continue
        for c in t.columns:
            if c in ("compartment_id",) or not np.issubdtype(t[c].dtype, np.number):
                continue
            ext[f"{struct}__{c}"] = float(t[c].median())
    return core, ext


def main():
    c = cfg(); set_global_seed(c["seed"])
    out = result_dir("data_pathomics")
    base = Path(c["raw_dir"]) / "pathomics"
    G = groups()
    slides, by_pid = [], {}
    for d in sorted(p for p in base.glob("*/*") if p.is_dir()):
        s, t = read_package(d)
        s["participant"] = d.parent.name; slides.append(s)
        by_pid.setdefault(d.parent.name, []).append((s, t))
    S = pd.DataFrame(slides); S.to_csv(out / "slide_table.tsv", sep="\t", index=False)
    rows = [summarise(p, v) for p, v in by_pid.items()]
    core = pd.DataFrame([r[0] for r in rows]); ext = pd.DataFrame([r[1] for r in rows]).dropna(axis=1, how="all")
    for X in (core, ext):
        X.insert(1, "group", X.participant.map(G.group).fillna("not_in_clinical_table"))
    core["layouts"] = core.participant.map(S.groupby("participant").layout.agg(lambda v: ",".join(sorted(set(map(str, v))))))
    core.to_csv(out / "pathomics_features.tsv", sep="\t", index=False)
    ext[ext.iloc[:, 2:].notna().any(axis=1)].to_csv(out / "pathomics_features_extended.tsv", sep="\t", index=False)
    write_dictionary(out, core, S)
    print(core.groupby("group").size().to_string(), "\nmissing mpp slides:", int(S.mpp.isna().sum()))
    write_provenance("data_pathomics", [Path(__file__).resolve().parents[2] / "results/data_download/pathomics/download_manifest_pathomics.tsv"],
                     [out / "pathomics_features.tsv", out / "pathomics_features_extended.tsv", out / "slide_table.tsv"], c["seed"], cfg=c,
                     extra={"n_participants": len(core), "n_slides": len(S)})


def write_dictionary(out: Path, core: pd.DataFrame, S: pd.DataFrame) -> None:
    desc = {"participant": "KPMP participant ID", "group": "study group (main-pipeline definition)", "n_slides": "segmented slides (packages)",
            "n_glomeruli": "non-globally-sclerotic glomeruli segmented", "n_sclerotic_glomeruli": "globally sclerotic glomeruli segmented",
            "pct_globally_sclerotic": "100 x sclerotic / all glomeruli", "n_tubules": "tubules segmented", "n_arteries": "arteries/arterioles segmented",
            "glom_area_um2": "median glomerular tuft area (um^2)", "glom_mesangial_fraction": "median mesangial area / glomerular area",
            "tubule_tbm_thickness_um": "median tubular basement membrane thickness (um)", "tubule_cell_thickness_um": "median tubular epithelial thickness (um)",
            "tubule_luminal_fraction": "median luminal area / tubule area", "artery_area_um2": "median arterial cross-sectional area (um^2)",
            "layouts": "package layout(s): A = 4-sheet workbook, B = per-compartment workbooks"}
    lines = ["# KPMP pathomics participant table: data dictionary", "",
             "Source: KPMP open repository, 'Segmentation Masks & Pathomics Vectors' (panoptic segmentation of PAS whole-slide images,",
             "pathologist-checked). Per-structure measures pooled over all slides of a participant and summarised by the median;",
             "pixel units converted to microns with each slide's svs microns-per-pixel.", "",
             "| column | description |", "|---|---|"] + [f"| {k} | {v} |" for k, v in desc.items()] + ["",
             "pathomics_features_extended.tsv: layout-B participants only; columns `<structure>__<feature>` = median over structures of the",
             "KPMP feature (definitions in each package README; areas/lengths converted to um^2/um, distance-transform features left in pixels,",
             "colour/texture/ratios unitless).", "", f"Slides: {len(S)}; participants: {len(core)}; slides without svs resolution: {int(S.mpp.isna().sum())}."]
    (out / "DATA_DICTIONARY.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
