"""11a_histology_embed: spot-centred H&E patches -> fixed-encoder embeddings + interpretable
morphology features, for GSE211785 Visium (same-section H&E) and KPMP Visium (full-res tif).

Idempotent: a section is skipped when its npz exists. KPMP tifs still downloading are skipped
until their size matches the download manifest. Outputs
  data/processed/histology_embeddings/<cohort>__<section>.npz  (barcode, emb float16)
  data/processed/histology_embeddings/<cohort>__<section>.features.tsv
  results/11a_histology_embed/section_index.tsv, features_<cohort>.tsv.gz
Usage: python 11a_histology_embed.py [gse211785|kpmp|all]
"""
from __future__ import annotations
import gzip, importlib, json, re, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from repro import ROOT, load_config, set_global_seed, stage_dir, write_provenance
from histo import SPOT_UM, crop, resize, he_features, background_rgb, Encoder

STAGE = "11a_histology_embed"
EMB = ROOT / "data/processed/histology_embeddings"


def slide_key(name: str) -> tuple:
    """(slide serial, slide number, capture area) e.g. V11Y24-79_A1 == V11Y24-079-A1."""
    m = re.search(r"(V1\d[A-Z]\d{2})[-_](\d+)[-_]([A-D]1)", name)
    return (m.group(1), int(m.group(2)), m.group(3)) if m else (name,)


def gse_sections(cfg) -> list[dict]:
    raw, he = ROOT / "data/raw/GSE211785", ROOT / "data/processed/gse211785_he"
    meta = pd.read_csv(raw / "GSE211785_ST_metadata.txt.gz", sep="\t", low_memory=False, index_col=0)
    sfx = meta.index.str.rsplit("_", n=1).str[1]
    sec_of_sfx = dict(zip(sfx, meta["orig.ident"]))
    bc = pd.Series(meta.index.str.rsplit("_", n=1).str[0], index=meta.index)
    amap = importlib.import_module("10_visium_common").barcode_array_map()
    jsons = {slide_key(p.name): p for p in list(raw.glob("*.json.gz")) + list(he.glob("*.json"))}
    out = []
    for tif in sorted(he.glob("*.tif")):
        hk = re.search(r"HK\d+", tif.name).group(0)
        js = [jsons[slide_key(tif.name)]] if slide_key(tif.name) in jsons else []
        if not js:
            continue
        sec = f"{hk}_ST"
        spots = meta[meta["orig.ident"] == sec]
        d = pd.DataFrame({"spot_id": spots.index, "barcode": bc[spots.index].values})
        d = d.join(amap, on="barcode")
        opener = gzip.open if js[0].name.endswith(".gz") else open
        with opener(js[0], "rt") as fh:
            ol = pd.DataFrame(json.load(fh)["oligo"])
        dia = float(ol.dia.median())
        d = d.merge(ol[["row", "col", "imageX", "imageY"]], left_on=["array_row", "array_col"],
                    right_on=["row", "col"], how="left")
        out.append({"cohort": "gse211785", "section": sec, "participant": hk, "tif": tif,
                    "um_per_px": SPOT_UM / dia, "spots": d, "orient": "loupe", "oligo": ol})
    return out


def kpmp_sections(cfg) -> list[dict]:
    man = pd.read_csv(ROOT / "data/raw/kpmp/download_manifest_visium_tif.tsv", sep="\t")
    size = dict(zip(man.file_name, man.file_size.astype(float).astype(int)))
    out = []
    for sfj in sorted((ROOT / "data/processed/kpmp_visium").glob("*/*/outs/spatial/scalefactors_json.json")):
        sp_dir = sfj.parent; sample = sp_dir.parents[1].name; pid = sp_dir.parents[2].name
        tifs = [t for t in (ROOT / "data/raw/kpmp/Spatial_Transcriptomics" / pid).glob(f"*{sample}*.tif")]
        tifs = [t for t in tifs if t.stat().st_size == size.get(t.name, -1)]
        if not tifs:
            continue
        sf = json.loads(sfj.read_text())
        tp = sp_dir / "tissue_positions.csv"
        tp = tp if tp.exists() else sp_dir / "tissue_positions_list.csv"
        d = pd.read_csv(tp, header=0 if tp.name == "tissue_positions.csv" else None)
        d.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        off = d[d.in_tissue == 0][["pxl_row", "pxl_col"]].sample(frac=1, random_state=0).head(300)
        d = d[d.in_tissue == 1].copy(); d["spot_id"] = d.barcode
        out.append({"cohort": "kpmp", "section": f"{pid}__{sample}", "participant": pid, "tif": tifs[0],
                    "um_per_px": SPOT_UM / float(sf["spot_diameter_fullres"]), "spots": d, "orient": "rowcol",
                    "off_pts": list(zip(off.pxl_row, off.pxl_col))})
    return out


def load_image(path: Path):
    t = tifffile.TiffFile(path)
    try:
        return tifffile.memmap(path)
    except (ValueError, TypeError):
        return t.series[0].levels[0].asarray() if hasattr(t.series[0], "levels") else t.asarray()


def centres(sec, img, half, hcfg):
    d = sec["spots"]
    if sec["orient"] == "rowcol":
        sec["bg"] = background_rgb(img, sec["off_pts"], half)
        return d.pxl_row.values, d.pxl_col.values
    # Loupe json orientation: the one under which in-tissue spots carry more tissue than the
    # off-tissue oligo positions of the same slide (AUC of tissue fraction) is used.
    from sklearn.metrics import roc_auc_score
    ins = d.dropna(subset=["imageX"]).sample(min(300, len(d)), random_state=0)
    off = sec["oligo"].merge(d[["array_row", "array_col"]], left_on=["row", "col"],
                             right_on=["array_row", "array_col"], how="left", indicator=True)
    off = off[off._merge == "left_only"]
    off = off.sample(min(300, len(off)), random_state=0)
    # background colour from off-tissue oligo positions (the glass margin dominates them)
    sec["bg"] = background_rgb(img, list(zip(off.imageY, off.imageX)), half)
    H, W = img.shape[:2]

    def tf(tab, name):
        t, fr, fc = name.split("_")
        r, c = (tab.imageY.values, tab.imageX.values) if t == "yx" else (tab.imageX.values, tab.imageY.values)
        return (H - 1 - r if fr == "f" else r), (W - 1 - c if fc == "f" else c)

    names = [f"{t}_{a}_{b}" for t in ("yx", "xy") for a in ("n", "f") for b in ("n", "f")]
    score = {}
    for name in names:
        fr = []
        for tab in (ins, off):
            for r, c in zip(*tf(tab, name)):
                p = crop(img, r, c, half)
                fr.append(he_features(p, hcfg, sec["bg"])["tissue_frac"] if p is not None else 0.0)
        y = np.r_[np.ones(len(ins)), np.zeros(len(off))]
        score[name] = round(float(roc_auc_score(y, fr)), 4) if len(off) else np.nan
    best = max(score, key=lambda k: score[k])
    sec["orient_chosen"], sec["orient_tissue"] = best, score
    if score[best] < hcfg["min_orientation_auc"]:
        return None, None
    return tf(d, best)


def process(sec, enc, hcfg) -> dict:
    stem = EMB / f"{sec['cohort']}__{sec['section']}"
    if Path(f"{stem}.npz").exists():
        return {"section": sec["section"], "status": "exists"}
    t0 = time.time()
    img = load_image(sec["tif"])
    half = int(round(hcfg["field_um"] / sec["um_per_px"] / 2))
    rr, cc = centres(sec, img, half, hcfg)
    if rr is None:   # image cannot be registered to the spot array -> section excluded, reported
        return {"cohort": sec["cohort"], "section": sec["section"], "participant": sec["participant"],
                "status": "excluded_unregistered", "orient_tissue": json.dumps(sec["orient_tissue"])}
    patches, feats, ids = [], [], []
    for sid, r, c in zip(sec["spots"].spot_id.values, rr, cc):
        if not (np.isfinite(r) and np.isfinite(c)):
            continue
        p = crop(img, r, c, half)
        if p is None:
            continue
        p = resize(p, hcfg["patch_px"])
        f = he_features(p, hcfg, sec["bg"])
        f.update({"spot_id": sid, "px_row": r, "px_col": c})
        feats.append(f)
        if f["tissue_frac"] >= hcfg["min_tissue_frac"]:
            patches.append(p); ids.append(sid)
    F = pd.DataFrame(feats); F["tissue_pass"] = F.spot_id.isin(ids)
    emb = enc(np.stack(patches)) if patches else np.zeros((0, 1))
    np.savez_compressed(f"{stem}.npz", spot_id=np.array(ids), emb=emb.astype(np.float16))
    F.to_csv(f"{stem}.features.tsv", sep="\t", index=False)
    return {"cohort": sec["cohort"], "section": sec["section"], "participant": sec["participant"],
            "tif": sec["tif"].name, "um_per_px": sec["um_per_px"], "field_px_native": 2 * half,
            "n_spots": len(sec["spots"]), "n_patches": len(F), "n_tissue_pass": len(ids),
            "orientation": sec.get("orient_chosen", "rowcol"), "bg_rgb": ",".join(f"{x:.3f}" for x in sec["bg"]), "orient_tissue": json.dumps(sec.get("orient_tissue", {})),
            "minutes": (time.time() - t0) / 60, "status": "done"}


def main():
    cfg = load_config(); set_global_seed(cfg["seed"]); h = cfg["histology"]
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    EMB.mkdir(parents=True, exist_ok=True)
    out = stage_dir(STAGE)
    secs = (gse_sections(cfg) if which in ("gse211785", "all") else []) + \
           (kpmp_sections(cfg) if which in ("kpmp", "all") else [])
    enc = Encoder(h["encoder"], h["encoder_revision"])
    rows = []
    for s in secs:
        r = process(s, enc, h); print(r, flush=True); rows.append(r)
    idx_f = out / "section_index.tsv"
    old = pd.read_csv(idx_f, sep="\t") if idx_f.exists() else pd.DataFrame()
    new = pd.DataFrame([r for r in rows if r.get("status") in ("done", "excluded_unregistered")])
    idx = pd.concat([old, new]).drop_duplicates("section", keep="last") if len(new) or len(old) else new
    idx.to_csv(idx_f, sep="\t", index=False)
    for coh in ["gse211785", "kpmp"]:
        fs = sorted(EMB.glob(f"{coh}__*.features.tsv"))
        if fs:
            pd.concat([pd.read_csv(f, sep="\t").assign(section=f.name.split("__", 1)[1].replace(".features.tsv", ""))
                       for f in fs]).to_csv(out / f"features_{coh}.tsv.gz", sep="\t", index=False)
    write_provenance(STAGE, [s["tif"] for s in secs], [idx_f, *sorted(out.glob("features_*.tsv.gz"))],
                     cfg["seed"], extra={"encoder": h["encoder"], "revision": h["encoder_revision"],
                                         "n_sections": int(len(idx))}, hash_limit=0)


if __name__ == "__main__":
    main()
