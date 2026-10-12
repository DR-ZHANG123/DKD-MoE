"""01b_embed_lowres: matched-resolution patch embeddings for the lowres track (ALIGN_TRACK=lowres).

GSE339455 (Visium CytAssist, hypertensive nephropathy) publishes only the down-sampled H&E
(tissue_hires_image, 2.9-4.0 um/px), so all cohorts are re-embedded at one common coarse scale:
a field of `field_um` centred on each spot is cropped from the native image, resampled to
target_um_per_px (area interpolation), optionally stain-normalised (Reinhard: per-section LAB mean/sd of
tissue pixels mapped to the median of KPMP sections; label-free), then resized to patch_px and passed
through the same fixed encoder. KPMP / GSE211785 use the same spots as the primary track
(tissue-passing patches of stage 11a / 01_embed_new); GSE339455 regions (one tissue region = one patient,
from the data module) are filtered for tissue with the stage-11a rule on the coarse patch.
Outputs <emb_dir>/<cohort>__<section>.npz|.features.tsv|.lowres.npz, <emb_dir>/section_index.tsv
"""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("ALIGN_TRACK", "lowres")
import common as C  # noqa: E402

os.environ.setdefault("HF_HOME", str(C.ROOT / "data/hf_cache"))
histo = importlib.import_module("histo"); s11 = importlib.import_module("11a_histology_embed")
RES = "align_embed"


def primary_sections() -> list[dict]:
    """KPMP + GSE211785 sections of the primary registry, with stage-11a spot centres."""
    out = []
    for r in C.registry(primary=True).itertuples():
        F = pd.read_csv(r.feats, sep="\t")
        F = F[F.tissue_pass]
        tif = C.tif_path(r.cohort, r.section, r.tif)
        if tif is None:
            continue
        out.append({"cohort": r.cohort, "section": r.section, "participant": r.participant, "group": r.group,
                    "image": tif, "um_per_px": float(r.um_per_px), "spot_id": F.spot_id.values,
                    "rc": np.c_[F.px_row.values, F.px_col.values], "tissue_filter": False})
    return out


def gse339455_sections(cfg) -> list[dict]:
    t = pd.read_csv(C.p(cfg["paths"]["external_sections"]), sep="\t")
    reg = pd.read_csv(C.p(cfg["paths"]["external_regions"]), sep="\t")
    t = t[t.usable.astype(str) == "True"]
    out = []
    for r in t.itertuples():
        if r.n_spots < cfg["data"]["min_spots_section"]:
            continue
        sp = Path(r.processed_dir) / "outs/spatial"
        sf = json.loads((sp / "scalefactors_json.json").read_text())
        pos = pd.read_csv(sp / "tissue_positions.csv")
        pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        ids = reg[(reg.slide == r.section) & (reg.region == r.region)].barcode
        d = pos.set_index("barcode").loc[ids]
        s = sf["tissue_hires_scalef"]
        off = pos[pos.in_tissue == 0].sample(frac=1, random_state=0).head(300)
        out.append({"cohort": "gse339455", "section": f"{r.section}_r{r.region}", "participant": r.donor_unit,
                    "group": r.disease, "subgroup": r.group, "image": sp / "tissue_hires_image.png",
                    "um_per_px": histo.SPOT_UM / (sf["spot_diameter_fullres"] * s), "spot_id": d.index.values,
                    "rc": np.c_[d.pxl_row.values * s, d.pxl_col.values * s],
                    "off": np.c_[off.pxl_row.values * s, off.pxl_col.values * s], "tissue_filter": True})
    return out


def load(path: Path):
    if str(path).endswith(".png"):
        import cv2
        return cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    return s11.load_image(path)


def coarse_patches(sec, H) -> tuple[np.ndarray, np.ndarray]:
    img = load(sec["image"])
    half = int(round(H["field_um"] / sec["um_per_px"] / 2))
    n_low = int(round(H["field_um"] / H["target_um_per_px"]))
    P, keep = [], []
    for i, (r, c) in enumerate(sec["rc"]):
        p = histo.crop(img, r, c, max(half, 1))
        if p is None:
            continue
        P.append(histo.resize(p, n_low)); keep.append(i)
    bg = None
    if "off" in sec:
        bg = histo.background_rgb(img, list(map(tuple, sec["off"])), max(half, 1))
    return np.stack(P), np.array(keep), bg


def lab_stats(P, bg, hcfg):
    import cv2
    lab = np.stack([cv2.cvtColor(p, cv2.COLOR_RGB2LAB) for p in P]).reshape(-1, 3).astype(np.float32)
    f = P.reshape(-1, 3).astype(np.float32) / 255.0
    bgv = np.array([0.95, 0.95, 0.95]) if bg is None else bg
    tis = (np.sqrt(((f - bgv) ** 2).sum(-1)) > hcfg["tissue_delta"]) & (f.mean(-1) < bgv.mean() - hcfg["tissue_min_darker"])
    x = lab[tis] if tis.sum() > 1000 else lab
    return x.mean(0), x.std(0) + 1e-3


def reinhard(P, mu, sd, mu_t, sd_t):
    import cv2
    out = []
    for p in P:
        lab = cv2.cvtColor(p, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab = (lab - mu) / sd * sd_t + mu_t
        out.append(cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB))
    return np.stack(out)


def main():
    cfg = C.load_cfg(); rc = C.run_cfg(); C.set_seed(cfg["seed"]); H = cfg["histology_lowres"]; h = rc["histology"]
    emb = C.p(cfg["paths"]["emb_dir"]); emb.mkdir(parents=True, exist_ok=True)
    secs = primary_sections() + gse339455_sections(cfg)
    # pass 1: coarse patches + stain statistics (cached)
    for s in secs:
        f = emb / f"{s['cohort']}__{s['section']}.lowres.npz"
        if not f.exists():
            P, keep, bg = coarse_patches(s, H)
            mu, sd = lab_stats(P, bg, h)
            np.savez_compressed(f, patches=P, spot_id=s["spot_id"][keep].astype(str), lab_mu=mu, lab_sd=sd,
                                bg=bg if bg is not None else np.array([np.nan] * 3))
            print("cropped", s["section"], len(keep), flush=True)
    st = {s["section"]: np.load(emb / f"{s['cohort']}__{s['section']}.lowres.npz") for s in secs}
    kp = [s["section"] for s in secs if s["cohort"] == "kpmp"]
    mu_t = np.median([st[k]["lab_mu"] for k in kp], 0); sd_t = np.median([st[k]["lab_sd"] for k in kp], 0)
    enc = histo.Encoder(h["encoder"], h["encoder_revision"], device=cfg["device"])
    rows = []
    for s in secs:
        stem = emb / f"{s['cohort']}__{s['section']}"
        z = st[s["section"]]; P = z["patches"]; ids = z["spot_id"]
        if not Path(f"{stem}.npz").exists():
            if H["stain_normalise"]:
                P = reinhard(P, z["lab_mu"], z["lab_sd"], mu_t, sd_t)
            big = np.stack([histo.resize(p, h["patch_px"]) for p in P])
            bg = z["bg"] if np.isfinite(z["bg"]).all() else None
            F = pd.DataFrame([histo.he_features(p, h, bg) for p in big]); F["spot_id"] = ids
            F["px_row"] = s["rc"][pd.Index(s["spot_id"].astype(str)).get_indexer(ids), 0]
            F["px_col"] = s["rc"][pd.Index(s["spot_id"].astype(str)).get_indexer(ids), 1]
            F["tissue_pass"] = F.tissue_frac >= h["min_tissue_frac"] if s["tissue_filter"] else True
            ok = F.tissue_pass.values
            e = np.vstack([enc(big[ok][i:i + 512], bs=cfg["embed_batch"]) for i in range(0, ok.sum(), 512)])
            np.savez_compressed(f"{stem}.npz", spot_id=ids[ok], emb=e.astype(np.float16))
            F.to_csv(f"{stem}.features.tsv", sep="\t", index=False)
            print("embedded", s["section"], int(ok.sum()), flush=True)
        n = len(np.load(f"{stem}.npz")["spot_id"])
        rows.append({"cohort": s["cohort"], "section": s["section"], "participant": s["participant"],
                     "group": s["group"], "subgroup": s.get("subgroup", ""), "image": str(s["image"]),
                     "um_per_px": s["um_per_px"], "field_px_native": 2 * int(round(H["field_um"] / s["um_per_px"] / 2)),
                     "n_tissue_pass": n, "status": "done"})
    I = pd.DataFrame(rows); I.to_csv(emb / "section_index.tsv", sep="\t", index=False)
    out = C.result_dir(RES); I.to_csv(out / "section_index.tsv", sep="\t", index=False)
    pd.DataFrame({"lab_mu_target": mu_t, "lab_sd_target": sd_t}, index=list("Lab")).to_csv(out / "stain_target.tsv", sep="\t")
    print(I.groupby(["cohort", "group"]).agg(sections=("section", "size"), spots=("n_tissue_pass", "sum")))
    C.provenance(RES, [C.p(cfg["paths"]["external_sections"])], [out / "section_index.tsv", out / "stain_target.tsv"],
                 cfg["seed"], extra={"histology_lowres": H})


if __name__ == "__main__":
    main()
