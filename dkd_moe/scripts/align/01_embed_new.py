"""01_embed_new: embed newly downloaded KPMP Visium sections with the stage-11a procedure
(same fixed encoder and revision, same patch geometry, tissue filter and morphology features),
reusing scripts/stages/11a_histology_embed.process read-only; outputs go to paths.emb_new.

Sections come from dkd_moe/results/data_visium/section_table.tsv (data module). Idempotent:
sections already embedded (here or by stage 11a) are skipped, so rerun as more sections arrive;
then rerun 02_prepare.py and the downstream steps.
"""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

os.environ.setdefault("HF_HOME", str(C.ROOT / "data/hf_cache"))
RES = "align_embed_new"


def build_sections(done: set) -> list[dict]:
    h = importlib.import_module("histo")
    n = C.new_section_table()
    out = []
    for r in n.to_dict("records"):
        sec = f"{r['participant']}__{r['sample']}"
        if sec in done or not r.get("tif") or not r.get("positions") or not Path(r["tif"]).exists():
            continue
        sf = json.loads(Path(r["scalefactors"]).read_text())
        tp = Path(r["positions"])
        d = pd.read_csv(tp, header=0 if tp.name == "tissue_positions.csv" else None)
        d.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
        off = d[d.in_tissue == 0][["pxl_row", "pxl_col"]].sample(frac=1, random_state=0).head(300)
        d = d[d.in_tissue == 1].copy(); d["spot_id"] = d.barcode
        out.append({"cohort": "kpmp", "section": sec, "participant": r["participant"], "tif": Path(r["tif"]),
                    "um_per_px": h.SPOT_UM / float(sf["spot_diameter_fullres"]), "spots": d, "orient": "rowcol",
                    "off_pts": list(zip(off.pxl_row, off.pxl_col))})
    return out


def main():
    cfg = C.load_cfg(); rc = C.run_cfg(); C.set_seed(cfg["seed"])
    if not C.p(cfg["paths"]["new_sections"]).exists():
        raise SystemExit("no new-section table yet")
    emb = C.p(cfg["paths"]["emb_new"]); emb.mkdir(parents=True, exist_ok=True)
    s11 = importlib.import_module("11a_histology_embed")
    s11.EMB = emb                       # write into the dkd_moe embedding folder, not stage 11a's
    old = pd.read_csv(C.ROOT / "results/11a_histology_embed/section_index.tsv", sep="\t")
    idx_f = emb / "section_index.tsv"
    prev = pd.read_csv(idx_f, sep="\t") if idx_f.exists() else pd.DataFrame(columns=["section"])
    secs = build_sections(set(old.section) | set(prev.section))
    print(len(secs), "sections to embed", flush=True)
    if not secs:
        return
    h = rc["histology"]
    enc = importlib.import_module("histo").Encoder(h["encoder"], h["encoder_revision"], device=cfg["device"])
    rows = []
    small = lambda x: enc(x, bs=cfg.get("embed_batch", 16))   # shared GPU: small inference batches
    for s in secs:
        try:
            r = s11.process(s, small, h)
        except Exception as e:          # unreadable image: reported, section not used
            r = {"cohort": "kpmp", "section": s["section"], "participant": s["participant"], "status": f"error:{e}"}
        print(r, flush=True); rows.append(r)
        idx = pd.concat([prev, pd.DataFrame(rows)]).drop_duplicates("section", keep="last")
        idx.to_csv(idx_f, sep="\t", index=False)
    out = C.result_dir(RES)
    idx.to_csv(out / "section_index.tsv", sep="\t", index=False)
    C.provenance(RES, [C.p(cfg["paths"]["new_sections"])], [out / "section_index.tsv"], cfg["seed"],
                 extra={"encoder": h["encoder"], "revision": h["encoder_revision"]})


if __name__ == "__main__":
    main()
