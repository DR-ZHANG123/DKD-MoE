"""06_heatmaps: image-side DKD-state score over the H&E of example sections.

Examples: the KPMP DKD and HKD participants with the most PT-rich spots (cross-fitted scores) and
the GSE211785 DKD and HKD sections with the most spots. Writes a down-sampled H&E thumbnail per
section (plotting input, with its scale) and a quick-look overlay; final figures are drawn later
from figures/source_data/align_spot_scores.tsv.gz + align_heatmap_sections.tsv + thumbnails.
"""
from __future__ import annotations
import importlib, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

RES = "align_heatmaps"


def thumbnail(img, max_px=1600):
    step = max(1, int(np.ceil(max(img.shape[:2]) / max_px)))
    t = np.asarray(img[::step, ::step])
    t = t[..., :3] if t.ndim == 3 else np.repeat(t[..., None], 3, -1)
    return t, step


def main():
    cfg = C.load_cfg(); C.set_seed(cfg["seed"])
    out = C.result_dir(RES)
    S = pd.read_csv(C.src("align_spot_scores.tsv.gz"), sep="\t", dtype={"participant": str})
    sec = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str}).set_index("section")
    n = S[S.pt_rich].groupby(["cohort", "group", "section"]).size().rename("n_pt_rich").reset_index()
    pick = []
    for coh, grp in [("kpmp", "DKD"), ("kpmp", "HKD"), ("gse211785", "DKD"), ("gse211785", "HKD"),
                     ("gse339455", "HKD")]:
        d = n[(n.cohort == coh) & (n.group == grp)].sort_values("n_pt_rich", ascending=False)
        if len(d):
            pick.append(d.iloc[0])
    P = pd.DataFrame(pick)
    s11 = importlib.import_module("11a_histology_embed")
    import cv2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows = []
    fig, ax = plt.subplots(1, len(P), figsize=(4 * len(P), 4.4))
    v = np.nanpercentile(np.abs(S.image_score), 98)
    for i, r in enumerate(P.itertuples()):
        tif = C.tif_path(r.cohort, r.section, sec.loc[r.section, "tif"])
        img = cv2.cvtColor(cv2.imread(str(tif)), cv2.COLOR_BGR2RGB) if str(tif).endswith(".png") else s11.load_image(tif)
        th, step = thumbnail(img)
        fn = f"thumb_{r.cohort}__{r.section}.png"
        cv2.imwrite(str(out / fn), cv2.cvtColor(th, cv2.COLOR_RGB2BGR))
        d = S[S.section == r.section]
        rows.append({"cohort": r.cohort, "group": r.group, "section": r.section, "participant": d.participant.iloc[0],
                     "n_spots": len(d), "n_pt_rich": r.n_pt_rich, "thumbnail": fn, "downsample": step,
                     "mean_image_score_pt_rich": d[d.pt_rich].image_score.mean()})
        ax[i].imshow(th)
        sc = ax[i].scatter(d.px_col / step, d.px_row / step, c=d.image_score, cmap="RdBu_r", vmin=-v, vmax=v,
                           s=4, alpha=0.8, linewidths=0)
        ax[i].set_title(f"{r.cohort} {r.group}", fontsize=8); ax[i].axis("off")
    fig.colorbar(sc, ax=ax, shrink=0.6, label="image-side DKD-state score (z)")
    fig.savefig(out / "heatmaps_quicklook.png", dpi=150); plt.close(fig)
    T = pd.DataFrame(rows)
    T.to_csv(out / "heatmap_sections.tsv", sep="\t", index=False)
    T.to_csv(C.src("align_heatmap_sections.tsv"), sep="\t", index=False)
    print(T.to_string())
    C.provenance(RES, [C.src("align_spot_scores.tsv.gz")],
                 [out / "heatmap_sections.tsv", *sorted(out.glob("*.png"))], cfg["seed"])


if __name__ == "__main__":
    main()
