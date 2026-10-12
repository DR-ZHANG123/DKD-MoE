"""05_attribution: pixel attribution of the image-side DKD-state score for example patches.

Patches: PT-rich spots of KPMP DKD and HKD participants, highest and lowest cross-fitted image-side
scores (at most 2 per participant), each explained with the model for which its participant was held
out (seed = first configured seed). Score(x) = <image head(phikon-v2 CLS(x)), DKD direction>; the
full ViT is run with gradients; attribution = integrated gradients from a background-coloured
(section glass) baseline (gradient x input when n_attr_steps = 1). Patch re-extraction is checked
against the stored stage-11a embedding (cosine similarity).
Outputs results/align_attrib/{patches.tsv, patch_grid.png, patches/*.png, attributions.npz};
figures/source_data/align_example_patches.tsv
"""
from __future__ import annotations
import importlib, json, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import interp_lib as L  # noqa: E402
from model import AlignModel  # noqa: E402

os.environ.setdefault("HF_HOME", str(C.ROOT / "data/hf_cache"))
RES = "align_attrib"


def pick(S: pd.DataFrame, n: int) -> pd.DataFrame:
    d = S[(S.cohort == "kpmp") & S.pt_rich & S.group.isin(["DKD", "HKD"])].copy()
    out = []
    for asc, lab in [(False, "high"), (True, "low")]:
        d2 = d.sort_values("image_score", ascending=asc).groupby("participant").head(2).head(n)
        out.append(d2.assign(extreme=lab))
    return pd.concat(out).reset_index(drop=True)


def load_patch(row, sec_row, s11, histo, patch_px):
    tif = C.tif_path(row.cohort, row.section, sec_row.tif)
    img = s11.load_image(tif)
    half = int(sec_row.field_px_native) // 2
    p = histo.crop(img, row.px_row, row.px_col, half)
    return histo.resize(p, patch_px)


def main():
    cfg = C.load_cfg(); rc = C.run_cfg(); C.set_seed(cfg["seed"]); I = cfg["interpret"]; h = rc["histology"]
    dev = torch.device(cfg["device"]); out = C.result_dir(RES); (out / "patches").mkdir(exist_ok=True)
    ck = C.p(cfg["paths"]["ckpt"])
    S = pd.read_csv(C.src("align_spot_scores.tsv.gz"), sep="\t", dtype={"participant": str})
    meta = pd.read_csv(ck / "spot_meta.tsv.gz", sep="\t", dtype={"participant": str})
    folds = C.splits().fold
    sec = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str}).set_index("section")
    s11 = importlib.import_module("11a_histology_embed"); histo = importlib.import_module("histo")
    from transformers import AutoModel
    vit = AutoModel.from_pretrained(h["encoder"], revision=h["encoder_revision"]).eval().to(dev)
    for q in vit.parameters():
        q.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    name = f"aligned_{cfg['train']['primary_batch_mode']}"; seed = cfg["seeds"][0]
    a, b = cfg["data"]["primary_contrast"]
    X = pick(S, I["n_example_patches"])
    rows, attrs, models = [], [], {}
    for r in X.itertuples():
        f = int(folds[r.participant])
        if f not in models:
            st = torch.load(ck / f"{name}_fold{f}_seed{seed}.pt", map_location=dev)
            mc = dict(cfg["model"]); mc.update(st["cfg_model"])
            m = AlignModel(st["n_genes"], mc, np.zeros(mc["emb_dim"], np.float32), np.ones(mc["emb_dim"], np.float32),
                           np.zeros(st["n_genes"], np.float32), np.ones(st["n_genes"], np.float32)).to(dev)
            m.load_state_dict(st["state"]); m.eval()
            z = np.load(ck / f"{name}_fold{f}_seed{seed}_emb.npz")
            d, _, _ = L.direction(z["ze"].astype(np.float32), meta, z["train"], a, b, I["min_pt_rich_spots"])
            models[f] = (m, torch.tensor(d, dtype=torch.float32, device=dev))
        m, d = models[f]
        srow = sec.loc[r.section]
        p = load_patch(r, srow, s11, histo, h["patch_px"])
        x = torch.from_numpy(p).to(dev).permute(2, 0, 1)[None].float() / 255.0
        bg = np.array([float(v) for v in str(pd.read_csv(C.ROOT / "results/11a_histology_embed/section_index.tsv", sep="\t")
                                             .set_index("section").bg_rgb.get(r.section, "0.95,0.95,0.95")).split(",")])
        base = torch.tensor(bg, dtype=torch.float32, device=dev).view(1, 3, 1, 1).expand_as(x)

        def score(xx):
            cls = vit(pixel_values=(xx - mean) / std).last_hidden_state[:, 0]
            return m.encode_img(cls) @ d, cls

        with torch.no_grad():
            s0, cls = score(x)
        e = np.load(sec.loc[r.section, "cache"])
        stored = torch.tensor(e["emb"][list(e["spot_id"]).index(r.spot_id)].astype(np.float32), device=dev)
        cos = F.cosine_similarity(cls[0], stored, dim=0).item()
        steps = I["n_attr_steps"]; grads = torch.zeros_like(x)
        for k in range(1, steps + 1):
            xx = (base + (x - base) * k / steps).requires_grad_(True)
            s_, _ = score(xx)
            g, = torch.autograd.grad(s_.sum(), xx); grads += g / steps
        att = ((x - base) * grads).sum(1)[0].detach().cpu().numpy()
        attrs.append(att)
        fn = f"patches/{r.extreme}_{len(rows):02d}_{r.participant}.png"
        import cv2
        cv2.imwrite(str(out / fn), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
        rows.append({"idx": len(rows), "extreme": r.extreme, "participant": r.participant, "group": r.group,
                     "section": r.section, "spot_id": r.spot_id, "fold": f, "image_score_crossfit": r.image_score,
                     "score_this_model": float(s0.item()), "attr_sum": float(att.sum()),
                     "cos_to_stored_embedding": cos, "patch_png": fn,
                     **{k: getattr(r, k) for k in I["morph_features"]}})
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(out / "patches.tsv", sep="\t", index=False)
    T.to_csv(C.src("align_example_patches.tsv"), sep="\t", index=False)
    np.savez_compressed(out / "attributions.npz", attr=np.stack(attrs).astype(np.float32), idx=T.idx.values)
    grid(out, T, np.stack(attrs))
    C.provenance(RES, [C.src("align_spot_scores.tsv.gz")],
                 [out / "patches.tsv", out / "attributions.npz", out / "patch_grid.png"], cfg["seed"],
                 extra={"model": name, "seed_model": seed, "ig_steps": I["n_attr_steps"], "baseline": "section background colour"})


def grid(out, T, A):
    """Quick-look grid (H&E above, signed attribution overlay below); final figures are drawn
    later from patches/*.png and attributions.npz."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.ndimage import gaussian_filter
    import cv2
    n = T.groupby("extreme").size().max()
    fig, ax = plt.subplots(4, n, figsize=(1.3 * n, 5.6))
    for i, lab in enumerate(["high", "low"]):
        t = T[T.extreme == lab].reset_index(drop=True)
        for j in range(n):
            for k in (0, 1):
                ax[2 * i + k, j].axis("off")
            if j >= len(t):
                continue
            im = cv2.cvtColor(cv2.imread(str(out / t.patch_png[j])), cv2.COLOR_BGR2RGB)
            a = gaussian_filter(A[t.idx[j]], 3); v = np.abs(a).max() + 1e-12
            ax[2 * i, j].imshow(im); ax[2 * i, j].set_title(f"{t.group[j]} {t.image_score_crossfit[j]:.1f}", fontsize=6)
            ax[2 * i + 1, j].imshow(im); ax[2 * i + 1, j].imshow(a, cmap="RdBu_r", vmin=-v, vmax=v, alpha=0.55)
    fig.tight_layout(); fig.savefig(out / "patch_grid.png", dpi=200); plt.close(fig)


if __name__ == "__main__":
    main()
