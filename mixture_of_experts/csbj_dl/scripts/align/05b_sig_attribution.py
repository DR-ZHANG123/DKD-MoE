"""05b_sig_attribution: pixel attribution along the transfer-signature image direction.

Direction per model = ridge regression (07_signature settings) of the final transfer-signature spot score on
the expression embedding over PT-rich spots of KPMP training-part participants (seed = first configured seed).
Image-side score c = u . w, standardised on training PT-rich spots. Patches: PT-rich spots of held-out KPMP
participants, scored by the model for which their participant was held out; the n highest and n lowest,
at most one patch per participant. Attribution = integrated gradients through the full fixed encoder and
image head from a section-background baseline (as 05_attribution); re-crop checked against the stored embedding.
Outputs results/align_sig_attrib/{patches.tsv, attributions.npz, directions.npz, patches/*.png, patch_grid.png};
figures/source_data/align_sig_attrib_patches.tsv
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import sig_lib as G  # noqa: E402
from model import AlignModel  # noqa: E402

os.environ.setdefault("HF_HOME", str(C.ROOT / "data/hf_cache"))
A05 = importlib.import_module("05_attribution"); S07 = importlib.import_module("07_signature")
RES = "align_sig_attrib"
N_EXTREME = 6


def directions(cfg, meta, name, seed):
    """fold -> (w, mean, sd) of the standardised image-side score; also every spot's score for its held-out fold."""
    ck = C.p(cfg["paths"]["ckpt"]); folds = pd.read_csv(C.rpath("align_train") / "spot_folds.tsv.gz", sep="\t").fold.values
    out, score = {}, np.full(len(meta), np.nan)
    for f in range(5):
        z = np.load(ck / f"{name}_fold{f}_seed{seed}_emb.npz")
        zi, ze = z["zi"].astype(np.float32), z["ze"].astype(np.float32)
        ref = z["train"] & meta.pt_rich.values & (meta.cohort == "kpmp").values
        w, _, _ = G.ridge_direction(ze[ref], meta.sig_final.values[ref], S07.ALPHAS)
        c = zi @ w; mu, sd = c[ref].mean(), c[ref].std()
        out[f] = (w, mu, sd); score[folds == f] = ((c - mu) / sd)[folds == f]
    return out, score


def main():
    cfg = C.load_cfg(); rc = C.run_cfg(); C.set_seed(cfg["seed"]); I = cfg["interpret"]; h = rc["histology"]
    dev = torch.device(cfg["device"]); out = C.result_dir(RES); (out / "patches").mkdir(exist_ok=True)
    ck = C.p(cfg["paths"]["ckpt"]); name = f"aligned_{cfg['train']['primary_batch_mode']}"; seed = cfg["seeds"][0]
    meta = pd.read_csv(ck / "spot_meta.tsv.gz", sep="\t", dtype={"participant": str})
    meta = meta.merge(pd.read_csv(C.p(cfg["paths"]["cache"]) / "signature_spot_scores.tsv.gz", sep="\t")[["section", "spot_id", "sig_final"]],
                      on=["section", "spot_id"], how="left")
    D, meta["image_sig_final"] = directions(cfg, meta, name, seed)
    np.savez_compressed(out / "directions.npz", **{f"w_fold{f}": v[0] for f, v in D.items()},
                        **{f"mu_sd_fold{f}": np.array(v[1:]) for f, v in D.items()})
    sec = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str}).set_index("section")
    feats = I["morph_features"]; Fs = []
    for r in sec.reset_index().itertuples():
        z = np.load(r.cache)
        Fs.append(pd.DataFrame(z["feats"], columns=list(z["feat_names"])).assign(section=r.section, spot_id=z["spot_id"]))
    cand = meta[(meta.cohort == "kpmp") & meta.pt_rich & meta.image_sig_final.notna()].merge(pd.concat(Fs), on=["section", "spot_id"])
    hi = cand.sort_values("image_sig_final", ascending=False).groupby("participant").head(1).head(N_EXTREME)
    lo = cand[~cand.participant.isin(hi.participant)].sort_values("image_sig_final").groupby("participant").head(1).head(N_EXTREME)
    X = pd.concat([hi.assign(extreme="high"), lo.assign(extreme="low")]).reset_index(drop=True)   # one patch per participant
    s11 = importlib.import_module("11a_histology_embed"); histo = importlib.import_module("histo")
    from transformers import AutoModel
    vit = AutoModel.from_pretrained(h["encoder"], revision=h["encoder_revision"]).eval().to(dev)
    for q in vit.parameters():
        q.requires_grad_(False)
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev).view(1, 3, 1, 1)
    bgs = pd.read_csv(C.ROOT / "results/11a_histology_embed/section_index.tsv", sep="\t").set_index("section").bg_rgb
    folds = C.splits().fold
    rows, attrs, models = [], [], {}
    import cv2
    for r in X.itertuples():
        f = int(folds[r.participant])
        if f not in models:
            st = torch.load(ck / f"{name}_fold{f}_seed{seed}.pt", map_location=dev)
            mc = dict(cfg["model"]); mc.update(st["cfg_model"])
            m = AlignModel(st["n_genes"], mc, np.zeros(mc["emb_dim"], np.float32), np.ones(mc["emb_dim"], np.float32),
                           np.zeros(st["n_genes"], np.float32), np.ones(st["n_genes"], np.float32)).to(dev)
            m.load_state_dict(st["state"]); m.eval()
            models[f] = (m, torch.tensor(D[f][0], device=dev))
        m, w = models[f]
        p = A05.load_patch(r, sec.loc[r.section], s11, histo, h["patch_px"])
        x = torch.from_numpy(p).to(dev).permute(2, 0, 1)[None].float() / 255.0
        bg = np.array([float(v) for v in str(bgs.get(r.section, "0.95,0.95,0.95")).split(",")])
        base = torch.tensor(bg, dtype=torch.float32, device=dev).view(1, 3, 1, 1).expand_as(x)

        def score(xx):
            cls = vit(pixel_values=(xx - mean) / std).last_hidden_state[:, 0]
            return (m.encode_img(cls) @ w - D[f][1]) / D[f][2], cls

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
        att = ((x - base) * grads).sum(1)[0].detach().cpu().numpy(); attrs.append(att)
        fn = f"patches/{r.extreme}_{len(rows):02d}_{r.participant}.png"
        cv2.imwrite(str(out / fn), cv2.cvtColor(p, cv2.COLOR_RGB2BGR))
        rows.append({"idx": len(rows), "extreme": r.extreme, "participant": r.participant, "group": r.group,
                     "section": r.section, "spot_id": r.spot_id, "fold": f, "score": float(s0.item()),
                     "measured_sig_final": r.sig_final, "attr_sum": float(att.sum()), "cos_to_stored_embedding": cos,
                     "patch_png": fn, **{k: getattr(r, k) for k in feats}})
        print(rows[-1], flush=True)
    T = pd.DataFrame(rows)
    T.to_csv(out / "patches.tsv", sep="\t", index=False)
    T.to_csv(C.src("align_sig_attrib_patches.tsv"), sep="\t", index=False)
    np.savez_compressed(out / "attributions.npz", attr=np.stack(attrs).astype(np.float32), idx=T.idx.values)
    T2 = T.rename(columns={"score": "image_score_crossfit"})
    A05.grid(out, T2, np.stack(attrs))
    C.provenance(RES, [C.p(cfg["paths"]["cache"]) / "signature_spot_scores.tsv.gz", C.rpath("align_train") / "spot_folds.tsv.gz"],
                 [out / "patches.tsv", out / "attributions.npz", out / "directions.npz", out / "patch_grid.png"], cfg["seed"],
                 extra={"model": name, "seed_model": seed, "ig_steps": I["n_attr_steps"], "target": "final transfer signature",
                        "baseline": "section background colour", "n_extreme": N_EXTREME})


if __name__ == "__main__":
    main()
