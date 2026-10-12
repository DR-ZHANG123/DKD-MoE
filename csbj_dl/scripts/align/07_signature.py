"""07_signature: the expression model's transfer signature in tissue, measured and image-side.

1. Measured spot score of the signature (sig_lib), PT-rich spots, gene coverage per platform.
2. Per primary model (fold f, seed s): ridge regression of the measured score on the expression
   embedding v_s over PT-rich spots of KPMP training-part participants -> w; image-side score
   c_s = u_s . w. For KPMP fold-f participants the fold-f signature (built without fold f) is used for
   both the ridge target and the measured score; external cohorts use the final signature with their
   cohort-calibrated embeddings (03c). Image-side scores are standardised on training PT-rich spots
   (external: within cohort) and averaged over seeds / models.
3. Within-section Spearman (image-side vs measured, PT-rich spots) summarised per participant (sign test,
   Wilcoxon); participant-level DKD vs HKD (KPMP out-of-part) for measured and image-side scores.
4. Morphology correlates of the image-side score. 5. Heatmap pair for one held-out KPMP DKD section.
Run per track (ALIGN_TRACK unset = full resolution: KPMP + GSE211785; lowres: + GSE339455).
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import interp_lib as L  # noqa: E402
import sig_lib as G  # noqa: E402
from stats import compare  # noqa: E402

RES = "align_sig"
ALPHAS = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]


def measured(cfg, out):
    f = C.p(cfg["paths"]["cache"]) / "signature_spot_scores.tsv.gz"
    sec = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    if not f.exists():
        reg = C.registry()
        S, cov = G.score_sections(sec, reg, G.signatures())
        S.to_csv(f, sep="\t", index=False); cov.to_csv(out / "gene_coverage_sections.tsv", sep="\t", index=False)
    cov = pd.read_csv(out / "gene_coverage_sections.tsv", sep="\t")
    cov.groupby(["cohort", "signature"])[["n_up", "n_down", "up_on_platform", "down_on_platform", "up_measured",
                                         "down_measured"]].median().to_csv(out / "gene_coverage.tsv", sep="\t")
    return pd.read_csv(f, sep="\t")


def model_scores(cfg, meta, sig):
    ck = C.p(cfg["paths"]["ckpt"]); name = f"aligned_{cfg['train']['primary_batch_mode']}"
    folds = pd.read_csv(C.rpath("align_train") / "spot_folds.tsv.gz", sep="\t").fold.values
    ext = (meta.cohort != "kpmp").values; pr = meta.pt_rich.values
    img = np.zeros(len(meta)); cnt = np.zeros(len(meta)); rows = []
    for f in range(5):
        for s in cfg["seeds"]:
            z = np.load(ck / f"{name}_fold{f}_seed{s}_emb.npz")
            zi, ze, train = z["zi"].astype(np.float32), z["ze"].astype(np.float32), z["train"]
            ref = train & pr & (meta.cohort == "kpmp").values
            out = {}
            for tgt in (f"sig_fold{f}", "sig_final"):
                w, b, a = G.ridge_direction(ze[ref], sig[tgt].values[ref], ALPHAS)
                fit_r = L.spearman(ze[ref] @ w, sig[tgt].values[ref])
                out[tgt] = w
                rows.append({"fold": f, "seed": s, "target": tgt, "alpha": a, "train_spearman_v_w": fit_r})
            c = zi @ out[f"sig_fold{f}"]; c = (c - c[ref].mean()) / c[ref].std()
            ho = folds == f
            img[ho] += c[ho]; cnt[ho] += 1
            e = np.load(ck / f"{name}_fold{f}_seed{s}_extcal.npz")
            ce = e["zi"].astype(np.float32) @ out["sig_final"]; rr = e["rows"]; co = meta.cohort.values[rr]
            for k in set(co):
                m = co == k; p_ = pr[rr] & m
                img[rr[m]] += (ce[m] - ce[p_].mean()) / ce[p_].std(); cnt[rr[m]] += 1
    meta = meta.copy(); meta["image_sig"] = img / np.maximum(cnt, 1); meta["n_models"] = cnt
    return meta[cnt > 0], pd.DataFrame(rows)


def main():
    cfg = C.load_cfg(); C.set_seed(cfg["seed"]); I = cfg["interpret"]; out = C.result_dir(RES)
    meta = pd.read_csv(C.p(cfg["paths"]["ckpt"]) / "spot_meta.tsv.gz", sep="\t", dtype={"participant": str})
    M = measured(cfg, out)
    meta = meta.merge(M, on=["section", "spot_id"], how="left")
    folds = pd.read_csv(C.rpath("align_train") / "spot_folds.tsv.gz", sep="\t").fold.values
    # measured score used for evaluation: fold-f signature for KPMP fold-f spots, final signature elsewhere
    ms = meta.sig_final.values.copy()
    for f in range(5):
        ms[folds == f] = meta[f"sig_fold{f}"].values[folds == f]
    meta["measured_sig"] = ms
    S, R = model_scores(cfg, meta, meta)
    R.to_csv(out / "ridge_models.tsv", sep="\t", index=False)
    sp = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    feats = I["morph_features"]; F = []
    for r in sp.itertuples():
        z = np.load(r.cache)
        F.append(pd.DataFrame(z["feats"], columns=list(z["feat_names"])).assign(section=r.section, spot_id=z["spot_id"]))
    S = S.merge(pd.concat(F), on=["section", "spot_id"], how="left")
    keep = ["cohort", "section", "participant", "group", "spot_id", "px_row", "px_col", "um_per_px", "pt_rich", "pt_score",
            "measured_sig", "sig_final", "image_sig", "n_models", *feats]
    S[keep].to_csv(C.src("align_sig_spot_scores.tsv.gz"), sep="\t", index=False)
    mins = I["min_pt_rich_spots"]
    # within-section correspondence + morphology
    W = L.within_section_corr(S, "image_sig", ["measured_sig", "pt_score", *feats], "pt_rich", mins)
    W.to_csv(out / "within_section_correlations.tsv", sep="\t", index=False)
    Wp = W.groupby(["participant", "cohort", "group"])[["measured_sig", "pt_score", *feats]].mean().reset_index()
    Wp.to_csv(C.src("align_sig_within_section_correlations.tsv"), sep="\t", index=False)
    summ = []
    for c in sorted(Wp.cohort.unique()):
        for y in ["measured_sig", "pt_score", *feats]:
            x = Wp[Wp.cohort == c][y].dropna()
            r = L.one_sample(x)
            r["sign_test_p"] = stats.binomtest(int((x > 0).sum()), len(x), 0.5).pvalue if len(x) else np.nan
            summ.append({"cohort": c, "variable": y, **r})
    SU = pd.DataFrame(summ); SU.to_csv(out / "within_section_summary.tsv", sep="\t", index=False)
    # participant level
    pr = S[S.pt_rich]
    P = pr.groupby("participant").agg(cohort=("cohort", "first"), group=("group", "first"), n_pt_rich=("spot_id", "size"),
                                      measured_sig=("measured_sig", "mean"), image_sig=("image_sig", "mean"),
                                      **{f: (f, "mean") for f in feats})
    P = P[P.n_pt_rich >= mins]
    P.to_csv(out / "participant_scores.tsv", sep="\t"); P.to_csv(C.src("align_sig_participant_scores.tsv"), sep="\t")
    T = []
    for coh in sorted(P.cohort.unique()):
        Pc = P[P.cohort == coh]
        for a, b in [("DKD", "HKD"), ("DKD", "HKD_withDM"), ("DKD", "Reference"), ("DKD", "DM_noCKD")]:
            for m in ["measured_sig", "image_sig"]:
                T.append({"cohort": coh, "contrast": f"{a}-{b}", "metric": m, **compare(Pc[m], Pc.group, a, b)})
        for f_ in feats + ["measured_sig"]:
            T.append({"cohort": coh, "contrast": "participant_spearman_with_image_sig", "metric": f_, "n_a": len(Pc),
                      "spearman": L.spearman(Pc.image_sig.values, Pc[f_].values)})
    T = pd.DataFrame(T); T.to_csv(out / "participant_tests.tsv", sep="\t", index=False)
    heatmap_pair(S, sp, out)
    pd.set_option("display.width", 250)
    print(pd.read_csv(out / "gene_coverage.tsv", sep="\t").to_string())
    print(R.groupby("target")[["alpha", "train_spearman_v_w"]].describe().round(3).to_string())
    print(SU.round(4).to_string()); print(T.round(4).to_string())
    C.provenance(RES, [C.CSBJ / "results/expr_transfer/transfer_signature.tsv", C.CSBJ / "results/expr_transfer/fold_signatures.tsv"],
                 sorted(out.glob("*.tsv")) + [C.src("align_sig_spot_scores.tsv.gz")], cfg["seed"], extra={"ridge_alphas": ALPHAS})


def heatmap_pair(S, sp, out):
    """Measured vs image-side signature score over the H&E of the held-out KPMP DKD section with most PT-rich spots."""
    import importlib, cv2
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d0 = S[(S.cohort == "kpmp") & (S.group == "DKD")]
    sec = d0[d0.pt_rich].groupby("section").size().idxmax()
    d = S[S.section == sec]
    r = sp.set_index("section").loc[sec]
    tif = C.tif_path("kpmp", sec, r.tif)
    img = cv2.cvtColor(cv2.imread(str(tif)), cv2.COLOR_BGR2RGB) if str(tif).endswith(".png") else \
        importlib.import_module("11a_histology_embed").load_image(tif)
    step = max(1, int(np.ceil(max(img.shape[:2]) / 1600))); th = np.asarray(img[::step, ::step])[..., :3]
    cv2.imwrite(str(out / f"thumb_{sec}.png"), cv2.cvtColor(th, cv2.COLOR_RGB2BGR))
    fig, ax = plt.subplots(1, 2, figsize=(9, 4.6))
    for a, col, lab in [(ax[0], "measured_sig", "measured signature score (z)"), (ax[1], "image_sig", "image-side signature score (z)")]:
        v = d[col] - d[col].mean(); v = v / v.std(); lim = np.nanpercentile(np.abs(v), 98)
        a.imshow(th); sc = a.scatter(d.px_col / step, d.px_row / step, c=v, cmap="RdBu_r", vmin=-lim, vmax=lim, s=5, linewidths=0)
        a.set_title(lab, fontsize=8); a.axis("off"); fig.colorbar(sc, ax=a, shrink=0.6)
    rho = L.spearman(d.measured_sig.values, d.image_sig.values)
    fig.suptitle(f"{sec} (held-out DKD), spot Spearman {rho:.2f}", fontsize=8)
    fig.savefig(out / "heatmap_pair_quicklook.png", dpi=150); plt.close(fig)
    pd.DataFrame([{"section": sec, "participant": d.participant.iloc[0], "thumbnail": f"thumb_{sec}.png", "downsample": step,
                   "n_spots": len(d), "spearman_all_spots": rho}]).to_csv(C.src("align_sig_heatmap_section.tsv"), sep="\t", index=False)


if __name__ == "__main__":
    main()
