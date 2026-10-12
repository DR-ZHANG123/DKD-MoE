"""04_interpret: DKD-state direction in the shared space, image-side scores, morphology correlates.

Per primary model (fold f, seed s): direction = mean PT-rich expression embedding of DKD minus HKD
training participants (KPMP folds != f). Each spot gets an image-side score (its H&E embedding
projected on the direction) and an expression-side score (its measured expression through the
expression encoder), standardised on training PT-rich spots of that model. KPMP spots are scored
only by models for which they were held out (cross-fitting; averaged over seeds); GSE211785 spots
by all models (averaged). Image-side scores are used to localise and describe tissue morphology; no
image-inferred expression is used to test disease differences in expression.
Outputs results/align_interpret/*.tsv and figures/source_data/align_*.tsv(.gz).
"""
from __future__ import annotations
import importlib, os, sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import interp_lib as L  # noqa: E402
from stats import compare  # noqa: E402

RES = "align_interpret"
CONTRASTS = [("DKD", "HKD"), ("DKD", "HKD_withDM"), ("DKD", "DM_noCKD"), ("DKD", "Reference"), ("HKD_withDM", "HKD")]


def score_spots(cfg, model_name):
    ck = C.p(cfg["paths"]["ckpt"])
    meta = pd.read_csv(ck / "spot_meta.tsv.gz", sep="\t", dtype={"participant": str})
    folds = pd.read_csv(C.rpath("align_train") / "spot_folds.tsv.gz", sep="\t").fold.values
    a, b = cfg["data"]["primary_contrast"]; mins = cfg["interpret"]["min_pt_rich_spots"]
    img = np.zeros(len(meta)); exp = np.zeros(len(meta)); cnt = np.zeros(len(meta))
    per_seed, dirs = [], []
    ext = (meta.cohort != "kpmp").values
    for f in sorted(np.unique(folds[~np.isnan(folds)]).astype(int)):
        for s in cfg["seeds"]:
            fn = ck / f"{model_name}_fold{f}_seed{s}_emb.npz"
            if not fn.exists():
                continue
            z = np.load(fn); zi, ze, train = z["zi"].astype(np.float32), z["ze"].astype(np.float32), z["train"]
            d, na, nb = L.direction(ze, meta, train, a, b, mins)
            ref = train & meta.pt_rich.values
            si, se = L.standardise(zi @ d, ref), L.standardise(ze @ d, ref)
            if cfg["interpret"].get("external_calibration") == "cohort":
                c = np.load(str(fn).replace("_emb.npz", "_extcal.npz"))
                ci, ce = c["zi"].astype(np.float32) @ d, c["ze"].astype(np.float32) @ d
                rr = c["rows"]; cc = meta.cohort.values[rr]; pr = meta.pt_rich.values[rr]
                for co in set(cc):     # standardised within each external cohort (PT-rich spots)
                    k = cc == co
                    si[rr[k]] = (ci[k] - ci[k & pr].mean()) / ci[k & pr].std()
                    se[rr[k]] = (ce[k] - ce[k & pr].mean()) / ce[k & pr].std()
            use = (folds == f) | ext
            img[use] += si[use]; exp[use] += se[use]; cnt[use] += 1
            per_seed.append(meta.loc[folds == f, ["participant", "pt_rich"]].assign(seed=s, fold=f, image_score=si[folds == f]))
            per_seed.append(meta.loc[ext, ["participant", "pt_rich"]].assign(seed=s, fold=f, image_score=si[ext]))
            dirs.append({"fold": f, "seed": s, "n_dkd_train": na, "n_hkd_train": nb,
                         "cos_img_expr_gap": float(np.mean(np.sum(zi[ref] * ze[ref], 1)))})
    meta["image_score"] = img / np.maximum(cnt, 1); meta["expr_score"] = exp / np.maximum(cnt, 1)
    meta["n_models"] = cnt
    return meta[cnt > 0].copy(), pd.concat(per_seed), pd.DataFrame(dirs)


def participant_table(S: pd.DataFrame, feats: list[str], mins: int) -> pd.DataFrame:
    pr = S[S.pt_rich]
    P = pr.groupby("participant").agg(cohort=("cohort", "first"), group=("group", "first"),
                                      n_pt_rich=("spot_id", "size"), image_score=("image_score", "mean"),
                                      expr_score=("expr_score", "mean"), **{f: (f, "mean") for f in feats})
    P["image_score_all_spots"] = S.groupby("participant").image_score.mean().reindex(P.index)
    P["n_sections"] = S.groupby("participant").section.nunique().reindex(P.index)
    return P[P.n_pt_rich >= mins]


def contrasts(P: pd.DataFrame, cols: list[str], cohort: str) -> pd.DataFrame:
    rows = []
    for a, b in CONTRASTS:
        for c in cols:
            r = compare(P[c], P.group, a, b)
            rows.append({"cohort": cohort, "contrast": f"{a}-{b}", "metric": c, **r})
    return pd.DataFrame(rows)


def main():
    global RES
    cfg = C.load_cfg(); C.set_seed(cfg["seed"]); I = cfg["interpret"]
    if os.environ.get("ALIGN_EXTCAL") == "none":   # external spots on the KPMP-training scale (cross-cohort placement)
        I["external_calibration"] = None; RES = "align_interpret_uncalibrated"
        _src = C.src; C.src = lambda f: _src(f.replace("align_", "align_uncal_", 1))
    out = C.result_dir(RES)
    model_name = f"aligned_{cfg['train']['primary_batch_mode']}"
    S, per_seed, dirs = score_spots(cfg, model_name)
    dirs.to_csv(out / "direction_models.tsv", sep="\t", index=False)
    # morphology features and measured-expression programme scores per spot
    sec = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    feats = I["morph_features"]
    F = []
    for r in sec.itertuples():
        z = np.load(r.cache)
        F.append(pd.DataFrame(z["feats"], columns=list(z["feat_names"])).assign(section=r.section, spot_id=z["spot_id"]))
    S = S.merge(pd.concat(F), on=["section", "spot_id"], how="left")
    S = S.join(L.measured_programme_scores(), on=["section", "spot_id"])
    keep = ["cohort", "section", "participant", "group", "spot_id", "px_row", "px_col", "um_per_px", "pt_rich", "pt_score",
            "image_score", "expr_score", "n_models", "DKDiPT_up", "SharedInjury_up", "injury_index", *feats]
    S[keep].to_csv(C.src("align_spot_scores.tsv.gz"), sep="\t", index=False)

    # participant level (unit of inference)
    mins = I["min_pt_rich_spots"]
    P = participant_table(S, feats, mins)
    P.to_csv(out / "participant_scores.tsv", sep="\t"); P.to_csv(C.src("align_participant_scores.tsv"), sep="\t")
    T = pd.concat([contrasts(P[P.cohort == "kpmp"], ["image_score", "image_score_all_spots", "expr_score", *feats], "kpmp_out_of_fold"),
                   contrasts(P[P.cohort == "gse211785"], ["image_score", "expr_score"], "gse211785_external")])
    # cohorts without a within-cohort contrast (e.g. GSE339455, hypertensive nephropathy only): distribution
    # of participant scores by cohort and group, on each cohort's own scale
    P.groupby(["cohort", "group"])[["image_score", "expr_score"]].describe().to_csv(out / "participant_score_distribution.tsv", sep="\t")
    T.to_csv(out / "participant_contrasts.tsv", sep="\t", index=False)
    # seed stability of the primary contrast (out-of-fold KPMP, image side)
    ps = per_seed[per_seed.pt_rich].groupby(["seed", "participant"]).image_score.mean().reset_index()
    grp = P.group
    stab = []
    for s, g in ps.groupby("seed"):
        g = g.set_index("participant").image_score
        g = g[g.index.isin(P.index[P.cohort == "kpmp"])]
        stab.append({"seed": s, **compare(g, grp.reindex(g.index), *cfg["data"]["primary_contrast"])})
    pd.DataFrame(stab).to_csv(out / "primary_contrast_by_seed.tsv", sep="\t", index=False)

    # spot-level correspondence within sections (PT-rich spots), summarised by participant
    ys = ["expr_score", "pt_score", "DKDiPT_up", "SharedInjury_up", "injury_index", *feats]
    W = L.within_section_corr(S, "image_score", ys, "pt_rich", mins)
    W.to_csv(out / "within_section_correlations.tsv", sep="\t", index=False)
    Wp = W.groupby(["participant", "cohort", "group"])[ys].mean().reset_index()
    Wp.to_csv(C.src("align_within_section_correlations.tsv"), sep="\t", index=False)
    summ = [{"cohort": c, "variable": y, **L.one_sample(Wp[Wp.cohort == c][y])} for c in sorted(P.cohort.unique()) for y in ys]
    pd.DataFrame(summ).to_csv(out / "within_section_correlation_summary.tsv", sep="\t", index=False)
    # participant-level morphology correlates of the image-side score
    mc = []
    for c in sorted(P.cohort.unique()):
        Pc = P[P.cohort == c]
        for f in feats + ["expr_score"]:
            mc.append({"cohort": c, "feature": f, "n": len(Pc), "spearman": L.spearman(Pc.image_score.values, Pc[f].values)})
    pd.DataFrame(mc).to_csv(out / "participant_morphology_correlations.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(dirs.describe().round(3))
    print(T[T.metric.isin(["image_score", "expr_score", "image_score_all_spots"])].round(4).to_string())
    print(pd.DataFrame(stab).round(3).to_string())
    print(pd.DataFrame(summ).round(4).to_string())
    print(pd.DataFrame(mc).round(3).to_string())
    C.provenance(RES, [C.rpath("align_train") / "spot_folds.tsv.gz", C.rpath("align_prep") / "sections.tsv"],
                 sorted(out.glob("*.tsv")) + [C.src(x) for x in ["align_spot_scores.tsv.gz", "align_participant_scores.tsv", "align_within_section_correlations.tsv"]], cfg["seed"], extra={"model": model_name})


if __name__ == "__main__":
    main()
