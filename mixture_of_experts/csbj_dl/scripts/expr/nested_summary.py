"""Nested cross-validation, summary: outer-fold participant scores of the nested pipeline (setting selected inside
each outer fold, final procedure at that setting) and of the reported pipeline (K=6, w_cond=10 in every fold),
scored identically (readout.py: condition-head log-odds averaged over injured-PT cells per participant x technology
with >= prep: min_cells_per_participant cells, mean over technologies, then mean over seeds).
AUC (DKD vs HKD) pooled over outer folds and per outer fold; DeLong CI for pooled AUCs and their paired difference;
participant bootstrap (stratified by outer fold x diagnosis) for pooled AUC, mean of per-fold AUCs and differences.
Outputs: results/expr_nested_cv/{cvw_reuse_check.tsv, outer_participant_scores.tsv, outer_fold_auc.tsv, summary.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from readout import part_means  # noqa: E402
from nested_outer import outer_name, reported_setting  # noqa: E402
from stats_utils import auc  # noqa: E402


def delong(y, scores):
    """DeLong (1988) AUCs and covariance for one or more score vectors on the same participants."""
    y = np.asarray(y, bool); S = np.atleast_2d(scores)
    m, n = y.sum(), (~y).sum()
    V10, V01, A = [], [], []
    for s in S:
        x, z = s[y], s[~y]
        psi = (x[:, None] > z[None, :]) + 0.5 * (x[:, None] == z[None, :])
        V10.append(psi.mean(1)); V01.append(psi.mean(0)); A.append(psi.mean())
    cov = np.cov(np.array(V10)) / m + np.cov(np.array(V01)) / n
    return np.array(A), np.atleast_2d(cov)


def fold_aucs(P, col):
    return {f: auc(q[col][q.group == "DKD"], q[col][q.group == "HKD"]) for f, q in P.groupby("fold")}


def seed_scores(cfg, obs, src: dict, mc):
    """src: outer fold -> (directory, name function(seed, fold)). Returns pooled participant table."""
    fold = obs.fold.values.astype(int); seeds = cfg["readout_stability"]["stability"]["cv_seeds"]
    vals = {f"seed{s}": np.full(len(obs), np.nan) for s in seeds}
    for o, (d, nm) in src.items():
        for s in seeds:
            z = np.load(d / f"{nm(s, o)}.npz")
            assert np.array_equal(np.sort(z["idx"]), np.where(fold == o)[0])
            vals[f"seed{s}"][z["idx"]] = z["logodds"]
    pm = part_means(obs, vals, mc)
    pm = pm[(pm.tech == "pooled") & pm.group.isin(["DKD", "HKD"])].copy()
    pm["score"] = pm[list(vals)].mean(1)
    return pm.set_index("participant")


def reuse_check(cfg, obs, ck, C, rep):
    """For outer folds whose selection equals the reported setting: the cvw models are the shared-start procedure
    on exactly the outer-training participants (held-out cells = outer fold; training-cell count = capped count of
    the other folds; all seeds present; 20 mixture epochs; single-expert start file present)."""
    st = cfg["readout_stability"]["stability"]; fold = obs.fold.values.astype(int); cap = cfg["train"]["cells_per_participant_cap"]
    rows = []
    for o in range(cfg["n_folds"]):
        n_exp = int(obs[fold != o].groupby("participant", observed=True).size().clip(upper=cap).sum())
        for s in st["cv_seeds"]:
            name = f"{st['prefix']}_K{rep[0]}_wc{rep[1]}_s{s}_f{o}"
            z = np.load(ck / st["prefix"] / f"{name}.npz")
            c = pd.read_csv(result_dir("expr_stability") / "curves" / f"{name}.tsv", sep="\t")
            cs = pd.read_csv(result_dir("expr_stability") / "curves" / f"single_f{o}.tsv", sep="\t")
            rows.append({"outer_fold": o, "seed": s, "selected_in_nested": bool(((C.outer_fold == o) & (C.K == rep[0]) & (C.w_cond == rep[1])).any()),
                         "heldout_cells_equal_outer_fold": bool(np.array_equal(np.sort(z["idx"]), np.where(fold == o)[0])),
                         "train_cells_per_epoch": int(c.n_cells.iloc[0]), "expected_train_cells": n_exp,
                         "single_start_train_cells": int(cs.n_cells.iloc[0]), "moe_epochs": int(c.epoch.max()),
                         "single_epochs": int(cs.epoch.max()), "single_start_file": (ck / st["prefix"] / f"single_f{o}.pt").exists()})
    R = pd.DataFrame(rows)
    R["pass"] = (R.heldout_cells_equal_outer_fold & (R.train_cells_per_epoch == R.expected_train_cells)
                 & (R.single_start_train_cells == R.expected_train_cells) & (R.moe_epochs == st["moe_epochs"])
                 & (R.single_epochs == st["warm_epochs"]) & R.single_start_file)
    return R


def bootstrap(P, B, rng):
    """Stratified (outer fold x diagnosis) participant bootstrap of pooled and mean-of-fold AUCs."""
    strata = [np.where((P.fold == f) & (P.group == g))[0] for f in sorted(P.fold.unique()) for g in ("DKD", "HKD")]
    y = (P.group == "DKD").values; fv = P.fold.values; a, b = P.nested.values, P.reported.values
    out = np.empty((B, 4))
    for r in range(B):
        ix = np.concatenate([rng.choice(s, len(s), replace=True) for s in strata])
        yy, ff = y[ix], fv[ix]
        res = []
        for v in (a[ix], b[ix]):
            pooled = auc(v[yy], v[~yy])
            mf = np.mean([auc(v[yy & (ff == f)], v[~yy & (ff == f)]) for f in np.unique(ff)])
            res += [pooled, mf]
        out[r] = res
    return pd.DataFrame(out, columns=["nested_pooled", "nested_meanfold", "reported_pooled", "reported_meanfold"])


def main():
    cfg = load_cfg("expr"); nc = cfg["nested_cv"]; out = result_dir("expr_nested_cv")
    ck = Path(cfg["paths"]["ckpt_dir"]); st = cfg["readout_stability"]["stability"]; mc = cfg["prep"]["min_cells_per_participant"]
    obs = ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs.copy()
    C = pd.read_csv(out / "selection.tsv", sep="\t"); rep = reported_setting()
    R = reuse_check(cfg, obs, ck, C, rep); R.to_csv(out / "cvw_reuse_check.tsv", sep="\t", index=False)
    assert R["pass"].all(), R
    cvw = (ck / st["prefix"], lambda s, o: f"{st['prefix']}_K{rep[0]}_wc{rep[1]}_s{s}_f{o}")
    nsrc = {}
    for _, r in C.iterrows():
        o, K, wc = int(r.outer_fold), int(r.K), float(r.w_cond)
        nsrc[o] = cvw if (K, wc) == rep else (ck / nc["outer_prefix"], lambda s, o, K=K, wc=wc: outer_name(K, wc, s, o))
    N = seed_scores(cfg, obs, nsrc, mc); F = seed_scores(cfg, obs, {o: cvw for o in range(cfg["n_folds"])}, mc)
    pfold = obs.groupby("participant", observed=True).fold.first().astype(int)
    P = pd.DataFrame({"group": F.group, "fold": pfold.reindex(F.index), "reported": F.score, "nested": N.score.reindex(F.index)})
    P = P.join(C.set_index("outer_fold")[["K", "w_cond"]], on="fold").rename(columns={"K": "nested_K", "w_cond": "nested_w_cond"})
    for c in ("nested", "reported"):      # within-fold z-score (label-free) to remove between-fold scale differences
        P[f"{c}_z_within_fold"] = P.groupby("fold")[c].transform(lambda v: (v - v.mean()) / v.std(ddof=1))
    assert P.nested.notna().all()
    P.reset_index().rename(columns={"index": "participant"}).to_csv(out / "outer_participant_scores.tsv", sep="\t", index=False)
    fn, fr = fold_aucs(P, "nested"), fold_aucs(P, "reported")
    FA = pd.DataFrame([{"outer_fold": f, "nested_K": int(C.K[C.outer_fold == f].iloc[0]), "nested_w_cond": float(C.w_cond[C.outer_fold == f].iloc[0]),
                        "n_dkd": int(((P.fold == f) & (P.group == "DKD")).sum()), "n_hkd": int(((P.fold == f) & (P.group == "HKD")).sum()),
                        "auc_nested": fn[f], "auc_reported": fr[f]} for f in sorted(fn)])
    FA.to_csv(out / "outer_fold_auc.tsv", sep="\t", index=False)
    y = (P.group == "DKD").values; z = stats.norm.ppf(0.5 + nc["ci"] / 2)
    A, S = delong(y, np.vstack([P.nested.values, P.reported.values]))
    Az, _ = delong(y, np.vstack([P.nested_z_within_fold.values, P.reported_z_within_fold.values]))
    se_d = np.sqrt(S[0, 0] + S[1, 1] - 2 * S[0, 1])
    bs = bootstrap(P, nc["bootstrap_reps"], np.random.default_rng(cfg["seed"]))
    lo, hi = (1 - nc["ci"]) / 2, 1 - (1 - nc["ci"]) / 2
    q = lambda v: (float(np.nanquantile(v, lo)), float(np.nanquantile(v, hi)))
    rows = []
    for name, est, d_ci, b_ci in [
            ("nested_pooled_auc", A[0], (A[0] - z * np.sqrt(S[0, 0]), A[0] + z * np.sqrt(S[0, 0])), q(bs.nested_pooled)),
            ("reported_pooled_auc", A[1], (A[1] - z * np.sqrt(S[1, 1]), A[1] + z * np.sqrt(S[1, 1])), q(bs.reported_pooled)),
            ("pooled_auc_difference_nested_minus_reported", A[0] - A[1], (A[0] - A[1] - z * se_d, A[0] - A[1] + z * se_d),
             q(bs.nested_pooled - bs.reported_pooled)),
            ("nested_mean_of_fold_aucs", FA.auc_nested.mean(), (np.nan, np.nan), q(bs.nested_meanfold)),
            ("reported_mean_of_fold_aucs", FA.auc_reported.mean(), (np.nan, np.nan), q(bs.reported_meanfold)),
            ("mean_fold_auc_difference_nested_minus_reported", FA.auc_nested.mean() - FA.auc_reported.mean(), (np.nan, np.nan),
             q(bs.nested_meanfold - bs.reported_meanfold)),
            ("nested_pooled_auc_within_fold_z", Az[0], (np.nan, np.nan), (np.nan, np.nan)),
            ("reported_pooled_auc_within_fold_z", Az[1], (np.nan, np.nan), (np.nan, np.nan))]:
        rows.append({"quantity": name, "estimate": float(est), "delong_ci_low": d_ci[0], "delong_ci_high": d_ci[1],
                     "bootstrap_ci_low": b_ci[0], "bootstrap_ci_high": b_ci[1]})
    rows.append({"quantity": "pooled_difference_delong_p", "estimate": float(2 * stats.norm.sf(abs(A[0] - A[1]) / se_d)) if se_d > 0 else np.nan})
    rows.append({"quantity": "bootstrap_fraction_mean_fold_difference_ge_0", "estimate": float((bs.nested_meanfold - bs.reported_meanfold >= 0).mean())})
    rows.append({"quantity": "n_dkd", "estimate": int(y.sum())}); rows.append({"quantity": "n_hkd", "estimate": int((~y).sum())})
    rows.append({"quantity": "n_outer_folds_selecting_reported_setting", "estimate": int(((C.K == rep[0]) & (C.w_cond == rep[1])).sum())})
    T = pd.DataFrame(rows); T.to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(C.to_string()); print(FA.round(4).to_string()); print(T.round(4).to_string())
    write_provenance("expr_nested_cv", [out / "selection.tsv"],
                     [out / f for f in ("cvw_reuse_check.tsv", "outer_participant_scores.tsv", "outer_fold_auc.tsv", "summary.tsv")],
                     cfg["seed"], cfg, extra={"bootstrap_reps": nc["bootstrap_reps"], "reported_setting": list(rep)}, tag="summary")


if __name__ == "__main__":
    main()
