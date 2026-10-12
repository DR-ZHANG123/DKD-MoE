"""Extended clinical models of the KPMP held-out readouts (participants with CKD: DKD, HKD, HKD with diabetes).

For every readout column of the participant table (final model, supervised model without experts, and any
further readouts passed with --extra <tsv with participant + score columns>):
  proportional-odds models  band ~ z(readout) + covariates, covariate sets groups_clinical.clinical.covariate_sets
    (group = diagnosis dummies, reference DKD; tech_set and site_c as dummies; continuous covariates z-scored)
  participant bootstrap of the odds ratio (groups_clinical.clinical.n_boot resamples, percentile 95% CI)
  cross-validated incremental value: repeated stratified K-fold, out-of-fold log-likelihood per participant
    of the covariate model with and without the readout (difference > 0 = readout adds information)
  missing data: participants with / without each band by group
Also both readouts in one model (final model and no-expert readout together) and albuminuria / proteinuria in
the participants with both measured.
Outputs: results/expr_clinical_models/{models.tsv, bootstrap.tsv, cv_incremental.tsv, missing.tsv, joint.tsv}
"""
from __future__ import annotations
import argparse, sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from clinical_common import participant_table, readout_table  # noqa: E402

warnings.filterwarnings("ignore")
OUTC = {"albuminuria_band": "alb_ord", "proteinuria_band": "prot_ord", "egfr_band": "egfr_ord"}
CAT = {"group": "DKD", "tech_set": "both", "site_c": None}


def design(d: pd.DataFrame, cols: list[str], mean=None, sd=None):
    """Continuous columns z-scored (with given mean / sd if supplied), categorical columns as dummies."""
    parts, cont = [], [c for c in cols if c not in CAT]
    mean = d[cont].mean() if mean is None else mean; sd = d[cont].std() if sd is None else sd
    parts.append((d[cont] - mean) / sd)
    for c in cols:
        if c in CAT:
            ref = CAT[c] if CAT[c] is not None else d[c].value_counts().idxmax()
            lv = sorted(set(d[c]) - {ref})
            parts.append(pd.DataFrame({f"{c}_{l}": (d[c] == l).astype(float) for l in lv}, index=d.index))
    X = pd.concat(parts, axis=1)
    return X.loc[:, X.std() > 0], mean, sd


def fit_po(y: pd.Series, X: pd.DataFrame):
    codes = pd.Categorical(y, categories=sorted(y.unique()), ordered=True)
    for m in ("bfgs", "newton", "nm"):
        r = OrderedModel(pd.Series(codes, index=y.index), X, distr="logit").fit(method=m, disp=False, maxiter=5000)
        if r.mle_retvals.get("converged", False):
            break
    return r, list(codes.categories)


def boot_one(d, y, s, cov, seed):
    """Odds-ratio coefficient of one participant bootstrap resample (None when the fit is not possible)."""
    db = d.iloc[np.random.default_rng(seed).integers(0, len(d), len(d))]
    if db[y].nunique() < 3:
        return None
    try:
        Xb, _, _ = design(db, [s] + cov); rb, _ = fit_po(db[y], Xb)
        return float(rb.params[s])
    except Exception:  # noqa: BLE001  (singular resample)
        return None


def oof_rep(d, y, cols, score, folds, seed):
    """Out-of-fold log-likelihood per participant of one repeat: (covariates only, covariates + readout)."""
    skf = StratifiedKFold(folds, shuffle=True, random_state=seed)
    ll = {"cov": 0.0, "full": 0.0}
    for tr, te in skf.split(d, d[y]):
        a, b = d.iloc[tr], d.iloc[te]
        for k, cc in (("cov", cols), ("full", [score] + cols)):
            Xa, mu, sd = design(a, cc); Xb, _, _ = design(b, cc, mu, sd)
            Xb = Xb.reindex(columns=Xa.columns, fill_value=0.0)
            r, cats = fit_po(a[y], Xa)
            pr = r.model.predict(r.params, exog=Xb.values)
            idx = pd.Categorical(b[y], categories=cats).codes
            ll[k] += float(np.log(np.clip(pr[np.arange(len(b)), idx], 1e-12, 1)).sum())
    return ll["cov"] / len(d), ll["full"] / len(d)


def oof_loglik(d, y, cols, score, folds, repeats, seed, n_jobs):
    """Mean out-of-fold log-likelihood per participant, covariates only vs covariates + readout (repeated K-fold)."""
    r = Parallel(n_jobs=n_jobs)(delayed(oof_rep)(d, y, cols, score, folds, seed + k) for k in range(repeats))
    res = {"cov": [x[0] for x in r], "full": [x[1] for x in r]}
    c, f = np.array(res["cov"]), np.array(res["full"])
    return {"oof_ll_cov": c.mean(), "oof_ll_full": f.mean(), "delta_oof_ll": (f - c).mean(),
            "delta_q025": np.quantile(f - c, 0.025), "delta_q975": np.quantile(f - c, 0.975),
            "frac_repeats_positive": float((f > c).mean())}


def or_row(r, t):
    b, se = r.params[t], r.bse[t]
    return {"or_per_sd": np.exp(b), "ci_low": np.exp(b - 1.96 * se), "ci_high": np.exp(b + 1.96 * se), "p": r.pvalues[t]}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--extra", nargs="*", default=[])
    a = ap.parse_args()
    cfg = load_cfg("expr"); pc = cfg["groups_clinical"]["clinical"]; out = result_dir("expr_clinical_models")
    P = participant_table(cfg, cfg["groups_clinical"]["min_site_participants"]).join(readout_table(), how="inner")
    scores = ["moe", "no_experts", "M1", "M3", "sup_injured", "sup_patient"]
    for f in a.extra:
        E = pd.read_csv(f, sep="\t", dtype={"participant": str}).set_index("participant")
        P = P.join(E); scores += list(E.columns)
    C = P[P.group.isin(pc["population"])].copy()
    miss = []
    for b, o in OUTC.items():
        for g, d in C.groupby("group"):
            miss.append({"band": b, "group": g, "n_total": len(d), "n_measured": int(d[o].notna().sum())})
    pd.DataFrame(miss).to_csv(out / "missing.tsv", sep="\t", index=False)
    rows, boots, cvs = [], [], []
    rng = np.random.default_rng(cfg["seed"])
    for b, y in OUTC.items():
        for cname, cov in pc["covariate_sets"].items():
            cov = [c for c in cov if not (y == "egfr_ord" and c == "egfr_ord")]
            for s in scores:
                d = C.dropna(subset=[y, s] + [c for c in cov if c not in CAT])
                X, _, _ = design(d, [s] + cov); r, _ = fit_po(d[y], X)
                rows.append({"band": b, "covariates": cname, "readout": s, "n": len(d), **or_row(r, s),
                             "converged": bool(r.mle_retvals.get("converged", False))})
                if cname in ("diagnosis_egfr", "full") and b == "albuminuria_band" and s in pc["resampled_readouts"]:
                    seeds = rng.integers(0, 2**31, pc["n_boot"])
                    bs = np.array([v for v in Parallel(n_jobs=pc["n_jobs"])(delayed(boot_one)(d, y, s, cov, int(k)) for k in seeds)
                                   if v is not None])
                    boots.append({"band": b, "covariates": cname, "readout": s, "n": len(d), "n_boot_ok": len(bs),
                                  "or_median": np.exp(np.median(bs)), "or_q025": np.exp(np.quantile(bs, 0.025)),
                                  "or_q975": np.exp(np.quantile(bs, 0.975)), "frac_or_gt_1": float((bs > 0).mean())})
                    cvs.append({"band": b, "covariates": cname, "readout": s, "n": len(d),
                                **oof_loglik(d, y, cov, s, pc["cv_folds"], pc["cv_repeats"], cfg["seed"], pc["n_jobs"])})
        print("done", b, flush=True)
    M, B, V = pd.DataFrame(rows), pd.DataFrame(boots), pd.DataFrame(cvs)
    M.to_csv(out / "models.tsv", sep="\t", index=False); B.to_csv(out / "bootstrap.tsv", sep="\t", index=False)
    V.to_csv(out / "cv_incremental.tsv", sep="\t", index=False)
    # both readouts together; albuminuria and proteinuria on the same participants
    J = []
    cov = pc["covariate_sets"]["diagnosis_egfr"]
    for b, y in (("albuminuria_band", "alb_ord"), ("proteinuria_band", "prot_ord")):
        for pair in (("moe", "no_experts"), ("M3", "no_experts"), ("M3", "moe")):
            d = C.dropna(subset=[y, *pair, "egfr_ord"])
            X, _, _ = design(d, list(pair) + cov); r, _ = fit_po(d[y], X)
            J += [{"analysis": "both_readouts:" + "+".join(pair), "band": b, "readout": s, "n": len(d), **or_row(r, s)} for s in pair]
        for s in ("moe", "no_experts", "M3"):
            d2 = C.dropna(subset=["alb_ord", "prot_ord", s, "egfr_ord"])
            X, _, _ = design(d2, [s] + cov); r, _ = fit_po(d2[y], X)
            J.append({"analysis": "common_subset", "band": b, "readout": s, "n": len(d2), **or_row(r, s)})
    pd.DataFrame(J).to_csv(out / "joint.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(M.round(3).to_string()); print(B.round(3).to_string()); print(V.round(4).to_string()); print(pd.DataFrame(J).round(3).to_string())
    write_provenance("expr_clinical_models", [result_dir("expr_readout") / "participant_scores.tsv"] + [Path(f) for f in a.extra],
                     [out / f for f in ("models.tsv", "bootstrap.tsv", "cv_incremental.tsv", "missing.tsv", "joint.tsv")],
                     cfg["seed"], cfg, extra={"readouts": scores})


if __name__ == "__main__":
    main()
