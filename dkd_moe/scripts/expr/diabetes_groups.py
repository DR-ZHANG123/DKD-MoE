"""Diabetes and CKD groups on the KPMP held-out readout.

All disease groups (DKD, HKD with diabetes, HKD without diabetes, diabetes without CKD) come from KPMP
recruitment-site biopsies. For each readout (final model; supervised model without experts):
  pairwise Hedges g / Wilcoxon (config groups_clinical.group_comparisons);
  OLS (HC3), one row per participant, recruitment-site groups:
    readout ~ C(group, ref HKD) + covariates                 (groups_clinical.group_covariates)
    readout ~ C(group, ref HKD) + covariates + eGFR band     (participants with eGFR)
    factorial coding: readout ~ diabetes + CKD + covariates  (diabetes: DKD, HKD_withDM, DM_noCKD;
                                                              CKD: DKD, HKD, HKD_withDM)
Outputs: results/expr_diabetes_groups/{participant_table.tsv, group_tests.tsv, group_models.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import hedges_g  # noqa: E402
from scipy import stats  # noqa: E402
from clinical_common import participant_table, readout_table  # noqa: E402


def ols_rows(d, f, name, score):
    m = smf.ols(f, d).fit(cov_type="HC3"); ci = m.conf_int()
    keep = [t for t in m.params.index if t.startswith(("C(group", "diabetes", "ckd"))]
    rows = [{"readout": score, "model": name, "term": t.replace("C(group, Treatment('HKD'))[T.", "").rstrip("]"),
             "beta": m.params[t], "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p": m.pvalues[t],
             "n": int(m.nobs), "formula": f} for t in keep]
    k = "C(group, Treatment('HKD'))"
    if f"{k}[T.DKD]" in m.params and f"{k}[T.DM_noCKD]" in m.params:      # DKD versus diabetes without CKD
        ct = m.t_test(f"{k}[T.DKD] - {k}[T.DM_noCKD] = 0"); lo, hi = ct.conf_int()[0]
        rows.append({"readout": score, "model": name, "term": "DKD_minus_DM_noCKD", "beta": float(ct.effect[0]),
                     "ci_low": float(lo), "ci_high": float(hi), "p": float(ct.pvalue), "n": int(m.nobs), "formula": f})
    return rows


def severity_matched(P, sm):
    """Nearest-neighbour matching (Mahalanobis on the covariates, greedy, without replacement) of DKD participants to
    each participant with diabetes without CKD; balance before and after; readout differences in the matched set
    (Hedges g with Wilcoxon rank-sum, and Wilcoxon signed-rank of each participant against the mean of its matches)."""
    cov = sm["covariates"]
    D = P[P.group == "DKD"].dropna(subset=cov); M = P[P.group == "DM_noCKD"].dropna(subset=cov)
    S = np.cov(pd.concat([D, M])[cov].values.T); Si = np.linalg.pinv(S)
    dist = lambda a, b: float(np.sqrt((a - b) @ Si @ (a - b)))
    pairs, used = [], set()
    order = M.index[np.argsort([-min(dist(M.loc[i, cov].values.astype(float), D.loc[j, cov].values.astype(float)) for j in D.index)
                                for i in M.index])]
    for i in order:
        cand = sorted((dist(M.loc[i, cov].values.astype(float), D.loc[j, cov].values.astype(float)), j) for j in D.index if j not in used)
        for dd, j in cand[:sm["ratio"]]:
            used.add(j); pairs.append((i, j, dd))
    Dm = D.loc[[j for _, j, _ in pairs]]
    bal = []
    for c in cov:
        for name, dd in (("before", D), ("after", Dm)):
            sd = np.sqrt((dd[c].var() + M[c].var()) / 2)
            bal.append({"covariate": c, "set": name, "mean_DKD": dd[c].mean(), "mean_DM_noCKD": M[c].mean(),
                        "std_mean_difference": (dd[c].mean() - M[c].mean()) / sd})
    res = []
    for r in sm["readouts"]:
        g = hedges_g(Dm[r], M[r])
        mm = pd.Series({i: P.loc[[j for ii, j, _ in pairs if ii == i], r].mean() for i in M.index})
        w = stats.wilcoxon(mm.values - M[r].values)
        res.append({"readout": r, **g, "n_matched_sets": len(M), "paired_mean_difference": float((mm - M[r]).mean()),
                    "p_signed_rank": float(w.pvalue)})
    return pd.DataFrame(bal), pd.DataFrame(res), pd.DataFrame(pairs, columns=["DM_noCKD", "DKD", "distance"])


def composition_given_readout(P, covs, egfr):
    """Expert contributions compared between groups at equal total readout: OLS (HC3) of the standardized
    contribution on group (reference HKD) and the standardized readout, without and with the covariates and eGFR
    category, in the recruitment-site disease groups; DKD minus diabetes without CKD from the same model."""
    rows = []
    d0 = P[P.group.isin(["DKD", "HKD", "HKD_withDM", "DM_noCKD"])].dropna(subset=["age_ord", "male", egfr]).copy()
    d0["r"] = (d0.moe - d0.moe.mean()) / d0.moe.std()
    for comp in ("M3", "M3_train_only", "M1"):
        d = d0.copy(); d["y"] = (d[comp] - d[comp].mean()) / d[comp].std()
        for name, f in (("readout", "y ~ C(group, Treatment('HKD')) + r"),
                        ("readout_covariates_egfr", f"y ~ C(group, Treatment('HKD')) + r + {' + '.join(covs)} + {egfr}")):
            rows += [dict(x, component=comp) for x in ols_rows(d, f, name, comp)]
    return pd.DataFrame(rows)


def main():
    cfg = load_cfg("expr"); pc = cfg["groups_clinical"]; out = result_dir("expr_diabetes_groups")
    P = participant_table(cfg, pc["min_site_participants"]).join(readout_table(), how="inner")
    P.to_csv(out / "participant_table.tsv", sep="\t")
    tests, models = [], []
    for score in ("moe", "no_experts", "M1", "M3", "M3_train_only", "nonmoe_Lpos", "nonmoe_Lneg"):
        for a, b in pc["group_comparisons"]:
            tests.append({"readout": score, "comparison": f"{a}_vs_{b}",
                          **hedges_g(P.loc[P.group == a, score], P.loc[P.group == b, score])})
        d = P[P.group.isin(pc["groups_same_procurement"])].copy()
        d["y"] = (d[score] - d[score].mean()) / d[score].std()     # standardised readout
        d["diabetes"] = d.group.isin(["DKD", "HKD_withDM", "DM_noCKD"]).astype(int)
        d["ckd"] = d.group.isin(["DKD", "HKD", "HKD_withDM"]).astype(int)
        cov = " + ".join(pc["group_covariates"])
        dd = d.dropna(subset=["age_ord", "male"])
        de = dd.dropna(subset=[pc["egfr_covariate"]])
        models += ols_rows(d, "y ~ C(group, Treatment('HKD'))", "unadjusted", score)
        models += ols_rows(dd, f"y ~ C(group, Treatment('HKD')) + {cov}", "adjusted", score)
        models += ols_rows(de, f"y ~ C(group, Treatment('HKD')) + {cov} + {pc['egfr_covariate']}", "adjusted_egfr", score)
        models += ols_rows(dd, f"y ~ diabetes + ckd + {cov}", "factorial_adjusted", score)
        models += ols_rows(de, f"y ~ diabetes + ckd + {cov} + {pc['egfr_covariate']}", "factorial_adjusted_egfr", score)
        # diabetes effect without CKD vs HKD, adjusted for eGFR (DM_noCKD has higher eGFR by definition)
    T = pd.DataFrame(tests); M = pd.DataFrame(models)
    CR = composition_given_readout(P, pc["group_covariates"], pc["egfr_covariate"])
    CR.to_csv(out / "composition_given_readout.tsv", sep="\t", index=False)
    print(CR[CR.term.isin(["DKD", "DM_noCKD", "DKD_minus_DM_noCKD"])].drop(columns="formula").round(3).to_string())
    Bal, Res, Pairs = severity_matched(P, cfg["severity_matching"])
    Bal.to_csv(out / "matched_balance.tsv", sep="\t", index=False); Res.to_csv(out / "matched_tests.tsv", sep="\t", index=False)
    Pairs.to_csv(out / "matched_pairs.tsv", sep="\t", index=False)
    print(Bal.round(3).to_string()); print(Res.round(3).to_string())
    T.to_csv(out / "group_tests.tsv", sep="\t", index=False); M.to_csv(out / "group_models.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(P.groupby("group")[["moe", "no_experts", "injured_fraction", "egfr_ord", "age_ord"]].agg(["count", "mean"]).round(2))
    print(T.round(3).to_string()); print(M.drop(columns="formula").round(3).to_string())
    write_provenance("expr_diabetes_groups", [result_dir("expr_readout") / "participant_scores.tsv",
                     result_dir("bench_supervised_single") / "donor_scores.tsv"],
                     [out / "participant_table.tsv", out / "group_tests.tsv", out / "group_models.tsv", out / "matched_balance.tsv",
                      out / "matched_tests.tsv", out / "matched_pairs.tsv", out / "composition_given_readout.tsv"], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
