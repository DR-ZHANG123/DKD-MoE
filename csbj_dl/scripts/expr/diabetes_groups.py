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
from clinical_common import participant_table, readout_table  # noqa: E402


def ols_rows(d, f, name, score):
    m = smf.ols(f, d).fit(cov_type="HC3"); ci = m.conf_int()
    keep = [t for t in m.params.index if t.startswith(("C(group", "diabetes", "ckd"))]
    return [{"readout": score, "model": name, "term": t.replace("C(group, Treatment('HKD'))[T.", "").rstrip("]"),
             "beta": m.params[t], "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p": m.pvalues[t],
             "n": int(m.nobs), "formula": f} for t in keep]


def main():
    cfg = load_cfg("expr"); pc = cfg["groups_clinical"]; out = result_dir("expr_diabetes_groups")
    P = participant_table(cfg, pc["min_site_participants"]).join(readout_table(), how="inner")
    P.to_csv(out / "participant_table.tsv", sep="\t")
    tests, models = [], []
    for score in ("moe", "no_experts"):
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
    T.to_csv(out / "group_tests.tsv", sep="\t", index=False); M.to_csv(out / "group_models.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(P.groupby("group")[["moe", "no_experts", "injured_fraction", "egfr_ord", "age_ord"]].agg(["count", "mean"]).round(2))
    print(T.round(3).to_string()); print(M.drop(columns="formula").round(3).to_string())
    write_provenance("expr_diabetes_groups", [result_dir("expr_readout") / "participant_scores.tsv",
                     result_dir("bench_supervised_single") / "donor_scores.tsv"],
                     [out / "participant_table.tsv", out / "group_tests.tsv", out / "group_models.tsv"], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
