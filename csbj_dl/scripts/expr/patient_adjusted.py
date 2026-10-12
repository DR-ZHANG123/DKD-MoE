"""Participant-level adjusted DKD-vs-HKD readout (KPMP held-out readout, DKD and HKD only).

Primary (one row per participant, pooled readout = mean over technologies, n = 122):
  OLS (HC3)  logodds ~ DKD + injured_fraction + log_n_cells + C(site) + C(tech_set)
  injured_fraction and n_cells are taken over the technologies that contributed to the pooled readout;
  tech_set = sc only / sn only / both; sites with < min_site_participants participants are pooled as 'other';
  a site level that still has fewer participants than that is left out of site-adjusted models (its own
  indicator would absorb it completely, and leverage 1 makes HC3 undefined).
Sensitivity (participant x technology rows, 144): OLS with cluster-robust SEs by participant, with and
  without DKD x technology; linear mixed model with a random intercept per participant (REML).
Also: exact KPMP held-out values for DKD vs HKD_withDM and HKD_withDM vs HKD (from expr_readout).
Outputs: results/expr_patient_adjusted/{participant_data.tsv, adjusted_models.tsv, model_fit.tsv,
         subgroup_comparisons.tsv}
"""
from __future__ import annotations
import sys, warnings
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402

KEEP = ("DKD", "injured", "log_n", "C(tech", "C(tech_set")


def build(cfg):
    S = pd.read_csv(result_dir("expr_readout") / "participant_scores.tsv", sep="\t", dtype={"participant": str})
    S = S[(S.cohort == "KPMP_heldout") & S.group.isin(["DKD", "HKD"])].copy()
    tab = pd.read_csv(result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv", sep="\t", dtype={"participant": str})
    tab["injured_fraction"] = tab.injured_PT / (tab.injured_PT + tab.healthy_PT)
    rows = S[S.tech != "pooled"].merge(tab[["participant", "tech", "injured_fraction"]], on=["participant", "tech"], how="left")
    assert rows.injured_fraction.notna().all()
    rows["log_n_cells"] = np.log(rows.n_cells)
    pooled = S[S.tech == "pooled"].set_index("participant")[["group", "logodds", "n_cells"]]
    agg = rows.groupby("participant").agg(injured_fraction=("injured_fraction", "mean"),
                                          tech_set=("tech", lambda t: "both" if t.nunique() > 1 else t.iloc[0]))
    P = pooled.join(agg)
    assert P.tech_set.notna().all() and len(P) == len(pooled)
    P["log_n_cells"] = np.log(P.n_cells)
    m = cfg["patient_level_models"]["adjusted"]["min_site_participants"]
    site = P.index.to_series().str.extract(r"^(\d+)-", expand=False).fillna("other")
    big = site.map(site.value_counts()) >= m
    P["site"] = site; P["site_c"] = np.where(big, site, "other")
    small_other = (P.site_c == "other").sum() < m
    P["site_model"] = ~((P.site_c == "other") & small_other)
    P["DKD"] = (P.group == "DKD").astype(int)
    rows["DKD"] = (rows.group == "DKD").astype(int)
    rows = rows.merge(P[["site_c", "site_model"]], left_on="participant", right_index=True)
    return P.reset_index(), rows


def tidy(m, name, level, se_type, n_part, extra=None):
    ci = m.conf_int()
    out = []
    for t in [t for t in m.params.index if t.startswith(KEEP)]:
        out.append({"model": name, "level": level, "se": se_type, "term": t, "beta": m.params[t],
                    "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "se_value": m.bse[t], "p": m.pvalues[t],
                    "n_participants": n_part, "n_rows": int(m.nobs), **(extra or {})})
    return out


def fit_mixed(f, d):
    """REML random-intercept model; tries lbfgs then powell / nm. Returns (result, info)."""
    for method in ("lbfgs", "powell", "nm"):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                r = smf.mixedlm(f, d, groups=d["participant"]).fit(reml=True, method=method, maxiter=2000)
            except Exception as e:  # noqa: BLE001
                info = {"converged": False, "method": method, "warnings": repr(e)}; continue
        msgs = "; ".join(sorted({str(x.message)[:120] for x in w}))
        info = {"converged": bool(r.converged), "method": method, "warnings": msgs,
                "re_var_participant": float(r.cov_re.iloc[0, 0]), "resid_var": float(r.scale)}
        if r.converged:
            return r, info
    return None, info


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_patient_adjusted")
    P, R = build(cfg)
    P.to_csv(out / "participant_data.tsv", sep="\t", index=False)
    res, fit = [], []
    Ps = P[P.site_model]; Rs = R[R.site_model]
    base = "logodds ~ DKD + injured_fraction + log_n_cells"
    forms = {"p0_unadjusted": ("logodds ~ DKD", P),
             "p1_cell_covariates": (base, P),
             "p2_plus_site": (base + " + C(site_c)", Ps),
             "p3_primary_plus_tech_set": (base + " + C(site_c) + C(tech_set, Treatment('sc'))", Ps)}
    for name, (f, d) in forms.items():
        m = smf.ols(f, d).fit(cov_type="HC3")
        res += tidy(m, name, "participant", "HC3", d.participant.nunique())
        fit.append({"model": name, "formula": f, "n_rows": int(m.nobs), "r2": m.rsquared, "converged": True})
    # participant x technology rows, cluster-robust by participant
    cl = {"s1_rows_cluster": "logodds ~ DKD + C(tech) + injured_fraction + log_n_cells + C(site_c)",
          "s2_rows_cluster_interaction": "logodds ~ DKD * C(tech) + injured_fraction + log_n_cells + C(site_c)"}
    for name, f in cl.items():
        g = pd.factorize(Rs.participant)[0]
        m = smf.ols(f, Rs).fit(cov_type="cluster", cov_kwds={"groups": g})
        res += tidy(m, name, "participant_x_technology", "cluster(participant)", Rs.participant.nunique())
        fit.append({"model": name, "formula": f, "n_rows": int(m.nobs), "r2": m.rsquared, "converged": True})
        if "interaction" in name:   # technology-specific DKD effects from the interaction model
            cov = m.cov_params(); b = m.params
            k = "DKD:C(tech)[T.sn]"; est = b["DKD"] + b[k]
            se = float(np.sqrt(cov.loc["DKD", "DKD"] + cov.loc[k, k] + 2 * cov.loc["DKD", k]))
            from scipy import stats
            for tech, e, s in (("sc", b["DKD"], m.bse["DKD"]), ("sn", est, se)):
                res.append({"model": name, "level": "participant_x_technology", "se": "cluster(participant)",
                            "term": f"DKD_effect_in_{tech}", "beta": e, "ci_low": e - 1.96 * s, "ci_high": e + 1.96 * s,
                            "se_value": s, "p": 2 * stats.norm.sf(abs(e / s)),
                            "n_participants": Rs.participant.nunique(), "n_rows": int(m.nobs)})
    for name, f in {"s3_mixed_random_intercept": cl["s1_rows_cluster"],
                    "s4_mixed_random_intercept_interaction": cl["s2_rows_cluster_interaction"]}.items():
        r, info = fit_mixed(f, Rs)
        fit.append({"model": name, "formula": f + " + (1 | participant)", "n_rows": len(Rs), **info})
        if r is not None:
            res += tidy(r, name, "participant_x_technology", "REML model-based", Rs.participant.nunique(),
                        {"converged": info["converged"]})
    A = pd.DataFrame(res); A.to_csv(out / "adjusted_models.tsv", sep="\t", index=False)
    F = pd.DataFrame(fit); F.to_csv(out / "model_fit.tsv", sep="\t", index=False)
    # subgroup comparisons, extracted verbatim
    T = pd.read_csv(result_dir("expr_readout") / "donor_level_tests.tsv", sep="\t")
    sub = T[(T.cohort == "KPMP_heldout") & T.comparison.isin(["DKD_vs_HKD", "DKD_vs_HKD_withDM", "HKD_withDM_vs_HKD"])]
    sub.to_csv(out / "subgroup_comparisons.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(P.groupby(["group", "tech_set"]).size().unstack(), "\nsite_c:", P.site_c.value_counts().to_dict(),
          "\nleft out of site models:", P.loc[~P.site_model, "participant"].tolist())
    print(A[A.term.str.startswith("DKD")].round(4).to_string()); print(F.to_string()); print(sub.round(4).to_string())
    write_provenance("expr_patient_adjusted", [result_dir("expr_readout") / "participant_scores.tsv",
                     result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv",
                     result_dir("expr_readout") / "donor_level_tests.tsv"],
                     [out / "participant_data.tsv", out / "adjusted_models.tsv", out / "model_fit.tsv",
                      out / "subgroup_comparisons.tsv"], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
