"""Clinical bands vs the KPMP held-out DKD-vs-HKD readout, adjusted for diagnosis group, and the readout's
incremental value over diagnosis group + injured-PT fraction (+ eGFR band).

Participant table, readout (pooled held-out log-odds, mean of seeds), band coding and covariates are those of
clinical.py (read from results/expr_clinical/participant_table.tsv). Proportional-odds (logit) models:
  CKD (DKD, HKD, HKD_withDM): band ~ z(readout) + group (reference DKD) + z(injured fraction) [+ z(eGFR band)]
  CKD without group (reproduces clinical) and within DKD (no group term).
Incremental value on identical rows: likelihood-ratio test (1 df) of adding the readout to the covariate model,
McFadden pseudo-R2 (null = thresholds only) and Harrell's C of the linear predictor for the ordinal outcome.
Output: results/expr_patient_clinical/{clinical_group_adjusted.tsv, clinical_incremental.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402

BANDS = ["albuminuria_band", "proteinuria_band", "egfr_band"]


def harrell_c(lp: np.ndarray, y: np.ndarray) -> float:
    """Concordance over pairs with different outcome levels; ties in the predictor count 1/2."""
    dy = np.sign(y[:, None] - y[None, :]); dl = np.sign(lp[:, None] - lp[None, :])
    m = dy > 0
    return float(((dl[m] > 0).sum() + 0.5 * (dl[m] == 0).sum()) / m.sum())


def fit(d: pd.DataFrame, y: str, cont: list[str], dummies: list[str]) -> dict:
    """Proportional-odds fit; continuous covariates z-scored within the rows used, dummies 0/1."""
    codes = pd.Categorical(d[y], categories=sorted(d[y].unique()), ordered=True)
    X = (d[cont] - d[cont].mean()) / d[cont].std()
    X = pd.concat([X, d[dummies].astype(float)], axis=1)
    fits = []
    for method in ("bfgs", "newton", "nm"):     # first converged optimiser is kept
        fits.append(OrderedModel(pd.Series(codes, index=d.index), X, distr="logit").fit(method=method, disp=False, maxiter=5000))
        if fits[-1].mle_retvals.get("converged", False):
            break
    r = fits[-1]
    k = X.shape[1]
    lp = X.values @ r.params.values[:k]
    n_k = pd.Series(codes.codes).value_counts().values
    ll0 = float((n_k * np.log(n_k / n_k.sum())).sum())
    out = {"llf": r.llf, "ll_null": ll0, "mcfadden_r2": 1 - r.llf / ll0, "c_index": harrell_c(lp, codes.codes),
           "converged": bool(r.mle_retvals.get("converged", False)), "_res": r}
    return out


def term(r, t):
    b, se = r.params[t], r.bse[t]
    return {"or_per_sd": np.exp(b), "or_ci_low": np.exp(b - 1.96 * se), "or_ci_high": np.exp(b + 1.96 * se),
            "p": r.pvalues[t]}


def main():
    cfg = load_cfg("expr"); pc = cfg["patient_level_models"]["clinical"]
    out = result_dir("expr_patient_clinical"); src = result_dir("expr_clinical") / "participant_table.tsv"
    df = pd.read_csv(src, sep="\t", dtype={"participant": str}).set_index("participant")
    groups = pc["groups"]; ref = groups[0]
    present = sorted(df.group[df.group.isin(groups)].unique())
    dums = [f"group_{g}" for g in groups[1:] if g in present]
    for g in groups[1:]:
        df[f"group_{g}"] = (df.group == g).astype(int)
    pops = {"CKD_group_adjusted": (df[df.group.isin(groups)], dums),
            "CKD_no_group (as clinical)": (df[df.group.isin(groups)], []),
            "DKD_only": (df[df.group == ref], [])}
    adj, inc = [], []
    for pop, (d0, dm) in pops.items():
        for b in BANDS:
            y = b + "_ord"
            for cov in pc["covariate_sets"]:
                cov = [c for c in cov if not (b == "egfr_band" and c == "egfr_band_ord")]
                d = d0.dropna(subset=[y, "score"] + cov)
                if d[y].nunique() < 3 or len(d) < 15:
                    continue
                key = {"population": pop, "band": b, "covariates": "+".join((["group"] if dm else []) + cov),
                       "n": len(d), "n_levels": d[y].nunique(),
                       "n_by_group": "; ".join(f"{g}={n}" for g, n in d.group.value_counts().items())}
                full = fit(d, y, ["score"] + cov, dm); red = fit(d, y, cov, dm)
                r = full["_res"]
                row = {**key, **term(r, "score"), "converged": full["converged"]}
                for g in dm:
                    t = term(r, g)
                    row.update({f"{g}_or": t["or_per_sd"], f"{g}_ci_low": t["or_ci_low"],
                                f"{g}_ci_high": t["or_ci_high"], f"{g}_p": t["p"]})
                adj.append(row)
                lr = 2 * (full["llf"] - red["llf"])
                inc.append({**key, "lr_chi2": lr, "lr_df": 1, "p_lr": stats.chi2.sf(lr, 1),
                            "mcfadden_r2_covariates": red["mcfadden_r2"], "mcfadden_r2_with_readout": full["mcfadden_r2"],
                            "delta_mcfadden_r2": full["mcfadden_r2"] - red["mcfadden_r2"],
                            "c_covariates": red["c_index"], "c_with_readout": full["c_index"],
                            "delta_c": full["c_index"] - red["c_index"],
                            "converged_both": full["converged"] and red["converged"]})
    k = ["population", "band", "covariates"]
    A = pd.DataFrame(adj).drop_duplicates(k); I = pd.DataFrame(inc).drop_duplicates(k)
    A.to_csv(out / "clinical_group_adjusted.tsv", sep="\t", index=False)
    I.to_csv(out / "clinical_incremental.tsv", sep="\t", index=False)
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 40)
    print(A.drop(columns=["n_by_group"]).round(4).to_string()); print(I.drop(columns=["n_by_group"]).round(4).to_string())
    write_provenance("expr_patient_clinical", [src], [out / "clinical_group_adjusted.tsv", out / "clinical_incremental.tsv"],
                     cfg["seed"], cfg, extra={"note": "C-index = Harrell's C of the proportional-odds linear predictor "
                                              "(apparent, not cross-validated)"})


if __name__ == "__main__":
    main()
