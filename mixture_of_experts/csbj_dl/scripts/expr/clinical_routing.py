"""KPMP participant-level routing to the DKD-preferred expert vs clinical bands (ordinal, no midpoints).

Gate = held-out (cross-validated) participant mean routing among injured-PT cells (mean across techs).
Covariate = injured-PT fraction of PT cells (participant mean across techs).
Models: Spearman; proportional-odds logistic regression band ~ z(gate) + z(injured fraction).
Populations: all participants with the band; CKD (DKD, HKD, HKD_withDM); DKD only.
Output: results/expr_clinical_routing/{participant_table.tsv, clinical_associations.tsv}
"""
from __future__ import annotations
import json, re, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.miscmodels.ordinal_model import OrderedModel

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402

BANDS = ["albuminuria_band", "proteinuria_band", "egfr_band"]


def band_order(v: str) -> float:
    """Ordinal position from the band's lower bound ('<30' -> 0, '30 to <300' -> 30, '>=1000' -> 1000)."""
    if v in ("nan", "", "None"):
        return np.nan
    s = v.replace(",", "")
    if s.strip().startswith("<"):
        return -1.0
    m = re.search(r"(\d+(?:\.\d+)?)", s)
    return float(m.group(1)) if m else np.nan


def ordinal_fit(df, y, xs):
    d = df.dropna(subset=[y] + xs)
    if d[y].nunique() < 3 or len(d) < 15:
        return {}
    codes = pd.Categorical(d[y], categories=sorted(d[y].unique()), ordered=True)
    X = (d[xs] - d[xs].mean()) / d[xs].std()
    r = OrderedModel(pd.Series(codes, index=d.index), X, distr="logit").fit(method="bfgs", disp=False, maxiter=500)
    b, se = r.params[xs[0]], r.bse[xs[0]]
    return {"or_per_sd": np.exp(b), "or_ci_low": np.exp(b - 1.96 * se), "or_ci_high": np.exp(b + 1.96 * se),
            "p_ordinal": r.pvalues[xs[0]], "n_ordinal": len(d)}


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_clinical_routing"); bulk = Path(cfg["paths"]["bulk_dir"])
    dk = json.loads((result_dir("expr_gates") / "dkd_expert.json").read_text())["dkd_preferred_expert"]
    pm = pd.read_csv(result_dir("expr_gates") / "participant_mean_gates.tsv", sep="\t", dtype={"participant": str})
    pm = pm[(pm.cohort == "KPMP_heldout") & (pm.cells == "injured_PT") & (pm.tech == "pooled")]
    tab = pd.read_csv(result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv", sep="\t", dtype={"participant": str})
    tab["frac"] = tab.injured_PT / (tab.injured_PT + tab.healthy_PT)
    frac = tab.groupby("participant").frac.mean()
    obs = ad.read_h5ad(bulk / "kpmp_pt_hvg.h5ad", backed="r").obs
    clin = obs.groupby("participant", observed=True)[BANDS].first().astype(str)
    df = pm.set_index("participant")[["group", dk]].rename(columns={dk: "gate"})
    df["injured_fraction"] = frac.reindex(df.index)
    df = df.join(clin)
    for b in BANDS:
        df[b + "_ord"] = df[b].map(band_order)
    df.to_csv(out / "participant_table.tsv", sep="\t")
    rows = []
    pops = {"all": df.index, "CKD": df.index[df.group.isin(["DKD", "HKD", "HKD_withDM"])],
            "DKD": df.index[df.group == "DKD"]}
    for pop, idx in pops.items():
        d = df.loc[idx]
        for b in BANDS:
            y = b + "_ord"; dd = d.dropna(subset=[y, "gate", "injured_fraction"])
            if len(dd) < 10:
                continue
            rho, p = stats.spearmanr(dd.gate, dd[y])
            # partial Spearman adjusting for injured fraction (rank residuals)
            rk = dd[["gate", y, "injured_fraction"]].rank()
            res = lambda a: rk[a] - np.polyval(np.polyfit(rk.injured_fraction, rk[a], 1), rk.injured_fraction)
            prho, pp = stats.pearsonr(res("gate"), res(y))
            r = {"population": pop, "band": b, "n": len(dd), "n_levels": dd[y].nunique(), "spearman_rho": rho,
                 "p_spearman": p, "partial_rho_adj_injured_fraction": prho, "p_partial": pp}
            r.update({f"{k}_adj": v for k, v in ordinal_fit(dd, y, ["gate", "injured_fraction"]).items()})
            r.update({f"{k}_unadj": v for k, v in ordinal_fit(dd, y, ["gate"]).items()})
            rows.append(r)
    res = pd.DataFrame(rows)
    res.to_csv(out / "clinical_associations.tsv", sep="\t", index=False)
    print(res.round(4).to_string())
    write_provenance("expr_clinical_routing", [result_dir("expr_gates") / "participant_mean_gates.tsv"],
                     [out / "participant_table.tsv", out / "clinical_associations.tsv"], cfg["seed"], cfg,
                     extra={"dkd_expert": dk, "note": "egfr bands ordered by lower bound; higher = better function"})


if __name__ == "__main__":
    main()
