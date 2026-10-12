"""Clinical associations of the identifiable participant score (KPMP held-out DKD-vs-HKD log-odds, mean of seeds).

Ordinal bands (no midpoints): albuminuria, proteinuria, eGFR. Proportional-odds models with the score
(z) and covariate sets from expr.yaml readout_stability.clinical (injured-PT fraction; + eGFR band for the urine bands),
plus Spearman. Populations: all, CKD (DKD/HKD/HKD_withDM), DKD only.
Output: results/expr_clinical/{participant_table.tsv, clinical_associations.tsv}
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
from clinical import band_order, ordinal_fit, BANDS  # noqa: E402


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_clinical"); bulk = Path(cfg["paths"]["bulk_dir"])
    S = pd.read_csv(result_dir("expr_readout") / "participant_scores.tsv", sep="\t", dtype={"participant": str})
    S = S[(S.cohort == "KPMP_heldout") & (S.tech == "pooled")].set_index("participant")
    tab = pd.read_csv(result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv", sep="\t", dtype={"participant": str})
    tab["frac"] = tab.injured_PT / (tab.injured_PT + tab.healthy_PT)
    obs = ad.read_h5ad(bulk / "kpmp_pt_hvg.h5ad", backed="r").obs
    clin = obs.groupby("participant", observed=True)[BANDS].first().astype(str)
    df = S[["group", "logodds"]].rename(columns={"logodds": "score"})
    df["injured_fraction"] = tab.groupby("participant").frac.mean().reindex(df.index)
    df = df.join(clin)
    for b in BANDS:
        df[b + "_ord"] = df[b].map(band_order)
    df.to_csv(out / "participant_table.tsv", sep="\t")
    pops = {"all": df.index, "CKD": df.index[df.group.isin(["DKD", "HKD", "HKD_withDM"])], "DKD": df.index[df.group == "DKD"]}
    rows = []
    for pop in cfg["readout_stability"]["clinical"]["populations"]:
        d = df.loc[pops[pop]]
        for b in BANDS:
            y = b + "_ord"
            for cov in cfg["readout_stability"]["clinical"]["covariates"]:
                cov = [c for c in cov if not (b == "egfr_band" and c == "egfr_band_ord")]
                dd = d.dropna(subset=[y, "score"] + cov)
                if len(dd) < 15:
                    continue
                rho, p = stats.spearmanr(dd.score, dd[y])
                r = {"population": pop, "band": b, "covariates": "+".join(cov), "n": len(dd), "n_levels": dd[y].nunique(),
                     "spearman_rho": rho, "p_spearman": p}
                r.update(ordinal_fit(dd, y, ["score"] + cov))
                rows.append(r)
    R = pd.DataFrame(rows).drop_duplicates(["population", "band", "covariates"])
    R.to_csv(out / "clinical_associations.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); print(R.round(4).to_string())
    write_provenance("expr_clinical", [result_dir("expr_readout") / "participant_scores.tsv"],
                     [out / "participant_table.tsv", out / "clinical_associations.tsv"], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
