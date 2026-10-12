"""Is the stronger sc than sn DKD-vs-HKD readout difference explained by procurement, injured-PT fraction,
cell number or recruitment site? Participant x technology level, KPMP held-out readout (DKD and HKD only).
Models (OLS, HC3): readout ~ DKD + tech + injured_fraction + log(n_cells) [+ site] and with DKD x tech.
Output: results/expr_transfer/{confounding_models.tsv, confounding_data.tsv, confounding_balance.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_transfer"); bulk = Path(cfg["paths"]["bulk_dir"])
    S = pd.read_csv(result_dir("expr_readout") / "participant_scores.tsv", sep="\t", dtype={"participant": str})
    S = S[(S.cohort == "KPMP_heldout") & (S.tech != "pooled") & S.group.isin(["DKD", "HKD"])].copy()
    tab = pd.read_csv(result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv", sep="\t", dtype={"participant": str})
    tab["injured_fraction"] = tab.injured_PT / (tab.injured_PT + tab.healthy_PT)
    obs = ad.read_h5ad(bulk / "kpmp_pt_hvg.h5ad", backed="r").obs
    ts = obs.groupby("participant", observed=True).tissue_source.first().astype(str)
    d = S.merge(tab[["participant", "tech", "injured_fraction"]], on=["participant", "tech"], how="left")
    d["log_n_cells"] = np.log(d.n_cells); d["DKD"] = (d.group == "DKD").astype(int)
    d["tissue_source"] = d.participant.map(ts)
    d["site"] = d.participant.str.extract(r"^(\d+)-", expand=False).fillna("other")
    bal = d.groupby(["tech", "group"]).agg(n=("participant", "size"), readout=("logodds", "mean"),
                                           injured_fraction=("injured_fraction", "mean"),
                                           log_n_cells=("log_n_cells", "mean"),
                                           n_tissue_sources=("tissue_source", "nunique"), n_sites=("site", "nunique")).reset_index()
    bal.to_csv(out / "confounding_balance.tsv", sep="\t", index=False)
    pd.crosstab([d.tech, d.group], d.site).to_csv(out / "confounding_site_table.tsv", sep="\t")
    d.to_csv(out / "confounding_data.tsv", sep="\t", index=False)
    site_ok = d.groupby("site").size().ge(3)
    d["site_c"] = np.where(d.site.map(site_ok), d.site, "other")
    ts_varies = d.groupby("group").tissue_source.nunique().max() > 1
    forms = {"m1_group_tech": "logodds ~ DKD + C(tech)",
             "m2_adjusted": "logodds ~ DKD + C(tech) + injured_fraction + log_n_cells",
             "m3_adjusted_site": "logodds ~ DKD + C(tech) + injured_fraction + log_n_cells + C(site_c)",
             "m4_interaction": "logodds ~ DKD * C(tech) + injured_fraction + log_n_cells + C(site_c)"}
    if ts_varies:
        forms["m5_tissue_source"] = "logodds ~ DKD + C(tech) + injured_fraction + log_n_cells + C(tissue_source)"
    for tech in ("sc", "sn"):
        forms[f"{tech}_only_adjusted"] = ("logodds ~ DKD + injured_fraction + log_n_cells + C(site_c)", tech)
    rows = []
    for name, f in forms.items():
        dd = d if isinstance(f, str) else d[d.tech == f[1]]
        f = f if isinstance(f, str) else f[0]
        m = smf.ols(f, dd).fit(cov_type="HC3")
        for term in [t for t in m.params.index if t.startswith(("DKD", "injured", "log_n", "C(tech)"))]:
            rows.append({"model": name, "term": term, "beta": m.params[term], "se": m.bse[term], "p": m.pvalues[term],
                         "n": int(m.nobs), "r2": m.rsquared})
    R = pd.DataFrame(rows); R.to_csv(out / "confounding_models.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(bal.round(3).to_string()); print("tissue_source varies within DKD/HKD:", ts_varies)
    print(R.round(4).to_string())
    write_provenance("expr_transfer", [result_dir("expr_readout") / "participant_scores.tsv"],
                     [out / "confounding_models.tsv", out / "confounding_balance.tsv"], cfg["seed"], cfg, tag="confounding")


if __name__ == "__main__":
    main()
