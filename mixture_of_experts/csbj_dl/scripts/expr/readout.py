"""Identifiable participant score: condition-head DKD-vs-HKD log-odds averaged over each participant's
injured-PT cells, then over seeds. KPMP = held-out cross-validation predictions; external = final models.
Also seed agreement (ICC(2,1), mean pairwise Spearman), donor-level tests, and the decomposition of the
score into meta-expert contributions (head weight x mean routing; final models).
Outputs: results/expr_readout/{participant_scores.tsv, seed_agreement.tsv, donor_level_tests.tsv,
         meta_contributions.tsv, meta_contribution_tests.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import group_tests  # noqa: E402
from stability_compare import load_heldout, seed_agreement  # noqa: E402
from readout_common import final_set, cohort_pred  # noqa: E402

KPMP_COMPS = [("DKD", "HKD"), ("DKD", "Reference"), ("HKD", "Reference"), ("DKD", "HKD_withDM"), ("HKD_withDM", "HKD")]
EXT = {"gse211785": [("DKD", "HKD"), ("DKD", "Control"), ("HKD", "Control")], "gse195460": [("DKD", "Control")]}


def part_means(obs, vals: dict, mc):
    """vals: name -> per-cell array. Participant x tech means over injured PT, then 'pooled' = mean over techs."""
    df = obs[["participant", "group", "tech"]].astype(str).reset_index(drop=True)
    for k, v in vals.items():
        df[k] = v
    df = df[(obs.pt_state.astype(str) == "injured_PT").values]
    g = df.groupby(["participant", "tech"]); cols = list(vals)
    t = g[cols].mean(); t["n_cells"] = g.size(); t["group"] = g.group.first()
    t = t[t.n_cells >= mc].reset_index()
    pooled = t.groupby("participant").agg(group=("group", "first"), n_cells=("n_cells", "sum"),
                                          **{c: (c, "mean") for c in cols}).reset_index().assign(tech="pooled")
    return pd.concat([t, pooled], ignore_index=True)


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_readout"); bulk = Path(cfg["paths"]["bulk_dir"]); fs = final_set(cfg)
    mc = cfg["prep"]["min_cells_per_participant"]
    obs = {c: ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad", backed="r").obs.copy() for c in ("kpmp", "gse211785", "gse195460")}
    scores, agree, tests = [], [], []
    # KPMP held-out
    vals = {}
    for s in fs["cv_seeds"]:
        idx, P, nll, lo = load_heldout(cfg, fs["cv_prefix"], fs["K"], fs["w_cond"], s)
        v = np.full(len(obs["kpmp"]), np.nan); v[idx] = lo; vals[f"seed{s}"] = v
    cohorts = {"KPMP_heldout": (obs["kpmp"], vals)}
    for c in EXT:
        cohorts[c] = (obs[c], {f"seed{s}": cohort_pred(cfg, fs, s, c)["logodds"] for s in fs["seeds"]})
    for name, (o, v) in cohorts.items():
        pm = part_means(o, v, mc)
        sc = [k for k in v]
        pm["logodds"] = pm[sc].mean(1)
        pm["cohort"] = name
        scores.append(pm)
        for tech in pm.tech.unique():
            q = pm[pm.tech == tech]
            agree.append({"cohort": name, "tech": tech, **seed_agreement(q.set_index("participant")[sc])})
            comps = KPMP_COMPS if name == "KPMP_heldout" else EXT[name]
            tests.append(group_tests(q, "group", comps, ["logodds"], {"cohort": name, "tech": tech}))
    S = pd.concat(scores, ignore_index=True); S.to_csv(out / "participant_scores.tsv", sep="\t", index=False)
    pd.DataFrame(agree).to_csv(out / "seed_agreement.tsv", sep="\t", index=False)
    T = pd.concat(tests, ignore_index=True); T.to_csv(out / "donor_level_tests.tsv", sep="\t", index=False)
    # meta-expert decomposition (final models): contribution_k = head weight (DKD - HKD) x participant mean routing
    U = pd.read_csv(result_dir("expr_meta") / "units.tsv", sep="\t")
    M = pd.read_csv(result_dir("expr_meta") / "meta_experts.tsv", sep="\t")
    metas = M.meta_expert.tolist(); conts, ctests = [], []
    for c in ("kpmp", "gse211785", "gse195460"):
        per_seed = []
        for s in fs["seeds"]:
            P = cohort_pred(cfg, fs, s, c)["p"]; u = U[U.seed == s]
            vals = {m: np.zeros(len(P)) for m in metas}
            for _, r in u.iterrows():
                vals[r.meta_expert] = vals[r.meta_expert] + r.head_weight_dkd_minus_hkd * P[:, int(r.expert)]
            pm = part_means(obs[c], vals, mc); pm["seed"] = s; per_seed.append(pm)
        A = pd.concat(per_seed).groupby(["participant", "tech"]).agg(group=("group", "first"), **{m: (m, "mean") for m in metas}).reset_index()
        A["cohort"] = "KPMP_final" if c == "kpmp" else c
        conts.append(A)
        comps = KPMP_COMPS[:3] if c == "kpmp" else EXT[c]
        for tech in A.tech.unique():
            ctests.append(group_tests(A[A.tech == tech], "group", comps, metas, {"cohort": A.cohort.iloc[0], "tech": tech}))
    pd.concat(conts).to_csv(out / "meta_contributions.tsv", sep="\t", index=False)
    CT = pd.concat(ctests, ignore_index=True)
    CT["reproducible_meta"] = CT.feature.isin(M[M.reproducible].meta_expert)
    CT.to_csv(out / "meta_contribution_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(pd.DataFrame(agree).round(3).to_string())
    print(T[["cohort", "tech", "comparison", "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon"]].round(3).to_string())
    print(CT[CT.reproducible_meta & CT.comparison.isin(["DKD_vs_HKD", "DKD_vs_Control"]) & (CT.tech == "pooled")]
          [["cohort", "comparison", "feature", "n1", "n2", "mean1", "mean2", "g", "p_wilcoxon"]].round(3).to_string())
    write_provenance("expr_readout", [], [out / "participant_scores.tsv", out / "seed_agreement.tsv",
                                          out / "donor_level_tests.tsv", out / "meta_contribution_tests.tsv"],
                     cfg["seed"], cfg, extra={"final_set": str(fs["dir"]), "cv_prefix": fs["cv_prefix"]})


if __name__ == "__main__":
    main()
