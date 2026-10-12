"""Participant-level routing tests.

1. KPMP held-out predictions -> expert roles (injury enrichment, DKD vs HKD) and the DKD-preferred expert
   (chosen on KPMP held-out participants only, before looking at external cohorts).
2. Participant mean gate weights per expert (KPMP held-out, GSE211785, GSE195460).
3. Donor-level tests: Hedges g (95% CI) + Wilcoxon rank-sum P, primary = injured-PT cells.
4. Seed robustness of the DKD-preferred expert in external cohorts (final models, per seed).
Outputs: results/expr_gates/*.tsv, dkd_expert.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
import anndata as ad

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import hedges_g, group_tests  # noqa: E402

KPMP_COMPS = [("DKD", "HKD"), ("DKD", "Reference"), ("HKD", "Reference"), ("DKD", "HKD_withDM"),
              ("HKD_withDM", "HKD")]
EXT_COMPS = {"gse211785": [("DKD", "HKD"), ("DKD", "Control"), ("HKD", "Control")],
             "gse195460": [("DKD", "Control")]}


def pmeans(df, ecols, state, min_cells):
    """Participant x tech means, then the participant mean across technologies ('pooled')."""
    d = df if state == "all_PT" else df[df.pt_state == state]
    g = d.groupby(["participant", "tech"])
    t = g[ecols].mean(); t["n_cells"] = g.size(); t["group"] = g.group.first()
    t = t[t.n_cells >= min_cells].reset_index()
    pooled = t.groupby("participant").agg(group=("group", "first"), n_cells=("n_cells", "sum"),
                                          **{c: (c, "mean") for c in ecols}).reset_index()
    pooled["tech"] = "pooled"
    return pd.concat([t, pooled], ignore_index=True).assign(cells=state)


def expert_roles(df, ecols, min_cells):
    """Per expert on KPMP held-out: injured vs healthy (paired within participant), disease contrasts."""
    inj = pmeans(df, ecols, "injured_PT", min_cells); hea = pmeans(df, ecols, "healthy_PT", min_cells)
    inj, hea = inj[inj.tech == "pooled"].set_index("participant"), hea[hea.tech == "pooled"].set_index("participant")
    both = inj.index.intersection(hea.index)
    rows = []
    for c in ecols:
        dif = inj.loc[both, c] - hea.loc[both, c]
        r = {"expert": c, "mean_injured": inj[c].mean(), "mean_healthy": hea[c].mean(),
             "injured_minus_healthy": dif.mean(), "p_paired": stats.wilcoxon(dif).pvalue, "n_paired": len(both)}
        for a, b in [("DKD", "HKD"), ("DKD", "Reference"), ("HKD", "Reference")]:
            h = hedges_g(inj.loc[inj.group == a, c], inj.loc[inj.group == b, c])
            r[f"g_{a}_vs_{b}"] = h["g"]; r[f"p_{a}_vs_{b}"] = h["p_wilcoxon"]
        rows.append(r)
    t = pd.DataFrame(rows)
    dk = t.loc[t.g_DKD_vs_HKD.idxmax(), "expert"]

    def role(r):
        if r.expert == dk:
            return "DKD-preferred"
        if r.injured_minus_healthy > 0 and r.p_paired < 0.05 and abs(r.g_DKD_vs_HKD) < 0.5:
            return "shared injury"
        if r.injured_minus_healthy < 0 and r.p_paired < 0.05:
            return "healthy-PT"
        return "other"
    t["role"] = t.apply(role, axis=1)
    return t, dk


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_gates"); bulk = Path(cfg["paths"]["bulk_dir"])
    mc = cfg["prep"]["min_cells_per_participant"]
    kp = pd.read_csv(bulk / "gates/kpmp_heldout_gates.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    ecols = [c for c in kp.columns if c.startswith("expert_")]
    roles, dk = expert_roles(kp, ecols, mc)
    roles.to_csv(out / "expert_roles_kpmp_heldout.tsv", sep="\t", index=False)
    # secondary readout (defined with the roles, before external testing): DKD-preferred share of injury
    # routing = p_dkd / (p_dkd + sum of shared-injury experts); insensitive to a global routing shift
    shared = roles.loc[roles.role == "shared injury", "expert"].tolist()

    def add_share(df):
        df["dkd_share_of_injury"] = df[dk] / (df[dk] + df[shared].sum(1) + 1e-9) if shared else np.nan
        return df
    kp = add_share(kp)
    pm_all, tests = [], []
    for state in ("injured_PT", "all_PT"):
        feats = ecols + ["p_dkd_vs_hkd", "dkd_share_of_injury"]
        pm = pmeans(kp, feats, state, mc)
        pm_all.append(pm.assign(cohort="KPMP_heldout"))
        for tech in ("pooled", "sc", "sn"):
            tests.append(group_tests(pm[pm.tech == tech], "group", KPMP_COMPS, feats,
                                     {"cohort": "KPMP_heldout", "cells": state, "tech": tech}))
    dom = [kp.groupby(["pt_state", "group", "tech"])[ecols].mean().reset_index().assign(cohort="KPMP_heldout")]
    for c, comps in EXT_COMPS.items():
        e = add_share(pd.read_csv(bulk / f"gates/{c}_gates.tsv.gz", sep="\t", index_col=0, dtype={"participant": str}))
        dom.append(e.groupby(["pt_state", "group", "tech"])[ecols].mean().reset_index().assign(cohort=c))
        for state in ("injured_PT", "all_PT"):
            pm = pmeans(e, feats, state, mc)
            pm_all.append(pm.assign(cohort=c))
            for tech in pm.tech.unique():
                tests.append(group_tests(pm[pm.tech == tech], "group", comps, feats,
                                         {"cohort": c, "cells": state, "tech": tech}))
    pd.concat(dom, ignore_index=True).to_csv(out / "routing_by_cohort.tsv", sep="\t", index=False)
    pm_all = pd.concat(pm_all, ignore_index=True)
    pm_all.to_csv(out / "participant_mean_gates.tsv", sep="\t", index=False)
    T = pd.concat(tests, ignore_index=True)
    T["dkd_preferred"] = T.feature == dk
    T.to_csv(out / "donor_level_tests.tsv", sep="\t", index=False)
    # diabetic-exposure gradient within KPMP held-out: HKD < HKD_withDM < DKD (Spearman, participants)
    grad = []
    for state in ("injured_PT", "all_PT"):
        p = pm_all[(pm_all.cohort == "KPMP_heldout") & (pm_all.cells == state) & (pm_all.tech == "pooled")]
        p = p[p.group.isin(["HKD", "HKD_withDM", "DKD"])]
        o = p.group.map({"HKD": 0, "HKD_withDM": 1, "DKD": 2})
        for c in ecols:
            r, pv = stats.spearmanr(o, p[c])
            grad.append({"cells": state, "feature": c, "rho": r, "p": pv, "n": len(p), "dkd_preferred": c == dk})
    pd.DataFrame(grad).to_csv(out / "kpmp_exposure_gradient.tsv", sep="\t", index=False)
    # per-seed robustness of the DKD-preferred expert in external cohorts
    ps = np.load(bulk / "gates/per_seed_matched.npz"); k = int(dk.split("_")[1]); rob = []
    for c, comps in EXT_COMPS.items():
        e = ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad", backed="r").obs[["participant", "group", "tech", "pt_state"]].astype(str)
        for si in range(ps[c].shape[0]):
            df = e.assign(w=ps[c][si][:, k].astype(float))
            pm = pmeans(df, ["w"], "injured_PT", mc)
            for tech in pm.tech.unique():
                for a, b in comps:
                    q = pm[pm.tech == tech]
                    rob.append({"cohort": c, "seed_index": si, "tech": tech, "comparison": f"{a}_vs_{b}",
                                **hedges_g(q.loc[q.group == a, "w"], q.loc[q.group == b, "w"])})
    pd.DataFrame(rob).to_csv(out / "dkd_expert_seed_robustness.tsv", sep="\t", index=False)
    info = {"dkd_preferred_expert": dk, "shared_injury_experts": shared, "selected_on": "KPMP held-out participants (injured PT, pooled tech)",
            "roles": roles.set_index("expert").role.to_dict()}
    (out / "dkd_expert.json").write_text(json.dumps(info, indent=2))
    show = T[T.feature.isin([dk, "dkd_share_of_injury"]) & (T.cells == "injured_PT")]
    print(roles.round(3).to_string()); print(show.round(3).to_string())
    write_provenance("expr_gates", [bulk / "gates/kpmp_heldout_gates.tsv.gz", bulk / "gates/gse211785_gates.tsv.gz",
                                    bulk / "gates/gse195460_gates.tsv.gz"],
                     sorted(out.glob("*.tsv")) + [out / "dkd_expert.json"], cfg["seed"], cfg, extra=info)


if __name__ == "__main__":
    main()
