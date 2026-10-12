"""Nested (leave-one-fold-out) check of the K / condition-weight selection on the existing CV grid runs.

For each outer fold f, the selection rule of cv_summary.py (eligibility on routing specificity; ranks over all
grid settings of held-out NLL per count, seed ARI and participant DKD-vs-HKD AUC; lowest rank sum among
eligible settings, ties -> smaller K) is applied to the held-out predictions of the other four folds only.
The chosen setting is then scored on fold f. Metric definitions follow cv_summary.metrics() on the chosen cells.
Outputs: results/expr_nested_selection/{inner_tables.tsv, outer_choice.tsv, fold_auc_by_setting.tsv,
         outer_participant_scores.tsv, summary.tsv}
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import auc  # noqa: E402


def load(cfg, prefix, K, wc, s, f):
    return np.load(Path(cfg["paths"]["ckpt_dir"]) / prefix / f"{prefix}_K{K}_wc{wc}_s{s}_f{f}.npz")


def part_scores(obs, z, iD, iH, min_cells):
    o = obs.iloc[z["idx"]][["participant", "group", "pt_state"]].astype(str).reset_index(drop=True)
    o["nll"] = z["nll"]
    inj = (o.pt_state == "injured_PT").values
    cp = z["cond_prob"][inj]
    oi = o[inj].assign(pD=cp[:, iD] / (cp[:, iD] + cp[:, iH] + 1e-9))
    pm = oi.groupby("participant").agg(group=("group", "first"), n=("pD", "size"), pD=("pD", "mean"))
    return o.groupby("participant").nll.mean(), pm[pm.n >= min_cells]


def metrics(Z: dict, folds, seeds, K):
    """Z[(seed, fold)] -> (nll_by_participant, participant pD table, p). Metrics on the union of `folds`."""
    nll, aucs = [], []
    for s in seeds:
        nll.append(pd.concat([Z[(s, f)][0] for f in folds]).mean())
        pm = pd.concat([Z[(s, f)][1] for f in folds])
        aucs.append(auc(pm.pD[pm.group == "DKD"], pm.pD[pm.group == "HKD"]))
    P = {s: np.concatenate([Z[(s, f)][2] for f in folds]) for s in seeds}
    p = P[seeds[0]]
    ent, frac = [], []
    for s in seeds:
        pp = np.clip(P[s], 1e-12, 1)
        ent.append(float((-(pp * np.log(pp)).sum(1)).mean() / np.log(K))); frac.append(float((P[s].max(1) > 0.5).mean()))
    ari = np.mean([adjusted_rand_score(P[a].argmax(1), P[b].argmax(1)) for a, b in itertools.combinations(seeds, 2)])
    return {"nll": float(np.mean(nll)), "auc_dkd_hkd": float(np.mean(aucs)), "stability_ari": float(ari),
            "entropy_over_logK": float(np.mean(ent)), "frac_top1_gt_0.5": float(np.mean(frac)), "n_cells": len(p)}


def select(tab, cv):
    tab = tab.copy()
    tab["eligible"] = (tab["frac_top1_gt_0.5"] >= cv["min_frac_top1_gt_half"]) & (tab.entropy_over_logK <= cv["max_entropy_over_logK"])
    tab["rank_nll"] = tab.nll.rank(ascending=True)
    tab["rank_stability"] = tab.stability_ari.rank(ascending=False)
    tab["rank_separation"] = tab.auc_dkd_hkd.rank(ascending=False)
    tab["rank_sum"] = tab[["rank_nll", "rank_stability", "rank_separation"]].sum(1)
    return tab.sort_values(["eligible", "rank_sum", "K"], ascending=[False, True, True]).reset_index(drop=True)


def main():
    cfg = load_cfg("expr"); cv = cfg["cv"]; out = result_dir("expr_nested_selection")
    prefix = cfg["patient_level_models"]["nested_selection"]["prefix"]
    obs = ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs.copy()
    cls = cfg["condition"]["classes"]; iD, iH = cls.index("DKD"), cls.index("HKD")
    seeds, nf = cv["seeds_for_stability"], cfg["n_folds"]
    grid = list(itertools.product(cv["K_grid"], [float(w) for w in cv["w_cond_grid"]]))
    Z = {}
    for K, wc in grid:
        for s in seeds:
            for f in range(nf):
                z = load(cfg, prefix, K, wc, s, f)
                nll, pm = part_scores(obs, z, iD, iH, cv["min_injured_cells"])
                Z[(K, wc, s, f)] = (nll, pm, z["p"])
    sub = lambda K, wc: {(s, f): Z[(K, wc, s, f)] for s in seeds for f in range(nf)}
    # all-fold selection (reproduces expr_cv/cv_table.tsv) and per-fold AUC of every setting
    full = select(pd.DataFrame([{"K": K, "w_cond": wc, **metrics(sub(K, wc), range(nf), seeds, K)} for K, wc in grid]), cv)
    full_choice = (int(full.K[0]), float(full.w_cond[0]))
    fold_auc = []
    for K, wc in grid:
        for f in range(nf):
            m = metrics(sub(K, wc), [f], seeds, K)
            fold_auc.append({"K": K, "w_cond": wc, "fold": f, "auc_dkd_hkd": m["auc_dkd_hkd"],
                             "eligible_all_folds": bool(full.set_index(["K", "w_cond"]).loc[(K, wc), "eligible"])})
    FA = pd.DataFrame(fold_auc)
    inner, choice, scores = [], [], []
    for f in range(nf):
        others = [g for g in range(nf) if g != f]
        t = select(pd.DataFrame([{"K": K, "w_cond": wc, **metrics(sub(K, wc), others, seeds, K)} for K, wc in grid]), cv)
        inner.append(t.assign(outer_fold=f))
        K, wc = int(t.K[0]), float(t.w_cond[0])
        m = metrics(sub(K, wc), [f], seeds, K)
        fa = FA[(FA.fold == f) & FA.eligible_all_folds]
        choice.append({"outer_fold": f, "chosen_K": K, "chosen_w_cond": wc, "chosen_eligible": bool(t.eligible[0]),
                       "same_as_all_fold_choice": (K, wc) == full_choice, "outer_auc_chosen": m["auc_dkd_hkd"],
                       "outer_auc_all_fold_choice": FA[(FA.fold == f) & (FA.K == full_choice[0]) & (FA.w_cond == full_choice[1])].auc_dkd_hkd.iloc[0],
                       "outer_auc_min_eligible": fa.auc_dkd_hkd.min(), "outer_auc_max_eligible": fa.auc_dkd_hkd.max(),
                       "n_dkd": int(sum((Z[(K, wc, s, f)][1].group == "DKD").sum() for s in seeds[:1])),
                       "n_hkd": int(sum((Z[(K, wc, s, f)][1].group == "HKD").sum() for s in seeds[:1]))})
        per_seed = [Z[(K, wc, s, f)][1].pD.rename(f"seed{s}") for s in seeds]
        sc = pd.concat(per_seed, axis=1).join(Z[(K, wc, seeds[0], f)][1].group)
        scores.append(sc.assign(outer_fold=f, K=K, w_cond=wc).reset_index())
    C = pd.DataFrame(choice); S = pd.concat(scores, ignore_index=True)
    S["pD_seed_mean"] = S[[f"seed{s}" for s in seeds]].mean(1)
    pooled = lambda col: auc(S[col][S.group == "DKD"], S[col][S.group == "HKD"])
    ref = {(s, f): Z[(full_choice[0], full_choice[1], s, f)] for s in seeds for f in range(nf)}
    ref_pm = pd.concat([pd.concat([ref[(s, f)][1].pD.rename(f"seed{s}") for s in seeds], axis=1).join(ref[(seeds[0], f)][1].group)
                        for f in range(nf)])
    ref_pm["m"] = ref_pm[[f"seed{s}" for s in seeds]].mean(1)
    elig = FA[FA.eligible_all_folds].groupby(["K", "w_cond"]).auc_dkd_hkd.mean()
    summ = [{"quantity": "nested_pooled_outer_auc_mean_of_seed_aucs", "value": float(np.mean([pooled(f"seed{s}") for s in seeds]))},
            {"quantity": "nested_pooled_outer_auc_seed_mean_score", "value": pooled("pD_seed_mean")},
            {"quantity": "nested_mean_of_fold_aucs", "value": C.outer_auc_chosen.mean()},
            {"quantity": "all_fold_choice", "value": f"K={full_choice[0]}, w_cond={full_choice[1]}"},
            {"quantity": "all_fold_choice_pooled_auc_mean_of_seed_aucs", "value": float(full.auc_dkd_hkd[0])},
            {"quantity": "all_fold_choice_pooled_auc_seed_mean_score", "value": auc(ref_pm.m[ref_pm.group == "DKD"], ref_pm.m[ref_pm.group == "HKD"])},
            {"quantity": "all_fold_choice_mean_of_fold_aucs", "value": C.outer_auc_all_fold_choice.mean()},
            {"quantity": "pooled_auc_range_eligible_settings", "value": f"{full[full.eligible].auc_dkd_hkd.min():.4f}-{full[full.eligible].auc_dkd_hkd.max():.4f}"},
            {"quantity": "pooled_auc_range_all_settings", "value": f"{full.auc_dkd_hkd.min():.4f}-{full.auc_dkd_hkd.max():.4f}"},
            {"quantity": "mean_fold_auc_range_eligible_settings", "value": f"{elig.min():.4f}-{elig.max():.4f}"},
            {"quantity": "n_outer_folds_choice_equal_all_fold_choice", "value": int(C.same_as_all_fold_choice.sum())}]
    pd.concat(inner).to_csv(out / "inner_tables.tsv", sep="\t", index=False)
    C.to_csv(out / "outer_choice.tsv", sep="\t", index=False); FA.to_csv(out / "fold_auc_by_setting.tsv", sep="\t", index=False)
    S.to_csv(out / "outer_participant_scores.tsv", sep="\t", index=False)
    full.to_csv(out / "all_fold_table_recomputed.tsv", sep="\t", index=False)
    pd.DataFrame(summ).to_csv(out / "summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(full.round(4).to_string()); print(C.round(4).to_string()); print(FA.pivot_table(index=["K", "w_cond"], columns="fold", values="auc_dkd_hkd").round(3))
    print(pd.DataFrame(summ).to_string())
    write_provenance("expr_nested_selection", [result_dir("expr_cv") / "cv_table.tsv"],
                     [out / f for f in ("inner_tables.tsv", "outer_choice.tsv", "fold_auc_by_setting.tsv",
                                        "outer_participant_scores.tsv", "all_fold_table_recomputed.tsv", "summary.tsv")],
                     cfg["seed"], cfg, extra={"prefix": prefix, "seeds": seeds})


if __name__ == "__main__":
    main()
