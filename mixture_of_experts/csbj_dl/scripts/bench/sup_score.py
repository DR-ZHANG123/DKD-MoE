"""Participant scores and summary for the supervised single-expert baseline (sup_single.py), evaluated with
the harmonised rules of harm_score.py / harm_summary.py.

Scores per fold model (injured-PT cells of qualifying participant x technology samples):
  condition_head : raw mean DKD-vs-HKD log-odds over injured cells, then mean over technologies (as the
                   final model's readout); KPMP = held-out partition, externals = mean over the five fold models
  latent_LR      : logistic regression on the standardised participant mean latent (as for every method)
Seed-mean scores give AUC / Hedges g with 95% CI; KPMP DKD vs HKD is also compared with the final model's
participant readout (results/expr_readout) by a paired stratified participant bootstrap of the AUC
difference. Seed agreement of participant scores (ICC(2,1), mean pairwise Spearman over the five seeds) is
computed for these models and, from results/bench_harmonised, for the final model's condition head.
Outputs: results/bench_supervised_single/{donor_scores.tsv, summary.tsv, paired_vs_moe_final.tsv,
         seed_agreement.tsv} + PROVENANCE.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "expr"))
from bench_metrics import hedges_g  # noqa: E402
from bench_score import latent_lr_score, participant_table  # noqa: E402
from harm_score import sample_meta, cell_means, CASE, CTRL  # noqa: E402
from harm_summary import tidy, contrast_rows, paired_auc  # noqa: E402
from stability_compare import seed_agreement  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import CDL, load_cfg, result_dir, write_provenance  # noqa: E402

VARIANTS = {"supervised_single_z": "Supervised single expert (condition head on latent mean)",
            "supervised_single_h": "Supervised single expert (condition head on encoder output)"}
CARRIED = {"moe_final_warm/condition_head": "Mixture of experts, final (harmonised condition head)",
           "moe_warm_K1/latent_LR": "Single expert, no condition head (latent LR)",
           "moe_warm_no_cond/condition_head": "Mixture of experts, no condition classifier"}
FINAL = "moe_final/participant_readout"       # results/expr_readout seed-mean participant readout


def final_readout() -> pd.DataFrame:
    """Final model participant readout (results/expr_readout): KPMP held-out and external cohorts, per seed and
    seed-mean, in the long format of harm_summary.tidy."""
    e = pd.read_csv(CDL / "results/expr_readout/participant_scores.tsv", sep="\t", dtype={"participant": str})
    ds = {"KPMP_heldout": "KPMP", "gse211785": "GSE211785", "gse195460": "GSE195460"}
    e = e[(e.tech == "pooled") & e.cohort.isin(ds)].assign(dataset=lambda x: x.cohort.map(ds))
    seeds = [c for c in e.columns if c.startswith("seed")]
    rows = [e.assign(seed=s.replace("seed", ""), score=e[s]) for s in seeds] + [e.assign(seed="mean", score=e.logodds)]
    return pd.concat(rows)[["seed", "dataset", "participant", "group", "score"]].assign(method=FINAL,
                                                                                      score_name="condition_head")


def score_variant(v, ck_root, obs, meta, std, cfg):
    rows = []
    for npz in sorted((ck_root / f"warm_{v}").glob("seed*_fold*/injured_cells.npz")):
        seed, fold = map(int, npz.parent.name.replace("seed", "").split("_fold"))
        z = dict(np.load(npz))
        fit = ((meta.dataset == "KPMP") & (meta.fold != fold) & meta.condition.isin([CASE, CTRL])).values
        s_head = cell_means(z["logodds"][:, None], obs, meta)[0]
        s_lat = latent_lr_score(cell_means(z["emb"], obs, meta), meta, std, CASE, cfg["scvi"]["lr_C"], seed, fit)
        rows.append(participant_table(s_head, meta, std, v, "condition_head", seed, fold, standardise=False))
        rows.append(participant_table(s_lat.reindex(meta.index), meta, std, v, "latent_LR", seed, fold))
    return rows


def main():
    cfg = load_cfg("bench"); out = result_dir("bench_supervised_single"); ck_root = Path(cfg["paths"]["ckpt"])
    obs = pd.read_csv(ck_root / "harmonised_cells.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    meta = sample_meta(obs); std = (meta.dataset == "KPMP").values
    rows, n_models = [], {}
    for v in VARIANTS:
        r = score_variant(v, ck_root, obs, meta, std, cfg); rows += r; n_models[v] = len(r) // 2
    D = pd.concat(rows).rename(columns={"condition": "group"})
    H = pd.read_csv(CDL / "results/bench_harmonised/donor_scores.tsv", sep="\t", dtype={"participant": str})
    H = H[(H.method + "/" + H.score_name).isin(CARRIED)]
    T = tidy(pd.concat([D, H], ignore_index=True)); T["method"] = T.method + "/" + T.score_name
    R = final_readout(); A = pd.concat([T, R], ignore_index=True)
    A.to_csv(out / "donor_scores.tsv", sep="\t", index=False)
    S = []
    for m, d in A[A.seed == "mean"].groupby("method"):
        S.append({"method": m, **dict(contrast_rows(d, cfg))})
    S = pd.DataFrame(S); S.to_csv(out / "summary.tsv", sep="\t", index=False)
    rk = R[(R.seed == "mean") & (R.dataset == "KPMP") & R.group.isin(["DKD", "HKD"])].set_index("participant")
    P = []
    for m, d in A[(A.seed == "mean") & (A.dataset == "KPMP") & A.group.isin(["DKD", "HKD"])].groupby("method"):
        d = d.set_index("participant"); cm = rk.index.intersection(d.index); grp = rk.group[cm]
        dd, lo, hi = paired_auc(rk.score[cm], d.score[cm], grp, cfg["metrics"]["n_boot"], 1)
        g, gl, gh = hedges_g(d.score[cm][grp == "DKD"], d.score[cm][grp == "HKD"])
        P.append({"method": m, "n_DKD": int((grp == "DKD").sum()), "n_HKD": int((grp == "HKD").sum()),
                  "auc_diff_vs_moe_final": dd, "ci_lo": lo, "ci_hi": hi, "g_method": g, "g_lo": gl, "g_hi": gh})
    P = pd.DataFrame(P); P.to_csv(out / "paired_vs_moe_final.tsv", sep="\t", index=False)
    G = []
    for (m, ds), d in A[A.seed != "mean"].groupby(["method", "dataset"]):
        W = d.pivot_table(index="participant", columns="seed", values="score").dropna()
        if W.shape[1] >= 2:
            G.append({"method": m, "dataset": ds, "n_seeds": W.shape[1], **seed_agreement(W)})
    G = pd.DataFrame(G); G.to_csv(out / "seed_agreement.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    cols = ["method", "KPMP_DKD_vs_HKD_auc", "KPMP_DKD_vs_HKD_auc_lo", "KPMP_DKD_vs_HKD_auc_hi", "KPMP_DKD_vs_HKD_g",
            "GSE211785_DKD_vs_HKD_g", "GSE211785_DKD_vs_HKD_g_lo", "GSE211785_DKD_vs_HKD_g_hi",
            "GSE195460_DKD_vs_Control_g", "GSE195460_DKD_vs_Control_g_lo", "GSE195460_DKD_vs_Control_g_hi"]
    print(S[cols].round(3).to_string()); print(P.round(3).to_string()); print(G.round(3).to_string())
    write_provenance("bench_supervised_single", [ck_root / "harmonised_cells.tsv.gz",
                                                 CDL / "results/bench_harmonised/donor_scores.tsv",
                                                 CDL / "results/expr_readout/participant_scores.tsv"],
                     [out / f for f in ("donor_scores.tsv", "summary.tsv", "paired_vs_moe_final.tsv",
                                        "seed_agreement.tsv")], cfg["seed"], cfg,
                     extra={"models_per_variant": n_models, "labels": {**VARIANTS, **CARRIED, FINAL: "final model participant readout (results/expr_readout)"},
                            "external_rule": "mean over the five fold models"})


if __name__ == "__main__":
    main()
