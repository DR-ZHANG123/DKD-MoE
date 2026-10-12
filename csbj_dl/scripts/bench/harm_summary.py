"""Harmonised benchmark summary (same evaluation as the expr readout).

Reads results/bench_harmonised/donor_scores.tsv and the final readout
results/expr_readout/participant_scores.tsv (KPMP held-out = out-of-part, seed-mean log-odds; externals
from the final models). Writes results/bench_summary/:
  benchmark.tsv          method, seed, fold, metric, value (seed = 'mean' for seed-mean scores)
  summary_table.tsv      primary table: seed-mean scores, g / AUC with 95% CI, plus mean ± sd over seeds
  paired_vs_moe_final.tsv  KPMP held-out DKD vs HKD: AUC difference vs the final MoE readout (paired
                         participant bootstrap) and Hedges g side by side
  participant_lists.tsv  participants per cohort x contrast, with the check against the expr lists
  donor_scores.tsv       harmonised participant scores (KPMP held-out pooled over parts; externals
                         averaged over fold models), per seed and seed-mean
  mixing.tsv             technology mixing (iLISI, same-technology neighbour share) carried from the
                         per-method summary (all PT cells of held-out participants)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_metrics import hedges_g, auc_ci  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import CDL, load_cfg, result_dir, write_provenance  # noqa: E402

CONTRASTS = {"KPMP": [("DKD", "HKD"), ("DKD", "HKD_withDM"), ("HKD_withDM", "HKD"), ("DKD", "Reference")],
             "GSE211785": [("DKD", "HKD"), ("DKD", "Control"), ("HKD", "Control")],
             "GSE195460": [("DKD", "Control")]}
REPORTED = "MoE final model readout"


def tidy(D):
    """KPMP: held-out rows pooled over parts; external: mean over fold models. Adds seed-mean rows."""
    k = D[D.role == "test"][["method", "score_name", "seed", "dataset", "participant", "group", "score"]]
    e = (D[D.role == "external"].groupby(["method", "score_name", "seed", "dataset", "participant", "group"],
                                        as_index=False).score.mean())
    T = pd.concat([k, e]); T["seed"] = T.seed.astype(str)
    M = T.groupby(["method", "score_name", "dataset", "participant", "group"], as_index=False).score.mean()
    return pd.concat([T, M.assign(seed="mean")], ignore_index=True)


def reported():
    e = pd.read_csv(CDL / "results/expr_readout/participant_scores.tsv", sep="\t", dtype={"participant": str})
    e = e[e.tech == "pooled"]
    ds = {"KPMP_heldout": "KPMP", "gse211785": "GSE211785", "gse195460": "GSE195460"}
    e = e[e.cohort.isin(ds)].assign(dataset=lambda x: x.cohort.map(ds))
    seeds = [c for c in e.columns if c.startswith("seed")]
    rows = [e.assign(seed=s.replace("seed", ""), score=e[s]) for s in seeds] + [e.assign(seed="mean", score=e.logodds)]
    R = pd.concat(rows)[["seed", "dataset", "participant", "group", "score"]]
    return R.assign(method=REPORTED, score_name="condition_head")


def contrast_rows(d, cfg, seed_for_boot=1):
    out = []
    for ds, comps in CONTRASTS.items():
        for a, b in comps:
            x = d[(d.dataset == ds) & (d.group == a)].score.dropna().values
            y = d[(d.dataset == ds) & (d.group == b)].score.dropna().values
            if len(x) < 2 or len(y) < 2:
                continue
            key = f"{ds}_{a}_vs_{b}"
            g, gl, gh = hedges_g(x, y, cfg["metrics"]["ci"])
            au, al, ah = auc_ci(x, y, cfg["metrics"]["n_boot"], seed_for_boot, cfg["metrics"]["ci"])
            out += [(f"{key}_g", g), (f"{key}_g_lo", gl), (f"{key}_g_hi", gh), (f"{key}_auc", au),
                    (f"{key}_auc_lo", al), (f"{key}_auc_hi", ah), (f"{key}_n1", len(x)), (f"{key}_n2", len(y))]
    return out


def paired_auc(ref: pd.Series, oth: pd.Series, grp: pd.Series, n_boot: int, seed: int):
    """AUC(other) - AUC(reference), DKD vs HKD, same participants; stratified participant bootstrap."""
    def auc(s, pos, neg):
        C = (s[pos][:, None] > s[neg][None, :]) + 0.5 * (s[pos][:, None] == s[neg][None, :])
        return C.mean()
    pos = np.where(grp.values == "DKD")[0]; neg = np.where(grp.values == "HKD")[0]
    r, o = ref.values, oth.values
    d0 = auc(o, pos, neg) - auc(r, pos, neg)
    rng = np.random.default_rng(seed); bs = np.empty(n_boot)
    for i in range(n_boot):
        p = rng.choice(pos, len(pos)); n = rng.choice(neg, len(neg))
        bs[i] = auc(o, p, n) - auc(r, p, n)
    return d0, np.quantile(bs, 0.025), np.quantile(bs, 0.975)


def participant_lists(T, R):
    rows = []
    for ds, comps in CONTRASTS.items():
        for a, b in comps:
            for grp in (a, b):
                mine = T[(T.seed == "mean") & (T.dataset == ds) & (T.group == grp)]
                ref = set(R[(R.seed == "mean") & (R.dataset == ds) & (R.group == grp)].participant)
                for m, d in mine.groupby("method"):
                    s = set(d.participant)
                    rows.append({"dataset": ds, "contrast": f"{a}_vs_{b}", "group": grp, "method": m, "n": len(s),
                                 "n_expr": len(ref), "identical_to_expr": s == ref,
                                 "participants": ",".join(sorted(s))})
    return pd.DataFrame(rows)


def main():
    cfg = load_cfg("bench"); out = result_dir("bench_summary")
    D = pd.read_csv(CDL / "results/bench_harmonised/donor_scores.tsv", sep="\t", dtype={"participant": str})
    T = tidy(D); R = reported()
    T["method"] = T.method + "/" + T.score_name; R["method"] = R.method
    A = pd.concat([T, R], ignore_index=True)
    A.to_csv(out / "donor_scores.tsv", sep="\t", index=False)
    rows = []
    for (m, s), d in A.groupby(["method", "seed"]):
        for k, v in contrast_rows(d, cfg):
            rows.append((m, s, "oof" if k.startswith("KPMP") else "ens", k, v))
    B = pd.DataFrame(rows, columns=["method", "seed", "fold", "metric", "value"])
    # paired comparison with the final readout on KPMP held-out (seed-mean scores)
    rk = R[(R.seed == "mean") & (R.dataset == "KPMP") & R.group.isin(["DKD", "HKD"])].set_index("participant")
    P = []
    for m, d in A[(A.seed == "mean") & (A.dataset == "KPMP") & A.group.isin(["DKD", "HKD"])].groupby("method"):
        d = d.set_index("participant"); common = rk.index.intersection(d.index)
        dd, lo, hi = paired_auc(rk.score[common], d.score[common], rk.group[common], cfg["metrics"]["n_boot"], 1)
        g, gl, gh = hedges_g(d.score[common][rk.group[common] == "DKD"], d.score[common][rk.group[common] == "HKD"])
        g0, g0l, g0h = hedges_g(rk.score[common][rk.group[common] == "DKD"], rk.score[common][rk.group[common] == "HKD"])
        P.append({"method": m, "n_DKD": int((rk.group[common] == "DKD").sum()), "n_HKD": int((rk.group[common] == "HKD").sum()),
                  "auc_diff_vs_moe_final": dd, "ci_lo": lo, "ci_hi": hi, "g_method": g, "g_lo": gl, "g_hi": gh,
                  "g_moe_final": g0, "g_moe_final_lo": g0l, "g_moe_final_hi": g0h})
    P = pd.DataFrame(P); P.to_csv(out / "paired_vs_moe_final.tsv", sep="\t", index=False)
    PL = participant_lists(A[A.method != REPORTED], R); PL.to_csv(out / "participant_lists.tsv", sep="\t", index=False)
    # technology mixing from the per-method summary
    ub = CDL / "results/bench_method_summary/benchmark.tsv"
    Mx = pd.read_csv(ub, sep="\t") if ub.exists() else pd.DataFrame(columns=B.columns)
    Mx = Mx[Mx.metric.str.startswith(("iLISI", "kNN30"))]
    Mx.to_csv(out / "mixing.tsv", sep="\t", index=False)
    B = pd.concat([B, Mx.assign(fold=Mx.fold.astype(str))], ignore_index=True)
    B.to_csv(out / "benchmark.tsv", sep="\t", index=False)
    # summary: seed-mean estimates with CI + mean ± sd over individual seeds
    sm = B[B.seed.astype(str) == "mean"].pivot_table(index="method", columns="metric", values="value")
    per = B[~B.seed.astype(str).isin(["mean", "all"]) & ~B.metric.str.startswith(("iLISI", "kNN30"))]
    sd = per.groupby(["method", "metric"]).value.agg(["mean", "std", "count"]).unstack("metric")
    mix = Mx.groupby(["method", "metric"]).value.mean().unstack("metric") if len(Mx) else pd.DataFrame()
    S = sm.join(sd["std"].add_suffix("_sd_over_seeds")).join(sd["count"].iloc[:, :1].rename(
        columns=lambda c: "n_seeds"))
    S.to_csv(out / "summary_table.tsv", sep="\t")
    if len(mix):
        mix.to_csv(out / "mixing_summary.tsv", sep="\t")
    pd.set_option("display.width", 250)
    cols = ["KPMP_DKD_vs_HKD_auc", "KPMP_DKD_vs_HKD_g", "GSE211785_DKD_vs_HKD_g", "GSE195460_DKD_vs_Control_g"]
    print(S[[c for c in cols if c in S]].round(2).to_string()); print(P.round(3).to_string())
    print(PL.groupby(["dataset", "contrast", "group"]).identical_to_expr.all())
    write_provenance("bench_summary", [CDL / "results/bench_harmonised/donor_scores.tsv",
                                       CDL / "results/expr_readout/participant_scores.tsv"],
                     [out / f for f in ["benchmark.tsv", "summary_table.tsv", "paired_vs_moe_final.tsv",
                                        "participant_lists.tsv", "donor_scores.tsv", "mixing.tsv"]],
                     cfg["seed"], cfg, extra={"evaluation": "harmonised with expr readout",
                                              "external_rule": "mean over the five fold models for every method"})


if __name__ == "__main__":
    main()
