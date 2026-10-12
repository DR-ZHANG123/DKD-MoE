"""03b_summarise: merge fold-group outputs of 03_train and summarise retrieval at participant level.

Section metrics -> participant means (per fold/seed/model) -> per participant mean over seeds ->
summary across participants (mean, sd); aligned vs shuffled-pair baseline compared per participant
(paired Wilcoxon over participants). Global-pool retrieval summarised over folds x seeds.
Outputs results/align_train/{retrieval_sections.tsv, retrieval_global.tsv, runs.tsv, training_curves.tsv,
retrieval_participants.tsv, retrieval_summary.tsv, retrieval_vs_baseline.tsv}; figures/source_data/align_retrieval_*.tsv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import evaluate as EV  # noqa: E402

RES = "align_train"


def merge(out: Path, stem: str) -> pd.DataFrame:
    fs = sorted(out.glob(f"{stem}_f*.tsv"))
    d = pd.concat([pd.read_csv(f, sep="\t", dtype={"participant": str}) for f in fs], ignore_index=True)
    d.to_csv(out / f"{stem}.tsv", sep="\t", index=False)
    return d


def main():
    cfg = C.load_cfg(); out = C.result_dir(RES)
    K = cfg["eval"]["topk"]
    R = merge(out, "retrieval_sections"); G = merge(out, "retrieval_global")
    cal = out / "retrieval_sections_external_calibrated.tsv"
    if cal.exists():
        R = pd.concat([R, pd.read_csv(cal, sep="\t", dtype={"participant": str})], ignore_index=True)
    merge(out, "runs"); merge(out, "training_curves")
    cols = [c for c in R.columns if c.startswith(("top", "chance", "median_rank"))]
    P = EV.by_participant(R, cols)                                   # per fold/seed/model/participant
    Pm = P.groupby(["model", "eval_set", "participant", "cohort", "group"])[cols].mean().reset_index()  # over seeds
    Pm.to_csv(out / "retrieval_participants.tsv", sep="\t", index=False)
    Pm.to_csv(C.src("align_retrieval_participants.tsv"), sep="\t", index=False)
    S = Pm.groupby(["model", "eval_set"])[cols].agg(["mean", "std"])
    S.columns = [f"{a}_{b}" for a, b in S.columns]
    S["n_participants"] = Pm.groupby(["model", "eval_set"]).size()
    # seed range of the participant-mean metric
    Ps = P.groupby(["model", "eval_set", "seed"])[[f"top{k}_img2expr" for k in K]].mean()
    for k in K:
        S[f"top{k}_img2expr_seed_min"] = Ps[f"top{k}_img2expr"].groupby(level=[0, 1]).min()
        S[f"top{k}_img2expr_seed_max"] = Ps[f"top{k}_img2expr"].groupby(level=[0, 1]).max()
    S = S.reset_index(); S.to_csv(out / "retrieval_summary.tsv", sep="\t", index=False)
    S.to_csv(C.src("align_retrieval_summary.tsv"), sep="\t", index=False)
    rows = []
    base = f"shuffled_{cfg['train']['primary_batch_mode']}"
    for mdl in sorted(Pm.model.unique()):
        if mdl == base:
            continue
        for es in Pm.eval_set.unique():
            a = Pm[(Pm.model == mdl) & (Pm.eval_set == es)].set_index("participant")
            b = Pm[(Pm.model == base) & (Pm.eval_set == es)].set_index("participant")
            j = a.index.intersection(b.index)
            for met in [f"top{k}_img2expr" for k in K] + [f"top{k}_expr2img" for k in K] + \
                       ["median_rank_pct_img2expr", "top1_within150um_img2expr"]:
                x, y = a.loc[j, met], b.loc[j, met]
                ch = a.loc[j, met.replace("_img2expr", "").replace("_expr2img", "").replace("top", "chance_top")] \
                    if met.startswith("top") and "within" not in met else None
                rows.append({"model": mdl, "baseline": base, "eval_set": es, "metric": met, "n_participants": len(j),
                             "mean_model": x.mean(), "mean_baseline": y.mean(),
                             "mean_chance": ch.mean() if ch is not None else np.nan,
                             "frac_participants_model_better": float((x > y).mean()) if "median_rank" not in met else float((x < y).mean()),
                             "wilcoxon_p": stats.wilcoxon(x, y).pvalue if len(j) >= 5 else np.nan})
    V = pd.DataFrame(rows); V.to_csv(out / "retrieval_vs_baseline.tsv", sep="\t", index=False)
    Gs = G.groupby(["model", "eval_set"])[[c for c in G.columns if c.startswith(("top", "chance"))]].agg(["mean", "min", "max"])
    Gs.columns = [f"{a}_{b}" for a, b in Gs.columns]; Gs = Gs.reset_index()
    Gs.to_csv(out / "retrieval_global_summary.tsv", sep="\t", index=False)
    Gs.to_csv(C.src("align_retrieval_global.tsv"), sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(S[["model", "eval_set", "n_participants"] + [f"top{k}_img2expr_mean" for k in K] + [f"chance_top{k}_mean" for k in K]
            + ["top1_within150um_img2expr_mean", "chance_top1_within150um_mean", "median_rank_pct_img2expr_mean"]].round(4).to_string())
    print(V[V.metric.str.startswith("top")].round(4).to_string())
    print(Gs.round(4).to_string())
    C.provenance(RES, sorted(out.glob("*_f*.tsv")), sorted(out.glob("retrieval_*.tsv")) + [out / "runs.tsv"],
                 cfg["seed"], extra={"merged_from": [f.name for f in sorted(out.glob("*_f*.tsv"))]})


if __name__ == "__main__":
    main()
