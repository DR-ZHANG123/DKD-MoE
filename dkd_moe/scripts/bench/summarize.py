"""Donor-level benchmark summary across all bench methods.

Reads results/bench_<method>/donor_scores.tsv and lisi.tsv (development runs excluded) and writes
results/bench_method_summary/{benchmark.tsv, donor_scores.tsv, summary_table.tsv, PROVENANCE.json}.
benchmark.tsv columns: method (method/score), seed, fold, metric, value.
  fold = 0..4   held-out KPMP fold (and that fold's model applied to external cohorts)
  fold = oof    KPMP held-out scores pooled over the five folds
  fold = ens    external cohorts scored by the mean of the five fold models
Usage: python summarize.py [--include-dev]
"""
from __future__ import annotations
import argparse, itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_metrics import hedges_g, auc_ci  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import CDL, load_cfg, result_dir, write_provenance  # noqa: E402

CONTRASTS = [("KPMP", "DKD", "HKD"), ("GSE211785", "DKD", "HKD"), ("GSE211785", "DKD", "Control"),
             ("GSE195460", "DKD", "Control"), ("KPMP", "DKD", "Reference"), ("KPMP", "HKD", "Reference")]


def contrast_metrics(d: pd.DataFrame, ds: str, a: str, b: str, cfg: dict, seed: int) -> list[tuple[str, float]]:
    x = d[(d.dataset == ds) & (d.condition == a)].score.dropna().values
    y = d[(d.dataset == ds) & (d.condition == b)].score.dropna().values
    if len(x) < 2 or len(y) < 2:
        return []
    m = cfg["metrics"]; key = f"{ds}_{a}vs{b}"
    g, gl, gh = hedges_g(x, y, m["ci"])
    au, al, ah = auc_ci(x, y, m["n_boot"], seed, m["ci"])
    return [(f"{key}_g", g), (f"{key}_g_lo", gl), (f"{key}_g_hi", gh), (f"{key}_auc", au),
            (f"{key}_auc_lo", al), (f"{key}_auc_hi", ah), (f"{key}_n_case", len(x)), (f"{key}_n_ctrl", len(y))]


def load_all(include_dev: bool, root: Path):
    D, L = [], []
    for d in sorted(root.glob("bench_*")):
        if d.name == "bench_method_summary" or (d.name.endswith("_dev") and not include_dev):
            continue
        if (d / "donor_scores.tsv").exists():
            D.append(pd.read_csv(d / "donor_scores.tsv", sep="\t", dtype={"participant": str}))
        if (d / "lisi.tsv").exists() and (d / "lisi.tsv").stat().st_size > 1:
            L.append(pd.read_csv(d / "lisi.tsv", sep="\t"))
    D = pd.concat(D)
    # one expert = constant routing: routing- and condition-head scores are structurally undefined
    D = D[~((D.method == "moe_K1") & D.score_name.isin(["MoE_gate_LR", "MoE_condition_head"]))]
    D["method"] = D.method + "/" + D.score_name
    return D, (pd.concat(L) if L else pd.DataFrame())


def knn_purity(cfg) -> list[tuple]:
    """Share of each held-out cell's 30 nearest neighbours from the same technology (saved test
    embeddings); complements the median iLISI, which saturates at 0 once most neighbourhoods are pure.
    Balanced sc/sn mixing would give about the sc (or sn) share, i.e. ~0.5."""
    from sklearn.neighbors import NearestNeighbors
    rows = []
    for f in sorted(Path(cfg["paths"]["ckpt"]).glob("*/seed*_fold*/test_embedding.npz")):
        name = f.parents[1].name
        if name.startswith("dev") or "_dev" in name:
            continue
        m = name.rsplit("_s", 1)[0]
        seed, fold = f.parent.name.replace("seed", "").split("_fold")
        d = np.load(f); E, t, st = d["emb"], d["tech"], d["pt_state"]
        nn = NearestNeighbors(n_neighbors=31).fit(E).kneighbors(E, return_distance=False)[:, 1:]
        same = (t[nn] == t[:, None]).mean(1)
        rows.append((m, int(seed), int(fold), "kNN30_same_tech", float(same.mean())))
        for s_ in ("healthy_PT", "injured_PT"):
            rows.append((m, int(seed), int(fold), f"kNN30_same_tech_{s_}", float(same[st == s_].mean())))
    return rows


def ensemble(D: pd.DataFrame) -> pd.DataFrame:
    e = D[D.role == "external"]
    e = (e.groupby(["method", "score_name", "seed", "dataset", "participant", "condition"], as_index=False)
         .agg(score=("score", "mean"), raw=("raw", "mean"), n_fold_models=("fold", "nunique")))
    e["fold"] = "ens"; e["role"] = "external"
    return e


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--include-dev", action="store_true")
    ap.add_argument("--root", default=None, help="folder holding bench_<method> dirs (default dkd_moe/results)")
    a = ap.parse_args()
    cfg = load_cfg("bench")
    root = Path(a.root) if a.root else CDL / "results"
    out = (root / "bench_method_summary") if a.root else result_dir("bench_method_summary"); out.mkdir(exist_ok=True)
    D, L = load_all(a.include_dev, root)
    E = ensemble(D)
    rows = []
    for (m, s), d in D.groupby(["method", "seed"]):
        for f, df in d.groupby("fold"):
            te = df[df.role == "test"]
            for k, v in contrast_metrics(te, "KPMP", "DKD", "HKD", cfg, s):
                rows.append((m, s, f, k, v))
            ex = df[df.role == "external"]
            for ds, x, y in CONTRASTS[1:4]:
                for k, v in contrast_metrics(ex, ds, x, y, cfg, s):
                    rows.append((m, s, f, k, v))
        oof = d[d.role == "test"]
        for ds, x, y in [c for c in CONTRASTS if c[0] == "KPMP"]:
            for k, v in contrast_metrics(oof, ds, x, y, cfg, s):
                rows.append((m, s, "oof", k, v))
    for (m, s), d in E.groupby(["method", "seed"]):
        for ds, x, y in CONTRASTS[1:4]:
            for k, v in contrast_metrics(d, ds, x, y, cfg, s):
                rows.append((m, s, "ens", k, v))
    # stability across seeds: mean pairwise Spearman of donor scores
    for m, d in pd.concat([D[D.role == "test"].assign(fold="oof"), E]).groupby("method"):
        for (ds, f), dd in d.groupby(["dataset", "fold"]):
            W = dd.pivot_table(index="participant", columns="seed", values="score")
            if W.shape[1] < 2:
                continue
            r = [W[i].corr(W[j], method="spearman") for i, j in itertools.combinations(W.columns, 2)]
            rows.append((m, "all", f, f"{ds}_seed_spearman", float(np.mean(r))))
    if len(L):
        L["method"] = L.method + "/" + L.embedding
        for _, r in L.iterrows():
            rows.append((r.method, r.seed, r.fold, r.metric, r.value))
    if len(L):
        emb_name = L.drop_duplicates("method").set_index(L.drop_duplicates("method").method.str.split("/").str[0]).method
        if not a.root:
            for m, s_, f, k, v in knn_purity(cfg):
                rows.append((emb_name.get(m, m), s_, f, k, v))
    B = pd.DataFrame(rows, columns=["method", "seed", "fold", "metric", "value"])
    B.to_csv(out / "benchmark.tsv", sep="\t", index=False)
    keep = ["method", "score_name", "seed", "fold", "dataset", "participant", "condition", "role", "score", "raw"]
    pd.concat([D[keep], E[keep]]).to_csv(out / "donor_scores.tsv", sep="\t", index=False)
    # methods x key metrics, mean ± sd over seeds (iLISI: mean over folds first)
    key = {("oof", "KPMP_DKDvsHKD_auc"), ("oof", "KPMP_DKDvsHKD_g"), ("ens", "GSE211785_DKDvsHKD_auc"),
           ("ens", "GSE211785_DKDvsHKD_g"), ("ens", "GSE211785_DKDvsControl_auc"), ("ens", "GSE211785_DKDvsControl_g"),
           ("ens", "GSE195460_DKDvsControl_auc"), ("ens", "GSE195460_DKDvsControl_g")}
    K = B[[(f, mt) in key for f, mt in zip(B.fold.astype(str), B.metric)]]
    li = B[B.metric.str.startswith(("iLISI", "kNN30"))].groupby(["method", "seed", "metric"], as_index=False).value.mean()
    K = pd.concat([K[["method", "seed", "metric", "value"]], li])
    S = K.groupby(["method", "metric"]).value.agg(["mean", "std", "count"]).reset_index()
    st = B[B.seed.astype(str) == "all"][["method", "fold", "metric", "value"]]
    st = st.assign(metric=st.fold.astype(str) + "_" + st.metric, mean=st.value, std=np.nan, count=1)
    S = pd.concat([S, st[["method", "metric", "mean", "std", "count"]]])
    S.to_csv(out / "summary_table.tsv", sep="\t", index=False)
    print(S.pivot_table(index="method", columns="metric", values="mean").round(3).T.to_string())
    ins = [p for d in sorted(root.glob("bench_*")) if d.name != "bench_method_summary"
           for p in [d / "donor_scores.tsv"] if p.exists() and (a.include_dev or not d.name.endswith("_dev"))]
    if a.root:   # development summaries: provenance next to the tables, the main record untouched
        (out / "INPUTS.txt").write_text("\n".join(map(str, ins)) + "\n")
        return
    write_provenance("bench_method_summary", ins, [out / "benchmark.tsv", out / "donor_scores.tsv", out / "summary_table.tsv"],
                     cfg["seed"], cfg, extra={"include_dev": a.include_dev})


if __name__ == "__main__":
    main()
