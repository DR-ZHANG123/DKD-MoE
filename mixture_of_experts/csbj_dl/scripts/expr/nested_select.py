"""Nested cross-validation, selection: within each outer fold, apply the cv: eligibility and rank-sum rule
(identical to cv_summary.py) to the inner held-out predictions of every grid setting.

Metrics per setting = held-out NLL per count (participant mean, then mean), seed ARI of top-1 expert, participant
DKD-vs-HKD AUC of the condition-head probability among injured-PT cells (>= cv: min_injured_cells), routing
entropy / log K and fraction of cells with top-1 weight > 0.5, all computed on the union of the 4 inner folds.
Outputs: results/expr_nested_cv/{inner_tables.tsv, selection.tsv}
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from nested_selection import part_scores, metrics, select  # noqa: E402
from nested_inner import inner_name  # noqa: E402


def main():
    cfg = load_cfg("expr"); cv = cfg["cv"]; nf = cfg["n_folds"]; out = result_dir("expr_nested_cv")
    d = Path(cfg["paths"]["ckpt_dir"]) / cfg["nested_cv"]["inner_prefix"]
    obs = ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs.copy()
    fold = obs.fold.values.astype(int)
    cls = cfg["condition"]["classes"]; iD, iH = cls.index("DKD"), cls.index("HKD")
    seeds = cv["seeds_for_stability"]
    grid = list(itertools.product(cv["K_grid"], [float(w) for w in cv["w_cond_grid"]]))
    tabs, choice, inputs = [], [], []
    for o in range(nf):
        inner = [i for i in range(nf) if i != o]
        rows = []
        for K, wc in grid:
            Z = {}
            for s in seeds:
                for i in inner:
                    f = d / f"{inner_name(o, K, wc, s, i)}.npz"; z = np.load(f); inputs.append(f)
                    assert np.array_equal(np.sort(z["idx"]), np.where(fold == i)[0]), f
                    nll, pm = part_scores(obs, z, iD, iH, cv["min_injured_cells"])
                    Z[(s, i)] = (nll, pm, z["p"])
            rows.append({"K": K, "w_cond": wc, **metrics(Z, inner, seeds, K),
                         "n_dkd": int((pd.concat([Z[(seeds[0], i)][1] for i in inner]).group == "DKD").sum()),
                         "n_hkd": int((pd.concat([Z[(seeds[0], i)][1] for i in inner]).group == "HKD").sum())})
        t = select(pd.DataFrame(rows), cv).assign(outer_fold=o)
        tabs.append(t)
        choice.append({"outer_fold": o, "K": int(t.K[0]), "w_cond": float(t.w_cond[0]), "eligible": bool(t.eligible[0]),
                       "rank_sum": float(t.rank_sum[0]), "n_eligible_settings": int(t.eligible.sum()),
                       "inner_auc": float(t.auc_dkd_hkd[0]), "inner_nll": float(t.nll[0]),
                       "inner_ari": float(t.stability_ari[0]),
                       "runner_up": f"K={int(t.K[1])}, w_cond={float(t.w_cond[1])} (rank sum {t.rank_sum[1]:g})"})
    T = pd.concat(tabs, ignore_index=True); C = pd.DataFrame(choice)
    T.to_csv(out / "inner_tables.tsv", sep="\t", index=False); C.to_csv(out / "selection.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(T.round(4).to_string()); print(C.to_string())
    write_provenance("expr_nested_cv", inputs[:1], [out / "inner_tables.tsv", out / "selection.tsv"], cfg["seed"], cfg,
                     extra={"n_inner_prediction_files": len(inputs), "seeds": seeds}, tag="select")


if __name__ == "__main__":
    main()
