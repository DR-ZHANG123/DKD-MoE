"""Cell-level map of DKD-MoE: where cells lie in the latent space and which expert routes them.

Layout: UMAP of the latent means of final model seed 1 on a participant-balanced sample of KPMP proximal tubule
cells (at most cells_per_participant per participant). Colours come from the cross-validation models (cvw, five
seeds) applied to the participants they had not seen: routing probabilities of the consensus experts M1-M3
(expert identities from results/expr_heldout_experts/matching.tsv) and the cell log-odds, averaged over seeds.
Densities: on a grid over the layout, each participant's cells weigh 1 / (its number of sampled cells), so every
participant contributes equally; group densities are means over participants, for all cells and for injured cells
(participants with at least min_cells_density cells in the set).
Outputs: results/expr_cell_atlas/{cells.tsv, density.tsv, state_by_expert.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402

META = ["M1", "M2", "M3"]


def heldout_routing(ck: Path, pre: str, K: int, wc: float, seeds, M: pd.DataFrame, n: int):
    """Seed-mean held-out routing to M1-M3 and log-odds for every cell."""
    p = np.zeros((n, 3)); lo = np.zeros(n); cnt = np.zeros(n)
    for s in seeds:
        for f in range(5):
            z = np.load(ck / pre / f"{pre}_K{K}_wc{wc}_s{s}_f{f}.npz"); idx = z["idx"]
            row = M[(M.seed == s) & (M.fold == f)].iloc[0]
            p[idx] += np.column_stack([z["p"][:, int(row[f"{m}_expert"])] for m in META])
            lo[idx] += z["logodds"]; cnt[idx] += 1
    return p / cnt[:, None], lo / cnt


def main():
    cfg = load_cfg("expr"); ac = cfg["cell_atlas"]; set_global_seed(ac["random_state"])
    import umap
    from cdl import result_dir as rd
    sel = pd.read_json(rd("expr_cv") / "selection.json", typ="series"); K, wc = int(sel["K"]), float(sel["w_cond"])
    ck = Path(cfg["paths"]["ckpt_dir"]); st = cfg["readout_stability"]["stability"]
    h5 = Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad"
    obs = ad.read_h5ad(h5, backed="r").obs[["participant", "group", "pt_state", "tech", "label"]].reset_index(drop=True)
    M = pd.read_csv(rd("expr_heldout_experts") / "matching.tsv", sep="\t")
    p, lo = heldout_routing(ck, st["prefix"], K, wc, st["cv_seeds"], M, len(obs))
    rng = np.random.default_rng(ac["random_state"])
    take = np.sort(np.concatenate([rng.choice(ix, min(len(ix), ac["cells_per_participant"]), replace=False)
                                   for ix in obs.groupby("participant", observed=True).indices.values()]))
    zf = np.load(ck / "finalw" / f"finalw_K{K}_s{ac['layout_seed']}_kpmp.npz")["z"][take]
    xy = umap.UMAP(n_neighbors=ac["n_neighbors"], min_dist=ac["min_dist"], random_state=ac["random_state"]).fit_transform(zf)
    C = obs.iloc[take].copy(); C["cell"] = take; C["umap1"], C["umap2"] = xy[:, 0], xy[:, 1]
    for j, m in enumerate(META):
        C[f"p_{m}"] = p[take, j]
    C["logodds"] = lo[take]
    out = result_dir("expr_cell_atlas"); C.to_csv(out / "cells.tsv", sep="\t", index=False)
    # participant-balanced densities per group on a common grid
    b = ac["grid_bins"]; ex = [np.quantile(xy[:, 0], [0.001, 0.999]), np.quantile(xy[:, 1], [0.001, 0.999])]
    e1, e2 = np.linspace(*ex[0], b + 1), np.linspace(*ex[1], b + 1); rows = []
    for cells_set, Cs in (("all", C), ("injured", C[C.pt_state == "injured_PT"])):
        for g, Cg in Cs.groupby("group", observed=True):
            hs = [np.histogram2d(d.umap1, d.umap2, bins=[e1, e2])[0] / len(d)
                  for _, d in Cg.groupby("participant", observed=True) if len(d) >= ac["min_cells_density"]]
            Hm = np.mean(hs, 0)
            for i in range(b):
                for k in range(b):
                    rows.append((cells_set, g, (e1[i] + e1[i + 1]) / 2, (e2[k] + e2[k + 1]) / 2, Hm[i, k], len(hs)))
    pd.DataFrame(rows, columns=["cells", "group", "x", "y", "density", "n_participants"]).to_csv(
        out / "density.tsv", sep="\t", index=False)
    # held-out routing of healthy and injured cells, participant means then group means
    T = obs.assign(**{f"p_{m}": p[:, j] for j, m in enumerate(META)})
    P = T.groupby(["participant", "group", "pt_state"], observed=True)[[f"p_{m}" for m in META]].mean().reset_index()
    P.to_csv(out / "state_by_expert.tsv", sep="\t", index=False)
    write_provenance("expr_cell_atlas", [h5, rd("expr_heldout_experts") / "matching.tsv"],
                     [out / f for f in ("cells.tsv", "density.tsv", "state_by_expert.tsv")], ac["random_state"], cfg)
    print(C.shape, P.groupby(["group", "pt_state"], observed=True)[[f"p_{m}" for m in META]].mean().round(3))


if __name__ == "__main__":
    main()
