"""Does the participant readout depend on routing mass that the forward computation actually uses?

The condition head reads the dense routing probabilities p; the expert mixture uses the top-2 renormalised
weights g. For the final models this compares
  dense readout  s_i = mean over injured-PT cells of w.p + b   (the participant readout)
  top-2 readout  s_i = mean over injured-PT cells of w.g + b
with participants, cells, model sets and averaging exactly as in readout.py (participant x technology means
over injured-PT cells with >= min_cells cells, mean over technologies, mean over the five seeds; KPMP =
held-out cross-validation models, external cohorts = final models). It also reports, per seed x expert, the
share of the expert's routing mass (and hence of its contribution w_k p_k) carried by cells for which the
expert is in the top 2, aggregated to the meta-experts of results/expr_meta.
Outputs: results/expr_routing_faithfulness/{participant_readouts.tsv, readout_comparison.tsv,
         final_unit_topk_share.tsv, meta_topk_share.tsv, cell_topk_mass.tsv} + PROVENANCE.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from readout import part_means  # noqa: E402
from readout_common import final_set  # noqa: E402
from bench_metrics import hedges_g, auc_ci  # noqa: E402
from harm_summary import paired_auc  # noqa: E402

COMPS = {"KPMP_heldout": ("DKD", "HKD"), "gse211785": ("DKD", "HKD"), "gse195460": ("DKD", "Control")}


def head(pt: Path):
    sd = torch.load(pt, map_location="cpu")
    sd = sd.get("state_dict", sd)
    W, b = sd["cond_head.weight"].numpy(), sd["cond_head.bias"].numpy()
    return W[1] - W[0], float(b[1] - b[0])                 # classes [HKD, DKD]


def heldout(cfg, fs, seed, n):
    """KPMP held-out p, g and head weights (per cell, from that cell's fold model), in KPMP row order."""
    d = Path(cfg["paths"]["ckpt_dir"]) / fs["cv_prefix"]
    P = np.full((n, fs["K"]), np.nan, np.float32); G = P.copy(); Wc = P.copy(); Bc = np.full(n, np.nan)
    LO = np.full(n, np.nan)
    for f in range(cfg["n_folds"]):
        stem = f"{fs['cv_prefix']}_K{fs['K']}_wc{fs['w_cond']}_s{seed}_f{f}"
        z = np.load(d / f"{stem}.npz"); w, b = head(d / f"{stem}.pt"); i = z["idx"]
        P[i], G[i], LO[i] = z["p"], z["g"], z["logodds"]; Wc[i] = w; Bc[i] = b
    return P, G, Wc, Bc, LO


def final(fs, seed, cohort):
    z = np.load(fs["dir"] / (fs["stem"].format(s=seed) + f"_{cohort}.npz"))
    w, b = head(fs["dir"] / (fs["stem"].format(s=seed) + ".pt"))
    n = len(z["p"])
    return z["p"], z["g"], np.tile(w, (n, 1)), np.full(n, b), z["logodds"]


def topk_mask(P, k=2):
    M = np.zeros_like(P, dtype=bool)
    np.put_along_axis(M, np.argsort(-P, 1)[:, :k], True, 1)
    return M


def main():
    cfg = load_cfg("expr"); fs = final_set(cfg); out = result_dir("expr_routing_faithfulness")
    bulk = Path(cfg["paths"]["bulk_dir"]); mc = cfg["prep"]["min_cells_per_participant"]
    obs = {c: ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad", backed="r").obs.copy() for c in ("kpmp", "gse211785", "gse195460")}
    mcfg = load_cfg("bench")["metrics"]
    rows, cellmass = [], []
    for name, c in (("KPMP_heldout", "kpmp"), ("gse211785", "gse211785"), ("gse195460", "gse195460")):
        o = obs[c]; inj = (o.pt_state.astype(str) == "injured_PT").values; vals = {}
        for s in fs["cv_seeds"] if name == "KPMP_heldout" else fs["seeds"]:
            P, G, W, B, LO = heldout(cfg, fs, s, len(o)) if name == "KPMP_heldout" else final(fs, s, c)
            dense = (W * P).sum(1) + B; top2 = (W * G).sum(1) + B
            ok = ~np.isnan(LO)
            assert np.allclose(dense[ok], LO[ok], atol=1e-4), "dense readout does not match stored log-odds"
            vals[f"dense_seed{s}"] = dense; vals[f"top2_seed{s}"] = top2
            sel = inj & ok
            top2mass = np.sort(np.nan_to_num(P[sel]), 1)[:, -2:].sum(1)
            cellmass.append({"cohort": name, "seed": s, "n_injured_cells": int(sel.sum()),
                             "mean_top2_mass_injured": float(top2mass.mean()),
                             "median_top2_mass_injured": float(np.median(top2mass)),
                             "mean_abs_dense_minus_top2_logodds": float(np.abs(dense - top2)[sel].mean())})
        pm = part_means(o, vals, mc)
        pm["dense"] = pm[[k for k in vals if k.startswith("dense")]].mean(1)
        pm["top2"] = pm[[k for k in vals if k.startswith("top2")]].mean(1)
        pm["cohort"] = name; rows.append(pm)
    S = pd.concat(rows, ignore_index=True)
    S.to_csv(out / "participant_readouts.tsv", sep="\t", index=False)
    # check against the stored participant readout
    R = pd.read_csv(result_dir("expr_readout") / "participant_scores.tsv", sep="\t", dtype={"participant": str})
    chk = S.merge(R[["cohort", "participant", "tech", "logodds"]], on=["cohort", "participant", "tech"])
    assert len(chk) == len(R[R.cohort.isin(COMPS)]) and np.allclose(chk.dense, chk.logodds, atol=1e-4)
    C = []
    for name, (a, b) in COMPS.items():
        q = S[(S.cohort == name) & (S.tech == "pooled")]
        r = {"cohort": name, "comparison": f"{a}_vs_{b}", "n_participants": len(q),
             "spearman_dense_vs_top2": stats.spearmanr(q.dense, q.top2)[0],
             "pearson_dense_vs_top2": stats.pearsonr(q.dense, q.top2)[0]}
        for col in ("dense", "top2"):
            x, y = q[q.group == a][col].values, q[q.group == b][col].values
            g, gl, gh = hedges_g(x, y, mcfg["ci"]); au, al, ah = auc_ci(x, y, mcfg["n_boot"], 1, mcfg["ci"])
            r.update({f"{col}_g": g, f"{col}_g_lo": gl, f"{col}_g_hi": gh, f"{col}_auc": au, f"{col}_auc_lo": al,
                      f"{col}_auc_hi": ah, "n1": len(x), "n2": len(y)})
        q2 = q[q.group.isin([a, b])].replace({"group": {b: "HKD", a: "DKD"}}).set_index("participant")
        d0, dl, dh = paired_auc(q2.dense, q2.top2, q2.group, mcfg["n_boot"], 1)
        r.update({"auc_top2_minus_dense": d0, "auc_diff_lo": dl, "auc_diff_hi": dh})
        C.append(r)
    C = pd.DataFrame(C); C.to_csv(out / "readout_comparison.tsv", sep="\t", index=False)
    pd.DataFrame(cellmass).to_csv(out / "cell_topk_mass.tsv", sep="\t", index=False)
    # meta-experts: final-model units (expr_meta/units.tsv maps seed x expert -> meta-expert)
    mu = pd.read_csv(result_dir("expr_meta") / "units.tsv", sep="\t")[["seed", "expert", "meta_expert"]]
    rep = pd.read_csv(result_dir("expr_meta") / "meta_experts.tsv", sep="\t").set_index("meta_expert").reproducible
    F = []
    for name, c in (("KPMP_final", "kpmp"), ("gse211785", "gse211785"), ("gse195460", "gse195460")):
        o = obs[c]; inj = (o.pt_state.astype(str) == "injured_PT").values
        for s in fs["seeds"]:
            P, G, W, B, LO = final(fs, s, c); M = topk_mask(P)
            for k in range(fs["K"]):
                for cells, lab in ((inj, "injured_PT"), (np.ones(len(o), bool), "all_PT")):
                    pk = P[cells, k]
                    F.append({"cohort": name, "seed": s, "expert": k, "cells": lab,
                              "head_weight_dkd_minus_hkd": float(W[0, k]), "mean_p": float(pk.mean()),
                              "frac_cells_expert_in_top2": float(M[cells, k].mean()),
                              "mass_in_top2_share": float(pk[M[cells, k]].sum() / pk.sum()),
                              "contribution_in_top2_share": float((W[0, k] * pk)[M[cells, k]].sum() / (W[0, k] * pk).sum()),
                              "routing_mass": float(pk.sum())})
    F = pd.DataFrame(F).merge(mu, on=["seed", "expert"])
    F.to_csv(out / "final_unit_topk_share.tsv", sep="\t", index=False)
    MM = (F.assign(m=F.mass_in_top2_share * F.routing_mass)
          .groupby(["cohort", "cells", "meta_expert"])
          .agg(n_units=("seed", "size"), n_seeds=("seed", "nunique"), mean_unit_share=("mass_in_top2_share", "mean"),
               min_unit_share=("mass_in_top2_share", "min"), m=("m", "sum"), tot=("routing_mass", "sum")).reset_index())
    MM["pooled_share"] = MM.m / MM.tot; MM = MM.drop(columns=["m", "tot"])
    MM["reproducible"] = MM.meta_expert.map(rep)
    MM.to_csv(out / "meta_topk_share.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(C.round(3).to_string()); print(pd.DataFrame(cellmass).round(3).to_string())
    print(MM[MM.reproducible].round(3).to_string())
    write_provenance("expr_routing_faithfulness", [result_dir("expr_readout") / "participant_scores.tsv",
                                                   result_dir("expr_meta") / "units.tsv"],
                     [out / f for f in ("participant_readouts.tsv", "readout_comparison.tsv",
                                        "final_unit_topk_share.tsv", "meta_topk_share.tsv", "cell_topk_mass.tsv")],
                     cfg["seed"], cfg, extra={"model_dir": str(fs["dir"]), "cv_prefix": fs["cv_prefix"],
                                              "top_k": cfg["model"]["top_k"]})


if __name__ == "__main__":
    main()
