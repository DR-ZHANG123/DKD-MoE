"""Robustness of the held-out readout to cell annotation, cell quality, technology and non-PT RNA.

(a) readout recomputed from injured-PT cells labelled by the KPMP atlas only / by the KPMP-internal classifier only,
    and from cells above / below the median number of detected genes of their technology;
(b) technology agreement: Spearman correlation of single-cell and single-nucleus readouts of participants with both;
(c) non-PT RNA in PT cells: injured-PT expression of immune, stromal and endothelial marker genes per participant
    (mean z of log CP10k), its correlation with the readout, and the DKD-vs-HKD difference adjusted for it;
    sample composition (immune and stromal share of all cells of the participant) as further adjustment;
(d) cellular source of the up-regulated signature genes: mean CP10k in injured PT relative to fibroblasts / stromal,
    immune and endothelial cells of the same KPMP samples.
Outputs: results/expr_readout_robustness/{subset_tests.tsv, tech_agreement.tsv, ambient_models.tsv, ambient_participant.tsv,
         signature_gene_sources.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp
import statsmodels.formula.api as smf
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import auc, hedges_g  # noqa: E402
from stability_compare import load_heldout  # noqa: E402
from readout import part_means  # noqa: E402


def logcp10k(X, lib):
    X = X.toarray() if sp.issparse(X) else np.asarray(X)
    return np.log1p(X / np.maximum(lib[:, None], 1) * 1e4)


def tests(name, r, grp):
    return [{"subset": name, "comparison": f"{a}_vs_{b}", "auc": auc(r[grp == a], r[grp == b]) if b == "HKD" else np.nan,
             **hedges_g(r[grp == a], r[grp == b])} for a, b in (("DKD", "HKD"), ("DM_noCKD", "HKD"), ("HKD_withDM", "HKD"))]


def main():
    cfg = load_cfg("expr"); pc = cfg["robustness"]; out = result_dir("expr_readout_robustness")
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    A = ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad"); obs = A.obs.copy()
    lab = pd.concat([pd.read_csv(result_dir("expr_labels") / f"kpmp_cell_labels_{t}.tsv.gz", sep="\t", index_col=0,
                                 usecols=[0, 1, 2, 5]) for t in ("sc", "sn")])
    obs["origin"] = lab.label_origin.reindex(obs.index).values
    lo = np.full(len(obs), np.nan); st = cfg["readout_stability"]["stability"]
    los = []
    for s in st["cv_seeds"]:
        idx, _, _, l = load_heldout(cfg, st["prefix"], K, wc, s); v = np.full(len(obs), np.nan); v[idx] = l; los.append(v)
    lo = np.nanmean(los, 0)
    med = obs.groupby("tech", observed=True).n_genes.transform("median").values
    subsets = {"all": np.ones(len(obs), bool), "atlas_labels": (obs.origin == "kpmp_atlas").values,
               "classifier_labels": (obs.origin == "classifier").values,
               "high_quality": obs.n_genes.values >= med, "low_quality": obs.n_genes.values < med}
    T = []
    for name, m in subsets.items():
        pm = part_means(obs[m], {"r": lo[m]}, pc["min_cells_subset"]); pm = pm[pm.tech == "pooled"].set_index("participant")
        T += tests(name, pm.r, pm.group)
    T = pd.DataFrame(T); T.to_csv(out / "subset_tests.tsv", sep="\t", index=False)
    # technology agreement
    pm = part_means(obs, {"r": lo}, cfg["prep"]["min_cells_per_participant"])
    W = pm[pm.tech != "pooled"].pivot_table(index="participant", columns="tech", values="r").dropna()
    g = pm.drop_duplicates("participant").set_index("participant").group.reindex(W.index)
    TA = [{"set": "all", "n": len(W), "spearman": stats.spearmanr(W.sc, W.sn)[0], "p": stats.spearmanr(W.sc, W.sn)[1]}]
    d = W[g.isin(["DKD", "HKD"])]; TA.append({"set": "DKD_HKD", "n": len(d), "spearman": stats.spearmanr(d.sc, d.sn)[0],
                                              "p": stats.spearmanr(d.sc, d.sn)[1]})
    pd.DataFrame(TA).to_csv(out / "tech_agreement.tsv", sep="\t", index=False)
    # non-PT marker expression in injured-PT cells
    inj = (obs.pt_state.astype(str) == "injured_PT").values; ii = np.where(inj)[0]
    genes = sorted({x for v in pc["marker_sets"].values() for x in v})
    X = A[ii][:, genes].X; X = X.toarray() if sp.issparse(X) else np.asarray(X)
    L = logcp10k(X, obs.total_counts.values[ii])
    df = pd.DataFrame(L, columns=genes); df["participant"] = obs.participant.astype(str).values[ii]; df["tech"] = obs.tech.astype(str).values[ii]
    pb = df.groupby(["participant", "tech"])[genes].mean()
    pb = (pb - pb.groupby(level="tech").transform("mean")) / pb.groupby(level="tech").transform("std")
    amb = pd.DataFrame({k: pb[v].mean(1) for k, v in pc["marker_sets"].items()}).groupby(level="participant").mean()
    allab = lab.assign(participant=lab.participant.astype(str))
    comp = allab.groupby("participant").label.agg(lambda s: pd.Series({
        "immune_share": s.isin(pc["source_celltypes"]["immune"]).mean(),
        "stromal_share": s.isin(pc["source_celltypes"]["fibroblast_stromal"]).mean()}).to_dict())
    comp = pd.DataFrame(comp.tolist(), index=comp.index)
    P = pm[pm.tech == "pooled"].set_index("participant")[["group", "r"]].join(amb).join(comp)
    P.to_csv(out / "ambient_participant.tsv", sep="\t")
    rows = []
    for k in list(pc["marker_sets"]) + ["immune_share", "stromal_share"]:
        q = P.dropna(subset=[k]); rho, p = stats.spearmanr(q.r, q[k]); rows.append({"model": f"spearman_readout_{k}", "beta": rho, "p": p, "n": len(q)})
    D = P[P.group.isin(["DKD", "HKD"])].dropna().assign(DKD=lambda x: (x.group == "DKD").astype(int))
    D["y"] = (D.r - D.r.mean()) / D.r.std()
    for name, f in {"unadjusted": "y ~ DKD", "markers": "y ~ DKD + immune + stromal + endothelial",
                    "markers_composition": "y ~ DKD + immune + stromal + endothelial + immune_share + stromal_share"}.items():
        m = smf.ols(f, D).fit(cov_type="HC3"); ci = m.conf_int()
        rows.append({"model": name, "beta": m.params["DKD"], "ci_low": ci.loc["DKD", 0], "ci_high": ci.loc["DKD", 1],
                     "p": m.pvalues["DKD"], "n": int(m.nobs)})
    pd.DataFrame(rows).to_csv(out / "ambient_models.tsv", sep="\t", index=False)
    # cellular source of up-regulated signature genes (full KPMP objects, all cell types)
    sig = pd.read_csv(result_dir("expr_transfer") / "transfer_signature.tsv", sep="\t"); up = sig.gene[sig.direction == "up"].tolist()
    src = []
    for t, f in pc["full_counts"].items():
        B = ad.read_h5ad(ROOT / f)
        B.obs["label"] = lab.label.reindex(B.obs_names).astype(str).values
        lib = np.asarray(B.X.sum(1)).ravel()
        gs = [x for x in up if x in B.var_names]
        Lg = pd.DataFrame(logcp10k(B[:, gs].X, lib), columns=gs)
        Lg = np.expm1(Lg)                       # mean on the CP10k scale
        for ct, labs in pc["source_celltypes"].items():
            m = B.obs.label.isin(labs).values
            src.append(pd.DataFrame({"tech": t, "celltype": ct, "gene": gs, "mean_cp10k": Lg[m].mean(0).values,
                                     "n_cells": int(m.sum())}))
        del B
    SRC = pd.concat(src).pivot_table(index=["tech", "gene"], columns="celltype", values="mean_cp10k").reset_index()
    oth = [c for c in pc["source_celltypes"] if c != "injured_PT"]
    SRC["pt_over_max_other"] = SRC.injured_PT / SRC[oth].max(1)
    SRC["pt_share_of_total"] = SRC.injured_PT / SRC[["injured_PT"] + oth].sum(1)
    SRC.to_csv(out / "signature_gene_sources.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
    print(T.round(3).to_string()); print(pd.DataFrame(TA).round(3)); print(pd.DataFrame(rows).round(3).to_string())
    print(SRC.round(3).sort_values("pt_over_max_other").to_string())
    write_provenance("expr_readout_robustness", [], [out / f for f in ("subset_tests.tsv", "tech_agreement.tsv", "ambient_models.tsv",
                                                            "ambient_participant.tsv", "signature_gene_sources.tsv")], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
