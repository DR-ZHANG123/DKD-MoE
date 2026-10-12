"""Transferable signature built from genes expressed mainly by proximal tubule cells.

Constructions (each with the readout-association rule of the transferable signature, 50 up / 50 down):
  pt_intrinsic                  candidate genes restricted to those whose injured-PT expression is at least
                                pt_signature.min_pt_ratio of the highest fibroblast / stromal, immune or endothelial
                                expression in both technologies (full KPMP objects, all cell types)
  marker_adjusted               readout-gene Spearman correlations computed on ranks residualised for injured-PT
                                expression of immune, stromal and endothelial marker genes (within technology)
  pt_intrinsic_marker_adjusted  both
Cross-fitted in KPMP with the partition models (out-of-fold DKD-vs-HKD AUC), rebuilt on all KPMP with the final
models and scored in the external cohorts exactly as the transferable signature.
Outputs: results/expr_pt_signature/{gene_sources.tsv, internal_tests.tsv, signatures.tsv, external_tests.tsv,
         overlap.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from readout_common import final_set  # noqa: E402
from transfer_common import kpmp_pseudobulk, expressed, score_kpmp, top_bottom, load_state, cell_logodds  # noqa: E402
from transfer_signatures import evaluate  # noqa: E402
import external_signature as es  # noqa: E402


def gene_sources(cfg, genes) -> pd.DataFrame:
    pc = cfg["robustness"]
    lab = pd.concat([pd.read_csv(result_dir("expr_labels") / f"kpmp_cell_labels_{t}.tsv.gz", sep="\t", index_col=0,
                                 usecols=[0, 2]) for t in ("sc", "sn")]).label
    rows = []
    for t, f in pc["full_counts"].items():
        B = ad.read_h5ad(ROOT / f); g = [x for x in genes if x in B.var_names]
        X = B[:, g].X; X = X.tocsr() if sp.issparse(X) else sp.csr_matrix(X)
        lib = np.asarray(B.X.sum(1)).ravel()
        Xn = sp.diags(1e4 / np.maximum(lib, 1)) @ X
        L = lab.reindex(B.obs_names).astype(str).values
        for ct, labs in pc["source_celltypes"].items():
            m = np.isin(L, labs)
            rows.append(pd.DataFrame({"tech": t, "celltype": ct, "gene": g, "mean_cp10k": np.asarray(Xn[m].mean(0)).ravel()}))
        del B
    S = pd.concat(rows).pivot_table(index=["gene", "tech"], columns="celltype", values="mean_cp10k").reset_index()
    oth = [c for c in pc["source_celltypes"] if c != "injured_PT"]
    S["pt_ratio"] = S.injured_PT / S[oth].max(1).clip(lower=1e-9)
    return S


def resid_rank(Y: pd.DataFrame, C: pd.DataFrame) -> pd.DataFrame:
    """Ranks of each column of Y residualised on ranks of the columns of C (least squares with intercept)."""
    R = Y.rank(); Z = np.column_stack([np.ones(len(C)), C.rank().values])
    beta, *_ = np.linalg.lstsq(Z, R.values, rcond=None)
    return pd.DataFrame(R.values - Z @ beta, index=Y.index, columns=Y.columns)


def assoc(L, M, r, genes, markers, adjust):
    """Readout-gene Spearman within technology (optionally partial on marker scores), Fisher-combined."""
    num, den = 0, 0
    for t in M.tech.unique():
        idx = M.index[(M.tech == t) & M.index.isin(r.index)]
        if len(idx) < 8:
            continue
        Y = L.loc[idx, genes]; y = r[idx].to_frame("r")
        if adjust:
            C = pd.DataFrame({k: L.loc[idx, v].apply(lambda c: (c - c.mean()) / c.std()).mean(1) for k, v in markers.items()})
            Y, y = resid_rank(Y, C), resid_rank(y, C)
        else:
            Y, y = Y.rank(), y.rank()
        c = Y.corrwith(y.r)
        num = num + np.arctanh(c.clip(-0.999, 0.999)) * (len(idx) - 3); den += len(idx) - 3
    return np.tanh(num / den)


def readout(models, data, M, parts):
    st = data.obs.pt_state.astype(str).values
    idx = np.where(data.obs.participant.astype(str).isin(parts).values & (st == "injured_PT"))[0]
    lo = np.mean([cell_logodds(m, data, idx) for m in models], 0); o = data.obs.iloc[idx]
    r = pd.Series(lo, index=(o.participant.astype(str) + "|" + o.tech.astype(str)).values).groupby(level=0).mean()
    return r[r.index.isin(M.index)]


def main():
    cfg = load_cfg("expr"); ps = cfg["pt_signature"]; set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]
    rc = {**cfg["transfer"], **cfg["transfer"]["pseudobulk"]}; out = result_dir("expr_pt_signature"); fs = final_set(cfg)
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    L, M = kpmp_pseudobulk(data, rc); genes = expressed(L, M, rc)
    markers = {k: [g for g in v if g in L.columns] for k, v in cfg["robustness"]["marker_sets"].items()}
    S = gene_sources(cfg, data.genes); S.to_csv(out / "gene_sources.tsv", sep="\t", index=False)
    ok = S.groupby("gene").pt_ratio.min() >= ps["min_pt_ratio"]
    pt_genes = [g for g in genes if ok.get(g, False)]
    universe = {"pt_intrinsic": (pt_genes, False), "marker_adjusted": (genes, True), "pt_intrinsic_marker_adjusted": (pt_genes, True)}
    ck = Path(cfg["paths"]["ckpt_dir"]) / fs["cv_prefix"]; oof = {k: pd.Series(dtype=float) for k in universe}; rows = []
    for f in range(cfg["n_folds"]):
        models = [load_state(cfg, ck / f"{fs['cv_prefix']}_K{fs['K']}_wc{fs['w_cond']}_s{s}_f{f}.pt", fs["K"], dev) for s in fs["cv_seeds"]]
        Mt = M[M.fold != f]; r = readout(models, data, Mt, set(Mt.participant)); test = M.index[M.fold == f]
        for k, (g, adj) in universe.items():
            up, down = top_bottom(assoc(L, Mt, r, g, markers, adj), rc["n_up"], rc["n_down"])
            oof[k] = pd.concat([oof[k], score_kpmp(L, M, up, down)[test]])
        print("fold", f, flush=True)
    for k, s in oof.items():
        rows += evaluate(s, M, k)
    T = pd.DataFrame(rows); T.to_csv(out / "internal_tests.tsv", sep="\t", index=False)
    models = [load_state(cfg, fs["dir"] / (fs["stem"].format(s=s) + ".pt"), fs["K"], dev) for s in fs["seeds"]]
    r = readout(models, data, M, set(M.participant))
    orig = pd.read_csv(result_dir("expr_transfer") / "transfer_signature.tsv", sep="\t")
    sigs, ext, ov = [], [], []
    es.OUT = "expr_pt_signature"
    nmin, mc = cfg["readout_stability"]["signature"]["min_genes_per_direction"], cfg["prep"]["min_cells_per_participant"]
    for k, (g, adj) in universe.items():
        up, down = top_bottom(assoc(L, M, r, g, markers, adj), rc["n_up"], rc["n_down"])
        sigs += [{"construction": k, "direction": d, "gene": x} for d, gl in (("up", up), ("down", down)) for x in gl]
        for d, gl in (("up", up), ("down", down)):
            ov.append({"construction": k, "direction": d, "n_shared_with_transfer_signature": len(set(gl) & set(orig.gene[orig.direction == d]))})
        ext += [{"construction": k, **x} for x in es.ercb(cfg, up, down, nmin) + es.zenodo(cfg, up, down, nmin, mc) + es.sc_cohorts(cfg, up, down, nmin, mc)]
    pd.DataFrame(sigs).to_csv(out / "signatures.tsv", sep="\t", index=False)
    E = pd.DataFrame(ext); E.to_csv(out / "external_tests.tsv", sep="\t", index=False)
    pd.DataFrame(ov).to_csv(out / "overlap.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
    print(len(genes), "expressed;", len(pt_genes), "PT-intrinsic")
    print(T.round(3).to_string()); print(pd.DataFrame(ov).to_string())
    print(E[[c for c in ["construction", "cohort", "platform", "celltype", "comparison", "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon"] if c in E]].round(3).to_string())
    for k in universe:
        print(k, [x["gene"] for x in sigs if x["construction"] == k and x["direction"] == "up"][:30])
    write_provenance("expr_pt_signature", [result_dir("expr_transfer") / "transfer_signature.tsv"],
                     [out / f for f in ("gene_sources.tsv", "internal_tests.tsv", "signatures.tsv", "external_tests.tsv", "overlap.tsv")],
                     cfg["seed"], cfg, extra={"n_expressed": len(genes), "n_pt_intrinsic": len(pt_genes)})


if __name__ == "__main__":
    main()
