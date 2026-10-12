"""Signature of the DKD-weighted expert contribution itself (rather than of the total readout).

Construction as for the readout-derived signature (readout association, 50 up / 50 down), with the
DKD-weighted expert contribution in place of the readout: for each part f, the five partition models trained without
f give the contribution w_k * p_k of their DKD-weighted expert (identities from training participants only,
results/expr_trainonly_experts/assignments.tsv) for injured-PT cells of the training participants; participant x
technology means are correlated with injured-PT expression (Spearman within technology, Fisher-combined); the
held-out participants of part f are scored. The final signature uses the five final models and their DKD-weighted
expert (reproducible expert M3, results/expr_meta/units.tsv) on all KPMP participants and is scored in the external
cohorts exactly as the readout-derived signature. Overlap with the readout-derived signature is reported.
Outputs: results/expr_m3_signature/{internal_tests.tsv, oof_scores.tsv, signature.tsv, fold_signatures.tsv,
         external_tests.tsv, overlap.tsv} + PROVENANCE.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from readout_common import final_set  # noqa: E402
from transfer_common import kpmp_pseudobulk, expressed, score_kpmp, top_bottom, fisher_combined_spearman, load_state  # noqa: E402
from transfer_signatures import evaluate  # noqa: E402
import external_signature as es  # noqa: E402


def contribution(models_experts, data, M, parts):
    """Mean over models of w_k p_k (DKD-weighted expert) per participant x technology over injured-PT cells."""
    st = data.obs.pt_state.astype(str).values
    idx = np.where(data.obs.participant.astype(str).isin(parts).values & (st == "injured_PT"))[0]
    vals = []
    for m, k in models_experts:
        p = forward_all(m, data, idx)["p"]; w = (m.cond_head.weight[1] - m.cond_head.weight[0]).detach().cpu().numpy()
        vals.append(w[k] * p[:, k])
    o = data.obs.iloc[idx]
    r = pd.Series(np.mean(vals, 0), index=(o.participant.astype(str) + "|" + o.tech.astype(str)).values).groupby(level=0).mean()
    return r[r.index.isin(M.index)]


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]
    rc = {**cfg["transfer"], **cfg["transfer"]["pseudobulk"]}; out = result_dir("expr_m3_signature"); fs = final_set(cfg)
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    L, M = kpmp_pseudobulk(data, rc); genes = expressed(L, M, rc)
    A = pd.read_csv(result_dir("expr_trainonly_experts") / "assignments.tsv", sep="\t").set_index("fold")
    ck = Path(cfg["paths"]["ckpt_dir"]) / fs["cv_prefix"]; oof = pd.Series(dtype=float); fsig = []
    for f in range(cfg["n_folds"]):
        me = [(load_state(cfg, ck / f"{fs['cv_prefix']}_K{fs['K']}_wc{fs['w_cond']}_s{s}_f{f}.pt", fs["K"], dev),
               int(A.loc[f, f"seed{s}_dkd_expert"])) for s in fs["cv_seeds"]]
        Mt = M[M.fold != f]; r = contribution(me, data, Mt, set(Mt.participant))
        up, down = top_bottom(fisher_combined_spearman(L, Mt, r, genes), rc["n_up"], rc["n_down"])
        oof = pd.concat([oof, score_kpmp(L, M, up, down)[M.index[M.fold == f]]])
        fsig += [{"fold": f, "direction": d, "gene": g} for d, gl in (("up", up), ("down", down)) for g in gl]
        print("part", f, flush=True)
    T = pd.DataFrame(evaluate(oof, M, "dkd_weighted_expert"))
    T.to_csv(out / "internal_tests.tsv", sep="\t", index=False); oof.rename("score").to_csv(out / "oof_scores.tsv", sep="\t")
    FS = pd.DataFrame(fsig); FS.to_csv(out / "fold_signatures.tsv", sep="\t", index=False)
    U = pd.read_csv(result_dir("expr_meta") / "units.tsv", sep="\t")
    me = [(load_state(cfg, fs["dir"] / (fs["stem"].format(s=s) + ".pt"), fs["K"], dev),
           int(U[(U.seed == s) & (U.meta_expert == "M3")].expert.iloc[0])) for s in fs["seeds"]]
    r = contribution(me, data, M, set(M.participant))
    up, down = top_bottom(fisher_combined_spearman(L, M, r, genes), rc["n_up"], rc["n_down"])
    sig = pd.concat([pd.DataFrame({"gene": up, "direction": "up"}), pd.DataFrame({"gene": down, "direction": "down"})])
    sig.to_csv(out / "signature.tsv", sep="\t", index=False)
    orig = pd.read_csv(result_dir("expr_transfer") / "transfer_signature.tsv", sep="\t")
    ov = pd.DataFrame([{"direction": d, "n_shared_with_readout_signature": len(set(sig.gene[sig.direction == d]) & set(orig.gene[orig.direction == d])),
                        "fold_stability_frac_in_3_of_5": float(FS[FS.direction == d].groupby("gene").fold.nunique().ge(3).reindex(sig.gene[sig.direction == d]).fillna(False).mean())}
                       for d in ("up", "down")])
    ov.to_csv(out / "overlap.tsv", sep="\t", index=False)
    es.OUT = "expr_m3_signature"
    nmin, mc = cfg["readout_stability"]["signature"]["min_genes_per_direction"], cfg["prep"]["min_cells_per_participant"]
    E = pd.DataFrame(es.ercb(cfg, up, down, nmin) + es.zenodo(cfg, up, down, nmin, mc) + es.sc_cohorts(cfg, up, down, nmin, mc))
    E.to_csv(out / "external_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(T.round(3).to_string()); print(ov.to_string()); print("up:", up[:25]); print("down:", down[:25])
    print(E[[c for c in ["cohort", "platform", "celltype", "comparison", "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon"] if c in E]].round(3).to_string())
    write_provenance("expr_m3_signature", [result_dir("expr_trainonly_experts") / "assignments.tsv", result_dir("expr_meta") / "units.tsv"],
                     [out / f for f in ("internal_tests.tsv", "oof_scores.tsv", "signature.tsv", "fold_signatures.tsv",
                                        "external_tests.tsv", "overlap.tsv")], cfg["seed"], cfg)


if __name__ == "__main__":
    main()
