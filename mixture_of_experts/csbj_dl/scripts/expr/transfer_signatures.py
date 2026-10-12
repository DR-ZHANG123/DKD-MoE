"""Internal diagnosis of the consensus IG signature and cross-fitted signature constructions in KPMP.

1. Consensus IG signature (healthy-PT baseline, final models) scored in KPMP injured-PT pseudobulk.
2. Cross-fitted constructions (expr.yaml transfer.constructions): per fold, built on training participants with
   that fold's stabilised models; held-out participants scored; out-of-fold scores pooled.
3. Selection = highest pooled DKD-vs-HKD AUC; the chosen construction is rebuilt on all KPMP (final models).
Outputs: results/expr_transfer/{kpmp_pseudobulk_meta.tsv, internal_signature_tests.tsv, oof_scores.tsv,
         selection.json, transfer_signature.tsv, fold_signatures.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from model import log_norm  # noqa: E402
from stats_utils import hedges_g, auc  # noqa: E402
from readout_common import final_set  # noqa: E402
from transfer_common import (kpmp_pseudobulk, expressed, score_kpmp, top_bottom, fisher_combined_spearman,  # noqa: E402
                       load_state, cell_logodds, ig_logodds)


def evaluate(S: pd.Series, M: pd.DataFrame, name: str) -> list[dict]:
    d = M.assign(score=S.reindex(M.index))
    rows = []
    for tech in ["pooled", "sc", "sn"]:
        q = (d.groupby("participant").agg(group=("group", "first"), score=("score", "mean")) if tech == "pooled"
             else d[d.tech == tech].set_index("participant")[["group", "score"]])
        D, H = q[q.group == "DKD"].score, q[q.group == "HKD"].score
        rows.append({"construction": name, "tech": tech, "auc_dkd_hkd": auc(D, H), **hedges_g(D, H)})
    return rows


def build(kind, cfg, data, L, M, genes, models, train_parts, rc, rng):
    """Return (up, down) for one construction from the given training participants and models."""
    tr_cells = np.where(data.obs.participant.astype(str).isin(train_parts).values)[0]
    st = data.obs.pt_state.astype(str).values; grp = data.obs.group.astype(str).values
    if kind == "readout_assoc":
        idx = tr_cells[st[tr_cells] == "injured_PT"]
        lo = np.mean([cell_logodds(m, data, idx) for m in models], 0)
        o = data.obs.iloc[idx]
        r = pd.Series(lo, index=(o.participant.astype(str) + "|" + o.tech.astype(str)).values).groupby(level=0).mean()
        r = r[r.index.isin(M.index)]
        return top_bottom(fisher_combined_spearman(L, M, r, genes), rc["n_up"], rc["n_down"])
    inj_tr = tr_cells[st[tr_cells] == "injured_PT"]
    base_idx = inj_tr[grp[inj_tr] == "HKD"] if kind == "ig_hkd_baseline" else tr_cells[st[tr_cells] == "healthy_PT"]
    with torch.no_grad():
        x0 = torch.cat([log_norm(data.rows(base_idx[s:s + 8192])) for s in range(0, len(base_idx), 8192)]).mean(0, keepdim=True)
        cells = np.sort(rng.choice(inj_tr, min(rc["ig_cells_per_fold"], len(inj_tr)), replace=False))
        xl = log_norm(data.rows(cells))
    A = np.mean([ig_logodds(m, xl, x0, cfg["interpret"]["ig_steps"]) for m in models], 0)
    s = pd.Series(A, index=data.genes)[genes]
    return top_bottom(s, rc["n_up"], rc["n_down"])


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]; rc = cfg["transfer"]
    rc = {**rc, **rc["pseudobulk"]}
    out = result_dir("expr_transfer"); bulk = Path(cfg["paths"]["bulk_dir"]); fs = final_set(cfg)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    L, M = kpmp_pseudobulk(data, rc); genes = expressed(L, M, rc)
    M.to_csv(out / "kpmp_pseudobulk_meta.tsv", sep="\t")
    rows = []
    # 1. consensus IG signature inside KPMP
    sig2 = pd.read_csv(result_dir("expr_signature") / "signature.tsv", sep="\t")
    S2 = score_kpmp(L, M, sig2[sig2.direction == "up"].gene, sig2[sig2.direction == "down"].gene)
    rows += [{**r, "setting": "consensus_signature_in_KPMP (models trained on all KPMP)"} for r in evaluate(S2, M, "ig_healthy_baseline_consensus")]
    # 2. cross-fitted constructions
    rng = np.random.default_rng(cfg["seed"]); ck = Path(cfg["paths"]["ckpt_dir"]) / fs["cv_prefix"]
    oof = {k: pd.Series(dtype=float) for k in rc["constructions"]}; fsig = []
    for f in range(cfg["n_folds"]):
        models = [load_state(cfg, ck / f"{fs['cv_prefix']}_K{fs['K']}_wc{fs['w_cond']}_s{s}_f{f}.pt", fs["K"], dev)
                  for s in fs["cv_seeds"]]
        train_parts = set(M.participant[M.fold != f]); test = M.index[M.fold == f]
        for kind in rc["constructions"]:
            up, down = build(kind, cfg, data, L, M[M.fold != f], genes, models, train_parts, rc, rng)
            oof[kind] = pd.concat([oof[kind], score_kpmp(L, M, up, down)[test]])
            fsig += [{"construction": kind, "fold": f, "direction": d, "gene": g} for d, gl in (("up", up), ("down", down)) for g in gl]
        print("fold", f, "done", flush=True)
    for kind, S in oof.items():
        rows += [{**r, "setting": "cross-fitted out-of-fold"} for r in evaluate(S, M, kind)]
    T = pd.DataFrame(rows); T.to_csv(out / "internal_signature_tests.tsv", sep="\t", index=False)
    pd.DataFrame(oof).to_csv(out / "oof_scores.tsv", sep="\t")
    FS = pd.DataFrame(fsig); FS.to_csv(out / "fold_signatures.tsv", sep="\t", index=False)
    stab = {k: float(FS[FS.construction == k].groupby("gene").fold.nunique().ge(3).sum() / (rc["n_up"] + rc["n_down"]))
            for k in rc["constructions"]}
    cv = T[(T.setting == "cross-fitted out-of-fold") & (T.tech == "pooled")].set_index("construction")
    chosen = cv.auc_dkd_hkd.idxmax()
    # 3. rebuild the chosen construction on all KPMP with the final models
    models = [load_state(cfg, fs["dir"] / (fs["stem"].format(s=s) + ".pt"), fs["K"], dev) for s in fs["seeds"]]
    up, down = build(chosen, cfg, data, L, M, genes, models, set(M.participant), rc, rng)
    pd.concat([pd.DataFrame({"gene": up, "direction": "up"}), pd.DataFrame({"gene": down, "direction": "down"})]
              ).to_csv(out / "transfer_signature.tsv", sep="\t", index=False)
    sel = {"chosen": chosen, "rule": rc["selection_metric"], "pooled_auc": cv.auc_dkd_hkd.to_dict(),
           "pooled_g": cv.g.to_dict(), "fold_gene_stability_frac_in_>=3_folds": stab, "n_genes_expressed": len(genes)}
    (out / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    pd.set_option("display.width", 250)
    print(T[["setting", "construction", "tech", "n1", "n2", "auc_dkd_hkd", "g", "ci_low", "ci_high", "p_wilcoxon"]].round(3).to_string())
    print(json.dumps(sel, indent=1, default=float)); print("up:", up[:25]); print("down:", down[:25])
    write_provenance("expr_transfer", [bulk / "kpmp_pt_hvg.h5ad"], [out / "internal_signature_tests.tsv", out / "selection.json",
                     out / "transfer_signature.tsv"], cfg["seed"], cfg, extra=sel, tag="signatures")


if __name__ == "__main__":
    main()
