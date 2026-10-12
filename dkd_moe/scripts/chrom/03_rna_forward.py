"""03_rna_forward: the fixed expression model on Multiome RNA proximal-tubule nuclei.

Per nucleus and seed: routing probabilities and the condition-head DKD-vs-HKD log-odds from (a) the
cross-fitted fold model whose training excluded the nucleus's participant (primary, since the Multiome GEX
nuclei are part of the KPMP snRNA training data) and (b) the final model (all KPMP); meta-expert contributions
(head weight x routing; final models, expr_meta units). All 3,000 input genes are measured in Multiome RNA.
Participant readout = mean over injured-PT nuclei, then over seeds. Agreement with the KPMP sc/sn held-out
readout (expr_readout) is reported per participant and on Multiome participants absent from each technology.
Outputs: dkd_moe/results/chrom_rna/{nucleus_readout.tsv.gz, participant_readout.tsv, tests.tsv,
         agreement_with_expr.tsv, consistency_check.json, gene_coverage.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import (cfg, P, result_dir, write_provenance, set_global_seed, pick_device, expr_model,  # noqa: E402
                          participant_table, tests)
from chrom_data import load_multiome  # noqa: E402
from model import log_norm, cond_logodds  # noqa: E402


@torch.no_grad()
def forward(m, X, device, bs=4096):
    P_, L_ = [], []
    for s in range(0, X.shape[0], bs):
        x = torch.from_numpy(X[s:s + bs].toarray()).to(device)
        h = m.encode_h(log_norm(x)); p, _ = m.gate_probs(h)
        P_.append(p.cpu().numpy()); L_.append(cond_logodds(m, p).cpu().numpy())
    return np.concatenate(P_), np.concatenate(L_)


def main():
    c = cfg(); R = c["rna"]; set_global_seed(c["seed"]); out = result_dir("chrom_rna")
    dev = pick_device(R["device"]); print("device", dev, flush=True)
    genes = pd.read_csv(P(c, "hvg"), sep="\t").gene.tolist()
    mvars = ad.read_h5ad(P(c, "multiome_rna"), backed="r").var_names
    miss = [g for g in genes if g not in set(mvars)]
    cov = {"n_input_genes": len(genes), "n_in_merged_object": len(genes) - len(miss),
           "read_by_symbol_from_library_files": miss,
           "note": "genes absent from the merged object (reference-specific Ensembl ids) are read by symbol from every "
                   "library's own RNA file; load fails if any library lacks a unique match (no zero-filling)"}
    obs, X, _, _ = load_multiome(c, rna_genes=genes, peaks=False)
    cov["median_hvg_counts_per_nucleus"] = float(np.median(np.asarray(X.sum(1)).ravel()))
    (out / "gene_coverage.json").write_text(json.dumps(cov, indent=2))
    fold = obs.fold.values
    U = pd.read_csv(P(c, "expr_meta_units"), sep="\t"); M = pd.read_csv(P(c, "expr_meta"), sep="\t")
    metas = M.meta_expert.tolist()
    cols_ = {}
    for s in R["seeds"]:
        lo_cf = np.full(len(obs), np.nan)
        for f in sorted(pd.unique(fold[~pd.isna(fold)]).astype(int)):
            idx = np.where(fold == f)[0]
            m, _ = expr_model(f, s, dev)
            _, lo_cf[idx] = forward(m, X[idx], dev)
        mf, _ = expr_model(None, s, dev)
        pf, lo_f = forward(mf, X, dev)
        cols_[f"cf_s{s}"] = lo_cf; cols_[f"final_s{s}"] = lo_f
        u = U[U.seed == s]
        for meta in metas:
            v = np.zeros(len(obs))
            for _, r in u[u.meta_expert == meta].iterrows():
                v += r.head_weight_dkd_minus_hkd * pf[:, int(r.expert)]
            cols_[f"{meta}_s{s}"] = v
        print("seed", s, "done", flush=True)
    S = R["seeds"]
    nuc = pd.DataFrame(cols_, index=obs.index)
    nuc["readout_cf"] = nuc[[f"cf_s{s}" for s in S]].mean(1)
    nuc["readout_final"] = nuc[[f"final_s{s}" for s in S]].mean(1)
    for meta in metas:
        nuc[meta] = nuc[[f"{meta}_s{s}" for s in S]].mean(1)
    nuc = pd.concat([obs[["participant", "library", "group", "fold", "pt_state", "matched", "sn_cell"]], nuc], axis=1)
    nuc.to_csv(out / "nucleus_readout.tsv.gz", sep="\t")
    # consistency with the expression module's held-out predictions for the same (matched) nuclei
    check = consistency(c, nuc)
    (out / "consistency_check.json").write_text(json.dumps(check, indent=2))
    # participant level
    cols = ["readout_cf", "readout_final"] + [f"cf_s{s}" for s in S] + metas
    t = participant_table(nuc, {k: nuc[k].values for k in cols}, min_cells=R["min_cells"])
    sp_ = pd.read_csv(P(c, "splits"), sep="\t", dtype={"participant": str}).set_index("participant")
    t["in_sc"] = sp_.in_sc.reindex(t.participant).fillna(0).astype(int).values
    t["in_sn"] = sp_.in_sn.reindex(t.participant).fillna(0).astype(int).values
    t.to_csv(out / "participant_readout.tsv", sep="\t", index=False)
    comps = c["kpmp_comparisons"]
    T = [tests(t, comps, ["readout_cf", "readout_final"] + metas, {"subset": "all_multiome"})]
    for tech in ("sc", "sn"):
        q = t[t[f"in_{tech}"] == 0]
        T.append(tests(q, comps, ["readout_cf"], {"subset": f"not_in_kpmp_{tech}"}))
    T = pd.concat(T, ignore_index=True); T.to_csv(out / "tests.tsv", sep="\t", index=False)
    # agreement with the KPMP sc / sn held-out participant readout
    E = pd.read_csv(P(c, "expr_readout"), sep="\t", dtype={"participant": str})
    E = E[E.cohort == "KPMP_heldout"]
    ag = []
    for tech in ("sc", "sn", "pooled"):
        e = E[E.tech == tech].set_index("participant").logodds
        j = t.set_index("participant").join(e.rename("expr"), how="inner")
        for grp in (None, ["DKD", "HKD"]):
            jj = j if grp is None else j[j.group.isin(grp)]
            if len(jj) >= 3:
                r = stats.spearmanr(jj.readout_cf, jj.expr)
                ag.append({"expr_tech": tech, "participants": "all" if grp is None else "DKD+HKD", "n": len(jj),
                           "spearman": r.statistic, "p": r.pvalue})
    ag = pd.DataFrame(ag); ag.to_csv(out / "agreement_with_expr.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(json.dumps(check, indent=1)); print(t.groupby("group").size())
    print(T[T.feature.isin(["readout_cf", "readout_final"])][["subset", "comparison", "feature", "n1", "n2", "g",
                                                               "ci_low", "ci_high", "p_wilcoxon"]].round(3).to_string())
    print(ag.round(3).to_string())
    write_provenance("chrom_rna", [P(c, "multiome_rna"), result_dir("chrom_labels") / "multiome_labels.tsv.gz"],
                     [out / f for f in ("nucleus_readout.tsv.gz", "participant_readout.tsv", "tests.tsv",
                                        "agreement_with_expr.tsv", "consistency_check.json", "gene_coverage.json")],
                     c["seed"], c, extra={**cov, **check, "device": dev})


def consistency(c: dict, nuc: pd.DataFrame) -> dict:
    """Correlation of this forward pass with the expression module's saved held-out log-odds (seed 1)."""
    R = c["rna"]; ck = Path(c["paths"]["expr_ckpt"]) / R["expr_prefix_cv"]
    k = ad.read_h5ad(Path(c["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs_names
    pos = pd.Series(np.arange(len(k)), index=k)
    a, b = [], []
    for f in range(c["n_folds"]):
        z = np.load(ck / f"{R['expr_prefix_cv']}_K{R['expr_K']}_wc{R['expr_wc']}_s1_f{f}.npz")
        saved = pd.Series(z["logodds"], index=z["idx"])
        q = nuc[(nuc.fold == f) & nuc.sn_cell.notna()]
        i = pd.Series(pos.reindex(q.sn_cell.values).values, index=q.index).dropna().astype(int)
        hit = i[i.isin(saved.index)]
        a.append(q.loc[hit.index, "cf_s1"].values); b.append(saved.loc[hit.values].values)
    a, b = np.concatenate(a), np.concatenate(b)
    if len(a) < 3:
        return {"n_matched_kpmp_pt_cells": int(len(a))}
    return {"n_matched_kpmp_pt_cells": int(len(a)), "pearson_vs_saved_heldout_logodds": float(np.corrcoef(a, b)[0, 1]),
            "max_abs_diff": float(np.abs(a - b).max())}


if __name__ == "__main__":
    main()
