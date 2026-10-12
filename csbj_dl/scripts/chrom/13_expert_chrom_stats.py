"""13_expert_chrom_stats: participant-level tests of the expert components and motif accessibility in chromatin.

Participant means over injured-PT nuclei (>= rna.min_cells) of the RNA-side and chromatin-side M1 / M3
contributions (11_expert_chromatin) and motif scores (12_motif_activity), joined with the chromatin-side total
score (05_chrom_readout) and participant ATAC quality and injured fraction over all PT nuclei (10_qc_adjust).
 (a) agreement: Spearman of chromatin-side with RNA-side contributions, and partial Spearman adjusting for the
     RNA-side total readout and ATAC quality (ranks residualised);
 (b) group differences (Hedges g, Wilcoxon) for expert.comparisons;
 (c) OLS (HC3) of each standardized score on group (reference HKD) with ATAC quality and injured fraction,
     with and without the RNA-side total readout, in the recruitment-site disease groups;
 (d) motif scores against the RNA-side M3 and M1 contributions: Spearman and partial Spearman adjusting for ATAC
     quality, injured fraction and the RNA-side total readout; Benjamini-Hochberg q over motifs x components within
     each participant set.
Outputs: csbj_dl/results/chrom_expert/{participant_table.tsv, agreement.tsv, group_tests.tsv, group_models.tsv,
         motif_vs_expert.tsv} + PROVENANCE_stats.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, result_dir, write_provenance, participant_table, tests  # noqa: E402

QC = ["log10_fragments_allPT", "tss_enrichment_allPT", "frip_allPT"]


def partial_spearman(d, x, y, cov):
    q = d.dropna(subset=[x, y] + cov)
    Z = np.column_stack([np.ones(len(q))] + [q[k].rank().values for k in cov])
    res = lambda v: v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
    r = stats.pearsonr(res(q[x].rank().values), res(q[y].rank().values))
    return r.statistic, r.pvalue, len(q)


def main():
    c = cfg(); E = c["expert"]; out = result_dir("chrom_expert"); mc = c["rna"]["min_cells"]
    A = pd.read_csv(out / "nucleus_scores.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    B = pd.read_csv(result_dir("chrom_motif_activity") / "nucleus_scores.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    motifs = [k for k in B.columns if k not in A.columns]
    vals = {k: A[k].values for k in A.columns if k.startswith(("rna_", "chrom_", "input_"))}
    vals.update({m: B.loc[A.index, m].values for m in motifs})
    T = participant_table(A, vals, min_cells=mc)
    Q = pd.read_csv(result_dir("chrom_qc_adjusted") / "participant_table.tsv", sep="\t", dtype={"participant": str})
    Q = Q[Q.feature == "peaks"].set_index("participant")[QC + ["injured_frac_allPT", "rna_readout", "chrom_score"]]
    T = T.set_index("participant").join(Q).reset_index()
    T.to_csv(out / "participant_table.tsv", sep="\t", index=False)
    comps = [tuple(x) for x in E["comparisons"]]
    scores = [f"{p}_{m}" for m in E["components"] for p in ("rna", "chrom")] + motifs + ["chrom_score", "rna_readout"]
    G = tests(T, comps, scores); G.to_csv(out / "group_tests.tsv", sep="\t", index=False)
    ag = []
    for m in E["components"]:
        for sub, d in (("all", T), ("disease", T[T.group.isin(E["groups"])])):
            r = stats.spearmanr(d[f"chrom_{m}"], d[f"rna_{m}"], nan_policy="omit")
            pr = partial_spearman(d, f"chrom_{m}", f"rna_{m}", ["rna_readout"] + QC)
            ag.append({"component": m, "participants": sub, "n": int(d[[f"chrom_{m}", f"rna_{m}"]].dropna().shape[0]),
                       "spearman": r.statistic, "p": r.pvalue, "partial_spearman_total_readout_quality": pr[0], "partial_p": pr[1]})
    pd.DataFrame(ag).to_csv(out / "agreement.tsv", sep="\t", index=False)
    D = T[T.group.isin(E["groups"])].copy(); rows = []
    base = " + ".join(QC + ["injured_frac_allPT"])
    for s in scores:
        D["y"] = (D[s] - D[s].mean()) / D[s].std()
        for name, f in (("quality_injury", f"y ~ C(group, Treatment('HKD')) + {base}"),
                        ("quality_injury_total", f"y ~ C(group, Treatment('HKD')) + {base} + rna_readout")):
            if s == "rna_readout" and name.endswith("total"):
                continue
            q = D.dropna(subset=["y"] + QC + ["injured_frac_allPT", "rna_readout"])
            mfit = smf.ols(f, q).fit(cov_type="HC3"); ci = mfit.conf_int()
            for t in [t for t in mfit.params.index if t.startswith("C(group")]:
                rows.append({"score": s, "model": name, "term": t.split("[T.")[1].rstrip("]"), "beta_sd": mfit.params[t],
                             "ci_low": ci.loc[t, 0], "ci_high": ci.loc[t, 1], "p": mfit.pvalues[t], "n": int(mfit.nobs)})
            k = "C(group, Treatment('HKD'))"
            ct = mfit.t_test(f"{k}[T.DKD] - {k}[T.DM_noCKD] = 0")      # DKD versus diabetes without CKD
            lo_, hi_ = ct.conf_int()[0]
            rows.append({"score": s, "model": name, "term": "DKD_minus_DM_noCKD", "beta_sd": float(ct.effect[0]),
                         "ci_low": float(lo_), "ci_high": float(hi_), "p": float(ct.pvalue), "n": int(mfit.nobs)})
    GM = pd.DataFrame(rows); GM.to_csv(out / "group_models.tsv", sep="\t", index=False)
    mv = []
    for mo in motifs:
        for m in E["components"]:
            for sub, d in (("all", T), ("disease", T[T.group.isin(E["groups"])])):
                r = stats.spearmanr(d[mo], d[f"rna_{m}"], nan_policy="omit")
                pr = partial_spearman(d, mo, f"rna_{m}", QC + ["injured_frac_allPT", "rna_readout"])
                mv.append({"motif": mo, "component": m, "participants": sub, "spearman": r.statistic, "p": r.pvalue,
                           "partial_spearman": pr[0], "partial_p": pr[1], "n": pr[2]})
    MV = pd.DataFrame(mv)
    for col in ("p", "partial_p"):          # Benjamini-Hochberg over motifs x components within each participant set
        MV[col.replace("p", "q", 1) if col == "p" else "partial_q"] = MV.groupby("participants")[col].transform(
            lambda v: multipletests(v, method="fdr_bh")[1])
    MV.to_csv(out / "motif_vs_expert.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 400)
    print(T.groupby("group").size()); print(pd.DataFrame(ag).round(3).to_string())
    print(G[G.feature.isin(scores)][["comparison", "feature", "n1", "n2", "g", "p_wilcoxon"]].round(3).to_string())
    print(GM.round(3).to_string()); print(MV.round(3).to_string())
    write_provenance("chrom_expert", [out / "nucleus_scores.tsv.gz", result_dir("chrom_motif_activity") / "nucleus_scores.tsv.gz",
                                      result_dir("chrom_qc_adjusted") / "participant_table.tsv"],
                     [out / f for f in ("participant_table.tsv", "agreement.tsv", "group_tests.tsv", "group_models.tsv",
                                        "motif_vs_expert.tsv")], c["seed"], c, tag="stats")


if __name__ == "__main__":
    main()
