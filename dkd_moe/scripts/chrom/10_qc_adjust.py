"""10_qc_adjust: chromatin-side readout and ATAC quality at the participant level (no retraining).

Inputs: results/chrom_readout/participant_scores.tsv (participant means over injured-PT nuclei of the
held-out chromatin-side score, the RNA-side readout and per-nucleus QC: log10 fragments, TSS enrichment,
FRiP; n_cells = injured-PT nuclei) and results/chrom_align/nuclei.tsv.gz (all PT nuclei with QC).
Analyses (participants are the units):
 (a) partial Spearman chrom_score ~ rna_readout | QC (ranks residualised on ranked covariates), all
     participants and DKD+HKD;
 (b) group contrasts DKD-HKD, DKD-Reference, HKD-Reference for chrom_score and rna_readout from OLS
     score ~ group + QC. Primary: model fitted on all participants (every diagnostic group as a factor
     level, QC slopes estimated from all); sensitivity: model fitted on the two contrasted groups only,
     and a rank-transformed outcome. Effects in SD units = coefficient / residual SD of the model
     (unadjusted model: residual SD of score ~ group).
 (c) QC by group: participant-level Wilcoxon of each QC metric (DKD vs HKD etc.) and nucleus-level
     medians (descriptive only).
QC covariate sets: 'injPT' = means over injured-PT nuclei (as in the readout table); 'allPT' = means over
all PT nuclei of the participant (closer to library quality, less tied to injured-cell biology).
Outputs: dkd_moe/results/chrom_qc_adjusted/{partial_assoc.tsv, group_contrasts.tsv, qc_by_group.tsv,
         qc_nucleus_medians.tsv, participant_table.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, result_dir, write_provenance, set_global_seed  # noqa: E402

QC = ["tss_enrichment", "frip", "log10_fragments", "log_n_cells"]
COMPS = [("DKD", "HKD"), ("DKD", "Reference"), ("HKD", "Reference")]
SCORES = ["chrom_score", "rna_readout"]


def partial_spearman(x, y, Z):
    """Pearson correlation of rank residuals; t-test with n-2-k df, Fisher-z CI with n-3-k."""
    rk = lambda v: stats.rankdata(v)
    X = np.column_stack([np.ones(len(x))] + [rk(z) for z in Z.T]) if Z.size else np.ones((len(x), 1))
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    rx, ry = res(rk(x)), res(rk(y))
    r = float(np.corrcoef(rx, ry)[0, 1]); n = len(x); k = X.shape[1] - 1
    df = n - 2 - k
    t = r * np.sqrt(df / max(1e-12, 1 - r ** 2)); p = 2 * stats.t.sf(abs(t), df)
    se = 1 / np.sqrt(max(1, n - 3 - k)); z = np.arctanh(r)
    return {"n": n, "k_covariates": k, "rho": r, "ci_low": float(np.tanh(z - 1.96 * se)),
            "ci_high": float(np.tanh(z + 1.96 * se)), "p": float(p)}


def ols(y, X):
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta; n, k = X.shape; df = n - k
    s2 = r @ r / df; XtXi = np.linalg.pinv(X.T @ X)
    se = np.sqrt(np.diag(XtXi) * s2)
    h = np.einsum("ij,jk,ik->i", X, XtXi, X)                      # HC3 robust covariance
    u = np.where(h < 1 - 1e-8, r / np.clip(1 - h, 1e-8, None), 0.0)   # single-member groups: zero residual
    meat = X.T @ (X * u[:, None] ** 2)
    se_hc3 = np.sqrt(np.diag(XtXi @ meat @ XtXi))
    return beta, se, se_hc3, df, np.sqrt(s2)


def contrast(t, score, a, b, cov, fit_on, rank=False):
    d = t if fit_on == "all" else t[t.group.isin([a, b])]
    d = d.dropna(subset=[score] + cov)
    y = d[score].values.astype(float)
    if rank:
        y = stats.rankdata(y)
    lv = [b] + sorted(set(d.group) - {b})                         # reference level = b
    G = np.column_stack([(d.group == g).values.astype(float) for g in lv[1:]])
    C = (d[cov].values - d[cov].values.mean(0)) / d[cov].values.std(0) if cov else np.empty((len(d), 0))
    X = np.column_stack([np.ones(len(d)), G, C])
    beta, se, se3, df, sd = ols(y, X)
    j = 1 + lv[1:].index(a); tq = stats.t.ppf(0.975, df)
    return {"estimate": beta[j], "se": se[j], "ci_low": beta[j] - tq * se[j], "ci_high": beta[j] + tq * se[j],
            "p": 2 * stats.t.sf(abs(beta[j] / se[j]), df), "p_hc3": 2 * stats.t.sf(abs(beta[j] / se3[j]), df),
            "resid_sd": sd, "d_sd": beta[j] / sd, "d_ci_low": (beta[j] - tq * se[j]) / sd,
            "d_ci_high": (beta[j] + tq * se[j]) / sd, "n_a": int((d.group == a).sum()), "n_b": int((d.group == b).sum()),
            "n_model": len(d), "df": df}


def allpt_qc(nuc):
    nuc = nuc.assign(log10_fragments=np.log10(nuc.n_fragments.astype(float)))
    g = nuc.groupby("participant")
    q = g[["tss_enrichment", "frip", "log10_fragments"]].mean()
    q["n_pt_nuclei"] = g.size(); q["injured_frac"] = g.pt_state.apply(lambda s: (s == "injured_PT").mean())
    return q.add_suffix("_allPT").reset_index()


def main():
    c = cfg(); set_global_seed(c["seed"]); out = result_dir("chrom_qc_adjusted")
    src = result_dir("chrom_readout") / "participant_scores.tsv"; nf = result_dir("chrom_align") / "nuclei.tsv.gz"
    T = pd.read_csv(src, sep="\t", dtype={"participant": str})
    nuc = pd.read_csv(nf, sep="\t", index_col=0, dtype={"participant": str})
    T = T.merge(allpt_qc(nuc), on="participant", how="left")
    T["log_n_cells"] = np.log10(T.n_cells.astype(float))
    T["log_n_pt_nuclei_allPT"] = np.log10(T.n_pt_nuclei_allPT.astype(float))
    covsets = {"injPT": QC, "allPT": ["tss_enrichment_allPT", "frip_allPT", "log10_fragments_allPT", "log_n_cells"],
               "none": []}
    PA, GC, QB = [], [], []
    for ft, t in T.groupby("feature"):
        for sub, q in (("all", t), ("DKD+HKD", t[t.group.isin(["DKD", "HKD"])])):
            for cs, cov in covsets.items():
                Z = q[cov].values.astype(float) if cov else np.empty((len(q), 0))
                PA.append({"feature": ft, "participants": sub, "covariates": cs,
                           **partial_spearman(q.chrom_score.values, q.rna_readout.values, Z)})
            for sc in SCORES:                                       # score vs each QC metric, adjusted for group-free
                for m in QC:
                    r = stats.spearmanr(q[sc], q[m])
                    PA.append({"feature": ft, "participants": sub, "covariates": f"marginal:{sc}~{m}", "n": len(q),
                               "rho": r.statistic, "p": r.pvalue})
        for sc in SCORES:
            for a, b in COMPS:
                for cs, cov in covsets.items():
                    for fit_on in ("all", "pair"):
                        for rank in (False, True):
                            GC.append({"feature": ft, "score": sc, "comparison": f"{a}_vs_{b}", "covariates": cs,
                                       "fit_on": fit_on, "outcome": "rank" if rank else "raw",
                                       **contrast(t, sc, a, b, cov, fit_on, rank)})
        for a, b in COMPS:
            for m in QC + ["n_cells", "tss_enrichment_allPT", "frip_allPT", "log10_fragments_allPT",
                           "n_pt_nuclei_allPT", "injured_frac_allPT"]:
                x, y = t.loc[t.group == a, m].dropna(), t.loc[t.group == b, m].dropna()
                QB.append({"feature": ft, "comparison": f"{a}_vs_{b}", "qc": m, "n_a": len(x), "n_b": len(y),
                           "median_a": x.median(), "median_b": y.median(), "iqr_a": f"{x.quantile(.25):.4g}-{x.quantile(.75):.4g}",
                           "iqr_b": f"{y.quantile(.25):.4g}-{y.quantile(.75):.4g}",
                           "p_wilcoxon": stats.mannwhitneyu(x, y).pvalue})
    # nucleus-level QC medians (descriptive), participants of the readout table, by group x PT state
    keep = set(T.participant)
    nn = nuc[nuc.participant.isin(keep)].assign(log10_fragments=lambda d: np.log10(d.n_fragments.astype(float)))
    NM = (nn.groupby(["group", "pt_state"])[["tss_enrichment", "frip", "log10_fragments"]]
          .median().join(nn.groupby(["group", "pt_state"]).size().rename("n_nuclei"))
          .join(nn.groupby(["group", "pt_state"]).participant.nunique().rename("n_participants")).reset_index())
    PA, GC, QB = pd.DataFrame(PA), pd.DataFrame(GC), pd.DataFrame(QB)
    files = {"partial_assoc.tsv": PA, "group_contrasts.tsv": GC, "qc_by_group.tsv": QB,
             "qc_nucleus_medians.tsv": NM, "participant_table.tsv": T}
    for k, v in files.items():
        v.to_csv(out / k, sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(PA[~PA.covariates.str.startswith("marginal")].round(4).to_string())
    print(GC[(GC.outcome == "raw")].drop(columns=["se", "df"]).round(3).to_string())
    print(QB.round(4).to_string()); print(NM.round(3).to_string())
    write_provenance("chrom_qc_adjusted", [src, nf], [out / k for k in files], c["seed"], c,
                     extra={"qc_covariates": covsets})


if __name__ == "__main__":
    main()
