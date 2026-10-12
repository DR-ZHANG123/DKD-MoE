"""Panels: comparison methods, model variants, technology mixing and forms of supervision (used by Fig2_model.py and FigS_comparison_external.py).

Original figure description — Fig. 3 — the mixture-of-experts readout against other approaches, model variants and forms of supervision.

Sources (results/bench_summary/, harmonised with the expression readout; external cohorts = mean of the five fold
models): summary_table.tsv (AUC / Hedges' g with 95% CI, means over seeds), paired_vs_moe_final.tsv (paired
participant bootstrap of the AUC difference vs the final model, on the participants both methods score);
results/bench_warm_mixing/technology_mixing.tsv (share of each held-out cell's 30 nearest neighbours from the same
technology, final model and comparison methods).
(a) KPMP held-out DKD vs HKD AUC and paired AUC difference vs the final model; (b) external Hedges' g
(GSE211785 DKD vs HKD, GSE195460 DKD vs control); (c) technology mixing; (d) model variants without one component and a
supervised model without experts (results/bench_supervised_single/; paired AUC difference vs the final model);
(e) cell-level, injured-cell and participant-level supervision: held-out AUC and the largest between-diagnosis
difference in mean routing of injured cells to one expert, in training and in held-out participants
(results/expr_supervision/summary.tsv)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _style as S

FINAL = "MoE final model readout"
ROWS = [(FINAL, "DKD-MoE: readout", "moe"),
        ("pseudobulk/DE_set", "Participant-level DE: gene set", "pseudobulk"),
        ("scvi/DE_set", "scVI: DE gene set", "scvi"),
        ("scvi/latent_LR", "scVI: latent classifier", "scvi"),
        ("scdisinfact/CKG_set", "scDisInFact: key genes", "scdisinfact"),
        ("scdisinfact/zd_LR", "scDisInFact: condition latent", "scdisinfact"),
        ("scdisinfact/latent_LR", "scDisInFact: shared latent", "scdisinfact"),
        ("cvae/decoded_set", "Conditional VAE: decoded genes", "cvae"),
        ("cvae/counterfactual_LLR", "Conditional VAE: counterfactual", "cvae"),
        ("cvae/latent_LR", "Conditional VAE: latent classifier", "cvae")]
FAM_COL = {"sup": "#8172B3", "moe": "#C44E52", "pseudobulk": "#8C8C8C", "scvi": "#4C72B0", "scdisinfact": "#937860", "cvae": "#DD8452"}
MIX = [("pseudobulk/PCA_uncorrected", "PCA, uncorrected", "pseudobulk"), ("scvi/scVI_latent", "scVI latent", "scvi"),
       ("scdisinfact/scDisInFact_common_latent", "scDisInFact shared latent", "scdisinfact"),
       ("cvae/CVAE_basal_latent", "Conditional VAE basal latent", "cvae"),
       ("Mixture of experts latent", "DKD-MoE latent", "moe"),
       ("Supervised single expert (condition head on latent mean)", "Supervised single expert latent", "sup")]
# readout of each variant: condition classifier log-odds; the single-expert variant has no routing, so its
# participant latent state is read by logistic regression
ABL = [("moe_warm_K1", "Single\nexpert", "latent_LR"), ("moe_warm_dense", "Dense\nrouting", "condition_head"),
       ("moe_warm_no_adv", "No\ndiscriminator", "condition_head"),
       ("moe_warm_no_cond", "No\nclassifier", "condition_head"),
       ("supervised_single_z", "Supervised,\nno experts", "condition_head")]


def table(Sm, P):
    recs = []
    for m, lab, fam in ROWS:
        r = Sm[Sm.method == m].iloc[0]; p = P[P.method == m].iloc[0]
        recs.append(dict(method=m, label=lab, family=fam, n_seeds=int(r.n_seeds),
                         kpmp_auc=r.KPMP_DKD_vs_HKD_auc, kpmp_auc_lo=r.KPMP_DKD_vs_HKD_auc_lo,
                         kpmp_auc_hi=r.KPMP_DKD_vs_HKD_auc_hi, n_dkd_paired=int(p.n_DKD), n_hkd_paired=int(p.n_HKD),
                         auc_diff=p.auc_diff_vs_moe_final, diff_lo=p.ci_lo, diff_hi=p.ci_hi,
                         g211=r.GSE211785_DKD_vs_HKD_g, g211_lo=r.GSE211785_DKD_vs_HKD_g_lo,
                         g211_hi=r.GSE211785_DKD_vs_HKD_g_hi, g195=r.GSE195460_DKD_vs_Control_g,
                         g195_lo=r.GSE195460_DKD_vs_Control_g_lo, g195_hi=r.GSE195460_DKD_vs_Control_g_hi))
    d = pd.DataFrame(recs)
    S.src_out(d.drop(columns="method"), "cmp_ab_methods")
    return d


def forest(ax, d, c, lo, hi, ref, xlabel, title, labels=False):
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        col = FAM_COL[r.family]
        ax.plot([getattr(r, lo), getattr(r, hi)], [yi, yi], color=col, lw=0.9)
        ax.scatter([getattr(r, c)], [yi], s=14, color=col, zorder=3, lw=0,
                   marker="D" if r.method == FINAL else "o")
    ax.axvline(ref, color=S.GREY, lw=0.6, ls="--")
    ax.axhspan(y[0] - 0.5, y[0] + 0.5, color="#C44E52", alpha=0.06, lw=0)
    lab = [r.label + ("" if r.n_dkd_paired == d.n_dkd_paired.iloc[0] else "†") for r in d.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(lab if labels else [], fontsize=5.8)
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel(xlabel, fontsize=6); ax.set_title(title, fontsize=6.3, pad=3)
    if not labels:
        ax.spines["left"].set_visible(False)


def panel_c(ax):
    M = S.read("bench_supervised_single/technology_mixing.tsv").set_index("method")
    recs = [dict(method=m, label=l, family=f, kNN30_same_tech=M.loc[m, "kNN30_same_tech"],
                 healthy_PT=M.loc[m, "kNN30_same_tech_healthy_PT"], injured_PT=M.loc[m, "kNN30_same_tech_injured_PT"])
            for m, l, f in MIX]
    d = S.src_out(pd.DataFrame(recs), "cmp_c_mixing")
    y = np.arange(len(d))[::-1]
    ax.barh(y, d.kNN30_same_tech, color=[FAM_COL[f] for f in d.family], height=0.6, ec="white", lw=0.3)
    ax.scatter(d.healthy_PT, y + 0.0, marker="|", s=40, color=S.INK, lw=0.8, zorder=3, label="Healthy PT")
    ax.scatter(d.injured_PT, y, marker="x", s=12, color=S.INK, lw=0.7, zorder=3, label="Injured PT")
    for yi, v in zip(y, d.kNN30_same_tech):
        ax.text(1.02, yi, f"{v:.2f}", va="center", ha="left", fontsize=5.5)
    ax.set_yticks(y); ax.set_yticklabels(d.label, fontsize=5.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(0.5, 1.12); ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    ax.set_xlabel("Same-technology share of 30 nearest\nneighbours (lower = better mixed)", fontsize=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, fontsize=5.6, frameon=False,
              handletextpad=0.3)


def panel_d(ax, P):
    recs = []
    P = pd.concat([P, S.read("bench_supervised_single/paired_vs_moe_final.tsv")])
    for v, vl, sc in ABL:
        r = P[P.method == f"{v}/{sc}"].iloc[0]
        recs.append(dict(variant=v, variant_label=vl.replace("\n", " "), score=sc, auc_diff=r.auc_diff_vs_moe_final,
                         ci_lo=r.ci_lo, ci_hi=r.ci_hi, n_dkd=int(r.n_DKD), n_hkd=int(r.n_HKD)))
    d = S.src_out(pd.DataFrame(recs), "cmp_d_variants")
    y = np.arange(len(d))[::-1]
    ax.errorbar(d.auc_diff, y, xerr=[d.auc_diff - d.ci_lo, d.ci_hi - d.auc_diff], fmt="o", ms=3.6, color="#C44E52",
                ecolor="#C44E52", elinewidth=0.8, capsize=1.6)
    ax.axvline(0, color=S.GREY, lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([vl.replace("\n", " ") for _, vl, _ in ABL], fontsize=5.8)
    ax.tick_params(axis="y", length=0); ax.set_ylim(-0.6, len(ABL) - 0.4)
    ax.set_xlabel("AUC difference vs final model\n(paired bootstrap 95% CI)", fontsize=6.3)
    n1, n2 = d.n_dkd.max(), d.n_hkd.max()
    ax.text(0.03, 0.03, f"KPMP held-out,\n{n1} DKD vs {n2} HKD", transform=ax.transAxes, ha="left", va="bottom",
            fontsize=5.5, color="#555555")


SUP = [("cell", "Cell level\n(final model)"), ("injured", "Injured cells\nonly"), ("patient", "Participant\nlevel")]


def panel_e(ax1, ax2):
    T = S.read("expr_supervision/summary.tsv").set_index("supervision")
    d = S.src_out(T.loc[[k for k, _ in SUP]].reset_index(), "cmp_e_supervision")
    y = np.arange(len(SUP))[::-1]
    for yi, r in zip(y, d.itertuples()):
        ax1.plot([r.auc_lo, r.auc_hi], [yi, yi], color="#C44E52", lw=0.9)
        ax1.scatter([r.auc_heldout], [yi], s=16, color="#C44E52", zorder=3, lw=0)
        ax1.text(0.97, yi, f"{r.auc_heldout:.2f}", va="center", ha="left", fontsize=5.6)
    ax1.axvline(0.5, color=S.GREY, lw=0.6, ls="--")
    ax1.set_yticks(y); ax1.set_yticklabels([l for _, l in SUP], fontsize=5.8); ax1.tick_params(axis="y", length=0)
    ax1.set_xlim(0.45, 1.06); ax1.set_ylim(-0.6, len(SUP) - 0.4)
    ax1.set_xlabel("Held-out AUC\n(95% CI)", fontsize=6.3)
    h = 0.36
    ax2.barh(y + h / 2, d.routing_gap_in_sample, height=h, color="#8C8C8C", label="Training participants")
    ax2.barh(y - h / 2, d.routing_gap_heldout, height=h, color="#C44E52", label="Held-out participants")
    ax2.set_yticks(y); ax2.set_yticklabels([]); ax2.tick_params(axis="y", length=0); ax2.set_ylim(-0.6, len(SUP) - 0.4)
    ax2.set_xlim(0, 1.0)
    ax2.set_xlabel("Largest DKD-HKD routing\ndifference, injured cells", fontsize=6.3)
    ax2.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=1, fontsize=5.6, frameon=False, handlelength=1.0)
