"""Panels of the readout-derived signature (used by FigS_signature.py; source names start with sig_):
box   cross-fitted signature score of held-out KPMP participants, DKD versus HKD by technology
      (results/expr_transfer/oof_scores.tsv, internal_signature_tests.tsv rows 'cross-fitted out-of-fold');
genes the 12 highest-ranked up- and down-regulated genes (results/expr_transfer/transfer_signature.tsv), shaded by the
      number of training parts that selected each gene (fold_signatures.tsv); genes expressed mainly outside the
      proximal tubule marked (results/expr_pt_signature/gene_sources.tsv);
forest independent cohorts, readout and signature (results/expr_readout/donor_level_tests.tsv,
      results/expr_transfer/{external_signature_tests,external_meta_analysis}.tsv, results/expr_pt_signature/external_tests.tsv)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import _style as S

COMPLEMENT = {"C3", "C1S", "C1R", "CFH"}
MATRIX = {"VCAN", "LAMA4", "NPNT", "ADAM12", "TGM2", "TNFAIP6", "COL23A1", "TIMP3"}


def box(ax):
    O = S.read("expr_transfer/oof_scores.tsv", index_col=0)
    O = O[["readout_assoc"]].rename(columns={"readout_assoc": "score"})
    O["participant"] = [i.split("|")[0] for i in O.index]; O["tech"] = [i.split("|")[1] for i in O.index]
    grp = S.read("../config/splits.tsv", dtype={"participant": str}).set_index("participant").group
    O["group"] = O.participant.map(grp)
    d = O[O.group.isin(["HKD", "DKD"])].reset_index(drop=True)
    S.src_out(d[["participant", "tech", "group", "score"]], "sig_box_scores")
    T = S.read("expr_transfer/internal_signature_tests.tsv")
    T = T[T.setting.str.startswith("cross-fitted") & (T.construction == "readout_assoc")].set_index("tech")
    S.src_out(T.reset_index()[["tech", "auc_dkd_hkd", "g", "ci_low", "ci_high", "p_wilcoxon", "n1", "n2"]],
              "sig_box_tests")
    pos, data, cols, labs = [], [], [], []
    for k, tech in enumerate(["sc", "sn"]):
        for j, g in enumerate(["HKD", "DKD"]):
            x = d[(d.tech == tech) & (d.group == g)].score.values
            pos.append(k * 2.6 + j); data.append(x); cols.append(S.GROUP_COL[g]); labs.append(f"{g}\n(n = {len(x)})")
    S.strip_box(ax, data, pos, cols, s=5)
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=6)
    top = max(np.max(x) for x in data); span = top - min(np.min(x) for x in data)
    for k, tech in enumerate(["sc", "sn"]):
        r = T.loc[tech]
        x1, x2, y = k * 2.6, k * 2.6 + 1, top + 0.08 * span
        ax.plot([x1, x1, x2, x2], [y - 0.03 * span, y, y, y - 0.03 * span], color=S.INK, lw=0.6)
        ax.text((x1 + x2) / 2, y + 0.02 * span, f"AUC {r.auc_dkd_hkd:.2f}\n{S.fmt_p(r.p_wilcoxon)}",
                ha="center", va="bottom", fontsize=5.8)
        ax.text((x1 + x2) / 2, top + 0.42 * span, S.TECH_LABEL[tech], ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")
    ax.set_ylim(ax.get_ylim()[0], top + 0.58 * span)
    ax.set_ylabel("Signature score\n(held-out participants)")


def genes(ax, n_show=12):
    sig = S.read("expr_transfer/transfer_signature.tsv")
    fs = S.read("expr_transfer/fold_signatures.tsv"); fs = fs[fs.construction == "readout_assoc"]
    nf = fs.groupby(["gene", "direction"]).fold.nunique(); nfold = fs.fold.nunique()
    src = S.read("expr_pt_signature/gene_sources.tsv").groupby("gene").pt_ratio.min()
    sig["rank"] = sig.groupby("direction").cumcount() + 1
    sig["n_folds"] = [int(nf.get((g, d), 0)) for g, d in zip(sig.gene, sig.direction)]
    sig["min_pt_ratio"] = sig.gene.map(src)
    sig["annotation"] = ["complement" if g in COMPLEMENT else "extracellular matrix" if g in MATRIX else ""
                         for g in sig.gene]
    S.src_out(sig, "sig_genes")
    sig = sig[sig["rank"] <= n_show]
    blocks = [("up", "Higher in DKD", "#C44E52", plt.cm.Reds), ("down", "Lower in DKD", "#4C72B0", plt.cm.Blues)]
    ncol = n_show
    for row, (d, title, col, cmap) in enumerate(blocks):
        g = sig[sig.direction == d].reset_index(drop=True); y = row * 2.0
        ax.text(-0.5, y - 0.62, f"{title} (top {len(g)} of 50)", ha="left", va="bottom", fontsize=6.3,
                fontweight="bold", color=col)
        for i, r in g.iterrows():
            fc = cmap(0.15 + 0.6 * r.n_folds / nfold)
            ec = {"complement": "#6A3D9A", "extracellular matrix": "#111111"}.get(r.annotation, "white")
            ax.add_patch(Rectangle((i - 0.47, y - 0.42), 0.94, 0.84, fc=fc, ec=ec, lw=1.3 if r.annotation else 0.5))
            lab = r.gene + ("*" if r.min_pt_ratio < 0.5 else "")
            ax.text(i, y, lab, ha="center", va="center", fontsize=5.2, fontstyle="italic",
                    color="white" if r.n_folds / nfold > 0.75 else S.INK)
    ax.set_xlim(-0.55, ncol - 0.45); ax.axis("off")
    ly = 3.15
    ax.add_patch(Rectangle((-0.45, ly - 0.22), 0.6, 0.45, fc="white", ec="#6A3D9A", lw=1.3))
    ax.text(0.3, ly, "complement", va="center", fontsize=5.8)
    ax.add_patch(Rectangle((2.3, ly - 0.22), 0.6, 0.45, fc="white", ec="#111111", lw=1.3))
    ax.text(3.05, ly, "extracellular matrix", va="center", fontsize=5.8)
    ax.text(5.9, ly, "* expressed mainly outside the proximal tubule", va="center", fontsize=5.8)
    ax.text(-0.45, ly + 0.8, f"Shading: number of training parts (of {nfold}) selecting the gene", va="center",
            ha="left", fontsize=5.8)
    ax.set_ylim(ly + 1.15, -1.1)


def forest(ax):
    T = S.read("expr_transfer/external_signature_tests.tsv")
    T = T[(T.signature == "transfer_readout_assoc") & T.g.notna()]
    A = S.read("expr_pt_signature/external_tests.tsv")
    A = A[A.construction == "marker_adjusted"]
    RT = S.read("expr_readout/donor_level_tests.tsv")
    Mm = S.read("expr_transfer/external_meta_analysis.tsv")
    Mm = Mm[Mm.signature == "transfer_readout_assoc"].set_index("summary")

    def sig_row(D, coh, plat, comp, ct=""):
        return D[(D.cohort == coh) & (D.platform == plat) & (D.comparison == comp) & (D.celltype.fillna("") == ct)].iloc[0]
    rows = []
    for lab, grp, r in [
            ("ERCB tubulointerstitium, DKD vs hypertensive nephropathy: signature", "dis", sig_row(T, "GSE104954", "GPL24120", "DKD_vs_HTN")),
            ("    signature adjusted for non-PT marker genes", "dis", sig_row(A, "GSE104954", "GPL24120", "DKD_vs_HTN")),
            ("GSE211785, DKD vs HKD: readout", "dis", RT[(RT.cohort == "gse211785") & (RT.tech == "pooled") & (RT.comparison == "DKD_vs_HKD")].iloc[0]),
            ("GSE211785, DKD vs HKD: signature", "dis", sig_row(T, "GSE211785", "pooled", "DKD_vs_HKD")),
            ("Xenium injured PT, DKD vs other CKD: signature", "dis", sig_row(T, "Zenodo19868428", "Xenium", "DKD_vs_other_CKD", "iPT")),
            ("ERCB, DKD vs living donor (GPL24120): signature", "hea", sig_row(T, "GSE104954", "GPL24120", "DKD_vs_LD")),
            ("ERCB, DKD vs living donor (GPL22945): signature", "hea", sig_row(T, "GSE104954", "GPL22945", "DKD_vs_LD")),
            ("GSE211785, DKD vs control: readout", "hea", RT[(RT.cohort == "gse211785") & (RT.tech == "pooled") & (RT.comparison == "DKD_vs_Control")].iloc[0]),
            ("GSE195460, DKD vs control: readout", "hea", RT[(RT.cohort == "gse195460") & (RT.tech == "pooled") & (RT.comparison == "DKD_vs_Control")].iloc[0]),
            ("GSE195460, DKD vs control: signature", "hea", sig_row(T, "GSE195460", "sn", "DKD_vs_Control"))]:
        rows.append((lab, grp, "study", r.n1, r.n2, r.g, r.ci_low, r.ci_high, r.p_wilcoxon))
    for key, grp, lab in (("DKD_vs_disease_control", "dis", "Summary of signature, disease controls (ERCB, GSE211785)"),
                          ("DKD_vs_healthy_control", "hea", "Summary of signature, healthy controls")):
        m = Mm.loc[key]
        rows.insert(max(i for i, x in enumerate(rows) if x[1] == grp) + 1,
                    (lab, grp, "summary", np.nan, np.nan, m.g_random, m.ci_low, m.ci_high, m.p))
    d = pd.DataFrame(rows, columns=["label", "contrast", "kind", "n_case", "n_ctrl", "g", "ci_low", "ci_high", "p"])
    S.src_out(d, "sig_forest")
    labels = [l if not np.isfinite(a) else f"{l} ({int(a)} vs {int(b)})" for l, a, b in zip(d.label, d.n_case, d.n_ctrl)]
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        c = "#C44E52" if r.contrast == "dis" else "#8C8C8C"
        if r.kind == "summary":
            ax.fill([r.ci_low, r.g, r.ci_high, r.g], [yi, yi + 0.32, yi, yi - 0.32],
                    color="#E6A3A5" if r.contrast == "dis" else "#BBBBBB", lw=0)
        else:
            hi = min(r.ci_high, 6.6)
            ax.plot([max(r.ci_low, -4.0), hi], [yi, yi], color=c, lw=1.0)
            ax.scatter([r.g], [yi], s=16, color=c, marker="s", lw=0, zorder=3)
            if r.ci_high > 6.6:
                ax.annotate("", (6.75, yi), (6.4, yi), arrowprops=dict(arrowstyle="-|>", color=c, lw=0.8, mutation_scale=5))
    ax.axvline(0, color=S.GREY, lw=0.6, ls="--")
    nd = int((d.contrast == "dis").sum())
    ax.axhline(y[nd - 1] - 0.5, color="#DDDDDD", lw=0.6)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=5.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for yi, r in zip(y, d.itertuples()):
        ax.text(7.0, yi, f"{r.g:.2f}  {S.fmt_p(r.p)}", va="center", ha="left", fontsize=5.4)
    ax.set_xlim(-4.0, 10.6); ax.set_xticks([-4, -2, 0, 2, 4, 6])
    ax.set_xlabel("Hedges' g (95% CI)")
