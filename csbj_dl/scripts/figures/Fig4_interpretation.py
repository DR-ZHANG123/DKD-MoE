"""Fig. 4 — genes of the experts, the transferable signature, exchanges of expert weight in held-out participants and
tests in independent cohorts.

(a) integrated-gradient consensus of the top genes of the three reproducible experts
    (results/expr_meta/{meta_top_genes,meta_ig_consensus,meta_experts}.tsv);
(b) cross-fitted signature score of held-out KPMP participants, DKD versus HKD by technology
    (results/expr_transfer/oof_scores.tsv, internal_signature_tests.tsv rows 'cross-fitted out-of-fold');
(c) the 15 highest-ranked up- and down-regulated genes of the signature (results/expr_transfer/transfer_signature.tsv),
    shaded by the number of training parts that selected each gene (fold_signatures.tsv); genes whose injured-PT
    expression is below half of that of fibroblast / stromal, immune or endothelial cells marked
    (results/expr_pt_signature/gene_sources.tsv);
(d) held-out HKD participants: signature change when the weight of the HKD-weighted expert is moved to another expert,
    and the share of alternative perturbations of the same model with an equal or larger change
    (results/expr_heldout_experts/{exchange_main_participants,exchange_ranks}.tsv);
(e) independent cohorts: readout and signature, disease controls first (results/expr_readout/donor_level_tests.tsv,
    results/expr_transfer/{external_signature_tests,external_meta_analysis}.tsv, results/expr_pt_signature/external_tests.tsv)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle
import _style_dl as S
from Fig3_readout import forest

COMPLEMENT = {"C3", "C1S", "C1R", "CFH"}
MATRIX = {"VCAN", "LAMA4", "NPNT", "ADAM12", "TGM2", "TNFAIP6", "COL23A1", "TIMP3"}


def panel_a(ax, n_top=6):
    me = S.read("expr_meta/meta_experts.tsv"); rep = me[me.reproducible].meta_expert.tolist()
    top = S.read("expr_meta/meta_top_genes.tsv")
    ig = S.read("expr_meta/meta_ig_consensus.tsv").set_index("gene")
    genes, owner = [], []
    for m in rep:
        for g in top[(top.meta_expert == m) & (top.direction == "up")].sort_values("rank").gene.head(n_top):
            if g not in genes:
                genes.append(g); owner.append(m)
    M = ig.loc[genes, rep]
    out = M.reset_index().melt(id_vars="gene", var_name="meta_expert", value_name="ig_consensus")
    out["top_gene_of"] = out.gene.map(dict(zip(genes, owner)))
    S.src_out(out, "Fig4_a_meta_genes")
    v = np.nanpercentile(np.abs(M.values), 98)
    im = ax.imshow(M.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=5.6, fontstyle="italic")
    hw = me.set_index("meta_expert").head_weight
    ax.set_xticks(range(len(rep)))
    rh = me.set_index("meta_expert")
    def role(m):    # the diagnosis-weighted experts carry the largest classifier weights; the third is the healthy-cell expert
        if abs(hw[m]) < hw.abs().max() / 2:
            return "mainly\nhealthy cells"
        return f"{'DKD' if hw[m] > 0 else 'HKD'}-\nweighted"
    ax.set_xticklabels([f"{m}\n({role(m)})" for m in rep], fontsize=5.8)
    ax.tick_params(length=0)
    for k in range(1, len(rep)):
        idx = owner.index(rep[k]) if rep[k] in owner else None
        if idx:
            ax.axhline(idx - 0.5, color="white", lw=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.06, pad=0.04)
    cb.set_label("Attribution\n(consensus)", fontsize=6); cb.ax.tick_params(labelsize=5.5); cb.outline.set_linewidth(0.4)


def panel_b(ax):
    O = S.read("expr_transfer/oof_scores.tsv", index_col=0)
    O = O[["readout_assoc"]].rename(columns={"readout_assoc": "score"})
    O["participant"] = [i.split("|")[0] for i in O.index]; O["tech"] = [i.split("|")[1] for i in O.index]
    grp = S.read("../config/splits.tsv", dtype={"participant": str}).set_index("participant").group
    O["group"] = O.participant.map(grp)
    d = O[O.group.isin(["HKD", "DKD"])].reset_index(drop=True)
    S.src_out(d[["participant", "tech", "group", "score"]], "Fig4_b_crossfitted_scores")
    T = S.read("expr_transfer/internal_signature_tests.tsv")
    T = T[T.setting.str.startswith("cross-fitted") & (T.construction == "readout_assoc")].set_index("tech")
    S.src_out(T.reset_index()[["tech", "auc_dkd_hkd", "g", "ci_low", "ci_high", "p_wilcoxon", "n1", "n2"]],
              "Fig4_b_tests")
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


def panel_c(ax, n_show=12):
    sig = S.read("expr_transfer/transfer_signature.tsv")
    fs = S.read("expr_transfer/fold_signatures.tsv"); fs = fs[fs.construction == "readout_assoc"]
    nf = fs.groupby(["gene", "direction"]).fold.nunique(); nfold = fs.fold.nunique()
    src = S.read("expr_pt_signature/gene_sources.tsv").groupby("gene").pt_ratio.min()
    sig["rank"] = sig.groupby("direction").cumcount() + 1
    sig["n_folds"] = [int(nf.get((g, d), 0)) for g, d in zip(sig.gene, sig.direction)]
    sig["min_pt_ratio"] = sig.gene.map(src)
    sig["annotation"] = ["complement" if g in COMPLEMENT else "extracellular matrix" if g in MATRIX else ""
                         for g in sig.gene]
    S.src_out(sig, "Fig4_c_transfer_signature")
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
    ax.text(5.6, ly, "* expressed mainly outside the proximal tubule", va="center", fontsize=5.8)
    ax.text(ncol - 0.5, ly, f"Shading: training parts (of {nfold})\nselecting the gene", va="center", ha="right",
            fontsize=5.8)
    ax.set_ylim(ly + 0.45, -1.1)


def panel_d(ax1, ax2):
    P = S.read("expr_heldout_experts/exchange_main_participants.tsv", dtype={"participant": str})
    R = S.read("expr_heldout_experts/exchange_ranks.tsv")
    cols = ["HKD-weighted to DKD-weighted", "HKD-weighted to healthy-cell", "HKD-weighted to non-reproducible"]
    S.src_out(P, "Fig4_d_exchange_participants"); S.src_out(R, "Fig4_d_exchange_ranks")
    data = [P[c].values * 100 for c in cols]
    S.strip_box(ax1, data, range(3), ["#C44E52", "#8C8C8C", "#CCB974"], s=4)
    ax1.axhline(0, color=S.GREY, lw=0.6, ls="--")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["to DKD-\nweighted", "to healthy-\ncell", "to non-\nreproducible"],
                                                  fontsize=5.6)
    ax1.set_ylabel("Signature change\n(×10$^{-2}$ log units)")
    ax1.set_title(f"Held-out HKD participants (n = {len(P)})", fontsize=6.3, pad=2)
    lab = {"mixture_of_experts": "Experts:\nexchange to DKD-\nweighted expert", "no_experts": "No experts:\nshift along\nclassifier"}
    data = [R[R.model == m].share_at_least_as_large.values for m in lab]
    S.strip_box(ax2, data, range(2), ["#C44E52", "#8172B3"], s=5)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(list(lab.values()), fontsize=5.4)
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("Share of alternatives with\nequal or larger change")
    ax2.set_title(f"Partition models (n = {len(data[0])})", fontsize=6.3, pad=2)


def panel_e(ax):
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
    S.src_out(d, "Fig4_e_external_forest")
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


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.25, 8.4))
    g1 = fig.add_gridspec(1, 2, width_ratios=[0.8, 1], wspace=0.55, left=0.1, right=0.98, top=0.97, bottom=0.74)
    g2 = fig.add_gridspec(1, 1, left=0.04, right=0.98, top=0.665, bottom=0.555)
    g3 = fig.add_gridspec(1, 2, width_ratios=[1.1, 0.8], wspace=0.55, left=0.12, right=0.75, top=0.48, bottom=0.36)
    g4 = fig.add_gridspec(1, 1, left=0.47, right=0.8, top=0.29, bottom=0.035)
    axa, axb = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])
    axc = fig.add_subplot(g2[0, 0])
    axd1, axd2 = fig.add_subplot(g3[0, 0]), fig.add_subplot(g3[0, 1])
    axe = fig.add_subplot(g4[0, 0])
    panel_a(axa); panel_b(axb); panel_c(axc); panel_d(axd1, axd2); panel_e(axe)
    S.auto_letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), ([axd1, axd2], "d"), (axe, "e")])
    S.save(fig, "Fig4_interpretation")


if __name__ == "__main__":
    main()
