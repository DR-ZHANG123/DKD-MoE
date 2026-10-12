"""Fig. 4 — what the experts represent: cells, genes and exchanges of expert weight in held-out participants.

(a) latent-space map of KPMP proximal tubule cells (layout: final model; colours: partition models that had not seen
    the participant): cell state and routing to the healthy-cell, HKD-weighted and DKD-weighted experts
    (results/expr_cell_atlas/cells.tsv);
(b) participant-balanced densities of injured cells, DKD and diabetes without CKD minus HKD
    (results/expr_cell_atlas/density.tsv);
(c) held-out routing of injured cells to the two disease-weighted experts, one point per participant
    (results/expr_cell_atlas/state_by_expert.tsv);
(d) integrated-gradient consensus of the top genes of the three reproducible experts
    (results/expr_meta/{meta_top_genes,meta_ig_consensus,meta_experts}.tsv);
(e) decoded change of gene groups when the weight of the HKD-weighted expert is moved to the DKD-weighted expert in
    held-out HKD participants (results/expr_heldout_experts/exchange_gene_groups.tsv);
(f) signature change of that exchange and of alternatives, and the share of alternatives with an equal or larger
    change (results/expr_heldout_experts/{exchange_main_participants,exchange_ranks}.tsv)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import _style as S
import panels_atlas as PA


def heatmap(ax, n_top=6):
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
    S.src_out(out, "Fig4_d_meta_genes")
    v = np.nanpercentile(np.abs(M.values), 98)
    im = ax.imshow(M.values, cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), aspect="auto")
    ax.set_yticks(range(len(genes))); ax.set_yticklabels(genes, fontsize=5.6, fontstyle="italic")
    hw = me.set_index("meta_expert").head_weight
    ax.set_xticks(range(len(rep)))
    rh = me.set_index("meta_expert")
    def role(m):    # the diagnosis-weighted experts carry the largest classifier weights; the third is the healthy-cell expert
        if abs(hw[m]) < hw.abs().max() / 2:
            return "healthy\ncells"
        return f"{'DKD' if hw[m] > 0 else 'HKD'}-\nweighted"
    ax.set_xticklabels([role(m) for m in rep], fontsize=5.6)
    ax.tick_params(length=0)
    for k in range(1, len(rep)):
        idx = owner.index(rep[k]) if rep[k] in owner else None
        if idx:
            ax.axhline(idx - 0.5, color="white", lw=1.2)
    for sp in ax.spines.values():
        sp.set_visible(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.08, pad=0.05)
    cb.set_label("Attribution", fontsize=6); cb.ax.tick_params(labelsize=5.5); cb.outline.set_linewidth(0.4)



def exchange(ax1, ax2):
    P = S.read("expr_heldout_experts/exchange_main_participants.tsv", dtype={"participant": str})
    R = S.read("expr_heldout_experts/exchange_ranks.tsv")
    cols = ["HKD-weighted to DKD-weighted", "HKD-weighted to healthy-cell", "HKD-weighted to non-reproducible"]
    S.src_out(P, "Fig4_f_exchange_participants"); S.src_out(R, "Fig4_f_exchange_ranks")
    data = [P[c].values * 100 for c in cols]
    S.strip_box(ax1, data, range(3), ["#C44E52", "#8C8C8C", "#CCB974"], s=4)
    ax1.axhline(0, color=S.GREY, lw=0.6, ls="--")
    ax1.set_xticks(range(3)); ax1.set_xticklabels(["DKD-\nwtd.", "healthy\ncell", "other"],
                                                  fontsize=5.4)
    ax1.set_ylabel("Signature change\n(×10$^{-2}$ log units)")
    ax1.set_title(f"Exchange target\n(held-out HKD, n = {len(P)})", fontsize=6.0, pad=2)
    lab = {"mixture_of_experts": "DKD-\nMoE", "no_experts": "no\nexperts"}
    data = [R[R.model == m].share_at_least_as_large.values for m in lab]
    S.strip_box(ax2, data, range(2), ["#C44E52", "#8172B3"], s=5)
    ax2.set_xticks(range(2)); ax2.set_xticklabels(list(lab.values()), fontsize=5.4)
    ax2.set_ylim(0, 1.0); ax2.set_ylabel("Share of alternatives with\nequal or larger change", fontsize=6)
    ax2.set_title(f"Specificity\n(models, n = {len(data[0])})", fontsize=6.0, pad=2)


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.12, 6.6))
    ga = fig.add_gridspec(1, 4, wspace=0.08, left=0.04, right=0.985, top=0.955, bottom=0.715)
    axa = [fig.add_subplot(ga[0, i]) for i in range(4)]
    gb = fig.add_gridspec(1, 2, wspace=0.08, left=0.04, right=0.5, top=0.63, bottom=0.40)
    axb = [fig.add_subplot(gb[0, i]) for i in range(2)]
    gc = fig.add_gridspec(1, 2, wspace=0.42, left=0.6, right=0.985, top=0.61, bottom=0.43)
    axc = [fig.add_subplot(gc[0, i]) for i in range(2)]
    axd = fig.add_subplot(fig.add_gridspec(1, 1, left=0.075, right=0.215, top=0.31, bottom=0.075)[0, 0])
    axe = fig.add_subplot(fig.add_gridspec(1, 1, left=0.47, right=0.63, top=0.31, bottom=0.095)[0, 0])
    gf = fig.add_gridspec(1, 2, wspace=0.75, left=0.715, right=0.985, top=0.31, bottom=0.095)
    axf = [fig.add_subplot(gf[0, i]) for i in range(2)]
    C = PA.cells()
    PA.state_map(axa[0], C, "Fig4_a_state")
    s2 = PA.expert_map(axa[1], C, "M2", "Fig4_a_routing_healthy_cell")
    s1 = PA.expert_map(axa[2], C, "M1", "Fig4_a_routing_HKD_weighted")
    s3 = PA.expert_map(axa[3], C, "M3", "Fig4_a_routing_DKD_weighted")
    im1, v = PA.density_diff(axb[0], C, "DKD", "HKD", "Fig4_b_density_DKD_minus_HKD")
    PA.density_diff(axb[1], C, "DM_noCKD", "HKD", "Fig4_b_density_DMnoCKD_minus_HKD", vmax=v)
    PA.routing_by_group(axc[0], "M1", "Fig4_c_routing_M1"); PA.routing_by_group(axc[1], "M3", "Fig4_c_routing_M3")
    axc[0].set_ylabel("Held-out routing of\ninjured cells (mean)")
    heatmap(axd); PA.exchange_groups(axe, "Fig4_e_gene_groups"); exchange(axf[0], axf[1])
    fig.canvas.draw()
    for ax_, sc_ in ((axa[1], s2), (axa[2], s1), (axa[3], s3)):
        PA.small_cbar(fig, ax_, sc_, "held-out routing probability", [0, 0.4, 0.8])
    cb = PA.small_cbar(fig, axb[0], im1, "density difference (% of injured cells per bin)")
    p0, p1 = axb[0].get_position(), axb[1].get_position()
    cb.set_position([p0.x0 + p0.width * 0.3, p0.y0 - 0.008, (p1.x1 - p0.x0) * 0.45, 0.007])
    S.auto_letters(fig, [(axa, "a"), (axb, "b"), (axc, "c"), (axd, "d"), (axe, "e"), (axf, "f")])
    S.save(fig, "Fig4_interpretation")


if __name__ == "__main__":
    main()
