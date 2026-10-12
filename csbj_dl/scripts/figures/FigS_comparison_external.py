"""Supplementary figure — comparison methods in external cohorts, technology mixing, and contributions of the
reproducible experts in external donors. Panels from panels_comparison.py and panels_selection.py."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style_dl as S
import panels_comparison as PC
import panels_selection as PS

S.SRC_RENAME.update({"cmp_ab": "FigS_comparison_a", "cmp_c": "FigS_comparison_b", "sel_d": "FigS_comparison_c"})


def main():
    S.setup()
    Sm = S.read("bench_summary/summary_table.tsv"); P = S.read("bench_summary/paired_vs_moe_final.tsv")
    fig = plt.figure(figsize=(S.MAX_W - 0.3, 6.6))
    g1 = fig.add_gridspec(1, 2, wspace=0.12, left=0.3, right=0.985, top=0.95, bottom=0.62)
    ax1, ax2 = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])
    d = PC.table(Sm, P)
    PC.forest(ax1, d, "g211", "g211_lo", "g211_hi", 0.0, "Hedges' g (95% CI)", "GSE211785, DKD vs HKD", labels=True)
    PC.forest(ax2, d, "g195", "g195_lo", "g195_hi", 0.0, "Hedges' g (95% CI)", "GSE195460, DKD vs control")
    g2 = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.5, left=0.24, right=0.985, top=0.47, bottom=0.08)
    axb = fig.add_subplot(g2[0, 0])
    sub = g2[0, 1].subgridspec(3, 1, hspace=0.12)
    axc = [fig.add_subplot(sub[k]) for k in range(3)]
    PC.panel_c(axb); PS.panel_d(axc)
    S.auto_letters(fig, [([ax1, ax2], "a"), (axb, "b"), (axc, "c")])
    S.save(fig, "FigS_comparison_external")


if __name__ == "__main__":
    main()
