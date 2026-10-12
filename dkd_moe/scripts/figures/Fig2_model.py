"""Fig. 2 — the expert model: selection, reproducibility, comparison with other approaches and forms of supervision.

(a) held-out criteria over the number of experts and condition weight; (b) agreement between random initializations;
(c) experts matched across five models and their classifier weights; (d) held-out KPMP DKD-versus-HKD AUC of the
readout and of the comparison readouts, with the paired AUC difference; (e) variants without one component and the
supervised model without experts; (f) supervision on cells, injured cells and participant bags.
Panels from panels_selection.py and panels_comparison.py (sources listed there)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style as S
import panels_selection as PS
import panels_comparison as PC

S.SRC_RENAME.update({"sel_a": "Fig2_a", "sel_b": "Fig2_b", "sel_c": "Fig2_c", "cmp_ab": "Fig2_d", "cmp_d": "Fig2_e",
                     "cmp_e": "Fig2_f"})


def main():
    S.setup()
    Sm = S.read("bench_summary/summary_table.tsv"); P = S.read("bench_summary/paired_vs_moe_final.tsv")
    fig = plt.figure(figsize=(S.MAX_W - 0.25, 8.3))
    g1 = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=0.42, left=0.13, right=0.985, top=0.93, bottom=0.735)
    axa, axb = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])
    g2 = fig.add_gridspec(2, 1, height_ratios=[1.35, 1], hspace=0.12, left=0.1, right=0.45, top=0.645, bottom=0.42)
    axc, axc2 = fig.add_subplot(g2[0]), fig.add_subplot(g2[1])
    g3 = fig.add_gridspec(1, 2, wspace=0.12, left=0.72, right=0.985, top=0.645, bottom=0.42)
    axd1, axd2 = fig.add_subplot(g3[0, 0]), fig.add_subplot(g3[0, 1])
    g4 = fig.add_gridspec(1, 1, left=0.17, right=0.44, top=0.29, bottom=0.06)
    axe = fig.add_subplot(g4[0, 0])
    g5 = fig.add_gridspec(1, 2, wspace=0.12, left=0.62, right=0.96, top=0.25, bottom=0.06)
    axf1, axf2 = fig.add_subplot(g5[0, 0]), fig.add_subplot(g5[0, 1])
    PS.panel_a(axa); PS.panel_b(axb); PS.panel_c(axc, axc2)
    d = PC.table(Sm, P)
    PC.forest(axd1, d, "kpmp_auc", "kpmp_auc_lo", "kpmp_auc_hi", 0.5, "AUC (95% CI)", "KPMP held-out", labels=True)
    PC.forest(axd2, d, "auc_diff", "diff_lo", "diff_hi", 0.0, "AUC difference\nvs final model", "Paired difference")
    axd1.set_xlim(0.35, 1.0); axd2.set_xlim(-0.4, 0.25)
    PC.panel_d(axe, P); PC.panel_e(axf1, axf2)
    fig.canvas.draw()
    pos, pos2 = axc.get_position(), axc2.get_position()
    axc2.set_position([pos2.x0, pos2.y0, pos.width, pos2.height])
    S.auto_letters(fig, [(axa, "a"), (axb, "b"), ([axc, axc2], "c"), ([axd1, axd2], "d"), (axe, "e"), ([axf1, axf2], "f")])
    S.save(fig, "Fig2_model")


if __name__ == "__main__":
    main()
