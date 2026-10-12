"""Fig. 6 — chromatin of the same nuclei: the readout, motifs and the DKD-weighted expert component.

(a) chromatin-side score against the RNA readout and by group; (b) motifs enriched in peaks contributing to the score;
(c) chromatin-side score of the DKD-weighted expert contribution by group; (d) motif accessibility against the
contributions of the DKD- and HKD-weighted experts. Panels from panels_chromatin.py (sources listed there)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style_dl as S
import panels_chromatin as PH

S.SRC_RENAME.update({"chr_b": "Fig6_a", "chr_d": "Fig6_b", "chr_f": "Fig6_c", "chr_g": "Fig6_d"})


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.3, 7.6))
    g1 = fig.add_gridspec(1, 2, width_ratios=[1.15, 0.85], wspace=0.42, left=0.09, right=0.985, top=0.955, bottom=0.71)
    axa1, axa2 = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])
    g2 = fig.add_gridspec(1, 2, wspace=0.75, left=0.1, right=0.55, top=0.56, bottom=0.05)
    axb1, axb2 = fig.add_subplot(g2[0, 0]), fig.add_subplot(g2[0, 1])
    axc = fig.add_axes([0.68, 0.37, 0.3, 0.2])
    axd = fig.add_axes([0.74, 0.045, 0.24, 0.2])
    PH.panel_b(axa1, axa2); PH.panel_d(axb1, axb2); PH.panel_f(axc); PH.panel_g(axd)
    S.letters(fig, [[(axa1, "a")], [(axb1, "b"), (axc, "c")]], col_align=False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = fig.transFigure.inverted().transform_bbox(axd.get_tightbbox(r, for_layout_only=True))
    bc = fig.transFigure.inverted().transform_bbox(axc.get_tightbbox(r, for_layout_only=True))
    S.letter_at(fig, bc.x0 - 0.012, bb.y1 + 0.006, "d")
    S.save(fig, "Fig6_chromatin")


if __name__ == "__main__":
    main()
