"""Supplementary figure — chromatin alignment: same-nucleus retrieval, external snATAC cohorts and nearest genes of the
peaks contributing to the chromatin-side score. Panels from panels_chromatin.py (sources listed there)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style_dl as S
import panels_chromatin as PH

S.SRC_RENAME.update({"chr_a": "FigS_chromatin_a", "chr_c": "FigS_chromatin_b", "chr_e": "FigS_chromatin_c"})


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.3, 4.6))
    g = fig.add_gridspec(2, 2, width_ratios=[1.1, 0.8], height_ratios=[1.2, 1], wspace=0.55, hspace=0.75,
                         left=0.09, right=0.95, top=0.93, bottom=0.1)
    axa = fig.add_subplot(g[0, 0]); axb = fig.add_subplot(g[1, 0]); axc = fig.add_subplot(g[:, 1])
    PH.panel_a(axa); PH.panel_c(axb); PH.panel_e(axc)
    S.letters(fig, [[(axa, "a"), (axc, "c")]], col_align=False)
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    bb = fig.transFigure.inverted().transform_bbox(axb.get_tightbbox(r, for_layout_only=True))
    S.letter_at(fig, 0.005, bb.y1 + 0.006, "b")
    S.save(fig, "FigS_chromatin_alignment")


if __name__ == "__main__":
    main()
