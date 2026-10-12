"""Graphical abstract, drawn entirely in code: vector shapes and editable text.

Left to right: KPMP proximal tubule cells (single-cell and single-nucleus) -> mixture-of-experts model with routing
(experts weighted towards HKD or DKD) -> held-out participant readout (HKD lower than diabetes without CKD and DKD)
-> three read-outs (transferable gene signature tested in external cohorts; H&E localization; chromatin NF-kB
motifs) -> albuminuria tracked by the DKD-weighted expert contribution.
Canvas in millimetres (130 x 50); output PDF, SVG and 600-dpi PNG."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle, Ellipse
import _style as S

W, H = 130.0, 50.0
RED, BLUE, GREY, INK = "#C44E52", "#4C72B0", "#8C8C8C", S.INK
PAIRS, BOXES = [], []


def box(ax, x0, y0, x1, y1, fc, ec, lw=0.7, r=1.6, z=2, name=None):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z)
    ax.add_patch(p)
    if name:
        BOXES.append((name, p))
    return p


def txt(ax, x, y, s, patch=None, fs=6.0, fw="normal", col=INK, ha="center", va="center", style="normal"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col, ha=ha, va=va, zorder=8, linespacing=1.1,
                fontstyle=style)
    if patch is not None:
        PAIRS.append((t, patch))
    return t


def arrow(ax, x0, y0, x1, y1, col=INK, lw=1.0, ms=7):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=ms, color=col, lw=lw, zorder=6,
                                 shrinkA=0, shrinkB=0))


def stage_cells(ax):
    p = box(ax, 1, 6, 22, 44, "#EEF3FA", BLUE, name="cells")
    rng = np.random.default_rng(3)
    for k in range(9):
        x, y = 5 + (k % 3) * 6 + rng.uniform(-0.8, 0.8), 30 + (k // 3) * 3.6 + rng.uniform(-0.5, 0.5)
        if k % 2 == 0:
            ax.add_patch(Circle((x, y), 1.6, fc="#F6D9D7", ec="#B9605F", lw=0.5, zorder=4))
        ax.add_patch(Circle((x, y), 0.8, fc="#7B78B8", ec="none", zorder=5))
    txt(ax, 11.5, 21, "Proximal\ntubule cells", p, fs=6.3, fw="bold")
    txt(ax, 11.5, 12.5, "KPMP single-cell\nand single-nucleus", p, fs=5.3)


def stage_model(ax):
    p = box(ax, 26, 6, 51, 44, "#EEF2F6", "#4A6278", name="model")
    ax.add_patch(Circle((31.5, 32), 2.2, fc="white", ec=INK, lw=0.6, zorder=4))   # gate
    txt(ax, 31.5, 27.6, "gate", None, fs=5.0, col="#555555")
    cols = [(BLUE, "#DCE6F4"), (GREY, "#EEEEEE"), (RED, "#F6D9D7")]
    for k, (ec, fc) in enumerate(cols):
        y = 38.5 - k * 6.5
        ax.add_patch(FancyBboxPatch((38, y - 2), 9.5, 4, boxstyle="round,pad=0,rounding_size=0.8", fc=fc, ec=ec,
                                    lw=0.7, zorder=4))
        arrow(ax, 33.7, 32, 37.9, y, col=INK, lw=0.5, ms=4)
    txt(ax, 42.75, 38.5, "HKD", None, fs=5.0, col=BLUE, fw="bold")
    txt(ax, 42.75, 25.5, "DKD", None, fs=5.0, col=RED, fw="bold")
    txt(ax, 38.5, 16.5, "DKD-MoE\nmixture of experts", p, fs=6.3, fw="bold")
    txt(ax, 38.5, 9.5, "routing per cell", p, fs=5.3)


def stage_readout(ax):
    p = box(ax, 55, 6, 76, 44, "#FBF0F0", RED, name="readout")
    rng = np.random.default_rng(5)
    for x, col, mu in [(59.5, BLUE, 30.0), (65.5, "#B59F3B", 34.5), (71.5, RED, 35.0)]:
        v = np.clip(rng.normal(mu, 2.4, 12), 25, 41)
        ax.scatter(x + rng.uniform(-1.1, 1.1, len(v)), v, s=4, color=col, lw=0, zorder=5)
        ax.add_patch(Rectangle((x - 1.8, np.percentile(v, 25)), 3.6, np.percentile(v, 75) - np.percentile(v, 25),
                               fc="none", ec=col, lw=0.6, zorder=4))
    txt(ax, 59.5, 22.8, "HKD", None, fs=4.8, col=BLUE, fw="bold")
    txt(ax, 65.5, 22.8, "DM", None, fs=4.8, col="#8A7A2A", fw="bold")
    txt(ax, 71.5, 22.8, "DKD", None, fs=4.8, col=RED, fw="bold")
    txt(ax, 65.5, 16.0, "Participant\nreadout", p, fs=6.3, fw="bold")
    txt(ax, 65.5, 9.5, "held-out\nparticipants", p, fs=5.0)


def stage_three(ax):
    x0, x1 = 79.5, 110.5
    items = [(31.5, 44, "Transferable signature", "external cohorts"),
             (19.0, 31.5, "H&E localization", "eosin-poor,\nhaematoxylin-rich"),
             (6.0, 18.5, "Chromatin", "NF-κB motifs")]
    for k, (y0, y1, a, b) in enumerate(items):
        p = box(ax, x0, y0, x1, y1 - 0.8, "white", "#B59F3B", r=1.2, name=f"ro{k}")
        txt(ax, x0 + 19.5, y1 - 3.3, a, p, fs=5.4, fw="bold")
        txt(ax, x0 + 19.5, y0 + (3.6 if "\n" in b else 2.5), b, p, fs=5.0)
        gx, gy = x0 + 4.6, (y0 + y1 - 0.8) / 2
        if k == 0:     # up/down gene arrows
            arrow(ax, gx - 1.6, gy - 3.0, gx - 1.6, gy + 3.2, col=RED, lw=1.2, ms=5)
            arrow(ax, gx + 1.6, gy + 3.0, gx + 1.6, gy - 3.2, col=BLUE, lw=1.2, ms=5)
        elif k == 1:   # H&E patch with nuclei
            ax.add_patch(Rectangle((gx - 3.2, gy - 3.6), 6.4, 7.2, fc="#F3DDEB", ec="#B06A9F", lw=0.5, zorder=4))
            rng = np.random.default_rng(8)
            for i in range(12):
                ax.add_patch(Circle((gx - 2.2 + (i % 3) * 2.2 + rng.uniform(-0.4, 0.4), gy - 2.7 + (i // 3) * 1.8
                                     + rng.uniform(-0.3, 0.3)), 0.5, fc="#5B4A9B",
                                    ec="none", zorder=5))
        else:          # accessibility peaks
            hs = [1.5, 3.8, 6.0, 2.6, 1.2]
            for i, h in enumerate(hs):
                ax.add_patch(Rectangle((gx - 3.5 + i * 1.5, gy - 3.4), 1.0, h, fc="#8172B3", ec="none", zorder=5))
            ax.plot([gx - 3.8, gx + 3.8], [gy - 3.45, gy - 3.45], color=INK, lw=0.5, zorder=5)
    # signature detail inside the first box (right of the arrows)
    txt(ax, x0 + 19.5, 37.0, "complement, matrix ↑", None, fs=4.8, col="#444444")


def stage_albuminuria(ax):
    p = box(ax, 113.5, 6, 129, 44, "#F7F3E6", "#B59F3B", name="alb")
    xs = np.linspace(117, 126.5, 5); ys = 26 + np.array([0, 1.8, 3.2, 5.6, 8.2])
    ax.plot([116.5, 116.5], [25, 36.5], color=INK, lw=0.5, zorder=4); ax.plot([116.5, 127.5], [25, 25], color=INK, lw=0.5,
                                                                        zorder=4)
    ax.plot(xs, ys, color=RED, lw=1.1, marker="o", ms=2.2, zorder=5)
    txt(ax, 121.25, 17.0, "Albuminuria", p, fs=6.3, fw="bold")
    txt(ax, 121.25, 10.5, "tracks the\nDKD-weighted\nexpert", p, fs=5.0)


def main():
    S.setup()
    sc = 128.0 / W; y0, y1 = 5.0, 45.0
    fig = plt.figure(figsize=(W * sc / 25.4, (y1 - y0) * sc / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(y0, y1); ax.set_aspect("equal"); ax.axis("off")
    stage_cells(ax); stage_model(ax); stage_readout(ax); stage_three(ax); stage_albuminuria(ax)
    for a, b in [(22.3, 25.7), (51.3, 54.7), (76.3, 79.2), (110.8, 113.2)]:
        arrow(ax, a, 25, b, 25, lw=1.2, ms=8)
    issues = S.check_inside(fig, PAIRS) + S.check_boxes_disjoint(fig, BOXES)
    fig.savefig(S.FIG / "GraphicalAbstract.svg")
    S.save(fig, "GraphicalAbstract", issues)


if __name__ == "__main__":
    main()
