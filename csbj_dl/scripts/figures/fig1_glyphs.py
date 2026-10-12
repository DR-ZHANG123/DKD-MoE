"""Vector glyphs for Fig. 1 (kidney, glucose drop, blood-pressure gauge, cells, tissue patch, chromatin peaks,
neural-network layers, softmax bars, Gaussian, count distribution, similarity matrix). Coordinates in millimetres;
every glyph is drawn from matplotlib patches and lines only (no raster artwork)."""
from __future__ import annotations
import numpy as np
from matplotlib.patches import Circle, Ellipse, FancyBboxPatch, Polygon, Rectangle, Wedge, PathPatch
from matplotlib.path import Path

INK = "#33373D"


def kidney(ax, cx, cy, s, fc, ec="none", z=4):
    """Bean-shaped kidney of height ~2s centred at (cx, cy), hilum facing right."""
    t = np.linspace(0, 2 * np.pi, 120)
    r = 1 - 0.32 * np.exp(-((t - 0.0) / 0.55) ** 2) - 0.32 * np.exp(-((t - 2 * np.pi) / 0.55) ** 2)
    x, y = cx + 0.72 * s * r * np.cos(t), cy + s * r * np.sin(t)
    ax.add_patch(Polygon(np.c_[x, y], closed=True, fc=fc, ec=ec, lw=0.6, zorder=z))
    # pelvis / ureter
    ax.plot([cx + 0.28 * s, cx + 0.62 * s, cx + 0.7 * s], [cy, cy - 0.15 * s, cy - 0.85 * s], color="white", lw=0.9,
            zorder=z + 1, solid_capstyle="round")


def drop(ax, cx, cy, s, fc, z=6):
    """Glucose drop (diabetes): round base of radius s centred at (cx, cy) with a point above."""
    a = np.linspace(np.radians(-35), np.radians(215), 50)
    arc = np.c_[cx + s * np.cos(a), cy + s * np.sin(a)][::-1]
    pts = np.vstack([[cx, cy + 2.1 * s], arc[::-1]])
    ax.add_patch(Polygon(pts, closed=True, fc=fc, ec="none", zorder=z))


def gauge(ax, cx, cy, s, col, z=6):
    """Blood-pressure gauge (hypertension)."""
    ax.add_patch(Wedge((cx, cy), s, 0, 180, fc="white", ec=col, lw=0.6, zorder=z))
    ax.add_patch(Wedge((cx, cy), s, 0, 45, fc=col, ec="none", alpha=0.85, zorder=z))
    ax.plot([cx, cx + 0.8 * s * np.cos(np.radians(30))], [cy, cy + 0.8 * s * np.sin(np.radians(30))], color=INK, lw=0.6,
            zorder=z + 1)


def cells(ax, cx, cy, w, h, nucleus_only, seed, z=4):
    rng = np.random.default_rng(seed)
    for _ in range(7):
        x, y = cx + rng.uniform(-w / 2, w / 2), cy + rng.uniform(-h / 2, h / 2)
        if not nucleus_only:
            ax.add_patch(Circle((x, y), 1.5, fc="#F6D9D7", ec="#B9605F", lw=0.4, zorder=z))
        ax.add_patch(Circle((x, y), 0.75, fc="#7B78B8", ec="none", zorder=z + 1))


def tissue(ax, cx, cy, w, h, z=4, spots=True):
    ax.add_patch(Rectangle((cx - w / 2, cy - h / 2), w, h, fc="#F3DDEB", ec="#B06A9F", lw=0.5, zorder=z))
    rng = np.random.default_rng(4)
    for _ in range(int(w * h / 9)):
        ax.add_patch(Circle((cx + rng.uniform(-w / 2 + 0.6, w / 2 - 0.6), cy + rng.uniform(-h / 2 + 0.6, h / 2 - 0.6)),
                            0.32, fc="#6B4F9E", ec="none", alpha=0.8, zorder=z + 1))
    if spots:
        for i in range(4):
            for j in range(3):
                ax.add_patch(Circle((cx - w / 2 + (i + 0.6) * w / 4.2, cy - h / 2 + (j + 0.6) * h / 3.2), 0.55,
                                    fc="none", ec="#2E7D32", lw=0.35, zorder=z + 2))


def peaks(ax, x0, y0, w, h, col="#55A868", seed=3, z=4):
    rng = np.random.default_rng(seed)
    n = 7; bw = w / (n * 1.5)
    for i in range(n):
        ax.add_patch(Rectangle((x0 + i * 1.5 * bw, y0), bw, h * rng.uniform(0.2, 1.0), fc=col, ec="none", zorder=z))
    ax.plot([x0 - 0.3, x0 + w], [y0, y0], color=INK, lw=0.5, zorder=z)


def layers(ax, x0, y0, x1, y1, sizes, col, z=4, r=0.75, links=True):
    """Fully connected layers drawn as columns of nodes with faint links; returns node centres per layer."""
    xs = np.linspace(x0, x1, len(sizes)); pos = []
    for x, n in zip(xs, sizes):
        ys = np.linspace(y0, y1, n + 2)[1:-1] if n > 1 else np.array([(y0 + y1) / 2])
        pos.append([(x, y) for y in ys])
    if links:
        for a, b in zip(pos[:-1], pos[1:]):
            for p in a:
                for q in b:
                    ax.plot([p[0], q[0]], [p[1], q[1]], color=col, lw=0.18, alpha=0.45, zorder=z)
    for L in pos:
        for x, y in L:
            ax.add_patch(Circle((x, y), r, fc="white", ec=col, lw=0.6, zorder=z + 1))
    return pos


def expr_vector(ax, x0, y0, w, h, n, seed, z=4, cmap=None):
    import matplotlib.pyplot as plt
    cm = cmap or plt.cm.Greens
    v = np.random.default_rng(seed).uniform(0.1, 1, n); ch = h / n
    for i, a in enumerate(v):
        ax.add_patch(Rectangle((x0, y0 + i * ch), w, ch * 0.9, fc=cm(0.2 + 0.7 * a), ec="none", zorder=z))


def softmax_bars(ax, x0, y0, w, h, probs, cols, top, z=5):
    n = len(probs); bw = w / (n * 1.4)
    for i, (p, c) in enumerate(zip(probs, cols)):
        ax.add_patch(Rectangle((x0 + i * 1.4 * bw, y0), bw, h * p / max(probs), fc=c if i in top else "#D6D6D6",
                               ec=INK if i in top else "none", lw=0.4, zorder=z))
    ax.plot([x0 - 0.3, x0 + w], [y0, y0], color=INK, lw=0.4, zorder=z)


def gaussian(ax, cx, y0, w, h, col, z=5):
    x = np.linspace(-3, 3, 60)
    ax.plot(cx + x * w / 6, y0 + h * np.exp(-x ** 2 / 2), color=col, lw=0.9, zorder=z)
    ax.plot([cx - w / 2, cx + w / 2], [y0, y0], color=INK, lw=0.4, zorder=z)


def nb_bars(ax, x0, y0, w, h, col, z=5):
    k = np.arange(9); p = (k + 1) * np.exp(-k / 1.6); p = p / p.max(); bw = w / 12
    for i, v in enumerate(p):
        ax.add_patch(Rectangle((x0 + i * 1.3 * bw, y0), bw, h * v, fc=col, ec="none", zorder=z))
    ax.plot([x0 - 0.3, x0 + w], [y0, y0], color=INK, lw=0.4, zorder=z)


def sim_matrix(ax, x0, y0, s, n, col, z=5):
    c = s / n
    for i in range(n):
        for j in range(n):
            ax.add_patch(Rectangle((x0 + j * c, y0 + (n - 1 - i) * c), c * 0.92, c * 0.92,
                                   fc=col if i == j else "#E6E6E6", alpha=1.0 if i == j else 0.9, ec="none", zorder=z))


def stacked(ax, x0, y0, w, parts, cols, scale, z=5):
    """Horizontal stacked bar of signed contributions (negative parts drawn to the left of x0)."""
    pos, neg = x0, x0
    for v, c in zip(parts, cols):
        if v >= 0:
            ax.add_patch(Rectangle((pos, y0), v * scale, w, fc=c, ec="white", lw=0.3, zorder=z)); pos += v * scale
        else:
            neg += v * scale
            ax.add_patch(Rectangle((neg, y0), -v * scale, w, fc=c, ec="white", lw=0.3, zorder=z))
    return neg, pos
