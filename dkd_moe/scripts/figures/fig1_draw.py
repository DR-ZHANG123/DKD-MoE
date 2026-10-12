"""Drawing helpers for Fig. 1: boxes, text, arrows, illustrated icons and network blocks (millimetre coordinates).

Icons are the transparent PNGs in figures/source_data/Fig1_icons/ (fig1_icons_generate.py); they are cropped to their
visible extent and placed by height, keeping their aspect ratio. Every text element and box is registered so that
Fig1_framework.py can check text-in-box containment and box overlap."""
from __future__ import annotations
import numpy as np
from PIL import Image
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
import _style as S

ICON_DIR = S.SRC / "Fig1_icons"
INK = S.INK
PAIRS, BOXES, ICONS = [], [], []
_cache: dict[str, np.ndarray] = {}


def box(ax, x0, y0, x1, y1, fc, ec, lw=0.6, r=1.5, z=2, name=None, track=True, alpha=1.0):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z, alpha=alpha)
    ax.add_patch(p)
    if track and name:
        BOXES.append((name, p))
    return p


def txt(ax, x, y, s, patch=None, fs=6.0, fw="normal", col=None, ha="center", va="center", z=7, style="normal"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col or INK, ha=ha, va=va, zorder=z, linespacing=1.15,
                fontstyle=style)
    if patch is not None:
        PAIRS.append((t, patch))
    return t


def arrow(ax, x0, y0, x1, y1, col=None, lw=0.9, ls="-", style="-|>", rad=0.0, ms=6):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=col or "#4A4A4A", lw=lw,
                        linestyle=ls, zorder=6, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def _load(name: str) -> np.ndarray:
    if name not in _cache:
        im = Image.open(ICON_DIR / f"{name}.png").convert("RGBA")
        a = np.asarray(im)[:, :, 3]; ys, xs = np.where(a > 8)
        _cache[name] = np.asarray(im.crop((xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)))
    return _cache[name]


def icon(ax, name, cx, cy, h=None, w=None, z=5):
    """Place an icon centred at (cx, cy) with height h (or width w) in millimetres; returns its extent."""
    im = _load(name); asp = im.shape[1] / im.shape[0]
    if h is None:
        h = w / asp
    w = h * asp
    ext = (cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2)
    ax.imshow(im, extent=ext, zorder=z, interpolation="lanczos")
    ICONS.append((name, ext))
    return ext


def trapezoid(ax, x0, x1, yc, h0, h1, fc, ec, label=None, fs=5.6, z=4, name=None):
    """Encoder (h0 > h1) or decoder (h0 < h1) block drawn as a trapezoid, as in machine-learning diagrams."""
    p = Polygon([(x0, yc - h0 / 2), (x1, yc - h1 / 2), (x1, yc + h1 / 2), (x0, yc + h0 / 2)], closed=True, fc=fc,
                ec=ec, lw=0.7, zorder=z, joinstyle="round")
    ax.add_patch(p)
    if name:
        BOXES.append((name, p))
    if label:
        txt(ax, (x0 + x1) / 2, yc, label, p, fs=fs, col="white", fw="bold")
    return p


def panel_frame(ax, x0, y0, x1, y1, fc, title, name):
    p = box(ax, x0, y0, x1, y1, fc, "none", lw=0, r=2.0, z=1, name=name)
    txt(ax, x0 + 3.0, y1 - 3.6, title, None, fs=7, fw="bold", ha="left")
    return p
