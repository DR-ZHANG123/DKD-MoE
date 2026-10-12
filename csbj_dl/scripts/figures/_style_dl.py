"""Shared style, layout checks and save helpers for the csbj_dl figures (plotting only; never recomputes analyses).

Style constants follow work/AIFFEL_Figs/CLAUDE.md and the main project's scripts/figures/_style.py
(sans-serif 8 pt, top/right spines off, bold lowercase panel letters, PDF with editable text + 600-dpi PNG).
Panel letters are placed with fig.text on one common baseline per row (not ax.set_title), so letters of a row
line up even when the axes have different heights or titles."""
from __future__ import annotations
from pathlib import Path
import itertools
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CDL = Path(__file__).resolve().parents[2]          # csbj_dl
RES = CDL / "results"
FIG = CDL / "figures"
SRC = FIG / "source_data"
FIG.mkdir(exist_ok=True); SRC.mkdir(exist_ok=True)

MAX_W = 17.5 / 2.54                                  # journal full width (inches)
MAX_H = 22.0 / 2.54

PALETTE = ['#4C72B0', '#DD8452', '#55A868', '#C44E52', '#8172B3', '#937860', '#DA8BC3', '#8C8C8C', '#CCB974', '#64B5CD']
GROUP_COL = {"Reference": "#8C8C8C", "Control": "#8C8C8C", "LD": "#8C8C8C", "DM_noCKD": "#CCB974",
             "HKD": "#4C72B0", "HTN": "#4C72B0", "HKD_withDM": "#8172B3", "other_CKD": "#4C72B0", "DKD": "#C44E52"}
GROUP_LABEL = {"Reference": "Reference", "Control": "Control", "DM_noCKD": "Diabetes,\nno CKD", "HKD": "HKD",
               "HKD_withDM": "HKD with\ndiabetes", "DKD": "DKD", "HTN": "HKD", "LD": "Living donor",
               "other_CKD": "Other CKD"}
TECH_LABEL = {"sc": "Single-cell", "sn": "Single-nucleus", "pooled": "Pooled"}
COHORT_LABEL = {"kpmp_heldout": "KPMP held-out", "kpmp": "KPMP held-out", "gse211785": "GSE211785",
                "gse339455": "GSE339455", "KPMP_heldout": "KPMP held-out"}
INK = "#33373D"
GREY = "#8C8C8C"


def setup():
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": True,
        "legend.fancybox": False, "legend.edgecolor": "#CCCCCC", "axes.labelweight": "bold",
        "axes.titleweight": "normal", "lines.linewidth": 0.8})


def src_out(df: pd.DataFrame, name: str):
    """Save the exact plotting input of a panel as figures/source_data/<name>.tsv."""
    df.to_csv(SRC / f"{name}.tsv", sep="\t", index=False)
    return df


def read(rel: str, **kw) -> pd.DataFrame:
    return pd.read_csv(RES / rel, sep="\t", **kw)


def letters(fig, rows: list[list[tuple]], dx: float = 0.012, dy: float = 0.006, col_align: bool = True):
    """rows: [[(ax, 'a'), (ax, 'b')], ...]. All letters of one row share the baseline just above the highest
    axes top (including its title) of that row; x = left edge of each axes' tight box."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    out = []; plan = []
    for row in rows:
        tops, lefts = [], []
        for ax, _ in row:
            axs = ax if isinstance(ax, (list, tuple)) else [ax]
            bbs = [inv.transform_bbox(a.get_tightbbox(r, for_layout_only=True)) for a in axs]
            tops.append(max(bb.y1 for bb in bbs)); lefts.append(min(bb.x0 for bb in bbs))
        plan.append((max(tops) + dy, [l for _, l in row], lefts))
    if col_align:   # letters in the same column position share x across rows
        ncol = max(len(p[2]) for p in plan)
        xs = [min(p[2][j] for p in plan if len(p[2]) > j) for j in range(ncol)]
        plan = [(y, ls, [xs[j] if len(p) == ncol else x for j, x in enumerate(p)]) for y, ls, p in plan]
    for y, ls, lefts in plan:
        for l, x in zip(ls, lefts):
            out.append(fig.text(x - dx, y, l, fontsize=10, fontweight="bold", ha="left", va="bottom"))
    return out


def letter_at(fig, x: float, y: float, l: str):
    return fig.text(x, y, l, fontsize=10, fontweight="bold", ha="left", va="bottom")


def _texts(fig):
    r = fig.canvas.get_renderer(); out = []
    for ax in fig.axes:
        if not ax.get_visible():
            continue
        items = [ax.title, ax._left_title, ax._right_title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
        if not ax.axison:   # axis('off'): tick labels are not drawn
            x0, x1, y0, y1 = np.inf, -np.inf, np.inf, -np.inf
        items += [t for t in ax.get_xticklabels() if t.get_visible() and x0 - 1e-9 <= t.get_position()[0] <= x1 + 1e-9]
        items += [t for t in ax.get_yticklabels() if t.get_visible() and y0 - 1e-9 <= t.get_position()[1] <= y1 + 1e-9]
        for t in items:
            if t.get_visible() and t.get_text().strip() and not getattr(t, "_skip_check", False):
                out.append((ax, t, t.get_window_extent(r)))
        leg = ax.get_legend()
        if leg is not None and leg.get_texts():
            out.append((ax, leg.get_texts()[0], leg.get_window_extent(r)))
    out += [(None, t, t.get_window_extent(r)) for t in fig.texts if t.get_text().strip()]
    for leg in fig.legends:
        out.append((None, leg.get_texts()[0], leg.get_window_extent(r)))
    return out


def check_overlaps(fig, tol: float = 0.5) -> list[str]:
    """Pairwise text-bbox overlaps, and text entering another axes' plotting area."""
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    T = _texts(fig); issues = []
    for (a1, t1, b1), (a2, t2, b2) in itertools.combinations(T, 2):
        if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > tol and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > tol:
            issues.append(f"text overlap: '{t1.get_text()[:28]}' x '{t2.get_text()[:28]}' [{a1.get_label() if a1 else 'fig'}|{a2.get_label() if a2 else 'fig'}]")
    boxes = [(ax, ax.get_window_extent(r)) for ax in fig.axes if ax.get_visible() and ax.axison]
    for ax_t, t, b in T:
        for ax, ab in boxes:
            if ax is ax_t or getattr(ax, "_host", None) is ax_t or getattr(ax_t, "_host", None) is ax:
                continue
            if min(b.x1, ab.x1) - max(b.x0, ab.x0) > tol and min(b.y1, ab.y1) - max(b.y0, ab.y0) > tol:
                issues.append(f"text '{t.get_text()[:28]}' enters another axes")
    return issues


def check_inside(fig, pairs: list[tuple], pad_px: float = 1.0) -> list[str]:
    """pairs: (text, patch) — every text must lie fully inside its box."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); issues = []
    for t, p in pairs:
        tb, pb = t.get_window_extent(r), p.get_window_extent(r)
        if tb.x0 < pb.x0 + pad_px or tb.x1 > pb.x1 - pad_px or tb.y0 < pb.y0 + pad_px or tb.y1 > pb.y1 - pad_px:
            issues.append(f"text '{t.get_text()[:30]}' leaves its box")
    return issues


def check_boxes_disjoint(fig, patches: list, tol_px: float = 0.5) -> list[str]:
    fig.canvas.draw(); r = fig.canvas.get_renderer(); issues = []
    for (n1, p1), (n2, p2) in itertools.combinations(patches, 2):
        b1, b2 = p1.get_window_extent(r), p2.get_window_extent(r)
        if min(b1.x1, b2.x1) - max(b1.x0, b2.x0) > tol_px and min(b1.y1, b2.y1) - max(b1.y0, b2.y0) > tol_px:
            issues.append(f"boxes overlap: {n1} x {n2}")
    return issues


def save(fig, name: str, extra_issues: list[str] | None = None):
    issues = check_overlaps(fig) + list(extra_issues or [])
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=600)
    from PIL import Image
    Image.MAX_IMAGE_PIXELS = None
    w, h = Image.open(FIG / f"{name}.png").size
    wcm, hcm = w / 600 * 2.54, h / 600 * 2.54
    if wcm > 17.5 + 0.05 or hcm > 22.0 + 0.05:
        issues.append(f"size {wcm:.1f} x {hcm:.1f} cm exceeds 17.5 x 22 cm")
    print(f"{name}: {wcm:.1f} x {hcm:.1f} cm; layout issues: {len(issues)}")
    for i in issues:
        print("   ", i)
    plt.close(fig)
    return issues


def fmt_p(p: float) -> str:
    if p is None or not np.isfinite(p):
        return "P n/a"
    if p < 1e-3:
        e = int(np.floor(np.log10(p))); m = p / 10 ** e
        return f"P = {m:.1f}×10$^{{{e}}}$"
    return f"P = {p:.2g}" if p < 0.1 else f"P = {p:.2f}"


def hedges_ci(g, na, nb, z=1.96):
    se = np.sqrt((na + nb) / (na * nb) + g ** 2 / (2 * (na + nb)))
    return g - z * se, g + z * se


def strip_box(ax, data: list[np.ndarray], pos, colors, width=0.55, s=6, rng_seed=0, alpha=0.75):
    """Box (no fliers) with jittered participant points on top."""
    rng = np.random.default_rng(rng_seed)
    bp = ax.boxplot(data, positions=pos, widths=width, showfliers=False, patch_artist=True,
                    medianprops=dict(color=INK, lw=0.9), whiskerprops=dict(lw=0.6, color=INK),
                    capprops=dict(lw=0.6, color=INK), boxprops=dict(lw=0.6, color=INK))
    for b, c in zip(bp["boxes"], colors):
        b.set_facecolor(matplotlib.colors.to_rgba(c, 0.18)); b.set_edgecolor(c)
    for x, d, c in zip(pos, data, colors):
        j = rng.uniform(-width * 0.32, width * 0.32, len(d))
        ax.scatter(np.full(len(d), x) + j, d, s=s, color=c, alpha=alpha, lw=0, zorder=3)
    return bp
