"""Shared style, layout checks and save helpers for the dkd_moe figures (plotting only; never recomputes analyses).

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

CDL = Path(__file__).resolve().parents[2]          # dkd_moe
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
        "font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 7, "axes.titlesize": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5,
        "legend.fontsize": 6.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
        "xtick.major.width": 0.6, "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
        "savefig.bbox": "tight", "savefig.pad_inches": 0.04, "legend.frameon": True,
        "legend.fancybox": False, "legend.edgecolor": "#CCCCCC", "axes.labelweight": "bold",
        "axes.titleweight": "normal", "lines.linewidth": 0.8})


SRC_RENAME: dict[str, str] = {}      # panel modules shared by several figures: old prefix -> figure-specific prefix


def src_out(df: pd.DataFrame, name: str):
    """Save the exact plotting input of a panel as figures/source_data/<name>.tsv (name rewritten by SRC_RENAME)."""
    for a, b in SRC_RENAME.items():
        if name.startswith(a):
            name = b + name[len(a):]
            break
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


def auto_letters(fig, items: list[tuple], pad_pt: float = 4.0, col_tol: float = 0.06, row_tol: float = 0.05):
    """Panel letters on shared rows and columns, computed from the drawn extent of each panel (rows and columns
    grouped from the drawn extents). items: [(ax or [axes], letter)]. Rows: panels whose drawn tops lie
    within row_tol of the figure height share one baseline, just above the highest drawn top of the row (a panel
    spanning several rows therefore does not pull the panels below it up). Columns: letters whose drawn left edges
    lie within col_tol of the figure width share the leftmost edge."""
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted(); rec = []
    for ax, letter in items:
        axs = ax if isinstance(ax, (list, tuple)) else [ax]
        tb = [inv.transform_bbox(a.get_tightbbox(r, for_layout_only=True)) for a in axs]
        rec.append({"letter": letter, "x0": min(b.x0 for b in tb), "top": max(b.y1 for b in tb)})
    rows = []
    for it in sorted(rec, key=lambda d: -d["top"]):
        if rows and rows[-1][0]["top"] - it["top"] < row_tol:
            rows[-1].append(it)
        else:
            rows.append([it])
    pad = pad_pt / 72 / fig.get_figheight()
    for row in rows:
        y = max(o["top"] for o in row) + pad
        for o in row:
            o["y"] = y
    groups = []
    for o in sorted(rec, key=lambda d: d["x0"]):
        if groups and o["x0"] - groups[-1][0]["x0"] < col_tol:
            groups[-1].append(o)
        else:
            groups.append([o])
    for g in groups:
        x = max(0.0, min(o["x0"] for o in g) - 0.004)
        for o in g:
            o["x"] = x
    return [fig.text(o["x"], o["y"], o["letter"], fontsize=10, fontweight="bold", ha="left", va="bottom") for o in rec]


def data_overlap_report(fig, pad_px: float = 1.0) -> list[str]:
    """Text boxes that cover plotted data in the same axes (scatter points, line vertices and segments, error
    bars, filled bars and boxes) and legends that cover data. Tick labels, axis labels and texts marked
    _overlap_exempt (heatmap values, labels drawn on their own tiles) are exempt; schematic axes (axis off) are
    checked by check_inside / check_boxes_disjoint instead."""
    from matplotlib.collections import PathCollection, LineCollection
    from matplotlib.patches import Rectangle, PathPatch
    fig.canvas.draw(); r = fig.canvas.get_renderer(); issues = []

    def samples(ax):
        pts = []
        for c in ax.collections:
            if not c.get_visible():
                continue
            if isinstance(c, PathCollection) and c.get_offsets().size:
                pts.append(c.get_offset_transform().transform(np.asarray(c.get_offsets())))
            elif isinstance(c, LineCollection):
                for seg in c.get_segments():
                    seg = c.get_transform().transform(np.asarray(seg))
                    for a, b in zip(seg[:-1], seg[1:]):
                        pts.append(np.linspace(a, b, 25))
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_linewidth() == 0 and ln.get_marker() in (None, "None", ""):
                continue
            xy = ln.get_transform().transform(np.asarray(ln.get_xydata(), float))
            if ln.get_linestyle() in ("None", "", " "):
                pts.append(xy)
            else:
                for a, b in zip(xy[:-1], xy[1:]):
                    if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                        pts.append(np.linspace(a, b, 40))
        return np.vstack(pts) if pts else np.zeros((0, 2))

    def solid(ax):
        out = []
        for p in ax.patches:
            if isinstance(p, (Rectangle, PathPatch)) and p.get_visible() and p.get_fill() and p.get_alpha() != 0:
                b = p.get_window_extent(r)
                if b.width > 1 and b.height > 1:
                    out.append(b)
        return out
    for ax in fig.axes:
        if not ax.get_visible() or not ax.axison:
            continue
        pts = samples(ax); bars = solid(ax)
        texts = [t for t in ax.texts if t.get_visible() and t.get_text().strip() and not getattr(t, "_overlap_exempt", False)]
        for t in texts:
            b = t.get_window_extent(r)
            if pts.size:
                inside = ((pts[:, 0] > b.x0 + pad_px) & (pts[:, 0] < b.x1 - pad_px) &
                          (pts[:, 1] > b.y0 + pad_px) & (pts[:, 1] < b.y1 - pad_px))
                if inside.any():
                    issues.append(f"text '{t.get_text()[:30]}' covers data")
            for bb in bars:
                if min(b.x1, bb.x1) - max(b.x0, bb.x0) > pad_px and min(b.y1, bb.y1) - max(b.y0, bb.y0) > pad_px:
                    issues.append(f"text '{t.get_text()[:30]}' covers a bar/box"); break
        lg = ax.get_legend()
        if lg is not None and pts.size:
            b = lg.get_window_extent(r)
            inside = ((pts[:, 0] > b.x0) & (pts[:, 0] < b.x1) & (pts[:, 1] > b.y0) & (pts[:, 1] < b.y1))
            if inside.any():
                issues.append(f"legend covers data in axes '{ax.get_label()}'")
    return sorted(set(issues))


def exempt(*texts):
    for t in texts:
        t._overlap_exempt = True
    return texts


def save(fig, name: str, extra_issues: list[str] | None = None):
    issues = check_overlaps(fig) + data_overlap_report(fig) + list(extra_issues or [])
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


def fmt_sci(v: float, lab: str = "q") -> str:
    """Value as a power of ten (journal style, 3 × 10$^{-5}$) below 0.001, else two significant digits."""
    if v < 1e-3:
        e = int(np.floor(np.log10(v))); m = v / 10 ** e
        return f"{lab} = {m:.0f}×10$^{{{e}}}$"
    return f"{lab} = {v:.1g}" if v < 0.01 else f"{lab} = {v:.2f}"


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
