"""Cell-level panels of Fig. 4 (sources: results/expr_cell_atlas/{cells,density,state_by_expert}.tsv and
results/expr_heldout_experts/exchange_gene_groups.tsv). Layout from the latent means of a final model; colours from
partition models that had not seen the participant."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm, LinearSegmentedColormap
from scipy.ndimage import gaussian_filter
import _style as S

GROUPS = ["HKD", "HKD_withDM", "DM_noCKD", "DKD"]
GLAB = {"HKD": "HKD", "HKD_withDM": "HKD\n+ DM", "DM_noCKD": "DM,\nno CKD", "DKD": "DKD", "Reference": "Ref."}
EXPERT_CMAP = {"M1": LinearSegmentedColormap.from_list("m1", ["#EDEDED", "#4C72B0", "#1F3F73"]),
               "M2": LinearSegmentedColormap.from_list("m2", ["#EDEDED", "#937860", "#5A4636"]),
               "M3": LinearSegmentedColormap.from_list("m3", ["#EDEDED", "#C44E52", "#7A1F22"])}
EXPERT_TITLE = {"M1": "HKD-weighted expert", "M2": "Healthy-cell expert", "M3": "DKD-weighted expert"}


def cells() -> pd.DataFrame:
    C = S.read("expr_cell_atlas/cells.tsv", dtype={"participant": str})
    return C.sample(frac=1.0, random_state=0).reset_index(drop=True)      # random drawing order: no group on top


def _frame(ax, title):
    ax.set_xticks([]); ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.set_title(title, fontsize=6.3, pad=2)
    ax.set_aspect("equal", adjustable="datalim")


def _axis_hint(ax):
    x0, x1 = ax.get_xlim(); y0, y1 = ax.get_ylim(); dx, dy = (x1 - x0) * 0.13, (y1 - y0) * 0.13
    ax.annotate("", (x0 + dx, y0), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.5, color=S.INK, mutation_scale=4))
    ax.annotate("", (x0, y0 + dy), (x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.5, color=S.INK, mutation_scale=4))
    ax.text(x0, y0 - (y1 - y0) * 0.02, "UMAP 1", fontsize=4.8, va="top", ha="left")
    ax.text(x0 - (x1 - x0) * 0.02, y0, "UMAP 2", fontsize=4.8, va="bottom", ha="right", rotation=90)


def state_map(ax, C, src):
    col = {"healthy_PT": "#B9C7D6", "injured_PT": "#D9822B"}
    S.src_out(C[["cell", "umap1", "umap2", "pt_state", "group"]], src)
    for st in ("healthy_PT", "injured_PT"):
        d = C[C.pt_state == st]
        ax.scatter(d.umap1, d.umap2, s=0.15, c=col[st], lw=0, rasterized=True)
    _frame(ax, "Cell state"); _axis_hint(ax)
    for i, (st, lab) in enumerate((("healthy_PT", "healthy"), ("injured_PT", "injured"))):
        ax.scatter([], [], s=8, c=col[st], label=lab)
    ax.legend(loc="upper right", fontsize=5.2, frameon=False, handletextpad=0.1, borderaxespad=0.0, markerscale=1.2)


def expert_map(ax, C, m, src):
    S.src_out(C[["cell", "umap1", "umap2", f"p_{m}"]], src)
    o = np.argsort(C[f"p_{m}"].values)
    sc = ax.scatter(C.umap1.values[o], C.umap2.values[o], c=C[f"p_{m}"].values[o], s=0.15, lw=0, cmap=EXPERT_CMAP[m],
                    vmin=0, vmax=0.8, rasterized=True)
    _frame(ax, EXPERT_TITLE[m])
    return sc


def logodds_map(ax, C, src):
    S.src_out(C[["cell", "umap1", "umap2", "logodds"]], src)
    v = np.nanpercentile(np.abs(C.logodds), 97)
    o = np.argsort(np.abs(C.logodds.values))
    sc = ax.scatter(C.umap1.values[o], C.umap2.values[o], c=C.logodds.values[o], s=0.15, lw=0, cmap="RdBu_r",
                    norm=TwoSlopeNorm(0, -v, v), rasterized=True)
    _frame(ax, "Cell log-odds (DKD vs HKD)")
    return sc


def small_cbar(fig, ax, sc, label, ticks=None):
    p = ax.get_position()
    cax = fig.add_axes([p.x0 + p.width * 0.2, p.y0 - 0.018, p.width * 0.6, 0.007])
    cb = fig.colorbar(sc, cax=cax, orientation="horizontal", ticks=ticks)
    cb.ax.tick_params(labelsize=5, length=1.5, pad=1); cb.outline.set_linewidth(0.3)
    cb.set_label(label, fontsize=5.3, labelpad=1)
    return cax


def density_diff(ax, C, g1, g2, src, vmax=None):
    """Difference of participant-balanced densities of injured cells, over the outline of all cells (grey)."""
    D = S.read("expr_cell_atlas/density.tsv"); D = D[D.cells == "injured"]
    A = D[D.group == g1].pivot(index="y", columns="x", values="density")
    B = D[D.group == g2].pivot(index="y", columns="x", values="density")
    diff = gaussian_filter(A.values - B.values, 1.5) * 1e2           # percentage points of a participant's injured cells
    both = gaussian_filter(A.values + B.values, 1.5)
    diff[both < np.nanpercentile(both[both > 0], 35)] = np.nan
    out = pd.DataFrame(diff, index=A.index, columns=A.columns).reset_index().melt(id_vars="y", var_name="x",
                                                                                  value_name="difference_smoothed")
    S.src_out(out, src)
    ax.scatter(C.umap1, C.umap2, s=0.12, c="#E4E4E4", lw=0, rasterized=True, zorder=0)
    v = vmax or np.nanpercentile(np.abs(diff), 98)
    dx = (A.columns[1] - A.columns[0]) / 2; dy = (A.index[1] - A.index[0]) / 2
    im = ax.imshow(diff, origin="lower", cmap="RdBu_r", norm=TwoSlopeNorm(0, -v, v), interpolation="bilinear",
                   extent=[A.columns.min() - dx, A.columns.max() + dx, A.index.min() - dy, A.index.max() + dy],
                   aspect="equal", zorder=1, alpha=0.95)
    lab = {"DKD": "DKD", "DM_noCKD": "Diabetes, no CKD", "HKD": "HKD"}
    _frame(ax, f"Injured cells: {lab[g1]} minus {lab[g2]}")
    ax.set_xlim(C.umap1.quantile(0.001), C.umap1.quantile(0.999)); ax.set_ylim(C.umap2.quantile(0.001), C.umap2.quantile(0.999))
    return im, v


def routing_by_group(ax, m, src):
    P = S.read("expr_cell_atlas/state_by_expert.tsv", dtype={"participant": str})
    P = P[(P.pt_state == "injured_PT") & P.group.isin(GROUPS)]
    S.src_out(P[["participant", "group", f"p_{m}"]], src)
    data = [P[P.group == g][f"p_{m}"].values for g in GROUPS]
    S.strip_box(ax, data, range(len(GROUPS)), [S.GROUP_COL[g] for g in GROUPS], s=3)
    ax.set_xticks(range(len(GROUPS))); ax.set_xticklabels([GLAB[g] for g in GROUPS], fontsize=5.4)
    ax.set_title(EXPERT_TITLE[m], fontsize=6.3, pad=2)
    ax.set_ylim(0, None)


def exchange_groups(ax, src):
    G = S.read("expr_heldout_experts/exchange_gene_groups.tsv")
    lab = {"complement": "Complement", "matrix": "Matrix and adhesion", "tnf_family": "TNF family",
           "glucocorticoid_responsive": "Glucocorticoid-responsive", "tubular_signature_down": "Differentiated tubule",
           "tubular_markers_not_in_signature": "Tubule markers"}
    G = G[G.gene_group.isin(lab)].copy(); G["label"] = G.gene_group.map(lab)
    G = G.set_index("gene_group").loc[list(lab)].reset_index()
    S.src_out(G, src)
    y = np.arange(len(G))[::-1]
    for yi, r in zip(y, G.itertuples()):
        c = "#C44E52" if r.mean_change > 0 and r.ci_low > 0 else "#4C72B0" if r.ci_high < 0 else S.GREY
        ax.plot([r.ci_low * 100, r.ci_high * 100], [yi, yi], color=c, lw=1.0)
        ax.scatter([r.mean_change * 100], [yi], s=14, color=c, zorder=3, lw=0)
    ax.axvline(0, color=S.GREY, lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(G.label, fontsize=5.6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel("Decoded change, HKD-weighted\nto DKD-weighted (×10$^{-2}$ log units)")
    ax.set_title(f"Gene groups, held-out HKD (n = {int(G.n_participants.iloc[0])})", fontsize=6.3, pad=2)
