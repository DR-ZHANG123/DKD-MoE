"""Shared panels for Fig. 6 and Supplementary Fig. histology-direction: per-participant within-section agreement
(box + points) and the morphology dot matrix. Plotting only; inputs are alignment result tables."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import _style_dl as S

FEAT_LABEL = {"nuclei_per_1e4px_tissue": "Nuclear density", "eosin_frac": "Eosin-rich area",
              "eosin_mean": "Eosin intensity", "hema_mean": "Haematoxylin intensity",
              "hema_frac": "Haematoxylin-rich area", "white_frac": "Unstained area",
              "pt_score": "PT-lineage score"}


def load_track(src_name: str, summary_rel: str, cohort: str, label: str, track: str):
    d = pd.read_csv(S.SRC / src_name, sep="\t", dtype={"participant": str})
    s = S.read(summary_rel)
    return d[d.cohort == cohort], s[s.cohort == cohort], label, track


def within_box(ax, parts, var, ylabel, out_name, color="#8172B3"):
    """parts: [(per-participant table, summary table, label, track)]; one box per cohort."""
    rows, summ, data = [], [], []
    for d, s, lab, track in parts:
        x = d[var].dropna().values; data.append(x)
        rows.append(d[["participant", "cohort", "group", var]].assign(track=track))
        summ.append(s[s.variable == var].assign(track=track))
    S.src_out(pd.concat(rows), f"{out_name}_participants")
    Sm = S.src_out(pd.concat(summ), f"{out_name}_summary")
    S.strip_box(ax, data, range(len(data)), [color] * len(data), s=3.5, width=0.6)
    ax.axhline(0, color=S.GREY, lw=0.6, ls="--")
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([f"{lab}\n(n = {len(x)})" for (_, _, lab, _), x in zip(parts, data)], fontsize=5.4)
    ax.set_ylabel(ylabel)
    top = max(np.max(x) for x in data)
    for k, (r, x) in enumerate(zip(Sm.itertuples(), data)):
        npos = int((x > 0).sum())
        ax.text(k, top + 0.05, f"{npos}/{len(x)} > 0\n" + S.fmt_p(r.wilcoxon_p).replace("P = ", "P =\n"), ha="center", va="bottom",
                fontsize=5.2)
    ax.set_ylim(min(np.min(x) for x in data) - 0.05, top + 0.36)
    return Sm


def morph_dots(ax, parts, feats, out_name, v=0.4, note_y=-2.05):
    """Rows = cohorts, columns = features; colour = mean within-section rho, size = -log10 P."""
    rows = []
    for d, s, lab, track in parts:
        for f in feats:
            r = s[s.variable == f]
            r = r.iloc[0] if len(r) else None
            rows.append(dict(cohort=lab, track=track, feature=f, label=FEAT_LABEL[f],
                             n=int(r.n) if r is not None else 0,
                             mean_rho=float(r["mean"]) if r is not None else np.nan,
                             wilcoxon_p=float(r.wilcoxon_p) if r is not None else np.nan))
    d = S.src_out(pd.DataFrame(rows), out_name)
    norm = TwoSlopeNorm(0, -v, v)
    labs = [p[2] for p in parts]
    for i, lab in enumerate(labs):
        for j, f in enumerate(feats):
            r = d[(d.cohort == lab) & (d.feature == f)].iloc[0]
            if r.n < 10 or not np.isfinite(r.mean_rho):
                ax.scatter(j, i, s=10, marker="x", color="#BBBBBB", lw=0.6)
                continue
            sz = 8 + 22 * min(4, -np.log10(r.wilcoxon_p))
            ax.scatter(j, i, s=sz, color=plt.cm.RdBu_r(norm(np.clip(r.mean_rho, -v, v))),
                       ec=S.INK if r.wilcoxon_p < 0.05 else "none", lw=0.5)
    ax.set_yticks(range(len(labs))); ax.set_yticklabels(labs, fontsize=5.6)
    ax.set_xticks(range(len(feats))); ax.set_xticklabels([FEAT_LABEL[f] for f in feats], fontsize=5.6, rotation=90)
    ax.set_ylim(len(labs) - 0.5, -0.6); ax.set_xlim(-0.6, len(feats) - 0.4)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cax = ax.inset_axes([0.05, 1.18, 0.9, 0.09])
    cb = plt.colorbar(sm, cax=cax, orientation="horizontal")
    cax.xaxis.set_ticks_position("top"); cax.xaxis.set_label_position("top")
    cb.set_label("Mean within-section ρ", fontsize=5.8); cb.ax.tick_params(labelsize=5.3)
    cb.outline.set_linewidth(0.4)
    ax.text(0.5, note_y, "Outlined: P < 0.05; size: −log10 P\n×: fewer than 10 sections", transform=ax.transAxes,
            ha="center", va="top", fontsize=5.2, color="#555555")
    return d
