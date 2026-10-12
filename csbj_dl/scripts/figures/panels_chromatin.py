"""Panels: chromatin alignment, motifs and expert components in chromatin (used by Fig6_chromatin.py and FigS_chromatin_alignment.py).

Original figure description — Fig. 7 — chromatin accessibility of the same nuclei carries the DKD-versus-HKD readout.

Sources (results/chrom_*): chrom_readout/retrieval_by_model.tsv (RNA-to-ATAC retrieval among 100 candidate nuclei of
held-out KPMP participants, aligned vs shuffled pairs; one point per model), participant_scores.tsv, kpmp_tests.tsv,
kpmp_agreement.tsv (participant-level chromatin-side readout vs RNA readout), external_tests.tsv (donor level),
chrom_motifs/motif_report.tsv (motif enrichment in the top-attributed peaks of each direction vs GC-matched
background), chrom_expert/{participant_table,group_tests}.tsv (chromatin-side score of the DKD-weighted expert
contribution by group), chrom_expert/motif_vs_expert.tsv (participant motif accessibility against the RNA-side
contributions of the DKD- and HKD-weighted experts, partial Spearman).
Chromatin features: 'peaks' = consensus peak matrix; 'gact' = gene-activity scores (used where an external cohort
has no peak matrix compatible with the KPMP consensus peaks)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style_dl as S

FEAT = {"peaks": "Peaks", "gact": "Gene activity"}
PRIMARY = "peaks"
FAMILY = {"REL": "NF-κB/REL", "RELA": "NF-κB/REL", "RELB": "NF-κB/REL", "NFKB1": "NF-κB/REL", "NFKB2": "NF-κB/REL",
          "STAT3": "STAT3", "NR3C1": "NR3C1", "HNF4A": "HNF4A/HNF1", "HNF4G": "HNF4A/HNF1", "HNF1A": "HNF4A/HNF1",
          "HNF1B": "HNF4A/HNF1"}
FAM_COL = {"NF-κB/REL": "#8172B3", "STAT3": "#DD8452", "NR3C1": "#937860", "HNF4A/HNF1": "#4C72B0",
           "AP-1": "#C44E52"}


def family(name: str) -> str:
    n = name.upper()
    if n in FAMILY:
        return FAMILY[n]
    if any(k in n for k in ("FOS", "JUN", "ATF3")):
        return "AP-1"
    return ""


def panel_a(ax):
    R = S.read("chrom_readout/retrieval_by_model.tsv")
    S.src_out(R, "chr_a_retrieval")
    pos, data, cols, ticks, labs, chance = [], [], [], [], [], []
    k = 0
    for f in [x for x in FEAT if x in set(R.feature)]:
        for metric, ch, ml in [("top1_at100", 0.01, "top 1"), ("top10_at100", 0.10, "top 10")]:
            for j, (pairs, c) in enumerate([("aligned", "#55A868"), ("shuffled", "#BFBFBF")]):
                pos.append(k * 2.4 + j); cols.append(c)
                data.append(R[(R.feature == f) & (R.pairs == pairs)][metric].values)
            ticks.append(k * 2.4 + 0.5); labs.append(f"{FEAT[f].replace(' ', chr(10))}\n{ml}"); chance.append((k * 2.4 - 0.45, k * 2.4 + 1.45, ch))
            k += 1
    S.strip_box(ax, data, pos, cols, s=4, width=0.7)
    for x0, x1, ch in chance:
        ax.plot([x0, x1], [ch, ch], color=S.GREY, lw=0.7, ls="--")
    ax.set_xticks(ticks); ax.set_xticklabels(labs, fontsize=5.6)
    ax.set_ylabel("Same nucleus retrieved among\n100 candidates (fraction)")
    ax.set_ylim(0, 1.08)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#55A868", label="Aligned"), Patch(fc="#BFBFBF", label="Shuffled pairs"),
                       plt.Line2D([], [], color=S.GREY, ls="--", lw=0.7, label="Chance")],
              loc="upper right", fontsize=5.5, frameon=False, handlelength=1.2)
    n = len(R[(R.feature == PRIMARY) & (R.pairs == "aligned")])
    ax.set_title(f"Held-out KPMP nuclei ({n} models per encoder)", fontsize=6, pad=3)


def bracket(ax, x1, x2, y, label, h):
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=S.INK, lw=0.6)
    ax.text((x1 + x2) / 2, y + h * 0.4, label, ha="center", va="bottom", fontsize=5.3)


def panel_b(ax_s, ax_g):
    P = S.read("chrom_readout/participant_scores.tsv", dtype={"participant": str})
    P = P[P.feature == PRIMARY]
    A = S.read("chrom_readout/kpmp_agreement.tsv"); T = S.read("chrom_readout/kpmp_tests.tsv")
    S.src_out(P.drop(columns=[c for c in P.columns if c.startswith("frac_open")], errors="ignore"),
              "chr_b_participants")
    show = ["Reference", "DM_noCKD", "AKI", "HKD", "HKD_withDM", "DKD"]
    for g in [x for x in show if x in set(P.group)]:
        d = P[P.group == g]
        ax_s.scatter(d.rna_readout, d.chrom_score, s=9, color=S.GROUP_COL.get(g, "#BBBBBB") if g != "AKI" else "#DA8BC3",
                     lw=0, alpha=0.85, label=S.GROUP_LABEL.get(g, g).replace("\n", " "))
    a = A[(A.feature == PRIMARY) & (A.participants == "all") & (A.score == "chrom_score") & (A.versus == "rna_readout")]
    a = a.iloc[0]
    ax_s.text(0.03, 0.97, f"Spearman ρ = {a.spearman:.2f}\n{S.fmt_p(a.p)}, n = {int(a.n)}", transform=ax_s.transAxes,
              ha="left", va="top", fontsize=5.6)
    ax_s.set_xlabel("RNA readout, cross-fitted (log-odds)"); ax_s.set_ylabel("Chromatin-side score")
    ax_s.legend(loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, fontsize=5.0, frameon=False,
                handletextpad=0.1, borderaxespad=0.1, columnspacing=0.6, labelspacing=0.25)
    groups = ["Reference", "HKD", "DKD"]
    data = [P[P.group == g].chrom_score.values for g in groups]
    t = T[(T.space == PRIMARY) & (T.feature == "chrom_score") &
          T.comparison.isin(["DKD_vs_HKD", "DKD_vs_Reference", "HKD_vs_Reference"])]
    S.src_out(t, "chr_b_tests")
    S.strip_box(ax_g, data, range(3), [S.GROUP_COL[g] for g in groups], s=5)
    ax_g.set_xticks(range(3)); ax_g.set_xticklabels([f"{g}\n(n = {len(x)})" for g, x in zip(groups, data)],
                                                    fontsize=5.6)
    top = max(np.max(x) for x in data); lo = min(np.min(x) for x in data); h = 0.035 * (top - lo)
    for (a_, b_, comp), lev in zip([(1, 2, "DKD_vs_HKD"), (0, 1, "HKD_vs_Reference"), (0, 2, "DKD_vs_Reference")],
                                   [0.1, 0.33, 0.56]):
        r = t[t.comparison == comp].iloc[0]
        bracket(ax_g, a_, b_, top + lev * (top - lo), f"g = {r.g:.2f}, {S.fmt_p(r.p_wilcoxon)}", h)
    ax_g.set_ylim(lo - 0.05 * (top - lo), top + 0.85 * (top - lo))
    ax_g.set_ylabel("Chromatin-side score")


def panel_c(ax):
    E = S.read("chrom_readout/external_tests.tsv")
    E = E[E.feature == "chrom_score"].set_index(["cohort", "space", "comparison"])
    keys = [("gse195460", "peaks", "DKD_vs_Control"), ("gse195460", "gact", "DKD_vs_Control"),
            ("gse211785", "gact", "DKD_vs_HKD")]
    d = E.loc[[k for k in keys if k in E.index]].reset_index()
    S.src_out(d, "chr_c_external")
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        c = "#C44E52"
        ax.plot([r.ci_low, r.ci_high], [yi, yi], color=c, lw=0.9)
        ax.scatter([r.g], [yi], s=16, marker="s", color=c, lw=0, zorder=3)
    ax.axvline(0, color=S.GREY, lw=0.6, ls="--")
    lab = [f"{r.cohort.upper()}, {r.comparison.replace('_vs_', ' vs ').replace('Control', 'control')}\n"
           f"({FEAT[r.space].lower()}; {int(r.n1)} vs {int(r.n2)})" for r in d.itertuples()]
    ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=5.5); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    xr = d.ci_high.max() + 0.2
    for yi, r in zip(y, d.itertuples()):
        ax.text(xr, yi, f"g = {r.g:.2f}\n{S.fmt_p(r.p_wilcoxon)}", va="center", ha="left", fontsize=5.3)
    ax.set_xlim(min(-1.0, d.ci_low.min() - 0.2), xr + 1.6); ax.set_ylim(-0.6, len(d) - 0.4)
    ax.set_xlabel("Hedges' g, donor level (95% CI)")


DKD_SHOW = [("RELA", "NF-κB/REL"), ("REL", "NF-κB/REL"), ("NFKB2", "NF-κB/REL"), ("NFKB1", "NF-κB/REL"),
            ("TEAD3", "TEAD"), ("TEAD1", "TEAD"), ("NFIX", "NFI"), ("NFIB", "NFI"), ("NR3C1", "NR3C1"),
            ("FOSL1::JUNB", "AP-1"), ("JUN", "AP-1"), ("HNF1B", "HNF1"), ("HNF1A", "HNF1"), ("STAT3", "STAT3")]
FAM_COL.update({"TEAD": "#55A868", "NFI": "#64B5CD", "HNF1": "#4C72B0"})


def _motif_bars(ax, d, col):
    y = np.arange(len(d))[::-1]
    for yi, r in zip(y, d.itertuples()):
        sig = r.q_bh < 0.05
        ax.barh(yi, r.log2_or, 0.66, color=col if sig else "white", ec=col, lw=0.7)
        ax.text(max(r.log2_or, 0) + 0.04, yi, f"q = {r.q_bh:.1g}" if sig else "ns", va="center", ha="left",
                fontsize=5.0, color="#555555")
    ax.axvline(0, color=S.INK, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels(d.motif, fontsize=5.6); ax.tick_params(axis="y", length=0)
    ax.set_xlabel("log2 odds ratio vs GC-matched\nbackground peaks", fontsize=6)
    ax.set_xlim(min(-0.4, d.log2_or.min() - 0.1), d.log2_or.max() + 0.55)


def panel_d(ax_dkd, ax_hkd, n_hkd=10):
    M = S.read("chrom_motifs/motif_report.tsv").sort_values("p_fisher")
    dk = M[M.direction == "DKD_side"].drop_duplicates("motif").set_index("motif")
    hk = M[M.direction == "HKD_side"].drop_duplicates("motif")
    d1 = pd.DataFrame([dict(motif=m, family=f, log2_or=dk.loc[m, "log2_or"], q_bh=dk.loc[m, "q_bh"],
                            p_fisher=dk.loc[m, "p_fisher"], rank=dk.loc[m, "rank_in_direction"])
                       for m, f in DKD_SHOW if m in dk.index])
    d2 = hk.sort_values("rank_in_direction").head(n_hkd)[["motif", "log2_or", "q_bh", "p_fisher", "rank_in_direction"]]
    S.src_out(d1, "chr_d_motifs_dkd_side"); S.src_out(d2, "chr_d_motifs_hkd_side")
    _motif_bars(ax_dkd, d1, "#C44E52"); _motif_bars(ax_hkd, d2.reset_index(drop=True), "#4C72B0")
    for t, f in zip(ax_dkd.get_yticklabels(), d1.family):
        t.set_color(FAM_COL.get(f, S.INK))
    ax_dkd.set_title("DKD-side attributed peaks", fontsize=6.2, pad=3)
    ax_hkd.set_title("HKD-side attributed peaks (top motifs)", fontsize=6.2, pad=3)


def panel_e(ax, n_top=10):
    G = S.read("chrom_attrib/top_genes.tsv")
    G = G.assign(abs_z=G.sum_z.abs())   # sum_z is signed (HKD-side peaks carry negative z)
    parts = [G[G.direction == "DKD_side"].sort_values("abs_z", ascending=False).head(n_top),
             G[G.direction == "HKD_side"].sort_values("abs_z", ascending=False).head(n_top)]
    d = S.src_out(pd.concat(parts), "chr_e_peak_genes")
    labs, vals, cols, npk = [], [], [], []
    for p, col, sgn in [(parts[0], "#C44E52", 1), (parts[1], "#4C72B0", -1)]:
        for r in p.itertuples():
            labs.append(r.nearest_gene); vals.append(sgn * abs(r.sum_z)); cols.append(col); npk.append(r.n_peaks)
    y = np.arange(len(labs))[::-1]
    ax.barh(y, vals, color=cols, height=0.7, ec="white", lw=0.3)
    for yi, v, n in zip(y, vals, npk):
        ax.text(v + (0.3 if v > 0 else -0.3), yi, f"{n}", va="center", ha="left" if v > 0 else "right", fontsize=5.0)
    ax.set_yticks(y); ax.set_yticklabels(labs, fontsize=5.6, fontstyle="italic"); ax.tick_params(axis="y", length=0)
    ax.axvline(0, color=S.INK, lw=0.5)
    m = max(abs(v) for v in vals) * 1.3; ax.set_xlim(-m, m)
    ax.set_xlabel("Summed attribution z of nearby top peaks\n(DKD side +, HKD side −; number of peaks)",
                  fontsize=6)


EXP_GROUPS = ["HKD", "HKD_withDM", "DM_noCKD", "DKD"]
MOTIF_SHOW = [("RELA", "RELA"), ("REL", "REL"), ("NFKB1", "NFKB1"), ("NFKB2", "NFKB2"), ("FOS::JUN (MA0099.4)", "FOS::JUN"),
              ("NR3C1", "NR3C1"), ("TEAD1", "TEAD1"), ("HNF1B", "HNF1B"), ("HNF4A (MA0114.5)", "HNF4A")]


def panel_f(ax):
    T = S.read("chrom_expert/participant_table.tsv", dtype={"participant": str})
    G = S.read("chrom_expert/group_tests.tsv"); G = G[G.feature == "chrom_M3"].set_index("comparison")
    d = T[T.group.isin(EXP_GROUPS)]
    S.src_out(d[["participant", "group", "chrom_M3", "rna_M3", "n_cells"]], "chr_f_expert_chromatin")
    data = [d[d.group == g].chrom_M3.values for g in EXP_GROUPS]
    S.strip_box(ax, data, range(4), [S.GROUP_COL[g] for g in EXP_GROUPS], s=4)
    ax.set_xticks(range(4)); ax.set_xticklabels([f"{l}\n(n = {len(x)})" for l, x in
                                                 zip(["HKD", "HKD\n+ DM", "DM,\nno CKD", "DKD"], data)], fontsize=5.4)
    ax.set_ylabel("Chromatin-side DKD-\nweighted expert score")
    top = max(np.max(x) for x in data); span = top - min(np.min(x) for x in data); rows = []
    for (i, j, comp), lev in zip([(2, 3, "DKD_vs_DM_noCKD"), (0, 2, "DM_noCKD_vs_HKD"), (0, 3, "DKD_vs_HKD")], [0.06, 0.26, 0.46]):
        r = G.loc[comp]; rows.append(r)
        y = top + lev * span
        ax.plot([i, i, j, j], [y - 0.03 * span, y, y, y - 0.03 * span], color=S.INK, lw=0.6)
        ax.text((i + j) / 2, y + 0.01 * span, f"g = {r.g:.2f}, {S.fmt_p(r.p_wilcoxon)}", ha="center", va="bottom", fontsize=5.4)
    S.src_out(pd.DataFrame(rows).reset_index(), "chr_f_tests")
    ax.set_ylim(ax.get_ylim()[0], top + 0.72 * span)


def panel_g(ax):
    M = S.read("chrom_expert/motif_vs_expert.tsv"); M = M[M.participants == "disease"]
    rows = []
    for key, lab in MOTIF_SHOW:
        for comp in ("M3", "M1"):
            r = M[(M.motif == key) & (M.component == comp)].iloc[0]
            rows.append(dict(motif=lab, component=comp, partial_spearman=r.partial_spearman, spearman=r.spearman,
                             p=r.partial_p, q=r.partial_q, n=int(r.n)))
    d = S.src_out(pd.DataFrame(rows), "chr_g_motif_vs_expert")
    y = np.arange(len(MOTIF_SHOW))[::-1]; h = 0.38
    for k, (comp, col, lab) in enumerate((("M3", "#C44E52", "DKD-weighted expert"), ("M1", "#4C72B0", "HKD-weighted expert"))):
        q = d[d.component == comp]
        ax.barh(y + (h / 2 if k == 0 else -h / 2), q.partial_spearman, height=h, color=col, label=f"{lab}, adjusted")
        yy = y + (h / 2 if k == 0 else -h / 2)
        ax.scatter(q.spearman, yy, s=7, facecolor="white", edgecolor=S.INK, lw=0.5, zorder=4,
                   label="unadjusted" if k == 0 else None)
        for yi, v, raw, p in zip(y, q.partial_spearman, q.spearman, q.q):
            if p < 0.05:                     # Benjamini-Hochberg q of the partial correlation
                x = max(v, raw) + 0.05 if v > 0 else min(v, raw) - 0.05
                ax.text(x, yi + (h / 2 if k == 0 else -h / 2), "*", ha="left" if v > 0 else "right", va="center", fontsize=6)
    ax.axvline(0, color=S.INK, lw=0.5)
    ax.set_yticks(y); ax.set_yticklabels([l for _, l in MOTIF_SHOW], fontsize=5.8); ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.6, 0.85)
    ax.set_xlabel(f"Spearman with expert contribution\n(n = {d.n.iloc[0]} participants)", fontsize=6)
    ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1, fontsize=5.5, frameon=False, handlelength=1.0)
