"""Fig. 4 — the participant readout by diagnosis and diabetes, and its decomposition into expert contributions.

(a) KPMP held-out participants (each scored by the partition models that never saw them; seed-mean log-odds) by
group (results/expr_diabetes_groups/{participant_table,group_tests}.tsv);
(b) DKD versus HKD within single-cell and single-nucleus data (results/expr_readout/{participant_scores,
donor_level_tests}.tsv);
(c) group differences in the standardized readout and in the standardized HKD- and DKD-weighted expert contributions,
adjusted for injured-cell fraction, cell number, age, sex, technology set, recruitment site and eGFR band
(results/expr_diabetes_groups/group_models.tsv);
(d) held-out contributions of the HKD-weighted (M1) and DKD-weighted (M3) experts by group
(results/expr_heldout_experts/{contributions,contribution_tests}.tsv)."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import _style_dl as S

GROUPS = ["Reference", "HKD", "HKD_withDM", "DM_noCKD", "DKD"]


def bracket(ax, x1, x2, y, label, h=0.08):
    ax.plot([x1, x1, x2, x2], [y - h, y, y, y - h], color=S.INK, lw=0.6)
    ax.text((x1 + x2) / 2, y + 0.05, label, ha="center", va="bottom", fontsize=5.8)


def test_row(T, cohort, tech, comp):
    r = T[(T.cohort == cohort) & (T.tech == tech) & (T.comparison == comp)].iloc[0]
    return r


def panel_a(ax):
    P = S.read("expr_diabetes_groups/participant_table.tsv", dtype={"participant": str})
    T = S.read("expr_diabetes_groups/group_tests.tsv"); T = T[T.readout == "moe"].set_index("comparison")
    d = P[P.group.isin(GROUPS)]
    S.src_out(d[["participant", "group", "moe", "n_cells"]].rename(columns={"moe": "readout"}), "Fig4_a_groups")
    data = [d[d.group == g].moe.values for g in GROUPS]
    S.strip_box(ax, data, range(len(GROUPS)), [S.GROUP_COL[g] for g in GROUPS], s=4)
    ax.set_xticks(range(len(GROUPS)))
    ax.set_xticklabels([f"{S.GROUP_LABEL[g]}\n(n = {len(x)})" for g, x in zip(GROUPS, data)], fontsize=5.8)
    ax.set_ylabel("Readout (log-odds, seed mean)")
    top = max(np.max(x) for x in data); rows = []
    for (i, j, comp), lev in zip([(3, 4, "DKD_vs_DM_noCKD"), (1, 2, "HKD_withDM_vs_HKD"), (1, 3, "DM_noCKD_vs_HKD"),
                                  (1, 4, "DKD_vs_HKD")], [0.35, 0.35, 1.3, 2.25]):
        r = T.loc[comp]; rows.append(r)
        bracket(ax, i, j, top + lev, f"g = {r.g:.2f}, {S.fmt_p(r.p_wilcoxon)}")
    S.src_out(pd.DataFrame(rows).reset_index(), "Fig4_a_tests")
    ax.set_ylim(ax.get_ylim()[0], top + 3.1)


def panel_b(ax, P, T):
    d = P[(P.cohort == "KPMP_heldout") & P.tech.isin(["sc", "sn"]) & P.group.isin(["HKD", "DKD"])]
    S.src_out(d[["participant", "tech", "group", "logodds", "n_cells"]], "Fig4_b_sc_sn")
    pos, data, cols, labs = [], [], [], []
    for k, tech in enumerate(["sc", "sn"]):
        for j, g in enumerate(["HKD", "DKD"]):
            x = d[(d.tech == tech) & (d.group == g)].logodds.values
            pos.append(k * 2.6 + j); data.append(x); cols.append(S.GROUP_COL[g]); labs.append(f"{g}\n(n = {len(x)})")
    S.strip_box(ax, data, pos, cols, s=5)
    ax.set_xticks(pos); ax.set_xticklabels(labs, fontsize=6)
    top = max(np.max(x) for x in data)
    rows = []
    for k, tech in enumerate(["sc", "sn"]):
        r = test_row(T, "KPMP_heldout", tech, "DKD_vs_HKD"); rows.append(r)
        bracket(ax, k * 2.6, k * 2.6 + 1, top + 0.35, f"g = {r.g:.2f}\n{S.fmt_p(r.p_wilcoxon)}")
        ax.text(k * 2.6 + 0.5, top + 1.75, S.TECH_LABEL[tech], ha="center", va="bottom", fontsize=6.5,
                fontweight="bold")
    S.src_out(pd.DataFrame(rows), "Fig4_b_tests")
    ax.set_ylim(ax.get_ylim()[0], top + 2.35)
    ax.set_ylabel("Readout (log-odds)")


def forest(ax, labels, est, lo, hi, colors, diamond_idx=None, ref=0.0):
    y = np.arange(len(labels))[::-1]
    for yi, e, l, h, c, k in zip(y, est, lo, hi, colors, range(len(labels))):
        if diamond_idx is not None and k == diamond_idx:
            ax.fill([l, e, h, e], [yi, yi + 0.28, yi, yi - 0.28], color=c, lw=0)
        else:
            ax.plot([l, h], [yi, yi], color=c, lw=1.0)
            ax.scatter([e], [yi], s=18, color=c, zorder=3, marker="s", lw=0)
    ax.axvline(ref, color=S.GREY, lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=6)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    return y


def panel_c(ax):
    M = S.read("expr_diabetes_groups/group_models.tsv")
    sets = [("moe", "Readout"), ("M1", "HKD-weighted expert"), ("M3", "DKD-weighted expert")]
    terms = [("DKD", "DKD vs HKD"), ("DM_noCKD", "DM, no CKD vs HKD"), ("DKD_minus_DM_noCKD", "DKD vs DM, no CKD")]
    col = {"DKD": S.GROUP_COL["DKD"], "DM_noCKD": S.GROUP_COL["DM_noCKD"], "DKD_minus_DM_noCKD": "#7A1F22"}
    rows = []
    for sc, sl in sets:
        for t, tl in terms:
            r = M[(M.readout == sc) & (M.model == "adjusted_egfr") & (M.term == t)].iloc[0]
            rows.append(dict(readout=sc, term=t, label=f"{sl}: {tl}", beta=r.beta, lo=r.ci_low, hi=r.ci_high, p=r.p, n=int(r.n)))
    d = S.src_out(pd.DataFrame(rows), "Fig4_c_models")
    y = forest(ax, d.label, d.beta, d.lo, d.hi, [col[t] for t in d.term])
    xr = d.hi.max() + 0.12
    for yi, p in zip(y, d.p):
        ax.text(xr, yi, S.fmt_p(p), va="center", ha="left", fontsize=5.6)
    for k in (3, 6):
        ax.axhline(y[k] + 0.5, color="#DDDDDD", lw=0.6)
    ax.set_xlim(min(-0.9, d.lo.min() - 0.1), xr + 0.75)
    ax.set_xlabel(f"Adjusted difference (SD, 95% CI)\nn = {d.n.iloc[0]} participants")


def panel_d(axs):
    C = S.read("expr_heldout_experts/contributions.tsv", dtype={"participant": str})
    T = S.read("expr_heldout_experts/contribution_tests.tsv").set_index(["expert", "comparison"])
    grp = ["HKD", "HKD_withDM", "DM_noCKD", "DKD"]
    d = C[C.group.isin(grp)]
    S.src_out(d[["participant", "group", "M1", "M3"]], "Fig4_d_contributions")
    rows = []
    for ax, m, title in zip(axs, ["M1", "M3"], ["HKD-weighted expert (M1)", "DKD-weighted expert (M3)"]):
        data = [d[d.group == g][m].values for g in grp]
        S.strip_box(ax, data, range(len(grp)), [S.GROUP_COL[g] for g in grp], s=4)
        ax.set_xticks(range(len(grp))); ax.set_xticklabels(["HKD", "HKD\n+ DM", "DM,\nno CKD", "DKD"], fontsize=5.6)
        ax.set_title(title, fontsize=6.5, pad=2)
        top = max(np.max(x) for x in data); span = top - min(np.min(x) for x in data)
        for (i, j, comp), lev in zip([(2, 3, "DKD_vs_DM_noCKD"), (0, 2, "DM_noCKD_vs_HKD"), (0, 3, "DKD_vs_HKD")],
                                     [0.06, 0.24, 0.42]):
            r = T.loc[(m, comp)]; rows.append(dict(expert=m, comparison=comp, **r.to_dict()))
            bracket(ax, i, j, top + lev * span, f"g = {r.g:.2f}, {S.fmt_p(r.p_wilcoxon)}", h=0.03 * span)
        ax.set_ylim(ax.get_ylim()[0], top + 0.7 * span)
    axs[0].set_ylabel("Contribution to readout\n(log-odds)")
    S.src_out(pd.DataFrame(rows), "Fig4_d_tests")


def main():
    S.setup()
    P = S.read("expr_readout/participant_scores.tsv", dtype={"participant": str})
    T = S.read("expr_readout/donor_level_tests.tsv")
    fig = plt.figure(figsize=(S.MAX_W - 0.25, 6.6))
    g1 = fig.add_gridspec(1, 2, width_ratios=[1.35, 0.85], wspace=0.3, left=0.09, right=0.985, top=0.96, bottom=0.62)
    g2 = fig.add_gridspec(1, 3, width_ratios=[1.25, 0.8, 0.8], wspace=0.45, left=0.24, right=0.955, top=0.43,
                          bottom=0.07)
    axa, axb = fig.add_subplot(g1[0, 0]), fig.add_subplot(g1[0, 1])
    axc, axd1, axd2 = fig.add_subplot(g2[0, 0]), fig.add_subplot(g2[0, 1]), fig.add_subplot(g2[0, 2])
    panel_a(axa); panel_b(axb, P, T); panel_c(axc); panel_d([axd1, axd2])
    S.letters(fig, [[(axa, "a"), (axb, "b")], [(axc, "c"), (axd1, "d")]])
    S.save(fig, "Fig4_readout")


if __name__ == "__main__":
    main()
