"""Panels: model selection, seed agreement, reproducible experts and their external contributions (used by Fig2_model.py and FigS_comparison_external.py).

Original figure description — Fig. 2 — choice of the number of experts, seed stability and reproducible meta-experts.

(a) held-out participant cross-validation over K x condition weight (results/expr_cv/cv_table.tsv);
(b) agreement between the five random initializations on held-out KPMP participants (results/expr_stability/before_after.tsv,
    row after_seeds1-5);
(c) meta-experts across five seeds: presence and condition-head weight per seed (results/expr_meta/units.tsv,
    meta_experts.tsv); (d) out-of-sample meta-expert contributions per donor in GSE211785 and GSE195460
    (results/expr_readout/meta_contributions.tsv, meta_contribution_tests.tsv)."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
from matplotlib.patches import Rectangle
import _style_dl as S


def panel_a(ax, cax_note=None):
    cv = S.read("expr_cv/cv_table.tsv")
    sel = json.load(open(S.RES / "expr_cv" / "selection.json"))
    cv = cv.sort_values(["K", "w_cond"]).reset_index(drop=True)
    cols = [("nll", "Held-out\nloss", False, "{:.4f}"), ("auc_dkd_hkd", "DKD vs\nHKD AUC", True, "{:.3f}"),
            ("frac_top1_gt_0.5", "Specific\nrouting", True, "{:.2f}"),
            ("entropy_over_logK", "Routing\nentropy", False, "{:.2f}"),
            ("stability_ari", "Seed\nARI", True, "{:.2f}"), ("rank_sum", "Rank\nsum", False, "{:.0f}")]
    out = cv[["K", "w_cond", "eligible"] + [c for c, *_ in cols]].copy()
    out["selected"] = (out.K == sel["K"]) & (out.w_cond == sel["w_cond"])
    S.src_out(out, "sel_a_cv_table")
    n, m = len(out), len(cols)
    M = np.zeros((n, m))
    for j, (c, _, hi, _) in enumerate(cols):
        v = out[c].astype(float).values
        z = (v - v.min()) / (np.ptp(v) + 1e-12)
        M[:, j] = z if hi else 1 - z
    ax.imshow(M, cmap="Greens", vmin=-0.15, vmax=1.35, aspect="auto")
    for i in range(n):
        for j, (c, _, _, f) in enumerate(cols):
            ax.text(j, i, f.format(out[c].iloc[i]), ha="center", va="center", fontsize=6,
                    color=("#E6E6E6" if M[i, j] > 0.7 else "#9A9A9A") if not out.eligible.iloc[i]
                    else ("white" if M[i, j] > 0.7 else S.INK))
    ax.set_xticks(range(m)); ax.set_xticklabels([l for _, l, *_ in cols], fontsize=5.8)
    ax.xaxis.tick_top(); ax.tick_params(axis="x", length=0, pad=2)
    ylab = [f"K = {k}, w = {w:g}" + ("" if e else "  (diffuse)") for k, w, e in zip(out.K, out.w_cond, out.eligible)]
    ax.set_yticks(range(n)); ax.set_yticklabels(ylab, fontsize=6); ax.tick_params(axis="y", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    i = int(np.where(out.selected)[0][0])
    ax.add_patch(Rectangle((-0.5, i - 0.5), m, 1, fill=False, ec="#C44E52", lw=1.2, zorder=5))
    ax.set_xlim(-0.5, m - 0.5); ax.set_ylim(n - 0.5, -0.5)
    ax.text(m / 2 - 0.5, n - 0.5 + 0.35, "Darker = better; red frame = selected; diffuse = routing not specific",
            ha="center", va="top", fontsize=5.8, color="#555555")


def panel_b(ax):
    r = S.read("expr_stability/before_after.tsv").set_index("run").loc["after_seeds1-5"]
    rows = [("Expert\nassignment\nARI", r.seed_ari), ("Matched\nexpert\ncorrelation", r.seed_matched_r),
            ("Readout\nICC", r.logodds_icc_2_1), ("DKD vs HKD\nAUC", r.auc_dkd_hkd_seedmean_logodds)]
    d = S.src_out(pd.DataFrame(rows, columns=["metric", "value"]), "sel_b_stability")
    x = np.arange(len(d))
    ax.bar(x, d.value, 0.55, color="#55A868", ec="white", lw=0.3)
    for xi, v in zip(x, d.value):
        ax.text(xi, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=5.8)
    ax.set_xticks(x); ax.set_xticklabels(d.metric, fontsize=6)
    ax.set_ylim(0, 1.12); ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0]); ax.set_ylabel("Value (five initializations,\nheld-out participants)")


def panel_c(ax, axb):
    u = S.read("expr_meta/units.tsv"); me = S.read("expr_meta/meta_experts.tsv")
    me["order"] = me.meta_expert.str[1:].astype(int)
    me = me.sort_values("order").reset_index(drop=True)
    order = me.meta_expert.tolist(); seeds = sorted(u.seed.unique())
    P = u.pivot_table(index="seed", columns="meta_expert", values="head_weight_dkd_minus_hkd", aggfunc="mean")
    P = P.reindex(index=seeds, columns=order)
    long = P.reset_index().melt(id_vars="seed", var_name="meta_expert", value_name="head_weight")
    long["present"] = long.head_weight.notna()
    S.src_out(long, "sel_c_presence")
    S.src_out(me[["meta_expert", "n_seeds", "head_weight", "reproducible", "mean_within_r"]], "sel_c_head_weight")
    v = np.nanmax(np.abs(P.values)); norm = TwoSlopeNorm(0, -v, v)
    ax.set_facecolor("white")
    for i, s in enumerate(seeds):
        for j, mname in enumerate(order):
            val = P.loc[s, mname]
            if np.isfinite(val):
                ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fc=plt.cm.RdBu_r(norm(val)), ec="none"))
            else:
                ax.add_patch(Rectangle((j - 0.42, i - 0.42), 0.84, 0.84, fc="#F3F3F3", ec="none"))
    ax.set_xlim(-0.5, len(order) - 0.5); ax.set_ylim(len(seeds) - 0.5, -0.5)
    ax.set_yticks(range(len(seeds))); ax.set_yticklabels([f"Seed {s}" for s in seeds], fontsize=6)
    ax.set_xticks([]); ax.tick_params(axis="y", length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    nrep = int(me.reproducible.sum())
    ax.add_patch(Rectangle((-0.5, -0.5), nrep, len(seeds), fill=False, ec=S.INK, lw=0.9, ls="--"))
    ax.text(nrep / 2 - 0.5, -0.65, "in all five seeds", ha="center", va="bottom", fontsize=6)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cb = plt.colorbar(sm, ax=ax, fraction=0.035, pad=0.015)
    cb.set_label("Condition-head weight\n(DKD minus HKD)", fontsize=6); cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    cols = ["#C44E52" if h > 0 else "#4C72B0" for h in me.head_weight]
    axb.bar(range(len(order)), me.head_weight, color=cols, width=0.7, ec="white", lw=0.3)
    axb.axhline(0, color=S.INK, lw=0.5)
    axb.set_xticks(range(len(order))); axb.set_xticklabels(order, fontsize=6, rotation=90)
    axb.set_xlim(-0.5, len(order) - 0.5); axb.set_ylabel("Mean head\nweight", fontsize=6.5)
    axb.set_xlabel("Meta-expert (experts matched across seeds)")
    return cb


GROUPS_EXT = [("gse211785", "Control"), ("gse211785", "HKD"), ("gse211785", "DKD"), ("gse195460", "Control"),
              ("gse195460", "DKD")]


def panel_d(axes):
    """Out-of-sample contribution (head weight x mean routing) of the reproducible meta-experts per donor in the
    external cohorts; models never saw these donors."""
    C = S.read("expr_readout/meta_contributions.tsv", dtype={"participant": str})
    T = S.read("expr_readout/meta_contribution_tests.tsv")
    me = S.read("expr_meta/meta_experts.tsv"); rep = me[me.reproducible].meta_expert.tolist()
    C = C[(C.tech == "pooled") & C.cohort.isin(["gse211785", "gse195460"])]
    S.src_out(C[["participant", "cohort", "group"] + rep], "sel_d_external_contributions")
    tt = T[(T.tech == "pooled") & T.cohort.isin(["gse211785", "gse195460"]) & T.feature.isin(rep)]
    S.src_out(tt, "sel_d_tests")
    pos = [0, 1, 2, 3.6, 4.6]; rng = np.random.default_rng(0)
    for ax, m in zip(axes, rep):
        for x, (coh, g) in zip(pos, GROUPS_EXT):
            v = C[(C.cohort == coh) & (C.group == g)][m].values
            ax.scatter(x + rng.uniform(-0.14, 0.14, len(v)), v, s=7, color=S.GROUP_COL[g], lw=0, alpha=0.85, zorder=3)
            ax.plot([x - 0.25, x + 0.25], [v.mean()] * 2, color=S.INK, lw=1.0, zorder=4)
        ax.axvline(2.8, color="#DDDDDD", lw=0.6)
        ax.set_ylabel(m, rotation=0, ha="right", va="center", fontsize=6.5, labelpad=4)
        ax.set_xlim(-0.5, 5.1)
        lo, hi = ax.get_ylim(); ax.set_ylim(lo, hi + (0.9 if m == rep[-1] else 0.3) * (hi - lo))
        if m == rep[-1]:
            for coh, comp, xc in [("gse211785", "DKD_vs_HKD", 1.5), ("gse195460", "DKD_vs_Control", 4.1)]:
                r = tt[(tt.cohort == coh) & (tt.comparison == comp) & (tt.feature == m)].iloc[0]
                lab = comp.replace("_vs_", " vs ").replace("Control", "control")
                ax.text(xc, 0.97, f"{lab}\ng = {r.g:.2f}, {S.fmt_p(r.p_wilcoxon)}", transform=ax.get_xaxis_transform(),
                        ha="center", va="top", fontsize=5.0)
        if m != rep[-1]:
            ax.set_xticks([])
            ax.spines["bottom"].set_visible(False)
    n = {k: int(((C.cohort == k[0]) & (C.group == k[1])).sum()) for k in GROUPS_EXT}
    axes[-1].set_xticks(pos)
    axes[-1].set_xticklabels([f"{g}\n({n[(c, g)]})" for c, g in GROUPS_EXT],
                             fontsize=5.6)
    axes[-1].text(1.0, -0.62, "GSE211785", transform=axes[-1].get_xaxis_transform(), ha="center", va="top",
                  fontsize=6, fontweight="bold")
    axes[-1].text(4.1, -0.62, "GSE195460", transform=axes[-1].get_xaxis_transform(), ha="center", va="top",
                  fontsize=6, fontweight="bold")
    axes[0].set_title("Meta-expert contribution per donor (head weight × routing)", fontsize=6.2, pad=3)
