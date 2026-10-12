"""Fig. 6 — the DKD-weighted expert contribution and urine albumin and protein in KPMP participants with CKD.

(a) odds ratio per SD from proportional-odds models of albuminuria and proteinuria bands (DKD, HKD and HKD with
    diabetes), adjusted for injured-cell fraction, diagnosis group and eGFR band (filled) and additionally for age, sex,
    technology set and recruitment site (open), for the total readout, the readout of the supervised model without
    experts and the held-out contributions of the HKD-weighted (M1) and DKD-weighted (M3) experts
    (results/expr_clinical_models/models.tsv);
(b) DKD-weighted expert contribution by albuminuria band (results/expr_diabetes_groups/participant_table.tsv,
    results/expr_heldout_experts/contributions.tsv);
(c) cross-validated gain in out-of-fold log-likelihood per participant from adding each readout to the covariate
    model (50 repeats of five-fold cross-validation; mean and 2.5-97.5 percentiles; results/expr_clinical_models/
    cv_incremental.tsv). Bands are ordered categories; no midpoints are used."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import _style_dl as S

READ = {"moe": ("Total readout", "#C44E52"), "no_experts": ("No-expert readout", "#8172B3"),
        "M1": ("HKD-weighted expert", "#4C72B0"), "M3": ("DKD-weighted expert", "#7A1F22")}
BAND = {"albuminuria_band": "Albuminuria", "proteinuria_band": "Proteinuria"}


def panel_a(ax):
    M = S.read("expr_clinical_models/models.tsv")
    rows, y, yt, ylab, sect = [], 0.0, [], [], []
    for b, bl in BAND.items():
        sect.append((y, bl)); y -= 0.8
        for r_, (lab, col) in READ.items():
            for k, cv in enumerate(["diagnosis_egfr", "full"]):
                r = M[(M.band == b) & (M.readout == r_) & (M.covariates == cv)].iloc[0]
                rows.append(dict(band=b, readout=r_, covariates=cv, n=int(r.n), or_per_sd=r.or_per_sd, ci_low=r.ci_low,
                                 ci_high=r.ci_high, p=r.p, y=y - 0.34 * k, colour=col))
            yt.append(y - 0.17); ylab.append(lab); y -= 0.95
        y -= 0.25
    d = S.src_out(pd.DataFrame(rows), "Fig6_a_odds_ratios")
    for r in d.itertuples():
        full = r.covariates == "full"
        ax.plot([r.ci_low, r.ci_high], [r.y, r.y], color=r.colour, lw=0.9)
        ax.scatter([r.or_per_sd], [r.y], s=15, facecolor="white" if full else r.colour, edgecolor=r.colour, lw=0.9, zorder=3)
        ax.text(7.2, r.y, S.fmt_p(r.p), va="center", ha="left", fontsize=5.5)
    ax.set_xscale("log"); ax.axvline(1, color=S.GREY, lw=0.6, ls="--")
    ax.set_xlim(0.6, 15); ax.set_xticks([1, 2, 4]); ax.set_xticklabels(["1", "2", "4"]); ax.minorticks_off()
    ax.set_yticks(yt); ax.set_yticklabels(ylab, fontsize=6); ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    for yy, lab in sect:
        n = d[d.band == [k for k, v in BAND.items() if v == lab][0]].n.iloc[0]
        ax.text(0.62, yy - 0.25, f"{lab} (n = {n})", fontsize=6.5, fontweight="bold", va="center", ha="left")
    ax.set_ylim(y + 0.5, 0.3); ax.set_xlabel("Odds ratio per SD (95% CI)")
    h = [Line2D([], [], marker="o", ls="", color=S.INK, ms=4, label="diagnosis, injured fraction, eGFR"),
         Line2D([], [], marker="o", ls="", mfc="white", mec=S.INK, ms=4, label="+ age, sex, technology, site")]
    ax.legend(handles=h, title="Adjusted for:", title_fontsize=6, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=1,
              fontsize=5.6, frameon=False, handletextpad=0.3)
    ax.get_legend()._legend_box.align = "left"


def panel_b(ax):
    P = S.read("expr_diabetes_groups/participant_table.tsv", dtype={"participant": str}).set_index("participant")
    C = S.read("expr_heldout_experts/contributions.tsv", dtype={"participant": str}).set_index("participant")
    d = P[P.group.isin(["DKD", "HKD", "HKD_withDM"]) & P.albuminuria_band.ne("nan") & P.alb_ord.notna()].join(C[["M3"]])
    order = d[["albuminuria_band", "alb_ord"]].drop_duplicates().sort_values("alb_ord").albuminuria_band.tolist()
    S.src_out(d.reset_index()[["participant", "group", "albuminuria_band", "alb_ord", "M3", "injured_fraction"]], "Fig6_b_M3_albuminuria")
    data = [d[d.albuminuria_band == b].M3.values for b in order]
    S.strip_box(ax, data, range(len(order)), plt.cm.Reds(np.linspace(0.3, 0.85, len(order))), s=4)
    labs = [b.replace(" mg/g cr", "").replace(" to ", "–").replace(">=", "≥") for b in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels([f"{l}\n(n = {len(x)})" for l, x in zip(labs, data)], fontsize=5.6)
    ax.set_xlabel("Albuminuria band (mg/g creatinine)"); ax.set_ylabel("DKD-weighted expert\ncontribution (log-odds)")


def panel_c(ax):
    V = S.read("expr_clinical_models/cv_incremental.tsv")
    V = V[(V.band == "albuminuria_band") & (V.covariates == "diagnosis_egfr") & V.readout.isin(READ)]
    V = V.set_index("readout").loc[list(READ)].reset_index()
    S.src_out(V, "Fig6_c_cv_gain")
    y = np.arange(len(V))[::-1]
    for yi, r in zip(y, V.itertuples()):
        c = READ[r.readout][1]
        ax.plot([r.delta_q025 * 100, r.delta_q975 * 100], [yi, yi], color=c, lw=1.0)
        ax.scatter([r.delta_oof_ll * 100], [yi], s=16, color=c, zorder=3, lw=0)
    ax.axvline(0, color=S.GREY, lw=0.6, ls="--")
    ax.set_yticks(y); ax.set_yticklabels([READ[r][0] for r in V.readout], fontsize=6); ax.tick_params(axis="y", length=0)
    ax.set_ylim(-0.6, len(V) - 0.4)
    ax.set_xlabel("Cross-validated gain in log-likelihood\nper participant (×10$^{-2}$), albuminuria")


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.25, 4.6))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.75, wspace=0.55, left=0.17, right=0.985, top=0.86,
                          bottom=0.12)
    axa = fig.add_subplot(gs[:, 0]); axb = fig.add_subplot(gs[0, 1]); axc = fig.add_subplot(gs[1, 1])
    panel_a(axa); panel_b(axb); panel_c(axc)
    L = S.letters(fig, [[(axa, "a"), (axb, "b")]], col_align=False)
    xb = L[1].get_position()[0]
    fig.canvas.draw(); r = fig.canvas.get_renderer()
    yc = fig.transFigure.inverted().transform_bbox(axc.get_tightbbox(r, for_layout_only=True)).y1 + 0.006
    S.letter_at(fig, xb, yc, "c")
    S.save(fig, "Fig6_clinical")


if __name__ == "__main__":
    main()
