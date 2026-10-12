"""Supplementary figure — H&E morphology aligned to spot expression.

(a) cross-modal retrieval: per-participant median rank of the true expression profile among candidate spots of the
    same section (image query; lower is better; 0.5 = chance), aligned model vs shuffled-pair baseline.
    KPMP held-out participants and GSE211785 at native resolution; GSE339455 (published only as down-sampled H&E)
    with the matched-resolution model (all cohorts resampled to 4 um/px). External cohorts standardised with their
    own statistics (label-free). Sources: figures/source_data/align_retrieval_participants.tsv,
    align_lowres_retrieval_participants.tsv; P from results/align_train*/retrieval_vs_baseline.tsv.
(b) within-section agreement between the image-side and the measured transfer-signature score (per-participant
    Spearman over spots); figures/source_data/align_sig_within_section_correlations.tsv (native) and
    align_lowres_sig_within_section_correlations.tsv (GSE339455), summaries in results/align_sig*/.
(c) within-section correlation of the image-side signature score with morphology features and the PT-lineage score.
(d) measured vs image-side signature score on one held-out DKD section (align_sig_spot_scores.tsv.gz).
(e) patches with the highest/lowest image-side signature score (one per participant) and their pixel attributions
    (results/align_sig_attrib/).
The DKD-HKD direction panels are in FigS_histology_direction.py.
(d, e) see histology_images.py."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import _style_dl as S
from histology_images import panel_patches, panel_sig_pair
from histology_common import load_track, within_box, morph_dots

COH = [("kpmp_heldout", "native", "KPMP\nheld-out"), ("gse211785_external_calibrated", "native", "GSE211785"),
       ("gse339455_external_calibrated", "matched", "GSE339455\n(4 µm/px)")]
MORPH_SIG = ["eosin_frac", "eosin_mean", "hema_mean", "hema_frac", "nuclei_per_1e4px_tissue", "pt_score"]


def panel_a(ax):
    nat = pd.read_csv(S.SRC / "align_retrieval_participants.tsv", sep="\t", dtype={"participant": str})
    low = pd.read_csv(S.SRC / "align_lowres_retrieval_participants.tsv", sep="\t", dtype={"participant": str})
    vb = {"native": S.read("align_train/retrieval_vs_baseline.tsv"),
          "matched": S.read("align_train_lowres/retrieval_vs_baseline.tsv")}
    rows, tests = [], []
    for ev, track, lab in COH:
        src = nat if track == "native" else low
        for model in ["aligned_within_section", "shuffled_within_section"]:
            d = src[(src.eval_set == ev) & (src.model == model)]
            rows.append(d.assign(track=track, panel_label=lab.replace("\n", " ")))
        t = vb[track]
        t = t[(t.model == "aligned_within_section") & (t.eval_set == ev) & (t.metric == "median_rank_pct_img2expr")]
        tests.append(t.assign(track=track))
    d = pd.concat(rows); T = pd.concat(tests)
    S.src_out(d[["participant", "group", "eval_set", "track", "model", "median_rank_pct_img2expr"]], "FigS_histology_a_retrieval")
    S.src_out(T, "FigS_histology_a_tests")
    pos, data, cols = [], [], []
    for k, (ev, track, lab) in enumerate(COH):
        for j, (model, c) in enumerate([("aligned_within_section", "#8172B3"), ("shuffled_within_section", "#BFBFBF")]):
            x = d[(d.eval_set == ev) & (d.model == model)].median_rank_pct_img2expr.values
            pos.append(k * 2.5 + j); data.append(x); cols.append(c)
    S.strip_box(ax, data, pos, cols, s=3.5, width=0.7)
    ax.axhline(0.5, color=S.GREY, lw=0.6, ls="--")
    ax.set_xticks([k * 2.5 + 0.5 for k in range(len(COH))])
    ns = [len(data[2 * k]) for k in range(len(COH))]
    ax.set_xticklabels([f"{lab}\n(n = {n})" for (_, _, lab), n in zip(COH, ns)], fontsize=5.8)
    ax.set_ylabel("Median rank of true spot\n(fraction of candidates)")
    ax.set_ylim(-0.02, 0.9)
    for k, (ev, _, _) in enumerate(COH):
        p = T[T.eval_set == ev].wilcoxon_p.iloc[0]
        ax.text(k * 2.5 + 0.5, 0.86, S.fmt_p(p).replace("P = ", "P =\n"), ha="center", va="top", fontsize=5.3)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(fc="#8172B3", label="Aligned"), Patch(fc="#BFBFBF", label="Shuffled pairs")],
              loc="center left", bbox_to_anchor=(0.0, 0.66), fontsize=5.6, handlelength=1.0, frameon=False)


def panel_b(ax):
    parts = [load_track("align_sig_within_section_correlations.tsv", "align_sig/within_section_summary.tsv", "kpmp",
                        "KPMP", "native"),
             load_track("align_sig_within_section_correlations.tsv", "align_sig/within_section_summary.tsv",
                        "gse211785", "GSE211785", "native"),
             load_track("align_lowres_sig_within_section_correlations.tsv",
                        "align_sig_lowres/within_section_summary.tsv", "gse339455", "GSE339455", "matched")]
    within_box(ax, parts, "measured_sig", "Within-section ρ, image-side\nvs measured signature",
               "FigS_histology_b_signature_agreement")


def panel_c(ax):
    parts = [load_track("align_sig_within_section_correlations.tsv", "align_sig/within_section_summary.tsv", "kpmp",
                        "KPMP", "native"),
             load_track("align_sig_within_section_correlations.tsv", "align_sig/within_section_summary.tsv",
                        "gse211785", "GSE211785", "native"),
             load_track("align_lowres_sig_within_section_correlations.tsv",
                        "align_sig_lowres/within_section_summary.tsv", "gse339455", "GSE339455", "matched")]
    morph_dots(ax, parts, MORPH_SIG, "FigS_histology_c_signature_morphology", v=0.4)


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.3, 7.0))
    g1 = fig.add_gridspec(1, 2, width_ratios=[1.12, 1.1], wspace=0.4, left=0.08, right=0.6, top=0.965,
                          bottom=0.72)
    axa, axb = (fig.add_subplot(g1[0, i]) for i in range(2))
    axc = fig.add_axes([0.76, 0.835, 0.225, 0.075])
    g2 = fig.add_gridspec(1, 4, width_ratios=[1, 1, 0.05, 0.75], wspace=0.08, left=0.03, right=0.97,
                          top=0.6, bottom=0.31)
    axd = [fig.add_subplot(g2[0, i]) for i in range(2)]; cax = fig.add_subplot(g2[0, 2])
    axnote = fig.add_subplot(g2[0, 3]); axnote.axis("off")
    ncol = 6
    g3 = fig.add_gridspec(2, 2 * ncol + 1, width_ratios=[1] * ncol + [0.35] + [1] * ncol, wspace=0.06, hspace=0.06,
                          left=0.03, right=0.93, top=0.235, bottom=0.085)
    hi = [[fig.add_subplot(g3[r, c]) for c in range(ncol)] for r in range(2)]
    lo = [[fig.add_subplot(g3[r, ncol + 1 + c]) for c in range(ncol)] for r in range(2)]
    cax2 = fig.add_axes([0.945, 0.095, 0.012, 0.12])
    panel_a(axa); panel_b(axb); panel_c(axc)
    sec, dd = panel_sig_pair(fig, axd[0], axd[1], cax)
    rho = dd[["measured_sig", "image_sig"]].corr(method="spearman").iloc[0, 1]
    axnote.text(0.42, 0.5, f"Held-out DKD participant\n{sec.participant}, one Visium section\n\n"
                f"Spot-level Spearman ρ = {rho:.2f}\n({len(dd)} spots)", transform=axnote.transAxes,
                ha="left", va="center", fontsize=6)
    vmax = panel_patches(hi, lo, which="signature")
    sm = plt.cm.ScalarMappable(norm=plt.Normalize(-vmax, vmax), cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=cax2, ticks=[-vmax, 0, vmax])
    cb.ax.set_yticklabels(["−", "0", "+"], fontsize=5.5); cb.outline.set_linewidth(0.4)
    cb.set_label("Pixel attribution", fontsize=5.8)
    fig.canvas.draw()
    for row, txt in [(hi, "Highest image-side signature scores"), (lo, "Lowest image-side signature scores")]:
        b0, b1 = row[0][0].get_position(), row[0][-1].get_position()
        fig.text((b0.x0 + b1.x1) / 2, b0.y1 + 0.04, txt, ha="center", va="bottom", fontsize=6.3, fontweight="bold")
    b = hi[0][0].get_position(); b2 = hi[1][0].get_position()
    fig.text(b.x0 - 0.006, (b.y0 + b.y1) / 2, "H&E", rotation=90, ha="right", va="center", fontsize=5.8)
    fig.text(b2.x0 - 0.006, (b2.y0 + b2.y1) / 2, "Attribution", rotation=90, ha="right", va="center", fontsize=5.8)
    L = S.letters(fig, [[(axa, "a"), (axb, "b"), (axc, "c")]], col_align=False)
    L[0].set_x(0.003)
    fig.canvas.draw(); r = fig.canvas.get_renderer(); inv = fig.transFigure.inverted()
    yd = max(inv.transform_bbox(a.get_tightbbox(r)).y1 for a in axd) + 0.006
    S.letter_at(fig, 0.003, yd, "d")
    tt = [t for t in fig.texts if t.get_text().startswith("Highest image-side")][0]
    S.letter_at(fig, 0.003, inv.transform_bbox(tt.get_window_extent(r)).y1 + 0.004, "e")
    S.save(fig, "FigS_histology")


if __name__ == "__main__":
    main()
