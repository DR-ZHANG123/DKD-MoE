"""Supplementary figure — the image-side DKD-HKD direction (the expression readout projected into the aligned space).

(a) within-section agreement between the image score along the DKD-HKD direction and the spot expression readout
    (per-participant Spearman); figures/source_data/align_within_section_correlations.tsv (native) and
    align_lowres_within_section_correlations.tsv (GSE339455), summaries in results/align_interpret*/;
(b) within-section correlation of that image score with morphology features (KPMP at native and matched
    4 um/px resolution, GSE339455 at 4 um/px);
(c) the image score on five H&E sections (figures/source_data/align_*spot_scores.tsv.gz, results/align_heatmaps*/)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style_dl as S
from histology_common import load_track, within_box, morph_dots
from histology_images import panel_heatmaps

NAT = ("align_within_section_correlations.tsv", "align_interpret/within_section_correlation_summary.tsv")
LOW = ("align_lowres_within_section_correlations.tsv", "align_interpret_lowres/within_section_correlation_summary.tsv")
MORPH = ["nuclei_per_1e4px_tissue", "eosin_frac", "eosin_mean", "hema_mean", "hema_frac", "white_frac"]


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.3, 4.4))
    g1 = fig.add_gridspec(1, 1, left=0.1, right=0.42, top=0.95, bottom=0.6)
    axa = fig.add_subplot(g1[0, 0])
    axb = fig.add_axes([0.62, 0.79, 0.3, 0.12])
    g2 = fig.add_gridspec(1, 6, width_ratios=[1, 1, 1, 1, 1, 0.07], wspace=0.12, left=0.03, right=0.93,
                          top=0.4, bottom=0.02)
    axc = [fig.add_subplot(g2[0, i]) for i in range(5)]; cax = fig.add_subplot(g2[0, 5])
    parts = [load_track(*NAT, "kpmp", "KPMP", "native"), load_track(*NAT, "gse211785", "GSE211785", "native"),
             load_track(*LOW, "gse339455", "GSE339455", "matched")]
    within_box(axa, parts, "expr_score", "Within-section ρ, image score\nvs expression readout",
               "FigS_histology_direction_a_agreement")
    mparts = [load_track(*NAT, "kpmp", "KPMP", "native"), load_track(*LOW, "kpmp", "KPMP 4 µm/px", "matched"),
              load_track(*LOW, "gse339455", "GSE339455", "matched")]
    morph_dots(axb, mparts, MORPH, "FigS_histology_direction_b_morphology", v=0.35, note_y=-2.3)
    panel_heatmaps(fig, axc, cax, prefix="FigS_histology_direction_c")
    S.auto_letters(fig, [(axa, "a"), (axb, "b"), (axc, "c")])
    S.save(fig, "FigS_histology_direction")


if __name__ == "__main__":
    main()
