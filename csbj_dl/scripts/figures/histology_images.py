"""Image panels of Fig. 6: spot-level image scores on H&E sections (d) and example patches with pixel
attributions (e). Plotting only: thumbnails come from results/align_heatmaps*/ (written by the alignment stage),
spot scores from figures/source_data/align_*spot_scores.tsv.gz, patches and attributions from results/align_attrib/.
Images are shown with their native aspect ratio (imshow, aspect='equal')."""
from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from PIL import Image
import _style_dl as S

Image.MAX_IMAGE_PIXELS = None
SIGMA_PX = 8          # display smoothing of attribution maps (224-px patches)


def section_table() -> pd.DataFrame:
    a = pd.read_csv(S.SRC / "align_heatmap_sections.tsv", sep="\t", dtype={"participant": str})
    a["track"] = "native"; a["thumb_dir"] = "align_heatmaps"
    b = pd.read_csv(S.SRC / "align_lowres_heatmap_sections.tsv", sep="\t", dtype={"participant": str})
    b = b[b.cohort == "gse339455"].assign(track="matched_4um", thumb_dir="align_heatmaps_lowres")
    order = [("kpmp", "DKD"), ("kpmp", "HKD"), ("gse211785", "DKD"), ("gse211785", "HKD"), ("gse339455", "HKD")]
    t = pd.concat([a, b]).set_index(["cohort", "group"]).loc[order].reset_index()
    return t


def spot_scores(t: pd.DataFrame) -> pd.DataFrame:
    nat = pd.read_csv(S.SRC / "align_spot_scores.tsv.gz", sep="\t", dtype={"participant": str},
                      usecols=["cohort", "section", "px_row", "px_col", "image_score", "pt_rich"])
    low = pd.read_csv(S.SRC / "align_lowres_spot_scores.tsv.gz", sep="\t", dtype={"participant": str},
                      usecols=["cohort", "section", "px_row", "px_col", "image_score", "pt_rich"])
    parts = []
    for r in t.itertuples():
        src = nat if r.track == "native" else low
        d = src[src.section == r.section].copy()
        d["x_thumb"] = d.px_col / r.downsample; d["y_thumb"] = d.px_row / r.downsample
        d["track"] = r.track; parts.append(d)
    return pd.concat(parts)


def panel_heatmaps(fig, axes, cax, prefix="FigS_histology_direction_c"):
    t = section_table(); sp = spot_scores(t)
    S.src_out(t, f"{prefix}_sections")
    S.src_out(sp, f"{prefix}_spot_scores")
    v = float(np.nanpercentile(np.abs(sp.image_score), 98))
    norm = TwoSlopeNorm(0, -v, v)
    for ax, r in zip(axes, t.itertuples()):
        img = np.asarray(Image.open(S.RES / r.thumb_dir / r.thumbnail).convert("RGB"))
        d = sp[sp.section == r.section]
        # crop to the spot area (+ margin) so the tissue fills the panel; aspect is preserved
        m = 0.06 * max(np.ptp(d.x_thumb.values), np.ptp(d.y_thumb.values))
        x0, x1 = max(0, d.x_thumb.min() - m), min(img.shape[1], d.x_thumb.max() + m)
        y0, y1 = max(0, d.y_thumb.min() - m), min(img.shape[0], d.y_thumb.max() + m)
        ax.imshow(img, aspect="equal", interpolation="lanczos")
        ax.scatter(d.x_thumb, d.y_thumb, c=d.image_score, cmap="RdBu_r", norm=norm,
                   s=spot_size(ax, fig, d, x1 - x0, y1 - y0), lw=0, alpha=0.8)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_visible(True); spn.set_linewidth(0.4); spn.set_color("#BBBBBB")
        coh = S.COHORT_LABEL.get(r.cohort, r.cohort).replace(" held-out", "")
        ax.set_title(f"{coh}, {r.group}", fontsize=6.3, pad=2)
    sm = plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r")
    cb = fig.colorbar(sm, cax=cax)
    cb.set_label("Image score along the\nDKD–HKD direction (z)", fontsize=5.8)
    cb.ax.tick_params(labelsize=5.5); cb.outline.set_linewidth(0.4)


def spot_size(ax, fig, d, wx, wy):
    """Marker area (pt^2) so that a spot covers about 70% of the nearest-neighbour distance."""
    xy = d[["x_thumb", "y_thumb"]].values
    if len(xy) > 2000:
        xy = xy[np.random.default_rng(0).choice(len(xy), 2000, replace=False)]
    dd = np.sqrt(((xy[:, None, :] - xy[None, :, :]) ** 2).sum(-1)); np.fill_diagonal(dd, np.inf)
    nn = np.median(dd.min(1))
    bb = ax.get_window_extent(); scale = min(bb.width / wx, bb.height / wy) * 72 / fig.dpi
    return float((0.75 * nn * scale) ** 2)


def smooth(a, sigma=SIGMA_PX):
    """Display-only Gaussian smoothing of the signed pixel attribution (ViT token-grid texture removed)."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(a, sigma)


def panel_sig_pair(fig, ax_m, ax_i, cax, out_name="FigS_histology_d_signature_pair"):
    """Measured vs image-side transfer-signature score on the same H&E section (native-resolution track).
    Both maps are standardised within the section for display (z), so they share one colour scale."""
    sec = pd.read_csv(S.SRC / "align_sig_heatmap_section.tsv", sep="\t", dtype={"participant": str}).iloc[0]
    sp = pd.read_csv(S.SRC / "align_sig_spot_scores.tsv.gz", sep="\t", dtype={"participant": str},
                     usecols=["section", "participant", "group", "spot_id", "px_row", "px_col", "measured_sig",
                              "image_sig", "pt_rich"])
    d = sp[sp.section == sec.section].copy()
    for c in ["measured_sig", "image_sig"]:
        d[c + "_z"] = (d[c] - d[c].mean()) / d[c].std()
    d["x_thumb"] = d.px_col / sec.downsample; d["y_thumb"] = d.px_row / sec.downsample
    S.src_out(d, out_name)
    img = np.asarray(Image.open(S.RES / "align_sig" / sec.thumbnail).convert("RGB"))
    v = float(np.nanpercentile(np.abs(d[["measured_sig_z", "image_sig_z"]].values), 98))
    norm = TwoSlopeNorm(0, -v, v)
    m = 0.06 * max(np.ptp(d.x_thumb.values), np.ptp(d.y_thumb.values))
    x0, x1 = max(0, d.x_thumb.min() - m), min(img.shape[1], d.x_thumb.max() + m)
    y0, y1 = max(0, d.y_thumb.min() - m), min(img.shape[0], d.y_thumb.max() + m)
    for ax, c, title in [(ax_m, "measured_sig_z", "Measured signature score"),
                         (ax_i, "image_sig_z", "Image-side signature score")]:
        ax.imshow(img, aspect="equal", interpolation="lanczos")
        ax.scatter(d.x_thumb, d.y_thumb, c=d[c], cmap="RdBu_r", norm=norm, s=spot_size(ax, fig, d, x1 - x0, y1 - y0),
                   lw=0, alpha=0.85)
        ax.set_xlim(x0, x1); ax.set_ylim(y1, y0); ax.set_aspect("equal")
        ax.set_xticks([]); ax.set_yticks([])
        for spn in ax.spines.values():
            spn.set_visible(True); spn.set_linewidth(0.4); spn.set_color("#BBBBBB")
        ax.set_title(title, fontsize=6.3, pad=2)
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap="RdBu_r"), cax=cax)
    cb.set_label("Score (z within section)", fontsize=5.8); cb.ax.tick_params(labelsize=5.5)
    cb.outline.set_linewidth(0.4)
    return sec, d


PATCH_SETS = {"direction": ("align_attrib", "image_score_crossfit"), "signature": ("align_sig_attrib", "score")}
SHORT_GROUP = {"DM_noCKD": "Diabetes", "HKD_withDM": "HKD + diabetes", "Reference": "Reference",
               "AKI": "AKI", "CKD_other": "Other CKD", "CKD_unadjudicated": "CKD"}


def panel_patches(axes_hi, axes_lo, which="signature", out_name="FigS_histology_e_patches"):
    """Highest/lowest-scoring patches and their pixel attributions (one patch per participant, as selected by the
    alignment stage); which = 'signature' (transfer-signature direction) or 'direction' (DKD-HKD direction)."""
    sub, col = PATCH_SETS[which]
    T = pd.read_csv(S.RES / sub / "patches.tsv", sep="\t", dtype={"participant": str})
    A = np.load(S.RES / sub / "attributions.npz")
    att = dict(zip(A["idx"], A["attr"]))
    n = len(axes_hi[0])
    sel = pd.concat([T[T.extreme == "high"].sort_values(col, ascending=False).head(n),
                     T[T.extreme == "low"].sort_values(col).head(n)])
    S.src_out(sel.drop(columns=["patch_png"]).assign(patch=f"{sub}/" + sel.patch_png), out_name)
    vmax = np.percentile(np.abs(np.stack([smooth(att[i]) for i in sel.idx])), 99.5)
    for axes, ext in [(axes_hi, "high"), (axes_lo, "low")]:
        rows = sel[sel.extreme == ext]
        for k, r in enumerate(rows.itertuples()):
            img = np.asarray(Image.open(S.RES / sub / r.patch_png).convert("RGB")).astype(float) / 255
            a0, a1 = axes[0][k], axes[1][k]
            a0.imshow(img, aspect="equal")
            a1.imshow(img.mean(-1), cmap="Greys_r", vmin=0, vmax=1.3, aspect="equal")
            a1.imshow(smooth(att[r.idx]), cmap="RdBu_r", vmin=-vmax, vmax=vmax, alpha=0.65, aspect="equal")
            score = getattr(r, col)
            a0.set_title(f"{SHORT_GROUP.get(r.group, r.group)}\n{score:+.1f}".replace("-", "−"), fontsize=5.2,
                         pad=1.5, linespacing=1.0)
            for a in (a0, a1):
                a.set_xticks([]); a.set_yticks([])
                for spn in a.spines.values():
                    spn.set_visible(True); spn.set_linewidth(0.3); spn.set_color("#BBBBBB")
    return vmax
