"""Fig. 1 — study design and DKD-MoE.

(a) participant groups, data types and the five participant-level parts of KPMP, and the external cohorts used only
for testing; (b) DKD-MoE: encoder, gating network, six experts, latent state, negative binomial decoder conditioned on
technology, the condition classifier on the routing, the technology discriminator and the training objective;
(c) the two contrastive alignments; (d) the participant readout as a sum of expert contributions (group means of the
held-out contributions, results/expr_heldout_experts/contributions.tsv) and the analyses built on it.
Illustrated icons: figures/source_data/Fig1_icons/ (fig1_icons_generate.py); all text, arrows and the model diagram
are vector. Participant counts come from dkd_moe/config/splits.tsv and the result tables (figures/source_data/
Fig1_counts.tsv); the number of experts from results/expr_cv/selection.json. Coordinates are millimetres."""
from __future__ import annotations
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, Rectangle
import _style as S
import fig1_glyphs as Gl
from fig1_draw import box, txt, arrow, icon, trapezoid, panel_frame, PAIRS, BOXES, INK

W, H = 175.0, 164.0
RED, BLUE, BROWN, GREY = "#C44E52", "#4C72B0", "#937860", "#8C8C8C"
NET = "#46607A"
EXPERT_COL = {0: BLUE, 1: BROWN, 2: RED}                   # HKD-weighted, healthy-cell, DKD-weighted


def counts():
    sp = pd.read_csv(S.CDL / "config" / "splits.tsv", sep="\t", dtype={"participant": str})
    ps = S.read("expr_readout/participant_scores.tsv", dtype={"participant": str})
    ps = ps[ps.tech == "pooled"]
    er = S.read("expr_transfer/scores_ercb.tsv"); zn = S.read("expr_transfer/scores_zenodo.tsv", dtype={"donor": str})
    vx = S.read("data_visium/external_section_table.tsv")
    vx = vx[vx.usable.astype(str) == "True"]
    pt = S.read("expr_prep/kpmp_pt_cells_by_participant.tsv", dtype={"participant": str})
    vs = S.read("align_prep/sections.tsv", dtype={"participant": str})
    ch = json.load(open(S.RES / "chrom_summary" / "summary.json"))
    rows = [("KPMP", "all", pt.participant.nunique()), ("KPMP", "parts", sp.fold.nunique())]
    gp = sp.set_index("participant").group.reindex(pt.participant.unique())
    rows += [("group", k, int((gp == k).sum())) for k in ["DKD", "HKD_withDM", "HKD", "DM_noCKD", "Reference"]]
    rows += [("KPMP", m, pt[pt.tech == m].participant.nunique()) for m in ["sc", "sn"]]
    rows += [("KPMP", "multiome", int(ch["participants_used"])),
             ("KPMP", "visium", vs[vs.cohort == "kpmp"].participant.nunique())]
    for coh, key in [("GSE211785", "gse211785"), ("GSE195460", "gse195460")]:
        rows.append((coh, "donors", ps[ps.cohort == key].participant.nunique()))
    rows.append(("GSE104954", "samples", len(er)))
    rows.append(("Zenodo19868428", "xenium_donors_scored", zn[zn.platform == "Xenium"].donor.nunique()))
    ar = pd.read_csv(S.SRC / "align_retrieval_participants.tsv", sep="\t", dtype={"participant": str})
    rows.append(("GSE211785", "visium_participants",
                 ar[ar.eval_set == "gse211785_external"].participant.nunique()))
    rows.append(("GSE339455", "tissue_regions", vx.donor_unit.nunique()))
    K = json.load(open(S.RES / "expr_cv" / "selection.json"))["K"]
    rows.append(("model", "K", K))
    return S.src_out(pd.DataFrame(rows, columns=["cohort", "item", "n"]), "Fig1_counts")


GROUPS = [("DKD", "DKD", RED, True, True), ("HKD_withDM", "HKD with diabetes", "#8172B3", True, True),
          ("HKD", "HKD", BLUE, False, True), ("DM_noCKD", "Diabetes, no CKD", "#B59F3B", True, False),
          ("Reference", "Reference tissue", GREY, False, False)]


def panel_a(ax, n):
    g = lambda c, i: int(n[(n.cohort == c) & (n.item == i)].n.iloc[0])
    panel_frame(ax, 4, 113, 117, 163, "#EEF3FA", f"KPMP kidney biopsies ({g('KPMP', 'all')} participants)", "kpmp")
    for i, (k, lab, col, dm, htn) in enumerate(GROUPS):
        cx = 15.5 + i * 22.0
        icon(ax, "kidney", cx - 1.5, 148.0, h=12.0)
        if dm:
            icon(ax, "glucose", cx + 5.2, 152.0, h=4.6)
        if htn:
            icon(ax, "bp_cuff", cx + 6.0, 145.2, h=4.2)
        txt(ax, cx, 139.6, lab, fs=5.7, fw="bold", col=col)
        txt(ax, cx, 136.4, f"n = {g('group', k)}", fs=5.5)
    mods = [("pt_cell", "Single-cell RNA", "sc"), ("nucleus", "Single-nucleus RNA", "sn"),
            ("chromatin", "Multiome (RNA + ATAC)", "multiome"), ("visium", "Visium + H&E", "visium")]
    for i, (ic, lab, k) in enumerate(mods):
        x = 7.0 + [0, 27.0, 55.0, 85.0][i]
        icon(ax, ic, x + 4.0, 126.0, h=8.0 if ic != "visium" else 6.0)
        txt(ax, x + 9.5, 127.6, lab, fs=5.6, ha="left")
        txt(ax, x + 9.5, 124.4, f"n = {g('KPMP', k)}", fs=5.4, ha="left", col="#555555")
    txt(ax, 7.0, 117.0, "Participants split into five parts; each part held out once", fs=5.5, ha="left")
    for i in range(g("KPMP", "parts")):
        ax.add_patch(Rectangle((80.0 + i * 5.6, 115.4), 4.8, 3.2, fc=BLUE if i < 4 else "#DD8452", ec="white", lw=0.4,
                               alpha=0.8, zorder=4))
    txt(ax, 108.4, 117.0, "held out", fs=5.3, ha="left", col="#B25F2A")
    panel_frame(ax, 122, 113, 174, 163, "#FBF0E6", "Independent cohorts: testing only", "ext")
    items = [("pt_cell", "Single-cell and spatial atlas",
              f"{g('GSE211785', 'donors')} donors; Visium, {g('GSE211785', 'visium_participants')} donors"),
             ("chromatin", "Single-nucleus RNA and ATAC", f"{g('GSE195460', 'donors')} donors"),
             ("microarray", "Tubulointerstitial microarrays", f"{g('GSE104954', 'samples')} biopsies"),
             ("xenium", "Imaging-based spatial transcriptomics", f"{g('Zenodo19868428', 'xenium_donors_scored')} donors"),
             ("visium", "Visium with H&E", f"{g('GSE339455', 'tissue_regions')} tissue regions")]
    for i, (ic, name, acc) in enumerate(items):
        ya = 152.4 - i * 8.6
        icon(ax, ic, 128.6, ya, h=6.2 if ic not in ("visium", "microarray") else 4.6)
        txt(ax, 134.2, ya + 1.6, name, fs=5.6, fw="bold", ha="left")
        txt(ax, 134.2, ya - 1.7, acc, fs=5.1, ha="left", col="#444444")
    arrow(ax, 117.6, 138.0, 121.4, 138.0, lw=1.2)


def panel_b(ax, K):
    panel_frame(ax, 4, 57, 174, 110, "#F3F4F8", "DKD-MoE: diagnosis-conditioned mixture of experts", "model")
    ym = 85.5
    icon(ax, "tubule", 11.0, 99.0, h=9.0)
    Gl.expr_vector(ax, 9.0, 74.0, 4.0, 18.0, 12, 2)
    txt(ax, 11.0, 71.6, "counts $\\mathbf{x}_n$\n(3,000 genes)", fs=5.2, va="top")
    arrow(ax, 13.6, ym, 17.0, ym)
    trapezoid(ax, 17.5, 29.5, ym, 22.0, 9.0, NET, NET, "Encoder", fs=5.3, name="enc")
    txt(ax, 23.5, ym - 13.6, "$\\mathbf{h}_n$", fs=6)
    arrow(ax, 30.0, ym, 33.0, ym)
    probs = [0.30, 0.05, 0.42, 0.08, 0.09, 0.06][:K]
    Gl.softmax_bars(ax, 33.6, ym - 4.5, 12.0, 10.0, probs, [EXPERT_COL.get(i, "#9E9E9E") for i in range(K)], top={0, 2})
    txt(ax, 39.6, ym + 8.6, "Gating network", fs=5.8, fw="bold")
    txt(ax, 39.6, ym - 7.6, "$\\mathbf{p}_n$: softmax, top-2", fs=5.2)
    arrow(ax, 46.4, ym, 49.4, ym)
    eh, eg = 3.3, 0.7; tot = K * eh + (K - 1) * eg; top_ = ym + tot / 2
    box(ax, 50.0, ym - tot / 2 - 1.2, 64.5, ym + tot / 2 + 1.2, "white", NET, r=1.0, z=3, name="experts")
    for i in range(K):
        ya = top_ - i * (eh + eg) - eh; col = EXPERT_COL.get(i, "#BDBDBD")
        p = box(ax, 51.0, ya, 63.5, ya + eh, col if i in EXPERT_COL else "#F2F2F2", col, lw=0.5, r=0.6, z=4,
                track=False, alpha=0.9 if i in EXPERT_COL else 1.0)
        txt(ax, 57.25, ya + eh / 2, f"$e_{{{i + 1}}}$", p, fs=5.6, col="white" if i in EXPERT_COL else "#757575")
    txt(ax, 57.25, ym + tot / 2 + 3.4, "Experts", fs=5.8, fw="bold")
    for i, lab in {0: "HKD-weighted", 1: "healthy cells", 2: "DKD-weighted"}.items():
        yy = ym - tot / 2 - 3.6 - 2.6 * i
        ax.add_patch(Rectangle((50.4, yy - 0.8), 1.6, 1.6, fc=EXPERT_COL[i], ec="none", zorder=5))
        txt(ax, 52.8, yy, lab, fs=4.9, ha="left")
    arrow(ax, 64.8, ym, 67.0, ym)
    ax.add_patch(Circle((69.2, ym), 2.1, fc="white", ec=INK, lw=0.6, zorder=5)); txt(ax, 69.2, ym, "Σ", fs=6.6)
    arrow(ax, 71.4, ym, 73.6, ym)
    Gl.gaussian(ax, 79.5, ym - 3.0, 10.5, 7.0, NET)
    txt(ax, 79.5, ym + 8.0, "Latent $\\mathbf{z}_n$", fs=5.8, fw="bold")
    txt(ax, 79.5, ym - 6.0, "$\\mathcal{N}(\\mu, \\sigma^2)$", fs=5.2)
    arrow(ax, 85.2, ym, 88.0, ym)
    trapezoid(ax, 88.5, 100.5, ym, 9.0, 22.0, NET, NET, "Decoder", fs=5.3, name="dec")
    arrow(ax, 101.0, ym, 104.0, ym)
    Gl.nb_bars(ax, 104.5, ym - 4.5, 11.0, 9.5, NET)
    txt(ax, 110.0, ym + 8.6, "Negative binomial", fs=5.8, fw="bold")
    txt(ax, 110.0, ym - 7.6, "$\\mu_{ng} = l_n \\rho_{ng}$, $\\theta_{tg}$", fs=5.0)
    tp = box(ax, 98.0, 60.0, 117.0, 64.6, "white", GREY, r=0.8, z=4, name="tech")
    txt(ax, 107.5, 62.3, "technology (sc / sn)", tp, fs=5.0)
    arrow(ax, 99.0, 64.6, 99.0, 75.0, col=GREY)
    cp = box(ax, 22.0, 59.5, 47.5, 68.0, "white", RED, r=1.0, z=4, name="cond")
    txt(ax, 34.75, 65.5, "Condition classifier", cp, fs=5.6, fw="bold")
    txt(ax, 34.75, 61.9, "$\\mathbf{W}_c\\mathbf{p}_n$: DKD vs HKD", cp, fs=5.2)
    arrow(ax, 39.6, ym - 9.6, 39.6, 68.0, col=RED)
    dp = box(ax, 66.0, 59.5, 93.0, 68.0, "white", GREY, r=1.0, z=4, name="disc")
    txt(ax, 79.5, 65.5, "Technology discriminator", dp, fs=5.6, fw="bold")
    txt(ax, 79.5, 61.9, "gradient reversal $-\\lambda\\nabla$", dp, fs=5.2)
    arrow(ax, 79.5, ym - 7.6, 79.5, 68.0, col=GREY)
    lb = box(ax, 121.5, 60.0, 172.0, 104.5, "white", NET, r=1.2, z=3, name="loss")
    txt(ax, 146.75, 100.8, "Training objective", lb, fs=6.2, fw="bold")
    lines = [("$\\mathcal{L}_{\\mathrm{rec}}$", "reconstruction", NET),
             ("$+\\,\\beta\\,\\mathcal{L}_{\\mathrm{KL}}$", "latent prior", NET),
             ("$+\\,\\lambda_{\\mathrm{bal}}\\mathcal{L}_{\\mathrm{bal}}$", "use of all experts", NET),
             ("$+\\,\\lambda_{\\mathrm{cond}}\\mathcal{L}_{\\mathrm{cond}}$", "diagnosis through routing", RED),
             ("$+\\,\\lambda_{\\mathrm{adv}}\\mathcal{L}_{\\mathrm{adv}}$", "technology removed", "#666666")]
    txt(ax, 124.0, 94.5, "$\\mathcal{L}\;=$", lb, fs=6.4, ha="left")
    for i, (eq, d, col) in enumerate(lines):
        yy = 94.5 - i * 7.0
        txt(ax, 132.0, yy, eq, lb, fs=6.4, ha="left", col=col)
        txt(ax, 148.0, yy, d, lb, fs=5.2, ha="left", col="#555555")


def panel_c(ax):
    panel_frame(ax, 4, 1, 92, 53, "#F5F1FA", "Contrastive alignment of paired measurements", "align")
    ec = "#7B68A8"
    rows = [(36.0, "he_patch", "H&E patch", "Image\nencoder", "visium", "spot expression", "Expression\nencoder",
             "same spot = positive pair"),
            (13.5, "chromatin", "ATAC peaks", "Chromatin\nencoder", "nucleus", "RNA, same nucleus", "RNA\nencoder",
             "same nucleus = positive pair")]
    for y, ia, la, ea, ib, lb_, eb, cap in rows:
        icon(ax, ia, 12.0, y + 1.0, h=10.0)
        txt(ax, 12.0, y - 6.8, la, fs=5.2)
        arrow(ax, 17.8, y, 20.6, y)
        trapezoid(ax, 21.0, 33.0, y, 14.0, 7.0, ec, ec, ea, fs=4.8, name=f"ea{y}")
        arrow(ax, 33.4, y, 41.0, y, col=ec, lw=1.0)
        Gl.sim_matrix(ax, 41.6, y - 4.4, 8.8, 5, ec)
        txt(ax, 46.0, y + 6.2, "InfoNCE", fs=5.6, fw="bold", col=ec)
        arrow(ax, 59.0, y, 51.0, y, col=ec, lw=1.0)
        trapezoid(ax, 71.0, 59.4, y, 14.0, 7.0, ec, ec, eb, fs=4.8, name=f"eb{y}")
        arrow(ax, 77.0, y, 71.4, y)
        icon(ax, ib, 83.0, y + 1.0, h=9.0 if ib == "nucleus" else 6.5)
        txt(ax, 83.0, y - 6.8, lb_, fs=5.2)
        txt(ax, 46.0, y - 8.6, cap, fs=5.4, col=ec, style="italic")


def panel_d(ax):
    bx = panel_frame(ax, 96, 1, 174, 53, "#FAF6EC", "Participant readout = sum of expert contributions", "read")
    txt(ax, 135.0, 44.6, "$s_i = \\sum_k w_k\\,\\bar{p}_{ik}$   (classifier weight × mean routing)", bx, fs=5.9)
    Cm = S.read("expr_heldout_experts/contributions.tsv", dtype={"participant": str})
    m = Cm.groupby("group")[["M1", "M3", "other"]].mean().loc[["DM_noCKD", "DKD"]]
    S.src_out(m.reset_index(), "Fig1_d_contributions")
    for j, (grp, lab) in enumerate((("DM_noCKD", "Diabetes, no CKD"), ("DKD", "DKD"))):
        yy = 36.0 - j * 6.0
        Gl.stacked(ax, 130.0, yy, 3.6, [m.loc[grp, "M1"], m.loc[grp, "other"], m.loc[grp, "M3"]],
                   [BLUE, "#BDBDBD", RED], 7.0)
        txt(ax, 99.5, yy + 1.8, lab, bx, fs=5.6, ha="left")
    ax.plot([130.0, 130.0], [28.6, 41.0], color=INK, lw=0.5, zorder=6)
    for i, (col, lab) in enumerate(((BLUE, "HKD-weighted"), ("#BDBDBD", "other experts"), (RED, "DKD-weighted"))):
        xx = 112.0 + i * 20.0
        ax.add_patch(Rectangle((xx, 24.2), 2.4, 2.0, fc=col, ec="none", zorder=5))
        txt(ax, xx + 3.2, 25.2, lab, bx, fs=5.0, ha="left")
    tiles = [(None, "expert exchange in\nunseen participants"), ("slides_stack", "independent\ncohorts"),
             ("urine_cup", "albuminuria and\nproteinuria"), ("dna", "chromatin motifs\nand histology")]
    w = 17.6
    for i, (ic, lab) in enumerate(tiles):
        xa = 98.5 + i * (w + 1.6)
        p = box(ax, xa, 3.0, xa + w, 20.5, "white", "#C9B87A", r=1.0, z=3, name=f"d{i}")
        if ic == "dna":
            icon(ax, "dna", xa + w / 2 - 3.0, 15.2, h=7.0); icon(ax, "microscope", xa + w / 2 + 3.4, 15.2, h=6.4)
        elif ic:
            icon(ax, ic, xa + w / 2, 15.2, h=7.0 if ic != "slides_stack" else 6.0)
        else:
            arrow(ax, xa + 4.0, 16.8, xa + w - 4.0, 16.8, col=BLUE, lw=1.1)
            arrow(ax, xa + w - 4.0, 13.6, xa + 4.0, 13.6, col=RED, lw=1.1)
        txt(ax, xa + w / 2, 7.4, lab, p, fs=4.9)


def main():
    S.setup()
    n = counts()
    K = int(n[(n.cohort == "model") & (n.item == "K")].n.iloc[0])
    sc = 171.0 / W
    fig = plt.figure(figsize=(W * sc / 25.4, H * sc / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(0, H); ax.set_aspect("equal"); ax.axis("off")
    panel_a(ax, n); panel_b(ax, K); panel_c(ax); panel_d(ax)
    for x, y, l in [(0.3, 163.6, "a"), (0.3, 110.6, "b"), (0.3, 53.6, "c"), (92.6, 53.6, "d")]:
        ax.text(x, y, l, fontsize=10, fontweight="bold", ha="left", va="top")
    top = [b for b in BOXES if b[0] in ("kpmp", "ext", "model", "align", "read")]
    inner = [b for b in BOXES if b[0] not in ("kpmp", "ext", "model", "align", "read")]
    issues = S.check_inside(fig, PAIRS) + S.check_boxes_disjoint(fig, top) + S.check_boxes_disjoint(fig, inner)
    S.save(fig, "Fig1_framework", issues)


if __name__ == "__main__":
    main()
