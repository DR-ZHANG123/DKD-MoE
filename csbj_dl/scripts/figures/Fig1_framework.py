"""Fig. 1 — study design and model, drawn entirely in code (vector shapes and editable text; no generated artwork).

(a) data and roles (KPMP training with participants split into five parts; external cohorts used for inference only);
(b) the mixture-of-experts expression model; (c) the two contrastive alignments; (d) read-outs.
Participant counts come from csbj_dl/config/splits.tsv and the external score tables; the number of experts from
results/expr_cv/selection.json. Coordinates are millimetres on a 175 mm wide canvas."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.pyplot as plt
import _style_dl as S

W, H, Y0 = 175.0, 178.0, 14.0
C = {"kpmp": "#E8EEF7", "kpmp_ec": "#4C72B0", "ext": "#FBEDE3", "ext_ec": "#DD8452", "model": "#EEF5EE",
     "model_ec": "#55A868", "align": "#F3EEF8", "align_ec": "#8172B3", "read": "#F7F3E6", "read_ec": "#B59F3B",
     "node": "white", "ink": S.INK, "red": "#C44E52", "blue": "#4C72B0"}
PAIRS, BOXES = [], []


def box(ax, x0, y0, x1, y1, fc, ec, lw=0.7, r=1.5, z=2, name=None, track=True):
    p = FancyBboxPatch((x0, y0), x1 - x0, y1 - y0, boxstyle=f"round,pad=0,rounding_size={r}", fc=fc, ec=ec, lw=lw,
                       zorder=z)
    ax.add_patch(p)
    if track and name:
        BOXES.append((name, p))
    return p


def txt(ax, x, y, s, patch=None, fs=6.5, fw="normal", col=None, ha="center", va="center", z=6, style="normal"):
    t = ax.text(x, y, s, fontsize=fs, fontweight=fw, color=col or C["ink"], ha=ha, va=va, zorder=z,
                linespacing=1.15, fontstyle=style)
    if patch is not None:
        PAIRS.append((t, patch))
    return t


def node(ax, x0, y0, x1, y1, s, ec, fs=6.5, fw="normal", fc="white", name=None):
    p = box(ax, x0, y0, x1, y1, fc, ec, lw=0.7, r=1.0, z=4, name=name)
    txt(ax, (x0 + x1) / 2, (y0 + y1) / 2, s, p, fs=fs, fw=fw)
    return p


def arrow(ax, x0, y0, x1, y1, col=None, lw=0.8, ls="-", style="-|>", rad=0.0, ms=6):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=ms, color=col or C["ink"], lw=lw,
                        linestyle=ls, zorder=5, shrinkA=0, shrinkB=0, connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(a)
    return a


def header(ax, x0, y1, x1, s, ec):
    return txt(ax, (x0 + x1) / 2, y1 - 3.2, s, None, fs=7, fw="bold", col=ec)


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


def panel_a(ax, n):
    g = lambda c, i: int(n[(n.cohort == c) & (n.item == i)].n.iloc[0])
    y0, y1 = 124.0, 176.0
    kp = box(ax, 1, y0, 101, y1, C["kpmp"], C["kpmp_ec"], name="kpmp")
    header(ax, 1, y1, 101, f"KPMP: training ({g('KPMP', 'all')} participants in the expression model)", C["kpmp_ec"])
    mods = [("Single-cell RNA", "sc"), ("Single-nucleus RNA", "sn"),
            ("Multiome\n(RNA + ATAC)", "multiome"), ("Visium + H&E", "visium")]
    w, gap, x = 22.6, 1.8, 3.6
    for i, (lab, k) in enumerate(mods):
        xa = x + i * (w + gap)
        p = box(ax, xa, 145.5, xa + w, 168.0, "white", C["kpmp_ec"], r=1.0, z=3, name=f"mod{i}")
        glyph(ax, k, xa + w / 2, 161.4)
        txt(ax, xa + w / 2, 152.2, f"{lab}\nn = {g('KPMP', k)}", p, fs=6)
    # five participant-level parts
    fb = box(ax, 3.6, 126.0, 98.4, 142.5, "white", C["kpmp_ec"], r=1.0, z=3, name="folds")
    txt(ax, 51, 138.4, "Participants split into five parts;", fb, fs=6)
    txt(ax, 51, 134.6, "all data types of one participant stay in one part", fb, fs=6)
    cols = ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]
    for i in range(g("KPMP", "parts")):
        xa = 30.5 + i * 8.4
        ax.add_patch(Rectangle((xa, 127.6), 7.0, 3.6, fc=cols[i], ec="white", lw=0.4, alpha=0.75, zorder=4))
    txt(ax, 75.5, 129.4, "held-out part", fs=5.5, ha="left", col="#B25F2A")
    # external cohorts
    ex = box(ax, 115, y0, 174, y1, C["ext"], C["ext_ec"], name="ext")
    header(ax, 115, y1, 174, "External cohorts: inference only", C["ext_ec"])
    items = [("GSE211785", f"sc/snRNA, {g('GSE211785', 'donors')} donors; Visium + H&E, {g('GSE211785', 'visium_participants')} donors"),
             ("GSE195460", f"snRNA; {g('GSE195460', 'donors')} donors"),
             ("GSE104954 (ERCB)", f"tissue arrays; {g('GSE104954', 'samples')} samples"),
             ("Zenodo 19868428", f"CosMx / Xenium; {g('Zenodo19868428', 'xenium_donors_scored')} Xenium donors"),
             ("GSE339455", f"Visium + H&E; {g('GSE339455', 'tissue_regions')} tissue regions")]
    for i, (a, b) in enumerate(items):
        ya = 165.0 - i * 8.2
        p = box(ax, 117.5, ya - 3.6, 171.5, ya + 3.6, "white", C["ext_ec"], r=1.0, z=3, name=f"ext{i}")
        txt(ax, 119.5, ya + 1.3, a, p, fs=6, fw="bold", ha="left")
        txt(ax, 119.5, ya - 1.7, b, p, fs=5.8, ha="left")
    arrow(ax, 102.2, 150, 113.8, 150, lw=1.0)
    txt(ax, 108, 155.6, "trained\nmodel", fs=5.5)
    txt(ax, 108, 144.4, "applied\nunchanged", fs=5.5)


def glyph(ax, k, cx, cy):
    rng = np.random.default_rng({"sc": 1, "sn": 2, "multiome": 3, "visium": 4}[k])
    if k in ("sc", "sn"):
        for _ in range(7):
            x, y = cx + rng.uniform(-7, 7), cy + rng.uniform(-3.4, 3.4)
            if k == "sc":
                ax.add_patch(Circle((x, y), 1.6, fc="#F6D9D7", ec="#B9605F", lw=0.5, zorder=4))
            ax.add_patch(Circle((x, y), 0.8, fc="#7B78B8", ec="none", zorder=5))
    elif k == "multiome":
        ax.add_patch(Circle((cx - 4.5, cy), 3.0, fc="#7B78B8", ec="none", alpha=0.85, zorder=4))
        for i in range(5):
            ax.add_patch(Rectangle((cx + 0.5 + i * 1.6, cy - 2.6), 1.0, 1.0 + 3.6 * rng.uniform(0.2, 1), fc="#55A868",
                                   ec="none", zorder=4))
        ax.plot([cx + 0.2, cx + 8.6], [cy - 2.7, cy - 2.7], color=C["ink"], lw=0.5, zorder=4)
    else:
        ax.add_patch(Rectangle((cx - 8.5, cy - 4.0), 17, 8.0, fc="#F3DDEB", ec="#B06A9F", lw=0.5, zorder=4))
        for i in range(7):
            for j in range(3):
                if j % 2 and i == 6:
                    continue
                ax.add_patch(Circle((cx - 6.9 + i * 2.2 + (j % 2) * 1.1, cy - 2.4 + j * 2.4), 0.6, fc="#8A5FA0",
                                    ec="none", alpha=0.8, zorder=5))


def panel_b(ax, K):
    y0, y1 = 70.0, 118.0
    bx = box(ax, 1, y0, 174, y1, C["model"], C["model_ec"], name="model")
    header(ax, 1, y1, 174, "Mixture-of-experts expression model (proximal tubule cells)", C["model_ec"])
    ec = C["model_ec"]; ym = 96.0
    node(ax, 4, ym - 6, 24, ym + 6, "Counts\n(3,000 genes)", ec, name="n_counts")
    node(ax, 29, ym - 6, 49, ym + 6, "Shared\nencoder", ec, name="n_enc")
    node(ax, 54, ym - 6, 76, ym + 6, "Gating network\n(top-2 routing)", ec, name="n_gate")
    # experts
    eh, eg = 3.3, 0.9
    tot = K * eh + (K - 1) * eg; ys = ym + tot / 2
    exp_frame = box(ax, 81, ym - tot / 2 - 1.6, 99, ym + tot / 2 + 1.6, "white", ec, r=1.0, z=3, name="experts")
    for i in range(K):
        ya = ys - i * (eh + eg) - eh
        hi = False
        p = box(ax, 82.5, ya, 97.5, ya + eh, "#F6D9D7" if hi else "#E4EFE4", "#C44E52" if hi else ec, lw=0.5, r=0.6,
                z=4, track=False)
        txt(ax, 90, ya + eh / 2, f"Expert {i + 1}", p, fs=5.5)
    node(ax, 104, ym - 6, 120, ym + 6, "Latent z", ec, name="n_z")
    node(ax, 125, ym - 6, 148, ym + 6, "Negative-\nbinomial\ndecoder", ec, name="n_dec")
    node(ax, 153, ym - 6, 172, ym + 6, "Reconstructed\ncounts", ec, name="n_rec")
    for a, b in [(24, 29), (49, 54), (99, 104), (120, 125), (148, 153)]:
        arrow(ax, a, ym, b, ym)
    arrow(ax, 76, ym, 81, ym)          # routing to the whole expert block (top-2 of all experts)
    # technology condition into decoder
    node(ax, 125, 108.0, 148, 113.5, "Technology (sc / sn)", "#8C8C8C", fs=6, name="n_tech")
    arrow(ax, 136.5, 108.0, 136.5, ym + 6)
    # condition head on routing probabilities
    node(ax, 38, 72.5, 79, 82.5, "Condition head on routing\nprobabilities: DKD vs HKD", C["red"], fs=6,
         name="n_cond")
    arrow(ax, 65, ym - 6, 65, 82.5, col=C["red"])
    # technology discriminator with gradient reversal
    node(ax, 101, 72.5, 131, 82.5, "Technology discriminator\nwith gradient reversal", "#8C8C8C", fs=6,
         name="n_disc")
    arrow(ax, 112, ym - 6, 112, 82.5, col="#8C8C8C")
    txt(ax, 150, 79.0, "Loss: reconstruction + KL\n+ load balance + condition\n− technology (reversed)",
        bx, fs=5.6, col="#3E6E46")


def panel_c(ax):
    y0, y1 = 15.0, 64.0
    bx = box(ax, 1, y0, 93, y1, C["align"], C["align_ec"], name="align")
    header(ax, 1, y1, 93, "Contrastive alignment of paired measurements", C["align_ec"])
    ec = C["align_ec"]
    rows = [(48.0, "H&E patch at\nthe spot", "Image encoder\n+ projection", "Spot expression\n+ projection",
             "matched spot = positive pair"),
            (28.0, "ATAC peaks of\nthe nucleus", "Chromatin encoder\n+ projection", "RNA of the same\nnucleus + projection",
             "same nucleus = positive pair")]
    for y, a, b, c, lab in rows:
        node(ax, 3.5, y - 6, 22, y + 6, a, ec, fs=5.8, name=f"c1_{y}")
        node(ax, 25, y - 6, 47, y + 6, b, ec, fs=5.8, name=f"c2_{y}")
        node(ax, 66, y - 6, 90.5, y + 6, c, ec, fs=5.8, name=f"c3_{y}")
        arrow(ax, 22, y, 25, y)
        arrow(ax, 47.4, y, 65.6, y, col=ec, style="<|-|>", lw=1.0)
        txt(ax, 56.5, y + 2.6, "InfoNCE", fs=5.8, fw="bold", col=ec)
        txt(ax, 55.5, y - 8.6, lab, bx, fs=5.6, col=ec, style="italic")


def panel_d(ax):
    y0, y1 = 15.0, 64.0
    bx = box(ax, 97, y0, 174, y1, C["read"], C["read_ec"], name="read")
    header(ax, 97, y1, 174, "Read-outs (participant level)", "#8A7820")
    items = ["DKD-versus-HKD readout and\nits expert contributions", "Integrated-gradient\ngene attributions",
             "Transfer gene signature\nfor external cohorts", "Morphology along the\nimage-side direction",
             "Chromatin accessibility\nand motifs", "Albuminuria and\nproteinuria bands"]
    w, h = 35.0, 11.0
    for i, s in enumerate(items):
        col, row = i % 2, i // 2
        xa = 99.5 + col * (w + 2.0); ya = 45.5 - row * (h + 2.5)
        node(ax, xa, ya, xa + w, ya + h, s, C["read_ec"], fs=5.8, name=f"d{i}")


def main():
    S.setup()
    n = counts()
    K = int(n[(n.cohort == "model") & (n.item == "K")].n.iloc[0])
    sc = 171.0 / W   # printed width 171 mm (tight bbox adds a small pad)
    fig = plt.figure(figsize=(W * sc / 25.4, (H - Y0) * sc / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.set_aspect("equal")
    ax.axis("off")
    panel_a(ax, n); panel_b(ax, K); panel_c(ax); panel_d(ax)
    for x, y, l in [(0.5, 176.6, "a"), (0.5, 118.6, "b"), (0.5, 64.6, "c"), (96.5, 64.6, "d")]:
        t = ax.text(x, y, l, fontsize=10, fontweight="bold", ha="left", va="bottom")
    top = [b for b in BOXES if b[0] in ("kpmp", "ext", "model", "align", "read")]
    inner = [b for b in BOXES if b[0] not in ("kpmp", "ext", "model", "align", "read")]
    issues = S.check_inside(fig, PAIRS) + S.check_boxes_disjoint(fig, top) + S.check_boxes_disjoint(fig, inner)
    S.save(fig, "Fig1_framework", issues)


if __name__ == "__main__":
    main()
