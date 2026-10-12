"""Fig. 1 — study design and model, drawn entirely in code (vector shapes and editable text; no generated artwork).

(a) participant groups, data and roles (KPMP training with participants split into five parts; external cohorts used
for inference only); (b) the mixture-of-experts expression model with its loss terms; (c) the two contrastive
alignments; (d) the participant readout as a sum of expert contributions (group means of the held-out contributions,
results/expr_heldout_experts/contributions.tsv) and the analyses built on it. Glyphs: fig1_glyphs.py.
Participant counts come from csbj_dl/config/splits.tsv and the external score tables; the number of experts from
results/expr_cv/selection.json. Coordinates are millimetres on a 175 mm wide canvas."""
from __future__ import annotations
import json
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import matplotlib.pyplot as plt
import _style_dl as S
import fig1_glyphs as Gl

W, H, Y0 = 175.0, 204.0, 14.0
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


GROUPS = [("DKD", "DKD", "#C44E52", True, True), ("HKD_withDM", "HKD with\ndiabetes", "#8172B3", True, True),
          ("HKD", "HKD", "#4C72B0", False, True), ("DM_noCKD", "Diabetes,\nno CKD", "#CCB974", True, False),
          ("Reference", "Reference\ntissue", "#8C8C8C", False, False)]


def panel_a(ax, n):
    g = lambda c, i: int(n[(n.cohort == c) & (n.item == i)].n.iloc[0])
    y0, y1 = 150.0, 202.0
    box(ax, 1, y0, 118, y1, C["kpmp"], C["kpmp_ec"], name="kpmp")
    header(ax, 1, y1, 118, f"KPMP: training ({g('KPMP', 'all')} participants in the expression model)", C["kpmp_ec"])
    # participant groups: kidney coloured by group, glucose drop = diabetes, gauge = hypertension
    for i, (k, lab, col, dm, htn) in enumerate(GROUPS):
        cx = 12 + i * 22.5
        Gl.kidney(ax, cx, 191.0, 3.6, col)
        if dm:
            Gl.drop(ax, cx + 5.4, 192.6, 1.1, "#B5891C")
        if htn:
            Gl.gauge(ax, cx + 5.6, 188.0, 1.6, "#4C72B0")
        txt(ax, cx + 13.0 if False else cx + 1.2, 183.6, f"{lab} (n = {g('group', k)})" if "\n" not in lab else
            lab.replace("\n", " ") + f"\n(n = {g('group', k)})", fs=5.4, va="top")
    mods = [("Single-cell RNA", "sc"), ("Single-nucleus RNA", "sn"),
            ("Multiome\n(RNA + ATAC)", "multiome"), ("Visium + H&E", "visium")]
    w, gap, x = 26.5, 2.4, 3.6
    for i, (lab, k) in enumerate(mods):
        xa = x + i * (w + gap)
        p = box(ax, xa, 159.0, xa + w, 176.5, "white", C["kpmp_ec"], r=1.0, z=3, name=f"mod{i}")
        glyph(ax, k, xa + w / 2, 171.3)
        txt(ax, xa + w / 2, 163.3, f"{lab}\nn = {g('KPMP', k)}", p, fs=5.8)
    cols = ["#4C72B0", "#4C72B0", "#4C72B0", "#4C72B0", "#DD8452"]
    txt(ax, 4.0, 154.6, "Participants in five parts (all data of a participant in one part):", fs=5.6, ha="left")
    for i in range(g("KPMP", "parts")):
        xa = 75.0 + i * 6.4
        ax.add_patch(Rectangle((xa, 152.8), 5.4, 3.6, fc=cols[i], ec="white", lw=0.4, alpha=0.75, zorder=4))
    txt(ax, 107.5, 154.6, "held out", fs=5.4, ha="left", col="#B25F2A")
    ex = box(ax, 125, y0, 174, y1, C["ext"], C["ext_ec"], name="ext")
    header(ax, 125, y1, 174, "External cohorts: inference only", C["ext_ec"])
    items = [("GSE211785", f"sc/snRNA, {g('GSE211785', 'donors')} donors; Visium, {g('GSE211785', 'visium_participants')}"),
             ("GSE195460", f"snRNA; {g('GSE195460', 'donors')} donors"),
             ("GSE104954 (ERCB)", f"tissue arrays; {g('GSE104954', 'samples')} samples"),
             ("Zenodo 19868428", f"Xenium; {g('Zenodo19868428', 'xenium_donors_scored')} donors"),
             ("GSE339455", f"Visium + H&E; {g('GSE339455', 'tissue_regions')} regions")]
    for i, (a, b) in enumerate(items):
        ya = 191.6 - i * 8.6
        p = box(ax, 127.0, ya - 3.7, 172.0, ya + 3.7, "white", C["ext_ec"], r=1.0, z=3, name=f"ext{i}")
        txt(ax, 128.8, ya + 1.4, a, p, fs=5.8, fw="bold", ha="left")
        txt(ax, 128.8, ya - 1.7, b, p, fs=5.5, ha="left")
    arrow(ax, 118.6, 176, 124.4, 176, lw=1.0)


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


EXPERT_COL = {0: "#4C72B0", 1: "#55A868", 2: "#C44E52"}     # HKD-weighted, healthy-cell, DKD-weighted


def panel_b(ax, K):
    y0, y1 = 84.0, 145.0
    bx = box(ax, 1, y0, 174, y1, C["model"], C["model_ec"], name="model")
    header(ax, 1, y1, 140, "Mixture-of-experts generative model of proximal tubule cells", C["model_ec"])
    ec = C["model_ec"]; ym = 121.0
    # input
    Gl.cells(ax, 9.0, 136.0, 7, 3.5, False, 5)
    Gl.expr_vector(ax, 7.0, 108.0, 4.0, 22.0, 14, 2)
    txt(ax, 9.0, 104.5, "counts $\\mathbf{x}_n$\n(3,000 genes)", fs=5.4, va="top")
    # encoder
    Gl.layers(ax, 16.0, ym - 10, 30.0, ym + 10, [6, 5, 4], ec)
    txt(ax, 23.0, ym + 12.6, "Shared encoder", fs=5.8, fw="bold", col="#3E6E46")
    txt(ax, 23.0, ym - 12.6, "$\\mathbf{h}_n$", fs=6)
    arrow(ax, 11.6, ym, 15.0, ym)
    # gate
    probs = [0.30, 0.05, 0.42, 0.08, 0.09, 0.06][:K]
    gcols = [EXPERT_COL.get(i, "#9E9E9E") for i in range(K)]
    Gl.softmax_bars(ax, 36.0, ym - 4.0, 13.0, 10.0, probs, gcols, top={0, 2})
    txt(ax, 42.5, ym + 9.0, "Gating network", fs=5.8, fw="bold", col="#3E6E46")
    txt(ax, 42.5, ym - 7.6, "$\\mathbf{p}_n$ = softmax, top-2", fs=5.4)
    arrow(ax, 31.0, ym, 35.2, ym)
    # experts
    eh, eg = 3.6, 0.8
    tot = K * eh + (K - 1) * eg; top_ = ym + tot / 2
    box(ax, 55.0, ym - tot / 2 - 1.4, 71.0, ym + tot / 2 + 1.4, "white", ec, r=1.0, z=3, name="experts")
    labs = {0: "HKD-weighted", 1: "healthy cells", 2: "DKD-weighted"}
    for i in range(K):
        ya = top_ - i * (eh + eg) - eh
        col = EXPERT_COL.get(i, "#BDBDBD")
        p = box(ax, 56.2, ya, 69.8, ya + eh, "white", col, lw=0.8 if i in EXPERT_COL else 0.5, r=0.6, z=4, track=False)
        Gl.layers(ax, 57.4, ya + 0.4, 61.0, ya + eh - 0.4, [2, 2], col, r=0.35, links=True)
        txt(ax, 65.6, ya + eh / 2, f"$e_{{{i + 1}}}$", p, fs=5.6, col=col if i in EXPERT_COL else "#757575")
    txt(ax, 63.0, ym + tot / 2 + 3.4, "Experts", fs=5.8, fw="bold", col="#3E6E46")
    for i, lab in labs.items():
        txt(ax, 63.0, ym - tot / 2 - 3.6 - 2.9 * i, lab, fs=5.0, col=EXPERT_COL[i])
    arrow(ax, 50.0, ym, 54.6, ym)
    # mixture and latent
    ax.add_patch(Circle((75.5, ym), 2.2, fc="white", ec=INK, lw=0.6, zorder=5))
    txt(ax, 75.5, ym, "Σ", fs=7)
    txt(ax, 75.5, ym - 4.6, "$\\sum_k g_{nk} e_k$", fs=5.2)
    arrow(ax, 71.2, ym, 73.2, ym)
    Gl.gaussian(ax, 87.0, ym - 3.0, 11.0, 7.0, "#3E6E46")
    txt(ax, 87.0, ym + 8.0, "Latent $\\mathbf{z}_n$", fs=5.8, fw="bold", col="#3E6E46")
    txt(ax, 87.0, ym - 6.4, "$\\mathcal{N}(\\mu, \\sigma^2)$", fs=5.4)
    arrow(ax, 77.8, ym, 81.2, ym)
    # decoder with technology
    Gl.layers(ax, 97.0, ym - 10, 111.0, ym + 10, [4, 5, 6], ec)
    txt(ax, 104.0, ym + 12.6, "Decoder", fs=5.8, fw="bold", col="#3E6E46")
    arrow(ax, 92.8, ym, 96.0, ym)
    tp = box(ax, 106.0, 100.5, 127.0, 105.5, "white", "#8C8C8C", r=0.8, z=4, name="tech")
    txt(ax, 116.5, 103.0, "technology (sc / sn)", tp, fs=5.2)
    arrow(ax, 111.0, 105.5, 111.0, ym - 9.0, col="#8C8C8C")
    Gl.nb_bars(ax, 116.5, ym - 4.0, 12.0, 9.0, "#3E6E46")
    txt(ax, 122.5, ym + 8.0, "Negative binomial", fs=5.8, fw="bold", col="#3E6E46")
    txt(ax, 122.5, ym - 7.0, "$\\mu_{ng} = l_n \\rho_{ng}$, $\\theta_{tg}$", fs=5.2)
    arrow(ax, 112.0, ym, 115.6, ym)
    # heads
    cp = box(ax, 28.0, 87.0, 60.0, 96.5, "white", C["red"], r=1.0, z=4, name="cond")
    txt(ax, 44.0, 93.6, "Condition classifier on routing", cp, fs=5.6, fw="bold", col=C["red"])
    txt(ax, 44.0, 89.6, "$\\mathbf{W}_c\\mathbf{p}_n$: DKD vs HKD", cp, fs=5.4)
    arrow(ax, 42.5, ym - 9.4, 42.5, 96.5, col=C["red"])
    dp = box(ax, 70.0, 87.0, 104.0, 96.5, "white", "#8C8C8C", r=1.0, z=4, name="disc")
    txt(ax, 87.0, 93.6, "Technology discriminator", dp, fs=5.6, fw="bold", col="#666666")
    txt(ax, 87.0, 89.6, "gradient reversal $-\\lambda\\nabla$", dp, fs=5.4)
    arrow(ax, 87.0, ym - 7.6, 87.0, 96.5, col="#8C8C8C")
    # loss
    lb = box(ax, 133.5, 87.0, 172.5, 142.0, "white", ec, r=1.0, z=3, name="loss")
    txt(ax, 153.0, 138.2, "Training objective", lb, fs=6.2, fw="bold", col="#3E6E46")
    lines = [("$\\mathcal{L}_{\\mathrm{rec}}$", "negative binomial reconstruction", "#3E6E46"),
             ("$+\\,\\beta\\,\\mathcal{L}_{\\mathrm{KL}}$", "latent prior", "#3E6E46"),
             ("$+\\,\\lambda_{\\mathrm{bal}}\\mathcal{L}_{\\mathrm{bal}}$", "use of all experts", "#3E6E46"),
             ("$+\\,\\lambda_{\\mathrm{cond}}\\mathcal{L}_{\\mathrm{cond}}$", "diagnosis through routing", C["red"]),
             ("$+\\,\\lambda_{\\mathrm{adv}}\\mathcal{L}_{\\mathrm{adv}}$", "technology removed", "#666666")]
    txt(ax, 136.0, 131.5, "$\\mathcal{L}\\;=$", lb, fs=6.4, ha="left")
    for i, (eq, d, col) in enumerate(lines):
        yy = 131.5 - i * 8.6
        txt(ax, 143.0, yy, eq, lb, fs=6.4, ha="left", col=col)
        txt(ax, 143.0, yy - 3.6, d, lb, fs=5.0, ha="left", col="#555555")


INK = S.INK


def panel_c(ax):
    y0, y1 = 15.0, 78.0
    bx = box(ax, 1, y0, 93, y1, C["align"], C["align_ec"], name="align")
    header(ax, 1, y1, 93, "Contrastive alignment of paired measurements", C["align_ec"])
    ec = C["align_ec"]
    rows = [(57.0, "H&E patch at\nthe spot", "Image encoder\n+ projection", "Spot expression\n+ projection",
             "matched spot = positive pair"),
            (31.0, "ATAC peaks of\nthe nucleus", "Chromatin encoder\n+ projection", "RNA of the same\nnucleus + projection",
             "same nucleus = positive pair")]
    for y, a, b, c, lab in rows:
        node(ax, 3.5, y - 6, 22, y + 6, a, ec, fs=5.8, name=f"c1_{y}")
        node(ax, 25, y - 6, 47, y + 6, b, ec, fs=5.8, name=f"c2_{y}")
        node(ax, 66, y - 6, 90.5, y + 6, c, ec, fs=5.8, name=f"c3_{y}")
        arrow(ax, 22, y, 25, y)
        arrow(ax, 47.4, y, 51.0, y, col=ec, style="<|-", lw=1.0)
        arrow(ax, 61.6, y, 65.6, y, col=ec, style="-|>", lw=1.0)
        Gl.sim_matrix(ax, 52.2, y - 4.6, 8.2, 5, ec)
        txt(ax, 56.3, y + 6.2, "InfoNCE", fs=5.6, fw="bold", col=ec)
        txt(ax, 55.5, y - 8.6, lab, bx, fs=5.6, col=ec, style="italic")


def panel_d(ax):
    y0, y1 = 15.0, 78.0
    bx = box(ax, 97, y0, 174, y1, C["read"], C["read_ec"], name="read")
    header(ax, 97, y1, 174, "Participant readout and its expert contributions", "#8A7820")
    txt(ax, 135.5, 68.0, "$s_i = \\sum_k w_k\\,\\bar{p}_{ik}$  (classifier weight × mean routing)", bx, fs=6.0)
    Cm = S.read("expr_heldout_experts/contributions.tsv", dtype={"participant": str})
    m = Cm.groupby("group")[["M1", "M3", "other"]].mean().loc[["DM_noCKD", "DKD"]]
    S.src_out(m.reset_index(), "Fig1_d_contributions")
    sc = 7.0
    for j, (grp, lab) in enumerate((("DM_noCKD", "Diabetes, no CKD"), ("DKD", "DKD"))):
        yy = 57.0 - j * 7.5
        Gl.stacked(ax, 130.0, yy, 4.0, [m.loc[grp, "M1"], m.loc[grp, "other"], m.loc[grp, "M3"]],
                   ["#4C72B0", "#BDBDBD", "#C44E52"], sc)
        txt(ax, 99.5, yy + 2.0, lab, bx, fs=5.6, ha="left")
    ax.plot([130.0, 130.0], [48.5, 62.5], color=INK, lw=0.5, zorder=6)
    for i, (col, lab) in enumerate((("#4C72B0", "HKD-weighted"), ("#BDBDBD", "other"), ("#C44E52", "DKD-weighted"))):
        xx = 118.0 + i * 18.0
        ax.add_patch(Rectangle((xx, 44.2), 2.6, 2.2, fc=col, ec="none", zorder=5))
        txt(ax, xx + 3.4, 45.3, lab, bx, fs=5.2, ha="left")
    items = ["Exchange of expert\nweights in unseen\nparticipants", "Independent\ncohorts",
             "Albuminuria and\nproteinuria", "Chromatin motifs\nand histology"]
    w, h = 17.2, 13.0
    for i, s_ in enumerate(items):
        xa = 99.0 + i * (w + 1.9)
        node(ax, xa, 19.0, xa + w, 19.0 + h, s_, C["read_ec"], fs=5.2, name=f"d{i}")
    txt(ax, 135.5, 38.0, "group means of held-out contributions (log-odds)", bx, fs=5.0, col="#555555")


def main():
    S.setup()
    n = counts()
    K = int(n[(n.cohort == "model") & (n.item == "K")].n.iloc[0])
    sc = 171.0 / W   # printed width 171 mm (tight bbox adds a small pad)
    fig = plt.figure(figsize=(W * sc / 25.4, (H - Y0) * sc / 25.4))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, W); ax.set_ylim(Y0, H); ax.set_aspect("equal")
    ax.axis("off")
    panel_a(ax, n); panel_b(ax, K); panel_c(ax); panel_d(ax)
    for x, y, l in [(0.5, 202.6, "a"), (0.5, 145.6, "b"), (0.5, 78.6, "c"), (96.5, 78.6, "d")]:
        t = ax.text(x, y, l, fontsize=10, fontweight="bold", ha="left", va="bottom")
    top = [b for b in BOXES if b[0] in ("kpmp", "ext", "model", "align", "read")]
    inner = [b for b in BOXES if b[0] not in ("kpmp", "ext", "model", "align", "read")]
    issues = S.check_inside(fig, PAIRS) + S.check_boxes_disjoint(fig, top) + S.check_boxes_disjoint(fig, inner)
    S.save(fig, "Fig1_framework", issues)


if __name__ == "__main__":
    main()
