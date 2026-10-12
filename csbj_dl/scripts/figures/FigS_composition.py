"""Supplementary figure — expert composition at a given total readout.

Held-out participants of the four disease groups: total readout against the contribution of the DKD-weighted (a) and
HKD-weighted (b) experts, with least-squares lines per group (results/expr_diabetes_groups/participant_table.tsv;
models in composition_given_readout.tsv)."""
from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
import _style_dl as S

GROUPS = ["HKD", "HKD_withDM", "DM_noCKD", "DKD"]
LAB = {"HKD": "HKD", "HKD_withDM": "HKD with diabetes", "DM_noCKD": "Diabetes, no CKD", "DKD": "DKD"}


def panel(ax, P, col, ylab):
    for g in GROUPS:
        d = P[P.group == g]
        ax.scatter(d.moe, d[col], s=9, color=S.GROUP_COL[g], lw=0, alpha=0.85, label=f"{LAB[g]} (n = {len(d)})")
        b = np.polyfit(d.moe, d[col], 1); xs = np.linspace(d.moe.min(), d.moe.max(), 20)
        ax.plot(xs, np.polyval(b, xs), color=S.GROUP_COL[g], lw=1.0)
    ax.set_xlabel("Total readout (log-odds)"); ax.set_ylabel(ylab)


def main():
    S.setup()
    P = S.read("expr_diabetes_groups/participant_table.tsv", dtype={"participant": str})
    P = P[P.group.isin(GROUPS)].dropna(subset=["moe", "M1", "M3"])
    S.src_out(P[["participant", "group", "moe", "M1", "M3", "injured_fraction"]], "FigS_composition_ab")
    fig = plt.figure(figsize=(S.MAX_W - 0.8, 3.0))
    g = fig.add_gridspec(1, 2, wspace=0.38, left=0.09, right=0.98, top=0.92, bottom=0.24)
    axa, axb = fig.add_subplot(g[0, 0]), fig.add_subplot(g[0, 1])
    panel(axa, P, "M3", "DKD-weighted expert\ncontribution (log-odds)"); panel(axb, P, "M1", "HKD-weighted expert\ncontribution (log-odds)")
    h, l = axa.get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", bbox_to_anchor=(0.53, 0.01), ncol=4, fontsize=5.8, frameon=False, handletextpad=0.2)
    S.auto_letters(fig, [(axa, "a"), (axb, "b")])
    S.save(fig, "FigS_composition")


if __name__ == "__main__":
    main()
