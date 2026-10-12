"""Supplementary figure — the readout-derived signature: held-out KPMP scores, its genes and the independent cohorts.

(a) cross-fitted signature score of held-out KPMP participants, DKD versus HKD by technology; (b) the 10 highest-ranked
up- and down-regulated genes; (c) readout and signature in the independent cohorts. Panels from panels_signature.py
(sources listed there)."""
from __future__ import annotations
import matplotlib.pyplot as plt
import _style as S
import panels_signature as PG

S.SRC_RENAME.update({"sig_box": "FigS_signature_a", "sig_genes": "FigS_signature_b", "sig_forest": "FigS_signature_c"})


def main():
    S.setup()
    fig = plt.figure(figsize=(S.MAX_W - 0.12, 5.0))
    axa = fig.add_subplot(fig.add_gridspec(1, 1, left=0.075, right=0.27, top=0.95, bottom=0.6)[0, 0])
    axb = fig.add_subplot(fig.add_gridspec(1, 1, left=0.33, right=0.99, top=0.93, bottom=0.66)[0, 0])
    axc = fig.add_subplot(fig.add_gridspec(1, 1, left=0.5, right=0.77, top=0.48, bottom=0.08)[0, 0])
    PG.box(axa); PG.genes(axb, n_show=10); PG.forest(axc)
    S.auto_letters(fig, [(axa, "a"), (axb, "b"), (axc, "c")])
    S.save(fig, "FigS_signature")


if __name__ == "__main__":
    main()
