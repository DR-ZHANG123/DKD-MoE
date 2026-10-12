"""00_extra_partitions: partitions for KPMP Visium participants absent from csbj_dl/config/splits.tsv.

These participants enter alignment training (and retrieval evaluation as held-out of their own part) only.
Deterministic assignment: within each group, participants sorted by ID, permuted with seed `seed`
(align.yaml) and dealt round-robin to the 5 parts. splits.tsv itself is not edited.
Output csbj_dl/config/align_extra_partitions.tsv (participant, group, fold, source)
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402


def main():
    cfg = C.load_cfg()
    t = pd.read_csv(C.p(cfg["paths"]["new_sections"]), sep="\t", dtype=str)
    sp = pd.read_csv(C.p(cfg["paths"]["splits"]), sep="\t", dtype=str)
    x = t[~t.participant.isin(sp.participant)].drop_duplicates("participant")[["participant", "group"]]
    rng = np.random.default_rng(cfg["seed"]); rows = []
    for g, d in x.sort_values(["group", "participant"]).groupby("group"):
        p = d.participant.values[rng.permutation(len(d))]
        rows += [{"participant": q, "group": g, "fold": i % 5, "source": "align_extra"} for i, q in enumerate(p)]
    T = pd.DataFrame(rows); T.to_csv(C.CSBJ / "config/align_extra_partitions.tsv", sep="\t", index=False)
    print(pd.crosstab(T.group, T.fold))


if __name__ == "__main__":
    main()
