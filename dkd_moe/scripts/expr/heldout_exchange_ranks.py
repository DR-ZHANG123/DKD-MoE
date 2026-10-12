"""Per-model specificity of the held-out exchanges (reads results/expr_heldout_experts tables; no model is run).

For every partition model (seed x part): participant-mean signature change of each ordered expert exchange in
held-out HKD injured cells; the share of all exchanges of that model with a change at least as large as that of the
HKD-weighted -> DKD-weighted exchange (M1 -> M3). For the supervised model without experts: the share of 50 random
latent directions of equal length with a change at least as large as that of the condition-head direction.
Outputs: results/expr_heldout_experts/{exchange_ranks.tsv, exchange_main_participants.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402


def main():
    cfg = load_cfg("expr"); d = result_dir("expr_heldout_experts")
    E = pd.read_csv(d / "exchange_pairs.tsv", sep="\t", dtype={"participant": str})
    L = pd.read_csv(d / "latent_shift.tsv", sep="\t", dtype={"participant": str})
    rows = []
    for (s, f), x in E[E.group == "HKD"].groupby(["seed", "fold"]):
        pm = x.groupby(["src", "dst"]).signature_change.mean(); v = pm[("M1", "M3")]
        rows.append({"model": "mixture_of_experts", "seed": s, "fold": f, "main_change": v,
                     "share_at_least_as_large": float((pm >= v).mean()), "n_alternatives": len(pm)})
    for (s, f), x in L.groupby(["seed", "fold"]):
        pm = x.groupby("direction").signature_change.mean(); v = pm["head"]; rnd = pm.drop("head")
        rows.append({"model": "no_experts", "seed": s, "fold": f, "main_change": v,
                     "share_at_least_as_large": float((rnd >= v).mean()), "n_alternatives": len(rnd)})
    R = pd.DataFrame(rows); R.to_csv(d / "exchange_ranks.tsv", sep="\t", index=False)
    H = E[E.group == "HKD"]
    keep = {("M1", "M3"): "HKD-weighted to DKD-weighted", ("M1", "M2"): "HKD-weighted to healthy-cell"}
    P = []
    for (a, b), lab in keep.items():
        P.append(H[(H.src == a) & (H.dst == b)].groupby("participant").signature_change.mean().rename(lab))
    P.append(H[(H.src == "M1") & H.dst.str.startswith("o")].groupby("participant").signature_change.mean()
             .rename("HKD-weighted to non-reproducible"))
    P = pd.concat(P, axis=1); P.to_csv(d / "exchange_main_participants.tsv", sep="\t")
    print(R.groupby("model").share_at_least_as_large.describe().round(3)); print(P.describe().round(4))
    write_provenance("expr_heldout_experts", [d / "exchange_pairs.tsv", d / "latent_shift.tsv"],
                     [d / "exchange_ranks.tsv", d / "exchange_main_participants.tsv"], cfg["seed"], cfg, tag="ranks")


if __name__ == "__main__":
    main()
