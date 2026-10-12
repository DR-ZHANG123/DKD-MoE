"""Components of the supervised model without experts, chosen like the experts, for a like-for-like comparison.

The supervised model without experts (one expert, condition classifier on the latent mean; scripts/bench/sup_single.py)
reads diagnosis as w^T mu, so a participant's readout is also a sum of components w_j * mean(mu_j). For each partition
model (part f, seed s), the latent dimension with the largest DKD-ward contribution difference between injured and
healthy cells of the training participants (w_j * (mean mu_j injured - mean mu_j healthy)) is the injury component, and
the dimension with the most negative weighted difference the opposite component; both are chosen from training
participants only, like the expert identities of trainonly_experts.py. Held-out contributions: w_j * mu_j over
injured-PT cells, participant x technology means, mean over technologies; seeds averaged.
Agreement between seeds (ICC(2,1) of participant contributions) is computed for these components and for the
DKD-weighted expert (expert identities from training participants only).
Outputs: results/expr_nonmoe_components/{choices.tsv, contributions.tsv, seed_agreement.tsv, group_tests.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "bench"))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from readout import part_means  # noqa: E402
from stats_utils import hedges_g  # noqa: E402
from stability_compare import icc_2_1  # noqa: E402

COMPS = [("DKD", "HKD"), ("DM_noCKD", "HKD"), ("HKD_withDM", "HKD"), ("DKD", "DM_noCKD")]


def main():
    cfg = load_cfg("expr"); ic = load_cfg("expert_intervention"); set_global_seed(ic["seed"]); rng = np.random.default_rng(ic["seed"])
    out = result_dir("expr_nonmoe_components"); dev = cfg["train"]["device"]; seeds = cfg["readout_stability"]["stability"]["cv_seeds"]
    from sup_single import make_builder
    build = make_builder("z"); bck = Path(load_cfg("bench")["paths"]["ckpt"]) / f"warm_{ic['heldout']['no_expert_variant']}"
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    fold = data.obs.fold.values.astype(int); st = data.obs.pt_state.astype(str).values
    vals = {"Lpos": {}, "Lneg": {}}; ch = []
    for f in range(cfg["n_folds"]):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        inj_tr = rng.choice(tr[st[tr] == "injured_PT"], 10000, replace=False); hea_tr = rng.choice(tr[st[tr] == "healthy_PT"], 10000, replace=False)
        te_inj = te[st[te] == "injured_PT"]
        for s in seeds:
            m = build(cfg, data.X.shape[1], 1).to(dev)
            m.load_state_dict(torch.load(bck / f"seed{s}_fold{f}" / "model.pt", map_location=dev)["state_dict"]); m.eval()
            w = (m.cond_head.weight[1] - m.cond_head.weight[0]).detach().cpu().numpy()
            zi, zh = forward_all(m, data, np.sort(inj_tr))["z"].astype(float), forward_all(m, data, np.sort(hea_tr))["z"].astype(float)
            d = w * (zi.mean(0) - zh.mean(0)); jp, jn = int(np.argmax(d)), int(np.argmin(d))
            ch.append({"fold": f, "seed": s, "pos_dim": jp, "pos_weighted_difference": d[jp], "neg_dim": jn,
                       "neg_weighted_difference": d[jn], "share_of_total_difference_pos": d[jp] / d.sum()})
            z = forward_all(m, data, te_inj)["z"].astype(float)
            for key, j in (("Lpos", jp), ("Lneg", jn)):
                v = np.full(len(data), np.nan); v[te_inj] = w[j] * z[:, j]
                vals[key].setdefault(s, np.full(len(data), np.nan))[te_inj] = v[te_inj]
        print("part", f, flush=True)
    C = pd.DataFrame(ch); C.to_csv(out / "choices.tsv", sep="\t", index=False)
    mc = cfg["prep"]["min_cells_per_participant"]; cols, per_seed = [], {}
    for key, dd in vals.items():
        pm = part_means(data.obs, {f"s{s}": v for s, v in dd.items()}, mc); pm = pm[pm.tech == "pooled"].set_index("participant")
        per_seed[key] = pm[[f"s{s}" for s in dd]]; cols.append(per_seed[key].mean(1).rename(f"nonmoe_{key}"))
    T = pd.concat(cols, axis=1)
    T["group"] = data.obs.groupby("participant", observed=True).group.first().astype(str).reindex(T.index)
    T.index.name = "participant"; T.to_csv(out / "contributions.tsv", sep="\t")
    E = pd.read_csv(result_dir("expr_trainonly_experts") / "per_seed_contributions.tsv", sep="\t", dtype={"participant": str}).set_index("participant")
    ag = [{"component": "non-MoE injury component", **{"icc_2_1": icc_2_1(per_seed["Lpos"].dropna().values)}},
          {"component": "non-MoE opposite component", **{"icc_2_1": icc_2_1(per_seed["Lneg"].dropna().values)}},
          {"component": "DKD-weighted expert (training-only identity)", "icc_2_1": icc_2_1(E[[c for c in E if c.startswith("M3_s")]].dropna().values)},
          {"component": "HKD-weighted expert (training-only identity)", "icc_2_1": icc_2_1(E[[c for c in E if c.startswith("M1_s")]].dropna().values)}]
    pd.DataFrame(ag).to_csv(out / "seed_agreement.tsv", sep="\t", index=False)
    G = pd.DataFrame([{"component": k, "comparison": f"{a}_vs_{b}", **hedges_g(T[k][T.group == a], T[k][T.group == b])}
                      for k in ("nonmoe_Lpos", "nonmoe_Lneg") for a, b in COMPS])
    G.to_csv(out / "group_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(C.round(3).to_string()); print(pd.DataFrame(ag).round(3)); print(G.round(3).to_string())
    write_provenance("expr_nonmoe_components", [], [out / f for f in ("choices.tsv", "contributions.tsv", "seed_agreement.tsv",
                                                                     "group_tests.tsv")], ic["seed"], cfg)


if __name__ == "__main__":
    main()
