"""Expert contributions with expert identities defined from training participants only.

For each part f, the five partition models (trained without part f) are run on a sample of cells of the training
participants; experts of seeds 2-5 are aligned to seed 1 by the correlation of routing probabilities (Hungarian
assignment). Per aligned expert: mean classifier weight (DKD minus HKD) over seeds and mean routing of injured and of
healthy training cells. DKD-weighted = largest mean weight among experts routed more by injured than by healthy
cells; HKD-weighted = most negative mean weight. These choices use no held-out participant of part f; the held-out
participants are then scored: contribution = weight x routing probability, participant x technology means over injured-PT cells,
mean over technologies and seeds. No final model and no held-out participant enters the definition.
Outputs: results/expr_trainonly_experts/{assignments.tsv, contributions.tsv, agreement.tsv, group_tests.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from train import build_model  # noqa: E402
from readout import part_means  # noqa: E402
from stats_utils import hedges_g  # noqa: E402

COMPS = [("DKD", "HKD"), ("DM_noCKD", "HKD"), ("HKD_withDM", "HKD"), ("DKD", "DM_noCKD")]


def main():
    cfg = load_cfg("expr"); ic = load_cfg("expert_intervention"); set_global_seed(ic["seed"])
    out = result_dir("expr_trainonly_experts"); ck = Path(cfg["paths"]["ckpt_dir"]); dev = cfg["train"]["device"]
    st = cfg["readout_stability"]["stability"]; pre = st["prefix"]; seeds = st["cv_seeds"]
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    fold = data.obs.fold.values.astype(int); state = data.obs.pt_state.astype(str).values
    rng = np.random.default_rng(ic["seed"])
    contrib = {"M1t": {}, "M3t": {}}; asg = []
    for f in range(cfg["n_folds"]):
        tr = np.where(fold != f)[0]; te = np.where(fold == f)[0]
        mi = np.sort(rng.choice(tr, ic["trainonly"]["match_cells"], replace=False))
        models, P_tr, W = {}, {}, {}
        for s in seeds:
            m = build_model(cfg, data.X.shape[1], K).to(dev)
            m.load_state_dict(torch.load(ck / pre / f"{pre}_K{K}_wc{wc}_s{s}_f{f}.pt", map_location=dev)); m.eval()
            models[s] = m; P_tr[s] = forward_all(m, data, mi)["p"]
            W[s] = (m.cond_head.weight[1] - m.cond_head.weight[0]).detach().cpu().numpy()
        ref = seeds[0]; perm = {ref: np.arange(K)}; mr = {}
        for s in seeds[1:]:
            C = np.corrcoef(P_tr[ref].T, P_tr[s].T)[:K, K:]
            r, c = linear_sum_assignment(-np.nan_to_num(C)); perm[s] = c[np.argsort(r)]; mr[s] = float(C[r, c].mean())
        inj, hea = state[mi] == "injured_PT", state[mi] == "healthy_PT"
        wbar = np.mean([W[s][perm[s]] for s in seeds], 0)
        rin = np.mean([P_tr[s][:, perm[s]][inj].mean(0) for s in seeds], 0)
        rhe = np.mean([P_tr[s][:, perm[s]][hea].mean(0) for s in seeds], 0)
        injured_experts = np.where(rin > rhe)[0]
        j3 = int(injured_experts[np.argmax(wbar[injured_experts])]); j1 = int(np.argmin(wbar))
        asg.append({"fold": f, "dkd_weighted_aligned": j3, "hkd_weighted_aligned": j1, "mean_weight_dkd": wbar[j3],
                    "mean_weight_hkd": wbar[j1], "routing_injured_dkd": rin[j3], "routing_healthy_dkd": rhe[j3],
                    "mean_alignment_r": float(np.mean(list(mr.values()))),
                    **{f"seed{s}_dkd_expert": int(perm[s][j3]) for s in seeds}, **{f"seed{s}_hkd_expert": int(perm[s][j1]) for s in seeds}})
        for s in seeds:
            p = forward_all(models[s], data, te)["p"]
            for key, j in (("M3t", j3), ("M1t", j1)):
                k = int(perm[s][j]); v = np.full(len(data), np.nan); v[te] = W[s][k] * p[:, k]
                contrib[key].setdefault(s, np.full(len(data), np.nan))
                contrib[key][s][te] = v[te]
        print("part", f, asg[-1], flush=True)
    A = pd.DataFrame(asg); A.to_csv(out / "assignments.tsv", sep="\t", index=False)
    mc = cfg["prep"]["min_cells_per_participant"]; cols, seedcols = [], []
    for key, d in contrib.items():
        pm = part_means(data.obs, {f"s{s}": v for s, v in d.items()}, mc); pm = pm[pm.tech == "pooled"].set_index("participant")
        cols.append(pm[[f"s{s}" for s in d]].mean(1).rename(key))
        seedcols.append(pm[[f"s{s}" for s in d]].rename(columns=lambda c, k=key: f"{k.rstrip('t')}_{c}"))
    S_ = pd.concat(seedcols, axis=1); S_.index.name = "participant"; S_.to_csv(out / "per_seed_contributions.tsv", sep="\t")
    C = pd.concat(cols, axis=1); C.columns = ["M1", "M3"] if list(contrib) == ["M1t", "M3t"] else C.columns
    C = C.rename(columns={"M1t": "M1", "M3t": "M3"})
    C["group"] = data.obs.groupby("participant", observed=True).group.first().astype(str).reindex(C.index)
    C.index.name = "participant"; C.to_csv(out / "contributions.tsv", sep="\t")
    T = pd.read_csv(result_dir("expr_heldout_experts") / "contributions.tsv", sep="\t", dtype={"participant": str}).set_index("participant")
    ag = [{"component": k, "spearman_with_template_matched": C[k].corr(T[k].reindex(C.index), method="spearman"),
           "n": int(C[k].notna().sum())} for k in ("M1", "M3")]
    pd.DataFrame(ag).to_csv(out / "agreement.tsv", sep="\t", index=False)
    G = pd.DataFrame([{"expert": k, "comparison": f"{a}_vs_{b}", **hedges_g(C[k][C.group == a], C[k][C.group == b])}
                      for k in ("M1", "M3") for a, b in COMPS])
    G.to_csv(out / "group_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220)
    print(A.round(3).to_string()); print(pd.DataFrame(ag).round(3)); print(G.round(3).to_string())
    write_provenance("expr_trainonly_experts", [], [out / f for f in ("assignments.tsv", "contributions.tsv", "agreement.tsv",
                                                                    "group_tests.tsv", "per_seed_contributions.tsv")], ic["seed"], cfg)


if __name__ == "__main__":
    main()
