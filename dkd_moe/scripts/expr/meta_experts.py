"""Expert consensus across seeds: cluster all seed x expert units into meta-experts.

Per unit: (i) IG gene profile of its routing probability (cells whose own top-1 expert it is; baseline = mean
healthy-PT profile) and (ii) decoded mean log profile with the gate fixed to that expert on a common cell set,
centred on the across-unit mean. Similarity = mean of the two Pearson correlations; average linkage, cut at
r = cut_r. Meta-experts present in >= min_seeds seeds are the interpretable ones.
Outputs: results/expr_meta/{units.tsv, meta_experts.tsv, meta_ig_consensus.tsv, meta_top_genes.tsv,
         unit_similarity.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from model import log_norm  # noqa: E402
from interpret_common import integrated_gradients  # noqa: E402
from readout_common import final_set, load_model, head_diff, cohort_pred  # noqa: E402


@torch.no_grad()
def decode_fixed(model, x, tech, k):
    h = model.encode_h(log_norm(x)); E = model.expert_out(h)
    g = F.one_hot(torch.full((x.shape[0],), k, device=x.device), model.K).float()
    rho, _ = model.decode(model.z_mu(model.mix(g, E)), F.one_hot(tech, model.n_tech).float(),
                          torch.ones(x.shape[0], 1, device=x.device))
    return torch.log(rho.mean(0) + 1e-7).cpu().numpy()


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]; mc = cfg["readout_stability"]["meta_experts"]
    out = result_dir("expr_meta"); bulk = Path(cfg["paths"]["bulk_dir"]); fs = final_set(cfg)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    st = data.obs.pt_state.astype(str).values; grp = data.obs.group.astype(str).values
    rng = np.random.default_rng(cfg["seed"])
    dec_idx = np.sort(rng.choice(len(data), mc["decode_cells"], replace=False))
    healthy = np.where(st == "healthy_PT")[0]
    with torch.no_grad():
        x0 = torch.cat([log_norm(data.rows(healthy[s:s + 8192])) for s in range(0, len(healthy), 8192)]).mean(0, keepdim=True)
    xd, td = data.rows(dec_idx), data.tech[torch.as_tensor(dec_idx, device=dev)]
    units, IG, DEC = [], [], []
    for s in fs["seeds"]:
        model, genes = load_model(cfg, fs, s, dev)
        P = cohort_pred(cfg, fs, s, "kpmp")["p"]; top = P.argmax(1); wdiff, _ = head_diff(model)
        for k in range(fs["K"]):
            cells = rng.permutation(np.where(top == k)[0])[:1000]
            if len(cells) < 50:
                continue
            fn = lambda z, k=k: model.gate_probs(model.encode_h(z))[0][:, k]
            with torch.no_grad():
                xl = log_norm(data.rows(cells))
            IG.append(integrated_gradients(fn, xl, x0, cfg["interpret"]["ig_steps"]).mean(0).cpu().numpy())
            DEC.append(decode_fixed(model, xd, td, k))
            inj = st == "injured_PT"
            units.append({"seed": s, "expert": k, "head_weight_dkd_minus_hkd": float(wdiff[k]),
                          "top1_share": float((top == k).mean()),
                          "mean_routing_injured": float(P[inj, k].mean()), "mean_routing_healthy": float(P[st == "healthy_PT", k].mean()),
                          "mean_routing_injured_DKD": float(P[inj & (grp == "DKD"), k].mean()),
                          "mean_routing_injured_HKD": float(P[inj & (grp == "HKD"), k].mean())})
        print("seed", s, "done", flush=True)
    IG, DEC = np.array(IG), np.array(DEC)
    Rig = np.corrcoef(IG); Rdec = np.corrcoef(DEC - DEC.mean(0, keepdims=True))
    R = (Rig + Rdec) / 2
    D = np.clip(1 - R, 0, 2); np.fill_diagonal(D, 0)
    lab = fcluster(linkage(squareform(D, checks=False), "average"), t=1 - mc["cut_r"], criterion="distance")
    U = pd.DataFrame(units); U["cluster"] = lab
    summ = U.groupby("cluster").agg(n_units=("seed", "size"), n_seeds=("seed", "nunique"),
                                    head_weight=("head_weight_dkd_minus_hkd", "mean"),
                                    routing_injured=("mean_routing_injured", "mean"),
                                    routing_healthy=("mean_routing_healthy", "mean"),
                                    routing_injured_DKD=("mean_routing_injured_DKD", "mean"),
                                    routing_injured_HKD=("mean_routing_injured_HKD", "mean")).reset_index()
    within = []
    for c in summ.cluster:
        m = np.where(lab == c)[0]
        within.append(np.nan if len(m) < 2 else float(R[np.ix_(m, m)][np.triu_indices(len(m), 1)].mean()))
    summ["mean_within_r"] = within
    summ = summ.sort_values(["n_seeds", "n_units"], ascending=False).reset_index(drop=True)
    summ["meta_expert"] = [f"M{i + 1}" for i in range(len(summ))]
    summ["reproducible"] = summ.n_seeds >= mc["min_seeds"]
    U["meta_expert"] = U.cluster.map(dict(zip(summ.cluster, summ.meta_expert)))
    U.to_csv(out / "units.tsv", sep="\t", index=False); summ.to_csv(out / "meta_experts.tsv", sep="\t", index=False)
    pd.DataFrame(R, index=U.seed.astype(str) + "_" + U.expert.astype(str), columns=U.seed.astype(str) + "_" + U.expert.astype(str)
                 ).to_csv(out / "unit_similarity.tsv", sep="\t")
    Z = (IG - IG.mean(1, keepdims=True)) / IG.std(1, keepdims=True)
    cons = pd.DataFrame({m: Z[(U.meta_expert == m).values].mean(0) for m in summ.meta_expert}, index=genes)
    cons.index.name = "gene"; cons.to_csv(out / "meta_ig_consensus.tsv", sep="\t")
    tops = pd.concat([pd.DataFrame({"meta_expert": m, "direction": d, "rank": range(1, 31),
                                    "gene": (cons[m].nlargest(30) if d == "up" else cons[m].nsmallest(30)).index})
                      for m in summ.meta_expert for d in ("up", "down")])
    tops.to_csv(out / "meta_top_genes.tsv", sep="\t", index=False)
    print(summ.round(3).to_string())
    for m in summ[summ.reproducible].meta_expert:
        print(m, cons[m].nlargest(15).index.tolist())
    write_provenance("expr_meta", [fs["dir"] / (fs["stem"].format(s=s) + ".pt") for s in fs["seeds"]],
                     [out / "units.tsv", out / "meta_experts.tsv", out / "meta_ig_consensus.tsv"], cfg["seed"], cfg,
                     extra={"final_set": str(fs["dir"]), "n_meta": int(len(summ)),
                            "n_reproducible": int(summ.reproducible.sum())})


if __name__ == "__main__":
    main()
