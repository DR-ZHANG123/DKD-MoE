"""In-silico routing perturbation: route KPMP HKD injured-PT cells to the DKD-preferred expert and decode.

For each final seed: original decoded profile (own routing, z = mu) vs perturbed decoded profile with the
gate replaced by (a) a one-hot on the DKD-preferred expert [primary], (b) the mean routing of DKD
injured-PT cells. Same technology and unit library for both. Per gene log2 fold change of mean decoded
expression, averaged across seeds; overlap with the 20 marker genes and Hallmark inflammatory sets.
Outputs: results/expr_interpret/{perturbation_genes.tsv, perturbation_overlap.tsv, perturbation_summary.json}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from model import log_norm  # noqa: E402
from interpret_common import load_final, marker_sets, hallmark_sets, overlap_tests  # noqa: E402


@torch.no_grad()
def decode_with_gate(model, x, tech, gate_fn, bs=4096):
    acc = None
    for s in range(0, x.shape[0], bs):
        xb, tb = x[s:s + bs], tech[s:s + bs]
        h = model.encode_h(log_norm(xb))
        p, _ = model.gate_probs(h)
        g = gate_fn(model.sparse_gate(p))
        m = model.mix(g, model.expert_out(h))
        rho, _ = model.decode(model.z_mu(m), F.one_hot(tb, model.n_tech).float(), torch.ones(len(xb), 1, device=x.device))
        acc = rho.sum(0) if acc is None else acc + rho.sum(0)
    return (acc / x.shape[0]).cpu().numpy()


@torch.no_grad()
def mean_gate(model, data, idx, bs=4096):
    acc = 0
    for s in range(0, len(idx), bs):
        acc = acc + model.sparse_gate(model.gate_probs(model.encode_h(log_norm(data.rows(idx[s:s + bs]))))[0]).sum(0)
    return acc / len(idx)


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]
    out = result_dir("expr_interpret"); bulk = Path(cfg["paths"]["bulk_dir"])
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    dk = int(json.loads((result_dir("expr_gates") / "dkd_expert.json").read_text())["dkd_preferred_expert"].split("_")[1])
    o = data.obs
    hk = np.where((o.group.astype(str) == "HKD").values & (o.pt_state.astype(str) == "injured_PT").values)[0]
    dkd = np.where((o.group.astype(str) == "DKD").values & (o.pt_state.astype(str) == "injured_PT").values)[0]
    xi, ti = data.rows(hk), data.tech[torch.as_tensor(hk, device=dev)]
    eps = 1e-7
    L1, L2, obs_lfc = [], [], None
    for s in cfg["final"]["seeds"]:
        model, perm, genes = load_final(cfg, s, dev)
        kk = int(perm[dk])
        base = decode_with_gate(model, xi, ti, lambda g: g)
        onehot = decode_with_gate(model, xi, ti, lambda g: F.one_hot(torch.full((g.shape[0],), kk, device=g.device),
                                                                         g.shape[1]).float())
        mg = mean_gate(model, data, dkd)
        dmean = decode_with_gate(model, xi, ti, lambda g: mg.expand_as(g))
        L1.append(np.log2((onehot + eps) / (base + eps))); L2.append(np.log2((dmean + eps) / (base + eps)))
        print("seed", s, "done", flush=True)
    # observed DKD vs HKD injured-PT difference (participant-averaged CP10k) as a reference direction
    X = data.X
    def part_cp10k(idx):
        xs = X[torch.as_tensor(idx)]
        cp = xs / xs.sum(1, keepdim=True).clamp_min(1) * 1e4
        df = pd.DataFrame(cp.numpy()); df["p"] = o.participant.values[idx]
        return df.groupby("p").mean().mean(0).values
    obs_lfc = np.log2((part_cp10k(dkd) + 1e-3) / (part_cp10k(hk) + 1e-3))
    res = pd.DataFrame({"gene": genes, "lfc_onehot_dkd_expert": np.mean(L1, 0), "lfc_onehot_sd_seeds": np.std(L1, 0),
                        "lfc_mean_dkd_routing": np.mean(L2, 0), "observed_lfc_dkd_vs_hkd_injured": obs_lfc})
    res = res.sort_values("lfc_onehot_dkd_expert", ascending=False)
    res.to_csv(out / "perturbation_genes.tsv", sep="\t", index=False)
    sets = {**marker_sets(cfg), **hallmark_sets(cfg)}
    s1 = res.set_index("gene").lfc_onehot_dkd_expert; s2 = res.set_index("gene").lfc_mean_dkd_routing
    ov = pd.concat([overlap_tests(s1, sets, cfg["interpret"]["top_n_genes"], "onehot_dkd_expert"),
                    overlap_tests(s2, sets, cfg["interpret"]["top_n_genes"], "mean_dkd_routing")])
    ov.to_csv(out / "perturbation_overlap.tsv", sep="\t", index=False)
    r = res.set_index("gene")
    summ = {"n_hkd_injured_cells": int(len(hk)), "dkd_expert": dk,
            "spearman_onehot_vs_observed": float(r.lfc_onehot_dkd_expert.corr(r.observed_lfc_dkd_vs_hkd_injured, method="spearman")),
            "spearman_meanrouting_vs_observed": float(r.lfc_mean_dkd_routing.corr(r.observed_lfc_dkd_vs_hkd_injured, method="spearman")),
            "top20_up": res.gene.head(20).tolist(), "top20_down": res.gene.tail(20).tolist()[::-1]}
    (out / "perturbation_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))
    print(ov.drop(columns=[c for c in ov.columns if c.endswith("_genes")]).round(4).to_string())
    write_provenance("expr_interpret", [bulk / "kpmp_pt_hvg.h5ad"],
                     [out / "perturbation_genes.tsv", out / "perturbation_overlap.tsv", out / "perturbation_summary.json"],
                     cfg["seed"], cfg, extra=summ, tag="perturbation")


if __name__ == "__main__":
    main()
