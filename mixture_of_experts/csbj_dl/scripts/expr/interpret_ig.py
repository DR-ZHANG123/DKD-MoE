"""Integrated-gradients gene attributions per expert (implemented directly in torch).

Input space: log1p(CP10k) expression; baseline = mean healthy-PT profile of KPMP.
Targets: (a) routing probability of each expert, on cells whose consensus top-1 expert is that expert;
         (b) log-ratio of routing to the DKD-preferred expert vs the strongest shared-injury expert, on
             injured-PT cells routed to either (what separates the DKD state from shared injury).
Averaged over the final seeds after expert matching. Overlap with the 20 marker genes and Hallmark sets.
Outputs: results/expr_interpret/{ig_attributions.tsv, ig_dkd_vs_shared.tsv, ig_top_genes.tsv, ig_overlap.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from model import log_norm  # noqa: E402
from interpret_common import (load_final, integrated_gradients, marker_sets, hallmark_sets,  # noqa: E402
                              overlap_tests, selected_K)


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]
    out = result_dir("expr_interpret"); bulk = Path(cfg["paths"]["bulk_dir"]); ic = cfg["interpret"]
    K = selected_K(cfg)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    gates = pd.read_csv(bulk / "gates/kpmp_final_gates.tsv.gz", sep="\t", index_col=0, usecols=["cell", "pt_state", "top1_expert"])
    info = json.loads((result_dir("expr_gates") / "dkd_expert.json").read_text())
    dk = int(info["dkd_preferred_expert"].split("_")[1])
    roles = pd.read_csv(result_dir("expr_gates") / "expert_roles_kpmp_heldout.tsv", sep="\t")
    sh = roles[roles.role == "shared injury"].sort_values("injured_minus_healthy", ascending=False)
    sh_k = int(sh.expert.iloc[0].split("_")[1]) if len(sh) else None
    rng = np.random.default_rng(cfg["seed"])
    with torch.no_grad():
        xlog_all = torch.cat([log_norm(data.X[s:s + 8192]) for s in range(0, len(data), 8192)])   # host memory
    healthy = np.where(gates.pt_state.values == "healthy_PT")[0]
    x0 = xlog_all[healthy].mean(0, keepdim=True).to(dev)
    top1 = gates.top1_expert.values
    cells = {k: rng.permutation(np.where(top1 == k)[0])[:ic["ig_cells_per_expert"]] for k in range(K)}
    inj = gates.pt_state.values == "injured_PT"
    pair_cells = None
    if sh_k is not None:
        pair_cells = rng.permutation(np.where(inj & np.isin(top1, [dk, sh_k]))[0])[:ic["ig_cells_per_expert"]]
    A = np.zeros((len(cfg["final"]["seeds"]), K, len(data.genes))); B = []
    for si, s in enumerate(cfg["final"]["seeds"]):
        model, perm, _ = load_final(cfg, s, dev)
        for p_ in model.parameters():
            p_.requires_grad_(False)
        for k in range(K):
            kk = int(perm[k])
            fn = lambda z, kk=kk: model.gate_probs(model.encode_h(z))[0][:, kk]
            att = integrated_gradients(fn, xlog_all[torch.as_tensor(cells[k])].to(dev), x0, ic["ig_steps"])
            A[si, k] = att.mean(0).cpu().numpy()
        if pair_cells is not None:
            a, b = int(perm[dk]), int(perm[sh_k])
            fn = lambda z: (lambda pr: torch.log(pr[:, a] + 1e-6) - torch.log(pr[:, b] + 1e-6))(
                model.gate_probs(model.encode_h(z))[0])
            B.append(integrated_gradients(fn, xlog_all[torch.as_tensor(pair_cells)].to(dev), x0,
                                          ic["ig_steps"]).mean(0).cpu().numpy())
        print("seed", s, "done", flush=True)
    att = pd.DataFrame(A.mean(0).T, index=data.genes, columns=[f"expert_{k}" for k in range(K)])
    att.index.name = "gene"
    att.to_csv(out / "ig_attributions.tsv", sep="\t")
    # seed agreement: mean pairwise Spearman of per-seed attribution vectors per expert
    agree = {f"expert_{k}": float(np.mean([pd.Series(A[i, k]).corr(pd.Series(A[j, k]), method="spearman")
                                          for i in range(len(A)) for j in range(i + 1, len(A))])) for k in range(K)}
    sets = {**marker_sets(cfg), **hallmark_sets(cfg)}
    tn = ic["top_n_genes"]
    ov = [overlap_tests(att[c], sets, tn, c) for c in att.columns]
    tops = [pd.DataFrame({"expert": c, "rank": range(1, 51), "gene": att[c].nlargest(50).index,
                          "attribution": att[c].nlargest(50).values}) for c in att.columns]
    if B:
        ds = pd.Series(np.mean(B, 0), index=data.genes, name="ig_dkd_vs_shared")
        ds.to_frame().rename_axis("gene").to_csv(out / "ig_dkd_vs_shared.tsv", sep="\t")
        ov.append(overlap_tests(ds, sets, tn, "dkd_vs_shared"))
        tops.append(pd.DataFrame({"expert": "dkd_vs_shared", "rank": range(1, 51), "gene": ds.nlargest(50).index,
                                  "attribution": ds.nlargest(50).values}))
    ov = pd.concat(ov, ignore_index=True); ov.to_csv(out / "ig_overlap.tsv", sep="\t", index=False)
    pd.concat(tops).to_csv(out / "ig_top_genes.tsv", sep="\t", index=False)
    summ = {"dkd_expert": dk, "shared_expert_for_contrast": sh_k, "seed_agreement_spearman": agree,
            "baseline": "mean log1p-CP10k profile of KPMP healthy PT cells", "ig_steps": ic["ig_steps"],
            "implementation": "direct torch integrated gradients (captum not installed)"}
    (out / "ig_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))
    print(ov[ov.score.isin([f"expert_{dk}", "dkd_vs_shared"])].drop(columns=[c for c in ov.columns if c.endswith("_genes")]).round(4).to_string())
    write_provenance("expr_interpret", [bulk / "kpmp_pt_hvg.h5ad", bulk / "gates/kpmp_final_gates.tsv.gz"],
                     [out / "ig_attributions.tsv", out / "ig_overlap.tsv", out / "ig_top_genes.tsv", out / "ig_summary.json"],
                     cfg["seed"], cfg, extra=summ, tag="ig")


if __name__ == "__main__":
    main()
