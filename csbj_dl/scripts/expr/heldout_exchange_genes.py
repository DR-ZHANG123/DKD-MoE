"""Gene-level effect of the main held-out exchange, in units of the observed DKD-HKD difference.

Partition models with expert identities from training participants only (results/expr_trainonly_experts/
assignments.tsv). In injured-PT cells of held-out HKD participants, the routing weight of the HKD-weighted expert is
moved to the DKD-weighted expert (intervention_common.run_intervention); the change in decoded log expression is
averaged per participant x technology, then over technologies and the five models of the part. Reference scale: the
difference between held-out DKD and HKD participants in the unperturbed decoded log expression of the same models.
Per gene and per gene group (expert_intervention.yaml gene_groups): mean change with participant bootstrap 95% CI,
Wilcoxon signed-rank P, observed DKD-HKD difference and the ratio of the two.
Outputs: results/expr_heldout_experts/{exchange_genes.tsv, exchange_gene_groups.tsv} + PROVENANCE_genes.json
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from train import build_model  # noqa: E402
from intervention_common import select_cells, run_intervention, participant_means  # noqa: E402


def boot_mean(v, rng, n=2000):
    v = np.asarray(v, float); b = rng.choice(v, (n, len(v))).mean(1)
    return float(v.mean()), float(np.quantile(b, 0.025)), float(np.quantile(b, 0.975))


def main():
    cfg = load_cfg("expr"); ic = load_cfg("expert_intervention"); set_global_seed(ic["seed"]); rng = np.random.default_rng(ic["seed"])
    out = result_dir("expr_heldout_experts"); ck = Path(cfg["paths"]["ckpt_dir"]); dev = cfg["train"]["device"]
    st = cfg["readout_stability"]["stability"]; pre = st["prefix"]; seeds = st["cv_seeds"]
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    A = pd.read_csv(result_dir("expr_trainonly_experts") / "assignments.tsv", sep="\t").set_index("fold")
    fold = data.obs.fold.values.astype(int); genes = data.genes
    delta, base = [], {"DKD": [], "HKD": []}
    for f in range(cfg["n_folds"]):
        te = np.where(fold == f)[0]; ob = data.obs.iloc[te]
        cells = {}
        for g in ("DKD", "HKD"):
            c = select_cells(ob, g, ic["cap_cells_per_unit"], ic["min_cells_per_unit"], rng); c["row"] = te[c.row.values]; cells[g] = c
        D_f, B_f = [], {"DKD": [], "HKD": []}
        for s in seeds:
            m = build_model(cfg, data.X.shape[1], K).to(dev)
            m.load_state_dict(torch.load(ck / pre / f"{pre}_K{K}_wc{wc}_s{s}_f{f}.pt", map_location=dev)); m.eval()
            src, dst = int(A.loc[f, f"seed{s}_hkd_expert"]), int(A.loc[f, f"seed{s}_dkd_expert"])
            r = run_intervention(m, data, cells["HKD"].row.values, src, dst)
            D_f.append(participant_means(cells["HKD"], r["delta"]))
            B_f["HKD"].append(participant_means(cells["HKD"], r["logrho0"]))
            B_f["DKD"].append(participant_means(cells["DKD"], run_intervention(m, data, cells["DKD"].row.values, src, src)["logrho0"]))
        delta.append(sum(D_f) / len(D_f))
        for g in base:
            base[g].append(sum(B_f[g]) / len(B_f[g]))
        print("part", f, flush=True)
    D = pd.concat(delta); D.columns = genes
    obs_diff = pd.concat(base["DKD"]).mean(0).values - pd.concat(base["HKD"]).mean(0).values
    obs_diff = pd.Series(obs_diff, index=genes)
    rows = []
    for g in genes:
        m_, lo, hi = boot_mean(D[g], rng, 500)
        rows.append({"gene": g, "mean_change": m_, "ci_low": lo, "ci_high": hi, "observed_DKD_minus_HKD": obs_diff[g]})
    Gt = pd.DataFrame(rows); Gt["ratio_to_observed"] = Gt.mean_change / Gt.observed_DKD_minus_HKD
    Gt.to_csv(out / "exchange_genes.tsv", sep="\t", index=False)
    grp = []
    for name, gl in ic["gene_groups"].items():
        gl = [x for x in gl if x in D.columns]; v = D[gl].mean(1)
        m_, lo, hi = boot_mean(v, rng)
        od = float(obs_diff[gl].mean())
        grp.append({"gene_group": name, "genes": ",".join(gl), "n_participants": len(v), "mean_change": m_, "ci_low": lo,
                    "ci_high": hi, "p_signed_rank": float(stats.wilcoxon(v).pvalue), "n_positive": int((v > 0).sum()),
                    "observed_DKD_minus_HKD": od, "ratio_to_observed": m_ / od if od else np.nan})
    G = pd.DataFrame(grp); G.to_csv(out / "exchange_gene_groups.tsv", sep="\t", index=False)
    pd.set_option("display.width", 220); print(G.round(4).to_string())
    print(Gt[Gt.gene.isin(sum(ic["gene_groups"].values(), []))].round(4).to_string())
    write_provenance("expr_heldout_experts", [result_dir("expr_trainonly_experts") / "assignments.tsv"],
                     [out / "exchange_genes.tsv", out / "exchange_gene_groups.tsv"], ic["seed"], cfg, tag="genes")


if __name__ == "__main__":
    main()
