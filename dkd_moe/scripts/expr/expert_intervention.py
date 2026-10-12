"""In-silico expert intervention on the five final models (KPMP injured proximal tubule cells).

For each final model, routing weight is moved between matched experts inside the model's own forward pass
(top-2 renormalised weights g; same h, same p, same technology, z = mu) and the decoded expression is
compared before and after:
  main      DKD injured-PT cells: M3 -> M1;  HKD injured-PT cells: M1 -> M3
  controls  M3 -> M2 / M3 -> each non-reproducible unit (DKD cells); M1 -> M2 / M1 -> non-reproducible (HKD)
Per-cell change in decoded log expression (natural log of rho) -> participant x technology means -> mean over
technologies -> per-participant scores (transferable signature, gene groups) -> seeds averaged per participant
-> Wilcoxon signed-rank across participants. Genome-wide concordance of the gene-level effect with the
participant-level readout-gene correlation (recomputed with the construction of the transferable signature).
Outputs: results/expr_expert_intervention/{summary.tsv, participant_scores.tsv, gene_effects.tsv,
         specificity.tsv, concordance.tsv, readout_gene_correlation.tsv, sanity.tsv, units_used.tsv,
         observed_difference.tsv (unperturbed decoded DKD-minus-HKD difference, the scale for the effects)} + PROVENANCE.json
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
from readout_common import final_set, load_model  # noqa: E402
from intervention_common import select_cells, run_intervention, participant_means, readout_gene_correlation  # noqa: E402
from intervention_stats import gene_set_scores, summary_table, specificity_table, concordance_rows  # noqa: E402


def unit_map(units: pd.DataFrame, meta: pd.DataFrame, seed: int) -> dict:
    """Meta-expert -> expert index for one model, plus the list of non-reproducible units."""
    u = units[units.seed == seed]
    rep = set(meta.meta_expert[meta.reproducible.astype(str) == "True"])
    m = {k: int(u.expert[u.meta_expert == k].iloc[0]) for k in ("M1", "M2", "M3")}
    m["nonrep"] = [int(e) for e, k in zip(u.expert, u.meta_expert) if k not in rep]
    return m


def run_seed(model, data, cells: dict, iv_cfg: list, um: dict, genes: list) -> tuple[dict, list, dict]:
    """Participant x gene mean change per intervention (non-reproducible destinations averaged), sanity rows
    and per-gene SD of the unperturbed decoded log expression per cell set."""
    deltas, sanity, sds = {}, [], {}
    for iv in iv_cfg:
        c = cells[iv["group"]]; src = um[iv["src"]]
        dsts = um["nonrep"] if iv["dst"] == "nonrep" else [um[iv["dst"]]]
        acc = []
        for dst in dsts:
            r = run_intervention(model, data, c.row.values, src, dst)
            acc.append(participant_means(c, r["delta"]))
            if iv["group"] not in sds:
                sds[iv["group"]] = pd.Series(r["logrho0"].std(0, ddof=1), index=genes)
                b = participant_means(c, r["logrho0"]); b.columns = genes; sds["base_" + iv["group"]] = b
            mv = participant_means(c, r["moved"][:, None])[0]
            fa = participant_means(c, (r["moved"] > 0).astype(float)[:, None])[0]
            sanity.append({"intervention": iv["name"], "src_expert": src, "dst_expert": dst, "n_cells": len(c),
                           "max_abs_delta_p": r["max_abs_dp"], "max_abs_delta_logodds": r["max_abs_dlogodds"],
                           "frac_cells_src_in_top2": float((r["moved"] > 0).mean()),
                           "participant_mean_moved_weight": float(mv.mean()),
                           "participant_mean_frac_cells_affected": float(fa.mean())})
        D = sum(acc) / len(acc); D.columns = genes
        deltas[iv["name"]] = D
    return deltas, sanity, sds


def main():
    cfg = load_cfg("expr"); ic = load_cfg("expert_intervention"); set_global_seed(ic["seed"])
    torch.set_num_threads(int(ic["threads"])); dev = ic["device"]
    out = result_dir("expr_expert_intervention"); bulk = Path(cfg["paths"]["bulk_dir"]); fs = final_set(cfg)
    rc = {**cfg["transfer"], **cfg["transfer"]["pseudobulk"]}
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    units = pd.read_csv(result_dir("expr_meta") / "units.tsv", sep="\t")
    meta = pd.read_csv(result_dir("expr_meta") / "meta_experts.tsv", sep="\t")
    sig = pd.read_csv(result_dir("expr_transfer") / "transfer_signature.tsv", sep="\t")
    up, down = sig.gene[sig.direction == "up"].tolist(), sig.gene[sig.direction == "down"].tolist()
    rng = np.random.default_rng(ic["seed"])
    cells = {g: select_cells(data.obs, g, ic["cap_cells_per_unit"], ic["min_cells_per_unit"], rng) for g in ("DKD", "HKD")}
    print({g: (len(c), c.participant.nunique()) for g, c in cells.items()}, flush=True)

    models, P, S, gene_eff, used, base = [], [], [], {}, [], []
    for s in fs["seeds"]:
        model, genes = load_model(cfg, fs, s, dev)
        assert list(genes) == data.genes
        models.append(model)
        um = unit_map(units, meta, s)
        used.append({"seed": s, **{k: (",".join(map(str, v)) if isinstance(v, list) else v) for k, v in um.items()}})
        deltas, san, sds = run_seed(model, data, cells, ic["interventions"], um, data.genes)
        S += [{"seed": s, **r} for r in san]
        for g in ("DKD", "HKD"):
            sd_pool = np.sqrt((sds["DKD"] ** 2 + sds["HKD"] ** 2) / 2)
            base.append(gene_set_scores(sds["base_" + g], sd_pool, up, down, ic["gene_groups"]).assign(group=g, seed=s))
        for iv in ic["interventions"]:
            D = deltas[iv["name"]]
            sc = gene_set_scores(D, sds[iv["group"]], up, down, ic["gene_groups"])
            P.append(sc.assign(intervention=iv["name"], seed=s))
            gene_eff[(iv["name"], s)] = (D.mean(0), (D / sds[iv["group"]].values).mean(0))
        print("seed", s, "done", flush=True)
    P = pd.concat(P, ignore_index=True)
    P.to_csv(out / "participant_scores.tsv", sep="\t", index=False)
    pd.DataFrame(S).to_csv(out / "sanity.tsv", sep="\t", index=False)
    pd.DataFrame(used).to_csv(out / "units_used.tsv", sep="\t", index=False)

    B = pd.concat(base, ignore_index=True)
    B = B.groupby(["measure", "units", "group", "participant"]).value.mean().unstack("group")
    ref = pd.DataFrame([{"measure": m, "units": u, "n_DKD": int(d.DKD.notna().sum()), "n_HKD": int(d.HKD.notna().sum()),
                         "mean_DKD": d.DKD.mean(), "mean_HKD": d.HKD.mean(), "observed_DKD_minus_HKD": d.DKD.mean() - d.HKD.mean()}
                        for (m, u), d in B.groupby(level=[0, 1])])
    ref.to_csv(out / "observed_difference.tsv", sep="\t", index=False)
    brng = np.random.default_rng(ic["seed"] + 1)
    summ = summary_table(P, ic["n_boot"], brng); summ.to_csv(out / "summary.tsv", sep="\t", index=False)
    spec = specificity_table(P, ic["specificity_pairs"], ic["n_boot"], brng)
    spec.to_csv(out / "specificity.tsv", sep="\t", index=False)

    # gene-level effects (participants averaged within seed, then seeds averaged)
    ge = []
    for iv in ic["interventions"]:
        L = pd.DataFrame({s: gene_eff[(iv["name"], s)][0] for s in fs["seeds"]})
        Z = pd.DataFrame({s: gene_eff[(iv["name"], s)][1] for s in fs["seeds"]})
        ge.append(pd.DataFrame({"intervention": iv["name"], "gene": L.index, "mean_delta_log": L.mean(1).values,
                                "mean_delta_sd": Z.mean(1).values, "seed_min_log": L.min(1).values, "seed_max_log": L.max(1).values,
                                "n_seeds_positive": (L > 0).sum(1).values,
                                "signature": L.index.map(lambda g: "up" if g in up else ("down" if g in down else "")).values}))
    GE = pd.concat(ge, ignore_index=True); GE.to_csv(out / "gene_effects.tsv", sep="\t", index=False)

    # readout-gene correlation (construction of the transferable signature) and genome-wide concordance
    r_expr, r_all, expr_genes = readout_gene_correlation(data, models, rc)
    pd.DataFrame({"gene": r_all.index, "r_all_genes": r_all.values, "r_expressed": r_expr.reindex(r_all.index).values}
                 ).to_csv(out / "readout_gene_correlation.tsv", sep="\t", index=False)
    s_r = r_expr.dropna().sort_values(ascending=False)
    check = {"n_expressed_genes": len(expr_genes),
             "overlap_top50_with_signature_up": len(set(s_r.index[:50]) & set(up)),
             "overlap_bottom50_with_signature_down": len(set(s_r.index[::-1][:50]) & set(down))}
    conc = []
    for iv in ic["interventions"]:
        sign = 1 if iv["group"] == "HKD" else -1     # positive = toward DKD (HKD cells gain M3; DKD cells lose M3)
        eff = GE[GE.intervention == iv["name"]].set_index("gene").mean_delta_log
        eff_seed = {s: gene_eff[(iv["name"], s)][0] for s in fs["seeds"]}
        conc += concordance_rows(iv["name"], eff, eff_seed, r_expr, r_all, up, down, sign)
    C = pd.DataFrame(conc); C.to_csv(out / "concordance.tsv", sep="\t", index=False)

    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    k = summ[summ.measure.isin(["signature_score", "signature_up_mean", "signature_down_mean"]) | summ.measure.str.startswith("group:")]
    print(k[["intervention", "measure", "units", "n_participants", "mean", "ci_low", "ci_high", "median", "q25", "q75",
             "p_wilcoxon", "seeds_same_sign"]].round(4).to_string())
    print(spec.round(4).to_string()); print(C.round(4).to_string()); print(pd.DataFrame(S).round(4).to_string()); print(check)
    write_provenance("expr_expert_intervention", [bulk / "kpmp_pt_hvg.h5ad", result_dir("expr_meta") / "units.tsv",
                     result_dir("expr_meta") / "meta_experts.tsv", result_dir("expr_transfer") / "transfer_signature.tsv"]
                     + [fs["dir"] / (fs["stem"].format(s=s) + ".pt") for s in fs["seeds"]],
                     [out / f for f in ("summary.tsv", "participant_scores.tsv", "gene_effects.tsv", "specificity.tsv",
                                        "concordance.tsv", "readout_gene_correlation.tsv", "sanity.tsv", "units_used.tsv",
                                        "observed_difference.tsv")],
                     ic["seed"], cfg, extra={"intervention_config": ic["_path"], "intervention_config_sha": ic["_sha"],
                                             "final_set": str(fs["dir"]), "signature_reconstruction_check": check,
                                             "n_cells": {g: int(len(c)) for g, c in cells.items()},
                                             "n_participants": {g: int(c.participant.nunique()) for g, c in cells.items()}})


if __name__ == "__main__":
    main()
