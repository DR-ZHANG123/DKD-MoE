"""Technology mixing of the supervised single-expert baseline, measured exactly as warm_mixing.py measures the
final model and its variants: per seed x fold, the held-out KPMP scoring cells of that fold (20,000 drawn with
the seed), embedded with the fold model (latent mean), scored by the share of each cell's 30 nearest
neighbours from the same technology.
Writes <ckpt>/warm_<variant>/seed{s}_fold{f}/test_embedding.npz and results/bench_supervised_single/
{mixing.tsv, mixing_summary.tsv, technology_mixing.tsv} + PROVENANCE_mixing.json.
Usage: python sup_mixing.py --device cuda:2
"""
from __future__ import annotations
import argparse, copy, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import load_bundle  # noqa: E402
from run_bench import scoring_cells  # noqa: E402
from warm_mixing import same_tech_share  # noqa: E402
from sup_single import make_builder, cell_outputs  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402

VARIANTS = {"supervised_single_z": "Supervised single expert (condition head on latent mean)",
            "supervised_single_h": "Supervised single expert (condition head on encoder output)"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--device", default="cuda:2")
    a = ap.parse_args()
    cfg = load_cfg("bench"); ecfg = copy.deepcopy(load_cfg("expr")); root = Path(cfg["paths"]["ckpt"])
    kpmp, ext = load_bundle(cfg)
    cells = scoring_cells(kpmp, ext, cfg["train_cells"]["score_cap_per_participant"], cfg["seed"])
    rows = []
    for v in VARIANTS:
        for ck in sorted((root / f"warm_{v}").glob("seed*_fold*")):
            if not (ck / "model.pt").exists():
                continue
            seed, f = map(int, ck.name.replace("seed", "").split("_fold"))
            te = np.where(((cells.obs.dataset == "KPMP") & (cells.obs.fold == f)).values)[0]
            te = np.random.default_rng(seed).choice(te, min(len(te), cfg["metrics"]["lisi_cells"]), replace=False)
            sub = cells[te].copy()
            st = torch.load(ck / "model.pt", map_location=a.device)
            m = make_builder(st["cond_source"])(ecfg, sub.n_vars, 1).to(a.device)
            m.load_state_dict(st["state_dict"])
            E = cell_outputs(m, sub, ecfg, a.device)["emb"]
            tech, pst = sub.obs.tech.values.astype(str), sub.obs.pt_state.values.astype(str)
            np.savez_compressed(ck / "test_embedding.npz", emb=E.astype(np.float32), tech=tech, pt_state=pst)
            same = same_tech_share(E, tech)
            rows.append({"variant": v, "seed": seed, "fold": f, "metric": "kNN30_same_tech", "value": same.mean()})
            for s_ in ("healthy_PT", "injured_PT"):
                rows.append({"variant": v, "seed": seed, "fold": f, "metric": f"kNN30_same_tech_{s_}",
                             "value": same[pst == s_].mean()})
            print(v, ck.name, round(float(same.mean()), 3), flush=True)
    out = result_dir("bench_supervised_single")
    R = pd.DataFrame(rows); R.to_csv(out / "mixing.tsv", sep="\t", index=False)
    S = (R.groupby(["variant", "seed", "metric"]).value.mean().groupby(["variant", "metric"])
         .agg(["mean", "std", "count"]).reset_index())
    S.to_csv(out / "mixing_summary.tsv", sep="\t", index=False)
    # side by side with the final model, its variants and the comparison methods (same cells and metric)
    W = pd.read_csv(result_dir("bench_warm_mixing") / "technology_mixing.tsv", sep="\t")
    N = S.pivot(index="variant", columns="metric", values="mean")
    T = pd.concat([W, pd.DataFrame([{"method": VARIANTS[v], **N.loc[v].to_dict()} for v in VARIANTS if v in N.index])],
                  ignore_index=True)
    T.to_csv(out / "technology_mixing.tsv", sep="\t", index=False)
    print(T.to_string(index=False))
    write_provenance("bench_supervised_single", [result_dir("bench_warm_mixing") / "technology_mixing.tsv"],
                     [out / "mixing.tsv", out / "mixing_summary.tsv", out / "technology_mixing.tsv"], cfg["seed"], cfg,
                     tag="mixing", extra={"cells": "held-out KPMP scoring cells per fold, 20,000 drawn with the seed"})


if __name__ == "__main__":
    main()
