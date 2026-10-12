"""Technology mixing of the final mixture-of-experts model and its variants (shared-start training).

For every variant / seed / fold of <bench ckpt>/warm_<variant>, the held-out KPMP cells are selected exactly as
in run_bench.py (scoring cells, fold f, 20,000 cells drawn with the seed), embedded with the fold model
(latent mean z) and scored by the share of each cell's 30 nearest neighbours from the same technology.
Writes <ckpt>/warm_<variant>/seed{s}_fold{f}/test_embedding.npz and results/bench_warm_mixing/{mixing.tsv,
mixing_summary.tsv} + PROVENANCE.json.
Also writes technology_mixing.tsv: the comparison methods (results/bench_summary/mixing_summary.tsv, other
methods only) together with the mixture-of-experts model and its variants from this script.
Usage: python warm_mixing.py [--variants full no_adv K1 dense no_cond] --device cuda:5 [--table-only]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import load_bundle  # noqa: E402
from run_bench import scoring_cells  # noqa: E402
from harm_cells import emb_moe  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, repro  # noqa: E402


def same_tech_share(E: np.ndarray, tech: np.ndarray, k: int = 30) -> np.ndarray:
    nn = NearestNeighbors(n_neighbors=k + 1).fit(E).kneighbors(E, return_distance=False)[:, 1:]
    return (tech[nn] == tech[:, None]).mean(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="*", default=["full", "no_adv", "K1", "dense", "no_cond"])
    ap.add_argument("--device", default="cuda:5"); ap.add_argument("--table-only", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg("bench"); root = Path(cfg["paths"]["ckpt"])
    if a.table_only:
        return mixing_table()
    kpmp, ext = load_bundle(cfg)
    cells = scoring_cells(kpmp, ext, cfg["train_cells"]["score_cap_per_participant"], cfg["seed"])
    rows = []
    for v in a.variants:
        for ck in sorted((root / f"warm_{v}").glob("seed*_fold*")):
            seed, f = map(int, ck.name.replace("seed", "").split("_fold"))
            te = np.where(((cells.obs.dataset == "KPMP") & (cells.obs.fold == f)).values)[0]
            te = np.random.default_rng(seed).choice(te, min(len(te), cfg["metrics"]["lisi_cells"]), replace=False)
            sub = cells[te].copy()
            E = emb_moe(ck, sub, cfg, a.device, seed, f, None)["emb"]
            tech, st = sub.obs.tech.values.astype(str), sub.obs.pt_state.values.astype(str)
            np.savez_compressed(ck / "test_embedding.npz", emb=E.astype(np.float32), tech=tech, pt_state=st)
            same = same_tech_share(E, tech)
            rows.append({"variant": v, "seed": seed, "fold": f, "metric": "kNN30_same_tech", "value": same.mean()})
            for s_ in ("healthy_PT", "injured_PT"):
                rows.append({"variant": v, "seed": seed, "fold": f, "metric": f"kNN30_same_tech_{s_}",
                             "value": same[st == s_].mean()})
            print(v, ck.name, round(float(same.mean()), 3), flush=True)
    out = result_dir("bench_warm_mixing")
    R = pd.DataFrame(rows); R.to_csv(out / "mixing.tsv", sep="\t", index=False)
    S = (R.groupby(["variant", "seed", "metric"]).value.mean().groupby(["variant", "metric"])
         .agg(["mean", "std", "count"]).reset_index())
    S.to_csv(out / "mixing_summary.tsv", sep="\t", index=False)
    print(S.to_string(index=False))
    repro.write_provenance("../dkd_moe/results/bench_warm_mixing", [HERE / "warm_mixing.py"],
                           [out / "mixing.tsv", out / "mixing_summary.tsv"], seed=cfg["seed"])


NAMES = {"full": "Mixture of experts latent", "no_adv": "Mixture of experts, no technology discriminator",
         "K1": "Mixture of experts, single expert", "dense": "Mixture of experts, dense routing",
         "no_cond": "Mixture of experts, no condition classifier"}


def mixing_table():
    out = result_dir("bench_warm_mixing")
    B = pd.read_csv(result_dir("bench_summary") / "mixing_summary.tsv", sep="\t")
    B = B[~B.method.str.startswith("moe")]
    rows = [{"method": m, "kNN30_same_tech": r.kNN30_same_tech, "kNN30_same_tech_healthy_PT": r.kNN30_same_tech_healthy_PT,
             "kNN30_same_tech_injured_PT": r.kNN30_same_tech_injured_PT} for m, r in zip(B.method, B.itertuples())]
    S = pd.read_csv(out / "mixing_summary.tsv", sep="\t").pivot(index="variant", columns="metric", values="mean")
    rows += [{"method": NAMES[v], **S.loc[v].to_dict()} for v in NAMES if v in S.index]
    T = pd.DataFrame(rows); T.to_csv(out / "technology_mixing.tsv", sep="\t", index=False)
    print(T.to_string(index=False))


if __name__ == "__main__":
    main()
