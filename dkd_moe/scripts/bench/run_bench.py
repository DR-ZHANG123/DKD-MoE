"""Run one comparison method over seeds x KPMP participant folds.

For each fold the method sees only the training-fold KPMP cells (and, for supervised parts, only the
training-fold DKD/HKD participants). It then scores every KPMP participant (held-out fold = test) and
every external participant (forward inference only). Outputs results/bench_<method>/:
donor_scores.tsv, lisi.tsv, tables/ and PROVENANCE.json; checkpoints under <ckpt>/<method>/.
Usage: python run_bench.py --method pseudobulk|scvi|scdisinfact|cvae|moe [--variant V] [--seeds 1 2]
       [--folds 0 1] [--device cuda:2] [--dev]
"""
from __future__ import annotations
import argparse, importlib, sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import load_bundle, bundle_inputs, subsample_cells, pseudobulk, log_cpm, path  # noqa: E402
from bench_score import ref_mask, participant_table  # noqa: E402
from bench_metrics import ilisi  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402

MODULES = {"pseudobulk": "m_pseudobulk", "scvi": "m_scvi", "scdisinfact": "m_scdisinfact", "cvae": "m_cvae",
           "moe": "m_moe"}


def scoring_cells(kpmp, ext, cap, seed):
    parts = [subsample_cells(kpmp, cap, seed)] + [subsample_cells(e, cap, seed) for e in ext.values()]
    cols = ["participant", "tech", "condition", "dataset", "library", "pt_state"]
    for p in parts:
        if "fold" not in p.obs:
            p.obs["fold"] = -1
        p.obs = p.obs[cols + ["fold"]].copy()
    c = ad.concat(parts, join="inner", index_unique=None)
    c.obs_names_make_unique()
    return c


def all_pseudobulk(kpmp, ext, min_cells):
    C, M = [], []
    for a in [kpmp] + list(ext.values()):
        c, m = pseudobulk(a, min_cells)
        if "fold" not in m:
            m["fold"] = -1
        C.append(c); M.append(m)
    C = pd.concat(C).fillna(0); M = pd.concat(M)
    M.index = M.dataset + ":" + M.index
    C.index = M.index
    return C, M


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=list(MODULES))
    ap.add_argument("--variant", default=None)
    ap.add_argument("--seeds", type=int, nargs="*")
    ap.add_argument("--folds", type=int, nargs="*")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dev", action="store_true")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--provenance-only", action="store_true",
                    help="write PROVENANCE.json for an existing complete run without recomputing")
    a = ap.parse_args()
    cfg = load_cfg("bench")
    seeds = a.seeds or cfg["seeds"]
    device = a.device or cfg["device"]
    name = a.method + (f"_{a.variant}" if a.variant else "") + ("_dev" if a.dev else "") + (f"_{a.tag}" if a.tag else "")
    label = a.method + (f"_{a.variant}" if a.variant else "")      # method label in tables (no tag)
    out = result_dir(f"bench_{name}"); (out / "tables").mkdir(exist_ok=True)
    ck = Path(cfg["paths"]["ckpt"]) / name; ck.mkdir(parents=True, exist_ok=True)
    if a.provenance_only:
        D = pd.read_csv(out / "donor_scores.tsv", sep="\t")
        write_provenance(f"bench_{name}", bundle_inputs(a.dev, cfg), [out / "donor_scores.tsv", out / "lisi.tsv"],
                         int(seeds[0]), cfg, extra={"method": a.method, "variant": a.variant,
                                                    "seeds": sorted(int(x) for x in D.seed.unique()),
                                                    "folds": sorted(int(x) for x in D.fold.unique()), "device": device,
                                                    "n_train_cap": cfg["train_cells"]["cap_per_participant"]})
        return
    mod = importlib.import_module(MODULES[a.method])
    kpmp, ext = load_bundle(cfg, dev=a.dev)
    folds = [int(x) for x in (a.folds if a.folds is not None else sorted(kpmp.obs.fold.unique()))]
    print(name, kpmp.shape, {k: v.shape for k, v in ext.items()}, "folds", folds, "seeds", seeds, flush=True)
    C, meta = all_pseudobulk(kpmp, ext, cfg["pseudobulk"]["min_cells"])
    L = log_cpm(C)
    # the same scoring cells for every seed/fold/method (fixed by the global seed)
    cells = scoring_cells(kpmp, ext, cfg["train_cells"]["score_cap_per_participant"], cfg["seed"])
    rows, lis = [], []
    for seed in seeds:
        for f in folds:
            t0 = time.time(); set_global_seed(seed)
            tr = subsample_cells(kpmp[(kpmp.obs.fold != f).values], cfg["train_cells"]["cap_per_participant"], seed)
            ref = ref_mask(meta, f, cfg["contrast"]["case"], cfg["contrast"]["control"])
            ctx = {"cfg": cfg, "seed": seed, "fold": f, "device": device, "train": tr, "cells": cells,
                   "pb_counts": C, "pb_logcpm": L, "meta": meta, "ref": ref, "ckdir": ck / f"seed{seed}_fold{f}",
                   "variant": a.variant, "dev": a.dev}
            ctx["ckdir"].mkdir(exist_ok=True)
            res = mod.run(ctx)
            for sname, s in res["scores"].items():
                rows.append(participant_table(s.reindex(meta.index), meta, ref, label,
                                              sname, seed, f))
            if res.get("emb") is not None:
                te = np.where(((cells.obs.dataset == "KPMP") & (cells.obs.fold == f)).values)[0]
                rng = np.random.default_rng(seed)
                te = rng.choice(te, min(len(te), cfg["metrics"]["lisi_cells"]), replace=False)
                v, vm = ilisi(res["emb"][te], cells.obs.tech.values[te], cfg["metrics"]["lisi_k"],
                          cfg["metrics"]["lisi_perplexity"])
                lis.append({"method": label, "embedding": res["emb_name"], "seed": seed,
                            "fold": f, "metric": "iLISI_tech", "value": v, "n_cells": len(te)})
                lis.append({**lis[-1], "metric": "iLISI_tech_mean", "value": vm})
                # within PT state: sc and sn differ in injured/healthy composition, which alone lowers iLISI
                for st in ("healthy_PT", "injured_PT"):
                    k = te[cells.obs.pt_state.values[te] == st]
                    v, _ = ilisi(res["emb"][k], cells.obs.tech.values[k], cfg["metrics"]["lisi_k"],
                                 cfg["metrics"]["lisi_perplexity"])
                    lis.append({**lis[-2], "metric": f"iLISI_tech_{st}", "value": v, "n_cells": len(k)})
                np.savez_compressed(ctx["ckdir"] / "test_embedding.npz", emb=res["emb"][te].astype(np.float32),
                                    tech=cells.obs.tech.values[te].astype(str),
                                    pt_state=cells.obs.pt_state.values[te].astype(str))
            for tname, t in res.get("tables", {}).items():
                t.to_csv(out / "tables" / f"seed{seed}_fold{f}_{tname}.tsv.gz", sep="\t")
            print(f"seed {seed} fold {f}: {time.time() - t0:.0f}s", flush=True)
            pd.concat(rows).to_csv(out / "donor_scores.tsv", sep="\t", index=False)
            pd.DataFrame(lis).to_csv(out / "lisi.tsv", sep="\t", index=False)
    D = pd.concat(rows)
    chk = D[D.role == "test"].groupby(["score_name", "seed"])[["condition", "score"]].apply(
        lambda d: d[d.condition == cfg["contrast"]["case"]].score.mean() - d[d.condition == cfg["contrast"]["control"]].score.mean())
    print("held-out mean difference DKD - HKD (standardised):\n", chk, flush=True)
    write_provenance(f"bench_{name}", bundle_inputs(a.dev, cfg), [out / "donor_scores.tsv", out / "lisi.tsv"],
                     int(seeds[0]), cfg, extra={"method": a.method, "variant": a.variant, "seeds": [int(x) for x in seeds], "folds": folds,
                                           "device": device, "n_train_cap": cfg["train_cells"]["cap_per_participant"]})


if __name__ == "__main__":
    main()
