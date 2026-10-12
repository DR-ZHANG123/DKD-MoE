"""Match experts across seeds / fold models (Hungarian, routing-probability correlation) and write
consensus per-cell gate weights.

Reference expert order = final model seed 1 on all KPMP cells.
Outputs (bulk, gzipped tsv): <bulk_dir>/gates/{kpmp_heldout,kpmp_final,gse211785,gse195460}_gates.tsv.gz
         (bulk npz)        : <bulk_dir>/gates/per_seed_matched.npz (per-seed matched routing, for robustness)
Tables: results/expr_gates/{seed_matching.tsv, fold_matching.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import match_experts  # noqa: E402

OBS = ["participant", "group", "tech", "pt_state"]


def write_cells(fn, obs, P, extra=None):
    K = P.shape[1]
    df = obs[[c for c in OBS if c in obs]].copy()
    for c in df.columns:
        df[c] = df[c].astype(str)
    for k in range(K):
        df[f"expert_{k}"] = P[:, k].round(5)
    df["top1_expert"] = P.argmax(1)
    for k, v in (extra or {}).items():
        df[k] = np.round(v, 5)
    df.index.name = "cell"
    df.to_csv(fn, sep="\t", compression="gzip")


def main():
    cfg = load_cfg("expr")
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K, wc = int(cfg["final"]["K"] or sel["K"]), float(sel["w_cond"])
    ck = Path(cfg["paths"]["ckpt_dir"]); bulk = Path(cfg["paths"]["bulk_dir"])
    gdir = bulk / "gates"; gdir.mkdir(parents=True, exist_ok=True)
    out = result_dir("expr_gates")
    seeds = cfg["final"]["seeds"]
    obs = {c: ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad", backed="r").obs.copy() for c in ("kpmp", "gse211785", "gse195460")}
    f = lambda s, c: np.load(ck / "final" / f"final_K{K}_s{s}_{c}.npz")
    ref = f(seeds[0], "kpmp")["p"]
    perms, rows, per_seed = {}, [], {}
    for s in seeds:
        perm, r = match_experts(ref, f(s, "kpmp")["p"])
        perms[s] = perm
        rows += [{"seed": s, "ref_expert": k, "matched_expert": int(perm[k]), "r": float(r[k])} for k in range(K)]
    pd.DataFrame(rows).to_csv(out / "seed_matching.tsv", sep="\t", index=False)
    classes = cfg["condition"]["classes"]; iD, iH = classes.index("DKD"), classes.index("HKD")
    outs = []
    for c in ("kpmp", "gse211785", "gse195460"):
        Ps = np.stack([f(s, c)["p"][:, perms[s]] for s in seeds])
        CP = np.stack([f(s, c)["cond_prob"] for s in seeds]).mean(0)
        per_seed[c] = Ps.astype(np.float16)
        name = "kpmp_final" if c == "kpmp" else c
        fn = gdir / f"{name}_gates.tsv.gz"
        write_cells(fn, obs[c], Ps.mean(0), {"p_dkd_vs_hkd": CP[:, iD] / (CP[:, iD] + CP[:, iH] + 1e-9)})
        outs.append(fn)
    # held-out KPMP predictions from the CV models at the selected configuration
    n = len(obs["kpmp"]); acc = np.zeros((n, K)); cnt = np.zeros(n); cpacc = np.zeros(n)
    frows = []
    for s in cfg["cv"]["seeds_for_stability"]:
        for fo in range(cfg["n_folds"]):
            d = np.load(ck / "cv" / f"cv_K{K}_wc{wc}_s{s}_f{fo}.npz")
            perm, r = match_experts(ref[d["idx"]], d["p"])
            acc[d["idx"]] += d["p"][:, perm]; cnt[d["idx"]] += 1
            cp = d["cond_prob"]; cpacc[d["idx"]] += cp[:, iD] / (cp[:, iD] + cp[:, iH] + 1e-9)
            frows += [{"cv_seed": s, "fold": fo, "ref_expert": k, "matched_expert": int(perm[k]), "r": float(r[k])}
                      for k in range(K)]
    pd.DataFrame(frows).to_csv(out / "fold_matching.tsv", sep="\t", index=False)
    fn = gdir / "kpmp_heldout_gates.tsv.gz"
    write_cells(fn, obs["kpmp"], acc / cnt[:, None], {"p_dkd_vs_hkd": cpacc / cnt})
    outs.append(fn)
    np.savez_compressed(gdir / "per_seed_matched.npz", **per_seed)
    fin = result_dir("expr_final")
    pd.concat([pd.read_csv(c, sep="\t").assign(model=c.stem) for c in sorted((fin / "curves").glob("final_*.tsv"))]
              ).to_csv(fin / "training_curves.tsv", sep="\t", index=False)
    summ = {"K": K, "w_cond": wc, "seed_match_r_mean": float(pd.DataFrame(rows).r.mean()),
            "fold_match_r_mean": float(pd.DataFrame(frows).r.mean())}
    (out / "matching_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))
    write_provenance("expr_gates", [ck / "final" / f"final_K{K}_s{s}.pt" for s in seeds],
                     outs + [out / "seed_matching.tsv", out / "fold_matching.tsv"], cfg["seed"], cfg,
                     extra=summ, tag="consensus")


if __name__ == "__main__":
    main()
