"""Random versus warm-start initialisation (KPMP held-out only).

random_init = CV models of the selected configuration (prefix cv, seeds 1-2, standard initialisation)
warm_start  = warm-started, noise-annealed CV models (prefix cvw, seeds 1-5)
Metrics: seed ARI of top-1 expert, Hungarian-matched routing correlation, held-out NLL, DKD-vs-HKD AUC of the
participant log-odds readout, routing specialisation, and seed agreement of participant log-odds
(ICC(2,1) and mean pairwise Spearman). Outputs: results/expr_stability/{random_vs_warm.tsv, initialization.json,
heldout_logodds_by_seed.tsv}
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import auc, match_experts  # noqa: E402


def icc_2_1(M: np.ndarray) -> float:
    """Two-way random-effects, absolute agreement, single rater ICC (Shrout & Fleiss 2,1); rows = subjects."""
    M = M[~np.isnan(M).any(1)]
    n, k = M.shape
    gm = M.mean()
    msr = k * ((M.mean(1) - gm) ** 2).sum() / (n - 1)
    msc = n * ((M.mean(0) - gm) ** 2).sum() / (k - 1)
    sse = ((M - M.mean(1, keepdims=True) - M.mean(0, keepdims=True) + gm) ** 2).sum()
    mse = sse / ((n - 1) * (k - 1))
    return float((msr - mse) / (msr + (k - 1) * mse + k * (msc - mse) / n))


def seed_agreement(S: pd.DataFrame) -> dict:
    rho = [S[a].corr(S[b], method="spearman") for a, b in itertools.combinations(S.columns, 2)]
    return {"icc_2_1": icc_2_1(S.values), "mean_pairwise_spearman": float(np.mean(rho)), "n_participants": int(len(S))}


def load_heldout(cfg, prefix, K, wc, seed):
    d = Path(cfg["paths"]["ckpt_dir"]) / prefix
    idx, P, NLL, LO = [], [], [], []
    for f in range(cfg["n_folds"]):
        z = np.load(d / f"{prefix}_K{K}_wc{wc}_s{seed}_f{f}.npz")
        idx.append(z["idx"]); P.append(z["p"]); NLL.append(z["nll"])
        if "logodds" in z:
            LO.append(z["logodds"])
        else:                                   # log-odds from the saved condition head
            sd = torch.load(d / f"{prefix}_K{K}_wc{wc}_s{seed}_f{f}.pt", map_location="cpu")
            W, b = sd["cond_head.weight"].numpy(), sd["cond_head.bias"].numpy()
            LO.append(z["p"] @ (W[1] - W[0]) + (b[1] - b[0]))
    idx = np.concatenate(idx); o = np.argsort(idx)
    return idx[o], np.concatenate(P)[o], np.concatenate(NLL)[o], np.concatenate(LO)[o]


def evaluate(cfg, obs, prefix, K, wc, seeds, tag):
    per, Ps, scores = [], {}, {}
    inj = (obs.pt_state.astype(str) == "injured_PT").values
    for s in seeds:
        idx, P, nll, lo = load_heldout(cfg, prefix, K, wc, s)
        Ps[s] = P
        o = obs.iloc[idx]
        df = pd.DataFrame({"participant": o.participant.astype(str).values, "group": o.group.astype(str).values,
                           "lo": lo, "inj": inj[idx], "nll": nll})
        pm = df[df.inj].groupby("participant").agg(group=("group", "first"), lo=("lo", "mean"), n=("lo", "size"))
        pm = pm[pm.n >= cfg["cv"]["min_injured_cells"]]
        scores[s] = pm.lo
        pp = np.clip(P, 1e-12, 1)
        per.append({"auc_dkd_hkd_logodds": auc(pm[pm.group == "DKD"].lo, pm[pm.group == "HKD"].lo),
                    "nll": df.groupby("participant").nll.mean().mean(),
                    "entropy_over_logK": float((-(pp * np.log(pp)).sum(1)).mean() / np.log(K)),
                    "frac_top1_gt_0.5": float((P.max(1) > 0.5).mean())})
        groups = pm.group
    ari, rr = [], []
    for a, b in itertools.combinations(seeds, 2):
        ari.append(adjusted_rand_score(Ps[a].argmax(1), Ps[b].argmax(1)))
        rr.append(np.mean(match_experts(Ps[a], Ps[b])[1]))
    S = pd.DataFrame(scores)
    sa = seed_agreement(S)
    mean_score = S.mean(1)
    res = {"run": tag, "seeds": len(seeds), **pd.DataFrame(per).mean().to_dict(),
           "seed_ari": float(np.mean(ari)), "seed_matched_r": float(np.mean(rr)),
           "logodds_icc_2_1": sa["icc_2_1"], "logodds_mean_pairwise_spearman": sa["mean_pairwise_spearman"],
           "auc_dkd_hkd_seedmean_logodds": auc(mean_score[groups.reindex(S.index) == "DKD"],
                                              mean_score[groups.reindex(S.index) == "HKD"])}
    return res, S.assign(group=groups.reindex(S.index))


def final_compare(cfg, K):
    """Pairwise seed ARI / matched r of the final (all-KPMP) models on all KPMP cells, random vs warm-start initialisation."""
    rows = []
    for tag, d, stem in [("final_random_init", "final", f"final_K{K}_s{{s}}_kpmp.npz"), ("final_warm_start", "finalw", f"finalw_K{K}_s{{s}}_kpmp.npz")]:
        fs = [Path(cfg["paths"]["ckpt_dir"]) / d / stem.format(s=s) for s in cfg["final"]["seeds"]]
        if not all(f.exists() for f in fs):
            continue
        P = [np.load(f)["p"] for f in fs]
        ari = [adjusted_rand_score(P[a].argmax(1), P[b].argmax(1)) for a, b in itertools.combinations(range(len(P)), 2)]
        rr = [np.mean(match_experts(P[a], P[b])[1]) for a, b in itertools.combinations(range(len(P)), 2)]
        rows.append({"run": tag, "seeds": len(P), "seed_ari": float(np.mean(ari)), "seed_matched_r": float(np.mean(rr))})
    return rows


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_stability"); st = cfg["readout_stability"]["stability"]
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    obs = ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs
    rows = []
    b, Sb = evaluate(cfg, obs, "cv", K, wc, cfg["cv"]["seeds_for_stability"], "random_init_seeds1-2")
    a2, _ = evaluate(cfg, obs, st["prefix"], K, wc, cfg["cv"]["seeds_for_stability"], "warm_start_seeds1-2")
    a5, Sa = evaluate(cfg, obs, st["prefix"], K, wc, st["cv_seeds"], "warm_start_seeds1-5")
    tab = pd.DataFrame([b, a2, a5] + final_compare(cfg, K)); tab.to_csv(out / "random_vs_warm.tsv", sep="\t", index=False)
    adopt = bool(a2["seed_ari"] > b["seed_ari"] and a2["seed_matched_r"] > b["seed_matched_r"]
                 and a2["auc_dkd_hkd_logodds"] > b["auc_dkd_hkd_logodds"] - st["max_auc_drop"])
    Sa.to_csv(out / "heldout_logodds_by_seed_warm_start.tsv", sep="\t"); Sb.to_csv(out / "heldout_logodds_by_seed_random_init.tsv", sep="\t")
    dec = {"adopt_stabilised": adopt, "rule": "ARI and matched r both improve (seeds 1-2) and AUC drop < "
           f"{st['max_auc_drop']}", "random_init": b, "warm_start_seeds1-2": a2, "warm_start_seeds1-5": a5}
    (out / "initialization.json").write_text(json.dumps(dec, indent=2, default=float))
    print(tab.round(4).T.to_string()); print("adopt:", adopt)
    write_provenance("expr_stability", [], [out / "random_vs_warm.tsv", out / "initialization.json"], cfg["seed"], cfg,
                     extra={"adopt": adopt}, tag="compare")


if __name__ == "__main__":
    main()
