"""Summarise participant-level cross-validation and select K / condition weight (rule in expr.yaml cv:)."""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import hedges_g, auc, match_experts  # noqa: E402


def load_obs(cfg):
    return ad.read_h5ad(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", backed="r").obs.copy()


def heldout(cfg, prefix, K, wc, seed):
    d = Path(cfg["paths"]["ckpt_dir"]) / prefix
    parts = [np.load(d / f"{prefix}_K{K}_wc{wc}_s{seed}_f{f}.npz") for f in range(cfg["n_folds"])]
    idx = np.concatenate([p["idx"] for p in parts])
    out = {k: np.concatenate([p[k] for p in parts]) for k in ("p", "nll", "cond_prob")}
    o = np.argsort(idx)
    return idx[o], {k: v[o] for k, v in out.items()}, parts


def metrics(cfg, obs, prefix, K, wc, seeds):
    classes = cfg["condition"]["classes"]; iD, iH = classes.index("DKD"), classes.index("HKD")
    res, P = {}, {}
    for s in seeds:
        idx, h, _ = heldout(cfg, prefix, K, wc, s)
        P[s] = h
        o = obs.iloc[idx].reset_index(drop=True)
        o["nll"] = h["nll"]
        res.setdefault("nll", []).append(o.groupby("participant").nll.mean().mean())
        inj = (o.pt_state == "injured_PT").values
        oi = o[inj].copy()
        cp = h["cond_prob"][inj]
        oi["pD"] = cp[:, iD] / (cp[:, iD] + cp[:, iH] + 1e-9)
        for k in range(K):
            oi[f"e{k}"] = h["p"][inj, k]
        pm = oi.groupby("participant").agg(group=("group", "first"), n=("pD", "size"),
                                            pD=("pD", "mean"), **{f"e{k}": (f"e{k}", "mean") for k in range(K)})
        pm = pm[pm.n >= cfg["cv"]["min_injured_cells"]]
        D, H = pm[pm.group == "DKD"], pm[pm.group == "HKD"]
        res.setdefault("auc_dkd_hkd", []).append(auc(D.pD, H.pD))
        gs = [hedges_g(D[f"e{k}"], H[f"e{k}"])["g"] for k in range(K)]
        res.setdefault("max_abs_g_expert", []).append(np.nanmax(np.abs(gs)))
        # per-cell specialisation (dense routing probabilities)
        pp = np.clip(h["p"], 1e-12, 1)
        res.setdefault("entropy_over_logK", []).append(float((-(pp * np.log(pp)).sum(1)).mean() / np.log(K)))
        res.setdefault("frac_top1_gt_0.5", []).append(float((h["p"].max(1) > 0.5).mean()))
        res.setdefault("mean_top1_weight", []).append(float(h["p"].max(1).mean()))
        hea = (o.pt_state == "healthy_PT").values
        res.setdefault("tv_healthy_vs_injured", []).append(
            float(0.5 * np.abs(h["p"][hea].mean(0) - h["p"][inj].mean(0)).sum()))
        gi = o.group.values
        res.setdefault("tv_dkd_vs_hkd_injured", []).append(float(0.5 * np.abs(
            h["p"][inj & (gi == "DKD")].mean(0) - h["p"][inj & (gi == "HKD")].mean(0)).sum()))
        tq = o.tech.values
        res.setdefault("tv_sc_vs_sn_healthy", []).append(float(0.5 * np.abs(
            h["p"][hea & (tq == "sc")].mean(0) - h["p"][hea & (tq == "sn")].mean(0)).sum()))
        share = np.bincount(h["p"].argmax(1), minlength=K) / len(h["p"])
        res.setdefault("min_top1_share", []).append(share.min())
        res.setdefault("max_top1_share", []).append(share.max())
        res.setdefault("n_dkd", []).append(len(D)); res.setdefault("n_hkd", []).append(len(H))
    if len(seeds) >= 2:
        ari, rr = [], []
        for a, b in itertools.combinations(seeds, 2):
            ari.append(adjusted_rand_score(P[a]["p"].argmax(1), P[b]["p"].argmax(1)))
            _, r = match_experts(P[a]["p"], P[b]["p"]); rr.append(np.mean(r))
        res["stability_ari"] = [np.mean(ari)]; res["stability_matched_r"] = [np.mean(rr)]
    return {k: float(np.mean(v)) for k, v in res.items()}


def usage_table(cfg, obs, K, wc, seed):
    """Mean held-out routing per expert by PT state x group x technology (one seed)."""
    idx, h, _ = heldout(cfg, "cv", K, wc, seed)
    o = obs.iloc[idx][["pt_state", "group", "tech"]].astype(str).reset_index(drop=True)
    df = pd.concat([o, pd.DataFrame(h["p"], columns=[f"expert_{k}" for k in range(K)])], axis=1)
    g = df.groupby(["pt_state", "group", "tech"])
    t = g.mean(); t["n_cells"] = g.size()
    return t.reset_index().assign(K=K, w_cond=wc, seed=seed)


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_cv")
    obs = load_obs(cfg)
    rows, usage = [], []
    for K, wc in itertools.product(cfg["cv"]["K_grid"], cfg["cv"]["w_cond_grid"]):
        try:
            m = metrics(cfg, obs, "cv", K, float(wc), cfg["cv"]["seeds_for_stability"])
        except FileNotFoundError as e:
            print("missing", K, wc, e); continue
        rows.append({"K": K, "w_cond": wc, **m})
        usage.append(usage_table(cfg, obs, K, float(wc), cfg["cv"]["seeds_for_stability"][0]))
    pd.concat(usage).to_csv(out / "routing_usage_by_state_group.tsv", sep="\t", index=False)
    tab = pd.DataFrame(rows)
    tab["eligible"] = ((tab["frac_top1_gt_0.5"] >= cfg["cv"]["min_frac_top1_gt_half"])
                       & (tab.entropy_over_logK <= cfg["cv"]["max_entropy_over_logK"]))
    tab["rank_nll"] = tab.nll.rank(ascending=True)
    tab["rank_stability"] = tab.stability_ari.rank(ascending=False)
    tab["rank_separation"] = tab.auc_dkd_hkd.rank(ascending=False)
    tab["rank_sum"] = tab[["rank_nll", "rank_stability", "rank_separation"]].sum(1)
    tab = tab.sort_values(["eligible", "rank_sum", "K"], ascending=[False, True, True]).reset_index(drop=True)
    tab.to_csv(out / "cv_table.tsv", sep="\t", index=False)
    best = tab.iloc[0]
    sel = {"K": int(best.K), "w_cond": float(best.w_cond), "rule": "among configurations with specific routing (expr.yaml cv eligibility), lowest rank sum of held-out NLL, "
           "seed stability (ARI) and held-out participant DKD vs HKD AUC; ties -> smaller K",
           "row": best.to_dict()}
    (out / "selection.json").write_text(json.dumps(sel, indent=2, default=float))
    print(tab.round(4).to_string()); print(json.dumps(sel, indent=1, default=float))
    curves = sorted((out / "curves").glob("cv_*.tsv"))
    pd.concat([pd.read_csv(c, sep="\t") for c in curves]).to_csv(out / "training_curves_cv.tsv", sep="\t", index=False)
    write_provenance("expr_cv", curves[:1], [out / "cv_table.tsv", out / "selection.json"], cfg["seed"], cfg,
                     extra={"n_curve_files": len(curves)})


if __name__ == "__main__":
    main()
