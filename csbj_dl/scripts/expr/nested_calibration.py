"""Nested cross-validation: one evaluation protocol with three summaries of discrimination.

Outer-part participant scores of the nested and of the reported pipelines (results/expr_nested_cv/
outer_participant_scores.tsv) are summarized as
  mean_part_auc       mean of the five per-part AUCs (no comparison of scores across parts)
  pooled_raw_auc      AUC of all held-out participants pooled without rescaling
  pooled_train_z_auc  AUC after standardizing each part's held-out scores with the mean and SD of the same models'
                      scores on that part's training participants (all groups, no diagnoses used), so that every
                      scaling parameter comes from training data only
Training-participant scores are computed with the saved outer models (seed-mean participant readout over
injured-PT cells, at most cap cells per participant x technology). Participant bootstrap stratified by part and
diagnosis. Fold 0 sensitivity: the summaries without the part on which the load-balance and discriminator weights
were compared. The reported pipeline's training-standardized held-out readout is also written for downstream use.
Outputs: results/expr_nested_cv/{protocol_summary.tsv, train_scale.tsv, reported_readout_train_z.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from train import build_model  # noqa: E402
from readout import part_means  # noqa: E402
from nested_outer import outer_name, reported_setting  # noqa: E402
from transfer_common import cell_logodds  # noqa: E402
from stats_utils import auc  # noqa: E402

CAP = 300


def train_scale(cfg, data, path_fn, K, seeds, fold, rng):
    obs = data.obs; inj = (obs.pt_state.astype(str) == "injured_PT").values
    tr = np.where((obs.fold.values.astype(int) != fold) & inj)[0]
    o = obs.iloc[tr][["participant", "tech"]].astype(str).reset_index(drop=True)
    keep = np.concatenate([rng.choice(d.index.values, min(CAP, len(d)), replace=False) for _, d in o.groupby(["participant", "tech"])])
    rows = np.sort(tr[keep])
    vals = {}
    for s in seeds:
        m = build_model(cfg, data.X.shape[1], K).to(data.device)
        sd = torch.load(path_fn(s), map_location=data.device); m.load_state_dict(sd.get("state_dict", sd)); m.eval()
        v = np.full(len(obs), np.nan); v[rows] = cell_logodds(m, data, rows); vals[f"s{s}"] = v
    pm = part_means(obs, vals, cfg["prep"]["min_cells_per_participant"]); pm = pm[pm.tech == "pooled"]
    r = pm[list(vals)].mean(1).dropna()
    return float(r.mean()), float(r.std()), int(len(r))


def summaries(P, col, rng, n_boot):
    def stat(Q):
        z = Q[Q.group.isin(["DKD", "HKD"])]
        mpa = np.nanmean([auc(q[col][q.group == "DKD"], q[col][q.group == "HKD"]) for _, q in z.groupby("fold")])
        raw = auc(z[col.replace("_tz", "")][z.group == "DKD"], z[col.replace("_tz", "")][z.group == "HKD"]) if col.endswith("_tz") else np.nan
        return mpa, auc(z[col][z.group == "DKD"], z[col][z.group == "HKD"])
    est = stat(P); bs = []
    strata = [d.index.values for _, d in P[P.group.isin(["DKD", "HKD"])].groupby(["fold", "group"])]
    for _ in range(n_boot):
        idx = np.concatenate([rng.choice(s, len(s)) for s in strata]); bs.append(stat(P.loc[idx]))
    bs = np.array(bs)
    return est, np.quantile(bs, [0.025, 0.975], axis=0), bs


def main():
    cfg = load_cfg("expr"); nc = cfg["nested_cv"]; set_global_seed(cfg["seed"]); rng = np.random.default_rng(cfg["seed"])
    out = result_dir("expr_nested_cv"); ck = Path(cfg["paths"]["ckpt_dir"])
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, cfg["train"]["device"])
    P = pd.read_csv(out / "outer_participant_scores.tsv", sep="\t", dtype={"participant": str})
    sel = pd.read_csv(out / "selection.tsv", sep="\t").set_index("outer_fold")
    Kr, wr = reported_setting(); seeds = cfg["readout_stability"]["stability"]["cv_seeds"]; pre = cfg["readout_stability"]["stability"]["prefix"]
    sc = []
    for f in sorted(P.fold.unique()):
        K, w = int(sel.loc[f, "K"]), float(sel.loc[f, "w_cond"])
        nested_path = lambda s: (ck / nc["outer_prefix"] / f"{outer_name(K, w, s, f)}.pt")
        if not nested_path(seeds[0]).exists():          # the selected setting equals the final one: those partition models apply
            nested_path = lambda s: ck / pre / f"{pre}_K{Kr}_wc{wr}_s{s}_f{f}.pt"
        for name, fn, KK in (("nested", nested_path, K), ("reported", lambda s: ck / pre / f"{pre}_K{Kr}_wc{wr}_s{s}_f{f}.pt", Kr)):
            mu, sd, n = train_scale(cfg, data, fn, KK, seeds, f, rng)
            sc.append({"fold": f, "pipeline": name, "K": KK, "train_mean": mu, "train_sd": sd, "n_train_participants": n})
        print("part", f, flush=True)
    S = pd.DataFrame(sc); S.to_csv(out / "train_scale.tsv", sep="\t", index=False)
    for name in ("nested", "reported"):
        s = S[S.pipeline == name].set_index("fold")
        P[f"{name}_tz"] = (P[name] - P.fold.map(s.train_mean)) / P.fold.map(s.train_sd)
    rows = []
    for subset, Q in (("all parts", P), ("parts 1-4", P[P.fold != 0])):
        Q = Q.reset_index(drop=True)
        for name in ("nested", "reported"):
            (mpa, ptz), ci, _ = summaries(Q, f"{name}_tz", rng, nc["bootstrap_reps"] // 4)
            zz = Q[Q.group.isin(["DKD", "HKD"])]
            raw = auc(zz[name][zz.group == "DKD"], zz[name][zz.group == "HKD"])
            rows += [{"subset": subset, "pipeline": name, "summary": "mean_part_auc", "auc": mpa, "ci_low": ci[0, 0], "ci_high": ci[1, 0]},
                     {"subset": subset, "pipeline": name, "summary": "pooled_train_z_auc", "auc": ptz, "ci_low": ci[0, 1], "ci_high": ci[1, 1]},
                     {"subset": subset, "pipeline": name, "summary": "pooled_raw_auc", "auc": raw}]
    R = pd.DataFrame(rows); R.to_csv(out / "protocol_summary.tsv", sep="\t", index=False)
    P[["participant", "group", "fold", "reported", "reported_tz"]].to_csv(out / "reported_readout_train_z.tsv", sep="\t", index=False)
    pd.set_option("display.width", 200); print(S.round(3).to_string()); print(R.round(3).to_string())
    write_provenance("expr_nested_cv", [out / "outer_participant_scores.tsv", out / "selection.tsv"],
                     [out / "protocol_summary.tsv", out / "train_scale.tsv", out / "reported_readout_train_z.tsv"],
                     cfg["seed"], cfg, tag="protocol")


if __name__ == "__main__":
    main()
