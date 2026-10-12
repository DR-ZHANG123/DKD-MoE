"""Expert contributions and expert exchanges in held-out KPMP participants.

1. Matching: each cross-validation model (cvw, 5 seeds x 5 folds) is run on a common sample of KPMP cells; its
   experts are matched to the reproducible experts M1-M3 of the final models by the correlation of routing
   probabilities (Hungarian assignment). 
2. Contributions: in held-out participants, contribution of expert k = (DKD - HKD head weight) x routing
   probability, participant x technology means over injured-PT cells, mean over technologies and seeds.
3. Exchanges (held-out injured-PT cells): every ordered pair (src -> dst) of the model's experts; the change in
   decoded log expression is scored with the signature built without the held-out fold; participant means.
   Main: HKD cells M1 -> M3; DKD cells M3 -> M1 (sign reversed).
4. Supervised model without experts (same folds and seeds): held-out HKD injured cells, latent mean shifted along
   the condition-head direction by the distance between DKD and HKD injured-cell latent means of the training
   participants, versus random directions of the same length.
Outputs: results/expr_heldout_experts/{matching.tsv, contributions.tsv, contribution_tests.tsv, exchange_pairs.tsv,
         exchange_summary.tsv, latent_shift.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common")); sys.path.insert(0, str(HERE)); sys.path.insert(0, str(HERE.parents[0] / "bench"))
from cdl import load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from train import build_model  # noqa: E402
from readout import part_means  # noqa: E402
from stats_utils import hedges_g  # noqa: E402
from intervention_common import select_cells, run_intervention, participant_means  # noqa: E402
from intervention_stats import summarise  # noqa: E402

META = ["M1", "M2", "M3"]


def load_cv(cfg, K, path, dev):
    m = build_model(cfg, 3000, K).to(dev); m.load_state_dict(torch.load(path, map_location=dev)); m.eval()
    return m


def match(cfg, data, fs_dir, K, units, idx, m):
    """Expert index of model m for M1-M3 and the routing correlations."""
    ref = {}
    for me in META:
        cols = [np.load(fs_dir / f"finalw_K{K}_s{s}_kpmp.npz")["p"][idx, int(e)]
                for s, e in units[units.meta_expert == me][["seed", "expert"]].values]
        ref[me] = np.mean(cols, 0)
    P = forward_all(m, data, idx)["p"]
    C = np.array([[np.corrcoef(ref[me], P[:, k])[0, 1] for k in range(K)] for me in META])
    r, c = linear_sum_assignment(-np.nan_to_num(C))
    return {META[i]: int(c[j]) for j, i in enumerate(r)}, {META[i]: float(C[i, c[j]]) for j, i in enumerate(r)}


def sig_score(D: pd.DataFrame, up, down):
    u = [g for g in up if g in D.columns]; d = [g for g in down if g in D.columns]
    return D[u].mean(1) - D[d].mean(1)


@torch.no_grad()
def latent_shift(model, data, rows, direction, bs=4096):
    """Change in decoded log expression when the latent mean is shifted by `direction`."""
    out = []
    for s in range(0, len(rows), bs):
        r = rows[s:s + bs]; x = data.rows(r); t = data.tech[torch.as_tensor(r, device=data.device)]
        o = model(x, t, sample=False); toh = F.one_hot(t, model.n_tech).float(); lib = torch.ones(len(r), 1, device=x.device)
        a, _ = model.decode(o["mu"], toh, lib); b, _ = model.decode(o["mu"] + direction, toh, lib)
        out.append((torch.log(b.clamp_min(1e-30)) - torch.log(a.clamp_min(1e-30))).cpu())
    return torch.cat(out).numpy()


def main():
    cfg = load_cfg("expr"); ic = load_cfg("expert_intervention"); hc = ic["heldout"]; set_global_seed(ic["seed"])
    dev = cfg["train"]["device"]; out = result_dir("expr_heldout_experts")
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    ck = Path(cfg["paths"]["ckpt_dir"]); pre = cfg["readout_stability"]["stability"]["prefix"]; seeds = cfg["readout_stability"]["stability"]["cv_seeds"]
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, dev)
    units = pd.read_csv(result_dir("expr_meta") / "units.tsv", sep="\t")
    FS = pd.read_csv(result_dir("expr_transfer") / "fold_signatures.tsv", sep="\t"); FS = FS[FS.construction == hc["signature_construction"]]
    rng = np.random.default_rng(ic["seed"]); midx = np.sort(rng.choice(len(data), hc["match_cells"], replace=False))
    fold = data.obs.fold.values.astype(int); genes = data.genes
    match_rows, contrib, pairs, shifts = [], {me: {} for me in META + ["other"]}, [], []
    from sup_single import make_builder
    sup_build = make_builder("z"); bck = Path(load_cfg("bench")["paths"]["ckpt"]) / f"warm_{hc['no_expert_variant']}"
    for f in range(cfg["n_folds"]):
        up = FS.gene[(FS.fold == f) & (FS.direction == "up")].tolist(); down = FS.gene[(FS.fold == f) & (FS.direction == "down")].tolist()
        te_obs = data.obs.iloc[np.where(fold == f)[0]]
        cells = {}
        for g in ("DKD", "HKD"):
            c = select_cells(te_obs, g, ic["cap_cells_per_unit"], ic["min_cells_per_unit"], rng)
            c["row"] = np.where(fold == f)[0][c.row.values]; cells[g] = c
        for s in seeds:
            name = f"{pre}_K{K}_wc{wc}_s{s}_f{f}"
            m = load_cv(cfg, K, ck / pre / f"{name}.pt", dev)
            mp, mr = match(cfg, data, ck / "finalw", K, units, midx, m)
            match_rows.append({"seed": s, "fold": f, **{f"{k}_expert": v for k, v in mp.items()}, **{f"{k}_r": v for k, v in mr.items()}})
            z = np.load(ck / pre / f"{name}.npz"); w = (m.cond_head.weight[1] - m.cond_head.weight[0]).detach().cpu().numpy()
            v = {me: np.full(len(data), np.nan) for me in contrib}
            oth = [k for k in range(K) if k not in mp.values()]
            for me in META:
                v[me][z["idx"]] = w[mp[me]] * z["p"][:, mp[me]]
            v["other"][z["idx"]] = (z["p"][:, oth] * w[oth]).sum(1)
            for me in contrib:
                contrib[me][f"s{s}_f{f}"] = v[me]
            inv = {e: me for me, e in mp.items()}
            for g, c in cells.items():
                for a in range(K):
                    for b in range(K):
                        if a == b:
                            continue
                        r = run_intervention(m, data, c.row.values, a, b)
                        D = participant_means(c, r["delta"]); D.columns = genes
                        sc = sig_score(D, up, down)
                        pairs += [{"seed": s, "fold": f, "group": g, "src": inv.get(a, f"o{a}"), "dst": inv.get(b, f"o{b}"),
                                   "participant": p, "signature_change": val} for p, val in sc.items()]
            # supervised model without experts: latent shift along the condition-head direction vs random directions
            sm = sup_build(cfg, 3000, 1).to(dev)
            sm.load_state_dict(torch.load(bck / f"seed{s}_fold{f}" / "model.pt", map_location=dev)["state_dict"]); sm.eval()
            tr = np.where(fold != f)[0]; st = data.obs.pt_state.astype(str).values; gg = data.obs.group.astype(str).values
            mu = {}
            for g in ("DKD", "HKD"):
                rr = rng.choice(tr[(st[tr] == "injured_PT") & (gg[tr] == g)], 5000, replace=False)
                mu[g] = forward_all(sm, data, np.sort(rr))["z"].mean(0)
            length = float(np.linalg.norm(mu["DKD"] - mu["HKD"]))
            hw = (sm.cond_head.weight[1] - sm.cond_head.weight[0]).detach()
            dirs = {"head": hw / hw.norm() * length}
            gen = torch.Generator(device="cpu").manual_seed(ic["seed"] + 100 * f + s)
            for i in range(hc["n_random_directions"]):
                rd = torch.randn(hw.shape, generator=gen).to(dev); dirs[f"random{i}"] = rd / rd.norm() * length
            c = cells["HKD"]
            for dn, dv in dirs.items():
                D = participant_means(c, latent_shift(sm, data, c.row.values, dv)); D.columns = genes
                sc = sig_score(D, up, down)
                shifts += [{"seed": s, "fold": f, "direction": dn, "participant": p, "signature_change": val} for p, val in sc.items()]
            print("done", name, mp, {k: round(v, 2) for k, v in mr.items()}, flush=True)
    M = pd.DataFrame(match_rows); M.to_csv(out / "matching.tsv", sep="\t", index=False)
    mc = cfg["prep"]["min_cells_per_participant"]; C = []
    for me, vals in contrib.items():
        pm = part_means(data.obs, vals, mc); pm = pm[pm.tech == "pooled"].set_index("participant")
        C.append(pm[list(vals)].mean(1).rename(me))
    C = pd.concat(C, axis=1); C["group"] = data.obs.groupby("participant", observed=True).group.first().astype(str).reindex(C.index)
    C.to_csv(out / "contributions.tsv", sep="\t")
    CT = pd.DataFrame([{"expert": me, "comparison": f"{a}_vs_{b}", **hedges_g(C[me][C.group == a], C[me][C.group == b])}
                       for me in META + ["other"] for a, b in (("DKD", "HKD"), ("DM_noCKD", "HKD"), ("HKD_withDM", "HKD"), ("DKD", "DM_noCKD"))])
    CT.to_csv(out / "contribution_tests.tsv", sep="\t", index=False)
    Pp = pd.DataFrame(pairs); Pp.to_csv(out / "exchange_pairs.tsv", sep="\t", index=False)
    Sh = pd.DataFrame(shifts); Sh.to_csv(out / "latent_shift.tsv", sep="\t", index=False)
    brng = np.random.default_rng(ic["seed"] + 1); summ = []
    for (g, a, b), d in Pp.groupby(["group", "src", "dst"]):
        if not (a in META and b in META) and not (g == "HKD" and a == "M1") and not (g == "DKD" and a == "M3"):
            continue
        summ.append({"analysis": "exchange", "group": g, "exchange": f"{a}->{b}", **summarise(d.groupby("participant").signature_change.mean(), 2000, brng)})
    # rank of the main exchange among all ordered pairs of the same model (participant means first)
    for g, a, b in (("HKD", "M1", "M3"), ("DKD", "M3", "M1")):
        ranks = []
        for (s, f), d in Pp[Pp.group == g].groupby(["seed", "fold"]):
            pm = d.groupby(["src", "dst"]).signature_change.mean()
            val = pm.get((a, b), np.nan); sign = 1 if g == "HKD" else -1
            ranks.append(float((sign * pm >= sign * val).mean()))
        summ.append({"analysis": "rank_among_all_pairs", "group": g, "exchange": f"{a}->{b}", "mean": np.mean(ranks),
                     "median": np.median(ranks), "n_participants": len(ranks), "n_positive": int(sum(r <= 0.1 for r in ranks))})
    head = Sh[Sh.direction == "head"].groupby("participant").signature_change.mean()
    summ.append({"analysis": "no_expert_latent_shift", "group": "HKD", "exchange": "head_direction", **summarise(head, 2000, brng)})
    rnd = Sh[Sh.direction != "head"].groupby(["seed", "fold", "direction"]).signature_change.mean()
    hd = Sh[Sh.direction == "head"].groupby(["seed", "fold"]).signature_change.mean()
    pct = [float((rnd.loc[(s, f)] >= hd.loc[(s, f)]).mean()) for s, f in hd.index]
    summ.append({"analysis": "no_expert_rank_among_random", "group": "HKD", "exchange": "head_direction", "mean": np.mean(pct),
                 "median": np.median(pct), "n_participants": len(pct)})
    S = pd.DataFrame(summ); S.to_csv(out / "exchange_summary.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(M.describe().round(2).to_string()); print(CT.round(3).to_string()); print(S.round(4).to_string())
    write_provenance("expr_heldout_experts", [result_dir("expr_meta") / "units.tsv", result_dir("expr_transfer") / "fold_signatures.tsv"],
                     [out / f for f in ("matching.tsv", "contributions.tsv", "contribution_tests.tsv", "exchange_pairs.tsv",
                                        "exchange_summary.tsv", "latent_shift.tsv")], ic["seed"], cfg,
                     extra={"intervention_config_sha": ic["_sha"], "K": K})


if __name__ == "__main__":
    main()
