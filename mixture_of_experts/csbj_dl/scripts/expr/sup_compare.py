"""Cell-level, injured-only and participant-level supervision compared on held-out KPMP participants,
in the training participants (generalisation gap) and in the external cohorts.

Per supervision (final procedure = cvw/finalw; injured = cvinj/finalinj; patient = cvpat/finalpat):
  held-out readout (participant x technology means of injured-PT log-odds, mean over technologies, mean over
  seeds), DKD-vs-HKD AUC with stratified participant bootstrap and the paired AUC difference from cell-level
  supervision; Hedges g for DKD vs HKD, DM_noCKD vs HKD, HKD_withDM vs HKD; seed agreement (ICC(2,1)); seed ARI
  of top-1 expert and Hungarian-matched routing correlation on held-out cells;
  training-participant AUC of the all-KPMP models (in-sample) and routing concentration of injured DKD / HKD
  cells (largest between-diagnosis difference in mean routing probability of one expert), in-sample vs held-out;
  external donors: GSE211785 DKD vs HKD / control, GSE195460 DKD vs control.
Outputs: results/expr_supervision/{summary.tsv, group_tests.tsv, heldout_readouts.tsv, external_tests.tsv}
"""
from __future__ import annotations
import itertools, json, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "bench"))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import auc, hedges_g, match_experts  # noqa: E402
from stability_compare import load_heldout, seed_agreement  # noqa: E402
from readout import part_means  # noqa: E402
from harm_summary import paired_auc  # noqa: E402

COMPS = [("DKD", "HKD"), ("DM_noCKD", "HKD"), ("HKD_withDM", "HKD"), ("DKD", "DM_noCKD")]
EXT = {"gse211785": [("DKD", "HKD"), ("DKD", "Control")], "gse195460": [("DKD", "Control")]}


def boot_auc(s, grp, n, seed):
    pos, neg = s[grp == "DKD"].values, s[grp == "HKD"].values
    rng = np.random.default_rng(seed)
    b = [auc(rng.choice(pos, len(pos)), rng.choice(neg, len(neg))) for _ in range(n)]
    return np.quantile(b, 0.025), np.quantile(b, 0.975)


def concentration(P, obs_sub):
    inj = obs_sub.pt_state.astype(str).values == "injured_PT"; g = obs_sub.group.astype(str).values
    d, h = P[inj & (g == "DKD")].mean(0), P[inj & (g == "HKD")].mean(0)
    return float(np.max(np.abs(d - h)))


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_supervision"); sv = cfg["supervision_variants"]
    st = cfg["readout_stability"]["stability"]; mc = cfg["prep"]["min_cells_per_participant"]
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    ck = Path(cfg["paths"]["ckpt_dir"]); bulk = Path(cfg["paths"]["bulk_dir"])
    obs = {c: ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad", backed="r").obs.copy() for c in ("kpmp", "gse211785", "gse195460")}
    runs = {"cell": (st["prefix"], "finalw"), "injured": (sv["cv_prefix"]["injured"], sv["final_prefix"]["injured"]),
            "patient": (sv["cv_prefix"]["patient"], sv["final_prefix"]["patient"])}
    seeds = st["cv_seeds"]; summ, tests, reads, ext = [], [], {}, []
    for name, (cvp, fp) in runs.items():
        vals, Ps, conc_ho = {}, {}, []
        for s in seeds:
            idx, P, nll, lo = load_heldout(cfg, cvp, K, wc, s)
            v = np.full(len(obs["kpmp"]), np.nan); v[idx] = lo; vals[f"seed{s}"] = v; Ps[s] = P
            conc_ho.append(concentration(P, obs["kpmp"].iloc[idx]))
        pm = part_means(obs["kpmp"], vals, mc); pm = pm[pm.tech == "pooled"].set_index("participant")
        S = pm[list(vals)]; r = S.mean(1); grp = pm.group
        reads[name] = r
        lo_, hi_ = boot_auc(r, grp, 2000, cfg["seed"])
        ari = [adjusted_rand_score(Ps[a].argmax(1), Ps[b].argmax(1)) for a, b in itertools.combinations(seeds, 2)]
        mr = [np.mean(match_experts(Ps[a], Ps[b])[1]) for a, b in itertools.combinations(seeds, 2)]
        # in-sample (all-KPMP models scored on their own training participants)
        fin = {f"seed{s}": np.load(ck / fp / f"{fp}_K{K}_s{s}_kpmp.npz")["logodds"] for s in seeds}
        fpm = part_means(obs["kpmp"], fin, mc); fpm = fpm[fpm.tech == "pooled"].set_index("participant")
        fr = fpm[list(fin)].mean(1)
        conc_in = np.mean([concentration(np.load(ck / fp / f"{fp}_K{K}_s{s}_kpmp.npz")["p"], obs["kpmp"]) for s in seeds])
        summ.append({"supervision": name, "n_DKD": int((grp == "DKD").sum()), "n_HKD": int((grp == "HKD").sum()),
                     "auc_heldout": auc(r[grp == "DKD"], r[grp == "HKD"]), "auc_lo": lo_, "auc_hi": hi_,
                     "auc_per_seed_mean": np.mean([auc(S[c][grp == "DKD"], S[c][grp == "HKD"]) for c in S]),
                     "auc_in_sample": auc(fr[fpm.group == "DKD"], fr[fpm.group == "HKD"]),
                     "routing_gap_heldout": float(np.mean(conc_ho)), "routing_gap_in_sample": conc_in,
                     "icc_readout": seed_agreement(S.dropna())["icc_2_1"], "seed_ari": float(np.mean(ari)),
                     "seed_matched_r": float(np.mean(mr))})
        for a, b in COMPS:
            tests.append({"supervision": name, "comparison": f"{a}_vs_{b}", **hedges_g(r[grp == a], r[grp == b])})
        for c, comps in EXT.items():
            ev = {f"seed{s}": np.load(ck / fp / f"{fp}_K{K}_s{s}_{c}.npz")["logodds"] for s in seeds}
            e = part_means(obs[c], ev, mc); e = e[e.tech == "pooled"]; e["score"] = e[list(ev)].mean(1)
            for a, b in comps:
                ext.append({"supervision": name, "cohort": c, "comparison": f"{a}_vs_{b}",
                            **hedges_g(e.score[e.group == a], e.score[e.group == b])})
        print("done", name, flush=True)
    R = pd.DataFrame(reads); R.index.name = "participant"
    grp = obs["kpmp"].groupby("participant", observed=True).group.first().astype(str).reindex(R.index)
    for name in ("injured", "patient"):
        m = grp.isin(["DKD", "HKD"]) & R[name].notna() & R["cell"].notna()
        d0, lo, hi = paired_auc(R.cell[m], R[name][m], grp[m], 2000, 1)
        row = next(x for x in summ if x["supervision"] == name)
        row.update({"auc_diff_vs_cell": d0, "auc_diff_lo": lo, "auc_diff_hi": hi,
                    "spearman_with_cell_readout": float(R[name].corr(R.cell, method="spearman"))})
    Sm = pd.DataFrame(summ); T = pd.DataFrame(tests); E = pd.DataFrame(ext)
    Sm.to_csv(out / "summary.tsv", sep="\t", index=False); T.to_csv(out / "group_tests.tsv", sep="\t", index=False)
    E.to_csv(out / "external_tests.tsv", sep="\t", index=False)
    R.rename(columns=lambda c: f"sup_{c}").drop(columns="sup_cell").to_csv(out / "heldout_readouts.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(Sm.round(3).T.to_string()); print(T.round(3).to_string()); print(E.round(3).to_string())
    write_provenance("expr_supervision", [], [out / f for f in ("summary.tsv", "group_tests.tsv", "external_tests.tsv",
                                                                 "heldout_readouts.tsv")], cfg["seed"], cfg,
                     extra={"runs": runs, "K": K, "w_cond": wc})


if __name__ == "__main__":
    main()
