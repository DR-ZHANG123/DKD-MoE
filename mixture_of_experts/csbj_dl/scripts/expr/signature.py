"""Consensus gene ranking of the DKD-vs-HKD readout and the fixed external signature; pathway enrichment.

IG target: condition-head DKD-vs-HKD log-odds (linear in routing probabilities) on KPMP injured-PT cells;
baseline = mean healthy-PT profile. Per seed attributions are z-scored over genes and averaged across seeds.
Signature = top n_up / bottom n_down genes (sizes fixed in expr.yaml before any scoring).
Pathways: preranked GSEA (gseapy) on the consensus ranking against Hallmark + Reactome (+ two GO BP
glucocorticoid sets named in the config). Outputs: results/expr_signature/*
"""
from __future__ import annotations
import itertools, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, result_dir, write_provenance, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from model import log_norm, cond_logodds  # noqa: E402
from interpret_common import integrated_gradients, marker_sets, overlap_tests  # noqa: E402
from readout_common import final_set, load_model  # noqa: E402


def read_gmt(path, keep=None):
    out = {}
    for line in open(path):
        f = line.rstrip("\n").split("\t")
        if keep is None or f[0] in keep:
            out[f[0]] = f[2:]
    return out


def main():
    cfg = load_cfg("expr"); set_global_seed(cfg["seed"]); dev = cfg["train"]["device"]
    sc = cfg["readout_stability"]["signature"]; pw = cfg["readout_stability"]["pathways"]
    out = result_dir("expr_signature"); bulk = Path(cfg["paths"]["bulk_dir"]); fs = final_set(cfg)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    st = data.obs.pt_state.astype(str).values
    rng = np.random.default_rng(cfg["seed"])
    cells = np.sort(rng.choice(np.where(st == "injured_PT")[0], sc["ig_cells"], replace=False))
    healthy = np.where(st == "healthy_PT")[0]
    with torch.no_grad():
        x0 = torch.cat([log_norm(data.rows(healthy[s:s + 8192])) for s in range(0, len(healthy), 8192)]).mean(0, keepdim=True)
        xl = log_norm(data.rows(cells))
    per = {}
    for s in fs["seeds"]:
        model, genes = load_model(cfg, fs, s, dev)
        fn = lambda z: cond_logodds(model, model.gate_probs(model.encode_h(z))[0])
        a = integrated_gradients(fn, xl, x0, cfg["interpret"]["ig_steps"]).mean(0).cpu().numpy()
        per[s] = (a - a.mean()) / a.std()
        print("seed", s, flush=True)
    A = pd.DataFrame(per, index=genes); A.index.name = "gene"
    A["consensus_z"] = A[fs["seeds"]].mean(1)
    A["n_seeds_positive"] = (A[fs["seeds"]] > 0).sum(1)
    A = A.sort_values("consensus_z", ascending=False)
    A.to_csv(out / "consensus_ig_logodds.tsv", sep="\t")
    agree = float(np.mean([A[a].corr(A[b], method="spearman") for a, b in itertools.combinations(fs["seeds"], 2)]))
    sig = pd.concat([pd.DataFrame({"gene": A.index[:sc["n_up"]], "direction": "up"}),
                     pd.DataFrame({"gene": A.index[::-1][:sc["n_down"]], "direction": "down"})])
    sig["consensus_z"] = A.consensus_z.reindex(sig.gene).values
    sig.to_csv(out / "signature.tsv", sep="\t", index=False)
    ov = overlap_tests(A.consensus_z, marker_sets(cfg), 100, "consensus_logodds")
    ov.to_csv(out / "marker_overlap.tsv", sep="\t", index=False)
    # preranked GSEA
    import gseapy as gp
    ref = ROOT / "data/raw/reference"
    sets = {}
    for lib in pw["libraries"]:
        sets.update(read_gmt(ref / lib))
    sets.update(read_gmt(ref / "c5.go.bp.v2024.1.Hs.symbols.gmt", set(pw["extra_sets_from_gobp"])))
    rnk = A.consensus_z.reset_index(); rnk.columns = [0, 1]
    pre = gp.prerank(rnk=rnk, gene_sets=sets, min_size=pw["min_size"], max_size=pw["max_size"], permutation_num=1000,
                     seed=cfg["seed"], outdir=None, threads=8, verbose=False)
    res = pre.res2d[["Term", "ES", "NES", "NOM p-val", "FDR q-val", "Tag %", "Lead_genes"]].copy()
    res["library"] = res.Term.str.split("_").str[0]
    res = res.sort_values("NES", ascending=False)
    res.to_csv(out / "gsea_prerank.tsv", sep="\t", index=False)
    summ = {"final_set": str(fs["dir"]), "stabilised": fs["stabilised"], "seed_agreement_spearman": agree,
            "n_up": sc["n_up"], "n_down": sc["n_down"], "gseapy": gp.__version__, "n_sets_tested": int(len(res))}
    pd.Series(summ).to_csv(out / "summary.tsv", sep="\t", header=False)
    print(summ); print("up:", sig[sig.direction == "up"].gene.head(25).tolist()); print("down:", sig[sig.direction == "down"].gene.head(25).tolist())
    r = res.copy(); r["NES"] = r.NES.astype(float); r["FDR q-val"] = r["FDR q-val"].astype(float)
    print(r[r["FDR q-val"] < 0.25].head(20)[["Term", "NES", "FDR q-val"]].to_string())
    print(r[r["FDR q-val"] < 0.25].tail(20)[["Term", "NES", "FDR q-val"]].to_string())
    write_provenance("expr_signature", [fs["dir"] / (fs["stem"].format(s=s) + ".pt") for s in fs["seeds"]],
                     [out / "consensus_ig_logodds.tsv", out / "signature.tsv", out / "gsea_prerank.tsv"], cfg["seed"], cfg,
                     extra=summ)


if __name__ == "__main__":
    main()
