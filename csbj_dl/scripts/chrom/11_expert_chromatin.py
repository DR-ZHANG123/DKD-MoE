"""11_expert_chromatin: chromatin-side scores of the HKD-weighted (M1) and DKD-weighted (M3) expert contributions.

For every part f and seed s, the partition model of the expression module (its own part held out) gives the
routing probabilities of each Multiome PT nucleus; the contributions of the experts assigned to M1 and M3
(results/expr_heldout_experts/matching.tsv) are head weight x routing probability. The saved peak aligner of the
same part and seed (04_align) embeds chromatin; ridge heads fitted on injured-PT nuclei of the training parts map
the chromatin embedding to each contribution and score the held-out nuclei. Seeds are averaged per nucleus.
Outputs: csbj_dl/results/chrom_expert/{nucleus_scores.tsv.gz, nucleus_eval.tsv} + PROVENANCE.json
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, P, result_dir, write_provenance, set_global_seed, pick_device, expr_model  # noqa: E402
from chrom_data import load_multiome, lsi_transform  # noqa: E402
from chrom_models import Aligner, embed, ridge_head, apply_head  # noqa: E402
from model import log_norm  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


@torch.no_grad()
def routing(m, X, device, bs=4096):
    out = []
    for s in range(0, X.shape[0], bs):
        x = torch.from_numpy(X[s:s + bs].toarray()).to(device)
        out.append(m.gate_probs(m.encode_h(log_norm(x)))[0].cpu().numpy())
    return np.concatenate(out)


def main():
    c = cfg(); E = c["expert"]; set_global_seed(c["seed"]); out = result_dir("chrom_expert")
    dev = pick_device(c["rna"]["device"]); ck = Path(c["paths"]["ckpt_dir"])
    hvg = pd.read_csv(P(c, "hvg"), sep="\t").gene.tolist()
    obs, R, Xp, _ = load_multiome(c, tuple(c["align"]["cells"]), rna_genes=hvg, peaks=True)
    match = pd.read_csv(ROOT / E["matching"], sep="\t")
    fold = obs.fold.values; inj = (obs.pt_state == "injured_PT").values
    acc = {k: {} for k in ["rna_" + m for m in E["components"]] + ["chrom_" + m for m in E["components"]]
           + ["input_" + m for m in E["components"]]}
    ev = []
    for f in range(c["n_folds"]):
        tr, te = np.where((fold != f) & ~pd.isna(fold))[0], np.where(fold == f)[0]
        if not len(te):
            continue
        L = dict(np.load(ck / f"lsi_f{f}.npz")); X = lsi_transform(Xp, L)
        trh = tr[inj[tr]]
        for s in c["rna"]["seeds"]:
            m, _ = expr_model(f, s, dev); p = routing(m, R, dev)
            w = (m.cond_head.weight[1] - m.cond_head.weight[0]).cpu().numpy(); del m
            row = match[(match.seed == s) & (match.fold == f)].iloc[0]
            d = torch.load(ck / f"peaks_f{f}_s{s}.pt", map_location=dev)
            al = Aligner(d["d_in"], 256, d["align_cfg"]).to(dev); al.load_state_dict(d["state_dict"]); al.eval()
            PA = embed(al, X, None, dev)[2]
            for comp in E["components"]:
                k = int(row[f"{comp}_expert"]); y = w[k] * p[:, k]
                h = ridge_head(PA[trh], y[trh], c["head"]["ridge_alpha"])
                hi = ridge_head(X[trh], y[trh], c["head"]["ridge_alpha"])
                for name, v in (("rna_", y[te]), ("chrom_", apply_head(PA[te], h)), ("input_", apply_head(X[te], hi))):
                    acc[name + comp].setdefault(s, np.full(len(obs), np.nan))[te] = v
                q = inj[te]
                ev.append({"fold": f, "seed": s, "component": comp, "n_injured_nuclei": int(q.sum()),
                           "pearson_chrom_vs_rna": float(np.corrcoef(apply_head(PA[te], h)[q], y[te][q])[0, 1])})
            print("part", f, "seed", s, flush=True)
            del PA; torch.cuda.empty_cache()
    S = pd.DataFrame({k: np.nanmean(list(v.values()), 0) for k, v in acc.items()}, index=obs.index)
    S = pd.concat([obs[["participant", "library", "group", "fold", "pt_state", "n_fragments", "tss_enrichment", "frip"]], S], axis=1)
    S.to_csv(out / "nucleus_scores.tsv.gz", sep="\t")
    V = pd.DataFrame(ev); V.to_csv(out / "nucleus_eval.tsv", sep="\t", index=False)
    print(V.groupby("component").pearson_chrom_vs_rna.describe().round(3))
    write_provenance("chrom_expert", [ROOT / E["matching"]] + sorted(ck.glob("peaks_f*_s*.pt")),
                     [out / "nucleus_scores.tsv.gz", out / "nucleus_eval.tsv"], c["seed"], c)


if __name__ == "__main__":
    main()
