"""06_attrib: peak attributions for the chromatin-side readout (integrated gradients over TF-IDF peak inputs).

For every peak model (fold f, seed s): score(x) = head(MLP(LSI(x))) on the TF-IDF vector x of kept peaks.
Baseline x0 = participant-balanced mean injured-PT TF-IDF profile of HKD training participants; attributions
(x - x0) * mean-path gradient, averaged over injured-PT nuclei of DKD training participants (participant-balanced,
<= attrib.max_cells_per_participant each). The LSI map is linear, so the path is integrated in LSI space and
mapped back to peaks exactly. Per model, attributions are z-scored over peaks; consensus = mean z over models;
sign consistency = share of models agreeing with the consensus sign. Nearest protein-coding TSS (GENCODE v32).
Outputs: csbj_dl/results/chrom_attrib/{peak_attributions.tsv.gz, top_peaks.tsv, top_genes.tsv, summary.json}
"""
from __future__ import annotations
import gzip, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, P, result_dir, write_provenance, set_global_seed, pick_device  # noqa: E402
from chrom_data import load_multiome, tfidf  # noqa: E402
from chrom_models import Aligner  # noqa: E402


def balanced_mean(X, groups: np.ndarray) -> np.ndarray:
    return np.mean([np.asarray(X[groups == g].mean(0)).ravel() for g in np.unique(groups)], 0)


def sample_cells(part: np.ndarray, idx: np.ndarray, cap: int, rng) -> np.ndarray:
    return np.concatenate([rng.permutation(idx[part[idx] == p])[:cap] for p in np.unique(part[idx])])


def ig_model(L: dict, ck: dict, Xt, x0: np.ndarray, part: np.ndarray, steps: int, dev: str, bs: int = 256) -> np.ndarray:
    A = ck["align_cfg"]; m = Aligner(ck["d_in"], 256, A).to(dev)
    m.load_state_dict(ck["state_dict"]); m.eval()
    hd = ck["head"]
    V = torch.from_numpy(L["V"]).to(dev); mu_l = torch.from_numpy(L["mu"]).to(dev); sd_l = torch.from_numpy(L["sd"]).to(dev)
    hmu, hsd = torch.from_numpy(hd["mu"]).float().to(dev), torch.from_numpy(hd["sd"]).float().to(dev)
    coef = torch.from_numpy(hd["coef"]).float().to(dev)
    x0t = torch.from_numpy(x0.astype(np.float32)).to(dev)
    z0 = (x0t @ V - mu_l) / sd_l
    alphas = (torch.arange(steps, device=dev, dtype=torch.float32) + 0.5) / steps
    per_part = {}
    for s in range(0, Xt.shape[0], bs):
        xb = torch.from_numpy(Xt[s:s + bs].toarray()).to(dev)
        z = (xb @ V - mu_l) / sd_l
        g = torch.zeros_like(z)
        for a in alphas:
            za = (z0 + a * (z - z0)).detach().requires_grad_(True)
            y = (((m.atac(za) - hmu) / hsd) @ coef).sum()
            g += torch.autograd.grad(y, za)[0]
        g /= steps
        attr = (xb - x0t) * ((g / sd_l) @ V.T)                      # B x kept peaks
        pb = part[s:s + bs]
        for p in np.unique(pb):
            a_ = attr[torch.as_tensor(pb == p, device=dev)].sum(0)
            per_part[p] = per_part.get(p, 0) + a_.cpu().numpy()
    counts = pd.Series(part).value_counts()
    return np.mean([per_part[p] / counts[p] for p in per_part], 0)


def nearest_tss(peaks: pd.DataFrame, gtf: Path, max_kb: float) -> pd.DataFrame:
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene" or 'gene_type "protein_coding"' not in f[8]:
                continue
            rows.append((f[0], int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1, re.search(r'gene_name "([^"]+)"', f[8]).group(1)))
    T = pd.DataFrame(rows, columns=["chrom", "tss", "gene"]).sort_values(["chrom", "tss"])
    out = pd.DataFrame(index=peaks.index, columns=["nearest_gene", "tss_distance"])
    for ch, d in peaks.groupby("chrom"):
        t = T[T.chrom == ch]
        if t.empty:
            continue
        cen = ((d.start + d.end) // 2).values; ts = t.tss.values
        j = np.clip(np.searchsorted(ts, cen), 1, len(ts) - 1)
        left, right = cen - ts[j - 1], ts[j] - cen
        k = np.where(np.abs(left) <= np.abs(right), j - 1, j)
        dist = cen - ts[k]
        ok = np.abs(dist) <= max_kb * 1e3
        out.loc[d.index, "nearest_gene"] = np.where(ok, t.gene.values[k], None)
        out.loc[d.index, "tss_distance"] = np.where(ok, dist, np.nan)
    return out


def main():
    c = cfg(); At = c["attrib"]; set_global_seed(c["seed"]); out = result_dir("chrom_attrib")
    dev = pick_device(c["rna"]["device"]); ck = Path(c["paths"]["ckpt_dir"])
    obs, _, Xp, peak_names = load_multiome(c, ("injured_PT",), rna_genes=None, peaks=True)
    part = obs.participant.astype(str).values; grp = obs.group.astype(str).values; fold = obs.fold.values
    rng = np.random.default_rng(c["seed"])
    Z, raw, models = [], [], []
    fallback = False
    for f_pt in sorted(ck.glob("peaks_f*_s*.pt")):
        m = re.match(r"peaks_f(\d+)_s(\d+)", f_pt.stem); f, s = int(m.group(1)), int(m.group(2))
        L = dict(np.load(ck / f"lsi_f{f}.npz")); mk = torch.load(f_pt, map_location="cpu", weights_only=False)
        tr = np.where(fold != f)[0]
        base_idx, tgt_idx = tr[grp[tr] == "HKD"], tr[grp[tr] == "DKD"]
        if len(base_idx) == 0 or len(tgt_idx) == 0:          # only for incomplete Multiome sets
            fallback = True; base_idx, tgt_idx = tr, tr
        Xk = Xp[:, L["keep"]]
        x0 = balanced_mean(tfidf(Xk[base_idx], L["idf"]), part[base_idx])
        ti = sample_cells(part, tgt_idx, At["max_cells_per_participant"], rng)
        a = ig_model(L, mk, tfidf(Xk[ti], L["idf"]), x0, part[ti], At["ig_steps"], dev)
        full = np.zeros(len(peak_names), np.float32); full[L["keep"]] = a
        raw.append(full); Z.append((full - full.mean()) / (full.std() + 1e-12)); models.append(f"f{f}_s{s}")
        print("attributed", f_pt.stem, "nuclei", len(ti), flush=True)
    Z, raw = np.array(Z), np.array(raw)
    cons = Z.mean(0)
    C = pd.read_csv(P(c, "consensus_peaks"), sep="\t").set_index("peak").loc[peak_names]
    D = C[["chrom", "start", "end"]].copy()
    D["consensus_z"] = cons; D["mean_attribution"] = raw.mean(0)
    D["sign_consistency"] = (np.sign(Z) == np.sign(cons)[None]).mean(0)
    D["n_models"] = len(models)
    # descriptive accessibility of injured-PT nuclei, DKD vs HKD participants (participant-balanced fraction open)
    for g in ("DKD", "HKD"):
        ii = np.where(grp == g)[0]
        D[f"frac_open_{g}"] = balanced_mean(Xp[ii], part[ii]) if len(ii) else np.nan
    D = D.join(nearest_tss(D, P(c, "gtf"), At["nearest_gene_max_kb"]))
    D.index.name = "peak"; D.to_csv(out / "peak_attributions.tsv.gz", sep="\t")
    n = At["n_top"]
    top = pd.concat([D.nlargest(n, "consensus_z").assign(direction="DKD_side"),
                     D.nsmallest(n, "consensus_z").assign(direction="HKD_side")])
    top.to_csv(out / "top_peaks.tsv", sep="\t")
    G = (top.dropna(subset=["nearest_gene"]).groupby(["direction", "nearest_gene"])
         .agg(n_peaks=("consensus_z", "size"), sum_z=("consensus_z", "sum"), best_z=("consensus_z", lambda x: x.abs().max()),
              median_tss_distance=("tss_distance", "median")).reset_index()
         .sort_values(["direction", "n_peaks", "best_z"], ascending=[True, False, False]))
    G.to_csv(out / "top_genes.tsv", sep="\t", index=False)
    summ = {"models": models, "n_peaks": int(len(D)), "label_fallback_incomplete_multiome": fallback,
            "median_sign_consistency_top": float(top.sign_consistency.median()) if len(top) else None}
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ)); print(G.groupby("direction").head(15).to_string(index=False))
    write_provenance("chrom_attrib", [P(c, "multiome_atac"), P(c, "gtf")] + sorted(ck.glob("peaks_f*_s*.pt")),
                     [out / f for f in ("peak_attributions.tsv.gz", "top_peaks.tsv", "top_genes.tsv", "summary.json")],
                     c["seed"], c, extra=summ)


if __name__ == "__main__":
    main()
