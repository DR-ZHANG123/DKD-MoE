"""04_align: contrastive alignment of same-nucleus RNA and chromatin, and the chromatin-side readout head.

Per outer fold f (participants of fold f held out) and seed s:
  RNA side   fixed backbone output h of the cross-fitted expression model (fold f, seed s) -> projection
  chromatin  peaks: TF-IDF + LSI fitted on training-fold nuclei -> MLP -> projection
             gact : gene activity of expression HVGs (log CP10k, training z) -> MLP -> projection
  loss       symmetric InfoNCE, learnable temperature, every batch drawn from one library
  baseline   identical training on within-library shuffled pairs
  head       ridge from the chromatin embedding to the RNA-side DKD-vs-HKD log-odds of the same model,
             fitted on training-fold PT nuclei; applied to held-out nuclei and external snATAC PT nuclei
Per model: <ckpt>/chrom/<feature>_f<f>_s<s>.pt (weights, head, transform) and _out.npz (held-out and external
scores, retrieval). Aggregation and statistics: 05_chrom_readout.py.
Usage: python 04_align.py [--features peaks gact] [--folds 0 1] [--seeds 1 2] [--force]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, P, result_dir, set_global_seed, pick_device, expr_model  # noqa: E402
from chrom_data import (load_multiome, load_multiome_gact, fit_lsi, lsi_transform, fit_ga, ga_transform,  # noqa: E402
                        ga_genes, binarize, align_peaks)
from chrom_models import train_aligner, embed, retrieval, ridge_head, apply_head, rna_embedding  # noqa: E402


def externals(c: dict, peak_names: pd.Index, genes: list[str]) -> dict:
    """PT nuclei of the external snATAC cohorts in each feature space (labels from 02_ext_labels)."""
    d = result_dir("chrom_ext_labels"); ext = {}
    for cohort, feats in c["external"]["features"].items():
        lab = pd.read_csv(d / f"{cohort}_labels.tsv.gz", sep="\t", index_col=0)
        lab = lab[lab.pt_state.isin(["healthy_PT", "injured_PT"])]
        for f in feats:
            key = {"gse211785": {"gact": "gse211785_atac"}, "gse195460": {"gact": "gse195460_gact", "peaks": "gse195460_atac"}}[cohort][f]
            a = ad.read_h5ad(P(c, key))
            a = a[a.obs_names.isin(lab.index)]
            if f == "peaks":
                X = align_peaks(binarize(a.X), a.var_names, peak_names)
            else:
                X = a[:, genes].X
            ext[(cohort, f)] = {"X": X, "obs": lab.loc[a.obs_names]}
    return ext


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", nargs="*", default=None); ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    c = cfg(); A = c["align"]; set_global_seed(c["seed"])
    dev = pick_device(c["rna"]["device"]); print("device", dev, flush=True)
    ck = Path(c["paths"]["ckpt_dir"]); ck.mkdir(parents=True, exist_ok=True)
    feats = args.features or A["features"]; seeds = args.seeds or c["rna"]["seeds"]
    hvg = pd.read_csv(P(c, "hvg"), sep="\t").gene.tolist()
    obs, R, Xp, peak_names = load_multiome(c, tuple(A["cells"]), rna_genes=hvg, peaks="peaks" in feats)
    gg = ga_genes(c)
    Xg = load_multiome_gact(c, obs.index, gg).X if "gact" in feats else None
    ext = externals(c, peak_names, gg)
    logfrag = np.log10(obs.n_fragments.values.astype(float))
    lib = obs.library.astype(str).values; fold = obs.fold.values
    inj = (obs.pt_state == "injured_PT").values
    print("nuclei", len(obs), "participants", obs.participant.nunique(), "peaks", None if Xp is None else Xp.shape[1],
          "ga genes", len(gg), flush=True)
    folds = args.folds if args.folds is not None else range(c["n_folds"])
    for f in folds:
        tr, te = np.where((fold != f) & ~pd.isna(fold))[0], np.where(fold == f)[0]
        if len(te) == 0 or len(tr) == 0:
            print("fold", f, "skipped (no nuclei)", flush=True); continue
        # inner validation participants (early stopping only), drawn from the training folds
        tp = np.sort(obs.participant.values[tr].astype(str)); tp = np.unique(tp)
        vrng = np.random.default_rng(c["seed"] + f)
        vp = set(vrng.choice(tp, max(1, int(round(A["val_frac_participants"] * len(tp)))), replace=False)) if len(tp) > 2 else set()
        isv = obs.participant.isin(vp).values
        tr_fit, val = tr[~isv[tr]], tr[isv[tr]]
        trh = tr[np.isin(obs.pt_state.values[tr], c["head"]["train_cells"])]        # nuclei used to fit the heads
        XA = {}
        if "peaks" in feats:
            L = fit_lsi(Xp[tr], logfrag[tr], c, c["seed"] + f)
            np.savez_compressed(ck / f"lsi_f{f}.npz", **{k: v for k, v in L.items()})
            XA["peaks"] = (lsi_transform(Xp, L), {k: lsi_transform(v["X"], L) for k, v in ext.items() if k[1] == "peaks"})
            print(f"fold {f} LSI: peaks kept {L['n_peaks_kept']}, first component r(depth) {L['r_depth_first']:.2f} "
                  f"dropped {L['dropped_first']}", flush=True)
        if "gact" in feats:
            G = fit_ga(Xg[tr], c)
            np.savez_compressed(ck / f"ga_f{f}.npz", genes=np.array(gg), **G)
            XA["gact"] = (ga_transform(Xg, G, c), {k: ga_transform(v["X"], G, c) for k, v in ext.items() if k[1] == "gact"})
        for s in seeds:
            todo = [ft for ft in feats if args.force or not (ck / f"{ft}_f{f}_s{s}_out.npz").exists()]
            if not todo:
                continue
            m_expr, _ = expr_model(f, s, dev)
            H, LO = rna_embedding(m_expr, R, dev)
            del m_expr
            for ft in todo:
                t0 = time.time(); X, XE = XA[ft]
                res = {"te": te, "readout_te": LO[te]}
                rng = np.random.default_rng(c["seed"] + 100 * f + s)
                perm = np.arange(len(obs))
                for l in np.unique(lib):
                    ii = np.where(lib == l)[0]; perm[ii] = rng.permutation(ii)
                ret = {}
                for shuffled in (False, True):
                    m = train_aligner(X, H, lib, tr_fit, A, s, dev, pair_perm=perm if shuffled else None, val=val,
                                      log=(lambda msg: print(f"[{ft} f{f} s{s}{' shuf' if shuffled else ''}]{msg}", flush=True)))
                    ZA, ZR, PA, PR = embed(m, X, H, dev)
                    ret["shuffled" if shuffled else "aligned"] = {**retrieval(ZA, ZR, lib, te, A["retrieval_pool"], A["topk"],
                                                                             A["retrieval_repeats"], c["seed"] + s),
                                                                   "best_epoch": m.best_epoch}
                    if shuffled:
                        continue
                    hd = ridge_head(PA[trh], LO[trh], c["head"]["ridge_alpha"])
                    res["chrom_te"] = apply_head(PA[te], hd)
                    hr = ridge_head(PR[trh], LO[trh], c["head"]["ridge_alpha"])          # RNA-side ceiling
                    res["rnaproj_te"] = apply_head(PR[te], hr)
                    hi = ridge_head(X[trh], LO[trh], c["head"]["ridge_alpha"])           # chromatin input, no alignment
                    res["input_te"] = apply_head(X[te], hi)
                    for (cohort, fe), Xe in XE.items():
                        Ee = embed(m, Xe, None, dev)[2]
                        res[f"ext_{cohort}"] = apply_head(Ee, hd)
                        res[f"extinput_{cohort}"] = apply_head(Xe, hi)
                    torch.save({"state_dict": m.state_dict(), "head": hd, "feature": ft, "fold": f, "seed": s,
                                "d_in": X.shape[1], "align_cfg": A}, ck / f"{ft}_f{f}_s{s}.pt")
                res["retrieval"] = np.array(json.dumps(ret))
                np.savez_compressed(ck / f"{ft}_f{f}_s{s}_out.npz", **res)
                q = inj[te]
                r_ch = np.corrcoef(res["chrom_te"][q], LO[te][q])[0, 1] if q.sum() > 2 else np.nan
                print(f"[{ft} f{f} s{s}] {time.time() - t0:.0f}s retrieval aligned {ret['aligned']} shuffled {ret['shuffled']} "
                      f"| held-out injured nucleus r(chrom, rna readout) {r_ch:.3f}", flush=True)
            del H
            torch.cuda.empty_cache()
    obs_out = result_dir("chrom_align") / "nuclei.tsv.gz"
    obs[["participant", "library", "group", "fold", "pt_state", "n_fragments", "tss_enrichment", "frip"]].to_csv(obs_out, sep="\t")
    for (cohort, fe), v in ext.items():
        v["obs"].to_csv(result_dir("chrom_align") / f"ext_nuclei_{cohort}_{fe}.tsv.gz", sep="\t")


if __name__ == "__main__":
    main()
