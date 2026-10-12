"""03c_multiome_merge: nucleus QC on both modalities and the merged paired AnnData files.

A nucleus is kept when RNA (genes, mitochondrial %) and ATAC (unique fragments, TSS enrichment,
nucleosome signal) all pass config `multiome`; libraries with < min_cells_library passing nuclei
are dropped. RNA and ATAC files hold exactly the same nuclei in the same order.
Outputs processed/dkd_moe/multiome_{rna,atac,atac_geneactivity}.h5ad and dkd_moe/results/data_multiome/
{library_qc.tsv, group_summary.tsv, qc_thresholds.tsv}."""
from __future__ import annotations
import os
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dcommon import cfg, result_dir, write_provenance, set_global_seed, clinical_obs, groups


def qc_mask(o: pd.DataFrame, M: dict) -> pd.DataFrame:
    return pd.DataFrame({
        "rna_genes": o.rna_genes.between(M["min_genes"], M["max_genes"]),
        "pct_mt": o.pct_mt <= M["max_pct_mt"],
        "n_fragments": o.n_fragments.between(M["min_frag"], M["max_frag"]),
        "tss_enrichment": o.tss_enrichment >= M["min_tss_enrichment"],
        "nucleosome_signal": o.nucleosome_signal <= M["max_nucleosome_signal"]}, index=o.index)


def load(lib: str, libdir: Path, M: dict):
    r = ad.read_h5ad(libdir / f"{lib}_rna.h5ad"); a = ad.read_h5ad(libdir / f"{lib}_atac.h5ad")
    g = ad.read_h5ad(libdir / f"{lib}_gact.h5ad")
    assert (r.obs_names == a.obs_names).all() and (r.obs_names == g.obs_names).all()
    q = qc_mask(r.obs, M); keep = q.all(1).values
    row = {"library": lib, "participant": r.obs.participant.iloc[0], "cells_cellranger": r.n_obs,
           **{f"pass_{k}": int(v.sum()) for k, v in q.items()}, "cells_qc": int(keep.sum())}
    if keep.any():
        o = r.obs[keep]
        row.update({"median_rna_genes": float(o.rna_genes.median()), "median_fragments": float(o.n_fragments.median()),
                    "median_tss_enrichment": float(o.tss_enrichment.median()), "median_frip": float(o.frip.median())})
    return r[keep].copy(), a[keep].copy(), row, g[keep].copy()


def main():
    c = cfg(); M = c["multiome"]; set_global_seed(c["seed"])
    out = result_dir("data_multiome")
    pdir = Path(c["processed_dir"]); libdir = pdir / "multiome_lib"
    L = pd.read_csv(out / "library_table.tsv", sep="\t")
    have = L.library.map(lambda l: all((libdir / f"{l}_{k}.h5ad").exists() for k in ("rna", "atac", "gact")))
    if (~have).any():
        print("not counted (see count_log.tsv):", list(L.library[~have]), flush=True)
    L = L[have]
    res = Parallel(n_jobs=16)(delayed(load)(lib, libdir, M) for lib in L.library)
    Q = pd.DataFrame([x[2] for x in res])
    Q["kept"] = Q.cells_qc >= M["min_cells_library"]
    keep = [i for i, k in enumerate(Q.kept) if k]
    # genes = Ensembl ids measured in every library (two CellRanger references: 2020-A and 2024-A); names = symbols
    # of the first 2020-A library (made unique), Ensembl id kept in var.gene_id
    genes = sorted(set.intersection(*[set(res[i][0].var_names) for i in keep]))
    R = ad.concat([res[i][0][:, genes] for i in keep], join="inner")
    ref = next(res[i][0] for i in keep if res[i][0].n_vars == min(res[j][0].n_vars for j in keep))
    R.var["gene_id"] = R.var_names.values
    R.var["symbol"] = ref.var.symbol.reindex(R.var_names).values
    R.var_names = pd.Index(R.var.symbol.astype(str).values); R.var_names_make_unique(); R.var.index.name = None
    A = ad.concat([res[i][1] for i in keep], join="inner")
    GA = ad.concat([res[i][3] for i in keep], join="inner")
    del res
    G = groups()
    for X in (R, A, GA):
        clin = clinical_obs(X.obs.participant.values, G)
        for col in clin.columns:
            X.obs[col] = clin[col].values
        X.obs["modality_pair"] = "10x Multiome"
    Q = Q.merge(L[["library", "group", "file_name"]], on="library", how="left")
    Q.to_csv(out / "library_qc.tsv", sep="\t", index=False)
    S = R.obs.groupby("group").agg(participants=("participant", "nunique"), libraries=("library", "nunique"),
                                   nuclei=("library", "size")).reset_index()
    S.to_csv(out / "group_summary.tsv", sep="\t", index=False)
    pd.Series({k: v for k, v in M.items() if not isinstance(v, (list, dict))}).to_csv(out / "qc_thresholds.tsv", sep="\t", header=False)
    fr, fa, fg = pdir / "multiome_rna.h5ad", pdir / "multiome_atac.h5ad", pdir / "multiome_atac_geneactivity.h5ad"
    for X, f in ((R, fr), (A, fa), (GA, fg)):          # atomic replace: readers of an older build keep a valid file
        tmp = f.with_suffix(".tmp.h5ad"); X.write_h5ad(tmp, compression="gzip"); os.replace(tmp, f)
    print("RNA", R.shape, "ATAC", A.shape, "nnz ATAC", A.X.nnz, flush=True)
    print(S.to_string(index=False), flush=True)
    write_provenance("data_multiome", [out / "library_table.tsv", out / "consensus_peaks.tsv.gz", out / "count_log.tsv"],
                     [fr, fa, fg, out / "library_qc.tsv", out / "group_summary.tsv", out / "qc_thresholds.tsv"], c["seed"], cfg=c,
                     extra={"step": "03c", "n_nuclei": R.n_obs, "n_genes": R.n_vars, "n_peaks": A.n_vars,
                            "n_libraries_kept": int(Q.kept.sum()), "n_libraries_dropped": int((~Q.kept).sum()),
                            "median_fragments": float(R.obs.n_fragments.median()),
                            "median_tss_enrichment": float(R.obs.tss_enrichment.median()),
                            "median_frip": float(R.obs.frip.median()), "median_rna_genes": float(R.obs.rna_genes.median())})


if __name__ == "__main__":
    main()
