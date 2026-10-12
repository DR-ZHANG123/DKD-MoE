"""03b_multiome_count: per library, RNA counts plus ATAC consensus-peak and gene-activity counts for the
same nuclei, with ATAC QC (single pass over the fragment file; definitions in fragcount.py).

The gene-activity matrix (protein-coding gene body + 2 kb upstream, Signac definition) is the feature
space shared with the external GSE211785 snATAC data, which are public only as gene-activity counts.
Writes processed/dkd_moe/multiome_lib/<library>_{rna,atac,gact}.h5ad (no cell filtering; 03c filters).
Usage: python 03b_multiome_count.py [--only LIBRARY]"""
from __future__ import annotations
import sys, time
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dcommon import cfg, result_dir, write_provenance, set_global_seed
from mio import read_h5, rna_by_id, iter_fragments, ensure_gtf, tss_table
import fragcount as fc


def count_library_safe(*a) -> dict:
    try:
        return count_library(*a)
    except Exception as e:          # one unreadable archive must not stop the build; it is reported in count_log.tsv
        return {"library": a[0].library, "status": f"error: {type(e).__name__}: {str(e)[:150]}"}


def count_library(lib: pd.Series, C: pd.DataFrame, tss: dict, genemap, M: dict, outdir: Path) -> dict:
    t0 = time.time()
    fr, fa, fg = (outdir / f"{lib.library}_{k}.h5ad" for k in ("rna", "atac", "gact"))
    if "--rna-only" in sys.argv and fr.exists():      # refresh the RNA file (Ensembl-id keyed) without recounting ATAC
        o = ad.read_h5ad(fr, backed="r").obs.copy()
        R, rvar = rna_by_id(read_h5(Path(lib.zip)))
        ad.AnnData(X=R, obs=o, var=rvar).write_h5ad(fr, compression="gzip")
        return {"library": lib.library, "status": "rna_refreshed"}
    if fr.exists() and fa.exists() and fg.exists():
        v = ad.read_h5ad(fa, backed="r").var_names
        if len(v) == len(C) and (v == C.peak.values).all():      # cached only if counted on this consensus
            return {"library": lib.library, "status": "cached"}
    zp = Path(lib.zip)
    h = read_h5(zp)
    R, rvar = rna_by_id(h); sym = pd.Index(rvar.symbol)
    bc = h["barcodes"]
    import mio
    lay = mio.fragment_layout(zp)        # logged: 11 KPMP libraries carry a 6th column; only the first 4 are read
    mio.clean_fragments.dropped = 0
    q, P, A, genes = fc.count(iter_fragments(zp, M["chunk_rows"]), bc, C, tss, genemap, M)
    dropped = mio.clean_fragments.dropped
    obs = q.reset_index()
    obs.index = [f"{lib.library}_{b}" for b in bc]
    obs.insert(1, "participant", lib.redcap_id); obs.insert(2, "library", lib.library)
    obs["rna_counts"] = np.asarray(R.sum(1)).ravel()
    obs["rna_genes"] = np.asarray((R > 0).sum(1)).ravel()
    obs["pct_mt"] = np.asarray(R[:, sym.str.startswith("MT-")].sum(1)).ravel() / np.maximum(obs.rna_counts.values, 1) * 100
    ad.AnnData(X=R, obs=obs, var=rvar).write_h5ad(fr, compression="gzip")   # Ensembl ids; symbols in var.symbol
    ad.AnnData(X=P.astype(np.float32), obs=obs, var=pd.DataFrame(index=C.peak.values)).write_h5ad(fa, compression="gzip")
    ad.AnnData(X=A.astype(np.float32), obs=obs, var=pd.DataFrame(index=genes)).write_h5ad(fg, compression="gzip")
    return {"library": lib.library, "status": "ok", "n_cells": len(bc), "median_fragments": float(obs.n_fragments.median()),
            "median_tss": float(obs.tss_enrichment.median()), "median_frip": float(obs.frip.median()),
            "malformed_fragment_rows": dropped, "fragment_fields": lay["n_fields"], "minutes": round((time.time() - t0) / 60, 1)}


def genome_refs(c: dict):
    M = c["multiome"]
    gtf = ensure_gtf(M["gtf_url"], M["gtf_md5"], Path(c["raw_dir"]) / "reference")
    tss = tss_table(gtf, M["chrom_keep_regex"])
    genemap = fc.segment_map(fc.gene_regions(gtf, M["chrom_keep_regex"], M["gene_upstream"]))
    return gtf, tss, genemap


def main():
    c = cfg(); M = c["multiome"]; set_global_seed(c["seed"])
    out = result_dir("data_multiome")
    L = pd.read_csv(out / "library_table.tsv", sep="\t")
    if "--only" in sys.argv:
        L = L[L.library == sys.argv[sys.argv.index("--only") + 1]]
    C = pd.read_csv(out / "consensus_peaks.tsv.gz", sep="\t")
    gtf, tss, genemap = genome_refs(c)
    libdir = Path(c["processed_dir"]) / "multiome_lib"; libdir.mkdir(parents=True, exist_ok=True)
    res = Parallel(n_jobs=M["n_jobs"], verbose=5)(delayed(count_library_safe)(r, C, tss, genemap, M, libdir) for _, r in L.iterrows())
    S = pd.DataFrame(res)
    print(S.to_string(), flush=True)
    if "--only" not in sys.argv and "--rna-only" not in sys.argv:
        S.to_csv(out / "count_log.tsv", sep="\t", index=False)
        write_provenance("data_multiome/prov_03b_counts", [out / "library_table.tsv", out / "consensus_peaks.tsv.gz", gtf],
                         [out / "count_log.tsv"], c["seed"], cfg=c, extra={"step": "03b", "libdir": str(libdir)})


if __name__ == "__main__":
    main()
