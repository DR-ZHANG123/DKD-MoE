"""07_ext_atac: external snATAC test matrices in the KPMP Multiome feature spaces (forward inference only).

- GSE195460 (13 snATAC libraries, Control1-6 / DN1-7): every library's fragment file is re-counted with
  the same code as KPMP (fragcount.py) on the KPMP consensus peaks AND as gene activity; nuclei = the
  CellRanger-ATAC cell barcodes, filtered with the same ATAC QC thresholds as KPMP.
  -> processed/dkd_moe/gse195460_atac.h5ad (peaks), gse195460_atac_geneactivity.h5ad
- GSE211785 (18 snATAC libraries, 14 donors): no fragments are public; the authors' gene-activity counts
  for the SN_ATAC nuclei are taken from the merged object (counts layer), restricted to the protein-coding
  genes of the KPMP gene-activity matrix. -> processed/dkd_moe/gse211785_atac.h5ad (gene activity)
Outputs dkd_moe/results/data_ext_atac/{library_qc.tsv, donor_summary.tsv}."""
from __future__ import annotations
import importlib, re
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dcommon import cfg, result_dir, write_provenance, set_global_seed, rpath
import fragcount as fc

genome_refs = importlib.import_module("03b_multiome_count").genome_refs


def gse195460_libraries(raw: Path) -> pd.DataFrame:
    rows = []
    for f in sorted(raw.glob("*_fragments.tsv.gz")):
        lib = re.search(r"_((?:Control|DN)\d+)_fragments", f.name).group(1)
        h5 = next(raw.glob(f"*_{lib}_filtered_peak_bc_matrix.h5"))
        rows.append({"library": lib, "donor": lib, "group": "Control" if lib.startswith("Control") else "DKD",
                     "fragments": str(f), "h5": str(h5)})
    return pd.DataFrame(rows)


def count_one(r, C, tss, genemap, M: dict) -> tuple:
    with h5py.File(r.h5, "r") as h:
        bc = np.array([x.decode() for x in h["matrix/barcodes"][:]])
    q, P, A, genes = fc.count(fc.read_fragments_plain(r.fragments, M["chunk_rows"]), bc, C, tss, genemap, M)
    q.index = [f"GSE195460_{r.library}_{b}" for b in bc]
    q = q.assign(library=r.library, donor=r.donor, group=r.group, cohort="GSE195460")
    keep = (q.n_fragments.between(M["min_frag"], M["max_frag"]) & (q.tss_enrichment >= M["min_tss_enrichment"])
            & (q.nucleosome_signal <= M["max_nucleosome_signal"])).values
    row = {"cohort": "GSE195460", "library": r.library, "donor": r.donor, "group": r.group, "cells_cellranger": len(bc),
           "cells_qc": int(keep.sum()), "median_fragments": float(q.n_fragments[keep].median()),
           "median_tss_enrichment": float(q.tss_enrichment[keep].median()), "median_frip_consensus": float(q.frip[keep].median())}
    return (ad.AnnData(P[keep].astype(np.float32), obs=q[keep], var=pd.DataFrame(index=C.peak.values)),
            ad.AnnData(A[keep].astype(np.float32), obs=q[keep], var=pd.DataFrame(index=genes)), row)


def gse211785(genes: pd.Index, c: dict) -> tuple[ad.AnnData, pd.DataFrame]:
    a = ad.read_h5ad(rpath("data/processed/GSE211785_PreSCVI.h5ad"), backed="r")
    m = np.where((a.obs.tech == "SN_ATAC").values)[0]
    X = a.layers["counts"][m[0]:m[-1] + 1][m - m[0]]
    obs = a.obs.iloc[m][["orig_ident", "sample", "group", "Status", "sex", "age", "passed_filters", "TSS.enrichment",
                         "nucleosome_signal", "pct_reads_in_peaks"]].copy()
    obs = obs.rename(columns={"orig_ident": "library", "sample": "donor", "passed_filters": "n_fragments",
                              "TSS.enrichment": "tss_enrichment"}).assign(cohort="GSE211785")
    for k in obs.columns:
        if obs[k].dtype.name == "category":
            obs[k] = obs[k].astype(str)
    for k in ["n_fragments", "tss_enrichment", "nucleosome_signal", "pct_reads_in_peaks", "age"]:
        obs[k] = pd.to_numeric(obs[k], errors="coerce")          # stored as text in the merged object
    g = ad.AnnData(X.tocsr().astype(np.float32), obs=obs, var=pd.DataFrame(index=a.var_names))
    shared = genes.intersection(g.var_names)
    g = g[:, shared].copy()
    q = obs.assign(gact_total=np.asarray(g.X.sum(1)).ravel()).groupby(["library", "donor", "group"]).agg(
        cells=("cohort", "size"), median_fragments=("n_fragments", "median"), median_tss_enrichment=("tss_enrichment", "median"),
        median_gene_activity=("gact_total", "median")).reset_index().assign(cohort="GSE211785")
    return g, q


def main():
    c = cfg(); M = c["multiome"]; set_global_seed(c["seed"])
    out = result_dir("data_ext_atac"); pdir = Path(c["processed_dir"])
    C = pd.read_csv(Path(__file__).resolve().parents[2] / "results/data_multiome/consensus_peaks.tsv.gz", sep="\t")
    gtf, tss, genemap = genome_refs(c)
    L = gse195460_libraries(Path(c["raw_dir"]).parent / "gse195460_atac")
    res = Parallel(n_jobs=min(len(L), M["n_jobs"]))(delayed(count_one)(r, C, tss, genemap, M) for r in L.itertuples())
    P = ad.concat([x[0] for x in res]); A = ad.concat([x[1] for x in res])
    f1, f2 = pdir / "gse195460_atac.h5ad", pdir / "gse195460_atac_geneactivity.h5ad"
    P.write_h5ad(f1, compression="gzip"); A.write_h5ad(f2, compression="gzip")
    g, q211 = gse211785(genemap[2], c)
    f3 = pdir / "gse211785_atac.h5ad"
    g.write_h5ad(f3, compression="gzip")
    Q = pd.concat([pd.DataFrame([x[2] for x in res]), q211], ignore_index=True)
    Q.to_csv(out / "library_qc.tsv", sep="\t", index=False)
    D = pd.concat([P.obs, g.obs]).groupby(["cohort", "group"]).agg(donors=("donor", "nunique"), libraries=("library", "nunique"),
                                                                  nuclei=("library", "size")).reset_index()
    D.to_csv(out / "donor_summary.tsv", sep="\t", index=False)
    print(Q.to_string(index=False)); print(D.to_string(index=False))
    write_provenance("data_ext_atac", [gtf, rpath("data/processed/GSE211785_PreSCVI.h5ad")] + list(L.fragments),
                     [f1, f2, f3, out / "library_qc.tsv", out / "donor_summary.tsv"], c["seed"], cfg=c,
                     extra={"gse195460_nuclei": P.n_obs, "gse211785_nuclei": g.n_obs, "gse211785_genes": g.n_vars,
                            "note": "GSE211785 has no public fragments; gene activity from the authors' merged object"})


if __name__ == "__main__":
    main()
