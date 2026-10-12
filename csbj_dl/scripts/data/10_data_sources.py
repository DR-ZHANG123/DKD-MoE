"""10_data_sources: csbj_dl/results/data_inventory/DATA_SOURCES.md from the download manifests and outputs.

Lists, for every dataset fetched by the data module: repository / accession and DOI, release, file counts,
bytes, download status and the checksum manifest that records each file's MD5."""
from __future__ import annotations
from pathlib import Path
import pandas as pd
from dcommon import cfg, CDL, result_dir, write_provenance, file_table

KPMP_DESC = {"multiome": "10x Multiome (snRNA + snATAC, same nuclei) expression archives (CellRanger-ARC filtered h5, fragments) + experimental metadata",
             "visium": "Visium spatial bundles (filtered h5, spatial/), full-resolution H&E tif and metadata for sections not already local",
             "pathomics": "Segmentation masks & pathomics vectors (feature workbooks, OME-XML, README extracted; masks/svs removed after reading svs resolution)",
             "annotation": "Aggregated clustered atlases (h5Seurat) with KPMP cell-type labels"}


def row(name, M: pd.DataFrame, doi: str, release: str, manifest: Path, desc: str) -> str:
    ok = M.status.str.startswith("ok")
    nbytes = M.loc[ok, "observed_bytes"].sum() if "observed_bytes" in M else 0
    return (f"| {name} | {desc} | {doi} | {release} | {int(ok.sum())}/{len(M)} | {nbytes / 1e9:.1f} | "
            f"`{manifest.relative_to(CDL)}` |")


def main():
    c = cfg(); out = result_dir("data_inventory")
    F = file_table(c)
    rel = "file table 2026-10-10"
    lines = ["# Data sources fetched by the csbj_dl data module", "",
             f"KPMP: Kidney Precision Medicine Project open-access repository (atlas.kpmp.org), dataset DOI {c['kpmp_doi']};",
             f"file table snapshot `{c['file_table']}` (harvested 2026-10-10; {len(F)} open files). Clinical table `{c['clinical_table']}`.",
             f"Raw files: `{c['raw_dir']}/<category>/`; processed: `{c['processed_dir']}/`.", "",
             "| dataset | content | DOI / accession | release | files ok/planned | GB | checksum manifest (MD5 per file) |",
             "|---|---|---|---|---|---|---|"]
    ins = []
    for cat, desc in KPMP_DESC.items():
        mf = CDL / f"results/data_download/{cat}/download_manifest_{cat}.tsv"
        if mf.exists():
            M = pd.read_csv(mf, sep="\t"); ins.append(mf)
            lines.append(row(f"KPMP {cat}", M, c["kpmp_doi"], rel, mf, desc))
    ext = CDL / "results/data_download/external/download_manifest_external.tsv"
    if ext.exists():
        E = pd.read_csv(ext, sep="\t"); ins.append(ext)
        for ds, d in E.groupby("dataset"):
            gse = ds.split("_")[0]
            lines.append(row(ds, d, f"GEO {gse}", "GEO public", ext,
                             {"GSE339455": "Visium CytAssist FFPE, biopsy-proven hypertensive nephropathy (4 slides x 3 patients)",
                              "GSE195460": "snATAC fragments + CellRanger-ATAC cell barcodes (Control1-6, DN1-7)"}[gse]))
    fails = []
    for mf in ins:
        M = pd.read_csv(mf, sep="\t")
        for r in M[~M.status.astype(str).str.startswith("ok")].itertuples():
            fails.append(f"- {getattr(r, 'redcap_id', getattr(r, 'dataset', ''))}: `{getattr(r, 'file_name', getattr(r, 'file', ''))}`")
    lines += ["", "Files that could not be fetched (KPMP server returned HTTP 500 / no data on every attempt, >= 3 attempts):"] + (fails or ["- none"])
    lines += ["", "Multiome libraries excluded downstream: see `results/data_multiome/excluded_core_archives.tsv` (archive not",
              "available) and `results/data_multiome/library_qc.tsv` (kept = False: < min_cells_library nuclei passing QC)."]
    lines += ["", "Not downloaded:",
              "- KPMP 'ATAC-Seq' expression archives: each holds the same fragments and filtered h5 as the 10x Multiome archive of the same library.",
              "- GSE211785 snATAC fragments: not public (raw data withheld by the submitters); gene-activity counts for its 55,955 snATAC nuclei",
              "  were taken from the already-local merged object `data/processed/GSE211785_PreSCVI.h5ad` (GEO GSE211785).",
              "", "Reference annotation: GENCODE release 32 primary-assembly GTF (annotation of the CellRanger-ARC GRCh38-2020-A reference),",
              f"MD5 {c['multiome']['gtf_md5']}, `{c['raw_dir']}/reference/`.", ""]
    (out / "DATA_SOURCES.md").write_text("\n".join(lines))
    print("\n".join(lines))
    write_provenance("data_inventory", ins + [out / "size_estimate.tsv"], [out / "DATA_SOURCES.md", out / "size_estimate.tsv"],
                     c["seed"], cfg=c, extra={"step": "10_data_sources"})


if __name__ == "__main__":
    main()
