"""03a_multiome_peaks: library table and consensus peak set for the KPMP 10x Multiome archives.

CellRanger-ARC calls peaks separately in every library (~900 bp wide), so coordinates differ between samples.
Consensus (ArchR-style): every core-group (DKD, HKD, HKD_withDM, Reference, DM_noCKD) library peak is re-centred
to a fixed peak_width window; windows are ranked by how many library peaks overlap them and kept greedily unless
they overlap a kept window; windows supported by < consensus_min_libraries peaks are dropped. Merging overlapping
peaks instead (Signac reduce) chains neighbouring peaks into ever wider regions as libraries accumulate (median
1 kb with only 10 libraries), losing resolution. Every library is then re-counted on this one peak set from its
own fragment file (03b); defined by the core groups, the set does not change when other participants are added.
Archives of one participant sharing > duplicate_barcode_overlap of barcodes are the same library
processed twice; the newer archive (by repository release order, then name) is kept.
Outputs: dkd_moe/results/data_multiome/{library_table.tsv, duplicate_archives.tsv, consensus_peaks.tsv.gz}
and processed/dkd_moe/multiome_consensus_peaks.bed."""
from __future__ import annotations
import re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from dcommon import cfg, result_dir, write_provenance, set_global_seed, groups, dest
from mio import read_h5, sample_peaks, fragment_header, fixed_width_consensus


def library_info(row) -> dict:
    zp = Path(row.local_path)
    h = read_h5(zp)
    hdr = fragment_header(zp)
    pk = sample_peaks(h)
    genes = h["features"][h["features"].type == "Gene Expression"].id
    return {"redcap_id": row.redcap_id, "file_name": row.file_name, "zip": str(zp),
            "library": f"{row.redcap_id}_{row.file_name[:8]}", "n_barcodes": len(h["barcodes"]),
            "n_peaks_called": len(pk), "n_genes": len(genes), "gene_ids_hash": __import__("hashlib").md5("|".join(genes).encode()).hexdigest()[:12],
            "pipeline_version": hdr.get("pipeline_version", ""), "reference_version": hdr.get("reference_version", ""),
            "reference_path": hdr.get("reference_path", ""), "_barcodes": set(h["barcodes"]),
            "_peaks": pk}


def dedupe(L: pd.DataFrame, bcs: dict, thr: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    drop, log = set(), []
    for pid, g in L.groupby("redcap_id"):
        rows = list(g.itertuples())
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                a, b = bcs[rows[i].file_name], bcs[rows[j].file_name]
                ov = len(a & b) / max(1, min(len(a), len(b)))
                if ov > thr:
                    old = sorted([rows[i].file_name, rows[j].file_name])[0]
                    drop.add(old)
                log.append({"participant": pid, "file_a": rows[i].file_name, "file_b": rows[j].file_name,
                            "barcode_overlap": round(ov, 3), "dropped": old if ov > thr else ""})
    return L[~L.file_name.isin(drop)], pd.DataFrame(log)


def main():
    c = cfg(); M = c["multiome"]; set_global_seed(c["seed"])
    out = result_dir("data_multiome")
    man_f = Path(__file__).resolve().parents[2] / "results/data_inventory/download_plan_multiome.tsv"
    man = pd.read_csv(man_f, sep="\t")
    man = man[man.file_name.str.endswith("_10x-multiome_expression_matrix.zip")].copy()
    man["local_path"] = [str(dest(r, c, "multiome")) for r in man.itertuples()]
    man["present"] = [Path(p).exists() and Path(p).stat().st_size == int(s) for p, s in zip(man.local_path, man.file_size)]
    core = man.group.isin(c["core_groups"])
    if not man.loc[core, "present"].all() and "--allow-partial" not in sys.argv:   # consensus = core study groups only
        raise SystemExit(f"{int((core & ~man.present).sum())} core-group archives still missing; consensus not built")
    miss = man[core & ~man.present][["redcap_id", "group", "file_name", "file_size"]]
    miss.assign(reason="not on disk at consensus build (server failure or download still running)").to_csv(
        out / "excluded_core_archives.tsv", sep="\t", index=False)
    man = man[man.present]
    info = Parallel(n_jobs=M["n_jobs"])(delayed(library_info)(r) for r in man.itertuples())
    bcs = {d["file_name"]: d.pop("_barcodes") for d in info}
    peaks = {d["file_name"]: d.pop("_peaks") for d in info}
    L = pd.DataFrame(info)
    L, dup = dedupe(L, bcs, M["duplicate_barcode_overlap"])
    dup.to_csv(out / "duplicate_archives.tsv", sep="\t", index=False)
    G = groups().group
    L["group"] = L.redcap_id.map(G).fillna("not_in_clinical_table")

    keep_chr = re.compile(M["chrom_keep_regex"])
    # consensus from core-group libraries only, so it does not change when other participants are added later
    P = pd.concat([peaks[r.file_name].assign(library=r.library) for r in L.itertuples() if r.group in c["core_groups"]],
                  ignore_index=True)
    P = P[P.chrom.map(lambda x: bool(keep_chr.match(x)))]
    n_all = len(P)
    if "--keep-consensus" in sys.argv and (out / "consensus_peaks.tsv.gz").exists():
        C = pd.read_csv(out / "consensus_peaks.tsv.gz", sep="\t")      # feature space already in use downstream
    else:
        C = fixed_width_consensus(P, M["peak_width"], M["consensus_min_libraries"])
        C["peak"] = C.chrom + ":" + C.start.astype(str) + "-" + C.end.astype(str)
    print(f"libraries {len(L)}; library peaks {n_all}; consensus peaks {len(C)}; median width {int((C.end - C.start).median())}", flush=True)
    bed = Path(c["processed_dir"]) / "multiome_consensus_peaks.bed"
    bed.parent.mkdir(parents=True, exist_ok=True)
    C[["chrom", "start", "end", "peak", "n_libraries"]].to_csv(bed, sep="\t", header=False, index=False)
    C.to_csv(out / "consensus_peaks.tsv.gz", sep="\t", index=False)
    L.to_csv(out / "library_table.tsv", sep="\t", index=False)
    write_provenance("data_multiome/prov_03a_peaks", [man_f], [out / "library_table.tsv", out / "duplicate_archives.tsv",
                     out / "consensus_peaks.tsv.gz", bed], c["seed"], cfg=c,
                     extra={"step": "03a", "n_libraries": len(L), "n_library_peaks": n_all, "n_consensus_peaks": len(C),
                            "n_reference_versions": L.reference_version.nunique(), "n_gene_feature_sets": L.gene_ids_hash.nunique()})


if __name__ == "__main__":
    main()
