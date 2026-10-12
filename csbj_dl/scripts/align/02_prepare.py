"""02_prepare: per-section cache pairing each embedded H&E patch with its spot's measured counts.

For every section in common.registry(): spots with a tissue-passing patch embedding are joined to
their raw counts on the gene set measured on both platforms (KPMP 10x whole transcriptome and
GSE211785 FFPE probe panel; no zero-filling of unmeasured genes). PT-rich spots follow the project
rule (config/run.yaml visium.pt_rich_top_frac): top fraction of the section's in-tissue spots by the
PT-lineage marker score (stage 10a/10b marker lists, z of log1p CP10k within section).
Idempotent: a section whose cache file exists is skipped (rerun after new sections are embedded).
Outputs  <cache>/<cohort>__<section>.npz, <cache>/genes.txt, results/align_prep/sections.tsv
"""
from __future__ import annotations
import importlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.io, scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

vc = importlib.import_module("10_visium_common")
RES = "align_prep"


def gse_matrix():
    proc = C.ROOT / "data/processed"
    M = scipy.io.mmread(proc / "GSE211785_ST_counts.mtx").tocsr().T.tocsr()
    genes = pd.Index(open(proc / "GSE211785_ST_genes.txt").read().split())
    bcs = pd.Index(open(proc / "GSE211785_ST_barcodes.txt").read().split())
    meta = pd.read_csv(C.ROOT / "data/raw/GSE211785/GSE211785_ST_metadata.txt.gz", sep="\t",
                       index_col=0, low_memory=False).reindex(bcs)
    return M, genes, bcs, meta["orig.ident"].values


def read_kpmp(h5, positions):
    import scanpy as sc
    a = sc.read_10x_h5(h5); a.var_names_make_unique()
    pos = pd.read_csv(positions, header=0 if str(positions).endswith("tissue_positions.csv") else None)
    pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row", "pxl_col"]
    keep = pos.barcode[pos.in_tissue == 1]
    a = a[a.obs_names.isin(keep)]
    return sp.csr_matrix(a.X), pd.Index(a.var_names), pd.Index(a.obs_names)


def pt_rich(X, genes, markers, frac):
    Z, _ = vc.lognorm_z(X, genes)
    s = Z[[g for g in markers if g in Z.columns]].mean(1).values
    return s, s >= np.nanquantile(s, 1 - frac)


def main():
    cfg = C.load_cfg(); rc = C.run_cfg(); C.set_seed(cfg["seed"])
    cache = C.p(cfg["paths"]["cache"]); cache.mkdir(parents=True, exist_ok=True)
    out = C.result_dir(RES)
    reg = C.registry()
    M, ggenes, gbcs, gsec = gse_matrix()
    gfile = cache / "genes.txt"
    if gfile.exists():
        common = pd.Index(gfile.read_text().split())
    else:   # GSE211785 panel genes measured in every KPMP section
        common = set(ggenes)
        for h5 in reg[reg.cohort != "gse211785"].h5.unique():   # KPMP (and GSE339455 when in the track)
            import h5py
            with h5py.File(h5, "r") as f:
                common &= set(x.decode() for x in f["matrix/features/name"][:])
        common = pd.Index(sorted(common)); gfile.write_text("\n".join(common))
    print("common genes", len(common), flush=True)
    frac = rc["visium"]["pt_rich_top_frac"]
    mk = {"kpmp": pd.read_csv(C.ROOT / "results/10b_visium_kpmp/pt_lineage_markers.tsv", sep="\t").gene.tolist(),
          "gse211785": pd.read_csv(C.ROOT / "results/10a_visium_gse211785/pt_lineage_markers.tsv", sep="\t").gene.tolist()}
    mk["gse339455"] = mk["kpmp"]      # 10x whole-slide probe panel; KPMP-derived PT-lineage markers
    regions = None
    rows = []
    for r in reg.itertuples():
        f = cache / f"{r.cohort}__{r.section}.npz"
        if not f.exists():
            if r.cohort == "kpmp":
                X, genes, bcs = read_kpmp(r.h5, r.positions)
            elif r.cohort == "gse339455":   # one tissue region (patient) of a multi-patient slide
                if regions is None:
                    regions = pd.read_csv(C.p(cfg["paths"]["external_regions"]), sep="\t")
                X, genes, bcs = read_kpmp(r.h5, r.positions)
                keep = bcs.isin(regions.barcode[regions.region_id == r.participant])
                X, bcs = X[keep], bcs[keep]
            else:
                m = gsec == r.section
                X, genes, bcs = M[m], ggenes, gbcs[m]
            score, rich = pt_rich(X, genes, mk[r.cohort], frac)
            e = np.load(r.emb)
            ids = pd.Index(e["spot_id"])
            pos = bcs.get_indexer(ids)
            ok = pos >= 0
            gi = genes.get_indexer(common)
            Xc = X[pos[ok]][:, gi]
            F = pd.read_csv(r.feats, sep="\t").set_index("spot_id").reindex(ids[ok])
            np.savez_compressed(f, spot_id=ids[ok].values.astype(str), emb=e["emb"][ok],
                                data=Xc.data, indices=Xc.indices, indptr=Xc.indptr, shape=Xc.shape,
                                lib=np.asarray(Xc.sum(1)).ravel(), lib_all=np.asarray(X[pos[ok]].sum(1)).ravel(),
                                pt_score=score[pos[ok]], pt_rich=rich[pos[ok]],
                                px_row=F.px_row.values, px_col=F.px_col.values,
                                feat_names=np.array(cfg["interpret"]["morph_features"]),
                                feats=F[cfg["interpret"]["morph_features"]].values.astype(np.float32))
            print(r.section, int(ok.sum()), "/", len(ids), "matched", flush=True)
        z = np.load(f, allow_pickle=False)
        rows.append({"cohort": r.cohort, "section": r.section, "participant": r.participant, "group": r.group,
                     "n_spots": int(len(z["spot_id"])), "n_pt_rich": int(z["pt_rich"].sum()),
                     "median_counts_common": float(np.median(z["lib"])), "cache": str(f),
                     "um_per_px": r.um_per_px, "field_px_native": r.field_px_native, "tif": r.tif})
    S = pd.DataFrame(rows)
    S = S[S.n_spots >= cfg["data"]["min_spots_section"]]
    S.to_csv(out / "sections.tsv", sep="\t", index=False)
    # rule check: PT-rich flags agree with the stage-10b assignment for already-scored sections
    sc10 = pd.read_csv(C.ROOT / "results/10b_visium_kpmp/spot_scores.tsv.gz", sep="\t",
                       usecols=["barcode", "participant", "sample", "PT_rich"], dtype={"participant": str})
    sc10["section"] = sc10.participant + "__" + sc10["sample"]
    agree = []
    for s in S[S.cohort == "kpmp"].itertuples():
        z = np.load(s.cache)
        ref = sc10[sc10.section == s.section].set_index("barcode").PT_rich.reindex(z["spot_id"])
        if ref.notna().any():
            agree.append({"section": s.section, "agree": float((ref.values.astype(bool) == z["pt_rich"]).mean())})
    A = pd.DataFrame(agree); A.to_csv(out / "pt_rich_agreement_10b.tsv", sep="\t", index=False)
    print(S.groupby(["cohort", "group"]).agg(sections=("section", "size"), participants=("participant", "nunique"),
                                             spots=("n_spots", "sum")))
    print("PT-rich agreement with 10b:", A.agree.describe().round(4).to_dict() if len(A) else None)
    C.provenance(RES, [gfile, C.ROOT / "results/11a_histology_embed/section_index.tsv"],
                 [out / "sections.tsv", out / "pt_rich_agreement_10b.tsv"], cfg["seed"],
                 extra={"n_common_genes": len(common), "n_sections": len(S)})


if __name__ == "__main__":
    main()
