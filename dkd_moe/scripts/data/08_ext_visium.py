"""08_ext_visium: GSE339455 (Visium CytAssist, FFPE, biopsy-proven hypertensive nephropathy; external test only)
in the KPMP processed layout: processed/dkd_moe/gse339455_visium/<slide>/outs/{filtered_feature_bc_matrix.h5, spatial/}.

Each capture area carries three patients' biopsies (GEO sample title lists the three registry numbers, but GEO
gives no spot-to-patient table). Patients are therefore separated as tissue regions: in-tissue spots are linked
to their hexagonal array neighbours, connected components are found, the three largest components are the
patient regions and stray fragments join the nearest region (array distance). Region numbers are ordered by
position (array column, then row); which registry number belongs to which region is not public.
GEO provides the high-resolution (and CytAssist) image only, no full-resolution H&E.
Outputs dkd_moe/results/data_visium/{external_section_table.tsv, gse339455_spot_regions.tsv.gz}."""
from __future__ import annotations
import gzip, re, shutil, tarfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.io
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from dcommon import cfg, result_dir, write_provenance, set_global_seed

GROUP = {"ES": "HKD_early_stable", "EP": "HKD_early_progressor", "LS": "HKD_late_stable", "LP": "HKD_late_progressor"}
TITLES = {"GSM9895920": "nr63,nr64,nr80", "GSM9895921": "nr94,nr95,nr122", "GSM9895922": "nr59,nr66,nr99",
          "GSM9895923": "nr67,nr68,nr100"}   # GEO sample titles (registry numbers of the three patients)


def write_10x_h5(path: Path, X: sp.csc_matrix, barcodes, feat: pd.DataFrame) -> None:
    """CellRanger v3 HDF5 layout (features x barcodes CSC), readable by scanpy.read_10x_h5."""
    enc = lambda v: np.array([str(x).encode() for x in v])
    with h5py.File(path, "w") as h:
        m = h.create_group("matrix")
        m["barcodes"] = enc(barcodes); m["data"] = X.data.astype(np.int32); m["indices"] = X.indices.astype(np.int64)
        m["indptr"] = X.indptr.astype(np.int64); m["shape"] = np.array(X.shape, np.int32)
        f = m.create_group("features")
        f["id"] = enc(feat[0]); f["name"] = enc(feat[1]); f["feature_type"] = enc(feat[2] if 2 in feat else ["Gene Expression"] * len(feat))
        f["genome"] = enc(["GRCh38"] * len(feat)); f["_all_tag_keys"] = enc(["genome"])


def regions(pos: pd.DataFrame, n_regions: int = 3) -> tuple[np.ndarray, list[int]]:
    xy = pos[["array_row", "array_col"]].values.astype(float)
    t = cKDTree(xy * [1.0, 0.5])                 # hex grid: columns step 2 within a row, 1 between rows
    pairs = t.query_pairs(1.2, output_type="ndarray")   # same-row (1.0) and diagonal (1.118) neighbours
    A = sp.coo_matrix((np.ones(len(pairs)), (pairs[:, 0], pairs[:, 1])), shape=(len(xy), len(xy)))
    _, lab = connected_components(A, directed=False)
    sizes = pd.Series(lab).value_counts()
    big = list(sizes.index[:n_regions])
    out = np.full(len(xy), -1)
    for k, b in enumerate(big):
        out[lab == b] = k
    stray = out < 0
    if stray.any():
        tb = cKDTree(xy[~stray]); _, j = tb.query(xy[stray]); out[stray] = out[~stray][j]
    # order regions by position
    cen = pd.DataFrame(xy).groupby(out).mean().sort_values([1, 0])
    remap = {old: new + 1 for new, old in enumerate(cen.index)}
    return np.vectorize(remap.get)(out), [int(sizes[b]) for b in big]


def plot_regions(R: pd.DataFrame, outdir: Path, png: Path) -> None:
    """QC view: spots coloured by assigned patient region on each slide's high-resolution image."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image
    slides = list(GROUP)
    fig, ax = plt.subplots(1, len(slides), figsize=(4 * len(slides), 4.5))
    for a, s in zip(ax, slides):
        d = outdir / s / "outs/spatial"
        sf = pd.read_json(d / "scalefactors_json.json", typ="series")["tissue_hires_scalef"]
        p = pd.read_csv(d / "tissue_positions.csv").set_index("barcode")[["pxl_row_in_fullres", "pxl_col_in_fullres"]]
        r = R[R.slide == s].set_index("barcode").join(p)
        a.imshow(Image.open(d / "tissue_hires_image.png"))
        a.scatter(r.pxl_col_in_fullres * sf, r.pxl_row_in_fullres * sf, c=r.region, s=2, cmap="tab10", vmin=0, vmax=9)
        a.set_title(f"{s}: regions 1-3"); a.axis("off")
    fig.tight_layout(); fig.savefig(png, dpi=80); plt.close(fig)


def main():
    c = cfg(); set_global_seed(c["seed"])
    raw = Path(c["raw_dir"]).parent / "gse339455"
    tar = raw / "GSE339455_RAW.tar"
    ex = raw / "extracted"; ex.mkdir(exist_ok=True)
    with tarfile.open(tar) as t:
        t.extractall(ex, filter="data")
    outdir = Path(c["processed_dir"]) / "gse339455_visium"
    res = result_dir("data_visium")
    rows, spots = [], []
    for gsm, pats in TITLES.items():
        files = {re.sub(rf"^{gsm}_[A-Z]{{2}}_", "", f.name): f for f in ex.glob(f"{gsm}_*")}
        slide = next(ex.glob(f"{gsm}_*")).name.split("_")[1]
        sdir = outdir / slide / "outs"; spd = sdir / "spatial"
        if sdir.exists():
            shutil.rmtree(sdir)
        spd.mkdir(parents=True)
        rd = lambda k: gzip.open(files[k])
        bc = pd.read_csv(rd("barcodes.tsv.gz"), header=None)[0].values
        feat = pd.read_csv(rd("features.tsv.gz"), header=None, sep="\t")
        X = scipy.io.mmread(rd("matrix.mtx.gz")).tocsc()
        write_10x_h5(sdir / "filtered_feature_bc_matrix.h5", X, bc, feat)
        for k in ["tissue_positions.csv", "scalefactors_json.json", "tissue_hires_image.png", "tissue_lowres_image.png",
                  "aligned_fiducials.jpg", "aligned_tissue_image.jpg", "cytassist_image.tiff"]:
            (spd / k).write_bytes(gzip.open(files[k + ".gz"]).read())
        pos = pd.read_csv(spd / "tissue_positions.csv")
        pos.columns = ["barcode", "in_tissue", "array_row", "array_col", "pxl_row_in_fullres", "pxl_col_in_fullres"]
        pos = pos[(pos.in_tissue == 1) & pos.barcode.isin(bc)].reset_index(drop=True)
        lab, sizes = regions(pos)
        pos["region"] = lab; pos["slide"] = slide; pos["region_id"] = [f"GSE339455_{slide}_r{x}" for x in lab]
        spots.append(pos[["slide", "barcode", "array_row", "array_col", "region", "region_id"]])
        for r_, d in pos.groupby("region"):
            rows.append({"cohort": "GSE339455", "gsm": gsm, "section": slide, "region": int(r_), "donor_unit": f"GSE339455_{slide}_r{r_}",
                         "group": GROUP[slide], "disease": "HKD", "patients_on_slide": pats, "n_spots": len(d),
                         "component_sizes_top3": ";".join(map(str, sizes)), "processed_dir": str(outdir / slide),
                         "tif_path": "", "hires_image": str(spd / "tissue_hires_image.png"),
                         "cytassist_image": str(spd / "cytassist_image.tiff")})
    T = pd.DataFrame(rows)
    T["usable"] = T.n_spots >= c["external_visium"]["min_region_spots"]   # biopsy mostly outside the capture area otherwise
    T.to_csv(res / "external_section_table.tsv", sep="\t", index=False)
    pd.concat(spots).to_csv(res / "gse339455_spot_regions.tsv.gz", sep="\t", index=False)
    plot_regions(pd.concat(spots), outdir, res / "gse339455_regions.png")
    print(T[["section", "region", "group", "n_spots", "component_sizes_top3", "usable"]].to_string(index=False))
    write_provenance("data_visium/prov_08_gse339455", [tar], [res / "external_section_table.tsv", res / "gse339455_spot_regions.tsv.gz"],
                     c["seed"], cfg=c, extra={"n_slides": T.section.nunique(), "n_regions": len(T)})


if __name__ == "__main__":
    main()
