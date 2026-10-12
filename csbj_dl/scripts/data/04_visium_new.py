"""04_visium_new: bring every KPMP Visium section not yet in data/processed/kpmp_visium to the same layout
(<participant>/<section>/outs/{filtered_feature_bc_matrix.h5, spatial/...}; always with outs/, whereas the
existing folder mixes bundles with and without outs/) under
processed/csbj_dl/kpmp_visium_new/, and tabulate all repository sections.

Spatial bundles come as .tar.gz (outs/ at the root) or .zip (Space Ranger outs/ nested under a server
path); only the filtered matrix and the spatial/ folder are extracted. Bundle, full-resolution tif and
metadata of one section share a repository package id, which is how the tif is paired with the section.
Section name: bundle name without uuid, 'spatial_' / 'space_ranger_outs_' prefixes and extension.
Output csbj_dl/results/data_visium/section_table.tsv (one row per repository section)."""
from __future__ import annotations
import io, json, shutil, tarfile, zipfile
from pathlib import Path
import pandas as pd
from dcommon import cfg, select, file_table, existing_raw_names, groups, rpath, dest, result_dir, write_provenance, set_global_seed

KEEP_SPATIAL = ("scalefactors_json.json", "tissue_positions.csv", "tissue_positions_list.csv", "tissue_hires_image.png",
                "tissue_lowres_image.png", "detected_tissue_image.jpg", "aligned_fiducials.jpg", "spatial_enrichment.csv")


def section_name(fname: str) -> str:
    s = fname.split("_", 1)[1]
    for pre in ("spatial_", "space_ranger_outs_"):
        if s.startswith(pre):
            s = s[len(pre):]
    for ext in (".tar.gz", ".zip"):
        if s.endswith(ext):
            s = s[: -len(ext)]
    return s


def locate(row, c: dict, local: dict) -> Path | None:
    if row.file_name in local:
        return local[row.file_name]
    p = dest(row, c, "visium")
    return p if p.exists() and p.stat().st_size == int(row.file_size) else None


def extract(bundle: Path, outs: Path) -> list[str]:
    """Write filtered_feature_bc_matrix.h5 and spatial/* of one bundle into outs/."""
    (outs / "spatial").mkdir(parents=True, exist_ok=True)
    got = []
    def want(n: str) -> str | None:
        base = n.rsplit("/", 1)[-1]
        if base == "filtered_feature_bc_matrix.h5":
            return base
        if base in KEEP_SPATIAL and "/spatial/" in "/" + n:
            return f"spatial/{base}"
        return None
    if bundle.name.endswith(".tar.gz"):
        with tarfile.open(bundle) as t:
            for m in t.getmembers():
                tgt = want(m.name) if m.isfile() else None
                if tgt:
                    (outs / tgt).write_bytes(t.extractfile(m).read()); got.append(tgt)
    else:
        with zipfile.ZipFile(bundle) as z:
            for n in z.namelist():
                tgt = want(n)
                if tgt:
                    (outs / tgt).write_bytes(z.read(n)); got.append(tgt)
    return got


def n_tissue_spots(spatial: Path) -> int:
    f = spatial / "tissue_positions.csv"
    if f.exists():
        return int(pd.read_csv(f).iloc[:, 1].eq(1).sum())
    f = spatial / "tissue_positions_list.csv"
    return int(pd.read_csv(f, header=None).iloc[:, 1].eq(1).sum()) if f.exists() else -1


def main():
    c = cfg(); set_global_seed(c["seed"])
    out = result_dir("data_visium")
    F = file_table(c); local = existing_raw_names(c)
    V = select("visium", c, F)
    bundles = V[V.workflow_type.eq("Spatial Imaging Data")]
    tifs = V[V.data_format.eq("tif")].set_index("package_id")
    # existing sections keep the bundle layout (outs/ present in some, absent in others)
    old = {p.name: p for p in rpath(c["existing_visium"]).glob("*/*") if next(p.rglob("filtered_feature_bc_matrix.h5"), None)}
    emb_f = rpath("results/11a_histology_embed/section_index.tsv")
    embedded = set(pd.read_csv(emb_f, sep="\t").query("cohort == 'kpmp'").section) if emb_f.exists() else set()
    newdir = Path(c["processed_dir"]) / "kpmp_visium_new"
    G = groups().group
    rows = []
    for r in bundles.itertuples():
        sec = section_name(r.file_name)
        t = tifs.loc[r.package_id] if r.package_id in tifs.index else None
        tif = locate(t, c, local) if t is not None else None
        rec = {"section": sec, "participant": r.redcap_id, "group": G.get(r.redcap_id, "not_in_clinical_table"),
               "package_id": r.package_id, "bundle": r.file_name, "tif_file": t.file_name if t is not None else "",
               "tif_path": str(tif) if tif else "", "tif_GB": round(t.file_size / 1e9, 2) if t is not None else None,
               "embedded_11a": sec in embedded}
        if sec in old:
            sp = next(old[sec].rglob("scalefactors_json.json")).parent
            rec.update(source="existing", processed_dir=str(old[sec]), n_spots=n_tissue_spots(sp), status="ok")
        else:
            b = locate(r, c, local)
            sdir = newdir / r.redcap_id / sec
            outs = sdir / "outs"
            if b is None:
                rec.update(source="new", processed_dir="", n_spots=-1, status="bundle missing")
            else:
                if not (outs / "filtered_feature_bc_matrix.h5").exists():
                    if sdir.exists():
                        shutil.rmtree(sdir)
                    extract(b, outs)
                ok = (outs / "filtered_feature_bc_matrix.h5").exists() and (outs / "spatial/scalefactors_json.json").exists()
                rec.update(source="new", processed_dir=str(sdir), n_spots=n_tissue_spots(outs / "spatial"),
                           status="ok" if ok else "incomplete bundle")
        rows.append(rec)
    T = pd.DataFrame(rows).sort_values(["source", "group", "participant", "section"])
    T.to_csv(out / "section_table.tsv", sep="\t", index=False)
    S = T[T.status == "ok"].groupby(["source", "group"]).agg(sections=("section", "size"), participants=("participant", "nunique"),
                                                          with_tif=("tif_path", lambda s: int((s != "").sum())),
                                                          spots=("n_spots", "sum")).reset_index()
    S.to_csv(out / "section_summary.tsv", sep="\t", index=False)
    print(S.to_string(index=False)); print(T.status.value_counts().to_string())
    write_provenance("data_visium", [rpath(c["file_table"])], [out / "section_table.tsv", out / "section_summary.tsv"],
                     c["seed"], cfg=c, extra={"new_dir": str(newdir), "n_sections": len(T),
                                              "status": json.loads(T.status.value_counts().to_json())})


if __name__ == "__main__":
    main()
