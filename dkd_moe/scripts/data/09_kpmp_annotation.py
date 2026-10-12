"""09_kpmp_annotation: KPMP's own published cell-type labels mapped onto the main pipeline's KPMP cells.

Sources: the newest KPMP clustered atlases in the open repository (config annotation_files; sc = PREMIERE
scRNA v2 2025, sn = single-nucleus explorer 2025). The h5Seurat metadata (cell names + label columns) is read
directly with h5py, without loading expression. Cells are matched to data/processed/KPMP_{sc,sn}_qc_counts.h5ad
on (participant, 16-nt barcode); a (participant, barcode) pair that occurs in more than one library on either
side is ambiguous and left unmatched rather than guessed.
Outputs dkd_moe/results/data_annotation/{kpmp_cell_labels_{sc,sn}.tsv.gz, match_summary.tsv, label_columns.tsv}."""
from __future__ import annotations
import re, sys
from pathlib import Path
import anndata as ad
import h5py
import numpy as np
import pandas as pd
from dcommon import cfg, result_dir, write_provenance, set_global_seed, rpath, file_table

def _dec(y) -> str:
    return y.decode("utf-8", errors="replace") if isinstance(y, bytes) else str(y)


BC = re.compile(r"([ACGT]{16})(-\d+)?")


def read_meta(path: Path) -> pd.DataFrame:
    with h5py.File(path, "r") as h:
        cells = np.array([_dec(x) for x in h["cell.names"][:]])
        md = h["meta.data"]; cols = {}
        for k in md.keys():
            v = md[k]
            if isinstance(v, h5py.Group) and "levels" in v and "values" in v:
                lev = np.array([_dec(x) for x in v["levels"][:]], dtype=object)
                idx = v["values"][:].astype(int)
                cols[k] = np.where(idx > 0, lev[np.clip(idx - 1, 0, len(lev) - 1)], None)
            elif isinstance(v, h5py.Dataset) and v.shape == (len(cells),):
                x = v[:]
                cols[k] = np.array([_dec(y) for y in x]) if x.dtype.kind in "SO" else x
    return pd.DataFrame(cols, index=cells)


LABELS = ["subclass.l1", "subclass.level1", "subclass.l2", "subclass.level2", "subclass.l3", "class", "state.l1", "state.l2", "state"]


def pick_columns(M: pd.DataFrame) -> tuple[str, list[str]]:
    """participant column and the KPMP label columns present (subclass.l1 = coarse lineage, l2/l3 = finer types)."""
    low = {c.lower(): c for c in M.columns}
    part = next(low[c] for c in ["datasourceid", "patient", "participant_id", "participant", "specimen_id"] if c in low)
    return part, [low[c] for c in LABELS if c in low]


def participant_map(c: dict) -> dict:
    """KPMP atlases use the sortable participant id (hyphen removed, e.g. 3110081); map to redcap ids (31-10081)."""
    F = file_table(c)[["redcap_id", "participant_id_sort"]].dropna().drop_duplicates()
    F = F[~F.redcap_id.str.contains("|", regex=False)]
    m = dict(zip(F.participant_id_sort.astype(str).str.replace(r"\.0$", "", regex=True), F.redcap_id))
    m.update({r: r for r in F.redcap_id})
    return m


def barcode_of(s: str) -> str | None:
    m = BC.search(s)
    return m.group(1) if m else None


def match(tag: str, M: pd.DataFrame, part_col: str, labels: list[str], pmap: dict, out: Path) -> dict:
    A = ad.read_h5ad(rpath(f"data/processed/KPMP_{tag}_qc_counts.h5ad"), backed="r").obs[["participant", "library"]].astype(str)
    A["bc"] = [barcode_of(x) for x in A.index]
    raw = M[part_col].astype(str)
    K = pd.DataFrame({"participant": raw.map(pmap).values, "atlas_participant": raw.values,
                      "bc": [barcode_of(x) for x in M.index]}, index=M.index)
    for l in labels:
        K["kpmp_" + l.replace(".level", ".l").replace(".", "_")] = M[l].astype(str).values   # same names for sc and sn
    unmapped = int(K.participant.isna().sum())
    K = K.dropna(subset=["participant"])
    ka, kk = A.groupby(["participant", "bc"], observed=True).size(), K.groupby(["participant", "bc"]).size()
    A = A.join((ka > 1).rename("amb_a"), on=["participant", "bc"]); K = K.join((kk > 1).rename("amb_k"), on=["participant", "bc"])
    J = A[~A.amb_a].reset_index(names="cell_id").merge(K[~K.amb_k], on=["participant", "bc"], how="left")
    lab = [c for c in K.columns if c.startswith("kpmp_")]
    J["matched"] = J[lab[0]].notna()
    J = J.rename(columns={"index": "atlas_cell"}) if "index" in J else J
    J[J.matched][["cell_id", "participant"] + lab].to_csv(out / f"kpmp_cell_labels_{tag}.tsv.gz", sep="\t", index=False)
    J.groupby("participant", observed=True).agg(our_cells=("cell_id", "size"), matched=("matched", "sum")).assign(
        in_atlas=lambda d: d.index.isin(set(K.participant)), match_rate=lambda d: d.matched / d.our_cells).to_csv(
        out / f"match_by_participant_{tag}.tsv", sep="\t")
    shared = set(A.participant) & set(K.participant)
    inshared = A.participant.isin(shared)
    return {"modality": tag, "atlas_cells": len(K), "atlas_participants": K.participant.nunique(), "our_cells": len(A),
            "our_participants": A.participant.nunique(), "shared_participants": len(shared),
            "matched_cells": int(J.matched.sum()), "match_rate_all": J.matched.sum() / len(A),
            "match_rate_shared_participants": J.matched.sum() / max(1, inshared.sum()),
            "ambiguous_ours": int(A.amb_a.sum()), "atlas_cells_unmapped_participant": unmapped,
            "participant_col": part_col, "label_cols": ",".join(labels)}


def main():
    c = cfg(); set_global_seed(c["seed"])
    out = result_dir("data_annotation")
    base = Path(c["raw_dir"]) / "annotation" / "_atlas"
    rows, cols, inputs = [], [], []
    pmap = participant_map(c)
    for tag, name in c["annotation_files"].items():
        f = next(base.glob(f"*{name}"), None)
        want = file_table(c).query("file_name.str.endswith(@name)", engine="python").file_size
        if f is None or f.stat().st_size != int(want.iloc[0]):
            print(f"{tag}: atlas file missing", file=sys.stderr); continue
        inputs.append(f)
        M = read_meta(f)
        for k in M.columns:
            cols.append({"modality": tag, "column": k, "n_unique": M[k].nunique(), "example": str(M[k].iloc[0])[:60]})
        part, labels = pick_columns(M)
        rows.append(match(tag, M, part, labels, pmap, out))
    pd.DataFrame(cols).to_csv(out / "label_columns.tsv", sep="\t", index=False)
    S = pd.DataFrame(rows); S.to_csv(out / "match_summary.tsv", sep="\t", index=False)
    print(S.T.to_string())
    write_provenance("data_annotation", inputs, [out / "match_summary.tsv", out / "label_columns.tsv"] +
                     [out / f"kpmp_cell_labels_{t}.tsv.gz" for t in S.modality], c["seed"], cfg=c)


if __name__ == "__main__":
    main()
