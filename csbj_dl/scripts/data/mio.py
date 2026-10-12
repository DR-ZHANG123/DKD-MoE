"""Readers for KPMP 10x Multiome archives (cellranger-arc outputs inside a zip) and genome helpers."""
from __future__ import annotations
import gzip, io, re, subprocess, zipfile
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp
from dcommon import md5


def _member(z: zipfile.ZipFile, suffix: str) -> str:
    hits = [n for n in z.namelist() if n.endswith(suffix)]
    if not hits:
        raise KeyError(f"{suffix} not in archive")
    return sorted(hits, key=len)[0]


def read_h5(zpath: Path) -> dict:
    """filtered_feature_bc_matrix.h5 -> barcodes, feature table and the full CSC matrix (features x cells)."""
    with zipfile.ZipFile(zpath) as z:
        buf = io.BytesIO(z.read(_member(z, "filtered_feature_bc_matrix.h5")))
    with h5py.File(buf, "r") as h:
        m = h["matrix"]
        dec = lambda k: np.array([x.decode() for x in m[k][:]])
        feat = pd.DataFrame({"id": dec("features/id"), "name": dec("features/name"),
                             "type": dec("features/feature_type"), "genome": dec("features/genome")})
        if "interval" in m["features"]:
            feat["interval"] = dec("features/interval")
        shape = tuple(m["shape"][:])
        X = sp.csc_matrix((m["data"][:], m["indices"][:], m["indptr"][:]), shape=shape)
        bc = dec("barcodes")
    return {"barcodes": bc, "features": feat, "X": X}


def rna_by_id(h: dict) -> tuple[sp.csr_matrix, pd.DataFrame]:
    """cells x genes counts keyed by Ensembl gene id (stable across the 2020-A and 2024-A references)."""
    keep = (h["features"].type == "Gene Expression").values
    var = pd.DataFrame({"symbol": h["features"].name.values[keep]}, index=pd.Index(h["features"].id.values[keep], name="gene_id"))
    return h["X"][keep].T.tocsr().astype(np.float32), var


def rna_matrix(h: dict) -> tuple[sp.csr_matrix, pd.Index, pd.Index]:
    """cells x genes counts (Gene Expression rows), duplicated symbols summed as in the main pipeline."""
    keep = (h["features"].type == "Gene Expression").values
    M = h["X"][keep].T.tocsr().astype(np.float32)
    sym = pd.Index(h["features"].name.values[keep])
    if sym.duplicated().any():
        codes, uniq = pd.factorize(sym)
        agg = sp.csr_matrix((np.ones(len(codes), np.float32), (np.arange(len(codes)), codes)), shape=(len(codes), len(uniq)))
        M, sym = (M @ agg).tocsr(), pd.Index(uniq)
    return M, sym, pd.Index(h["features"].id.values[keep])


def sample_peaks(h: dict) -> pd.DataFrame:
    p = h["features"][h["features"].type == "Peaks"]
    iv = p["interval"] if "interval" in p else p["id"]
    s = iv.str.extract(r"^(?P<chrom>[^:]+):(?P<start>\d+)-(?P<end>\d+)$")
    return s.assign(start=s.start.astype(np.int64), end=s.end.astype(np.int64))


def fragment_header(zpath: Path, n: int = 200) -> dict:
    with zipfile.ZipFile(zpath) as z, z.open(_member(z, "atac_fragments.tsv.gz")) as fh:
        hdr = {}
        for line in gzip.open(fh, "rt"):
            if not line.startswith("#"):
                break
            if "=" in line:
                k, v = line[2:].strip().split("=", 1)
                hdr.setdefault(k, v)
            n -= 1
            if n == 0:
                break
    return hdr


def clean_fragments(reader):
    """Coerce chunks to typed columns; malformed rows (shifted or truncated lines) are dropped, not fatal."""
    for ch in reader:
        st = pd.to_numeric(ch.start, errors="coerce"); en = pd.to_numeric(ch.end, errors="coerce")
        ok = st.notna() & en.notna() & ch.barcode.notna()
        if not ok.all():
            clean_fragments.dropped += int((~ok).sum())
        ch = ch[ok]
        yield pd.DataFrame({"chrom": ch.chrom.astype(str).values, "start": st[ok].astype(np.int64).values,
                            "end": en[ok].astype(np.int64).values, "barcode": ch.barcode.astype(str).values})


clean_fragments.dropped = 0


def iter_fragments(zpath: Path, chunk_rows: int):
    """Stream (chrom, start, end, barcode) chunks from the archive without extracting it."""
    z = zipfile.ZipFile(zpath)
    fh = z.open(_member(z, "atac_fragments.tsv.gz"))
    try:
        reader = pd.read_csv(gzip.open(fh), sep="\t", comment="#", header=None, chunksize=chunk_rows, usecols=range(4),
                             names=["chrom", "start", "end", "barcode"], dtype=str, on_bad_lines="skip")
        yield from clean_fragments(reader)
    finally:
        fh.close(); z.close()


def reduce_intervals(df: pd.DataFrame) -> pd.DataFrame:
    """Merge overlapping/adjacent intervals per chromosome; returns chrom/start/end and n_libraries support."""
    out = []
    for chrom, d in df.sort_values(["chrom", "start"]).groupby("chrom", sort=True, observed=True):
        s, e, lib = d.start.values, d.end.values, d.library.values
        run_end = np.maximum.accumulate(e)
        new = np.r_[True, s[1:] > run_end[:-1]]
        gid = np.cumsum(new) - 1
        g = pd.DataFrame({"gid": gid, "start": s, "end": e, "library": lib})
        agg = g.groupby("gid").agg(start=("start", "min"), end=("end", "max"), n_libraries=("library", "nunique"))
        out.append(agg.assign(chrom=chrom))
    return pd.concat(out, ignore_index=True)[["chrom", "start", "end", "n_libraries"]]


def ensure_gtf(url: str, want_md5: str, dst_dir: Path) -> Path:
    dst = dst_dir / url.rsplit("/", 1)[1]
    if not dst.exists():
        dst_dir.mkdir(parents=True, exist_ok=True)
        subprocess.run(["wget", "-q", "--tries=10", "-O", str(dst), url], check=True)
    got = md5(dst)
    if got != want_md5:
        raise SystemExit(f"GTF md5 mismatch {got} != {want_md5}")
    return dst


def tss_table(gtf: Path, chrom_regex: str) -> dict[str, np.ndarray]:
    """Sorted unique TSS positions (0-based) of protein-coding and lncRNA genes, per chromosome."""
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "transcript" or not re.match(chrom_regex, f[0]):
                continue
            if 'gene_type "protein_coding"' not in f[8] and 'gene_type "lncRNA"' not in f[8]:
                continue
            rows.append((f[0], int(f[3]) - 1 if f[6] == "+" else int(f[4]) - 1))
    T = pd.DataFrame(rows, columns=["chrom", "pos"]).drop_duplicates()
    return {c: np.sort(d.pos.values) for c, d in T.groupby("chrom")}


def fixed_width_consensus(P: pd.DataFrame, width: int, min_support: int) -> pd.DataFrame:
    """ArchR-style iterative overlap removal on fixed-width peaks.

    Every library peak is re-centred to `width` bp at its midpoint; each window's support is the number of
    library peaks (all libraries) overlapping it; windows are visited by decreasing support (ties: position)
    and kept unless a kept window already overlaps them. Kept windows are non-overlapping and wide-region
    chaining (a problem of reduce/merge with many samples) cannot occur."""
    half = width // 2
    out = []
    for chrom, d in P.groupby("chrom", sort=True, observed=True):
        mid = ((d.start.values + d.end.values) // 2).astype(np.int64)
        c = np.sort(mid)
        ws, we = c - half, c - half + width
        s_sorted, e_sorted = np.sort(d.start.values), np.sort(d.end.values)
        support = np.searchsorted(s_sorted, we, side="left") - np.searchsorted(e_sorted, ws, side="right")
        lo = np.searchsorted(c, c - width + 1, side="left")
        hi = np.searchsorted(c, c + width - 1, side="right")
        order = np.lexsort((c, -support))
        blocked = np.zeros(len(c), bool); keep = []
        for i in order:
            if support[i] < min_support:
                break
            if not blocked[i]:
                keep.append(i); blocked[lo[i]:hi[i]] = True
        k = np.sort(np.array(keep, dtype=np.int64))
        nlib = np.zeros(len(k), np.int64)               # distinct libraries with a peak overlapping the kept window
        for _, dl in d.groupby("library", observed=True):
            ls, le = np.sort(dl.start.values), np.sort(dl.end.values)
            nlib += (np.searchsorted(ls, we[k], side="left") - np.searchsorted(le, ws[k], side="right")) > 0
        sel = nlib >= min_support
        out.append(pd.DataFrame({"chrom": chrom, "start": np.maximum(ws[k][sel], 0), "end": we[k][sel],
                                 "n_libraries": nlib[sel], "support_peaks": support[k][sel]}))
    return pd.concat(out, ignore_index=True)


def fragment_layout(zpath: Path, n: int = 2000) -> dict:
    """Column layout of the first n data lines: number of fields, numeric start/end, barcode column guess."""
    with zipfile.ZipFile(zpath) as z, z.open(_member(z, "atac_fragments.tsv.gz")) as fh:
        rows = []
        for line in gzip.open(fh, "rt"):
            if line.startswith("#"):
                continue
            rows.append(line.rstrip("\n").split("\t"))
            if len(rows) >= n:
                break
    nf = pd.Series([len(r) for r in rows]).value_counts().to_dict()
    ok = sum(len(r) >= 4 and r[1].isdigit() and r[2].isdigit() and not r[3].isdigit() for r in rows)
    return {"n_fields": ";".join(f"{k}:{v}" for k, v in sorted(nf.items())), "standard_rows": ok, "checked_rows": len(rows),
            "example": "|".join(rows[0]) if rows else ""}
