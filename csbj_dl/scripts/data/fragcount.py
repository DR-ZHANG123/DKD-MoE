"""Single-pass fragment counter shared by KPMP Multiome and external snATAC libraries.

For a set of cell barcodes and a stream of fragment chunks (chrom, start, end, barcode, count):
- n_fragments: unique fragments (rows) per cell;
- peak matrix: Tn5 cut sites (start and end-1) inside the consensus peaks (non-overlapping intervals);
- gene-activity matrix: fragments overlapping gene body + gene_upstream bp of protein-coding genes,
  each fragment counted once per gene (Signac GeneActivity definition), via an elementary-segment map
  so that overlapping genes are handled exactly at fragment ends;
- frip, tss_enrichment (cut sites/bp within +/-W of a TSS over cut sites/bp in the flank band, flank
  pseudocount 1) and nucleosome_signal (147-294 bp over < 147 bp fragments, pseudocount 1)."""
from __future__ import annotations
import gzip
import numpy as np
import pandas as pd
import scipy.sparse as sp


def peak_index(C: pd.DataFrame) -> dict:
    idx = {}
    for chrom, d in C.groupby("chrom", sort=False):
        d = d.sort_values("start")
        idx[chrom] = (d.start.values, d.end.values, d.index.values)
    return idx


def in_interval(pos, starts, ends, ids) -> np.ndarray:
    i = np.searchsorted(starts, pos, side="right") - 1
    ok = i >= 0
    ok[ok] = pos[ok] < ends[i[ok]]
    out = np.full(len(pos), -1, np.int64)
    out[ok] = ids[i[ok]]
    return out


def tss_dist(pos, tss) -> np.ndarray:
    j = np.searchsorted(tss, pos)
    left = np.abs(pos - tss[np.clip(j - 1, 0, len(tss) - 1)])
    right = np.abs(tss[np.clip(j, 0, len(tss) - 1)] - pos)
    return np.minimum(left, right)


def gene_regions(gtf, chrom_regex: str, upstream: int) -> pd.DataFrame:
    """Protein-coding genes (GENCODE gene records): gene body extended `upstream` bp 5'."""
    import re
    rows = []
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 9)
            if f[2] != "gene" or 'gene_type "protein_coding"' not in f[8] or not re.match(chrom_regex, f[0]):
                continue
            name = re.search(r'gene_name "([^"]+)"', f[8]).group(1)
            s, e = int(f[3]) - 1, int(f[4])
            s, e = (max(0, s - upstream), e) if f[6] == "+" else (s, e + upstream)
            rows.append((f[0], s, e, name))
    return pd.DataFrame(rows, columns=["chrom", "start", "end", "gene"])


def segment_map(G: pd.DataFrame) -> tuple[dict, sp.csr_matrix, pd.Index]:
    """Elementary segments per chromosome (breakpoints at all gene edges) and a segment x gene 0/1 map."""
    genes = pd.Index(sorted(G.gene.unique()))
    gcode = genes.get_indexer(G.gene)
    seg_idx, rows, cols, off = {}, [], [], 0
    for chrom, d in G.assign(code=gcode).groupby("chrom"):
        bp = np.unique(np.r_[d.start.values, d.end.values])
        starts, ends = bp[:-1], bp[1:]
        ids = np.arange(len(starts)) + off
        for s, e, g in zip(d.start.values, d.end.values, d.code.values):
            a, b = np.searchsorted(starts, s), np.searchsorted(starts, e)
            rows.append(ids[a:b]); cols.append(np.full(b - a, g))
        seg_idx[chrom] = (starts, ends, ids)
        off += len(starts)
    r, c = np.concatenate(rows), np.concatenate(cols)
    S = sp.csr_matrix((np.ones(len(r), np.int8), (r, c)), shape=(off, len(genes)))
    S.data[:] = 1
    return seg_idx, S, genes


def count(chunks, barcodes, C: pd.DataFrame, tss: dict, genemap, M: dict):
    """Returns (obs DataFrame indexed by barcode, peak csr cells x peaks, gene csr cells x genes)."""
    n = len(barcodes)
    bmap = pd.Series(np.arange(n), index=pd.Index(barcodes))
    pidx = peak_index(C); npk = len(C)
    seg_idx, S, genes = genemap
    z = lambda: np.zeros(n, np.int64)
    nfrag, cuts, cuts_pk, tss_c, tss_f, mono, nfr = z(), z(), z(), z(), z(), z(), z()
    W, (f0, f1) = M["tss_window"], M["tss_flank"]
    P = sp.csr_matrix((n, npk), dtype=np.int32); A = sp.csr_matrix((n, len(genes)), dtype=np.int32)
    for ch in chunks:
        cell = ch.barcode.map(bmap)
        keep = cell.notna().values
        if not keep.any():
            continue
        ch = ch[keep]; cell = cell.values[keep].astype(np.int64)
        st, en = ch.start.values.astype(np.int64), ch.end.values.astype(np.int64)
        nfrag += np.bincount(cell, minlength=n)
        ln = en - st
        nfr += np.bincount(cell[ln < 147], minlength=n); mono += np.bincount(cell[(ln >= 147) & (ln <= 294)], minlength=n)
        chrom = ch.chrom.astype(str).values
        pr, pc, gr, gc = [], [], [], []
        for cname in np.unique(chrom):
            m = chrom == cname
            cm = cell[m]; cc = np.r_[cm, cm]
            pos = np.r_[st[m], en[m] - 1]
            cuts += np.bincount(cc, minlength=n)
            if cname in pidx:
                pk = in_interval(pos, *pidx[cname]); hit = pk >= 0
                pr.append(cc[hit]); pc.append(pk[hit])
            if cname in tss:
                d = tss_dist(pos, tss[cname])
                tss_c += np.bincount(cc[d <= W], minlength=n)
                tss_f += np.bincount(cc[(d >= f0) & (d <= f1)], minlength=n)
            if cname in seg_idx:
                sg = in_interval(pos, *seg_idx[cname])
                nf = m.sum(); fid = np.r_[np.arange(nf), np.arange(nf)]
                hit = sg >= 0
                if hit.any():
                    F = sp.csr_matrix((np.ones(hit.sum(), np.int8), (fid[hit], sg[hit])), shape=(nf, S.shape[0]))
                    FG = (F @ S).tocoo()                       # fragment x gene, >0 if either end in gene region
                    gr.append(cm[FG.row]); gc.append(FG.col)   # one count per fragment-gene pair
        if pr:
            r, c_ = np.concatenate(pr), np.concatenate(pc)
            cuts_pk += np.bincount(r, minlength=n)
            P = P + sp.csr_matrix((np.ones(len(r), np.int32), (r, c_)), shape=(n, npk))
        if gr:
            r, c_ = np.concatenate(gr), np.concatenate(gc)
            A = A + sp.csr_matrix((np.ones(len(r), np.int32), (r, c_)), shape=(n, len(genes)))
    obs = pd.DataFrame({"n_fragments": nfrag, "n_cutsites": cuts, "frip": cuts_pk / np.maximum(cuts, 1),
                        "tss_enrichment": (tss_c / (2 * W + 1)) / ((tss_f + 1) / (2 * (f1 - f0 + 1))),
                        "nucleosome_signal": (mono + 1) / (nfr + 1)}, index=pd.Index(barcodes, name="barcode"))
    return obs, P.tocsr(), A.tocsr(), genes


def read_fragments_plain(path, chunk_rows: int):
    from mio import clean_fragments
    reader = pd.read_csv(path, sep="\t", comment="#", header=None, chunksize=chunk_rows, usecols=range(4),
                         names=["chrom", "start", "end", "barcode"], dtype=str, on_bad_lines="skip")
    return clean_fragments(reader)
