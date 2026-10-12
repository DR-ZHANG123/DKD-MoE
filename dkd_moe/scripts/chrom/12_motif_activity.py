"""12_motif_activity: accessibility of transcription factor motifs per nucleus, corrected for GC content and peak
accessibility (deviation approach of chromVAR).

All consensus peaks are scanned (JASPAR 2024 CORE vertebrates, latest version of each named motif, MOODS at
motif.pvalue, either strand). For motif m with peak indicator h and nucleus n with accessible-peak count N_n,
observed O_nm = sum of accessible motif peaks; expected E_nm = N_n * sum_n O_nm / sum_n N_n; deviation
(O - E) / E. The same deviation is computed for motif_activity.n_background peak sets in which every motif peak is
replaced by a random peak of the same GC-content and accessibility bin; the motif score is the deviation
standardized by the mean and SD of its background deviations.
Outputs: dkd_moe/results/chrom_motif_activity/{nucleus_scores.tsv.gz, motif_peaks.tsv} + PROVENANCE.json
"""
from __future__ import annotations
import importlib.util, sys
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, P, result_dir, write_provenance, set_global_seed  # noqa: E402
from chrom_data import load_multiome  # noqa: E402

_spec = importlib.util.spec_from_file_location("motifs07", Path(__file__).resolve().parent / "07_motifs.py")
M07 = importlib.util.module_from_spec(_spec); sys.modules["motifs07"] = M07; _spec.loader.exec_module(M07)


def named_motifs(M: dict, names: list[str]):
    want = {n.upper() for n in names}
    return [m for m in M07.motifs(M) if m[1].upper() in want]


def bins(v: np.ndarray, k: int) -> np.ndarray:
    e = np.unique(np.quantile(v, np.linspace(0, 1, k + 1)))
    return np.clip(np.searchsorted(e, v, side="right") - 1, 0, len(e) - 2)


def background_sets(hit: np.ndarray, cell: np.ndarray, n_bg: int, rng) -> sp.csc_matrix:
    """Peaks x n_bg count matrix: each motif peak replaced by a random peak of the same bin, per set."""
    rows, cols = [], []
    pools = {b: np.where(cell == b)[0] for b in np.unique(cell)}
    hb = cell[hit]
    for j in range(n_bg):
        for b, n in zip(*np.unique(hb, return_counts=True)):
            rows.append(rng.choice(pools[b], n, replace=True)); cols.append(np.full(n, j))
    r, c_ = np.concatenate(rows), np.concatenate(cols)
    return sp.csc_matrix((np.ones(len(r), np.float32), (r, c_)), shape=(len(cell), n_bg))


def deviation(X: sp.csr_matrix, H: sp.csc_matrix, N: np.ndarray) -> np.ndarray:
    O = np.asarray((X @ H).todense(), dtype=np.float64)
    E = N[:, None] * (O.sum(0) / N.sum())[None, :]
    return (O - E) / np.maximum(E, 1e-9)


def main():
    c = cfg(); MA = c["motif_activity"]; M = c["motif"]; set_global_seed(c["seed"]); rng = np.random.default_rng(c["seed"])
    out = result_dir("chrom_motif_activity")
    obs, _, X, peak_names = load_multiome(c, tuple(c["align"]["cells"]), rna_genes=None, peaks=True)
    pk = pd.read_csv(P(c, "consensus_peaks"), sep="\t").set_index("peak").reindex(peak_names)
    assert pk.chrom.notna().all()
    sq = M07.seqs(P(c, "genome_fa"), pk.reset_index())
    gc = np.array([M07.gc(s) for s in sq]); gc = np.where(np.isnan(gc), np.nanmean(gc), gc)
    access = np.asarray(X.mean(0)).ravel()
    cell = bins(gc, MA["gc_bins"]) * MA["access_bins"] + bins(access, MA["access_bins"])
    import MOODS.tools
    bg = list(MOODS.tools.bg_from_sequence_dna("".join(sq[::50]), 1))
    mot = named_motifs(M, MA["motifs"])
    print("motifs", [m[1] for m in mot], "peaks", len(sq), "nuclei", X.shape[0], flush=True)
    Hit = M07.presence(sq, mot, bg, M)
    N = np.asarray(X.sum(1)).ravel().astype(np.float64)
    res, mp = {}, []
    dup = {n for n in [m[1] for m in mot] if [m[1] for m in mot].count(n) > 1}
    for j, (mid, nm, _) in enumerate(mot):
        name = f"{nm} ({mid})" if nm in dup else nm          # several JASPAR matrices share one factor name
        h = Hit[:, j]
        mp.append({"motif": name, "motif_id": mid, "n_peaks": int(h.sum()), "frac_peaks": float(h.mean())})
        d = deviation(X, sp.csc_matrix(h.astype(np.float32)[:, None]), N)[:, 0]
        B = deviation(X, background_sets(h, cell, MA["n_background"], rng), N)
        res[name] = (d - B.mean(1)) / (B.std(1) + 1e-9)
        print(name, int(h.sum()), "peaks", flush=True)
    S = pd.concat([obs[["participant", "library", "group", "fold", "pt_state", "n_fragments", "tss_enrichment", "frip"]],
                   pd.DataFrame(res, index=obs.index)], axis=1)
    S.to_csv(out / "nucleus_scores.tsv.gz", sep="\t")
    pd.DataFrame(mp).to_csv(out / "motif_peaks.tsv", sep="\t", index=False)
    write_provenance("chrom_motif_activity", [P(c, "consensus_peaks"), P(c, "genome_fa")],
                     [out / "nucleus_scores.tsv.gz", out / "motif_peaks.tsv"], c["seed"], c,
                     extra={"motifs": [m[1] for m in mot]})


if __name__ == "__main__":
    main()
