"""07_motifs: transcription-factor motif enrichment of the top-attributed peaks vs GC-matched background.

Foreground: top attrib.n_top peaks on each side of the consensus attribution (DKD side = raise the
chromatin-side DKD-vs-HKD score; HKD side = lower it). Background: motif.bg_ratio peaks per foreground peak,
drawn from the remaining attributed peaks within the same GC-content bin (quantile bins). Motifs: JASPAR 2024
CORE vertebrates (latest version of each matrix), scanned on both strands with MOODS at motif.pvalue against
the peak-pool base composition. Per motif: share of peaks with >= 1 hit, odds ratio, Fisher exact P, BH q.
Outputs: dkd_moe/results/chrom_motifs/{motif_enrichment.tsv, motif_report.tsv, background_gc.tsv, summary.json}
"""
from __future__ import annotations
import json, sys
from multiprocessing import Pool
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, P, result_dir, write_provenance, set_global_seed  # noqa: E402

_SC = None


def seqs(fa_path: Path, D: pd.DataFrame) -> list[str]:
    import pyfaidx
    fa = pyfaidx.Fasta(str(fa_path), sequence_always_upper=True)
    return [str(fa[ch][s:e]) for ch, s, e in zip(D.chrom, D.start, D.end)]


def gc(s: str) -> float:
    acgt = sum(s.count(b) for b in "ACGT")
    return (s.count("G") + s.count("C")) / acgt if acgt else np.nan


def motifs(M: dict):
    from pyjaspar import jaspardb
    ms = jaspardb(release=M["jaspar_release"]).fetch_motifs(collection=M["collection"], tax_group=[M["tax_group"]],
                                                            all_versions=False)
    return [(m.matrix_id, m.name, [list(map(float, m.counts[b])) for b in "ACGT"]) for m in ms]


def _init(mats, bg, thr):
    global _SC
    import MOODS.scan
    _SC = MOODS.scan.Scanner(7); _SC.set_motifs(mats, bg, thr)


def _scan(s: str) -> np.ndarray:
    res = _SC.scan(s)
    return np.array([len(r) > 0 for r in res], dtype=bool)


def presence(sq: list[str], mot, bg, M: dict, n_jobs: int = 32) -> np.ndarray:
    import MOODS.tools
    mats, thr = [], []
    for _, _, cnt in mot:
        lo = MOODS.tools.log_odds(cnt, bg, M["pseudocount"])
        for mm in (lo, MOODS.tools.reverse_complement(lo)):
            mats.append(mm); thr.append(MOODS.tools.threshold_from_p(mm, bg, M["pvalue"]))
    with Pool(n_jobs, initializer=_init, initargs=(mats, bg, thr)) as pool:
        H = np.array(pool.map(_scan, sq, chunksize=50))
    return H[:, 0::2] | H[:, 1::2]                  # either strand


def gc_matched(fg_gc: np.ndarray, pool_gc: pd.Series, ratio: int, bins: int, rng) -> pd.Index:
    edges = np.unique(np.quantile(pool_gc.dropna(), np.linspace(0, 1, bins + 1)))
    pb = np.clip(np.searchsorted(edges, pool_gc.values, side="right") - 1, 0, len(edges) - 2)
    fb = np.clip(np.searchsorted(edges, fg_gc, side="right") - 1, 0, len(edges) - 2)
    out = []
    for b, n in zip(*np.unique(fb, return_counts=True)):
        cand = pool_gc.index[pb == b]
        out.append(rng.choice(cand, min(len(cand), ratio * n), replace=False))
    return pd.Index(np.concatenate(out))


def main():
    c = cfg(); M = c["motif"]; set_global_seed(c["seed"]); rng = np.random.default_rng(c["seed"])
    out = result_dir("chrom_motifs"); att = result_dir("chrom_attrib")
    D = pd.read_csv(att / "peak_attributions.tsv.gz", sep="\t", index_col=0)
    D = D[D.mean_attribution != 0]                                     # peaks used by at least one model
    top = pd.read_csv(att / "top_peaks.tsv", sep="\t", index_col=0)
    pool = D.drop(index=top.index.intersection(D.index))
    print("pool peaks", len(pool), flush=True)
    pool_seq = seqs(P(c, "genome_fa"), pool)
    pool_gc = pd.Series([gc(s) for s in pool_seq], index=pool.index)
    import MOODS.tools
    bg = list(MOODS.tools.bg_from_sequence_dna("".join(pool_seq[::20]), 1))
    mot = motifs(M)
    rows, gcrows, summ = [], [], {"n_motifs": len(mot), "background_composition": bg}
    for direction in ("DKD_side", "HKD_side"):
        fg = top[top.direction == direction]
        fseq = seqs(P(c, "genome_fa"), fg); fgc = np.array([gc(s) for s in fseq])
        bidx = gc_matched(fgc, pool_gc, M["bg_ratio"], M["gc_bins"], rng)
        bseq = [pool_seq[i] for i in pool.index.get_indexer(bidx)]
        gcrows.append({"direction": direction, "n_fg": len(fg), "n_bg": len(bidx), "gc_fg_mean": float(np.nanmean(fgc)),
                       "gc_bg_mean": float(pool_gc.loc[bidx].mean())})
        Hf, Hb = presence(fseq, mot, bg, M), presence(bseq, mot, bg, M)
        for j, (mid, name, _) in enumerate(mot):
            a, b = int(Hf[:, j].sum()), int(Hb[:, j].sum())
            tab = [[a, len(fseq) - a], [b, len(bseq) - b]]
            odds = ((a + 0.5) * (len(bseq) - b + 0.5)) / ((len(fseq) - a + 0.5) * (b + 0.5))
            rows.append({"direction": direction, "motif_id": mid, "motif": name, "fg_hits": a, "fg_n": len(fseq),
                         "bg_hits": b, "bg_n": len(bseq), "fg_frac": a / len(fseq), "bg_frac": b / len(bseq),
                         "log2_or": float(np.log2(odds)), "p_fisher": stats.fisher_exact(tab)[1]})
        print(direction, "scanned", flush=True)
    E = pd.DataFrame(rows)
    E["q_bh"] = E.groupby("direction").p_fisher.transform(lambda p: multipletests(p, method="fdr_bh")[1])
    E["rank_in_direction"] = E.groupby("direction").p_fisher.rank(method="first")
    E = E.sort_values(["direction", "p_fisher"]); E.to_csv(out / "motif_enrichment.tsv", sep="\t", index=False)
    rep = set(x.upper() for x in M["report"])
    R = E[E.motif.str.upper().isin(rep) | (E.rank_in_direction <= 25)]
    R = R.assign(requested=R.motif.str.upper().isin(rep)); R.to_csv(out / "motif_report.tsv", sep="\t", index=False)
    pd.DataFrame(gcrows).to_csv(out / "background_gc.tsv", sep="\t", index=False)
    summ["n_significant_q05"] = E[E.q_bh < 0.05].groupby("direction").size().to_dict()
    (out / "summary.json").write_text(json.dumps(summ, indent=2))
    pd.set_option("display.width", 220)
    print(pd.DataFrame(gcrows)); print(R[["direction", "motif", "fg_frac", "bg_frac", "log2_or", "p_fisher", "q_bh",
                                          "rank_in_direction", "requested"]].round(4).to_string(index=False))
    write_provenance("chrom_motifs", [att / "peak_attributions.tsv.gz", att / "top_peaks.tsv", P(c, "genome_fa")],
                     [out / f for f in ("motif_enrichment.tsv", "motif_report.tsv", "background_gc.tsv", "summary.json")],
                     c["seed"], c, extra=summ)


if __name__ == "__main__":
    main()
