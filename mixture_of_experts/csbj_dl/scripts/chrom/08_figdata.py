"""08_figdata: figure source tables for the chromatin analyses (csbj_dl/figures/source_data/chrom_*.tsv) and a
run summary (csbj_dl/results/chrom_summary/summary.json) recording which Multiome build the run used."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import CDL, cfg, P, result_dir, write_provenance, multiome_complete, md5  # noqa: E402

TABLES = {
    "chrom_label_counts.tsv": ("chrom_labels", "counts_by_participant.tsv"),
    "chrom_ext_label_counts.tsv": ("chrom_ext_labels", "counts_by_donor.tsv"),
    "chrom_rna_participants.tsv": ("chrom_rna", "participant_readout.tsv"),
    "chrom_rna_tests.tsv": ("chrom_rna", "tests.tsv"),
    "chrom_retrieval.tsv": ("chrom_readout", "retrieval_by_model.tsv"),
    "chrom_retrieval_summary.tsv": ("chrom_readout", "retrieval_summary.tsv"),
    "chrom_kpmp_participants.tsv": ("chrom_readout", "participant_scores.tsv"),
    "chrom_kpmp_tests.tsv": ("chrom_readout", "kpmp_tests.tsv"),
    "chrom_kpmp_agreement.tsv": ("chrom_readout", "kpmp_agreement.tsv"),
    "chrom_external_donors.tsv": ("chrom_readout", "external_donor_scores.tsv"),
    "chrom_external_tests.tsv": ("chrom_readout", "external_tests.tsv"),
    "chrom_top_peaks.tsv": ("chrom_attrib", "top_peaks.tsv"),
    "chrom_top_genes.tsv": ("chrom_attrib", "top_genes.tsv"),
    "chrom_motifs.tsv": ("chrom_motifs", "motif_report.tsv"),
    "chrom_motifs_all.tsv": ("chrom_motifs", "motif_enrichment.tsv"),
}


def main():
    c = cfg(); sd = CDL / "figures" / "source_data"; sd.mkdir(parents=True, exist_ok=True)
    outs = []
    for name, (d, f) in TABLES.items():
        src = result_dir(d) / f
        if src.exists():
            pd.read_csv(src, sep="\t").to_csv(sd / name, sep="\t", index=False); outs.append(sd / name)
    lt = pd.read_csv(P(c, "library_table"), sep="\t")
    nuc = pd.read_csv(result_dir("chrom_align") / "nuclei.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    summ = {"multiome_build": multiome_complete(), "library_table_rows": int(len(lt)),
            "library_table_groups": lt.group.value_counts().to_dict(),
            "multiome_rna_md5": md5(P(c, "multiome_rna")), "multiome_atac_md5": md5(P(c, "multiome_atac")),
            "pt_nuclei_used": int(len(nuc)), "participants_used": int(nuc.participant.nunique()),
            "participants_by_group": nuc.groupby("group").participant.nunique().to_dict(),
            "injured_nuclei_by_group": nuc[nuc.pt_state == "injured_PT"].groupby("group").size().to_dict()}
    out = result_dir("chrom_summary"); (out / "summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))
    write_provenance("chrom_summary", [result_dir(d) / f for d, f in TABLES.values() if (result_dir(d) / f).exists()],
                     outs + [out / "summary.json"], c["seed"], c, extra=summ)


if __name__ == "__main__":
    main()
