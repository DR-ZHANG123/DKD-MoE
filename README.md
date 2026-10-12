# dkd-injured-tubule-state

Code, configuration and result tables for the study

> **Injured proximal tubule cells in diabetic kidney disease carry a transcriptional shift beyond the common injury response that accompanies albuminuria**

The study compares injured proximal tubule (iPT) cells between diabetic kidney disease (DKD) and hypertensive chronic kidney disease (HKD). A discovery atlas of single-nucleus data (GSE211785) defines a DKD-associated shift within iPT cells using donor-level pseudobulk statistics and a disentangled representation model (scDisInFact). The marker genes and the primary test were fixed before any validation data were analysed and were then tested in independent cohorts: Kidney Precision Medicine Project (KPMP) single-nucleus and single-cell data, GSE195460, ERCB tubulointerstitial arrays (GSE104954), an imaging-based spatial atlas (Zenodo 19868428), Visium sections with same-section H&E, whole-slide images and pathology descriptor scores.

## Mixture-of-experts model (CSBJ study)

The folder [`mixture_of_experts/`](mixture_of_experts/) holds the code for the study

> **Disease-Conditioned Expert Routing Identifies a Diabetes-Associated Proximal Tubule Injury Component Linked to Albuminuria**

in which a mixture-of-experts generative model of proximal tubule cells is trained on KPMP single-cell and single-nucleus data, with diagnosis expressed through the routing of cells to experts, and tested in held-out participants, external cohorts, histology, chromatin and clinical data. It is self-contained (own configuration, environment and run order); see [`mixture_of_experts/README.md`](mixture_of_experts/README.md).

## Data

All data are public. Accessions, files and roles are listed in `metadata/data_sources.tsv`; `scripts/stages/00_download.sh`, `00b_download_kpmp.py` and `00c_download_published_signatures.sh` fetch the inputs and record checksums. KPMP open-access files are listed from the public repository index (`scripts/stages/kpmp/`). Raw and processed data are written to `data/` (not tracked).

## Environment

```bash
conda env create -f environment.yml     # environment "project016"
```

## Running the pipeline

```bash
bash run_all.sh            # stages with an existing results/<stage>/PROVENANCE.json are skipped
bash run_all.sh --force    # rerun everything
```

All parameters are in `config/run.yaml`. Every stage writes its tables to `results/<stage>/` with a `PROVENANCE.json` (input and output checksums, seed, configuration hash, package versions). The global seed is 20260929.

| Stages | Content |
|---|---|
| 00, 00b, 00c | Downloads and checksums (GEO, Zenodo, KPMP open access, published signatures) |
| 01 | Donor-by-modality inventory and overlap between sources |
| 02–02c | Discovery pseudobulk, replication cohort preparation, library quality control and protocol batches |
| 03–03d | Donor-level differential expression, technical-axis diagnostic, leave-one-donor-out stability, composition |
| 04–05 | scDisInFact models and marker-gene definition |
| 06–08 | GSE195460, KPMP single-nucleus/single-cell and ERCB array tests; transcriptome-wide agreement |
| 09 | Pathway analysis |
| 10–13 | Visium, spot-level histology, whole-slide images |
| 12a–12c | Technical robustness, imaging spatial atlas, clinical and pathology associations |
| 14a–14h | Permutation-based agreement, albuminuria models, published proximal tubule states, leukocyte RNA, per-cohort pathways, recruitment site, diabetic-exposure ordering, injury-intensity adjustment |
| 15 | Summary of pre-specified tests |

