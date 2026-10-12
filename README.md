# Mixture-of-experts model of proximal tubule cells: analysis code

Code for "Disease-Conditioned Expert Routing Identifies a Diabetes-Associated Proximal Tubule Injury Component
Linked to Albuminuria". Every parameter is in `csbj_dl/config/*.yaml`; analysis scripts write tables to
`csbj_dl/results/`, and figure scripts read only those tables.

## Setup

1. Create the environment: `conda env create -f environment.yml` and `conda activate project016`.
2. Choose a data directory with about 1 TB of free space and replace `/path/to/project016_data` in
   `csbj_dl/config/*.yaml` and `config/run.yaml` with it (for example
   `sed -i 's#/path/to/project016_data#/your/data/dir#g' csbj_dl/config/*.yaml config/run.yaml`).
3. All data are public. `csbj_dl/scripts/data/run_downloads.sh` and `run_postdownload.sh` fetch and check the KPMP
   open-access files and the GEO and Zenodo cohorts listed in `csbj_dl/config/data.yaml`.

## Order of the analysis

Run the steps in this order from the repository root; every runner sets its own working directory and creates
`csbj_dl/logs/` for its log files. Each step reads the tables written by the steps above it.

| Step | Command | Content |
|---|---|---|
| 1 Data | `bash csbj_dl/scripts/data/run_downloads.sh`, then `run_postdownload.sh` and `run_multiome.sh` (same folder) | downloads with checksums, cell tables, Multiome peak counts |
| 2 Model | `bash csbj_dl/scripts/expr/run_expr.sh` | partitions, cell labels, model selection, final models, readouts, gene attribution, signature, external cohorts |
| 3 Comparison | `bash csbj_dl/scripts/bench/run_all_bench.sh`, then `harmonise_all.sh` (same folder) | comparison methods and model variants on the same participants |
| 4 Nested selection | `bash csbj_dl/scripts/expr/run_nested_cv.sh "<devices>"` | model selection repeated inside every training partition |
| 5 Follow-up | `bash csbj_dl/scripts/expr/run_followup.sh [device]` | model without experts and its components, forms of supervision, held-out expert contributions, identities and exchanges, summaries of the nested selection, diabetes groups, clinical models, robustness of the readout, signatures by cellular source |
| 6 Chromatin | `bash csbj_dl/scripts/chrom/run_chrom.sh` | chromatin–RNA alignment, motifs, expert components in chromatin and motif accessibility (needs step 5) |
| 7 Histology | `bash csbj_dl/scripts/align/run_track_full.sh` | H&E–expression contrastive alignment |
| 8 Figures | `bash csbj_dl/scripts/figures/run_all.sh` | all figures from the result tables |

A quick check that the code is complete, without data: `python -m compileall -q csbj_dl scripts` and
`for f in csbj_dl/scripts/*/*.sh; do bash -n "$f"; done`.

Each step records its inputs, outputs, random seed and software environment in `PROVENANCE.json` next to its
tables. GPU devices are set in the configuration files and on the command line of the runners.
