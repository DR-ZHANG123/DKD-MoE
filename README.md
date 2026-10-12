# DKD-MoE

DKD-MoE is a diagnosis-conditioned mixture-of-experts variational autoencoder for single-cell and single-nucleus
transcriptomes of human proximal tubule cells. A routing network assigns each cell to a small set of experts, a
condition head reads diabetic kidney disease (DKD) against hypertensive kidney disease (HKD) from the expert
mixture, and each participant's readout decomposes into additive expert contributions. This repository contains
the model, its training and evaluation, and every analysis and figure of the accompanying article, "DKD-MoE Reveals a Diabetes-Associated
Proximal Tubule Injury Component Linked to Albuminuria".

## Layout

```
DKD-MoE/
├── environment.yml          conda environment (dkd-moe)
├── config/run.yaml          parameters of the shared preprocessing modules
├── scripts/                 shared preprocessing (lib/: reproducibility helpers; stages/: atlas and spatial modules)
└── dkd_moe/
    ├── config/              all model and analysis parameters (*.yaml)
    ├── scripts/
    │   ├── data/            downloads, cell tables, Multiome peak counts
    │   ├── expr/            DKD-MoE model, training, readouts, attribution, signatures, clinical models
    │   ├── bench/           comparison methods and model variants
    │   ├── chrom/           chromatin–RNA alignment and motif accessibility
    │   ├── align/           H&E–expression contrastive alignment
    │   ├── figures/         figures drawn from the result tables
    │   └── common/          shared helpers
    └── logs/                written by the runners
```

Analysis scripts write tables to `dkd_moe/results/`; figure scripts read only those tables.

## Setup

1. Create the environment: `conda env create -f environment.yml` and `conda activate dkd-moe`.
2. Choose a data directory with about 1 TB of free space and replace `/path/to/dkd_moe_data` in
   `dkd_moe/config/*.yaml` and `config/run.yaml` with it (for example
   `sed -i 's#/path/to/dkd_moe_data#/your/data/dir#g' dkd_moe/config/*.yaml config/run.yaml`).
3. All data are public. `dkd_moe/scripts/data/run_downloads.sh` and `run_postdownload.sh` fetch and check the KPMP
   open-access files and the GEO and Zenodo cohorts listed in `dkd_moe/config/data.yaml`.

## Running the analysis

Run the steps in this order from the repository root; every runner sets its own working directory and writes its
log files to `dkd_moe/logs/`. Each step reads the tables written by the steps above it.

| Step | Command | Content |
|---|---|---|
| 1 Data | `bash dkd_moe/scripts/data/run_downloads.sh`, then `run_postdownload.sh` and `run_multiome.sh` (same folder) | downloads with checksums, cell tables, Multiome peak counts |
| 2 Model | `bash dkd_moe/scripts/expr/run_expr.sh` | partitions, cell labels, model selection, final models, readouts, gene attribution, signature, external cohorts |
| 3 Comparison | `bash dkd_moe/scripts/bench/run_all_bench.sh`, then `harmonise_all.sh` (same folder) | comparison methods and model variants on the same participants |
| 4 Nested selection | `bash dkd_moe/scripts/expr/run_nested_cv.sh "<devices>"` | model selection repeated inside every training partition |
| 5 Follow-up | `bash dkd_moe/scripts/expr/run_followup.sh [device]` | model without experts and its components, forms of supervision, held-out expert contributions, identities and exchanges, cell-level map of routing, diabetes groups, clinical models, robustness of the readout, signatures by cellular source |
| 6 Chromatin | `bash dkd_moe/scripts/chrom/run_chrom.sh` | chromatin–RNA alignment, motifs, expert components in chromatin and motif accessibility (needs step 5) |
| 7 Histology | `bash dkd_moe/scripts/align/run_track_full.sh` | H&E–expression contrastive alignment |
| 8 Figures | `bash dkd_moe/scripts/figures/run_all.sh` | all figures from the result tables |

GPU devices are set in the configuration files (`DKDMOE_DEVICE` overrides them) and on the command line of the
runners. A quick check that the code is complete, without data: `python -m compileall -q dkd_moe scripts` and
`for f in dkd_moe/scripts/*/*.sh; do bash -n "$f"; done`.

Each step records its inputs, outputs, random seed and software environment in `PROVENANCE.json` next to its
tables.
