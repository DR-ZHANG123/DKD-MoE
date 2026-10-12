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

| Step | Command | Content |
|---|---|---|
| Data | `bash csbj_dl/scripts/data/run_downloads.sh`, `run_postdownload.sh`, `run_multiome.sh` | downloads, cell tables, Multiome peak counts |
| Model | `bash csbj_dl/scripts/expr/run_expr.sh` | partitions, cell labels, model selection, final models, readouts, gene attribution, signature, external cohorts |
| Comparison | `bash csbj_dl/scripts/bench/run_all_bench.sh`, `harmonise_all.sh` | comparison methods and model variants on the same participants |
| Follow-up | `bash csbj_dl/scripts/expr/run_followup.sh`, `run_nested_cv.sh` | nested model selection, model without experts, forms of supervision, held-out expert contributions and exchanges, diabetes groups, clinical models, robustness of the readout, signatures by cellular source |
| Histology | `bash csbj_dl/scripts/align/run_track_full.sh` | H&E–expression contrastive alignment |
| Chromatin | `bash csbj_dl/scripts/chrom/run_chrom.sh` | chromatin–RNA contrastive alignment, motifs |
| Figures | `bash csbj_dl/scripts/figures/run_all.sh` | all figures from the result tables |

Each step records its inputs, outputs, random seed and software environment in `PROVENANCE.json` next to its
tables. GPU devices are set in the configuration files and on the command line of the runners.
