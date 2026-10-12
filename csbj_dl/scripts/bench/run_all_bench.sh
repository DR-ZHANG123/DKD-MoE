#!/usr/bin/env bash
# One command for the whole benchmark: waits for the final shared inputs (expr_prep label source and
# expr CV selection), runs all methods and MoE ablations (5 seeds x 5 folds) and writes bench_method_summary.
# Usage: bash csbj_dl/scripts/bench/run_all_bench.sh [--slots cuda:2 cuda:6 ...] [--no-wait]
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate project016
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
python orchestrate.py "$@"
