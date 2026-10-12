#!/usr/bin/env bash
# Full nested cross-validation (expr.yaml nested_cv:). Workers share one inner job list (atomic claims).
# Usage: bash csbj_dl/scripts/expr/run_nested_cv.sh "<device list, one entry per worker>" [threads_per_worker]
# The run in results/expr_nested_cv used 18 GPU workers (3 each on cuda:1-3, 2 each on cuda:0,4,6,7, 1 on cuda:5;
# the first six with 16 threads, the rest with 4; one early CPU worker was stopped before finishing a job).
# Devices are recorded per job in results/expr_nested_cv/curves/.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
L=csbj_dl/logs; S=csbj_dl/scripts/expr; DEVS=(${1:-cpu}); TH=${2:-4}
mkdir -p "$L"
ALL=$(python $S/nested_jobs.py 1)
n=0; for d in "${DEVS[@]}"; do
  python $S/nested_inner.py --jobs $ALL --shared --device $d --threads $TH > $L/expr_nested_inner_w$n.log 2>&1 &
  n=$((n+1)); done; wait
python $S/nested_select.py                    > $L/expr_nested_select.log 2>&1
n=0; for d in "${DEVS[@]}"; do
  python $S/nested_outer.py --shared --device $d --threads $TH > $L/expr_nested_outer_w$n.log 2>&1 &
  n=$((n+1)); done; wait
python $S/nested_summary.py                   > $L/expr_nested_summary.log 2>&1
