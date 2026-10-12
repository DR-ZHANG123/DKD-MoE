#!/usr/bin/env bash
# Harmonised evaluation (same units, groups and external rule as the expr readout):
#  1) component variants of the final MoE model + extraction of the final model (warm_ablation.py)
#  2) injured-PT per-cell arrays from every saved baseline / random-init MoE model (harm_cells.py)
#  3) participant scores and the summary (harm_score.py, harm_summary.py)
#  4) technology mixing of the final model and its variants (warm_mixing.py); supplementary-table inputs
#     restricted to the models in the article (common/final_tables.py)
# Usage: bash dkd_moe/scripts/bench/harmonise_all.sh "cuda:6 cuda:0 cuda:2 cuda:4 cuda:5 cuda:7"
set -uo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate dkd-moe
HERE="$(cd "$(dirname "$0")" && pwd)"; LOG="$HERE/../../logs"; cd "$HERE"
mkdir -p "$LOG"
read -r -a GPUS <<< "${1:-cuda:6}"
export PYTHONUNBUFFERED=1
i=0
pick() { DEV=${GPUS[$((i % ${#GPUS[@]}))]}; i=$((i + 1)); }
pick; python warm_ablation.py --variant full --seeds 1 2 3 4 5 --device "$DEV" > "$LOG/bench_harm_warm_full.log" 2>&1
pids=()
for v in K1 no_adv no_cond dense; do
  for s in 1 2 3 4 5; do
    pick; python warm_ablation.py --variant $v --seeds $s --device "$DEV" > "$LOG/bench_harm_warm_${v}_s$s.log" 2>&1 &
    pids+=($!)
  done
done
for r in scvi cvae scdisinfact moe_full moe_K1 moe_no_adv moe_no_cond moe_dense; do
  pick; ( for s in 1 2 3 4 5; do python harm_cells.py --run ${r}_s$s --device "$DEV"; done ) \
    > "$LOG/bench_harm_cells_$r.log" 2>&1 &
  pids+=($!)
done
fail=0; for p in "${pids[@]}"; do wait "$p" || fail=1; done
python harm_score.py > "$LOG/bench_harm_score.log" 2>&1 || fail=1
python harm_summary.py > "$LOG/bench_harm_summary.log" 2>&1 || fail=1
pick; python warm_mixing.py --device "$DEV" > "$LOG/bench_warm_mixing.log" 2>&1 || fail=1
python warm_mixing.py --table-only >> "$LOG/bench_warm_mixing.log" 2>&1 || fail=1
python ../common/final_tables.py > "$LOG/final_tables.log" 2>&1 || fail=1
exit $fail
