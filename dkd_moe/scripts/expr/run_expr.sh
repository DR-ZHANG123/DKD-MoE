#!/usr/bin/env bash
# expr module orchestrator (idempotent: CV / final jobs skip existing outputs).
# Usage: bash dkd_moe/scripts/expr/run_expr.sh [n_parallel_cv_workers] [device]
# CV and final models were trained on CPU (--device cpu --threads 24).
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate dkd-moe
L=dkd_moe/logs; S=dkd_moe/scripts; NW=${1:-6}; DEV=${2:-cpu}; DA="--device $DEV --threads 24"
mkdir -p "$L"
python $S/common/make_splits.py              > $L/expr_splits.log 2>&1
python $S/expr/kpmp_labels.py sn               > $L/expr_labels_sn.log 2>&1
python $S/expr/kpmp_labels.py sc               > $L/expr_labels_sc.log 2>&1
python $S/expr/prep_kpmp.py                  > $L/expr_prep.log 2>&1
python $S/expr/prep_external.py             >> $L/expr_prep.log 2>&1
# load-balance x technology-adversary weight comparison on KPMP fold 0 (K=6)
for ba in "10 5" "50 5" "10 20" "50 20"; do set -- $ba
  python $S/expr/cv_worker.py --jobs 6:10.0:1:0 --epochs 10 --prefix tune_b$1_a$2 --wbal $1 --wadv $2 > $L/expr_tune_b$1_a$2.log 2>&1 &
done; wait
# participant-level 5-fold CV grid
jobs=(); for K in $(python -c "import yaml;print(*yaml.safe_load(open('dkd_moe/config/expr.yaml'))['cv']['K_grid'])"); do
  for wc in $(python -c "import yaml;print(*yaml.safe_load(open('dkd_moe/config/expr.yaml'))['cv']['w_cond_grid'])"); do
    for s in 1 2; do for f in 0 1 2 3 4; do jobs+=("$K:$wc:$s:$f"); done; done; done; done
for w in $(seq 0 $((NW-1))); do sel=(); for i in "${!jobs[@]}"; do [ $((i%NW)) -eq $w ] && sel+=("${jobs[$i]}"); done
  python $S/expr/cv_worker.py --jobs "${sel[@]}" $DA > $L/expr_cv_w$w.log 2>&1 & done; wait
python $S/expr/cv_summary.py                 > $L/expr_cv_summary.log 2>&1
# final models (5 seeds, all KPMP) + external forward inference
python $S/expr/final_worker.py --seeds 1 2 3 $DA > $L/expr_final_a.log 2>&1 &
python $S/expr/final_worker.py --seeds 4 5 $DA > $L/expr_final_b.log 2>&1 &
wait
python $S/expr/gates_consensus.py            > $L/expr_gates.log 2>&1
python $S/expr/gate_tests.py                >> $L/expr_gates.log 2>&1
python $S/expr/interpret_ig.py               > $L/expr_interpret_ig.log 2>&1
python $S/expr/interpret_perturb.py          > $L/expr_interpret_perturb.log 2>&1
python $S/expr/clinical_routing.py                   > $L/expr_clinical_routing.log 2>&1
# ---- participant readout: identifiable readout, stabilised training, meta-experts, fixed signature, external tests
for fl in "0 1" "2 3" "4"; do python $S/expr/warm_worker.py --mode cv --folds $fl > $L/expr_cvw_$(echo $fl | tr ' ' '_').log 2>&1 & done
python $S/expr/warm_worker.py --mode final  > $L/expr_finalw.log 2>&1 &
wait
python $S/expr/stability_compare.py          > $L/expr_stability.log 2>&1
python $S/expr/meta_experts.py               > $L/expr_meta.log 2>&1
python $S/expr/signature.py                  > $L/expr_signature.log 2>&1
python $S/expr/readout.py                    > $L/expr_readout.log 2>&1
python $S/expr/external_signature.py         > $L/expr_external.log 2>&1
python $S/expr/clinical.py                  > $L/expr_clinical.log 2>&1
# ---- transfer signature: internal signature diagnosis, cross-fitted constructions, transfer signature, readout meta, sc/sn check
python $S/expr/transfer_signatures.py              > $L/expr_transfer_signatures.log 2>&1
python $S/expr/transfer_external.py                > $L/expr_transfer_external.log 2>&1
python $S/expr/readout_adjustment.py                > $L/expr_readout_adjustment.log 2>&1
