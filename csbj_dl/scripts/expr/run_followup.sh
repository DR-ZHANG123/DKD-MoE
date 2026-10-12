#!/usr/bin/env bash
# Analyses that build on the final models, the comparison methods and the supervised model without experts
# (run after run_expr.sh and the comparison step). Idempotent where the underlying scripts skip existing models.
# Usage: bash csbj_dl/scripts/expr/run_followup.sh [device]   (after run_nested_cv.sh; run from any directory)
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../../.." && pwd); cd "$ROOT"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
L=csbj_dl/logs; S=csbj_dl/scripts; DEV=${1:-cuda:5}
mkdir -p "$L"
export CSBJ_DATA_ON_DEVICE=1
# nested cross-validation of the model selection (see run_nested_cv.sh for the device list)
python $S/expr/nested_selection.py                         > $L/expr_nested_selection.log 2>&1
# participant-level adjustment, external summaries by control type, sparse versus dense readout
python $S/expr/patient_adjusted.py                         > $L/expr_patient_adjusted.log 2>&1
python $S/expr/external_by_control.py                      > $L/expr_external_by_control.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/routing_faithfulness.py    > $L/expr_routing_faithfulness.log 2>&1
# supervised model without experts (same procedure, one expert, classifier on the latent mean)
python $S/bench/sup_single.py --source z --seeds 1 2 3 4 5 --device $DEV > $L/bench_sup_single.log 2>&1
python $S/bench/sup_score.py                               > $L/bench_sup_score.log 2>&1
python $S/bench/sup_mixing.py --device $DEV                > $L/bench_sup_mixing.log 2>&1
# injured-cell and participant-level supervision (partition models, then all-KPMP models)
for v in injured patient; do
  python $S/expr/sup_variants.py --variant $v --mode cv --device $DEV    > $L/supv_${v}_cv.log 2>&1
  python $S/expr/sup_variants.py --variant $v --mode final --device $DEV > $L/supv_${v}_final.log 2>&1
done
python $S/expr/sup_compare.py                              > $L/sup_compare.log 2>&1
# expert contributions and exchanges of expert weight in held-out participants
CSBJ_DEVICE=$DEV python $S/expr/heldout_experts.py         > $L/heldout_experts.log 2>&1
python $S/expr/heldout_exchange_ranks.py                   > $L/heldout_exchange_ranks.log 2>&1
# three summaries of the nested cross-validation; expert identities from training participants; gene-level exchange
# effects; components of the model without experts chosen like the experts
CSBJ_DEVICE=$DEV python $S/expr/nested_calibration.py      > $L/nested_calibration.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/trainonly_experts.py       > $L/trainonly_experts.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/heldout_exchange_genes.py  > $L/heldout_exchange_genes.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/nonmoe_components.py       > $L/nonmoe_components.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/m3_signature.py            > $L/m3_signature.log 2>&1
# diabetes groups, clinical models, robustness of the readout, signatures restricted by cellular source
python $S/expr/diabetes_groups.py                          > $L/diabetes_groups.log 2>&1
python $S/expr/clinical_models.py                          > $L/clinical_models.log 2>&1
python $S/expr/readout_robustness.py                       > $L/readout_robustness.log 2>&1
CSBJ_DEVICE=$DEV python $S/expr/pt_signature.py            > $L/pt_signature.log 2>&1
