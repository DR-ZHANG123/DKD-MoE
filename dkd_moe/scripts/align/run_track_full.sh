#!/usr/bin/env bash
# Full alignment pipeline for one track after 02_prepare: 5 parts trained in parallel on one GPU (resumable,
# OOM-tolerant), then calibrated external evaluation, summaries, interpretation, signature, heatmaps.
# Usage: ALIGN_DEVICE=cuda:0 bash run_track_full.sh            (primary, full-resolution track)
#        ALIGN_TRACK=lowres ALIGN_DEVICE=cuda:5 bash run_track_full.sh
cd "$(dirname "$0")/../../.."
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate dkd-moe
A=dkd_moe/scripts/align; mkdir -p dkd_moe/logs
if [ "${ALIGN_TRACK:-}" = "lowres" ]; then
  CK=/path/to/dkd_moe_data/dkd_moe_ckpt/align_lowres; L=dkd_moe/logs/align_lowres
else
  CK=/path/to/dkd_moe_data/dkd_moe_ckpt/align; L=dkd_moe/logs/align
fi
for i in $(seq 1 20); do
  n=$(ls $CK/*_metrics.json 2>/dev/null | wc -l)
  [ "$n" -ge 75 ] && break
  for f in 0 1 2 3 4; do
    python $A/03_train.py --folds "$f" --tag "_f$f" >> ${L}_train_f$f.log 2>&1 &
    sleep 20
  done
  wait
done
echo "trained $(ls $CK/*_metrics.json | wc -l)"
python $A/03_train.py > ${L}_train_all.log 2>&1          # all runs resumed: consistent spot tables
python $A/03c_external_calibrated.py > ${L}_extcal.log 2>&1
python $A/03b_summarise.py > ${L}_summarise.log 2>&1
python $A/04_interpret.py > ${L}_interpret.log 2>&1
if [ "${ALIGN_TRACK:-}" = "lowres" ]; then
  ALIGN_EXTCAL=none python $A/04_interpret.py > ${L}_interpret_uncal.log 2>&1
fi
python $A/07_signature.py > ${L}_sig.log 2>&1
if [ "${ALIGN_TRACK:-}" != "lowres" ]; then
  python $A/05_attribution.py > ${L}_attrib.log 2>&1
  python $A/05b_sig_attribution.py > ${L}_sig_attrib.log 2>&1
fi
python $A/06_heatmaps.py > ${L}_heatmaps.log 2>&1
echo "done"
