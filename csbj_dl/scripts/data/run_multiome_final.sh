#!/usr/bin/env bash
# After the whole Multiome queue and the core-first build: one retry pass, then add the remaining
# (AKI/other) libraries on the unchanged core-group consensus and remerge.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$HERE/../../logs
mkdir -p "$LOG"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
cd "$HERE"
until grep -q "done:" "$LOG/data_download_multiome.log" && grep -q "multiome build finished" "$LOG/data_run_multiome.log"; do sleep 300; done
cp "$LOG/data_download_multiome.log" "$LOG/data_download_multiome_run2.log"
python 02_download.py multiome --workers 6 > "$LOG/data_download_multiome.log" 2>&1
set -e
python 03a_multiome_peaks.py --allow-partial --keep-consensus > "$LOG/data_multiome_03a_final.log" 2>&1
python 03b_multiome_count.py > "$LOG/data_multiome_03b_final.log" 2>&1
python 03c_multiome_merge.py > "$LOG/data_multiome_03c_final.log" 2>&1
python 10_data_sources.py > "$LOG/data_sources_10.log" 2>&1
echo "final multiome build finished $(date)"
