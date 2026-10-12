#!/usr/bin/env bash
# After the Visium and pathomics queues finish: one retry pass per category (completed files are skipped),
# a full annotation pass to write its complete manifest, then the Visium and pathomics tables.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$HERE/../../logs
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
cd "$HERE"
until grep -q "done:" "$LOG/data_download_visium.log" && grep -q "done:" "$LOG/data_download_pathomics.log"; do sleep 120; done
for cat in visium pathomics; do
  cp "$LOG/data_download_$cat.log" "$LOG/data_download_${cat}_run1.log"
  python 02_download.py "$cat" --workers 4 > "$LOG/data_download_$cat.log" 2>&1
done
python 02_download.py annotation --workers 2 > "$LOG/data_download_annotation.log" 2>&1
python 04_visium_new.py > "$LOG/data_visium_04.log" 2>&1
python 05_pathomics.py > "$LOG/data_pathomics_05.log" 2>&1
echo "post-download steps finished $(date)"
