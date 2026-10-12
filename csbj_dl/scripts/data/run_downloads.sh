#!/usr/bin/env bash
# Launch all csbj_dl KPMP downloads in the background (resume-safe; rerun to complete partial runs).
# Per-stream throughput from the KPMP server is ~1 MB/s, so categories run side by side.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/../../logs
mkdir -p "$LOG"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
cd "$HERE"
for spec in "multiome 6" "annotation 2" "pathomics 4" "visium 4"; do
  set -- $spec
  nohup python 02_download.py "$1" --workers "$2" > "$LOG/data_download_$1.log" 2>&1 &
done
