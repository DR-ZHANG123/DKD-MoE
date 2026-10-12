#!/usr/bin/env bash
# Multiome + external snATAC build. Waits until every core-group Multiome archive is on disk (03a exits
# non-zero otherwise), then consensus peaks -> per-library counts -> merged h5ads -> external snATAC matrices.
# Rerun after the remaining (AKI/other) archives arrive: the consensus is unchanged, counted libraries are cached.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
LOG=$HERE/../../logs
mkdir -p "$LOG"
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
cd "$HERE"
until python 03a_multiome_peaks.py >> "$LOG/data_multiome_03a.log" 2>&1; do sleep 300; done
set -e
python 03b_multiome_count.py > "$LOG/data_multiome_03b.log" 2>&1
python 03c_multiome_merge.py > "$LOG/data_multiome_03c.log" 2>&1
python 07_ext_atac.py > "$LOG/data_ext_atac.log" 2>&1
echo "multiome build finished $(date)"
