#!/usr/bin/env bash
# chrom module, all stages in order (one command). Usage: bash run_chrom.sh [--from NN]
# Every stage rewrites its outputs; 04_align skips fold x seed models whose outputs exist unless FORCE=1.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); LOG=$HERE/../../logs
source "$(conda info --base)/etc/profile.d/conda.sh" && conda activate project016
cd "$HERE"
FROM=${2:-00}; [ "${1:-}" = "--from" ] || FROM=00
run() { local n=$1; shift; if [[ "$n" < "$FROM" ]]; then return; fi
        echo "[$(date)] stage $n: $*"; "$@" > "$LOG/chrom_$n.log" 2>&1; echo "[$(date)] stage $n done"; }
run 00 bash 00_reference.sh
run 01 python 01_labels.py
run 02 python 02_ext_labels.py
run 03 python 03_rna_forward.py
run 04 python 04_align.py ${FORCE:+--force}
run 05 python 05_chrom_readout.py
run 06 python 06_attrib.py
run 07 python 07_motifs.py
run 08 python 08_figdata.py
run 10 python 10_qc_adjust.py
# expert components in chromatin (needs results/expr_heldout_experts/matching.tsv from run_followup.sh)
run 11 python 11_expert_chromatin.py
run 12 python 12_motif_activity.py
run 13 python 13_expert_chrom_stats.py
echo "chrom pipeline finished $(date)"
