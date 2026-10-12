#!/usr/bin/env bash
# Rebuild every csbj_dl figure from result tables (plotting only; no model is rerun).
# Usage: bash csbj_dl/scripts/figures/run_all.sh            (all figures)
#        bash csbj_dl/scripts/figures/run_all.sh Fig3 Fig7  (selected figures)
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate project016
cd "$(dirname "$0")"
ALL=(Fig1_framework Fig2_model Fig3_readout Fig4_interpretation Fig5_clinical Fig6_chromatin
     FigS_comparison_external FigS_chromatin_alignment FigS_histology FigS_histology_direction GraphicalAbstract)
SEL=("$@")
for f in "${ALL[@]}"; do
  if [ ${#SEL[@]} -gt 0 ]; then
    keep=0; for s in "${SEL[@]}"; do [[ "$f" == "$s"* ]] && keep=1; done
    [ $keep -eq 1 ] || continue
  fi
  echo "== $f"
  python "$f.py"
done
