"""Print the inner-loop job list of the nested cross-validation, one worker per line (round-robin split).
Usage: python nested_jobs.py N_WORKERS  -> N lines of space-separated outer:K:wcond:seed:inner jobs"""
from __future__ import annotations
import itertools, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg  # noqa: E402

cfg = load_cfg("expr"); cv = cfg["cv"]; nf = cfg["n_folds"]; nw = int(sys.argv[1])
jobs = [f"{o}:{K}:{float(w)}:{s}:{i}" for K, w in itertools.product(cv["K_grid"], cv["w_cond_grid"])
        for s in cv["seeds_for_stability"] for o in range(nf) for i in range(nf) if i != o]
for w in range(nw):
    print(" ".join(jobs[w::nw]))
