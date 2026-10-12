"""03a_select: choose the regularisation setting of the alignment heads on inner-validation
participants only (KPMP training folds; held-out folds and GSE211785 are not looked at).

A first unregularised run memorised training pairs within two epochs (training InfoNCE -> 0.3 while
the inner-validation loss rose from the first epoch), so capacity / dropout / weight decay are
selected here. Criterion: mean over folds of the best inner-validation within-section InfoNCE
(relative to chance, lower is better). Writes results/align_select/{grid.tsv, selected.yaml}.
Usage: python 03a_select.py [--folds 0,1,2,3,4] [--seed 0]
"""
from __future__ import annotations
import argparse, copy, importlib, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from data import SpotData, select_hvgs  # noqa: E402

T3 = importlib.import_module("03_train")
RES = "align_select"
GRID = {
    "reg_mlp512": {"model": {"img_hidden": [512], "expr_hidden": [512], "dropout": 0.3, "img_in_dropout": 0.2,
                             "expr_in_dropout": 0.5}, "train": {"lr": 3.0e-4, "weight_decay": 0.01}},
    "reg_mlp256": {"model": {"img_hidden": [256], "expr_hidden": [256], "dropout": 0.5, "img_in_dropout": 0.3,
                             "expr_in_dropout": 0.7}, "train": {"lr": 3.0e-4, "weight_decay": 0.05}},
    "linear": {"model": {"img_hidden": [], "expr_hidden": [], "dropout": 0.0, "img_in_dropout": 0.2,
                         "expr_in_dropout": 0.5}, "train": {"lr": 1.0e-3, "weight_decay": 0.01}},
    "reg_deep": {"model": {"img_hidden": [512], "expr_hidden": [1024, 512], "dropout": 0.5, "img_in_dropout": 0.3,
                           "expr_in_dropout": 0.7}, "train": {"lr": 3.0e-4, "weight_decay": 0.05}},
}
# same settings with HVGs selected on KPMP training spots (comparison of gene sets; not selectable)
for _k in list(GRID):
    GRID[f"{_k}__visium_hvg"] = {**GRID[_k], "data": {"hvg_source": "visium"}}


def merged(cfg, over):
    c = copy.deepcopy(cfg)
    for k, v in over.items():
        c[k].update(v)
    return c


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--folds", default="0,1,2,3,4"); ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    cfg = C.load_cfg(); dev = torch.device(cfg["device"]); out = C.result_dir(RES)
    S = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    D = SpotData(S); m = D.meta; sp = C.splits()
    fold = m.participant.map(sp.fold).where(m.cohort == "kpmp")
    rows = []
    for f in [int(x) for x in a.folds.split(",")]:
        trp = sorted(m.participant[(fold != f) & fold.notna()].unique())
        rng = np.random.default_rng(cfg["seed"] + f)   # same inner split as 03_train
        vap = set(rng.choice(trp, max(2, int(round(len(trp) * cfg["train"]["val_frac_participants"]))), replace=False))
        tr = np.where(m.participant.isin(trp) & ~m.participant.isin(vap))[0]
        va = np.where(m.participant.isin(vap))[0]
        xcache = {}
        for name, over in GRID.items():
            c = merged(cfg, over)
            n = (c["data"]["n_hvg"], c["data"]["hvg_source"])
            if n not in xcache:
                hvg, _ = select_hvgs(D, np.isin(np.arange(len(m)), tr), *n); xcache[n] = D.lognorm(hvg)
            for mode in [cfg["train"]["primary_batch_mode"]]:   # batch modes are compared in 03_train
                _, curve = T3.train_one(D, xcache[n], tr, va, mode, a.seed, c, dev)
                best = min(curve, key=lambda r: r["val_rel_loss"])
                rows.append({"setting": name, "mode": mode, "fold": f, "best_val_rel_loss": best["val_rel_loss"],
                             "best_epoch": best["epoch"], "train_loss_at_best": best["train_loss"]})
                print(rows[-1], flush=True)
                pd.DataFrame(rows).to_csv(out / "grid.tsv", sep="\t", index=False)
        del xcache; torch.cuda.empty_cache()
    G = pd.DataFrame(rows)
    summ = G.groupby(["setting", "mode"]).best_val_rel_loss.mean().reset_index().sort_values("best_val_rel_loss")
    summ.to_csv(out / "grid_summary.tsv", sep="\t", index=False)
    ok = (summ["mode"] == cfg["train"]["primary_batch_mode"]) & (summ.setting.str.contains("__visium_hvg") == (cfg["data"]["hvg_source"] == "visium"))
    best = summ[ok].iloc[0].setting
    (out / "selected.yaml").write_text(yaml.safe_dump({"setting": best, **GRID[best]}))
    print(summ.to_string()); print("selected", best)
    C.provenance(RES, [C.rpath("align_prep") / "sections.tsv"], [out / "grid.tsv", out / "grid_summary.tsv",
                 out / "selected.yaml"], a.seed, extra={"grid": GRID})


if __name__ == "__main__":
    main()
