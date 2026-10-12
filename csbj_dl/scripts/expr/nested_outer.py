"""Nested cross-validation, outer loop: the final (shared-start) procedure at the setting selected inside each
outer fold, trained on the 4 outer-training folds and applied to the outer held-out fold.

Procedure identical to warm_worker.py --mode cv: the single-expert start model of the outer training set
(K=1, seed 0, no routing / condition terms, readout_stability.stability warm_epochs; it does not depend on K or w_cond, so
the existing <ckpt>/cvw/single_f{o}.pt trained on exactly these participants is used), then readout_stability.stability
cv_seeds mixture models initialised from it with annealed gate noise. Outer folds whose selected setting equals
the reported one (K=6, w_cond=10) reuse the reported cvw models (checked in nested_summary.py) and are skipped.
Writes <ckpt>/<nested_cv.outer_prefix>/ncvo_K{K}_wc{w}_s{s}_f{o}.{npz,pt}; curves in results/expr_nested_cv/curves/
Usage: python nested_outer.py [--folds 0 1] [--device cuda:1] [--threads 8] [--shared]
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
from data import CellData  # noqa: E402
from train import train_model  # noqa: E402
from warm_worker import save_pred  # noqa: E402
from nested_inner import claim  # noqa: E402


def outer_name(K, wc, seed, o):
    return f"ncvo_K{K}_wc{float(wc)}_s{seed}_f{o}"


def reported_setting() -> tuple[int, float]:
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    return int(sel["K"]), float(sel["w_cond"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shared", action="store_true")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_cfg("expr"); st = cfg["readout_stability"]["stability"]
    if args.device:
        cfg["train"]["device"] = args.device
    ck = Path(cfg["paths"]["ckpt_dir"]); out = ck / cfg["nested_cv"]["outer_prefix"]; out.mkdir(parents=True, exist_ok=True)
    cdir = result_dir("expr_nested_cv") / "curves"; cdir.mkdir(exist_ok=True)
    C = pd.read_csv(result_dir("expr_nested_cv") / "selection.tsv", sep="\t")
    rep = reported_setting()
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, cfg["train"]["device"])
    fold = data.obs.fold.values.astype(int)
    kw = dict(epochs=st["moe_epochs"], log_every=5, anneal_noise=st["anneal_gate_noise"], init_noise=st["expert_init_noise"])
    for _, r in C.iterrows():
        o, K, wc = int(r.outer_fold), int(r.K), float(r.w_cond)
        if (args.folds is not None and o not in args.folds) or (K, wc) == rep:
            continue
        tr, te = np.where(fold != o)[0], np.where(fold == o)[0]
        sd = torch.load(ck / st["prefix"] / f"single_f{o}.pt", map_location=data.device)
        for s in st["cv_seeds"]:
            name = outer_name(K, wc, s, o)
            if (out / f"{name}.npz").exists() or (args.shared and not claim(out, name)):
                continue
            set_global_seed(s); t0 = time.time()
            m, curve = train_model(cfg, data, tr, K, s, w_over={"cond": wc}, init_from=sd, tag=name, **kw)
            save_pred(m, data, out / f"{name}.npz", te)
            torch.save(m.state_dict(), out / f"{name}.pt")
            curve["job"] = name; curve["train_minutes"] = (time.time() - t0) / 60; curve["device"] = str(data.device)
            curve.to_csv(cdir / f"{name}.tsv", sep="\t", index=False)
            print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
