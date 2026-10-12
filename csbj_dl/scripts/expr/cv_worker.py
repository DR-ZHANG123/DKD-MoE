"""Participant-level cross-validation worker: trains on 4 folds, predicts the held-out fold.

Usage: python cv_worker.py --jobs K:wcond:seed:fold [K:wcond:seed:fold ...] [--epochs N] [--outdir DIR]
Writes per job: <ckpt_dir>/cv/<job>.npz (held-out p, g, nll, condition probabilities, cell index),
                csbj_dl/results/expr_cv/curves/<job>.tsv
"""
from __future__ import annotations
import argparse, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from train import train_model  # noqa: E402


def job_name(K, wc, seed, fold, prefix="cv"):
    return f"{prefix}_K{K}_wc{wc}_s{seed}_f{fold}"


def run_job(cfg, data, K, wc, seed, fold, epochs, outdir, prefix, w_extra=None):
    name = job_name(K, wc, seed, fold, prefix)
    f_npz = outdir / f"{name}.npz"
    if f_npz.exists():
        print("skip", name, flush=True); return
    fold_arr = data.obs.fold.values.astype(int)
    tr_idx = np.where(fold_arr != fold)[0]   # all cells; head loss masks unlabelled groups
    te_idx = np.where(fold_arr == fold)[0]
    set_global_seed(seed)
    t0 = time.time()
    w = {"cond": wc, **(w_extra or {})}
    model, curve = train_model(cfg, data, tr_idx, K, seed, w_over=w, epochs=epochs, log_every=5, tag=name)
    res = forward_all(model, data, te_idx)
    with torch.no_grad():
        cp = torch.softmax(model.cond_head(torch.tensor(res["p"], device=data.device)), -1).cpu().numpy()
    np.savez_compressed(f_npz, idx=te_idx, p=res["p"].astype(np.float32), g=res["g"].astype(np.float32),
                        nll=res["nll"].astype(np.float32), cond_prob=cp.astype(np.float32),
                        z=res["z"].astype(np.float16))
    curve["job"] = name; curve["train_minutes"] = (time.time() - t0) / 60
    cdir = result_dir("expr_cv") / "curves"; cdir.mkdir(exist_ok=True)
    curve.to_csv(cdir / f"{name}.tsv", sep="\t", index=False)
    torch.save(model.state_dict(), outdir / f"{name}.pt")
    print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--prefix", default="cv")
    ap.add_argument("--wbal", type=float, default=None)
    ap.add_argument("--wadv", type=float, default=None)
    ap.add_argument("--device", default=None, help="override train.device (e.g. cpu when the GPU is full)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_cfg("expr")
    if args.device:
        cfg["train"]["device"] = args.device
    outdir = Path(cfg["paths"]["ckpt_dir"]) / args.prefix; outdir.mkdir(parents=True, exist_ok=True)
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, cfg["train"]["device"])
    epochs = args.epochs or cfg["cv"]["epochs"]
    extra = {k: v for k, v in (("balance", args.wbal), ("adv", args.wadv)) if v is not None}
    for j in args.jobs:
        K, wc, seed, fold = j.split(":")
        run_job(cfg, data, int(K), float(wc), int(seed), int(fold), epochs, outdir, args.prefix, extra)


if __name__ == "__main__":
    main()
