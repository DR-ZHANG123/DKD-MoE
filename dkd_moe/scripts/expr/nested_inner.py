"""Nested cross-validation, inner loop: one grid setting x seed x inner held-out fold inside one outer fold.

Job = outer:K:wcond:seed:inner. Training cells = participants in neither the outer nor the inner held-out fold;
held-out = inner fold. Training identical to cv_worker.py (train_model defaults, w_cond override, cv: epochs).
Writes <ckpt_dir>/<nested_cv.inner_prefix>/ncvi_o{o}_K{K}_wc{w}_s{s}_i{i}.{npz,pt} and
dkd_moe/results/expr_nested_cv/curves/<job>.tsv
Usage: python nested_inner.py --jobs 0:6:10.0:1:2 [...] [--device cuda:1] [--threads 8] [--shared]
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


def inner_name(o, K, wc, seed, i):
    return f"ncvi_o{o}_K{K}_wc{float(wc)}_s{seed}_i{i}"


def claim(outdir, name) -> bool:
    """Atomic per-job claim so that several workers can share one job list."""
    (outdir / "claims").mkdir(exist_ok=True)
    try:
        (outdir / "claims" / name).mkdir()
        return True
    except FileExistsError:
        return False


def run_job(cfg, data, o, K, wc, seed, i, outdir, shared=False):
    name = inner_name(o, K, wc, seed, i)
    f_npz = outdir / f"{name}.npz"
    if f_npz.exists() or (shared and not claim(outdir, name)):
        print("skip", name, flush=True); return
    assert o != i
    fold = data.obs.fold.values.astype(int)
    tr_idx = np.where((fold != o) & (fold != i))[0]
    te_idx = np.where(fold == i)[0]
    set_global_seed(seed); t0 = time.time()
    model, curve = train_model(cfg, data, tr_idx, K, seed, w_over={"cond": wc}, epochs=cfg["cv"]["epochs"],
                               log_every=5, tag=name)
    res = forward_all(model, data, te_idx)
    with torch.no_grad():
        cp = torch.softmax(model.cond_head(torch.tensor(res["p"], device=data.device)), -1).cpu().numpy()
    np.savez_compressed(f_npz, idx=te_idx, p=res["p"].astype(np.float32), g=res["g"].astype(np.float32),
                        nll=res["nll"].astype(np.float32), cond_prob=cp.astype(np.float32))
    curve["job"] = name; curve["train_minutes"] = (time.time() - t0) / 60; curve["device"] = str(data.device)
    cdir = result_dir("expr_nested_cv") / "curves"; cdir.mkdir(exist_ok=True)
    curve.to_csv(cdir / f"{name}.tsv", sep="\t", index=False)
    torch.save(model.state_dict(), outdir / f"{name}.pt")
    print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", nargs="+", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--shared", action="store_true", help="claim jobs from a list shared with other workers")
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_cfg("expr")
    if args.device:
        cfg["train"]["device"] = args.device
    outdir = Path(cfg["paths"]["ckpt_dir"]) / cfg["nested_cv"]["inner_prefix"]; outdir.mkdir(parents=True, exist_ok=True)
    data = CellData(Path(cfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", cfg, cfg["train"]["device"])
    for j in args.jobs:
        o, K, wc, seed, i = j.split(":")
        run_job(cfg, data, int(o), int(K), float(wc), int(seed), int(i), outdir, args.shared)


if __name__ == "__main__":
    main()
