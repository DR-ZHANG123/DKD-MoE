"""Final models: one seed trained on all KPMP PT cells at the selected K / condition weight,
then forward inference on all KPMP cells and on the external cohorts (never used for training).

Usage: python final_worker.py --seeds 1 2 ...
Writes: <ckpt_dir>/final/final_K{K}_s{seed}.pt and .npz per cohort; results/expr_final/curves/*.tsv
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
from data import CellData, forward_all  # noqa: E402
from train import train_model  # noqa: E402

COHORTS = ["kpmp", "gse211785", "gse195460"]


def selected(cfg):
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K = cfg["final"]["K"] or sel["K"]
    return int(K), float(sel["w_cond"])


def infer(model, data, f):
    res = forward_all(model, data)
    with torch.no_grad():
        cp = torch.softmax(model.cond_head(torch.tensor(res["p"], device=data.device)), -1).cpu().numpy()
    np.savez_compressed(f, p=res["p"].astype(np.float32), g=res["g"].astype(np.float32),
                        nll=res["nll"].astype(np.float32), cond_prob=cp.astype(np.float32),
                        z=res["z"].astype(np.float16))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--device", default=None, help="override train.device (e.g. cpu when the GPU is full)")
    ap.add_argument("--threads", type=int, default=0)
    args = ap.parse_args()
    if args.threads:
        torch.set_num_threads(args.threads)
    cfg = load_cfg("expr")
    if args.device:
        cfg["train"]["device"] = args.device
    K, wc = selected(cfg)
    bulk = Path(cfg["paths"]["bulk_dir"]); dev = cfg["train"]["device"]
    outdir = Path(cfg["paths"]["ckpt_dir"]) / "final"; outdir.mkdir(parents=True, exist_ok=True)
    cdir = result_dir("expr_final") / "curves"; cdir.mkdir(parents=True, exist_ok=True)
    kp = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    for s in args.seeds:
        name = f"final_K{K}_s{s}"
        if (outdir / f"{name}_gse195460.npz").exists():
            print("skip", name); continue
        set_global_seed(s); t0 = time.time()
        model, curve = train_model(cfg, kp, np.arange(len(kp)), K, s, w_over={"cond": wc}, log_every=5, tag=name)
        curve["train_minutes"] = (time.time() - t0) / 60
        curve.to_csv(cdir / f"{name}.tsv", sep="\t", index=False)
        torch.save({"state_dict": model.state_dict(), "K": K, "w_cond": wc, "genes": kp.genes}, outdir / f"{name}.pt")
        infer(model, kp, outdir / f"{name}_kpmp.npz")
        for c in COHORTS[1:]:
            ext = CellData(bulk / f"{c}_pt_hvg.h5ad", cfg, dev, genes=kp.genes)
            infer(model, ext, outdir / f"{name}_{c}.npz")
            del ext; torch.cuda.empty_cache()
        print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
