"""Ablations of the final (warm-start) mixture-of-experts model, trained exactly like the expr
module's stabilised cross-validation models (scripts/expr/warm_worker.py --mode cv): all training-fold
KPMP PT cells, initialised from that fold's single-expert model (<expr ckpt>/cvw/single_f{f}.pt),
readout_stability.stability settings (moe_epochs, expert_init_noise, annealed gate noise), selected K and w_cond.
Each variant changes one switch: K1 (one expert), no_adv (technology discriminator weight 0), no_cond
(condition-head weight 0), dense (top_k = K). Variant 'full' does not train: it loads the expr
module's own cvw checkpoints (the final model).
Writes <bench ckpt>/warm_<variant>/seed{s}_fold{f}/{model.pt, injured_cells.npz} and curves.
Usage: python warm_ablation.py --variant K1 --seeds 1 2 3 4 5 [--folds 0 1 2 3 4] --device cuda:6
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "expr"))
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
from harm_cells import injured_cells, cell_table, emb_moe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, choices=["full", "K1", "no_adv", "no_cond", "dense"])
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:6")
    a = ap.parse_args()
    from data import CellData
    from train import train_model
    cfg = load_cfg("bench"); ecfg = copy.deepcopy(load_cfg("expr")); st = ecfg["readout_stability"]["stability"]
    ecfg["train"]["device"] = a.device
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K, wc = int(sel["K"]), float(sel["w_cond"])
    cvw = Path(ecfg["paths"]["ckpt_dir"]) / st["prefix"]
    out = Path(cfg["paths"]["ckpt"]) / f"warm_{a.variant}"; out.mkdir(parents=True, exist_ok=True)
    cells = injured_cells(cfg); cell_table(cfg, cells)
    data = None
    w = {"cond": wc}; Kv, top_k = K, ecfg["model"]["top_k"]
    if a.variant == "K1":
        Kv = 1
    elif a.variant == "no_adv":
        w["adv"] = 0.0
    elif a.variant == "no_cond":
        w["cond"] = 0.0
    elif a.variant == "dense":
        top_k = K
    ecfg["model"]["top_k"] = top_k
    for s in a.seeds:
        for f in a.folds:
            d = out / f"seed{s}_fold{f}"; d.mkdir(exist_ok=True)
            if (d / "injured_cells.npz").exists():
                continue
            t0 = time.time()
            if a.variant == "full":
                state = torch.load(cvw / f"{st['prefix']}_K{K}_wc{wc}_s{s}_f{f}.pt", map_location=a.device)
            else:
                if data is None:
                    data = CellData(Path(ecfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", ecfg, a.device)
                tr = np.where(data.obs.fold.values.astype(int) != f)[0]
                sd = torch.load(cvw / f"single_f{f}.pt", map_location=a.device)
                set_global_seed(s)
                m, curve = train_model(ecfg, data, tr, Kv, s, w_over=w, epochs=st["moe_epochs"], log_every=5,
                                       tag=f"warm_{a.variant}_s{s}_f{f}", init_from=sd,
                                       anneal_noise=st["anneal_gate_noise"], init_noise=st["expert_init_noise"])
                state = m.state_dict()
                curve.assign(variant=a.variant, K=Kv, w=repr(w), expr_cfg_sha=ecfg["_sha"]).to_csv(
                    d / "curve.tsv", sep="\t", index=False)
            torch.save({"state_dict": state, "K": Kv, "top_k": top_k, "w": w, "variant": a.variant}, d / "model.pt")
            res = emb_moe(d, cells, cfg, a.device, s, f, None, state=state, K=Kv, top_k=top_k)
            np.savez_compressed(d / "injured_cells.npz", **{k: v.astype(np.float32) for k, v in res.items()})
            print(f"warm_{a.variant} seed {s} fold {f}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
