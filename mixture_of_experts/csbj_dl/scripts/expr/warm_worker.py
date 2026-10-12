"""Stabilised training: K=1 warm start + annealed gate noise, at the selected K / w_cond.

--mode cv    : per held-out fold, one K=1 model on the training folds, then MoE seeds warm-started from it;
               writes <ckpt>/cvw/cvw_K{K}_wc{wc}_s{seed}_f{fold}.npz (held-out p, g, nll, cond_prob, logodds)
--mode final : one K=1 model on all KPMP, MoE seeds warm-started from it, forward inference on KPMP and the
               external cohorts; writes <ckpt>/finalw/finalw_K{K}_s{seed}_{cohort}.npz
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
from model import cond_logodds  # noqa: E402
from train import train_model  # noqa: E402


def save_pred(model, data, f, idx=None):
    res = forward_all(model, data, idx)
    with torch.no_grad():
        p = torch.tensor(res["p"], device=data.device)
        cp = torch.softmax(model.cond_head(p), -1).cpu().numpy(); lo = cond_logodds(model, p).cpu().numpy()
    extra = {} if idx is None else {"idx": idx}
    np.savez_compressed(f, p=res["p"].astype(np.float32), g=res["g"].astype(np.float32), nll=res["nll"].astype(np.float32),
                        cond_prob=cp.astype(np.float32), logodds=lo.astype(np.float32), z=res["z"].astype(np.float16), **extra)


def single_model(cfg, data, tr_idx, f_pt, epochs, tag):
    if f_pt.exists():
        return torch.load(f_pt, map_location=data.device)
    set_global_seed(0)
    m, curve = train_model(cfg, data, tr_idx, 1, 0, w_over={"balance": 0.0, "cond": 0.0}, epochs=epochs,
                           log_every=5, tag=tag)
    sd = {k: v.detach().clone() for k, v in m.state_dict().items()}
    torch.save(sd, f_pt)
    curve.to_csv(result_dir("expr_stability") / "curves" / f"{f_pt.stem}.tsv", sep="\t", index=False)
    return sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["cv", "final"], required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--threads", type=int, default=4)
    args = ap.parse_args(); torch.set_num_threads(args.threads)
    cfg = load_cfg("expr"); st = cfg["readout_stability"]["stability"]
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K, wc = int(sel["K"]), float(sel["w_cond"])
    bulk = Path(cfg["paths"]["bulk_dir"]); ck = Path(cfg["paths"]["ckpt_dir"])
    (result_dir("expr_stability") / "curves").mkdir(parents=True, exist_ok=True)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, cfg["train"]["device"])
    seeds = args.seeds or st["cv_seeds"]
    kw = dict(w_over={"cond": wc}, epochs=st["moe_epochs"], log_every=5, anneal_noise=st["anneal_gate_noise"],
              init_noise=st["expert_init_noise"])
    if args.mode == "cv":
        out = ck / st["prefix"]; out.mkdir(parents=True, exist_ok=True)
        fold = data.obs.fold.values.astype(int)
        for f in (args.folds if args.folds is not None else range(cfg["n_folds"])):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            sd = single_model(cfg, data, tr, out / f"single_f{f}.pt", st["warm_epochs"], f"single_f{f}")
            for s in seeds:
                name = f"{st['prefix']}_K{K}_wc{wc}_s{s}_f{f}"
                if (out / f"{name}.npz").exists():
                    continue
                set_global_seed(s); t0 = time.time()
                m, curve = train_model(cfg, data, tr, K, s, init_from=sd, tag=name, **kw)
                save_pred(m, data, out / f"{name}.npz", te)
                torch.save(m.state_dict(), out / f"{name}.pt")
                curve.to_csv(result_dir("expr_stability") / "curves" / f"{name}.tsv", sep="\t", index=False)
                print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        out = ck / "finalw"; out.mkdir(parents=True, exist_ok=True)
        allidx = np.arange(len(data))
        sd = single_model(cfg, data, allidx, out / "single_all.pt", st["warm_epochs"], "single_all")
        for s in seeds:
            name = f"finalw_K{K}_s{s}"
            if (out / f"{name}_gse195460.npz").exists():
                continue
            set_global_seed(s); t0 = time.time()
            m, curve = train_model(cfg, data, allidx, K, s, init_from=sd, tag=name, **{**kw, "epochs": cfg["train"]["epochs"]})
            torch.save({"state_dict": m.state_dict(), "K": K, "w_cond": wc, "genes": data.genes}, out / f"{name}.pt")
            curve.to_csv(result_dir("expr_stability") / "curves" / f"{name}.tsv", sep="\t", index=False)
            save_pred(m, data, out / f"{name}_kpmp.npz")
            for c in ("gse211785", "gse195460"):
                ext = CellData(bulk / f"{c}_pt_hvg.h5ad", cfg, cfg["train"]["device"], genes=data.genes)
                save_pred(m, ext, out / f"{name}_{c}.npz")
            print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
