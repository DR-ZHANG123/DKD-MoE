"""Supervision variants of the final procedure (config: expr.yaml supervision_variants).

injured : cell-level condition loss on injured-PT cells only (healthy-PT cells keep every other loss term)
patient : participant-level condition loss on bags of injured-PT cells (mean routing -> condition head)
Everything else (architecture, selected K / condition weight, shared start from the single-expert models,
annealed gate noise, schedule, partitions, seeds) is that of warm_worker.py.

--mode cv    : held-out fold predictions  <ckpt>/<cv_prefix>/<cv_prefix>_K{K}_wc{wc}_s{s}_f{f}.{npz,pt}
--mode final : all-KPMP models + forward inference on KPMP and external cohorts
               <ckpt>/<final_prefix>/<final_prefix>_K{K}_s{s}{.pt,_kpmp.npz,_gse211785.npz,_gse195460.npz}
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
from data import CellData, capped_epoch_indices  # noqa: E402
from model import losses, log_norm  # noqa: E402
from train import build_model, warm_init, train_model  # noqa: E402
from warm_worker import save_pred  # noqa: E402


def bag_pool(data: CellData, tr_idx: np.ndarray, min_cells: int) -> dict:
    """Injured-PT training cells of DKD / HKD participants, grouped by participant and diagnosis."""
    st = data.obs.pt_state.astype(str).values; cond = data.cond.cpu().numpy()
    m = np.zeros(len(data), bool); m[tr_idx] = True
    m &= (st == "injured_PT") & (cond >= 0)
    df = pd.DataFrame({"row": np.where(m)[0], "p": data.participant[m], "c": cond[m]})
    pools = {c: [] for c in sorted(df.c.unique())}
    for (p, c), d in df.groupby(["p", "c"]):
        if len(d) >= min_cells:
            pools[c].append(d.row.values)
    return pools


def patient_loss(model, data, pools, n_per_class, cells, rng):
    """Cross-entropy of the condition head on bag-mean routing probabilities (balanced diagnoses)."""
    rows, bag, ys = [], [], []
    for c, plist in pools.items():
        for j in rng.choice(len(plist), n_per_class, replace=len(plist) < n_per_class):
            r = plist[j]; r = rng.choice(r, min(cells, len(r)), replace=False)
            rows.append(r); bag.append(np.full(len(r), len(ys))); ys.append(c)
    rows, bag = np.concatenate(rows), torch.as_tensor(np.concatenate(bag), device=data.device)
    p, _ = model.gate_probs(model.encode_h(log_norm(data.rows(rows))), noisy=True)
    nb = len(ys)
    pm = torch.zeros(nb, p.shape[1], device=p.device).index_add_(0, bag, p)
    pm = pm / torch.bincount(bag, minlength=nb).clamp_min(1).unsqueeze(1).float()
    return F.cross_entropy(model.cond_head(pm), torch.as_tensor(ys, device=data.device))


def train_patient(cfg, data, tr_idx, K, seed, wc, epochs, init_from, init_noise, anneal, tag):
    """warm_worker training loop with the cell-level condition term replaced by the participant-level term."""
    tr, pv = cfg["train"], cfg["supervision_variants"]["patient"]
    torch.manual_seed(seed); np.random.seed(seed); rng = np.random.default_rng(seed)
    w = {"kl": tr["w_kl"], "balance": tr["w_balance"], "cond": 0.0, "adv": tr["w_adv"]}
    model = build_model(cfg, data.X.shape[1], K).to(data.device)
    warm_init(model, init_from, init_noise, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=tr["lr"], weight_decay=tr["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=tr["lr"] * 0.05)
    pools = bag_pool(data, tr_idx, pv["min_injured_cells"]); npc = pv["participants_per_step"] // len(pools)
    rows, bs = [], tr["batch_size"]
    for ep in range(epochs):
        model.train(); t0 = time.time()
        if anneal:
            model.noise_mult = max(0.0, 1.0 - ep / max(1, epochs - 1))
        ww = dict(w, kl=w["kl"] * min(1.0, (ep + 1) / max(1, tr["kl_warmup_epochs"])))
        idx = capped_epoch_indices(data.participant, tr_idx, tr["cells_per_participant_cap"], rng)
        acc, nb = {}, 0
        for s in range(0, len(idx) - bs + 1, bs):
            bn = idx[s:s + bs]; b = torch.as_tensor(bn, device=data.device); xb = data.rows(bn)
            out = model(xb, data.tech[b], sample=True, grl_lambda=tr["grl_lambda"])
            loss, parts = losses(out, xb, data.tech[b], data.cond[b], ww)
            lp = patient_loss(model, data, pools, npc, pv["cells_per_bag"], rng)
            loss = loss + wc * lp; parts["ce_patient"] = lp.item()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); opt.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0.0) + v
            nb += 1
        sched.step()
        row = {"epoch": ep + 1, **{k: v / max(nb, 1) for k, v in acc.items()}, "sec": time.time() - t0, "n_cells": len(idx)}
        rows.append(row)
        if (ep + 1) % 5 == 0:
            print(tag, " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()), flush=True)
    return model, pd.DataFrame(rows)


def fit(variant, cfg, data, tr, K, wc, seed, epochs, sd, st, tag):
    if variant == "patient":
        return train_patient(cfg, data, tr, K, seed, wc, epochs, sd, st["expert_init_noise"], st["anneal_gate_noise"], tag)
    cond0 = data.cond
    inj = torch.as_tensor((data.obs.pt_state.astype(str) == "injured_PT").values, device=data.device)
    data.cond = torch.where(inj, cond0, torch.full_like(cond0, -1))
    try:
        return train_model(cfg, data, tr, K, seed, w_over={"cond": wc}, epochs=epochs, log_every=5, tag=tag,
                           init_from=sd, anneal_noise=st["anneal_gate_noise"], init_noise=st["expert_init_noise"])
    finally:
        data.cond = cond0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["injured", "patient"], required=True)
    ap.add_argument("--mode", choices=["cv", "final"], required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=None)
    ap.add_argument("--device", default=None)
    a = ap.parse_args()
    cfg = load_cfg("expr"); st = cfg["readout_stability"]["stability"]; sv = cfg["supervision_variants"]
    if a.device:
        cfg["train"]["device"] = a.device
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text()); K, wc = int(sel["K"]), float(sel["w_cond"])
    bulk = Path(cfg["paths"]["bulk_dir"]); ck = Path(cfg["paths"]["ckpt_dir"]); dev = cfg["train"]["device"]
    curves = result_dir("expr_supervision") / "curves"; curves.mkdir(parents=True, exist_ok=True)
    data = CellData(bulk / "kpmp_pt_hvg.h5ad", cfg, dev)
    seeds = a.seeds or st["cv_seeds"]
    if a.mode == "cv":
        pre = sv["cv_prefix"][a.variant]; out = ck / pre; out.mkdir(parents=True, exist_ok=True)
        fold = data.obs.fold.values.astype(int)
        for f in (a.folds if a.folds is not None else range(cfg["n_folds"])):
            tr, te = np.where(fold != f)[0], np.where(fold == f)[0]
            sd = torch.load(ck / st["prefix"] / f"single_f{f}.pt", map_location=dev)
            for s in seeds:
                name = f"{pre}_K{K}_wc{wc}_s{s}_f{f}"
                if (out / f"{name}.npz").exists():
                    continue
                set_global_seed(s); t0 = time.time()
                m, curve = fit(a.variant, cfg, data, tr, K, wc, s, st["moe_epochs"], sd, st, name)
                save_pred(m, data, out / f"{name}.npz", te); torch.save(m.state_dict(), out / f"{name}.pt")
                curve.to_csv(curves / f"{name}.tsv", sep="\t", index=False)
                print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)
    else:
        pre = sv["final_prefix"][a.variant]; out = ck / pre; out.mkdir(parents=True, exist_ok=True)
        sd = torch.load(ck / "finalw" / "single_all.pt", map_location=dev)
        for s in (a.seeds or cfg["final"]["seeds"]):
            name = f"{pre}_K{K}_s{s}"
            if (out / f"{name}_gse195460.npz").exists():
                continue
            set_global_seed(s); t0 = time.time()
            m, curve = fit(a.variant, cfg, data, np.arange(len(data)), K, wc, s, cfg["train"]["epochs"], sd, st, name)
            torch.save({"state_dict": m.state_dict(), "K": K, "w_cond": wc, "genes": data.genes}, out / f"{name}.pt")
            curve.to_csv(curves / f"{name}.tsv", sep="\t", index=False)
            save_pred(m, data, out / f"{name}_kpmp.npz")
            for c in ("gse211785", "gse195460"):
                save_pred(m, CellData(bulk / f"{c}_pt_hvg.h5ad", cfg, dev, genes=data.genes), out / f"{name}_{c}.npz")
            print("done", name, f"{(time.time() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
