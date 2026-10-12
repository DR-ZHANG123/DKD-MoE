"""Method 5: the expr module's mixture-of-experts model and its ablations, trained in the bench
harness on exactly the same training-fold cells as the comparison methods.

The expr code (dkd_moe/scripts/expr: model.py, train.py, data.py) is imported read-only; architecture
and training settings come from dkd_moe/config/expr.yaml, K and the condition weight from the expr
cross-validation selection (results/expr_cv/selection.json). Variants change one switch each:
  full    selected K, top-k sparse routing, condition head, technology discriminator
  K1      a single expert (no routing)
  no_adv  technology discriminator weight 0
  no_cond condition head weight 0
  dense   dense routing (top_k = K)
Scores: (a) logistic regression on participant mean routing probabilities; (b) mean log-odds DKD vs HKD
of the condition head; (c) logistic regression on participant mean latent.
"""
from __future__ import annotations
import copy, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from bench_data import dense
from bench_score import latent_lr_score, mean_by_sample

EXPR = Path(__file__).resolve().parents[1] / "expr"
sys.path.insert(0, str(EXPR))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
from cdl import load_cfg, result_dir  # noqa: E402


def BenchCells(a, ecfg, device):
    """expr data.CellData built from an in-memory AnnData slice (subclass created at call time so every
    current CellData method, e.g. rows(), is inherited; only construction differs)."""
    from data import CellData, TECHS

    class _Cells(CellData):
        def __init__(self):
            self.obs = a.obs.copy(); self.genes = list(a.var_names); self.device = device
            self.X = torch.from_numpy(dense(a.X))            # host memory, as in CellData
            self.tech = torch.tensor(self.obs.tech.map({t: i for i, t in enumerate(TECHS)}).values,
                                     dtype=torch.long, device=device)
            classes = ecfg["condition"]["classes"]
            grp = self.obs.group if "group" in self.obs else self.obs.condition
            cmap = {g: classes.index(c) for g, c in ecfg["condition"]["map"].items()}
            cmap.update({c: classes.index(c) for c in classes})
            self.cond = torch.tensor(grp.astype(str).map(cmap).fillna(-1).astype(int).values, dtype=torch.long,
                                     device=device)
            self.participant = self.obs.participant.astype(str).values
    return _Cells()


def expr_setup(variant: str, bench_cfg: dict, dev: bool = False) -> tuple[dict, int, dict, int]:
    ecfg = copy.deepcopy(load_cfg("expr"))
    sel_f = result_dir("expr_cv") / "selection.json"
    sel = json.loads(sel_f.read_text()) if sel_f.exists() else {}
    K = ecfg["final"]["K"] or sel.get("K") or (bench_cfg["moe_ablations"]["dev_K"] if dev else None)
    if K is None:
        raise SystemExit(f"expr selection not available yet: {sel_f}")
    K = int(K); w = {"cond": float(sel.get("w_cond", ecfg["train"]["w_cond"]))}
    if variant == "K1":
        K = 1
    elif variant == "no_adv":
        w["adv"] = 0.0
    elif variant == "no_cond":
        w["cond"] = 0.0
    elif variant == "dense":
        ecfg["model"]["top_k"] = K
    elif variant != "full":
        raise SystemExit(f"unknown variant {variant}")
    epochs = bench_cfg["moe_ablations"]["epochs"] or ecfg["train"]["epochs"]
    return ecfg, K, w, epochs


@torch.no_grad()
def forward_chunks(model, cells, ecfg, device, chunk=100000):
    from data import forward_all
    P, Z, LO = [], [], []
    cls = ecfg["condition"]["classes"]; i_d, i_h = cls.index("DKD"), cls.index("HKD")
    for i in range(0, cells.n_obs, chunk):
        d = BenchCells(cells[i:i + chunk], ecfg, device)
        r = forward_all(model, d)
        lg = torch.log_softmax(model.cond_head(torch.tensor(r["p"], device=device)), -1).cpu().numpy()
        P.append(r["p"]); Z.append(r["z"]); LO.append(lg[:, i_d] - lg[:, i_h])
        del d; torch.cuda.empty_cache()
    return np.vstack(P), np.vstack(Z), np.concatenate(LO)


def run(ctx) -> dict:
    from train import train_model
    cfg = ctx["cfg"]; seed = ctx["seed"]; dev = ctx["device"]; variant = ctx["variant"] or "full"
    case = cfg["contrast"]["case"]
    ecfg, K, w, epochs = expr_setup(variant, cfg, ctx.get("dev", False))
    tr = BenchCells(ctx["train"], ecfg, dev)
    model, curve = train_model(ecfg, tr, np.arange(len(tr)), K, seed, w_over=w, epochs=epochs, log_every=5,
                               tag=f"bench_{variant}_s{seed}_f{ctx['fold']}")
    del tr; torch.cuda.empty_cache()
    torch.save({"state_dict": model.state_dict(), "K": K, "w": w, "variant": variant}, ctx["ckdir"] / "model.pt")
    cells, meta, ref = ctx["cells"], ctx["meta"], ctx["ref"]
    P, Z, LO = forward_chunks(model, cells, ecfg, dev)
    s_gate = latent_lr_score(mean_by_sample(P, cells.obs, meta), meta, ref, case, cfg["scvi"]["lr_C"], seed)
    s_head = mean_by_sample(LO[:, None], cells.obs, meta)[0].rename("score")
    s_lat = latent_lr_score(mean_by_sample(Z, cells.obs, meta), meta, ref, case, cfg["scvi"]["lr_C"], seed)
    curve["K"] = K; curve["variant"] = variant; curve["w"] = repr(w); curve["expr_cfg_sha"] = ecfg["_sha"]
    scores = {"MoE_gate_LR": s_gate, "MoE_condition_head": s_head, "MoE_latent_LR": s_lat}
    if K == 1:   # a single expert has constant routing: gate and condition-head scores are undefined
        scores = {"MoE_latent_LR": s_lat}
    return {"scores": scores,
            "emb": Z, "emb_name": "MoE_latent", "tables": {"history": curve}}
