"""Readout helpers: which final models are used (per the KPMP-internal initialization rule), model loading,
per-seed cohort predictions with the DKD-vs-HKD log-odds readout."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import result_dir  # noqa: E402
from train import build_model  # noqa: E402


def final_set(cfg) -> dict:
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K, wc = int(sel["K"]), float(sel["w_cond"])
    ad_f = result_dir("expr_stability") / "initialization.json"
    adopt = ad_f.exists() and json.loads(ad_f.read_text())["adopt_stabilised"]
    d = Path(cfg["paths"]["ckpt_dir"]) / ("finalw" if adopt else "final")
    stem = f"finalw_K{K}_s{{s}}" if adopt else f"final_K{K}_s{{s}}"
    cvp = cfg["readout_stability"]["stability"]["prefix"] if adopt else "cv"
    cvs = cfg["readout_stability"]["stability"]["cv_seeds"] if adopt else cfg["cv"]["seeds_for_stability"]
    return {"K": K, "w_cond": wc, "dir": d, "stem": stem, "stabilised": bool(adopt), "seeds": cfg["final"]["seeds"],
            "cv_prefix": cvp, "cv_seeds": cvs}


def load_model(cfg, fs: dict, seed: int, device: str):
    ck = torch.load(fs["dir"] / (fs["stem"].format(s=seed) + ".pt"), map_location=device)
    m = build_model(cfg, len(ck["genes"]), fs["K"]).to(device)
    m.load_state_dict(ck["state_dict"]); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, ck["genes"]


def head_diff(model) -> tuple[np.ndarray, float]:
    W, b = model.cond_head.weight.detach().cpu().numpy(), model.cond_head.bias.detach().cpu().numpy()
    return W[1] - W[0], float(b[1] - b[0])


def cohort_pred(cfg, fs: dict, seed: int, cohort: str) -> dict:
    """p and log-odds for one seed and cohort ('kpmp', 'gse211785', 'gse195460')."""
    z = np.load(fs["dir"] / (fs["stem"].format(s=seed) + f"_{cohort}.npz"))
    out = {"p": z["p"]}
    if "logodds" in z:
        out["logodds"] = z["logodds"]
    else:
        sd = torch.load(fs["dir"] / (fs["stem"].format(s=seed) + ".pt"), map_location="cpu")["state_dict"]
        W, b = sd["cond_head.weight"].numpy(), sd["cond_head.bias"].numpy()
        out["logodds"] = z["p"] @ (W[1] - W[0]) + (b[1] - b[0])
    return out
