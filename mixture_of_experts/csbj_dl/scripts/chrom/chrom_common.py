"""Shared helpers for the chrom module: config, paths, device choice, expression-model access (read-only
imports from csbj_dl/scripts/expr), label mapping and participant-level statistics."""
from __future__ import annotations
import subprocess, sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
sys.path.insert(0, str(HERE.parents[0] / "expr"))
from cdl import ROOT, CDL, load_cfg, result_dir, write_provenance, set_global_seed, md5  # noqa: E402,F401
from stats_utils import hedges_g, auc  # noqa: E402,F401


def cfg() -> dict:
    return load_cfg("chrom")


def P(c: dict, key: str) -> Path:
    p = Path(c["paths"][key])
    return p if p.is_absolute() else ROOT / p


def expr_cfg() -> dict:
    return load_cfg("expr")


def pick_device(pref: str = "auto") -> str:
    """Visible GPU with the most free memory (or the configured device / CHROM_DEVICE override)."""
    import os
    import torch
    pref = os.environ.get("CHROM_DEVICE", pref)
    if pref != "auto":
        return pref
    if not torch.cuda.is_available():
        return "cpu"
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, check=True).stdout.strip().splitlines()
        free = {int(a): int(b) for a, b in (l.split(",") for l in q)}
        vis = os.environ.get("CUDA_VISIBLE_DEVICES")
        ids = [int(x) for x in vis.split(",")] if vis else sorted(free)
        best = max(range(len(ids)), key=lambda i: free.get(ids[i], 0))
        return f"cuda:{best}"
    except Exception:
        return "cuda:0"


def pt_state(labels: pd.Series, c: dict) -> pd.Series:
    lab = labels.astype(str)
    out = pd.Series(np.nan, index=labels.index, dtype=object)
    out[lab.isin(set(c["labels"]["healthy_pt"]))] = "healthy_PT"
    out[lab.isin(set(c["labels"]["injured_pt"]))] = "injured_PT"
    return out


def splits(c: dict) -> pd.DataFrame:
    return pd.read_csv(P(c, "splits"), sep="\t", dtype={"participant": str}).set_index("participant")


def expr_model(fold: int | None, seed: int, device: str):
    """Fixed MoE model: cross-fitted fold model (fold held out) or, with fold=None, the final model."""
    import torch
    from train import build_model
    e = expr_cfg(); c = cfg(); R = c["rna"]
    ck = Path(c["paths"]["expr_ckpt"])
    if fold is None:
        d = torch.load(ck / "finalw" / f"finalw_K{R['expr_K']}_s{seed}.pt", map_location=device)
        sd, genes = d["state_dict"], d["genes"]
    else:
        sd = torch.load(ck / R["expr_prefix_cv"] / f"{R['expr_prefix_cv']}_K{R['expr_K']}_wc{R['expr_wc']}_s{seed}_f{fold}.pt",
                        map_location=device)
        genes = pd.read_csv(P(c, "hvg"), sep="\t").gene.tolist()
    m = build_model(e, len(genes), R["expr_K"]).to(device)
    m.load_state_dict(sd); m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m, genes


def participant_table(obs: pd.DataFrame, vals: dict, unit: str = "participant", min_cells: int = 20,
                      state: str = "injured_PT") -> pd.DataFrame:
    """Mean of per-nucleus values over a participant's nuclei of one PT state (>= min_cells)."""
    df = obs[[unit, "group"]].astype(str).reset_index(drop=True)
    for k, v in vals.items():
        df[k] = np.asarray(v)
    df = df[(obs.pt_state.astype(str) == state).values]
    g = df.groupby(unit)
    t = g[list(vals)].mean(); t["n_cells"] = g.size(); t["group"] = g.group.first()
    return t[t.n_cells >= min_cells].reset_index()


def tests(t: pd.DataFrame, comps, cols, extra: dict | None = None) -> pd.DataFrame:
    rows = []
    for a, b in comps:
        for col in cols:
            r = hedges_g(t.loc[t.group == a, col], t.loc[t.group == b, col])
            x, y = t.loc[t.group == a, col].dropna(), t.loc[t.group == b, col].dropna()
            rows.append({**(extra or {}), "comparison": f"{a}_vs_{b}", "feature": col, **r,
                         "auc": auc(x, y) if len(x) and len(y) else np.nan})
    return pd.DataFrame(rows)


def multiome_complete() -> dict:
    """Whether the merged Multiome objects contain every core-group library listed for download."""
    txt = "".join(f.read_text() for f in sorted((CDL / "logs").glob("data_run_multiome*.log")))
    return {"core_build_finished": "multiome build finished" in txt,
            "final_build_finished": ("final multiome build finished" in txt) or ("multiome build finished (final" in txt)}
