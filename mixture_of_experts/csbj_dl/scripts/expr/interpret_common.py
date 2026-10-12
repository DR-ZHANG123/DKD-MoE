"""Shared pieces for interpretation: model loading with matched expert order, integrated gradients,
gene-set overlap tests."""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, path, result_dir  # noqa: E402
from train import build_model  # noqa: E402


def selected_K(cfg) -> int:
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    return int(cfg["final"]["K"] or sel["K"])


def load_final(cfg, seed: int, device: str):
    """Final model of one seed and its permutation onto the reference expert order
    (perm[k] = this seed's expert matched to reference expert k)."""
    K = selected_K(cfg)
    ck = torch.load(Path(cfg["paths"]["ckpt_dir"]) / "final" / f"final_K{K}_s{seed}.pt", map_location=device)
    model = build_model(cfg, len(ck["genes"]), K).to(device)
    model.load_state_dict(ck["state_dict"]); model.eval()
    m = pd.read_csv(result_dir("expr_gates") / "seed_matching.tsv", sep="\t")
    m = m[m.seed == seed].sort_values("ref_expert")
    return model, m.matched_expert.values.astype(int), ck["genes"]


def integrated_gradients(fn, x: torch.Tensor, x0: torch.Tensor, steps: int = 50, chunk: int = 256) -> torch.Tensor:
    """IG attributions (x - x0) * mean_alpha d fn(x0 + alpha (x - x0)) / dx (Riemann midpoint rule).
    fn maps a batch of inputs to one scalar per row."""
    alphas = (torch.arange(steps, device=x.device, dtype=x.dtype) + 0.5) / steps
    out = torch.zeros_like(x)
    for s in range(0, x.shape[0], chunk):
        xb = x[s:s + chunk]; d = xb - x0
        g = torch.zeros_like(xb)
        for a in alphas:
            xi = (x0 + a * d).detach().requires_grad_(True)
            y = fn(xi).sum()
            g += torch.autograd.grad(y, xi)[0]
        out[s:s + chunk] = d * g / steps
    return out


def marker_sets(cfg) -> dict[str, list[str]]:
    gs = pd.read_csv(path(cfg, "marker_sets"), sep="\t")
    return {"markers_up": gs[gs.set_id == "DKDiPT_up"].gene.tolist(),
            "markers_down": gs[gs.set_id == "DKDiPT_down"].gene.tolist()}


def hallmark_sets(cfg) -> dict[str, list[str]]:
    out = {}
    for line in open(path(cfg, "hallmark_gmt")):
        f = line.rstrip("\n").split("\t")
        if f[0] in cfg["interpret"]["pathways"]:
            out[f[0]] = f[2:]
    return out


def overlap_tests(score: pd.Series, sets: dict[str, list[str]], top_n: int, label: str) -> pd.DataFrame:
    """For a gene score (universe = scored genes): top-n positive and top-n negative overlap with each set
    (hypergeometric P), and a rank test (Mann-Whitney of set-gene scores vs the rest)."""
    s = score.dropna()
    uni = set(s.index); top = set(s.nlargest(top_n).index); bot = set(s.nsmallest(top_n).index)
    rows = []
    for name, genes in sets.items():
        g = set(genes) & uni
        r = {"score": label, "set": name, "set_size": len(genes), "in_universe": len(g)}
        for tag, T in (("top", top), ("bottom", bot)):
            k = len(g & T)
            r[f"{tag}{top_n}_overlap"] = k
            r[f"{tag}{top_n}_genes"] = ",".join(sorted(g & T))
            r[f"{tag}{top_n}_p"] = stats.hypergeom.sf(k - 1, len(uni), len(g), top_n) if g else np.nan
        if len(g) >= 2:
            inn, rest = s[list(g)], s.drop(list(g))
            r["median_rank_pct"] = float(s.rank(pct=True)[list(g)].median())
            r["p_rank_greater"] = stats.mannwhitneyu(inn, rest, alternative="greater").pvalue
            r["p_rank_less"] = stats.mannwhitneyu(inn, rest, alternative="less").pvalue
        rows.append(r)
    return pd.DataFrame(rows)
