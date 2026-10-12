"""Integration metric (iLISI overall and within PT state, healthy / injured) from saved checkpoints;
also provides the embedding functions used by harm_cells.py. The held-out cells are selected exactly as in run_bench.py,
so the overall iLISI must reproduce the stored value (checked and reported).
Usage: python lisi_posthoc.py --method scvi|cvae|scdisinfact [--tag s4] --device cuda:6
Rewrites results/bench_<name>/lisi.tsv (input table kept as lisi_overall_only.tsv).
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import load_bundle, subsample_cells, dense  # noqa: E402
from bench_metrics import ilisi  # noqa: E402
from run_bench import scoring_cells  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir  # noqa: E402


def embed_scvi(ck, cells, train, cfg, dev):
    import scvi
    c = cells.copy(); c.obs = c.obs[["participant", "tech", "condition", "dataset"]].astype(str)
    c.layers["counts"] = c.X.copy()
    m = scvi.model.SCVI.load(str(ck / "scvi_model"), adata=c, accelerator="gpu", device=[int(dev.split(":")[1])])
    return m.get_latent_representation(c, batch_size=4096)


def embed_cvae(ck, cells, train, cfg, dev):
    from cvae_net import CondVAE
    from m_cvae import apply
    p = cfg["cvae"]; st = torch.load(ck / "cvae.pt", map_location=dev)
    model = CondVAE(cells.n_vars, len(st["techs"]), len(st["conds"]), p["n_latent"], p["n_hidden"], p["n_layers"],
                    p["dropout"], p["adv_hidden"]).to(dev)
    model.load_state_dict(st["state_dict"])
    z, _ = apply(model, cells, st["techs"], st["conds"], "DKD", "HKD", torch.device(dev))
    return z


def embed_scdisinfact(ck, cells, train, cfg, dev):
    """Common encoder needs the training standardiser, recomputed from the identical training subsample."""
    from sklearn.preprocessing import StandardScaler
    from scDisInFact import base_model
    from m_scdisinfact import encode
    p = cfg["scdisinfact"]
    tr = train[train.obs.condition.isin(p["conditions"]).values]
    n = tr.obs[p["batch_key"]].value_counts()
    tr = tr[tr.obs[p["batch_key"]].isin(n[n >= p["min_cells_library"]].index).values]
    X = dense(tr.X); X = np.log1p(X / (X.sum(1, keepdims=True) / 100))
    scaler = StandardScaler().fit(X)
    sd = torch.load(ck / "scdisinfact.pt", map_location=dev)
    n_batch = sd["Enc_c.fc.fc_layers.Layer 0.0.weight"].shape[1] - cells.n_vars

    class Holder(torch.nn.Module):
        pass
    h = Holder()
    h.Enc_c = base_model.Encoder(n_input=cells.n_vars, n_output=p["Ks"][0], n_layers=2, n_hidden=128,
                                 n_cat_list=[n_batch], dropout_rate=0.2, use_batch_norm=False)
    h.Enc_ds = torch.nn.ModuleList([base_model.Encoder(n_input=cells.n_vars, n_output=p["Ks"][1], n_layers=1,
                                                       n_hidden=128, n_cat_list=None, dropout_rate=0.2,
                                                       use_batch_norm=False)])
    h.load_state_dict({k: v for k, v in sd.items() if k.startswith(("Enc_c.", "Enc_ds."))})
    h.to(dev)
    zc, _ = encode(h, scaler, cells.X, torch.device(dev))
    return zc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["scvi", "cvae", "scdisinfact"])
    ap.add_argument("--tag", default=None); ap.add_argument("--device", default="cuda:6")
    a = ap.parse_args()
    cfg = load_cfg("bench"); name = a.method + (f"_{a.tag}" if a.tag else "")
    out = result_dir(f"bench_{name}"); ckroot = Path(cfg["paths"]["ckpt"]) / name
    old = pd.read_csv(out / "lisi.tsv", sep="\t")
    kpmp, ext = load_bundle(cfg)
    cells = scoring_cells(kpmp, ext, cfg["train_cells"]["score_cap_per_participant"], cfg["seed"])
    fn = {"scvi": embed_scvi, "cvae": embed_cvae, "scdisinfact": embed_scdisinfact}[a.method]
    rows = []
    for (seed, f), r0 in old.groupby(["seed", "fold"]):
        ck = ckroot / f"seed{seed}_fold{f}"
        te_all = np.where(((cells.obs.dataset == "KPMP") & (cells.obs.fold == f)).values)[0]
        rng = np.random.default_rng(seed)
        te = rng.choice(te_all, min(len(te_all), cfg["metrics"]["lisi_cells"]), replace=False)
        sub = cells[te].copy()
        train = (subsample_cells(kpmp[(kpmp.obs.fold != f).values], cfg["train_cells"]["cap_per_participant"], seed)
                 if a.method == "scdisinfact" else None)
        emb = fn(ck, sub, train, cfg, a.device)
        base = r0.iloc[0][["method", "embedding", "seed", "fold"]].to_dict()
        tech, st = sub.obs.tech.values, sub.obs.pt_state.values
        v, vm = ilisi(emb, tech, cfg["metrics"]["lisi_k"], cfg["metrics"]["lisi_perplexity"])
        stored = float(r0[r0.metric == "iLISI_tech"].value.iloc[0])
        rows += [{**base, "metric": "iLISI_tech", "value": v, "n_cells": len(te), "stored_value": stored},
                 {**base, "metric": "iLISI_tech_mean", "value": vm, "n_cells": len(te)}]
        for s in ("healthy_PT", "injured_PT"):
            k = st == s
            vs, _ = ilisi(emb[k], tech[k], cfg["metrics"]["lisi_k"], cfg["metrics"]["lisi_perplexity"])
            rows.append({**base, "metric": f"iLISI_tech_{s}", "value": vs, "n_cells": int(k.sum())})
        np.savez_compressed(ck / "test_embedding.npz", emb=emb.astype(np.float32), tech=tech.astype(str),
                            pt_state=st.astype(str))
        print(seed, f, "overall recomputed", round(v, 4), "stored", round(stored, 4), flush=True)
    old.to_csv(out / "lisi_overall_only.tsv", sep="\t", index=False)
    pd.DataFrame(rows).to_csv(out / "lisi.tsv", sep="\t", index=False)


if __name__ == "__main__":
    main()
