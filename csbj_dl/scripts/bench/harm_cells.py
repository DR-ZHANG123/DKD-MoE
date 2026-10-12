"""Harmonised evaluation, step 1: per-cell representations of injured-PT cells from every saved model.

Evaluation unit (same as the expr readout, csbj_dl/scripts/expr/readout.py): injured-PT cells only;
participant x technology samples with >= expr.yaml prep.min_cells_per_participant injured cells; all such
cells are used (no subsampling). For each saved fold model this writes <ckpt>/<run>/seed{s}_fold{f}/
injured_cells.npz with the per-cell arrays the scores need (latent, routing, condition-head log-odds,
counterfactual likelihood ratio), in the row order of <ckpt>/harmonised_cells.tsv.gz.
Usage: python harm_cells.py --run scvi_s1 [--device cuda:6]   (run = checkpoint folder name)
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import load_bundle, subsample_cells, dense  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg  # noqa: E402


def injured_cells(cfg) -> ad.AnnData:
    """All injured-PT cells of qualifying participant x technology samples, KPMP then external cohorts."""
    mc = load_cfg("expr")["prep"]["min_cells_per_participant"]
    kpmp, ext = load_bundle(cfg)
    parts = []
    for a in [kpmp] + list(ext.values()):
        a = a[(a.obs.pt_state == "injured_PT").values]
        key = a.obs.participant + "|" + a.obs.tech
        n = key.value_counts()
        a = a[key.isin(n[n >= mc].index).values].copy()
        if "fold" not in a.obs:
            a.obs["fold"] = -1
        a.obs["group"] = a.obs.group.astype(str)
        a.obs = a.obs[["participant", "tech", "group", "condition", "dataset", "library", "pt_state", "fold"]].copy()
        parts.append(a)
    c = ad.concat(parts, join="inner", index_unique=None)
    c.obs_names_make_unique()
    return c


def cell_table(cfg, cells) -> None:
    f = Path(cfg["paths"]["ckpt"]) / "harmonised_cells.tsv.gz"
    o = cells.obs[["dataset", "participant", "tech", "group", "fold"]]
    if f.exists():
        old = pd.read_csv(f, sep="\t", index_col=0, dtype={"participant": str})
        assert (old.index == o.index).all(), "injured-cell table changed; rebuild all harmonised arrays"
    else:
        o.to_csv(f, sep="\t")


def emb_scvi(ck, cells, cfg, dev, seed, fold, kpmp):
    import scvi
    c = cells.copy(); c.obs = c.obs[["participant", "tech", "condition", "dataset"]].astype(str)
    c.layers["counts"] = c.X.copy()
    m = scvi.model.SCVI.load(str(ck / "scvi_model"), adata=c, accelerator="gpu", device=[int(dev.split(":")[1])])
    out = {"emb": m.get_latent_representation(c, batch_size=4096)}
    # scVI DE on injured-PT training cells, DKD vs HKD (group as in the expr readout)
    p = cfg["scvi"]; rng = np.random.default_rng(seed)
    tr = c[((cells.obs.dataset == "KPMP") & (cells.obs.fold != fold)).values].copy()
    g = cells.obs.group.values[((cells.obs.dataset == "KPMP") & (cells.obs.fold != fold)).values]
    i1 = np.where(g == "DKD")[0]; i2 = np.where(g == "HKD")[0]
    n = p["de_cells_per_group"]
    i1 = rng.choice(i1, min(n, len(i1)), replace=False); i2 = rng.choice(i2, min(n, len(i2)), replace=False)
    ar = np.arange(tr.n_obs)
    de = m.differential_expression(tr, idx1=np.isin(ar, i1), idx2=np.isin(ar, i2), mode="change", delta=0.25,
                                   batch_correction=True, batch_size=4096)
    de = de[np.maximum(de.non_zeros_proportion1, de.non_zeros_proportion2) >= p["de_min_nonzero"]]
    de[["lfc_mean", "is_de_fdr_0.05"]].to_csv(ck / "harmonised_scvi_de.tsv.gz", sep="\t")
    return out


def emb_cvae(ck, cells, cfg, dev, seed, fold, kpmp):
    from cvae_net import CondVAE
    from m_cvae import apply
    p = cfg["cvae"]; st = torch.load(ck / "cvae.pt", map_location=dev)
    model = CondVAE(cells.n_vars, len(st["techs"]), len(st["conds"]), p["n_latent"], p["n_hidden"], p["n_layers"],
                    p["dropout"], p["adv_hidden"]).to(dev)
    model.load_state_dict(st["state_dict"])
    z, llr = apply(model, cells, st["techs"], st["conds"], "DKD", "HKD", torch.device(dev))
    return {"emb": z, "llr": llr}


def emb_scdisinfact(ck, cells, cfg, dev, seed, fold, kpmp):
    from lisi_posthoc import embed_scdisinfact
    from m_scdisinfact import encode
    train = subsample_cells(kpmp[(kpmp.obs.fold != fold).values], cfg["train_cells"]["cap_per_participant"], seed)
    zc = embed_scdisinfact(ck, cells, train, cfg, dev)            # common latent (training standardiser)
    return {"emb": zc, **_scd_zd(ck, cells, train, cfg, dev)}


def _scd_zd(ck, cells, train, cfg, dev):
    from sklearn.preprocessing import StandardScaler
    from scDisInFact import base_model
    p = cfg["scdisinfact"]
    tr = train[train.obs.condition.isin(p["conditions"]).values]
    n = tr.obs[p["batch_key"]].value_counts()
    tr = tr[tr.obs[p["batch_key"]].isin(n[n >= p["min_cells_library"]].index).values]
    X = dense(tr.X); scaler = StandardScaler().fit(np.log1p(X / (X.sum(1, keepdims=True) / 100)))
    sd = torch.load(ck / "scdisinfact.pt", map_location=dev)
    enc = base_model.Encoder(n_input=cells.n_vars, n_output=p["Ks"][1], n_layers=1, n_hidden=128, n_cat_list=None,
                             dropout_rate=0.2, use_batch_norm=False)
    enc.load_state_dict({k[len("Enc_ds.0."):]: v for k, v in sd.items() if k.startswith("Enc_ds.0.")})
    enc.to(dev).eval(); zd = []
    with torch.no_grad():
        for i in range(0, cells.n_obs, 20000):
            x = dense(cells.X[i:i + 20000]); x = np.log1p(x / np.maximum(x.sum(1, keepdims=True) / 100, 1e-8))
            mu, _ = enc(torch.FloatTensor(scaler.transform(x)).to(dev))
            zd.append(mu.cpu().numpy())
    return {"zd": np.vstack(zd)}


def emb_moe(ck, cells, cfg, dev, seed, fold, kpmp, state=None, K=None, top_k=None):
    """Routing, condition-head log-odds and latent from a MoE checkpoint (bench run or expr warm-start)."""
    import copy
    sys.path.insert(0, str(HERE.parents[0] / "expr"))
    from train import build_model
    from model import cond_logodds
    from m_moe import BenchCells
    from data import forward_all
    ecfg = copy.deepcopy(load_cfg("expr"))
    if state is None:
        st = torch.load(ck / "model.pt", map_location=dev); state, K = st["state_dict"], st["K"]
        top_k = K if st.get("variant") == "dense" else ecfg["model"]["top_k"]
    ecfg["model"]["top_k"] = top_k
    m = build_model(ecfg, cells.n_vars, K).to(dev); m.load_state_dict(state); m.eval()
    P, Z, LO = [], [], []
    for i in range(0, cells.n_obs, 100000):
        d = BenchCells(cells[i:i + 100000], ecfg, dev)
        r = forward_all(m, d)
        with torch.no_grad():
            LO.append(cond_logodds(m, torch.tensor(r["p"], device=dev)).cpu().numpy())
        P.append(r["p"]); Z.append(r["z"])
    return {"emb": np.vstack(Z), "p": np.vstack(P), "logodds": np.concatenate(LO)}


FN = {"scvi": emb_scvi, "cvae": emb_cvae, "scdisinfact": emb_scdisinfact, "moe": emb_moe}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--run", required=True); ap.add_argument("--device", default="cuda:6")
    a = ap.parse_args()
    cfg = load_cfg("bench"); root = Path(cfg["paths"]["ckpt"]) / a.run
    method = "moe" if a.run.startswith("moe") else a.run.rsplit("_s", 1)[0]
    cells = injured_cells(cfg); cell_table(cfg, cells)
    kpmp = load_bundle(cfg)[0] if method == "scdisinfact" else None
    for ck in sorted(root.glob("seed*_fold*")):
        if (ck / "injured_cells.npz").exists():
            continue
        seed, fold = map(int, ck.name.replace("seed", "").split("_fold"))
        out = FN[method](ck, cells, cfg, a.device, seed, fold, kpmp)
        np.savez_compressed(ck / "injured_cells.npz", **{k: np.asarray(v, dtype=np.float32) for k, v in out.items()})
        print(a.run, ck.name, {k: np.shape(v) for k, v in out.items()}, flush=True)


if __name__ == "__main__":
    main()
