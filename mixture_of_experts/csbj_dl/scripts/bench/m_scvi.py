"""Method 2: scVI (scvi-tools) trained on KPMP training-fold PT cells, batch = technology.

Scores: (a) logistic regression on participant mean latent (training-fold DKD/HKD participants);
(b) signed mean z of the top up/down genes from scVI change-mode DE (DKD vs HKD training cells).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from bench_score import geneset_score, latent_lr_score, mean_by_sample


def _prep(a, cfg):
    a = a.copy()
    a.obs = a.obs[["participant", "tech", "condition", "dataset"]].astype(str).copy()
    a.layers["counts"] = a.X.copy()
    return a


def run(ctx) -> dict:
    import scvi, torch
    cfg = ctx["cfg"]; p = cfg["scvi"]; seed = ctx["seed"]
    case, ctrl = cfg["contrast"]["case"], cfg["contrast"]["control"]
    scvi.settings.seed = seed
    torch.set_float32_matmul_precision("high")
    tr = _prep(ctx["train"], cfg)
    scvi.model.SCVI.setup_anndata(tr, layer="counts", batch_key=p["batch_key"])
    m = scvi.model.SCVI(tr, n_latent=p["n_latent"], n_layers=p["n_layers"], n_hidden=p["n_hidden"],
                        gene_likelihood=p["gene_likelihood"])
    dev = ctx["device"]
    acc, idx = ("gpu", [int(dev.split(":")[1])]) if dev.startswith("cuda") else ("cpu", "auto")
    m.train(max_epochs=p["max_epochs"], batch_size=p["batch_size"], accelerator=acc, devices=idx,
            plan_kwargs={"lr": p["lr"]}, early_stopping=False, enable_progress_bar=False, train_size=1.0)
    m.save(str(ctx["ckdir"] / "scvi_model"), overwrite=True)
    hist = {k: v.iloc[:, 0].tolist() for k, v in m.history.items() if k in ("elbo_train", "reconstruction_loss_train")}
    cells = _prep(ctx["cells"], cfg)
    emb = m.get_latent_representation(cells, batch_size=4096)
    means = mean_by_sample(emb, cells.obs, ctx["meta"])
    s_lr = latent_lr_score(means, ctx["meta"], ctx["ref"], case, p["lr_C"], seed)
    # scVI DE on training cells of training-fold DKD vs HKD participants
    rng = np.random.default_rng(seed)
    o = tr.obs
    i1 = np.where(o.condition.values == case)[0]; i2 = np.where(o.condition.values == ctrl)[0]
    n = p["de_cells_per_group"]
    i1 = rng.choice(i1, min(n, len(i1)), replace=False); i2 = rng.choice(i2, min(n, len(i2)), replace=False)
    de = m.differential_expression(tr, idx1=np.isin(np.arange(tr.n_obs), i1), idx2=np.isin(np.arange(tr.n_obs), i2),
                                   mode="change", delta=0.25, batch_correction=True, batch_size=4096)
    de = de.sort_values("lfc_mean", ascending=False)
    # usual scVI practice: ignore genes detected in very few cells of both groups
    de = de[np.maximum(de.non_zeros_proportion1, de.non_zeros_proportion2) >= p["de_min_nonzero"]]
    sig = de[de["is_de_fdr_0.05"]] if "is_de_fdr_0.05" in de else de
    if (sig.lfc_mean > 0).sum() < p["de_n_top"] or (sig.lfc_mean < 0).sum() < p["de_n_top"]:
        sig = de
    up = sig[sig.lfc_mean > 0].index[:p["de_n_top"]].tolist()
    down = sig[sig.lfc_mean < 0].sort_values("lfc_mean").index[:p["de_n_top"]].tolist()
    s_de = geneset_score(ctx["pb_logcpm"], ctx["meta"], up, down, ctx["ref"])
    de = de.assign(selected=np.where(de.index.isin(up), "up", np.where(de.index.isin(down), "down", "")))
    return {"scores": {"scVI_latent_LR": s_lr, "scVI_DE_set": s_de}, "emb": emb, "emb_name": "scVI_latent",
            "tables": {"scvi_de": de, "history": pd.DataFrame(hist)}}
