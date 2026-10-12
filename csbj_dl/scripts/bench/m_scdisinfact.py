"""Method 3: scDisInFact on KPMP training-fold PT cells, condition = group (DKD/HKD/Reference),
batch = library (author defaults from config/run.yaml unless stated in bench.yaml).

scDisInFact iterates over all batch x condition datasets in lock-step, so one epoch is as long as the
smallest dataset allows; the number of epochs is therefore set from a target number of passes over
the training cells. The condition encoder has no batch input (enc_injection = False), so cells from
unseen libraries (held-out participants, external cohorts) are encoded without a batch label.
Scores: (a) logistic regression on participant mean condition latent; (b) signed mean z of the top
condition-associated genes, sign from the training-fold pseudobulk log fold change.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from bench_data import dense
from bench_score import geneset_score, latent_lr_score, mean_by_sample


def _norm(counts: np.ndarray, scaler) -> torch.Tensor:
    sf = counts.sum(1, keepdims=True) / 100
    x = np.log1p(counts / np.maximum(sf, 1e-8))
    return torch.FloatTensor(scaler.transform(x))


def encode(model, scaler, X, dev, chunk=20000):
    zc, zd = [], []
    model.eval()
    with torch.no_grad():
        for i in range(0, X.shape[0], chunk):
            x = _norm(dense(X[i:i + chunk]), scaler).to(dev)
            b = torch.zeros((x.shape[0], 1), dtype=torch.long, device=dev)  # used by the common encoder only
            mu_c, _ = model.Enc_c(x, b)
            mu_d, _ = model.Enc_ds[0](x, b)   # no batch injection: b is ignored
            zc.append(mu_c.cpu().numpy()); zd.append(mu_d.cpu().numpy())
    return np.vstack(zc), np.vstack(zd)


def run(ctx) -> dict:
    from scDisInFact import scdisinfact, create_scdisinfact_dataset
    cfg = ctx["cfg"]; p = cfg["scdisinfact"]; seed = ctx["seed"]
    case, ctrl = cfg["contrast"]["case"], cfg["contrast"]["control"]
    tr = ctx["train"]
    tr = tr[tr.obs.condition.isin(p["conditions"]).values]
    n = tr.obs[p["batch_key"]].value_counts()
    tr = tr[tr.obs[p["batch_key"]].isin(n[n >= p["min_cells_library"]].index).values].copy()
    counts = dense(tr.X)
    meta = tr.obs[["condition", p["batch_key"]]].astype(str).copy()
    torch.manual_seed(seed); np.random.seed(seed)
    dd = create_scdisinfact_dataset(counts, meta, condition_key=["condition"], batch_key=p["batch_key"])
    dev = torch.device(ctx["device"])
    if dev.type == "cpu":
        torch.set_num_threads(p["cpu_threads"])
    model = scdisinfact(data_dict=dd, Ks=p["Ks"], batch_size=p["batch_size"], interval=p["interval"], lr=p["lr"],
                        reg_mmd_comm=p["reg_mmd_comm"], reg_mmd_diff=p["reg_mmd_diff"], reg_gl=p["reg_gl"],
                        reg_class=p["reg_class"], reg_kl_comm=p["reg_kl_comm"], reg_kl_diff=p["reg_kl_diff"],
                        seed=seed, device=dev)
    sizes = [len(d) for d in dd["datasets"]]
    steps_per_epoch = int(np.ceil(min(sizes) / p["batch_size"]))
    cells_per_step = p["batch_size"] * len(sizes)
    nepochs = int(np.ceil(p["passes"] * sum(sizes) / (cells_per_step * steps_per_epoch)))
    model.train()
    losses = model.train_model(nepochs=nepochs, recon_loss="NB")
    model.eval()
    torch.save(model.state_dict(), ctx["ckdir"] / "scdisinfact.pt")
    zc, zd = encode(model, dd["scaler"], ctx["cells"].X, dev)
    means = mean_by_sample(zd, ctx["cells"].obs, ctx["meta"])
    s_lr = latent_lr_score(means, ctx["meta"], ctx["ref"], case, p["lr_C"], seed)
    ckg = pd.Series(model.extract_gene_scores()[0], index=tr.var_names).sort_values(ascending=False)
    top = ckg.index[:p["ckg_n_top"]]
    L, ref, meta_s = ctx["pb_logcpm"], ctx["ref"], ctx["meta"]
    lfc = (L.iloc[ref][meta_s.condition.values[ref] == case].mean(0)
           - L.iloc[ref][meta_s.condition.values[ref] == ctrl].mean(0))
    up = [g for g in top if lfc[g] > 0]; down = [g for g in top if lfc[g] <= 0]
    s_ckg = geneset_score(L, meta_s, up, down, ref)
    tab = pd.DataFrame({"ckg": ckg, "train_lfc": lfc.reindex(ckg.index)})
    info = pd.DataFrame([{"n_datasets": len(sizes), "min_dataset": min(sizes), "steps_per_epoch": steps_per_epoch,
                          "nepochs": nepochs, "n_cells": int(sum(sizes)),
                          "final_loss": float(np.ravel(losses[0])[-1]) if len(losses) else np.nan}])
    return {"scores": {"scDisInFact_zd_LR": s_lr, "scDisInFact_CKG_set": s_ckg}, "emb": zc,
            "emb_name": "scDisInFact_common_latent", "tables": {"ckg": tab, "train_info": info}}
