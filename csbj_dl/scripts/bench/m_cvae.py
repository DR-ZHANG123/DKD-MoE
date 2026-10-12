"""Method 4: CPA-like conditional VAE (cvae_net.CondVAE) trained on KPMP training-fold PT cells.

Condition (DKD/HKD/Reference) enters only through a decoder embedding; the basal latent is made
condition-free adversarially. Scores: (a) counterfactual log-likelihood ratio of each cell's counts
under DKD vs HKD (per count), averaged per participant x technology; (b) signed mean z of the top
genes of the decoded DKD-vs-HKD shift on training cells; (c) logistic regression on mean basal latent
(expected weak by construction; reported for completeness).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
import torch

from bench_data import dense
from bench_score import geneset_score, latent_lr_score, mean_by_sample
from cvae_net import CondVAE


def _tensors(a, techs, conds, dev):
    t = torch.as_tensor(pd.Categorical(a.obs.tech, categories=techs).codes.astype(np.int64))
    c = pd.Categorical(a.obs.condition, categories=conds).codes.astype(np.int64)
    return t, torch.as_tensor(np.where(c < 0, 0, c))


def train(model, a, techs, conds, p, dev, seed):
    X = a.X.tocsr(); t, c = _tensors(a, techs, conds, dev)
    opt = torch.optim.AdamW(model.parameters(), lr=p["lr"], weight_decay=p["weight_decay"])
    g = torch.Generator().manual_seed(seed)
    hist = []
    for ep in range(p["epochs"]):
        model.train()
        beta = min(1.0, (ep + 1) / max(1, p["kl_warmup_epochs"]))
        perm = torch.randperm(a.n_obs, generator=g).numpy()
        tot = np.zeros(3); nb = 0
        for i in range(0, len(perm), p["batch_size"]):
            idx = np.sort(perm[i:i + p["batch_size"]])
            x = torch.as_tensor(dense(X[idx])).to(dev)
            rec, kl, adv = model(x, t[idx].to(dev), c[idx].to(dev), p["adv_weight"])
            loss = rec + beta * kl + adv
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            opt.step()
            tot += [rec.item(), kl.item(), adv.item()]; nb += 1
        hist.append({"epoch": ep, "rec": tot[0] / nb, "kl": tot[1] / nb, "adv_ce": tot[2] / nb})
    return pd.DataFrame(hist)


@torch.no_grad()
def apply(model, a, techs, conds, case, ctrl, dev, chunk=8192):
    model.eval()
    X = a.X.tocsr(); t, _ = _tensors(a, techs, conds, dev)
    ic, ih = conds.index(case), conds.index(ctrl)
    zs, ll = [], []
    for i in range(0, a.n_obs, chunk):
        x = torch.as_tensor(dense(X[i:i + chunk])).to(dev); tt = t[i:i + chunk].to(dev)
        mu, _ = model.encode(x, tt)
        zs.append(mu.cpu().numpy()); ll.append(model.counterfactual_llr(x, tt, ic, ih).cpu().numpy())
    return np.vstack(zs), np.concatenate(ll)


def run(ctx) -> dict:
    cfg = ctx["cfg"]; p = cfg["cvae"]; seed = ctx["seed"]
    case, ctrl = cfg["contrast"]["case"], cfg["contrast"]["control"]
    conds = cfg["scdisinfact"]["conditions"]; techs = ["sc", "sn"]
    tr = ctx["train"][ctx["train"].obs.condition.isin(conds).values].copy()
    dev = torch.device(ctx["device"])
    torch.manual_seed(seed); np.random.seed(seed)
    model = CondVAE(tr.n_vars, len(techs), len(conds), p["n_latent"], p["n_hidden"], p["n_layers"], p["dropout"],
                    p["adv_hidden"]).to(dev)
    hist = train(model, tr, techs, conds, p, dev, seed)
    torch.save({"state_dict": model.state_dict(), "techs": techs, "conds": conds}, ctx["ckdir"] / "cvae.pt")
    cells = ctx["cells"]
    z, llr = apply(model, cells, techs, conds, case, ctrl, dev)
    meta, ref = ctx["meta"], ctx["ref"]
    s_llr = mean_by_sample(llr[:, None], cells.obs, meta)[0].rename("score")
    s_lr = latent_lr_score(mean_by_sample(z, cells.obs, meta), meta, ref, case, p["lr_C"], seed)
    # decoded shift on training cells (chunked mean)
    X = tr.X.tocsr(); t, _ = _tensors(tr, techs, conds, dev); acc = 0; n = 0
    model.eval()
    for i in range(0, tr.n_obs, 8192):
        x = torch.as_tensor(dense(X[i:i + 8192])).to(dev)
        acc = acc + model.decoded_shift(x, t[i:i + 8192].to(dev), conds.index(case), conds.index(ctrl)).cpu().numpy() * len(x)
        n += len(x)
    shift = pd.Series(acc / n, index=tr.var_names).sort_values(ascending=False)
    k = cfg["pseudobulk"]["n_top"]
    s_set = geneset_score(ctx["pb_logcpm"], meta, shift.index[:k].tolist(), shift.index[::-1][:k].tolist(), ref)
    return {"scores": {"CVAE_counterfactual_LLR": s_llr, "CVAE_decoded_set": s_set, "CVAE_basal_LR": s_lr},
            "emb": z, "emb_name": "CVAE_basal_latent",
            "tables": {"history": hist, "decoded_shift": shift.rename("log1p_cp10k_diff").to_frame()}}
