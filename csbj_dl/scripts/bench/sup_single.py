"""Supervised single-expert baseline: the final model without a mixture of experts.

Same encoder, latent heads, technology-conditioned NB decoder, KL / balance / technology-discriminator losses,
shared start (initialised from the fold's single-expert model <expr ckpt>/cvw/single_f{f}.pt), schedule
(stability settings of expr.yaml: moe_epochs, annealed gate noise, which is inert with one expert), training cells, partitions,
seeds, class-balanced condition supervision (DKD vs HKD only) and condition weight (selected w_cond) as the
final model (scripts/bench/warm_ablation.py). The only change: one expert, so there is no routing, and the
linear condition head reads
  z : the latent mean mu (variant supervised_single_z; the usual supervised-VAE choice), or
  h : the shared encoder output (variant supervised_single_h).
The technology discriminator keeps its input [z, p] (p == 1 with one expert), exactly as the K1 variant.
Writes <bench ckpt>/warm_<variant>/seed{s}_fold{f}/{model.pt, curve.tsv, injured_cells.npz (emb = mu,
logodds = DKD-vs-HKD condition log-odds per injured-PT cell, rows of harmonised_cells.tsv.gz)}.
Usage: python sup_single.py --source z --seeds 1 --device cuda:2
"""
from __future__ import annotations
import argparse, copy, json, sys, time
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[0] / "expr"))
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import load_cfg, result_dir, set_global_seed  # noqa: E402
import model as expr_model  # noqa: E402
from model import MoEVAE, GradReverse, log_norm  # noqa: E402


class SupSingleVAE(MoEVAE):
    """MoEVAE with K = 1 whose condition head reads the latent mean (source 'z') or encoder output ('h')."""

    def __init__(self, *args, cond_source: str = "z", **kw):
        super().__init__(*args, **kw)
        assert self.K == 1
        self.cond_source = cond_source
        d_in = self.z_mu.out_features if cond_source == "z" else self.gate.in_features
        self.cond_head = nn.Linear(d_in, self.cond_head.out_features)

    def forward(self, x, tech, sample=True, grl_lambda=1.0):
        xlog = log_norm(x)
        h = self.encode_h(xlog)
        p, logits = self.gate_probs(h, noisy=self.training)          # == 1 with one expert
        g = self.sparse_gate(p)
        m = self.mix(g, self.expert_out(h))
        mu, lv = self.z_mu(m), self.z_lv(m).clamp(-10, 6)
        z = mu + torch.randn_like(mu) * torch.exp(0.5 * lv) if sample else mu
        toh = F.one_hot(tech, self.n_tech).float()
        lib = x.sum(1, keepdim=True).clamp_min(1.0)
        rho, mean = self.decode(z, toh, lib)
        theta = torch.exp(self.log_theta[tech])
        cond_logits = self.cond_head(mu if self.cond_source == "z" else h)
        disc_logits = self.disc(GradReverse.apply(torch.cat([z, p], -1), grl_lambda))
        return dict(p=p, g=g, logits=logits, mu=mu, lv=lv, z=z, rho=rho, mean=mean, theta=theta, h=h,
                    cond_logits=cond_logits, disc_logits=disc_logits)


def stability_cfg(ecfg: dict) -> dict:
    """The shared-start training settings of expr.yaml (the section holding the 'stability' block)."""
    return next(v["stability"] for v in ecfg.values() if isinstance(v, dict) and "stability" in v)


def make_builder(source: str):
    def build_model(cfg: dict, n_genes: int, K: int) -> SupSingleVAE:
        from data import TECHS
        m = cfg["model"]
        return SupSingleVAE(n_genes, len(TECHS), len(cfg["condition"]["classes"]), K, n_hidden=m["n_hidden"],
                            n_backbone_layers=m["n_backbone_layers"], expert_hidden=m["expert_hidden"],
                            n_latent=m["n_latent"], top_k=m["top_k"], gate_temperature=m["gate_temperature"],
                            dropout=m["dropout"], gate_noise=m.get("gate_noise", 1.0), cond_source=source)
    return build_model


@torch.no_grad()
def cell_outputs(model: SupSingleVAE, cells, ecfg: dict, dev: str, chunk: int = 100000, bs: int = 4096) -> dict:
    """Deterministic pass (z = mu): latent mean and DKD-vs-HKD log-odds of the condition head per cell."""
    from m_moe import BenchCells
    cls = ecfg["condition"]["classes"]; i_d, i_h = cls.index("DKD"), cls.index("HKD")
    model.eval(); Z, LO = [], []
    for i in range(0, cells.n_obs, chunk):
        d = BenchCells(cells[i:i + chunk], ecfg, dev)
        for s in range(0, len(d), bs):
            bn = np.arange(s, min(s + bs, len(d))); b = torch.as_tensor(bn, device=dev)
            out = model(d.rows(bn), d.tech[b], sample=False)
            c = out["cond_logits"]
            Z.append(out["mu"].cpu().numpy()); LO.append((c[:, i_d] - c[:, i_h]).cpu().numpy())
        del d; torch.cuda.empty_cache()
    return {"emb": np.vstack(Z), "logodds": np.concatenate(LO)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["z", "h"], default="z")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--folds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--device", default="cuda:2")
    a = ap.parse_args()
    import train as expr_train
    from data import CellData
    from harm_cells import injured_cells, cell_table
    expr_train.build_model = make_builder(a.source)            # train_model builds the model through this name
    cfg = load_cfg("bench"); ecfg = copy.deepcopy(load_cfg("expr")); st = stability_cfg(ecfg)
    ecfg["train"]["device"] = a.device
    sel = json.loads((result_dir("expr_cv") / "selection.json").read_text())
    K, wc = int(sel["K"]), float(sel["w_cond"])
    variant = f"supervised_single_{a.source}"
    cvw = Path(ecfg["paths"]["ckpt_dir"]) / st["prefix"]
    out = Path(cfg["paths"]["ckpt"]) / f"warm_{variant}"; out.mkdir(parents=True, exist_ok=True)
    cells = injured_cells(cfg); cell_table(cfg, cells)
    data = None; w = {"cond": wc}
    for s in a.seeds:
        for f in a.folds:
            d = out / f"seed{s}_fold{f}"; d.mkdir(exist_ok=True)
            if (d / "injured_cells.npz").exists():
                continue
            t0 = time.time()
            if data is None:
                data = CellData(Path(ecfg["paths"]["bulk_dir"]) / "kpmp_pt_hvg.h5ad", ecfg, a.device)
            tr = np.where(data.obs.fold.values.astype(int) != f)[0]
            sd = torch.load(cvw / f"single_f{f}.pt", map_location=a.device)
            set_global_seed(s)
            m, curve = expr_train.train_model(ecfg, data, tr, 1, s, w_over=w, epochs=st["moe_epochs"], log_every=5,
                                              tag=f"warm_{variant}_s{s}_f{f}", init_from=sd,
                                              anneal_noise=st["anneal_gate_noise"],
                                              init_noise=st["expert_init_noise"])
            assert isinstance(m, SupSingleVAE)
            curve.assign(variant=variant, K=1, w=repr(w), selected_K=K, expr_cfg_sha=ecfg["_sha"]).to_csv(
                d / "curve.tsv", sep="\t", index=False)
            torch.save({"state_dict": m.state_dict(), "K": 1, "top_k": 1, "w": w, "variant": variant,
                        "cond_source": a.source}, d / "model.pt")
            res = cell_outputs(m, cells, ecfg, a.device)
            np.savez_compressed(d / "injured_cells.npz", **{k: v.astype(np.float32) for k, v in res.items()})
            print(f"warm_{variant} seed {s} fold {f}: {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
