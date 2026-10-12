"""Contrastive RNA-chromatin alignment: encoders, symmetric InfoNCE with learnable temperature (negatives
drawn within one library), retrieval metrics and the ridge readout head."""
from __future__ import annotations
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import Ridge


def mlp(d_in, d_hidden, d_out, dropout):
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_hidden, d_hidden), nn.LayerNorm(d_hidden), nn.GELU(), nn.Dropout(dropout),
                         nn.Linear(d_hidden, d_out))


class Aligner(nn.Module):
    def __init__(self, d_atac, d_rna, A: dict):
        super().__init__()
        self.atac = mlp(d_atac, A["d_hidden"], A["d_proj"], A["dropout"])
        self.rna = mlp(d_rna, A["d_hidden"], A["d_proj"], A["dropout"])
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1.0 / A["temp_init"])))
        self.max_scale = A["max_logit_scale"]

    def scale(self):
        return self.logit_scale.exp().clamp(max=self.max_scale)

    def forward(self, xa, hr):
        return F.normalize(self.atac(xa), dim=-1), F.normalize(self.rna(hr), dim=-1)


def library_batches(lib: np.ndarray, idx: np.ndarray, bs: int, rng, min_bs: int = 16) -> list[np.ndarray]:
    out = []
    for l in np.unique(lib[idx]):
        ii = rng.permutation(idx[lib[idx] == l])
        if len(ii) < min_bs:
            continue
        chunks = [ii[s:s + bs] for s in range(0, len(ii), bs)]
        if len(chunks) > 1 and len(chunks[-1]) < min_bs:
            last = chunks.pop()
            chunks[-1] = np.concatenate([chunks[-1], last])
        out += chunks
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def batch_loss(m, XA, H, b, pp, device):
    xa = torch.from_numpy(XA[pp[b]]).to(device); hr = H[torch.as_tensor(b, device=device)]
    za, zr = m(xa, hr)
    logits = m.scale() * za @ zr.T
    y = torch.arange(len(b), device=device)
    return 0.5 * (F.cross_entropy(logits, y) + F.cross_entropy(logits.T, y))


def train_aligner(XA: np.ndarray, H: torch.Tensor, lib: np.ndarray, tr: np.ndarray, A: dict, seed: int, device: str,
                  pair_perm: np.ndarray | None = None, log=None, val: np.ndarray | None = None) -> Aligner:
    """XA: chromatin features (host), H: fixed RNA embeddings (device). pair_perm: ATAC row used for each
    nucleus (identity = true pairs; within-library permutation = shuffled-pair baseline). With `val`
    (nuclei of held-aside training participants) the epoch with the lowest validation loss is kept."""
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    m = Aligner(XA.shape[1], H.shape[1], A).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=A["lr"], weight_decay=A["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=A["epochs"], eta_min=A["lr"] * 0.05)
    pp = np.arange(len(XA)) if pair_perm is None else pair_perm
    vb = library_batches(lib, val, A["batch_size"], np.random.default_rng(seed + 1)) if val is not None and len(val) else []
    best, best_state, best_ep = np.inf, None, A["epochs"]
    for ep in range(A["epochs"]):
        m.train(); tot, nb = 0.0, 0
        for b in library_batches(lib, tr, A["batch_size"], rng):
            loss = batch_loss(m, XA, H, b, pp, device)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        sched.step()
        vl = np.nan
        if vb:
            m.eval()
            with torch.no_grad():
                vl = float(np.mean([batch_loss(m, XA, H, b, pp, device).item() for b in vb]))
            if vl < best:
                best, best_ep = vl, ep + 1
                best_state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        if log is not None and (ep + 1) % 10 == 0:
            log(f"  epoch {ep + 1} loss {tot / max(nb, 1):.4f} val {vl:.4f} scale {m.scale().item():.2f}")
    if best_state is not None:
        m.load_state_dict(best_state)
    m.best_epoch = best_ep
    m.eval()
    return m


@torch.no_grad()
def embed(m: Aligner, XA: np.ndarray, H: torch.Tensor | None, device: str, bs: int = 8192):
    ZA, ZR, PA, PR = [], [], [], []
    for s in range(0, len(XA), bs):
        pa = m.atac(torch.from_numpy(XA[s:s + bs]).to(device)); PA.append(pa.cpu()); ZA.append(F.normalize(pa, dim=-1).cpu())
        if H is not None:
            pr = m.rna(H[s:s + bs]); PR.append(pr.cpu()); ZR.append(F.normalize(pr, dim=-1).cpu())
    cat = lambda L: torch.cat(L).numpy() if L else None  # noqa: E731
    return cat(ZA), cat(ZR), cat(PA), cat(PR)


def retrieval(ZA: np.ndarray, ZR: np.ndarray, lib: np.ndarray, te: np.ndarray, pool: int, topk, repeats: int,
              seed: int) -> dict:
    """Same-nucleus retrieval within each held-out library: top-k accuracy among `pool` candidates (true
    partner + random nuclei of the library) and FOSCTTM over the whole library (both directions averaged)."""
    rng = np.random.default_rng(seed)
    hits = {k: [] for k in topk}; fos = []
    for l in np.unique(lib[te]):
        ii = te[lib[te] == l]
        if len(ii) < 20:
            continue
        S = ZR[ii] @ ZA[ii].T                     # rows RNA query, cols ATAC candidates
        d = np.diag(S)
        fos.append(0.5 * ((S > d[:, None]).sum(1) / (len(ii) - 1) + (S > d[None, :]).sum(0) / (len(ii) - 1)))
        n = min(pool, len(ii))
        for _ in range(repeats):
            for SS in (S, S.T):                   # RNA->ATAC and ATAC->RNA
                Rn = rng.random(SS.shape); np.fill_diagonal(Rn, np.inf)
                cand = np.argpartition(Rn, n - 2, axis=1)[:, :n - 1]
                rank = (np.take_along_axis(SS, cand, 1) > d[:, None]).sum(1)
                for k in topk:
                    hits[k].append(rank < k)
    res = {f"top{k}_at{pool}": float(np.mean(np.concatenate(v))) if v else np.nan for k, v in hits.items()}
    res["foscttm"] = float(np.mean(np.concatenate(fos))) if fos else np.nan
    res["chance_top1"] = 1.0 / pool
    return res


def ridge_head(E_tr: np.ndarray, y_tr: np.ndarray, alpha: float) -> dict:
    mu, sd = E_tr.mean(0), E_tr.std(0) + 1e-6
    r = Ridge(alpha=alpha).fit((E_tr - mu) / sd, y_tr)
    return {"mu": mu, "sd": sd, "coef": r.coef_.astype(np.float32), "intercept": float(r.intercept_)}


def apply_head(E: np.ndarray, h: dict) -> np.ndarray:
    return ((E - h["mu"]) / h["sd"]) @ h["coef"] + h["intercept"]


@torch.no_grad()
def rna_embedding(m, X, device, bs: int = 4096) -> tuple[torch.Tensor, np.ndarray]:
    """Fixed MoE backbone output h and the DKD-vs-HKD log-odds for raw-count rows X (CSR)."""
    from model import log_norm, cond_logodds
    Hs, Ls = [], []
    for s in range(0, X.shape[0], bs):
        x = torch.from_numpy(X[s:s + bs].toarray()).to(device)
        h = m.encode_h(log_norm(x)); p, _ = m.gate_probs(h)
        Hs.append(h); Ls.append(cond_logodds(m, p).cpu().numpy())
    return torch.cat(Hs), np.concatenate(Ls)
