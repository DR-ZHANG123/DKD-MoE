"""03c_external_calibrated: GSE211785 evaluation with cohort-level input standardisation.

The GSE211785 sections differ from KPMP in platform (FFPE probe panel vs fresh-frozen whole
transcriptome) and in slide scanning/staining. Inputs of the external cohort are therefore
standardised with the external cohort's own mean and sd (image dimensions and genes; no labels or
pairings used) before the trained heads, instead of the KPMP training statistics. Retrieval is
re-evaluated for every trained model and the calibrated external embeddings are saved next to the
model (<stem>_extcal.npz) for 04_interpret (interpret.external_calibration).
Outputs results/align_train/retrieval_sections_external_calibrated.tsv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from data import SpotData  # noqa: E402
from model import AlignModel  # noqa: E402
import evaluate as EV  # noqa: E402

RES = "align_train"


def zs(A):
    sd = A.std(0)
    return ((A - A.mean(0)) / np.where(sd > 1e-6, sd, 1.0)).astype(np.float32)


def main():
    cfg = C.load_cfg(); C.set_seed(cfg["seed"]); out = C.result_dir(RES); ck = C.p(cfg["paths"]["ckpt"])
    S = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    D = SpotData(S); m = D.meta
    ext = np.where(m.cohort != "kpmp")[0]; mm = m.iloc[ext].reset_index(drop=True)
    coh = mm.cohort.values
    def zc(A):   # standardise each external cohort with its own statistics
        out_ = np.empty_like(A, dtype=np.float32)
        for c in set(coh):
            out_[coh == c] = zs(A[coh == c])
        return out_
    Ez = zc(D.E[ext]); xcache = {}
    rows = []
    for f in sorted(ck.glob("*_fold*_seed*.pt")):
        st = torch.load(f, map_location="cpu")
        key = tuple(st["hvg_idx"])
        if key not in xcache:
            xcache[key] = zc(D.lognorm(np.asarray(st["hvg_idx"]))[ext])
        mc = dict(cfg["model"]); mc.update(st["cfg_model"]); g = st["n_genes"]
        M = AlignModel(g, mc, np.zeros(mc["emb_dim"], np.float32), np.ones(mc["emb_dim"], np.float32),
                       np.zeros(g, np.float32), np.ones(g, np.float32))
        M.load_state_dict(st["state"]); M.eval()
        with torch.no_grad():
            zi = F.normalize(M.img(torch.from_numpy(Ez)), dim=-1).numpy()
            ze = F.normalize(M.expr(torch.from_numpy(xcache[key])), dim=-1).numpy()
        np.savez_compressed(str(f).replace(".pt", "_extcal.npz"), zi=zi.astype(np.float16), ze=ze.astype(np.float16), rows=ext)
        name, rest = f.stem.split("_fold")
        fold, seed = int(rest.split("_seed")[0]), int(rest.split("_seed")[1])
        for c in sorted(set(coh)):
            k = np.where(coh == c)[0]
            rr = EV.within_section(zi[k], ze[k], mm.iloc[k].reset_index(drop=True), cfg["eval"]["topk"])
            rows += [dict(r, fold=fold, seed=seed, model=name, eval_set=f"{c}_external_calibrated") for r in rr]
            print(f.name, c, round(pd.DataFrame(rr).top5_img2expr.mean(), 4), flush=True)
    R = pd.DataFrame(rows); R.to_csv(out / "retrieval_sections_external_calibrated.tsv", sep="\t", index=False)
    C.provenance("align_train_extcal", [C.rpath("align_prep") / "sections.tsv"],
                 [out / "retrieval_sections_external_calibrated.tsv"], cfg["seed"])


if __name__ == "__main__":
    main()
