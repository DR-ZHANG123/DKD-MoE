"""03_train: cross-fitted contrastive alignment of H&E patch embeddings and spot expression.

For each KPMP fold f (csbj_dl/config/splits.tsv): train on KPMP participants of the other folds
(an inner set of training participants is held aside for early stopping), evaluate on fold-f
participants (held out) and on GSE211785 Visium (external; tissue localisation). Runs:
batch modes x seeds, plus a shuffled-pair baseline (expression rows permuted within each training
section: same data, broken image-expression correspondence).
Usage: python 03_train.py [--seeds 0,1] [--folds 0,1] [--modes within_section] [--no-baseline]
Outputs results/align_train/{retrieval_sections.tsv, retrieval_global.tsv, training_curves.tsv,
runs.tsv}; checkpoints + evaluation embeddings under paths.ckpt.
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
from data import SpotData, select_hvgs, standardiser  # noqa: E402
from model import AlignModel, Sampler, info_nce, embed_all  # noqa: E402
import evaluate as EV  # noqa: E402

RES = "align_train"


def section_loss(model, E, Xe, rows, sec_idx, dev, max_n=2048):
    """Mean within-section InfoNCE over whole sections (validation criterion, mode-independent)."""
    model.eval(); ls = []
    with torch.no_grad():
        for s in np.unique(sec_idx[rows]):
            r = rows[sec_idx[rows] == s][:max_n]
            zi = model.encode_img(torch.from_numpy(E[r]).to(dev)); ze = model.encode_expr(torch.from_numpy(Xe[r]).to(dev))
            ls.append(info_nce(zi, ze, model.scale()).item() - np.log(len(r)))   # relative to chance
    return float(np.mean(ls))


def train_one(D, Xe, tr, va, mode, seed, cfg, dev, shuffle=False):
    C.set_seed(seed); rng = np.random.default_rng(seed)
    T, M = cfg["train"], cfg["model"]
    sec = D.meta.sec_idx.values
    X_pair = Xe
    if shuffle:   # permute expression within each training section
        perm = np.arange(len(Xe))
        for s in np.unique(sec[tr]):
            r = tr[sec[tr] == s]; perm[r] = rng.permutation(r)
        X_pair = Xe[perm]
    imu, isd = standardiser(D.E, tr); emu, esd = standardiser(Xe, tr)
    model = AlignModel(Xe.shape[1], M, imu, isd, emu, esd).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=T["lr"], weight_decay=T["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T["epochs"] * T["steps_per_epoch"])
    samp = Sampler(sec, tr, mode, T["batch_size"], rng)
    best, best_state, bad, curve = np.inf, None, 0, []
    for ep in range(T["epochs"]):
        model.train(); tl = []
        for _ in range(T["steps_per_epoch"]):
            r, blk = samp()
            zi = model.encode_img(torch.from_numpy(D.E[r]).to(dev))
            ze = model.encode_expr(torch.from_numpy(X_pair[r]).to(dev))
            b = torch.from_numpy(blk).to(dev) if blk is not None else None
            loss = info_nce(zi, ze, model.scale(), b)
            opt.zero_grad(); loss.backward(); opt.step(); sched.step(); tl.append(loss.item())
        vl = section_loss(model, D.E, Xe, va, sec, dev)
        curve.append({"epoch": ep, "train_loss": float(np.mean(tl)), "val_rel_loss": vl,
                      "temperature": float(1 / model.scale().item())})
        if vl < best - 1e-4:
            best, bad = vl, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= T["patience"]:
                break
    model.load_state_dict(best_state)
    return model, curve


def run_job(D, Xe, tr, va, ho, ext, mo, s, shuf, f, name, stem, hvg, hsrc, cfg, dev):
    """Train one model, evaluate retrieval, save checkpoint and all-spot embeddings."""
    t0 = time.time(); m = D.meta; K = cfg["eval"]["topk"]
    model, curve = train_one(D, Xe, tr, va, mo, s, cfg, dev, shuffle=shuf)
    ev = np.r_[ho, ext]
    zi, ze = embed_all(model, D.E, Xe, ev, dev)
    sec_rows, glob_rows = [], []
    sets = [("kpmp_heldout", np.arange(len(ho)))]
    coh = m.cohort.values[ext]
    sets += [(f"{c}_external", len(ho) + np.where(coh == c)[0]) for c in sorted(set(coh))]
    for setname, idx in sets:
        mm = m.iloc[ev[idx]].reset_index(drop=True)
        for r in EV.within_section(zi[idx], ze[idx], mm, K):
            sec_rows.append(dict(r, fold=f, seed=s, model=name, eval_set=setname))
        g = EV.global_pool(zi[idx], ze[idx], mm, K, cfg["eval"]["global_pool"], cfg["eval"]["n_global_queries"],
                           np.random.default_rng(s))
        glob_rows.append(dict(g, fold=f, seed=s, model=name, eval_set=setname))
    torch.save({"state": model.state_dict(), "hvg_idx": hvg, "n_genes": len(hvg), "cfg_model": cfg["model"]}, f"{stem}.pt")
    # embeddings of every spot (training ones are needed for the training-derived direction)
    zia, zea = embed_all(model, D.E, Xe, np.arange(len(m)), dev)
    np.savez_compressed(f"{stem}_emb.npz", zi=zia.astype(np.float16), ze=zea.astype(np.float16),
                        train=np.isin(np.arange(len(m)), np.r_[tr, va]))
    sw = pd.DataFrame(sec_rows)
    run = {"fold": f, "seed": s, "model": name, "epochs": len(curve), "minutes": (time.time() - t0) / 60,
           "temperature": curve[-1]["temperature"], "n_genes": len(hvg), "genes_source": hsrc,
           "heldout_top1": sw[sw.eval_set == "kpmp_heldout"].top1_img2expr.mean(),
           "heldout_top5": sw[sw.eval_set == "kpmp_heldout"].top5_img2expr.mean(),
           "chance_top5": sw[sw.eval_set == "kpmp_heldout"].chance_top5.mean(),
           "external_top5": sw[sw.eval_set == "gse211785_external"].top5_img2expr.mean()}
    del model; torch.cuda.empty_cache()
    return {"sections": sec_rows, "global": glob_rows, "curve": [dict(c, fold=f, seed=s, model=name) for c in curve], "run": run}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds"); ap.add_argument("--folds"); ap.add_argument("--modes")
    ap.add_argument("--no-baseline", action="store_true"); ap.add_argument("--tag", default="")
    a = ap.parse_args()
    cfg = C.load_cfg(); dev = torch.device(cfg["device"])
    sel = C.CSBJ / "results/align_select/selected.yaml"   # selected once, on the primary track
    if sel.exists():   # regularisation chosen on inner-validation participants (03a_select)
        import yaml
        over = yaml.safe_load(sel.read_text())
        for k in ("model", "train", "data"):
            cfg[k].update(over.get(k, {}))
        print("using selected setting", over["setting"], flush=True)
    seeds = [int(x) for x in a.seeds.split(",")] if a.seeds else cfg["seeds"]
    modes = a.modes.split(",") if a.modes else cfg["train"]["batch_modes"]
    out = C.result_dir(RES); ck = C.p(cfg["paths"]["ckpt"]); ck.mkdir(parents=True, exist_ok=True)
    S = pd.read_csv(C.rpath("align_prep") / "sections.tsv", sep="\t", dtype={"participant": str})
    D = SpotData(S)
    sp = C.splits()
    m = D.meta
    fold = m.participant.map(sp.fold).where(m.cohort == "kpmp")
    folds = [int(x) for x in a.folds.split(",")] if a.folds else sorted(int(x) for x in sp.fold.dropna().unique())
    ext = np.where(m.cohort != "kpmp")[0]          # external cohorts: never trained on
    K = cfg["eval"]["topk"]
    sec_rows, glob_rows, curves, runs = [], [], [], []
    for f in folds:
        trp = sorted(m.participant[(fold != f) & fold.notna()].unique())
        rng = np.random.default_rng(cfg["seed"] + f)
        vap = set(rng.choice(trp, max(2, int(round(len(trp) * cfg["train"]["val_frac_participants"]))), replace=False))
        tr = np.where(m.participant.isin(trp) & ~m.participant.isin(vap))[0]
        va = np.where(m.participant.isin(vap))[0]
        ho = np.where(fold == f)[0]
        hvg, hsrc = select_hvgs(D, np.isin(np.arange(len(m)), tr), cfg["data"]["n_hvg"], cfg["data"]["hvg_source"])
        Xe = D.lognorm(hvg)
        print(f"fold {f}: train {len(tr)} spots/{m.participant[tr].nunique()} participants, val {len(va)}, "
              f"held-out {len(ho)}/{m.participant[ho].nunique()}, genes {len(hvg)} ({hsrc})", flush=True)
        jobs = [(mo, s, False) for mo in modes for s in seeds]
        if not a.no_baseline:
            jobs += [(cfg["train"]["primary_batch_mode"], s, True) for s in seeds]
        for mo, s, shuf in jobs:
            name = f"{'shuffled' if shuf else 'aligned'}_{mo}"
            stem = ck / f"{name}_fold{f}_seed{s}"
            mfile = Path(f"{stem}_metrics.json")
            if not mfile.exists():          # resumable: finished runs are not retrained
                for attempt in range(cfg["train"].get("oom_retries", 20)):
                    try:
                        res = run_job(D, Xe, tr, va, ho, ext, mo, s, shuf, f, name, stem, hvg, hsrc, cfg, dev)
                        break
                    except torch.cuda.OutOfMemoryError:   # GPU shared with other jobs: wait and retry
                        torch.cuda.empty_cache(); print("OOM, retrying", attempt, flush=True); time.sleep(60)
                else:
                    raise SystemExit("repeated CUDA OOM")
                mfile.write_text(json.dumps(res, default=float))
            res = json.loads(mfile.read_text())
            sec_rows += res["sections"]; glob_rows += res["global"]; curves += res["curve"]; runs.append(res["run"])
            print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in res["run"].items()}, flush=True)
        m[["participant"]].assign(fold=fold).to_csv(out / "spot_folds.tsv.gz", sep="\t", index=False)
        sfx = a.tag
        pd.DataFrame(sec_rows).to_csv(out / f"retrieval_sections{sfx}.tsv", sep="\t", index=False)
        pd.DataFrame(glob_rows).to_csv(out / f"retrieval_global{sfx}.tsv", sep="\t", index=False)
        pd.DataFrame(curves).to_csv(out / f"training_curves{sfx}.tsv", sep="\t", index=False)
        pd.DataFrame(runs).to_csv(out / f"runs{sfx}.tsv", sep="\t", index=False)
    m.drop(columns=D.feat_names).to_csv(C.p(cfg["paths"]["ckpt"]) / "spot_meta.tsv.gz", sep="\t", index=False)
    outs = sorted(out.glob("*.tsv"))
    C.provenance(RES, [C.rpath("align_prep") / "sections.tsv", C.p(cfg["paths"]["splits"])], outs, cfg["seed"],
                 extra={"seeds": seeds, "modes": modes, "folds": folds, "tag": a.tag})


if __name__ == "__main__":
    main()
