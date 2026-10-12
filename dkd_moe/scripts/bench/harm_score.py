"""Harmonised evaluation, step 2: participant scores for every method from injured-PT cells.

Samples = participant x technology with >= expr min_cells injured-PT cells (harm_cells.injured_cells).
Supervised steps (DE gene lists, logistic regressions) are fitted on training-fold KPMP participants of
group DKD vs group HKD (hypertensive CKD without diabetes), the classes of the expr condition head;
HKD_withDM, DM_noCKD and Reference are scored but never fitted on. As in the expr readout, the MoE
condition-head log-odds is averaged raw over injured cells and then over technologies; every other score
(and every gene / latent feature) is z-scored within technology, label-free, across all KPMP samples
(external: within cohort x technology), then averaged over a participant's technologies. External cohorts: every fold model is applied and the
participant scores are averaged over the five fold models (one rule for every method, MoE included).
Output: results/bench_harmonised/donor_scores.tsv
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from bench_data import pseudobulk, log_cpm  # noqa: E402
from bench_score import geneset_score, latent_lr_score, participant_table  # noqa: E402
from harm_cells import injured_cells  # noqa: E402
from m_pseudobulk import top_genes  # noqa: E402
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import CDL, load_cfg, result_dir, write_provenance  # noqa: E402

CASE, CTRL = "DKD", "HKD"


def sample_meta(o: pd.DataFrame) -> pd.DataFrame:
    key = (o.dataset + ":" + o.participant + "|" + o.tech).values
    m = o.assign(key=key).groupby("key").agg(dataset=("dataset", "first"), participant=("participant", "first"),
                                             tech=("tech", "first"), condition=("group", "first"),
                                             fold=("fold", "first"), n_cells=("group", "size"))
    return m


def cell_means(arr, o, meta):
    key = (o.dataset + ":" + o.participant + "|" + o.tech).values
    return pd.DataFrame(np.asarray(arr), index=key).groupby(level=0).mean().reindex(meta.index)


def deseq_genes(C, meta, ref, cfg):
    from pydeseq2.dds import DeseqDataSet
    from pydeseq2.ds import DeseqStats
    from pydeseq2.default_inference import DefaultInference
    p = cfg["pseudobulk"]
    counts = C.iloc[ref].round().astype(int); counts = counts.loc[:, counts.sum(0) > 0]
    md = meta.iloc[ref][["tech", "condition"]].copy()
    inf = DefaultInference(n_cpus=p["n_cpus"])
    dds = DeseqDataSet(counts=counts, metadata=md, design_factors=["tech", "condition"], ref_level=["condition", CTRL],
                       refit_cooks=True, inference=inf, quiet=True)
    dds.deseq2()
    st = DeseqStats(dds, contrast=["condition", CASE, CTRL], inference=inf, quiet=True); st.summary()
    return top_genes(st.results_df, p["n_top"], p["padj_max"])


def train_lfc(L, meta, ref):
    c = meta.condition.values[ref]
    return L.iloc[ref][c == CASE].mean(0) - L.iloc[ref][c == CTRL].mean(0)


def scores_for(run, method, z, L, meta, ref, cfg, seed, fold, ck):
    """Raw per-sample scores (name -> Series over meta.index) for one fold model."""
    C = cfg["scvi"]["lr_C"]; k = cfg["pseudobulk"]["n_top"]; S = {}
    o = OBS; fit = ref; ref = STD
    if "emb" in z:
        S["latent_LR"] = latent_lr_score(cell_means(z["emb"], o, meta), meta, ref, CASE, C, seed, fit)
    if method == "scvi":
        de = pd.read_csv(ck / "harmonised_scvi_de.tsv.gz", sep="\t", index_col=0).sort_values("lfc_mean")
        S["DE_set"] = geneset_score(L, meta, de.index[::-1][:k].tolist(), de.index[:k].tolist(), ref)
    elif method == "cvae":
        S["counterfactual_LLR"] = cell_means(z["llr"][:, None], o, meta)[0]
        sh = pd.read_csv(CDL / f"results/{run_dir(run)}/tables/seed{seed}_fold{fold}_decoded_shift.tsv.gz",
                         sep="\t", index_col=0).iloc[:, 0].sort_values(ascending=False)
        S["decoded_set"] = geneset_score(L, meta, sh.index[:k].tolist(), sh.index[::-1][:k].tolist(), ref)
    elif method == "scdisinfact":
        S["zd_LR"] = latent_lr_score(cell_means(z["zd"], o, meta), meta, ref, CASE, C, seed, fit)
        ckg = pd.read_csv(CDL / f"results/{run_dir(run)}/tables/seed{seed}_fold{fold}_ckg.tsv.gz", sep="\t",
                          index_col=0).ckg.sort_values(ascending=False)
        top = ckg.index[:cfg["scdisinfact"]["ckg_n_top"]]; lfc = train_lfc(L, meta, fit)
        S["CKG_set"] = geneset_score(L, meta, [g for g in top if lfc[g] > 0], [g for g in top if lfc[g] <= 0], ref)
    elif method == "moe":
        if z["p"].shape[1] > 1:
            S["gate_LR"] = latent_lr_score(cell_means(z["p"], o, meta), meta, ref, CASE, C, seed, fit)
            S["condition_head"] = cell_means(z["logodds"][:, None], o, meta)[0]
    return S


def run_dir(run):
    return f"bench_{run}"


def label(run):
    if run.startswith("warm_"):
        v = run[len("warm_"):]
        return "moe_final_warm" if v == "full" else f"moe_warm_{v}"
    base = run.rsplit("_s", 1)[0]
    return f"{base}_randinit" if base.startswith("moe_") else base


def main():
    global OBS, STD
    cfg = load_cfg("bench"); out = result_dir("bench_harmonised"); ck_root = Path(cfg["paths"]["ckpt"])
    cells = injured_cells(cfg)
    OBS = pd.read_csv(ck_root / "harmonised_cells.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    assert (OBS.index == cells.obs_names).all()
    meta = sample_meta(OBS)
    STD = (meta.dataset == "KPMP").values          # label-free standardisation rows (all KPMP samples)
    Cs, Ms = [], []
    for ds in OBS.dataset.unique():
        c, m = pseudobulk(cells[(cells.obs.dataset == ds).values], 1)
        Cs.append(c.set_axis(ds + ":" + c.index, axis=0))
    Cnt = pd.concat(Cs).reindex(meta.index).fillna(0); L = log_cpm(Cnt)
    rows = []
    for f in range(5):                                   # pseudobulk DE: deterministic, one 'seed'
        ref = ((meta.dataset == "KPMP") & (meta.fold != f) & meta.condition.isin([CASE, CTRL])).values
        up, down = deseq_genes(Cnt, meta, ref, cfg)
        rows.append(participant_table(geneset_score(L, meta, up, down, STD), meta, STD, "pseudobulk", "DE_set", 0, f))
    runs = sorted({p.parents[1].name for p in ck_root.glob("*/seed*_fold*/injured_cells.npz")})
    for run in runs:
        method = "moe" if run.startswith(("moe", "warm")) else run.rsplit("_s", 1)[0]
        for npz in sorted((ck_root / run).glob("seed*_fold*/injured_cells.npz")):
            seed, fold = map(int, npz.parent.name.replace("seed", "").split("_fold"))
            ref = ((meta.dataset == "KPMP") & (meta.fold != fold) & meta.condition.isin([CASE, CTRL])).values
            S = scores_for(run, method, dict(np.load(npz)), L, meta, ref, cfg, seed, fold, npz.parent)
            for name, s in S.items():
                rows.append(participant_table(s.reindex(meta.index), meta, STD, label(run), name, seed, fold,
                                              standardise=(name != "condition_head")))
        print(run, flush=True)
    D = pd.concat(rows).rename(columns={"condition": "group"})
    D.to_csv(out / "donor_scores.tsv", sep="\t", index=False)
    meta.to_csv(out / "samples.tsv", sep="\t")
    write_provenance("bench_harmonised", [ck_root / "harmonised_cells.tsv.gz"],
                     [out / "donor_scores.tsv", out / "samples.tsv"], cfg["seed"], cfg,
                     extra={"runs": runs, "rule": "injured-PT samples; fit DKD vs HKD; externals = fold-model mean"})


if __name__ == "__main__":
    main()
