"""Score the fixed KPMP-derived signature (results/expr_signature/signature.tsv) in external cohorts, test-only.

Score = mean z of measured up genes - mean z of measured down genes; z across samples within each
cohort x platform / technology. Cohort is scored only with >= min_genes_per_direction measured genes per direction.
(a) GSE104954 ERCB tubulointerstitium arrays, platforms separate (loading as in scripts/stages/08).
(b) Zenodo 19868428 CosMx / Xenium: donor-level pooled log1p(CP10k) of measured panel genes in author iPT
    (primary) and PT cells; donors shared with GSE211785, HK2874, paediatric and mixed-aetiology donors excluded.
(c) GSE211785 / GSE195460 injured-PT pseudobulk per participant x technology (log2 CPM over the HVGs).
Random-effects (DerSimonian-Laird) summaries for DKD vs disease control and DKD vs healthy control.
Outputs: results/expr_external/{scores_*.tsv, signature_tests.tsv, coverage.tsv, meta_analysis.tsv}
"""
from __future__ import annotations
import gzip, importlib, sys
from io import StringIO
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import ROOT, load_cfg, result_dir, write_provenance  # noqa: E402
from stats_utils import hedges_g  # noqa: E402

OUT = "expr_external"          # result folder for score tables (overridable by callers)
DX = {"Diabetic nephropathy": "DKD", "Hypertensive nephropathy": "HTN", "Living donor": "LD"}


def score(expr: pd.DataFrame, up, down, nmin):
    """expr: samples x genes (log scale). Returns score Series (or None) and coverage."""
    z = (expr - expr.mean()) / expr.std(ddof=1).replace(0, np.nan)
    u = [g for g in up if g in z.columns and z[g].notna().all()]
    d = [g for g in down if g in z.columns and z[g].notna().all()]
    cov = {"n_up_measured": len(u), "n_down_measured": len(d)}
    if len(u) < nmin or len(d) < nmin:
        return None, cov
    return z[u].mean(1) - z[d].mean(1), cov


def ercb(cfg, up, down, nmin):
    raw = ROOT / "data/raw"
    hg = pd.read_csv(raw / "reference/hgnc_complete_set.txt", sep="\t", usecols=["symbol", "entrez_id", "status"], low_memory=False)
    hg = hg[(hg.status == "Approved") & hg.entrez_id.notna()]
    e2s = dict(zip(hg.entrez_id.astype(int).astype(str) + "_at", hg.symbol))
    samples = pd.read_csv(ROOT / "results/01_cohort_audit/sample_table_GSE104954.tsv", sep="\t").set_index("sample")
    rows, sc_all = [], []
    for plat, contrasts in cfg["readout_stability"]["external"]["ercb_platforms"].items():
        txt = gzip.open(raw / f"GSE104954/GSE104954-{plat}_series_matrix.txt.gz", "rt").read()
        m = pd.read_csv(StringIO(txt.split("!series_matrix_table_begin\n")[1].split("!series_matrix_table_end")[0]), sep="\t", index_col=0)
        m.index = m.index.map(lambda x: e2s.get(x)); m = m[m.index.notna()].groupby(level=0).mean()
        grp = samples.diagnosis.reindex(m.columns).map(DX); keep = grp.notna()
        expr = m.loc[:, keep].T; grp = grp[keep]
        expr = expr.loc[:, expr.mean() > np.quantile(expr.mean(), 0.25)]
        s, cov = score(expr, up, down, nmin)
        if s is None:
            rows.append({"cohort": "GSE104954", "platform": plat, **cov}); continue
        sc_all.append(pd.DataFrame({"sample": s.index, "score": s.values, "group": grp.values, "platform": plat}))
        for a, b in contrasts:
            rows.append({"cohort": "GSE104954", "platform": plat, "comparison": f"{a}_vs_{b}", **cov,
                         **hedges_g(s[grp == a], s[grp == b])})
    pd.concat(sc_all).to_csv(result_dir(OUT) / "scores_ercb.tsv", sep="\t", index=False)
    return rows


def zenodo(cfg, up, down, nmin, mc):
    sys.path.insert(0, str(ROOT / "scripts" / "stages"))
    z12 = importlib.import_module("12b_zenodo_gene_level")
    h5 = ROOT / cfg["readout_stability"]["external"]["zenodo_h5"]
    a = ad.read_h5ad(h5, backed="r"); var = pd.Index(a.var_names.astype(str))
    obs = a.obs[["orig_ident", "tech", "annotation_updated", "Condition", "Age"]].copy()
    obs["donor"] = obs.orig_ident.astype(str).str.replace(r"_2$", "", regex=True)
    obs["celltype"] = obs.annotation_updated.astype(str); obs["tech"] = obs.tech.astype(str)
    overlap = set(pd.read_csv(ROOT / "results/01_cohort_audit/donor_overlap.tsv", sep="\t").donor) | {"HK2874"}
    obs["group"] = obs.Condition.astype(str).map(z12.GROUP)
    obs["use"] = (obs.celltype.isin(["iPT", "PT"]) & ~obs.donor.isin(overlap) & obs.group.notna()
                  & (pd.to_numeric(obs.Age, errors="coerce") >= 18)).values
    genes = [g for g in list(up) + list(down) if g in var]
    rows, scs = [], []
    if not genes:
        return [{"cohort": "Zenodo19868428", "n_up_measured": 0, "n_down_measured": 0}]
    agg, det = z12.stream(obs, genes, var)
    grp = obs[obs.use].groupby("donor").group.first()
    agg = agg[agg.n_cells >= mc].copy(); agg["group"] = agg.donor.map(grp)
    for (tech, ct), d in agg.groupby(["tech", "celltype"]):
        meas = [g for g in genes if det[tech][g] > 0]
        d = d.set_index("donor")
        expr = np.log1p(d[meas].div(d.total, axis=0) * 1e4)
        s, cov = score(expr, up, down, nmin)
        base = {"cohort": "Zenodo19868428", "platform": tech, "celltype": ct, **cov}
        if s is None:
            rows.append(base); continue
        scs.append(pd.DataFrame({"donor": s.index, "score": s.values, "group": d.group.values, "platform": tech, "celltype": ct}))
        for a_, b_ in cfg["readout_stability"]["external"]["zenodo_contrasts"]:
            rows.append({**base, "comparison": f"{a_}_vs_{b_}", **hedges_g(s[d.group == a_], s[d.group == b_])})
    if scs:
        pd.concat(scs).to_csv(result_dir(OUT) / "scores_zenodo.tsv", sep="\t", index=False)
    return rows


def sc_cohorts(cfg, up, down, nmin, mc):
    bulk = Path(cfg["paths"]["bulk_dir"]); rows, scs = [], []
    comps = {"gse211785": [("DKD", "HKD"), ("DKD", "Control"), ("HKD", "Control")], "gse195460": [("DKD", "Control")]}
    for c, cc in comps.items():
        a = ad.read_h5ad(bulk / f"{c}_pt_hvg.h5ad")
        a = a[(a.obs.pt_state.astype(str) == "injured_PT").values]
        o = a.obs[["participant", "group", "tech"]].astype(str)
        per_tech = []
        for tech in o.tech.unique():
            m = (o.tech == tech).values
            n = o[m].groupby("participant").size(); keep = n[n >= mc].index
            pb = {p: np.asarray(a.X[m & (o.participant == p).values].sum(0)).ravel() for p in keep}
            pb = pd.DataFrame(pb, index=a.var_names).T
            lcpm = np.log2(pb.div(pb.sum(1), axis=0) * 1e6 + 1)
            lcpm = lcpm.loc[:, (lcpm > 1).mean() >= 0.5]
            s, cov = score(lcpm, up, down, nmin)
            g = o[m].groupby("participant").group.first().reindex(s.index)
            per_tech.append(pd.DataFrame({"participant": s.index, "score": s.values, "group": g.values, "tech": tech}))
            for a_, b_ in cc:
                rows.append({"cohort": c.upper(), "platform": tech, "comparison": f"{a_}_vs_{b_}", **cov,
                             **hedges_g(s[g == a_], s[g == b_])})
        P = pd.concat(per_tech)
        pooled = P.groupby("participant").agg(score=("score", "mean"), group=("group", "first")).reset_index().assign(tech="pooled")
        if P.tech.nunique() > 1:
            for a_, b_ in cc:
                rows.append({"cohort": c.upper(), "platform": "pooled", "comparison": f"{a_}_vs_{b_}",
                             **hedges_g(pooled.score[pooled.group == a_], pooled.score[pooled.group == b_])})
        scs.append(pd.concat([P, pooled]).assign(cohort=c.upper()))
    pd.concat(scs).to_csv(result_dir(OUT) / "scores_sc_pseudobulk.tsv", sep="\t", index=False)
    return rows


def dersimonian_laird(T: pd.DataFrame) -> dict:
    y = T.g.values; v = ((T.ci_high - T.ci_low).values / 3.92) ** 2
    w = 1 / v; fe = (w * y).sum() / w.sum(); Q = (w * (y - fe) ** 2).sum(); k = len(y)
    tau2 = max(0.0, (Q - (k - 1)) / (w.sum() - (w ** 2).sum() / w.sum())) if k > 1 else 0.0
    ws = 1 / (v + tau2); est = (ws * y).sum() / ws.sum(); se = np.sqrt(1 / ws.sum())
    return {"k": k, "g_random": est, "ci_low": est - 1.96 * se, "ci_high": est + 1.96 * se,
            "p": 2 * stats.norm.sf(abs(est / se)), "tau2": tau2, "Q": Q,
            "I2": max(0.0, (Q - (k - 1)) / Q) if Q > 0 else 0.0, "studies": "; ".join(T.label)}


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_external"); sc = cfg["readout_stability"]["signature"]
    sig = pd.read_csv(result_dir("expr_signature") / "signature.tsv", sep="\t")
    up, down = sig[sig.direction == "up"].gene.tolist(), sig[sig.direction == "down"].gene.tolist()
    nmin, mc = sc["min_genes_per_direction"], cfg["prep"]["min_cells_per_participant"]
    rows = ercb(cfg, up, down, nmin) + zenodo(cfg, up, down, nmin, mc) + sc_cohorts(cfg, up, down, nmin, mc)
    T = pd.DataFrame(rows); T.to_csv(out / "signature_tests.tsv", sep="\t", index=False)
    t = T.dropna(subset=["g"]).copy()
    t["label"] = t.cohort + ":" + t.platform.astype(str) + t.get("celltype", pd.Series("", index=t.index)).fillna("").map(lambda x: f":{x}" if x else "")
    disease = (t.comparison.isin(["DKD_vs_HTN", "DKD_vs_HKD", "DKD_vs_other_CKD"]))
    healthy = t.comparison.isin(["DKD_vs_LD", "DKD_vs_Control"])
    # one estimate per independent sample set: pooled technologies for GSE211785, iPT for Zenodo
    sel = (t.platform != "sc") & (t.platform != "sn") | (t.cohort == "GSE195460")
    zp = cfg["readout_stability"]["external"]["meta_zenodo_platform"]
    sel &= ~((t.cohort == "Zenodo19868428") & ((t.get("celltype") != "iPT") | (t.platform != zp)))
    meta = []
    for name, m in [("DKD_vs_disease_control", disease), ("DKD_vs_healthy_control", healthy)]:
        d = t[m & sel]
        if len(d):
            meta.append({"summary": name, **dersimonian_laird(d)})
    M = pd.DataFrame(meta); M.to_csv(out / "meta_analysis.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["cohort", "platform", "celltype", "comparison", "n_up_measured", "n_down_measured", "n1", "n2",
                        "g", "ci_low", "ci_high", "p_wilcoxon"] if c in T.columns]
    print(T[cols].round(3).to_string()); print(M.round(3).to_string())
    write_provenance("expr_external", [result_dir("expr_signature") / "signature.tsv"],
                     sorted(out.glob("*.tsv")), cfg["seed"], cfg, extra={"n_up": len(up), "n_down": len(down)})


if __name__ == "__main__":
    main()
