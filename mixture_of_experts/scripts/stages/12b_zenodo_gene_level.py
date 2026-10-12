"""12b_zenodo_gene_level: gene-level localisation of measured programme genes in the independent
CosMx / Xenium DKD atlas (Zenodo 19868428).

The DKDiPT marker programme has only 3 genes on the panels (< 5 required minimum), so no
programme score is computed; genes are reported individually. Measured genes = genes detected in
>= 1 cell of that platform. Donors shared with the discovery atlas, HK2874 (discordant diagnosis),
paediatric donors and mixed-aetiology samples are excluded. Unit = donor (replicate sections
merged). Per donor x platform x author cell type (iPT, PT): pooled log1p(counts per 10k)."""
from __future__ import annotations
import sys
from pathlib import Path
import h5py
import numpy as np
import pandas as pd
import anndata as ad
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "lib"))
from repro import ROOT, load_config, set_global_seed, stage_dir, write_provenance
from stats import compare

STAGE = "12b_zenodo_gene_level"
H5 = ROOT / "data/raw/zenodo_dkd_spatial/spatial_adata_xenium_cosmx_zenodo.h5ad"
GROUP = {"DKD": "DKD", "Control": "Control", "DM": "DM_noDKD", "DM/HTN": "DM_noDKD", "IgA": "other_CKD",
         "MN": "other_CKD", "FSGS": "other_CKD", "C3GN": "other_CKD", "AA amyloid": "other_CKD", "TMA": "other_CKD",
         "CKD": "other_CKD", "HTN": "other_CKD"}          # DKD+FSGS (mixed) excluded
GFR = {"<30": 0, "30-60": 1, ">60": 2}


def stream(obs: pd.DataFrame, genes: list[str], var: pd.Index, chunk: int = 250_000):
    """One pass over layers/counts: per-platform detection counts for all genes, and per
    (donor, platform, celltype) sums of target genes + total counts."""
    gi = np.array([var.get_loc(g) for g in genes])
    pos = -np.ones(len(var), int); pos[gi] = np.arange(len(gi))
    key = (obs.donor + "|" + obs.tech + "|" + obs.celltype).values
    use = obs.use.values
    codes, uniq = pd.factorize(key)
    sums = np.zeros((len(uniq), len(gi))); tots = np.zeros(len(uniq)); ncell = np.zeros(len(uniq))
    det = {t: np.zeros(len(var)) for t in obs.tech.unique()}
    tech = obs.tech.values
    with h5py.File(H5, "r") as f:
        g = f["layers/counts"]; ip = g["indptr"][:]
        for s in range(0, len(ip) - 1, chunk):
            e = min(s + chunk, len(ip) - 1)
            ind = g["indices"][ip[s]:ip[e]]; dat = g["data"][ip[s]:ip[e]]
            row = np.repeat(np.arange(s, e), np.diff(ip[s:e + 1]))
            for t in det:
                m = tech[row] == t
                det[t] += np.bincount(ind[m], minlength=len(var))
            rs = np.bincount(row - s, weights=dat, minlength=e - s)
            u = use[s:e]
            np.add.at(tots, codes[s:e][u], rs[u]); np.add.at(ncell, codes[s:e][u], 1)
            m = (pos[ind] >= 0) & use[row]
            np.add.at(sums, (codes[row[m]], pos[ind[m]]), dat[m])
    parts = pd.Series(uniq).str.split("|", expand=True)
    agg = pd.DataFrame(sums, columns=genes)
    agg["total"] = tots; agg["n_cells"] = ncell
    agg[["donor", "tech", "celltype"]] = parts.values
    return agg[agg.n_cells > 0], {t: pd.Series(v, index=var) for t, v in det.items()}


def main():
    cfg = load_config(); seed = cfg["seed"]; set_global_seed(seed); R = cfg["robustness"]
    out = stage_dir(STAGE)
    a = ad.read_h5ad(H5, backed="r")
    var = pd.Index(a.var_names.astype(str))
    obs = a.obs[["orig_ident", "tech", "annotation_updated", "Condition", "Age"]].copy()
    obs["donor"] = obs.orig_ident.astype(str).str.replace(r"_2$", "", regex=True)
    obs["celltype"] = obs.annotation_updated.astype(str); obs["tech"] = obs.tech.astype(str)
    overlap = set(pd.read_csv(ROOT / "results/01_cohort_audit/donor_overlap.tsv", sep="\t").donor) | {"HK2874"}
    obs["group"] = obs.Condition.astype(str).map(GROUP)
    obs["use"] = (obs.celltype.isin(["iPT", "PT"]) & ~obs.donor.isin(overlap) & obs.group.notna()
                  & (pd.to_numeric(obs.Age, errors="coerce") >= 18)).values
    gs = pd.read_csv(ROOT / "results/05_lock_programme/gene_sets.tsv", sep="\t")
    shared = [g for g in gs[gs.set_id == "SharedInjury_up"].gene if g in var]
    targets = [g for g in R["zenodo_genes"] if g in var]
    genes = list(dict.fromkeys(targets + shared))
    agg, det = stream(obs, genes, var)
    measured = pd.DataFrame({t: (d > 0) for t, d in det.items()})
    measured.to_csv(out / "measured_genes_by_platform.tsv", sep="\t")
    dx = pd.read_excel(ROOT / "data/raw/zenodo_dkd_spatial/Diagnosis.xlsx")
    dx["donor"] = dx["Sample ID"].str.replace(r"(_2)?_(CosMx|Xenium)$", "", regex=True)
    gfr = dx.drop_duplicates("donor").set_index("donor").GFR.map(GFR)
    grp = obs[obs.use].groupby("donor").group.first()
    agg = agg[agg.n_cells >= R["zenodo_min_cells"]].copy()
    agg["group"] = agg.donor.map(grp); agg["gfr_ord"] = agg.donor.map(gfr)
    for gname in genes:
        agg[f"lcpm_{gname}"] = np.log1p(agg[gname] / agg.total * 1e4)
    agg.to_csv(out / "donor_celltype_expression.tsv", sep="\t", index=False)
    rows, loc = [], []
    for tech, d in agg.groupby("tech"):
        meas = [g for g in genes if measured.loc[g, tech]]
        sh = [g for g in shared if g in meas]
        for ct, dd in d.groupby("celltype"):
            dd = dd.set_index("donor")
            feats = {g: dd[f"lcpm_{g}"] for g in targets if g in meas}
            if len(sh) >= 5:
                z = dd[[f"lcpm_{g}" for g in sh]]; z = (z - z.mean()) / z.std(ddof=1)
                feats["SharedInjury_up_score"] = z.mean(1)
            for fn, x in feats.items():
                for a_, b_ in [("DKD", "Control"), ("DKD", "DM_noDKD"), ("DKD", "other_CKD"), ("DM_noDKD", "Control")]:
                    rows.append({"platform": tech, "celltype": ct, "feature": fn, "contrast": f"{a_}-{b_}",
                                 "n_shared_measured": len(sh), **compare(x, dd.group, a_, b_)})
                ok = dd.gfr_ord.notna()
                if ok.sum() >= 6:
                    rr = stats.spearmanr(x[ok], dd.gfr_ord[ok])
                    rows.append({"platform": tech, "celltype": ct, "feature": fn, "contrast": "spearman_vs_GFR_band",
                                 "n_a": int(ok.sum()), "rho": rr.correlation, "p": rr.pvalue})
        # localisation: iPT vs PT within donor (paired)
        w = d.pivot_table(index="donor", columns="celltype", values=[f"lcpm_{g}" for g in targets if g in meas])
        for g in [g for g in targets if g in meas]:
            if ("lcpm_" + g, "iPT") in w and ("lcpm_" + g, "PT") in w:
                p = w[["lcpm_" + g]].dropna()
                diff = p[("lcpm_" + g, "iPT")] - p[("lcpm_" + g, "PT")]
                loc.append({"platform": tech, "gene": g, "n_donors": len(diff), "median_iPT_minus_PT": diff.median(),
                            "frac_donors_iPT_higher": float((diff > 0).mean()),
                            "wilcoxon_p": stats.wilcoxon(diff).pvalue if len(diff) >= 5 else np.nan})
    T = pd.DataFrame(rows); T.to_csv(out / "group_tests.tsv", sep="\t", index=False)
    L = pd.DataFrame(loc); L.to_csv(out / "iPT_vs_PT_localisation.tsv", sep="\t", index=False)
    donors = agg.drop_duplicates(["donor", "tech"]).groupby(["tech", "group"]).donor.nunique()
    donors.to_csv(out / "donors_by_platform_group.tsv", sep="\t")
    pd.set_option("display.width", 250)
    print(measured.loc[genes].to_string()); print(donors)
    print(T[T.contrast != "spearman_vs_GFR_band"][["platform", "celltype", "feature", "contrast", "n_a", "n_b", "hedges_g", "welch_p"]].round(3).to_string())
    print(T[T.contrast == "spearman_vs_GFR_band"][["platform", "celltype", "feature", "n_a", "rho", "p"]].round(3).to_string())
    print(L.round(3).to_string())
    write_provenance(STAGE, [H5, ROOT / "data/raw/zenodo_dkd_spatial/Diagnosis.xlsx",
                             ROOT / "results/01_cohort_audit/donor_overlap.tsv"], sorted(out.glob("*.tsv")), seed,
                     hash_limit=1 << 30)


if __name__ == "__main__":
    main()
