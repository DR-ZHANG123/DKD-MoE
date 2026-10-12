"""09_sig_composition: does the within-section agreement of the image-side and measured signature scores
remain after accounting for spot composition, histology intensity and injury? (no retraining)

Inputs: figures/source_data/align_sig_spot_scores.tsv.gz (full-resolution: KPMP + GSE211785) and
align_lowres_sig_spot_scores.tsv.gz (matched resolution: + GSE339455), written by 07_signature.py; spot-level
expression-derived injury proxies from stages 10a/10b (results/10{a,b}_visium_*/spot_scores.tsv.gz):
injury_index (injured-PT minus healthy-PT marker score) and prop_Stroma (deconvolved stromal fraction,
interstitial/fibrosis proxy); available for GSE211785 and part of the KPMP sections, not for GSE339455.
Per section (PT-rich spots, >= min_pt_rich_spots; image-side KPMP scores are out-of-fold):
  partial Spearman image_sig ~ measured_sig | covariates (ranks residualised on ranked covariates) for
  nested covariate sets; Spearman of measured_sig (and image_sig) with pt_score and injury proxies, on
  PT-rich spots and on all spots. Sections are averaged per participant; participants summarised per cohort.
Outputs: dkd_moe/results/align_composition_adjusted/{section_partial.tsv, participant_partial.tsv,
         summary.tsv, composition_links.tsv, composition_summary.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

OUT = "align_composition_adjusted"
MORPH = ["nuclei_per_1e4px_tissue", "hema_mean", "eosin_mean", "white_frac"]
INJ = ["injury_index", "prop_Stroma"]
COVSETS = {"unadjusted": [], "pt": ["pt_score"], "pt_morph": ["pt_score", *MORPH],
           "pt_morph_all": ["pt_score", *MORPH, "hema_frac", "eosin_frac"],
           "pt_morph_injury": ["pt_score", *MORPH, *INJ]}
LINKS = ["pt_score", *MORPH, "hema_frac", "eosin_frac", *INJ]


def partial_rho(x, y, Z):
    rk = lambda v: stats.rankdata(v)
    X = np.column_stack([np.ones(len(x))] + [rk(Z[:, j]) for j in range(Z.shape[1])])
    res = lambda v: v - X @ np.linalg.lstsq(X, v, rcond=None)[0]
    rx, ry = res(rk(x)), res(rk(y))
    if rx.std() == 0 or ry.std() == 0:
        return np.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def injury_table() -> pd.DataFrame:
    k = pd.read_csv(C.ROOT / "results/10b_visium_kpmp/spot_scores.tsv.gz", sep="\t", dtype={"participant": str},
                    usecols=["barcode", "participant", "sample", *INJ])
    k["section"] = k.participant + "__" + k["sample"]
    g = pd.read_csv(C.ROOT / "results/10a_visium_gse211785/spot_scores.tsv.gz", sep="\t", usecols=["barcode", "section", *INJ])
    return pd.concat([k[["section", "barcode", *INJ]], g[["section", "barcode", *INJ]]]).rename(columns={"barcode": "spot_id"})


def section_rows(S: pd.DataFrame, track: str, mins: int):
    P, L = [], []
    for (sec, part), g0 in S.groupby(["section", "participant"]):
        for spots, g in (("pt_rich", g0[g0.pt_rich]), ("all", g0)):
            if len(g) < mins:
                continue
            base = {"track": track, "cohort": g.cohort.iloc[0], "section": sec, "participant": part,
                    "group": g.group.iloc[0], "spots": spots}
            for cs, cov in COVSETS.items():
                d = g.dropna(subset=["image_sig", "measured_sig", *cov])
                ok = len(d) >= max(mins, len(cov) + 10)
                P.append({**base, "covariates": cs, "n_spots": len(d),
                          "rho": partial_rho(d.image_sig.values, d.measured_sig.values, d[cov].values.astype(float))
                          if ok else np.nan})
                if cs == "pt_morph_injury" and ok:                  # same spots, no adjustment (fair reference)
                    P.append({**base, "covariates": "unadjusted_injury_subset", "n_spots": len(d),
                              "rho": stats.spearmanr(d.image_sig, d.measured_sig).statistic})
            for score in ("measured_sig", "image_sig"):
                for v in LINKS:
                    d = g[[score, v]].dropna()
                    L.append({**base, "score": score, "versus": v, "n_spots": len(d),
                              "rho": stats.spearmanr(d[score], d[v]).statistic if len(d) >= mins and d[v].std() > 0 else np.nan})
    return pd.DataFrame(P), pd.DataFrame(L)


def summarise(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 3:
        return {"n_participants": len(x), "mean": x.mean() if len(x) else np.nan}
    return {"n_participants": len(x), "mean": x.mean(), "median": x.median(), "q25": x.quantile(.25),
            "q75": x.quantile(.75), "n_positive": int((x > 0).sum()), "frac_positive": (x > 0).mean(),
            "wilcoxon_p": stats.wilcoxon(x).pvalue, "sign_test_p": stats.binomtest(int((x > 0).sum()), len(x)).pvalue}


def main():
    cfg = C.load_cfg(); C.set_seed(cfg["seed"]); mins = cfg["interpret"]["min_pt_rich_spots"]
    out = C.PKG / "results" / OUT; out.mkdir(parents=True, exist_ok=True)
    src = C.PKG / "figures/source_data"
    inputs = {"full_resolution": src / "align_sig_spot_scores.tsv.gz", "matched_resolution": src / "align_lowres_sig_spot_scores.tsv.gz"}
    inj = injury_table()
    P, L = [], []
    for track, f in inputs.items():
        S = pd.read_csv(f, sep="\t", dtype={"participant": str}).merge(inj, on=["section", "spot_id"], how="left")
        p, l = section_rows(S, track, mins); P.append(p); L.append(l)
    P, L = pd.concat(P, ignore_index=True), pd.concat(L, ignore_index=True)
    keys = ["track", "cohort", "participant", "group", "spots"]
    PP = P.groupby(keys + ["covariates"]).rho.mean().reset_index()
    LP = L.groupby(keys + ["score", "versus"]).rho.mean().reset_index()
    SU = []
    for (tr, co, sp, cs), d in PP.groupby(["track", "cohort", "spots", "covariates"]):
        r = {"track": tr, "cohort": co, "spots": sp, "covariates": cs, **summarise(d.rho)}
        ref = "unadjusted_injury_subset" if cs == "pt_morph_injury" else "unadjusted"
        if cs not in ("unadjusted", "unadjusted_injury_subset"):    # paired change versus the unadjusted value
            u = PP[(PP.track == tr) & (PP.cohort == co) & (PP.spots == sp) & (PP.covariates == ref)]
            m = d.merge(u, on="participant", suffixes=("", "_u")).dropna(subset=["rho", "rho_u"])
            if len(m) >= 3:
                dd = m.rho - m.rho_u
                r.update({"reference": ref, "n_paired": len(m), "mean_unadjusted_paired": m.rho_u.mean(),
                          "mean_change": dd.mean(), "retained_frac": m.rho.mean() / m.rho_u.mean(),
                          "paired_wilcoxon_p": stats.wilcoxon(dd).pvalue if (dd != 0).any() else np.nan})
        SU.append(r)
    SU = pd.DataFrame(SU)
    LS = pd.DataFrame([{"track": tr, "cohort": co, "spots": sp, "score": sc, "versus": v, **summarise(d.rho)}
                       for (tr, co, sp, sc, v), d in LP.groupby(["track", "cohort", "spots", "score", "versus"])])
    files = {"section_partial.tsv": P, "participant_partial.tsv": PP, "summary.tsv": SU,
             "composition_links.tsv": LP, "composition_summary.tsv": LS}
    for k, v in files.items():
        v.to_csv(out / k, sep="\t", index=False)
    pd.set_option("display.width", 250); pd.set_option("display.max_rows", 500)
    print(SU.round(4).to_string())
    print(LS[LS.versus.isin(["pt_score", *INJ])].round(4).to_string())
    C.repro.write_provenance(f"../dkd_moe/results/{OUT}", [*inputs.values(), C.ROOT / "results/10b_visium_kpmp/spot_scores.tsv.gz",
                             C.ROOT / "results/10a_visium_gse211785/spot_scores.tsv.gz"], [out / k for k in files], cfg["seed"],
                             extra={"covariate_sets": COVSETS, "align_config_sha": C.repro.config_sha(C.CFG_PATH),
                                    "min_spots": mins})


if __name__ == "__main__":
    main()
