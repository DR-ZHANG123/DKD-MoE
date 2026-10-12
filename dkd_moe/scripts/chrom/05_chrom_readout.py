"""05_chrom_readout: retrieval, held-out chromatin-side readout and external donor-level tests.

Inputs: per-model outputs of 04_align (<ckpt>/chrom/<feature>_f<f>_s<s>_out.npz) and the nucleus tables.
KPMP: each held-out participant is scored by the models of its own fold (mean over seeds); participant score =
mean over injured-PT nuclei; agreement with the RNA-side readout of the same nuclei and DKD vs HKD on the
chromatin-side score. External: every fold x seed model scores every nucleus (mean over models); donors as
units (injured-PT nuclei, >= external.min_cells), Hedges g with 95% CI.
Outputs: dkd_moe/results/chrom_readout/{retrieval_by_model.tsv, retrieval_summary.tsv, nucleus_eval.tsv,
         participant_scores.tsv, kpmp_agreement.tsv, kpmp_tests.tsv, seed_agreement.tsv,
         external_donor_scores.tsv, external_tests.tsv}
"""
from __future__ import annotations
import itertools, json, re, sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from chrom_common import cfg, result_dir, write_provenance, participant_table, tests  # noqa: E402

COLS = ["readout_te", "chrom_te", "rnaproj_te", "input_te"]
NAMES = {"readout_te": "rna_readout", "chrom_te": "chrom_score", "rnaproj_te": "rna_projection_head",
         "input_te": "chrom_input_ridge"}


def model_files(ck: Path, ft: str):
    for f in sorted(ck.glob(f"{ft}_f*_s*_out.npz")):
        m = re.match(rf"{ft}_f(\d+)_s(\d+)_out", f.stem)
        yield int(m.group(1)), int(m.group(2)), f


def spearman(a, b):
    ok = ~(np.isnan(a) | np.isnan(b))
    if ok.sum() < 4:
        return np.nan, np.nan, int(ok.sum())
    r = stats.spearmanr(a[ok], b[ok]); return r.statistic, r.pvalue, int(ok.sum())


def main():
    c = cfg(); out = result_dir("chrom_readout"); ck = Path(c["paths"]["ckpt_dir"]); al = result_dir("chrom_align")
    nuc = pd.read_csv(al / "nuclei.tsv.gz", sep="\t", index_col=0, dtype={"participant": str})
    inj = (nuc.pt_state == "injured_PT").values
    ret, neval, parts, ag, tst, sag, ext_scores, ext_tests = [], [], [], [], [], [], [], []
    for ft in c["align"]["features"]:
        per_seed = {}; ext = {}
        for f, s, fn in model_files(ck, ft):
            z = np.load(fn)
            r = json.loads(str(z["retrieval"]))
            for cond, d in r.items():
                ret.append({"feature": ft, "fold": f, "seed": s, "pairs": cond, **d})
            te = z["te"]
            for k in COLS:
                per_seed.setdefault((s, k), np.full(len(nuc), np.nan))[te] = z[k]
            q = inj[te]
            for k in ("chrom_te", "rnaproj_te", "input_te"):
                if q.sum() > 2:
                    neval.append({"feature": ft, "fold": f, "seed": s, "score": NAMES[k], "n_nuclei": int(q.sum()),
                                  "pearson_vs_rna_readout": float(np.corrcoef(z[k][q], z["readout_te"][q])[0, 1])})
            for key in z.files:
                if key.startswith("ext"):
                    ext.setdefault(key, []).append(z[key])
        if not per_seed:
            continue
        seeds = sorted({s for s, _ in per_seed})
        vals = {NAMES[k]: np.nanmean([per_seed[(s, k)] for s in seeds], 0) for k in COLS}
        for s in seeds:
            vals[f"chrom_score_s{s}"] = per_seed[(s, "chrom_te")]
        vals["log10_fragments"] = np.log10(nuc.n_fragments.values.astype(float))
        for k in ("tss_enrichment", "frip"):
            if k in nuc:
                vals[k] = nuc[k].values.astype(float)
        t = participant_table(nuc, vals, min_cells=c["rna"]["min_cells"]).assign(feature=ft)
        for k in ("log10_fragments", "tss_enrichment", "frip", "n_cells"):       # technical covariates of the score
            if k in t:
                for sc_ in ("chrom_score", "rna_readout"):
                    r, p, n = spearman(t[sc_].values, t[k].values.astype(float))
                    ag.append({"feature": ft, "participants": "all", "score": sc_, "versus": k, "n": n,
                               "spearman": r, "p": p})
        parts.append(t)
        sc = [f"chrom_score_s{s}" for s in seeds]
        rho = [t[a].corr(t[b], method="spearman") for a, b in itertools.combinations(sc, 2)]
        sag.append({"feature": ft, "n_participants": len(t), "mean_pairwise_spearman": float(np.mean(rho)) if rho else np.nan})
        for sub, q in (("all", t), ("DKD+HKD", t[t.group.isin(["DKD", "HKD"])])):
            for k in ("chrom_score", "rna_projection_head", "chrom_input_ridge"):
                r, p, n = spearman(q[k].values, q.rna_readout.values)
                ag.append({"feature": ft, "participants": sub, "score": k, "versus": "rna_readout", "n": n, "spearman": r, "p": p})
        tst.append(tests(t, c["kpmp_comparisons"], ["chrom_score", "rna_readout", "chrom_input_ridge"], {"space": ft}))
        # external cohorts
        for key, arrs in ext.items():
            kind, cohort = key.split("_", 1)
            o = pd.read_csv(al / f"ext_nuclei_{cohort}_{ft}.tsv.gz", sep="\t", index_col=0)
            o["donor"] = o.donor.astype(str); o["group"] = o.group.astype(str)
            score = np.mean(arrs, 0)
            name = "chrom_score" if kind == "ext" else "chrom_input_ridge"
            d = participant_table(o.rename(columns={"donor": "participant"}), {name: score},
                                  min_cells=c["external"]["min_cells"]).assign(cohort=cohort, feature=ft, n_models=len(arrs))
            ext_scores.append(d)
            ext_tests.append(tests(d, c["external"]["comparisons"][cohort], [name], {"cohort": cohort, "space": ft}))
    R = pd.DataFrame(ret); R.to_csv(out / "retrieval_by_model.tsv", sep="\t", index=False)
    met = [m for m in R.columns if m.startswith("top") or m == "foscttm"]
    RS = (R.groupby(["feature", "pairs", "seed"])[met].mean().groupby(["feature", "pairs"])
          .agg(["mean", "std", "min", "max"]))
    RS.columns = ["_".join(x) for x in RS.columns]; RS = RS.reset_index()
    RS.to_csv(out / "retrieval_summary.tsv", sep="\t", index=False)
    pd.DataFrame(neval).to_csv(out / "nucleus_eval.tsv", sep="\t", index=False)
    T = pd.concat(parts, ignore_index=True); T.to_csv(out / "participant_scores.tsv", sep="\t", index=False)
    pd.DataFrame(ag).to_csv(out / "kpmp_agreement.tsv", sep="\t", index=False)
    KT = pd.concat(tst, ignore_index=True); KT.to_csv(out / "kpmp_tests.tsv", sep="\t", index=False)
    pd.DataFrame(sag).to_csv(out / "seed_agreement.tsv", sep="\t", index=False)
    ES = pd.concat(ext_scores, ignore_index=True) if ext_scores else pd.DataFrame()
    ET = pd.concat(ext_tests, ignore_index=True) if ext_tests else pd.DataFrame()
    ES.to_csv(out / "external_donor_scores.tsv", sep="\t", index=False); ET.to_csv(out / "external_tests.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(RS.round(3).to_string()); print(pd.DataFrame(neval).groupby(["feature", "score"]).pearson_vs_rna_readout.mean().round(3))
    print(pd.DataFrame(ag).round(3).to_string()); print(T.groupby(["feature", "group"]).size())
    print(KT[["space", "comparison", "feature", "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon", "auc"]].round(3).to_string())
    if len(ET):
        print(ET[["cohort", "space", "feature", "comparison", "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon"]].round(3).to_string())
    files = ["retrieval_by_model.tsv", "retrieval_summary.tsv", "nucleus_eval.tsv", "participant_scores.tsv",
             "kpmp_agreement.tsv", "kpmp_tests.tsv", "seed_agreement.tsv", "external_donor_scores.tsv", "external_tests.tsv"]
    write_provenance("chrom_readout", [al / "nuclei.tsv.gz"] + sorted(ck.glob("*_out.npz")), [out / f for f in files],
                     c["seed"], c, extra={"n_models": int(len(R) // 2)})


if __name__ == "__main__":
    main()
