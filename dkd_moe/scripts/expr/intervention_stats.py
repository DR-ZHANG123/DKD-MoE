"""Participant-level summaries for the expert intervention: signature / gene-group scores, Wilcoxon tests,
bootstrap intervals, specificity contrasts and genome-wide concordance with the readout-gene correlation."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats


def summarise(v: pd.Series, n_boot: int, rng) -> dict:
    """Mean with participant-bootstrap 95% CI, median, IQR and Wilcoxon signed-rank test against 0."""
    v = v.dropna().astype(float)
    bm = rng.choice(v.values, (n_boot, len(v)), replace=True).mean(1)
    w = stats.wilcoxon(v.values) if (v != 0).any() else None
    return {"n_participants": len(v), "mean": v.mean(), "ci_low": np.quantile(bm, 0.025), "ci_high": np.quantile(bm, 0.975),
            "median": v.median(), "q25": v.quantile(0.25), "q75": v.quantile(0.75), "n_positive": int((v > 0).sum()),
            "wilcoxon_W": float(w.statistic) if w else np.nan, "p_wilcoxon": float(w.pvalue) if w else np.nan}


def gene_set_scores(delta: pd.DataFrame, sd: pd.Series, up: list, down: list, groups: dict) -> pd.DataFrame:
    """Per-participant scores for one seed. Columns: measure, units, participant, value.
    'log' = change in natural-log decoded expression; 'sd' = change divided by the across-cell SD of the
    unperturbed decoded log expression of that gene (same cells, same model)."""
    rows = []
    for units, D in (("log", delta), ("sd", delta / sd.reindex(delta.columns).values)):
        u = [g for g in up if g in D.columns]; d = [g for g in down if g in D.columns]
        meas = {"signature_score": D[u].mean(1) - D[d].mean(1), "signature_up_mean": D[u].mean(1),
                "signature_down_mean": D[d].mean(1)}
        for k, gl in groups.items():
            gl = [g for g in gl if g in D.columns]
            meas[f"group:{k}"] = D[gl].mean(1)
            for g in gl:
                meas[f"gene:{g}"] = D[g]
        for m, s in meas.items():
            rows.append(pd.DataFrame({"measure": m, "units": units, "participant": s.index, "value": s.values}))
    return pd.concat(rows, ignore_index=True)


def summary_table(P: pd.DataFrame, n_boot: int, rng) -> pd.DataFrame:
    """P: intervention, seed, measure, units, participant, value. Seeds pooled by averaging per participant;
    seed-level agreement = number of seeds whose participant mean has the sign of the pooled mean."""
    rows = []
    for (iv, m, u), d in P.groupby(["intervention", "measure", "units"], sort=False):
        pooled = d.groupby("participant").value.mean()
        r = summarise(pooled, n_boot, rng)
        per_seed = d.groupby("seed").value.mean()
        rows.append({"intervention": iv, "measure": m, "units": u, "n_seeds": d.seed.nunique(), **r,
                     "seeds_same_sign": int((np.sign(per_seed) == np.sign(r["mean"])).sum()),
                     "seed_mean_min": per_seed.min(), "seed_mean_max": per_seed.max()})
    return pd.DataFrame(rows)


def specificity_table(P: pd.DataFrame, pairs: list, n_boot: int, rng) -> pd.DataFrame:
    """Main minus control signature-score change per participant (same cells), seeds averaged; for the
    direction of the main effect the difference is signed so that positive = main larger in its own direction."""
    rows = []
    d = P[P.measure == "signature_score"]
    for main, ctrl in pairs:
        for u in ("log", "sd"):
            a = d[(d.intervention == main) & (d.units == u)].groupby("participant").value.mean()
            b = d[(d.intervention == ctrl) & (d.units == u)].groupby("participant").value.mean()
            sgn = np.sign(a.mean())
            diff = (sgn * (a - b.reindex(a.index))).dropna()
            r = summarise(diff, n_boot, rng)
            rows.append({"main": main, "control": ctrl, "units": u, "main_mean": a.mean(), "control_mean": b.mean(),
                         "ratio_control_to_main": b.mean() / a.mean(), **{f"diff_{k}": v for k, v in r.items()}})
    return pd.DataFrame(rows)


def auc_up_down(x: pd.Series, up: list, down: list) -> float:
    """P(effect of a random up gene > effect of a random down gene)."""
    u = x.reindex(up).dropna().values; d = x.reindex(down).dropna().values
    return float(stats.mannwhitneyu(u, d).statistic / (len(u) * len(d)))


def concordance_rows(name: str, eff: pd.Series, eff_seed: dict, r_expr: pd.Series, r_all: pd.Series,
                     up: list, down: list, sign: int) -> list[dict]:
    """Genome-wide concordance of a gene-level effect (participants and seeds averaged; multiplied by
    `sign` so that positive = toward DKD) with the readout-gene correlation and the signature weights."""
    e = sign * eff
    w = pd.Series(0.0, index=e.index); w[w.index.isin(up)] = 1.0; w[w.index.isin(down)] = -1.0
    rows = []
    for label, ref, genes in (("readout_gene_correlation_expressed", r_expr, r_expr.dropna().index),
                              ("readout_gene_correlation_all_model_genes", r_all, r_all.dropna().index),
                              ("signature_weight_+1_0_-1", w, w.index)):
        g = [x for x in genes if x in e.index]
        rho, p = stats.spearmanr(e[g], ref[g])
        ps = [stats.spearmanr(sign * es[g], ref[g])[0] for es in eff_seed.values()]
        rows.append({"effect": name, "reference": label, "n_genes": len(g), "statistic_type": "spearman_rho", "statistic": rho, "p_gene_level": p,
                     "seed_min": min(ps), "seed_max": max(ps), "n_seeds_concordant": int(sum(x > 0 for x in ps))})
    uu = [x for x in up if x in e.index]; dd = [x for x in down if x in e.index]
    rows.append({"effect": name, "reference": "signature_up_vs_down_auc", "n_genes": len(uu) + len(dd),
                 "statistic_type": "auc", "statistic": auc_up_down(e, uu, dd), "p_gene_level": stats.mannwhitneyu(e[uu], e[dd]).pvalue,
                 "seed_min": min(auc_up_down(sign * es, uu, dd) for es in eff_seed.values()),
                 "seed_max": max(auc_up_down(sign * es, uu, dd) for es in eff_seed.values()),
                 "n_seeds_concordant": int(sum(auc_up_down(sign * es, uu, dd) > 0.5 for es in eff_seed.values()))})
    conc = int((e[uu] > 0).sum() + (e[dd] < 0).sum())
    rows.append({"effect": name, "reference": "signature_sign_concordance_fraction", "n_genes": len(uu) + len(dd),
                 "statistic_type": "fraction", "statistic": conc / (len(uu) + len(dd)),
                 "p_gene_level": stats.binomtest(conc, len(uu) + len(dd), 0.5).pvalue,
                 "n_up_increased": int((e[uu] > 0).sum()), "n_down_decreased": int((e[dd] < 0).sum())})
    return rows
