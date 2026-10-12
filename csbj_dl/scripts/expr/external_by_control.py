"""External cohorts: the DKD-vs-HKD held-out readout (final models) summarised separately by control type.

DKD vs disease control: only GSE211785 HKD is available -> single estimate, no pooling.
DKD vs healthy control: GSE211785 control and GSE195460 control, DerSimonian-Laird random effects on Hedges g
(one estimate per cohort, technologies pooled within cohort). Direction of every external contrast is listed,
including HKD vs control, with exact one- and two-sided binomial sign tests where more than one estimate exists.
Note: in GSE211785 the same DKD donors enter both the disease- and the healthy-control contrast.
Output: results/expr_external_by_control/{readout_by_control.tsv, readout_directions.tsv}
"""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
from external_signature import dersimonian_laird  # noqa: E402

COLS = ["n1", "n2", "mean1", "mean2", "g", "ci_low", "ci_high", "p_wilcoxon"]


def pick(T, c, tech, comp):
    r = T[(T.cohort == c) & (T.tech == tech) & (T.comparison == comp)]
    assert len(r) == 1, (c, tech, comp)
    return {"cohort": c, "tech": tech, "comparison": comp, **r.iloc[0][COLS].to_dict(), "label": f"{c}:{comp}"}


def main():
    cfg = load_cfg("expr"); em = cfg["patient_level_models"]["external_meta"]
    out = result_dir("expr_external_by_control"); src = result_dir("expr_readout") / "donor_level_tests.tsv"
    T = pd.read_csv(src, sep="\t")
    rows = []
    for summary, key in (("DKD_vs_disease_control", "disease_control"), ("DKD_vs_healthy_control", "healthy_control")):
        d = pd.DataFrame([pick(T, *x) for x in em[key]])
        for _, r in d.iterrows():
            rows.append({"summary": summary, "row": "cohort", **r.to_dict()})
        k_pos = int((d.g > 0).sum())
        s = {"summary": summary, "k": len(d), "n_positive": k_pos, "label": "; ".join(d.label),
             "n1": d.n1.sum(), "n2": d.n2.sum()}
        if len(d) == 1:
            r = d.iloc[0]
            s.update({"row": "single_estimate_no_pooling", "g": r.g, "ci_low": r.ci_low, "ci_high": r.ci_high,
                      "p_wilcoxon": r.p_wilcoxon})
        else:
            dl = dersimonian_laird(d)
            s.update({"row": "random_effects_DL", "g": dl["g_random"], "ci_low": dl["ci_low"], "ci_high": dl["ci_high"],
                      "p_random_effects_z": dl["p"], "tau2": dl["tau2"], "Q": dl["Q"], "I2": dl["I2"],
                      "sign_test_p_one_sided": stats.binomtest(k_pos, len(d), 0.5, alternative="greater").pvalue,
                      "sign_test_p_two_sided": stats.binomtest(k_pos, len(d), 0.5).pvalue})
        rows.append(s)
    R = pd.DataFrame(rows); R.to_csv(out / "readout_by_control.tsv", sep="\t", index=False)
    ext = T[T.cohort.isin(["gse211785", "gse195460"]) & (T.tech == "pooled")].copy()
    ext["direction"] = ext.g.map(lambda g: "higher_in_first" if g > 0 else ("lower_in_first" if g < 0 else "undefined"))
    kp = T[(T.cohort == "KPMP_heldout") & (T.tech == "pooled")][["comparison", "g"]].rename(columns={"g": "g_kpmp_heldout"})
    kp["comparison"] = kp.comparison.str.replace("Reference", "Control")
    ext = ext.merge(kp, on="comparison", how="left")
    ext["same_sign_as_kpmp"] = (ext.g * ext.g_kpmp_heldout) > 0
    ext.to_csv(out / "readout_directions.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    print(R.round(4).to_string()); print(ext.round(4).to_string())
    write_provenance("expr_external_by_control", [src], [out / "readout_by_control.tsv", out / "readout_directions.tsv"],
                     cfg["seed"], cfg, extra={"note": "KPMP Reference is the healthy-tissue comparator matched to external 'Control' "
                                              "in readout_directions.tsv only for the sign comparison"})


if __name__ == "__main__":
    main()
