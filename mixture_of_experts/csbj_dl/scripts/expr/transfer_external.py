"""External tests (test-only): the KPMP-selected transfer signature in all external cohorts, shown next to
the consensus IG signature; random-effects summaries; meta-analysis and sign test of the identifiable readout.
Outputs: results/expr_transfer/{external_signature_tests.tsv, external_meta_analysis.tsv, readout_meta.tsv, scores_*.tsv}
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import load_cfg, result_dir, write_provenance  # noqa: E402
import external_signature as es  # noqa: E402


def meta_rows(t: pd.DataFrame, cfg, label: str) -> list[dict]:
    t = t.dropna(subset=["g"]).copy()
    ct = t["celltype"] if "celltype" in t else pd.Series("", index=t.index)
    t["label"] = t.cohort + ":" + t.platform.astype(str) + ct.fillna("").map(lambda x: f":{x}" if x else "")
    sel = ((t.platform != "sc") & (t.platform != "sn")) | (t.cohort == "GSE195460")
    zp = cfg["readout_stability"]["external"]["meta_zenodo_platform"]
    sel &= ~((t.cohort == "Zenodo19868428") & ((ct != "iPT") | (t.platform != zp)))
    out = []
    for name, comps in [("DKD_vs_disease_control", ["DKD_vs_HTN", "DKD_vs_HKD", "DKD_vs_other_CKD"]),
                        ("DKD_vs_healthy_control", ["DKD_vs_LD", "DKD_vs_Control"])]:
        d = t[sel & t.comparison.isin(comps)]
        if len(d):
            out.append({"signature": label, "summary": name, **es.dersimonian_laird(d),
                        "n_positive": int((d.g > 0).sum()),
                        "sign_test_p": stats.binomtest(int((d.g > 0).sum()), len(d), 0.5).pvalue})
    return out


def main():
    cfg = load_cfg("expr"); out = result_dir("expr_transfer"); es.OUT = "expr_transfer"
    sel = json.loads((out / "selection.json").read_text())
    sig = pd.read_csv(out / "transfer_signature.tsv", sep="\t")
    up, down = sig[sig.direction == "up"].gene.tolist(), sig[sig.direction == "down"].gene.tolist()
    nmin, mc = cfg["readout_stability"]["signature"]["min_genes_per_direction"], cfg["prep"]["min_cells_per_participant"]
    T = pd.DataFrame(es.ercb(cfg, up, down, nmin) + es.zenodo(cfg, up, down, nmin, mc) + es.sc_cohorts(cfg, up, down, nmin, mc))
    T.insert(0, "signature", f"transfer_{sel['chosen']}")
    R2 = pd.read_csv(result_dir("expr_external") / "signature_tests.tsv", sep="\t"); R2.insert(0, "signature", "consensus_ig_healthy_baseline")
    A = pd.concat([T, R2], ignore_index=True); A.to_csv(out / "external_signature_tests.tsv", sep="\t", index=False)
    M = pd.DataFrame(meta_rows(T, cfg, T.signature.iloc[0]) + meta_rows(R2, cfg, "consensus_ig_healthy_baseline"))
    M.to_csv(out / "external_meta_analysis.tsv", sep="\t", index=False)
    # identifiable readout across external single-cell cohorts
    RT = pd.read_csv(result_dir("expr_readout") / "donor_level_tests.tsv", sep="\t")
    rows = []
    for c, tech, comp in cfg["transfer"]["readout_meta"]["cohorts"]:
        r = RT[(RT.cohort == c) & (RT.tech == tech) & (RT.comparison == comp)].iloc[0]
        rows.append({"cohort": c, "tech": tech, "comparison": comp, "n1": r.n1, "n2": r.n2, "g": r.g,
                     "ci_low": r.ci_low, "ci_high": r.ci_high, "p_wilcoxon": r.p_wilcoxon,
                     "label": f"{c}:{comp}"})
    d = pd.DataFrame(rows)
    dl = es.dersimonian_laird(d)
    k_pos = int((d.g > 0).sum())
    summ = pd.DataFrame([{"cohort": "random_effects_summary (mixed contrasts: DKD vs HKD; DKD vs control)",
                          "g": dl["g_random"], "ci_low": dl["ci_low"], "ci_high": dl["ci_high"], "p_wilcoxon": dl["p"],
                          "tau2": dl["tau2"], "I2": dl["I2"], "n_positive": k_pos,
                          "sign_test_p": stats.binomtest(k_pos, len(d), 0.5, alternative="greater").pvalue}])
    pd.concat([d, summ], ignore_index=True).to_csv(out / "readout_meta.tsv", sep="\t", index=False)
    pd.set_option("display.width", 250)
    cols = [c for c in ["signature", "cohort", "platform", "celltype", "comparison", "n_up_measured", "n_down_measured",
                        "n1", "n2", "g", "ci_low", "ci_high", "p_wilcoxon"] if c in A.columns]
    print(T[cols[1:]].round(3).to_string()); print(M.round(3).to_string()); print(pd.concat([d, summ]).round(3).to_string())
    write_provenance("expr_transfer", [out / "transfer_signature.tsv"], [out / "external_signature_tests.tsv",
                     out / "external_meta_analysis.tsv", out / "readout_meta.tsv"], cfg["seed"], cfg, tag="external")


if __name__ == "__main__":
    main()
