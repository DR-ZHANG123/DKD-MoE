"""final_tables: supplementary-table inputs restricted to the models and analyses reported in the article.

Reads result tables (no recomputation) and keeps the final mixture-of-experts model, the comparison methods,
the four component variants of the final model (one readout each), the transferable signature, and the
within-section histology models with the evaluation used for each cohort; labels are written for readers.
Outputs dkd_moe/results/dl_tables_final/{stability, signature_crossfit, signature_external_tests,
signature_external_meta, comparison_summary, comparison_paired, comparison_participant_lists, technology_mixing,
histology_retrieval,
histology_retrieval_vs_shuffled}.tsv + PROVENANCE.json."""
from __future__ import annotations
import sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import CDL, repro  # noqa: E402

R = CDL / "results"; OUT = R / "dl_tables_final"
METHODS = {"MoE final model readout": "Mixture of experts: readout",
           "pseudobulk/DE_set": "Participant-level DE: gene set",
           "scvi/DE_set": "scVI: DE gene set", "scvi/latent_LR": "scVI: latent classifier",
           "scdisinfact/CKG_set": "scDisInFact: key genes", "scdisinfact/zd_LR": "scDisInFact: condition latent",
           "scdisinfact/latent_LR": "scDisInFact: shared latent",
           "cvae/decoded_set": "Conditional VAE: decoded genes",
           "cvae/counterfactual_LLR": "Conditional VAE: counterfactual",
           "cvae/latent_LR": "Conditional VAE: latent classifier",
           "moe_warm_K1/latent_LR": "Variant: single expert (latent classifier)",
           "moe_warm_dense/condition_head": "Variant: dense routing",
           "moe_warm_no_adv/condition_head": "Variant: no technology discriminator",
           "moe_warm_no_cond/condition_head": "Variant: no condition classifier",
           "supervised_single_z/condition_head": "Supervised single expert: condition classifier on the latent mean",
           "supervised_single_h/condition_head": "Supervised single expert: condition classifier on the encoder output"}


def rd(rel):
    return pd.read_csv(R / rel, sep="\t")


def main():
    OUT.mkdir(parents=True, exist_ok=True); outs = []

    def w(df, name):
        f = OUT / f"{name}.tsv"; df.to_csv(f, sep="\t", index=False); outs.append(f)

    st = rd("expr_stability/before_after.tsv").set_index("run")
    st = st.loc[["after_seeds1-5", "final_after"]].rename(
        index={"after_seeds1-5": "five partition models per part, held-out KPMP participants",
               "final_after": "five final models, all KPMP cells"}).reset_index().rename(columns={"run": "models"})
    w(st.dropna(axis=1, how="all"), "stability")

    t = rd("expr_transfer/internal_signature_tests.tsv")
    t = t[t.construction == "readout_assoc"].drop(columns="construction")
    t["setting"] = t.setting.str.replace(r"\s*\(.*\)", "", regex=True)
    w(t, "signature_crossfit")
    e = rd("expr_transfer/external_signature_tests.tsv"); e = e[e.signature == "transfer_readout_assoc"].drop(columns="signature")
    w(e, "signature_external_tests")
    m = rd("expr_transfer/external_meta_analysis.tsv"); m = m[m.signature == "transfer_readout_assoc"].drop(columns="signature")
    w(m, "signature_external_meta")

    s = pd.concat([rd("bench_summary/summary_table.tsv"), rd("bench_supervised_single/summary.tsv")])
    s = s[s.method.isin(METHODS)].copy()
    s["method"] = s.method.map(METHODS); w(s, "comparison_summary")
    p = pd.concat([rd("bench_summary/paired_vs_moe_final.tsv"), rd("bench_supervised_single/paired_vs_moe_final.tsv")])
    p = p[p.method.isin(METHODS)].drop_duplicates("method").copy()
    p["method"] = p.method.map(METHODS); w(p, "comparison_paired")
    pl = rd("bench_summary/participant_lists.tsv"); pl = pl[pl.method.isin(METHODS)].copy()
    pl["method"] = pl.method.map(METHODS); w(pl, "comparison_participant_lists")
    mx = rd("bench_supervised_single/technology_mixing.tsv"); w(mx, "technology_mixing")
    w(rd("bench_supervised_single/seed_agreement.tsv"), "readout_reproducibility")
    w(rd("expr_routing_faithfulness/readout_comparison.tsv"), "routing_faithfulness")
    w(rd("expr_routing_faithfulness/meta_topk_share.tsv"), "routing_faithfulness_experts")

    # histology: within-section models; KPMP held-out, GSE211785 with cohort-level input standardisation
    keep = {"kpmp_heldout": "KPMP held-out", "gse211785_external_calibrated": "GSE211785"}
    r = rd("align_train/retrieval_summary.tsv")
    r = r[r.model.isin(["aligned_within_section", "shuffled_within_section"]) & r.eval_set.isin(keep)].copy()
    r["eval_set"] = r.eval_set.map(keep)
    r["model"] = r.model.map({"aligned_within_section": "aligned", "shuffled_within_section": "shuffled pairs"})
    w(r, "histology_retrieval")
    b = rd("align_train/retrieval_vs_baseline.tsv")
    b = b[(b.model == "aligned_within_section") & b.eval_set.isin(keep)].copy()
    b["eval_set"] = b.eval_set.map(keep); b["model"] = "aligned"; b["baseline"] = "shuffled pairs"
    w(b, "histology_retrieval_vs_shuffled")

    repro.write_provenance("../dkd_moe/results/dl_tables_final", [Path(__file__)], outs, seed=None)
    print("\n".join(str(o.relative_to(CDL)) for o in outs))


if __name__ == "__main__":
    main()
