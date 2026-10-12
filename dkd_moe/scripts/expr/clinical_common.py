"""Participant table for the KPMP held-out readouts: diagnosis group, PT composition, technology set,
recruitment site, banded age / sex / eGFR / albuminuria / proteinuria (one row per participant)."""
from __future__ import annotations
import re, sys
from pathlib import Path
import anndata as ad
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "common"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import CDL, result_dir  # noqa: E402
from clinical_routing import band_order  # noqa: E402

BANDS = {"albuminuria_band": "alb_ord", "proteinuria_band": "prot_ord", "egfr_band": "egfr_ord"}


def lower_bound(v) -> float:
    m = re.search(r"(\d+)", str(v))
    return float(m.group(1)) if m else np.nan


def participant_table(cfg, min_site: int = 3) -> pd.DataFrame:
    bulk = Path(cfg["paths"]["bulk_dir"])
    obs = ad.read_h5ad(bulk / "kpmp_pt_hvg.h5ad", backed="r").obs
    first = obs.groupby("participant", observed=True)[["group", "sex", "age_band", "tissue_source", *BANDS]].first().astype(str)
    tab = pd.read_csv(result_dir("expr_prep") / "kpmp_pt_cells_by_participant.tsv", sep="\t", dtype={"participant": str})
    mc = cfg["prep"]["min_cells_per_participant"]
    tab = tab[tab.injured_PT >= mc]                       # technologies that contribute to the readout
    tab["frac"] = tab.injured_PT / (tab.injured_PT + tab.healthy_PT)
    agg = tab.groupby("participant").agg(injured_fraction=("frac", "mean"), n_cells=("injured_PT", "sum"),
                                         tech_set=("tech", lambda t: "both" if t.nunique() > 1 else t.iloc[0]))
    P = first.join(agg, how="inner")
    P["log_n_cells"] = np.log(P.n_cells)
    P["age_ord"] = P.age_band.map(lower_bound)
    P["male"] = P.sex.map({"Male": 1.0, "Female": 0.0})
    for b, o in BANDS.items():
        P[o] = P[b].map(band_order)
    site = P.index.to_series().str.extract(r"^(\d+)-", expand=False).fillna("other")
    P["site"] = site
    P["site_c"] = np.where(site.map(site.value_counts()) >= min_site, site, "other")
    P.index.name = "participant"
    return P


def readout_table() -> pd.DataFrame:
    """Held-out KPMP participant readouts (pooled over technologies, seed means): final model, supervised model
    without experts (latent-mean condition head), expert contributions M1 / M3, supervision variants."""
    S = pd.read_csv(result_dir("expr_readout") / "participant_scores.tsv", sep="\t", dtype={"participant": str})
    S = S[(S.cohort == "KPMP_heldout") & (S.tech == "pooled")].set_index("participant")
    out = pd.DataFrame({"moe": S.logodds})
    B = pd.read_csv(CDL / "results/bench_supervised_single/donor_scores.tsv", sep="\t", dtype={"participant": str})
    B = B[(B.seed.astype(str) == "mean") & (B.dataset == "KPMP") & (B.method == "supervised_single_z/condition_head")]
    out["no_experts"] = B.set_index("participant").score
    # held-out contributions of the HKD- and DKD-weighted experts (heldout_experts.py) and the readouts of the
    # injured-only and participant-level supervision (sup_compare.py)
    C = pd.read_csv(result_dir("expr_heldout_experts") / "contributions.tsv", sep="\t", dtype={"participant": str})
    out = out.join(C.set_index("participant")[["M1", "M3"]])
    V = pd.read_csv(result_dir("expr_supervision") / "heldout_readouts.tsv", sep="\t", dtype={"participant": str})
    out = out.join(V.set_index("participant"))
    # expert identities defined from training participants only (trainonly_experts.py) and the readout standardized
    # with training-participant statistics of each part (nested_calibration.py)
    Tt = pd.read_csv(result_dir("expr_trainonly_experts") / "contributions.tsv", sep="\t", dtype={"participant": str})
    out = out.join(Tt.set_index("participant")[["M1", "M3"]].rename(columns=lambda c: c + "_train_only"))
    N = pd.read_csv(result_dir("expr_nonmoe_components") / "contributions.tsv", sep="\t", dtype={"participant": str})
    out = out.join(N.set_index("participant")[["nonmoe_Lpos", "nonmoe_Lneg"]])
    sc = pd.read_csv(result_dir("expr_nested_cv") / "train_scale.tsv", sep="\t")
    sc = sc[sc.pipeline == "reported"].set_index("fold")
    fold = pd.read_csv(CDL / "config/splits.tsv", sep="\t", dtype={"participant": str}).set_index("participant").fold
    f = fold.reindex(out.index)
    out["moe_train_z"] = (out.moe - f.map(sc.train_mean)) / f.map(sc.train_sd)
    return out
