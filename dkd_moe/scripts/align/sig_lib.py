"""Measured spot scores of the expression model's transfer signature (expr module, dkd_moe/results/expr_transfer).

Score per spot = mean z of measured up genes minus mean z of measured down genes, z of log1p(CP10k)
across the in-tissue spots of the section (genes detected in >= 1% of spots; unmeasured genes are
left out, never zero-filled). Signatures: the final transfer signature (all KPMP) and the fold-wise
signatures built without fold f (fold_signatures.tsv, chosen construction), used for KPMP fold-f
participants so that their disease labels never contributed to the gene list scoring them.
"""
from __future__ import annotations
import importlib, json
import numpy as np
import pandas as pd

import common as C

vc = importlib.import_module("10_visium_common")
P2 = importlib.import_module("02_prepare")


def signatures() -> dict[str, tuple[list, list]]:
    d = C.PKG / "results/expr_transfer"
    t = pd.read_csv(d / "transfer_signature.tsv", sep="\t")
    out = {"final": (t[t.direction == "up"].gene.tolist(), t[t.direction == "down"].gene.tolist())}
    chosen = json.loads((d / "selection.json").read_text())["chosen"]
    f = pd.read_csv(d / "fold_signatures.tsv", sep="\t")
    f = f[f.construction == chosen]
    for k, g in f.groupby("fold"):
        out[f"fold{k}"] = (g[g.direction == "up"].gene.tolist(), g[g.direction == "down"].gene.tolist())
    return out


def section_matrix(cohort, section, participant, h5, positions, cache):
    """(X, genes, barcodes) of all in-tissue spots of a section on its own platform's full gene list."""
    if cohort == "gse211785":
        if "gse" not in cache:
            cache["gse"] = P2.gse_matrix()
        M, genes, bcs, sec = cache["gse"]
        m = sec == section
        return M[m], genes, bcs[m]
    X, genes, bcs = P2.read_kpmp(h5, positions)
    if cohort == "gse339455":
        if "reg" not in cache:
            cache["reg"] = pd.read_csv(C.p(C.load_cfg()["paths"]["external_regions"]), sep="\t")
        reg = cache["reg"]
        keep = bcs.isin(reg.barcode[reg.region_id == participant])
        X, bcs = X[keep], bcs[keep]
    return X, genes, bcs


def score_sections(sections: pd.DataFrame, reg: pd.DataFrame, sigs: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Spot scores (embedded spots of each section) for every signature + per-section gene coverage."""
    reg = reg.set_index("section")
    rows, cov, cache = [], [], {}
    for s in sections.itertuples():
        r = reg.loc[s.section]
        X, genes, bcs = section_matrix(s.cohort, s.section, s.participant, r.h5, r.positions, cache)
        Z, _ = vc.lognorm_z(X, genes); Z.index = bcs
        z = np.load(s.cache)
        ids = pd.Index(z["spot_id"])
        d = pd.DataFrame({"section": s.section, "spot_id": ids})
        for k, (up, dn) in sigs.items():
            u = [g for g in up if g in Z.columns]; w = [g for g in dn if g in Z.columns]
            sc = (Z[u].mean(1) if u else 0) - (Z[w].mean(1) if w else 0)
            d[f"sig_{k}"] = pd.Series(sc, index=Z.index).reindex(ids).values
            cov.append({"cohort": s.cohort, "section": s.section, "signature": k, "n_up": len(up), "n_down": len(dn),
                        "up_measured": len(u), "down_measured": len(w),
                        "up_on_platform": sum(g in set(genes) for g in up), "down_on_platform": sum(g in set(genes) for g in dn)})
        rows.append(d)
        print("scored", s.section, flush=True)
    return pd.concat(rows, ignore_index=True), pd.DataFrame(cov)


def ridge_direction(V: np.ndarray, y: np.ndarray, alphas) -> tuple[np.ndarray, float, float]:
    from sklearn.linear_model import RidgeCV
    ok = np.isfinite(y)
    m = RidgeCV(alphas=alphas).fit(V[ok], y[ok])
    return m.coef_.astype(np.float32), float(m.intercept_), float(m.alpha_)
