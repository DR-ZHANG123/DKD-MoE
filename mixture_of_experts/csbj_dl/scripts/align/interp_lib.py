"""Helpers for the DKD-state direction in the shared image-expression space."""
from __future__ import annotations
import numpy as np
import pandas as pd
from scipy import stats

import common as C


def direction(ze: np.ndarray, meta: pd.DataFrame, train: np.ndarray, a: str, b: str, min_spots: int):
    """Unit vector: mean over group-a training participants of their mean PT-rich expression
    embedding minus the same for group b (participants weighted equally, not spots)."""
    m = meta[train & meta.pt_rich.values & meta.group.isin([a, b]).values]
    P = pd.DataFrame(ze[m.index.values], index=m.index).groupby(m.participant.values).mean()
    n = m.groupby("participant").size()
    P = P[n.reindex(P.index) >= min_spots]
    g = m.drop_duplicates("participant").set_index("participant").group.reindex(P.index)
    d = P[g == a].mean().values - P[g == b].mean().values
    return d / np.linalg.norm(d), int((g == a).sum()), int((g == b).sum())


def standardise(score: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (score - score[ref].mean()) / score[ref].std()


def spearman(x, y) -> float:
    ok = np.isfinite(x) & np.isfinite(y)
    return float(stats.spearmanr(x[ok], y[ok]).correlation) if ok.sum() >= 10 else np.nan


def one_sample(x: pd.Series) -> dict:
    x = x.dropna()
    if len(x) < 3:
        return {"n": len(x), "mean": x.mean() if len(x) else np.nan}
    return {"n": len(x), "mean": float(x.mean()), "sd": float(x.std()), "frac_positive": float((x > 0).mean()),
            "wilcoxon_p": float(stats.wilcoxon(x).pvalue)}


def measured_programme_scores() -> pd.DataFrame:
    """Spot-level scores from measured expression (stages 10a/10b) for spot-level correspondence:
    DKDiPT_up programme score and injury index; keyed by (section, spot_id)."""
    k = pd.read_csv(C.ROOT / "results/10b_visium_kpmp/spot_scores.tsv.gz", sep="\t", dtype={"participant": str},
                    usecols=["barcode", "participant", "sample", "DKDiPT_up", "SharedInjury_up", "injury_index"])
    k["section"] = k.participant + "__" + k["sample"]
    g = pd.read_csv(C.ROOT / "results/10a_visium_gse211785/spot_scores.tsv.gz", sep="\t",
                    usecols=["barcode", "section", "DKDiPT_up", "SharedInjury_up", "injury_index"])
    cols = ["section", "barcode", "DKDiPT_up", "SharedInjury_up", "injury_index"]
    return pd.concat([k[cols], g[cols]]).rename(columns={"barcode": "spot_id"}).set_index(["section", "spot_id"])


def within_section_corr(S: pd.DataFrame, x: str, ys: list[str], mask_col: str | None = "pt_rich",
                        min_spots: int = 20) -> pd.DataFrame:
    rows = []
    d = S[S[mask_col]] if mask_col else S
    for (sec, part), g in d.groupby(["section", "participant"]):
        if len(g) < min_spots:
            continue
        r = {"section": sec, "participant": part, "cohort": g.cohort.iloc[0], "group": g.group.iloc[0], "n": len(g)}
        for y in ys:
            r[y] = spearman(g[x].values, g[y].values)
        rows.append(r)
    return pd.DataFrame(rows)
