"""Participant-level scores shared by all methods.

Every score is computed per participant x technology sample and then standardised within technology:
KPMP samples against the training-fold DKD/HKD samples of the same technology, external samples
within their own cohort x technology (label-free). The participant score is the mean over its
technologies. Signs are oriented so that higher = more DKD-like.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def ref_mask(meta: pd.DataFrame, test_fold: int, case: str, ctrl: str) -> np.ndarray:
    """KPMP training-fold DKD/HKD samples: the only rows any score may be fitted on."""
    return ((meta.dataset == "KPMP") & (meta.fold != test_fold) & meta.condition.isin([case, ctrl])).values


def _standardise_within(values: pd.DataFrame, meta: pd.DataFrame, ref: np.ndarray) -> pd.DataFrame:
    """Column-wise z within (dataset, tech); KPMP uses ref rows for mean/sd, external all own rows."""
    out = values.copy() * np.nan
    for (ds, tech), idx in meta.groupby(["dataset", "tech"]).groups.items():
        rows = meta.index.get_indexer(idx)
        base = rows[ref[rows]] if ds == "KPMP" else rows
        if len(base) < 2:
            continue
        mu = values.iloc[base].mean(0)
        sd = values.iloc[base].std(0).replace(0, np.nan)
        out.iloc[rows] = ((values.iloc[rows] - mu) / sd).values
    return out


def geneset_score(logx: pd.DataFrame, meta: pd.DataFrame, up: list, down: list, ref: np.ndarray) -> pd.Series:
    """Signed mean z: mean z of up genes minus mean z of down genes (genes z-scored within dataset x tech)."""
    genes = [g for g in list(up) + list(down) if g in logx.columns]
    Z = _standardise_within(logx[genes], meta, ref)
    u = Z[[g for g in up if g in Z.columns]].mean(1)
    d = Z[[g for g in down if g in Z.columns]].mean(1) if len(down) else 0.0
    return (u - d).rename("score")


def latent_lr_score(emb_means: pd.DataFrame, meta: pd.DataFrame, ref: np.ndarray, case: str, C: float,
                    seed: int, fit: np.ndarray | None = None) -> pd.Series:
    """Logistic regression (DKD vs HKD) on per-sample mean latent, fitted on training-fold samples only.
    Features are standardised within dataset x tech before fitting/applying. Returns the decision value."""
    fit = ref if fit is None else fit          # rows the classifier is fitted on (default: the standardising rows)
    Z = _standardise_within(emb_means, meta, ref).fillna(0.0)
    y = (meta.condition.values[fit] == case).astype(int)
    clf = LogisticRegression(C=C, max_iter=5000, random_state=seed, class_weight="balanced")
    clf.fit(Z.values[fit], y)
    return pd.Series(clf.decision_function(Z.values), index=meta.index, name="score")


def mean_by_sample(emb: np.ndarray, obs: pd.DataFrame, meta: pd.DataFrame) -> pd.DataFrame:
    """Mean of a per-cell matrix within participant x technology samples listed in meta.index."""
    key = (obs.dataset.astype(str) + ":" + obs.participant.astype(str) + "|" + obs.tech.astype(str)).values
    df = pd.DataFrame(emb, index=key).groupby(level=0).mean()
    return df.reindex(meta.index)


def participant_table(raw: pd.Series, meta: pd.DataFrame, ref: np.ndarray, method: str, score_name: str,
                      seed: int, fold: int, standardise: bool = True) -> pd.DataFrame:
    """Within-technology standardisation (unless standardise=False), then mean over a participant's
    technologies."""
    z = _standardise_within(raw.to_frame("s"), meta, ref)["s"] if standardise else raw
    df = meta[["dataset", "participant", "tech", "condition"]].copy()
    df["raw"] = raw.values; df["z"] = z.values
    df["role"] = np.where(df.dataset != "KPMP", "external",
                          np.where(meta.fold.values == fold, "test", "train"))
    out = (df.groupby(["dataset", "participant", "condition", "role"], as_index=False)
           .agg(score=("z", "mean"), raw=("raw", "mean"), techs=("tech", lambda s: "+".join(sorted(s)))))
    out.insert(0, "fold", fold); out.insert(0, "seed", seed)
    out.insert(0, "score_name", score_name); out.insert(0, "method", method)
    return out
