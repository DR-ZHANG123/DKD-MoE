"""Readers for the KPMP open-access clinical table and TIV descriptor workbook."""
from __future__ import annotations
from pathlib import Path
import pandas as pd


def read_descriptors(path: str | Path) -> pd.DataFrame:
    """'Data Table' sheet: rows 0-2 are section / label / variable-id headers; data from row 3."""
    raw = pd.read_excel(path, sheet_name="Data Table", header=None)
    hdr_row = raw.index[raw.eq("Participant ID").any(axis=1)][0]
    df = raw.iloc[hdr_row + 1:].copy()
    df.columns = raw.iloc[hdr_row].astype(str).str.strip()
    df = df[df["Participant ID"].notna()].reset_index(drop=True)
    df["Participant ID"] = df["Participant ID"].astype(str).str.strip()
    return df
