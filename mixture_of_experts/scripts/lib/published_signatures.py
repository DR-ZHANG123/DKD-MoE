"""Parsers for published proximal-tubule injury-state signatures (stage 14c).

Every gene list is read from a downloaded supplementary table or from a sentence of the downloaded
full text; text-derived lists are only accepted when the quoted sentence is found verbatim
(whitespace-insensitive) in that full text. Sources are fetched by
scripts/stages/00c_download_published_signatures.sh."""
from __future__ import annotations
import re
from pathlib import Path
import pandas as pd

KIRITA = "Kirita et al. 2020 PNAS 117:15874 doi:10.1073/pnas.2005477117"
LAKE = "Lake et al. 2023 Nature 619:585 doi:10.1038/s41586-023-05769-3"
MUTO = "Muto et al. 2021 Nat Commun 12:2190 doi:10.1038/s41467-021-22368-w"
ABEDINI = "Abedini et al. 2024 Nat Genet 56:1712 doi:10.1038/s41588-024-01802-x"

# (signature, source, full-text file, verbatim quote, genes named in the quote as HGNC symbols, locator)
# alias -> HGNC: CD133 = PROM1, IL-18 = IL18; mouse Vcam1/Dcdc2a/Sema5a are mapped through MGI below.
TEXT_QUOTES = [
    ("Kirita2020_FRPTC_text", KIRITA, "kirita2020_fulltext.xml",
     "This cluster expressed a distinct set of genes not observed in either healthy or acutely injured mouse "
     "proximal tubule. These included Vcam1, Dcdc2a, and Sema5a (Fig. 2A and Dataset S2).",
     ["Vcam1", "Dcdc2a", "Sema5a"], "Results, 'Proximal Tubule Responses to Acute Injury'; Fig. 2A (mouse genes, MGI 1:1 orthologs)"),
    ("Muto2021_PTVCAM1_text", MUTO, "muto2021_fulltext.xml",
     "the PT_VCAM1 population showed increased expression of kidney injury molecule-1 (KIM1, HAVCR1)",
     ["VCAM1", "HAVCR1"], "Results, 'NF-kB regulates the molecular signature of a subpopulation of proximal tubule that expresses VCAM1'; Fig. 5"),
    ("Muto2021_PTVCAM1_text", MUTO, "muto2021_fulltext.xml",
     "PT_VCAM1 also expressed VIM (vimentin), CD24, and CD133 (Supplementary Fig. 14)",
     ["VIM", "CD24", "PROM1"], "same paragraph; Supplementary Fig. 14 (CD133 = PROM1)"),
    ("Abedini2024_iPT_VCAM1_text", ABEDINI, "abedini2024_fulltext.xml",
     "one of the iPT clusters expressed VCAM1, ACSL1, ASS1 and ASPA",
     ["VCAM1", "ACSL1", "ASS1", "ASPA"], "Results, 'Injured tubule cells in human kidney fibrosis'; Fig. 5f-g"),
    ("Abedini2024_iPT_HAVCR1_text", ABEDINI, "abedini2024_fulltext.xml",
     "The second iPT cluster expressed HAVCR1 (which encodes KIM1), NFKBIZ, IL-18, ITGA3, PDGFB and ITGB1",
     ["HAVCR1", "NFKBIZ", "IL18", "ITGA3", "PDGFB", "ITGB1"], "Results, 'Injured tubule cells in human kidney fibrosis'; Supplementary Fig. 25 (IL-18 = IL18)"),
]


def _squash(s: str) -> str:
    return re.sub(r"\s+", "", s)


def _plain_text(xml: Path) -> str:
    t = re.sub(r"<[^>]+>", " ", xml.read_text(errors="ignore"))
    return _squash(t.replace("&lt;", "<").replace("&gt;", ">").replace("&amp;", "&"))


def mgi_one_to_one(raw: Path) -> dict[str, str]:
    """mouse symbol -> human symbol for MGI homology classes with exactly one mouse and one human gene."""
    h = pd.read_csv(raw / "MGI_HOM_MouseHumanSequence.rpt", sep="\t", dtype=str)
    h["org"] = h["Common Organism Name"].str.startswith("mouse").map({True: "mouse", False: "human"})
    n = h.groupby(["DB Class Key", "org"]).size().unstack(fill_value=0)
    keep = n.index[(n.get("mouse", 0) == 1) & (n.get("human", 0) == 1)]
    h = h[h["DB Class Key"].isin(keep)]
    m = h[h.org == "mouse"].set_index("DB Class Key").Symbol
    u = h[h.org == "human"].set_index("DB Class Key").Symbol
    return dict(zip(m, u.reindex(m.index)))


def _rows(sig, genes, source, locator):
    return [{"signature": sig, "gene": g, "rank": i + 1, "source": source, "locator": locator}
            for i, g in enumerate(dict.fromkeys(genes))]


def text_signatures(raw: Path, orth: dict[str, str]) -> list[dict]:
    rows = []
    for sig, src, fn, quote, genes, loc in TEXT_QUOTES:
        if _squash(quote) not in _plain_text(raw / fn):
            raise ValueError(f"quote for {sig} not found verbatim in {fn}")
        hs = [orth.get(g, None) for g in genes] if sig.startswith("Kirita") else genes
        if any(g is None for g in hs):
            raise ValueError(f"{sig}: no 1:1 human ortholog for {genes}")
        rows += _rows(sig, hs, src, f'{loc}; quote: "{quote}"')
    # the two Muto quotes form one signature; PT_VCAM1 is defined by VCAM1 itself
    return rows


def _top(df, gene, fc, n, padj_col, padj):
    df = df.assign(**{c: pd.to_numeric(df[c], errors="coerce") for c in (fc, padj_col)})
    d = df[(df[fc] > 0) & (df[padj_col] < padj)].sort_values(fc, ascending=False)
    return d[gene].astype(str).drop_duplicates().head(n).tolist(), len(d)


def kirita_ds2(raw: Path, orth, cluster: int, padj: float, n: int) -> list[dict]:
    d = pd.read_excel(raw / "kirita2020_sd02.xlsx", "MousePT_DEG")
    d = d[d.cluster == cluster].copy()
    d["human"] = d.gene.astype(str).map(orth)
    d = d.dropna(subset=["human"])
    genes, tot = _top(d, "human", "avg_logFC", n, "p_val_adj", padj)
    loc = (f"Dataset S2 (pnas.2005477117.sd02.xlsx) sheet MousePT_DEG, cluster {cluster} (the only cluster "
           f"carrying Vcam1/Dcdc2a/Sema5a = FR-PTC); avg_logFC>0, p_val_adj<{padj}; MGI 1:1 orthologs "
           f"({tot} mapped genes), top {n} by avg_logFC")
    return _rows("Kirita2020_FRPTC_DS2", genes, KIRITA, loc)


def lake_tables(raw: Path) -> list[dict]:
    rows = []
    d = pd.read_excel(raw / "lake2023_ST13.xlsx", header=None).iloc[7:]
    for sig, gc, sc, lab in [("Lake2023_aPT_snCv3", 2, 3, "snCv3"), ("Lake2023_aPT_scCv3", 7, 8, "scCv3")]:
        g = d.loc[d[sc] == "aPT", gc].dropna().astype(str).str.strip().tolist()
        rows += _rows(sig, g, LAKE, f"Supplementary Table 13, {lab} 'Full Conserved Adaptive State Gene Set "
                                    f"(p val < 0.05 for Ref, AKI and CKD)', rows with State/Cluster = aPT (source order)")
    e = pd.read_excel(raw / "lake2023_ST29.xlsx", header=None)
    hdr = e.iloc[5]
    for col, sig in [("aPT", "Lake2023_aPT_clinical"), ("Degerenative", "Lake2023_degenerative_clinical")]:
        c = hdr[hdr == col].index[0]
        g = e.iloc[6:][c].dropna().astype(str).str.strip().tolist()
        rows += _rows(sig, g, LAKE, f"Supplementary Table 29 'Altered State Gene Sets', column '{col}' "
                                    f"(expanded gene sets used for clinical outcome analysis)")
    return rows


def muto_table(raw: Path, padj: float, n: int) -> list[dict]:
    d = pd.read_excel(raw / "muto2021_MOESM8.xlsx", header=1)
    genes, tot = _top(d, "gene", "avg_logFC", n, "p_val_adj", padj)
    return _rows("Muto2021_PTVCAM1_vs_PT", genes, MUTO,
                 f"Supplementary Data 5 (41467_2021_22368_MOESM8_ESM.xlsx, 'Differentially expressed genes in PT_VCAM1 vs PT'); "
                 f"avg_logFC>0, p_val_adj<{padj} ({tot} genes), top {n} by avg_logFC")


def abedini_table(raw: Path, padj: float, n: int) -> list[dict]:
    import openpyxl
    wb = openpyxl.load_workbook(raw / "abedini2024_MOESM3_supp_tables.xlsx", read_only=True)
    it = wb["Supplementary Table 8"].iter_rows(values_only=True)
    hdr = next(it)
    d = pd.DataFrame([r for r in it if r[6] == "iPT"], columns=hdr)
    wb.close()
    genes, tot = _top(d, "genes", "avg_log2FC", n, "p_val_adj", padj)
    return _rows("Abedini2024_iPT_ST8", genes, ABEDINI,
                 f"Supplementary Table 8 (cluster markers), cluster = iPT; avg_log2FC>0, p_val_adj<{padj} "
                 f"({tot} genes), top {n} by avg_log2FC. Same donor cohort as GSE211785 (discovery)")


def all_signatures(raw: Path, cfg: dict) -> pd.DataFrame:
    orth = mgi_one_to_one(raw)
    padj, n = cfg["padj"], cfg["max_genes"]
    rows = (text_signatures(raw, orth) + kirita_ds2(raw, orth, cfg["kirita_frptc_cluster"], padj, n)
            + lake_tables(raw) + muto_table(raw, padj, n) + abedini_table(raw, padj, n))
    df = pd.DataFrame(rows).drop_duplicates(["signature", "gene"])
    df["rank"] = df.groupby("signature").cumcount() + 1
    return df[["signature", "gene", "rank", "source", "locator"]]
