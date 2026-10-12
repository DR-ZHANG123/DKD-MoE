"""06_ext_download: fetch the external-test files from GEO (never used for training or tuning).

- GSE339455 (Visium CytAssist, FFPE, biopsy-proven hypertensive nephropathy, 4 slides x 3 patients):
  GSE339455_RAW.tar -> raw/gse339455/
- GSE195460 snATAC (Control1-6, DN1-7): fragments + filtered_peak_bc_matrix.h5 (cell barcodes) per
  library -> raw/gse195460_atac/
GSE211785 deposits no snATAC fragments ("raw data were not provided"); its snATAC nuclei exist only as
gene-activity counts inside the merged object already on disk (handled in 07_ext_atac.py).
File lists come from each series' GEO filelist.txt; every file is size-checked against it and MD5-hashed.
Output dkd_moe/results/data_download/external/download_manifest_external.tsv."""
from __future__ import annotations
import io, subprocess, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import pandas as pd
from dcommon import cfg, md5, result_dir, write_provenance

GEO = "https://ftp.ncbi.nlm.nih.gov/geo"


def series_dir(gse: str) -> str:
    return f"{GEO}/series/{gse[:-3]}nnn/{gse}/suppl"


def sample_url(name: str) -> str:
    gsm = name.split("_", 1)[0]
    return f"{GEO}/samples/{gsm[:-3]}nnn/{gsm}/suppl/{name}"


def content_length(url: str) -> int:
    req = urllib.request.Request(url, method="HEAD")
    return int(urllib.request.urlopen(req, timeout=60).headers.get("Content-Length", -1))


def filelist(gse: str) -> pd.DataFrame:
    txt = urllib.request.urlopen(f"{series_dir(gse)}/filelist.txt", timeout=60).read().decode()
    return pd.read_csv(io.StringIO(txt), sep="\t")


def plan() -> pd.DataFrame:
    c = cfg(); root = Path(c["raw_dir"]).parent
    rows = []
    fl = filelist("GSE339455")
    a = fl[fl["#Archive/File"] == "Archive"].iloc[0]
    rows.append({"dataset": "GSE339455", "file": a.Name, "bytes": int(a.Size), "url": f"{series_dir('GSE339455')}/{a.Name}",
                 "dest": str(root / "gse339455" / a.Name)})
    fl = filelist("GSE195460")
    want = lambda n: n.endswith("_fragments.tsv.gz") or n.endswith("_filtered_peak_bc_matrix.h5")
    for r in fl[(fl["#Archive/File"] == "File") & fl.Name.map(want)].itertuples():
        rows.append({"dataset": "GSE195460_atac", "file": r.Name, "bytes": int(r.Size), "url": sample_url(r.Name),
                     "dest": str(root / "gse195460_atac" / r.Name)})
    sdir = urllib.request.urlopen(series_dir("GSE195460") + "/", timeout=60).read().decode()
    for n in sorted(set(x.split('"')[0] for x in sdir.split('href="')[1:])):
        if want(n) and n.startswith("GSE195460_"):
            rows.append({"dataset": "GSE195460_atac", "file": n, "bytes": content_length(f"{series_dir('GSE195460')}/{n}"), "url": f"{series_dir('GSE195460')}/{n}",
                         "dest": str(root / "gse195460_atac" / n)})
    return pd.DataFrame(rows)


def fetch(r) -> dict:
    d = Path(r.dest); d.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    ok = d.exists() and (r.bytes < 0 or d.stat().st_size == r.bytes)
    if not ok:
        for attempt in range(5):
            p = subprocess.run(["wget", "-q", "-c", "--tries=10", "--timeout=120", "-O", str(d), r.url])
            ok = p.returncode == 0 and d.exists() and (r.bytes < 0 or d.stat().st_size == r.bytes)
            if ok:
                break
            time.sleep(20 * (attempt + 1))
    return {**r._asdict(), "status": "ok" if ok else "FAIL", "observed_bytes": d.stat().st_size if d.exists() else 0,
            "md5": md5(d) if ok else "", "seconds": round(time.time() - t0)}


def main():
    c = cfg()
    P = plan().drop_duplicates("file")
    print(P.groupby("dataset").agg(n=("file", "size"), GB=("bytes", lambda s: round(s[s > 0].sum() / 1e9, 1))), flush=True)
    with ThreadPoolExecutor(6) as ex:
        recs = list(ex.map(fetch, P.drop(columns=[]).itertuples(index=False)))
    M = pd.DataFrame(recs)
    out = result_dir("data_download/external")
    M.to_csv(out / "download_manifest_external.tsv", sep="\t", index=False)
    print(M.status.value_counts().to_string(), flush=True)
    write_provenance("data_download/external", [], [out / "download_manifest_external.tsv"], c["seed"], cfg=c,
                     extra={"n_ok": int((M.status == "ok").sum()), "n_fail": int((M.status != "ok").sum())})


if __name__ == "__main__":
    main()
