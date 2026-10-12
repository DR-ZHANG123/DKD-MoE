"""02_download: fetch one csbj_dl category from the KPMP open repository with resume, size and MD5 checks.

Usage: python 02_download.py <multiome|visium|pathomics|annotation> [--workers N] [--only SUBSTRING]
Reads csbj_dl/results/data_inventory/download_plan_<category>.tsv, writes
csbj_dl/results/data_download/<category>/download_manifest_<category>.tsv (one row per file: status, expected/observed bytes, md5, local path).
The KPMP server does not honour HTTP range requests, so resume is file-level: a file whose size
already matches the repository is kept, a partial file is fetched again.
Pathomics archives: only the feature workbooks, OME-XML (pixel size) and README are extracted; the
archive (masks + svs) is then removed and a .done.json record (archive size + md5) keeps resume working."""
from __future__ import annotations
import json, subprocess, sys, time, zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pandas as pd
from dcommon import cfg, dest, md5, result_dir, write_provenance

PATHO_KEEP = (".xlsx", ".ome.xml", "README.md")


def svs_mpp(zf: zipfile.ZipFile, member: str, tmpdir: Path) -> float | None:
    """Microns per pixel from the slide's Aperio ImageDescription (needed: v2 features are in pixels)."""
    import re, tifffile
    tmp = tmpdir / ("_tmp_" + Path(member).name)
    with zf.open(member) as src, open(tmp, "wb") as dst:
        while b := src.read(1 << 24):
            dst.write(b)
    try:
        with tifffile.TiffFile(tmp) as t:
            m = re.search(r"MPP\s*=\s*([0-9.]+)", t.pages[0].description or "")
        return float(m.group(1)) if m else None
    finally:
        tmp.unlink(missing_ok=True)


def extract_pathomics(zpath: Path) -> list[str]:
    outdir = zpath.parent / zpath.name.replace(".zip", "")
    outdir.mkdir(exist_ok=True)
    kept, mpp = [], {}
    with zipfile.ZipFile(zpath) as z:
        for n in z.namelist():
            if n.endswith(PATHO_KEEP) and not n.endswith("/"):
                target = outdir / Path(n).name
                target.write_bytes(z.read(n))
                kept.append(target.name)
            elif n.lower().endswith(".svs"):
                mpp[Path(n).name] = svs_mpp(z, n, outdir)
    (outdir / "slide_mpp.json").write_text(json.dumps(mpp))
    return kept + ["slide_mpp.json"]


def fetch(row, c: dict, category: str) -> dict:
    dst = dest(row, c, category)
    dst.parent.mkdir(parents=True, exist_ok=True)
    want = int(row.file_size)
    rec = {"redcap_id": row.redcap_id if category != "annotation" else "atlas", "file_name": row.file_name,
           "expected_bytes": want, "local_path": str(dst), "download_url": row.download_url}
    done = dst.with_name(dst.name + ".done.json")
    if done.exists():
        return {**rec, **json.loads(done.read_text()), "status": "ok_cached"}
    t0 = time.time()
    if not (dst.exists() and dst.stat().st_size == want):
        dl = c["download"]
        for attempt in range(dl.get("attempts", 5)):          # the server drops some connections immediately
            r = subprocess.run(["wget", "-q", f"--tries={dl['tries']}", f"--waitretry={dl['waitretry']}",
                                "--timeout=120", "-O", str(dst), row.download_url])
            if r.returncode == 0 and dst.exists() and dst.stat().st_size == want:
                break
            time.sleep(30 * (attempt + 1))
        else:
            got = dst.stat().st_size if dst.exists() else 0
            dst.unlink(missing_ok=True)
            return {**rec, "status": "FAIL", "observed_bytes": got, "md5": "", "seconds": round(time.time() - t0)}
    info = {"observed_bytes": dst.stat().st_size, "md5": md5(dst), "seconds": round(time.time() - t0)}
    if category == "pathomics":
        try:
            info["extracted"] = ";".join(extract_pathomics(dst))
        except zipfile.BadZipFile as e:
            dst.unlink(missing_ok=True)
            return {**rec, **info, "status": f"FAIL bad zip: {e}"}
        dst.unlink()
        done.write_text(json.dumps(info))
    return {**rec, **info, "status": "ok"}


def main():
    category = sys.argv[1]
    c = cfg()
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else c["download"]["workers"]
    out = result_dir("data_inventory")
    plan = pd.read_csv(out / f"download_plan_{category}.tsv", sep="\t", low_memory=False)
    only = sys.argv[sys.argv.index("--only") + 1] if "--only" in sys.argv else None
    if only:                                   # retry a subset in a separate process; full rerun later rebuilds the manifest
        plan = plan[plan.file_name.str.contains(only, regex=False)]
    # core study groups first (DKD, HKD, HKD_withDM, Reference, DM_noCKD), then the rest; small files first within a tier
    plan = plan.assign(_tier=(~plan.group.isin(c["core_groups"])).astype(int)).sort_values(["_tier", "file_size"]).drop(columns="_tier")
    print(category, len(plan), "files", round(plan.file_size.sum() / 1e9, 1), "GB", flush=True)
    recs = []
    with ThreadPoolExecutor(workers) as ex:
        futs = [ex.submit(fetch, r, c, category) for r in plan.itertuples()]
        for i, f in enumerate(as_completed(futs), 1):
            r = f.result(); recs.append(r)
            print(f"[{i}/{len(plan)}] {r['status']} {r['file_name']} {r.get('seconds', '')}s", flush=True)
    M = pd.DataFrame(recs).sort_values(["redcap_id", "file_name"])
    mdir = result_dir(f"data_download/{category}")
    mf = mdir / (f"download_manifest_{category}.tsv" if not only else f"retry_{category}_{only[:20]}.tsv")
    M.to_csv(mf, sep="\t", index=False)
    nfail = int(M.status.str.startswith("FAIL").sum())
    print(category, "done:", len(M) - nfail, "ok,", nfail, "failed", flush=True)
    write_provenance(f"data_download/{category}", [out / f"download_plan_{category}.tsv"], [mf], c["seed"], cfg=c,
                     extra={"category": category, "n_ok": len(M) - nfail, "n_fail": nfail})


if __name__ == "__main__":
    main()
