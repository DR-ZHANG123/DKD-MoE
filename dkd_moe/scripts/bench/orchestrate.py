"""One-command benchmark: waits for the final shared inputs, runs every method and MoE ablation
(5 seeds x 5 KPMP folds, one job per method x seed) on the configured GPU slots, then summarises.

Waits for (1) expr_prep built with the expected label source (bench.yaml rerun.expected_label_source),
external objects at least as new as the KPMP object and no prep process running; (2) before MoE jobs,
an expr cross-validation selection newer than the KPMP object with no CV worker running.
Usage: python orchestrate.py [--slots cuda:2 cuda:2 cuda:6] [--only pseudobulk scvi ...] [--no-wait]
Logs: dkd_moe/logs/bench_<job>.log, dkd_moe/logs/bench_orchestrate.log
"""
from __future__ import annotations
import argparse, json, subprocess, sys, threading, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "common"))
from cdl import CDL, load_cfg, result_dir  # noqa: E402

LOG = CDL / "logs"; LOG.mkdir(parents=True, exist_ok=True)
PY = sys.executable
lock = threading.Lock()


def log(msg):
    with lock:
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}"
        print(line, flush=True)
        with open(LOG / "bench_orchestrate.log", "a") as fh:
            fh.write(line + "\n")


def running(pattern: str) -> bool:
    r = subprocess.run(["pgrep", "-f", pattern], capture_output=True, text=True)
    return bool(r.stdout.strip())


def bulk() -> Path:
    return Path(load_cfg("expr")["paths"]["bulk_dir"])


def prep_ready(cfg) -> bool:
    ls = result_dir("expr_prep") / "label_source.json"
    k = bulk() / "kpmp_pt_hvg.h5ad"
    exts = [bulk() / "gse211785_pt_hvg.h5ad", bulk() / "gse195460_pt_hvg.h5ad"]
    if not (ls.exists() and k.exists() and all(e.exists() for e in exts)):
        return False
    if json.loads(ls.read_text()).get("label_source") != cfg["rerun"]["expected_label_source"]:
        return False
    if any(e.stat().st_mtime < k.stat().st_mtime for e in exts):
        return False
    return not (running("expr/prep_kpmp.py") or running("expr/prep_external.py"))


def selection_ready() -> bool:
    sel = result_dir("expr_cv") / "selection.json"
    k = bulk() / "kpmp_pt_hvg.h5ad"
    return (sel.exists() and sel.stat().st_mtime > k.stat().st_mtime
            and not running("expr/cv_worker.py") and not running("expr/cv_summary.py"))


def wait(pred, what, poll):
    t0 = time.time()
    while not pred():
        time.sleep(poll)
    log(f"ready: {what} (waited {(time.time() - t0) / 60:.0f} min)")


def jobs(cfg, only):
    J = []
    for seed in cfg["seeds"]:
        for m in ("scdisinfact", "scvi", "cvae"):
            J.append({"name": f"{m}_s{seed}", "cost": cfg["rerun"]["est_hours"][m], "moe": False,
                      "args": ["--method", m, "--seeds", str(seed), "--tag", f"s{seed}"]})
        for v in cfg["moe_ablations"]["variants"]:
            J.append({"name": f"moe_{v}_s{seed}", "cost": cfg["rerun"]["est_hours"]["moe"], "moe": True,
                      "args": ["--method", "moe", "--variant", v, "--seeds", str(seed), "--tag", f"s{seed}"]})
    if only:
        J = [j for j in J if any(j["name"].startswith(o) for o in only)]
    return J


def run_job(j, device):
    cmd = [PY, str(HERE / "run_bench.py")] + j["args"] + (["--device", device] if device else [])
    log(f"start {j['name']} on {device}")
    t0 = time.time()
    with open(LOG / f"bench_{j['name']}.log", "w") as fh:
        rc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=HERE,
                            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"}).returncode
    log(f"end {j['name']} rc={rc} {(time.time() - t0) / 3600:.2f} h")
    return rc


def slot_worker(device, queue, poll, failures, retries):
    """Runs its queue in order; a failed job (e.g. CUDA out of memory on a shared GPU) is retried later."""
    pending = [(j, 0) for j in queue]
    while pending:
        j, tries = pending.pop(0)
        if j["moe"]:
            wait(selection_ready, f"expr selection for {j['name']}", poll)
        if run_job(j, device) != 0:
            if tries < retries:
                log(f"retry later {j['name']} ({tries + 1}/{retries})"); time.sleep(poll)
                pending.append((j, tries + 1))
            else:
                failures.append(j["name"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--slots", nargs="*"); ap.add_argument("--only", nargs="*")
    ap.add_argument("--no-wait", action="store_true")
    a = ap.parse_args()
    cfg = load_cfg("bench"); poll = cfg["rerun"]["poll_seconds"]
    slots = a.slots or cfg["rerun"]["slots"]
    if not a.no_wait:
        wait(lambda: prep_ready(cfg), "expr_prep with the final label source", poll)
    J = jobs(cfg, a.only)
    cpu_jobs = [j for j in J if j["name"].rsplit("_s", 1)[0] in cfg["rerun"]["cpu_methods"]]
    J = [j for j in J if j not in cpu_jobs]
    # longest-processing-time assignment; within a slot, non-MoE jobs first (MoE waits for the selection)
    load = [0.0] * len(slots); queues = [[] for _ in slots]
    for j in sorted(J, key=lambda j: -j["cost"]):
        i = min(range(len(slots)), key=lambda k: load[k]); queues[i].append(j); load[i] += j["cost"]
    for q in queues:
        q.sort(key=lambda j: (j["moe"], -j["cost"]))
    log(f"slots {slots}; estimated hours per slot {[round(x, 1) for x in load]}")
    failures, threads = [], []
    if not a.only or "pseudobulk" in a.only:
        threads.append(threading.Thread(target=lambda: failures.append("pseudobulk") if run_job(
            {"name": "pseudobulk", "args": ["--method", "pseudobulk"]}, "cpu") else None))
    threads += [threading.Thread(target=slot_worker, args=("cpu", [j], poll, failures, cfg["rerun"]["retries"]))
                for j in cpu_jobs]
    threads += [threading.Thread(target=slot_worker, args=(d, q, poll, failures, cfg["rerun"]["retries"])) for d, q in zip(slots, queues)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    log(f"all jobs finished; failures: {failures}")
    with open(LOG / "bench_summary.log", "w") as fh:
        subprocess.run([PY, str(HERE / "summarize.py")], stdout=fh, stderr=subprocess.STDOUT, cwd=HERE)
    log("summary written")


if __name__ == "__main__":
    main()
