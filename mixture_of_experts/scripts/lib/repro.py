"""Reproducibility helpers shared by every analysis stage of project016."""
from __future__ import annotations
import hashlib, json, os, platform, random, subprocess, sys, time
from pathlib import Path
import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_config(path: str | Path = ROOT / "config" / "run.yaml") -> dict:
    with open(path) as fh:
        return yaml.safe_load(fh)


def config_sha(path: str | Path = ROOT / "config" / "run.yaml") -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def set_global_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


def md5(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        while b := fh.read(chunk):
            h.update(b)
    return h.hexdigest()


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "-C", str(ROOT), "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "untracked"


def _env_snapshot() -> dict:
    pk = {}
    for name in ["numpy", "pandas", "scipy", "scanpy", "anndata", "pydeseq2", "torch",
                 "scDisInFact", "statsmodels", "sklearn", "timm", "decoupler"]:
        try:
            mod = __import__(name)
            pk[name] = getattr(mod, "__version__", "?")
        except Exception:
            pass
    return {"python": sys.version.split()[0], "platform": platform.platform(), "packages": pk}


def stage_dir(stage: str) -> Path:
    d = ROOT / "results" / stage
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_provenance(stage: str, inputs: list, outputs: list, seed: int | None,
                     extra: dict | None = None, hash_limit: int = 2 << 30) -> Path:
    """Record input/output MD5s (files > hash_limit get size+mtime only)."""
    def desc(p):
        p = Path(p)
        if not p.exists():
            return {"path": str(p), "missing": True}
        st = p.stat()
        rel = str(p.resolve().relative_to(ROOT)) if str(p.resolve()).startswith(str(ROOT)) else str(p)
        d = {"path": rel, "bytes": st.st_size}
        d["md5"] = md5(p) if st.st_size <= hash_limit else f"size-mtime:{st.st_size}-{int(st.st_mtime)}"
        return d
    prov = {"stage": stage, "time": time.strftime("%Y-%m-%dT%H:%M:%S"), "seed": seed,
            "config_sha": config_sha(), "git_commit": _git_commit(),
            "inputs": [desc(p) for p in inputs], "outputs": [desc(p) for p in outputs],
            "environment": _env_snapshot()}
    if extra:
        prov["extra"] = extra
    out = stage_dir(stage) / "PROVENANCE.json"
    out.write_text(json.dumps(prov, indent=2))
    return out
