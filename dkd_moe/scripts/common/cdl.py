"""Shared helpers for dkd_moe scripts: paths, config loading and provenance into dkd_moe/results."""
from __future__ import annotations
import hashlib, os, sys
from pathlib import Path
import yaml

ROOT = Path(__file__).resolve().parents[3]          # DKD-MoE
CDL = ROOT / "dkd_moe"
sys.path.insert(0, str(ROOT / "scripts" / "lib"))
import repro  # noqa: E402  (project-wide reproducibility helpers)

set_global_seed = repro.set_global_seed
md5 = repro.md5


def load_cfg(name: str) -> dict:
    p = CDL / "config" / f"{name}.yaml"
    with open(p) as fh:
        cfg = yaml.safe_load(fh)
    cfg["_sha"] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
    cfg["_path"] = str(p)
    dev = os.environ.get("DKDMOE_DEVICE")          # optional runtime override (e.g. cpu when the GPU is full)
    if dev and "train" in cfg:
        cfg["train"]["device"] = dev
    return cfg


def path(cfg: dict, key: str) -> Path:
    p = Path(cfg["paths"][key])
    return p if p.is_absolute() else ROOT / p


def result_dir(name: str) -> Path:
    d = CDL / "results" / name
    d.mkdir(parents=True, exist_ok=True)
    return d


def write_provenance(name: str, inputs: list, outputs: list, seed, cfg: dict | None = None,
                     extra: dict | None = None, tag: str | None = None) -> Path:
    """PROVENANCE.json in dkd_moe/results/<name>/ (wraps scripts/lib/repro.write_provenance).
    With `tag`, the record goes to PROVENANCE_<tag>.json and an existing PROVENANCE.json is kept."""
    ex = dict(extra or {})
    if cfg is not None:
        ex["dkd_moe_config"] = cfg.get("_path"); ex["dkd_moe_config_sha"] = cfg.get("_sha")
    d = result_dir(name)
    main = d / "PROVENANCE.json"
    keep = main.read_bytes() if (tag and main.exists()) else None
    p = repro.write_provenance(f"../dkd_moe/results/{name}", inputs, outputs, seed, extra=ex)
    if tag:
        p = p.rename(d / f"PROVENANCE_{tag}.json")
        if keep is not None:
            main.write_bytes(keep)
    return p
