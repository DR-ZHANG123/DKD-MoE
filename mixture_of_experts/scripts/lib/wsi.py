"""Whole-slide tiling helpers (openslide): tissue mask, tile grid, tile QC.

Tissue mask: thumbnail at `thumb_downsample`, HSV saturation above max(Otsu, mask_sat_thr),
minus saturated pen-ink hues, small holes closed. Tiles: square fields of tile_px * target
um/px read at level 0 (native ~0.25 um/px) and resized to tile_px. Tile QC: mask coverage,
Laplacian-variance focus, ink fraction."""
from __future__ import annotations
import numpy as np
import cv2
import openslide


def mpp(slide: openslide.OpenSlide) -> float:
    v = slide.properties.get("openslide.mpp-x") or slide.properties.get("aperio.MPP")
    return float(v) if v else 0.25


def tissue_mask(slide, cfg) -> tuple[np.ndarray, float]:
    ds = cfg["thumb_downsample"]
    W, H = slide.dimensions
    th = np.asarray(slide.get_thumbnail((W // ds, H // ds)).convert("RGB"))
    hsv = cv2.cvtColor(th, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat = hsv[..., 0] / 180.0, hsv[..., 1] / 255.0
    otsu, _ = cv2.threshold((sat * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    m = sat > max(otsu / 255.0, cfg["mask_sat_thr"])
    ink = np.zeros_like(m)
    for lo, hi in cfg["pen_hue_ranges"]:
        ink |= (hue >= lo) & (hue <= hi) & (sat > cfg["pen_sat_thr"])
    m &= ~ink
    m = cv2.morphologyEx(m.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)).astype(bool)
    return m, W / th.shape[1]


def tile_grid(slide, mask: np.ndarray, scale: float, cfg) -> list[tuple[int, int, int]]:
    """Level-0 (x, y, size) of tiles whose mask coverage >= min_tile_tissue."""
    size0 = int(round(cfg["tile_px"] * cfg["target_um_per_px"] / mpp(slide)))
    W, H = slide.dimensions
    out = []
    for y in range(0, H - size0, size0):
        for x in range(0, W - size0, size0):
            y0, x0 = int(y / scale), int(x / scale)
            y1, x1 = max(y0 + 1, int((y + size0) / scale)), max(x0 + 1, int((x + size0) / scale))
            if mask[y0:y1, x0:x1].mean() >= cfg["min_tile_tissue"]:
                out.append((x, y, size0))
    return out


def read_tile(slide, x: int, y: int, size0: int, px: int) -> np.ndarray:
    t = np.asarray(slide.read_region((x, y), 0, (size0, size0)).convert("RGB"))
    return cv2.resize(t, (px, px), interpolation=cv2.INTER_AREA)


def tile_qc(t: np.ndarray, cfg) -> bool:
    g = cv2.cvtColor(t, cv2.COLOR_RGB2GRAY)
    if cv2.Laplacian(g, cv2.CV_64F).var() < cfg["min_blur_var"]:
        return False
    hsv = cv2.cvtColor(t, cv2.COLOR_RGB2HSV).astype(np.float32)
    hue, sat = hsv[..., 0] / 180.0, hsv[..., 1] / 255.0
    ink = np.zeros(hue.shape, bool)
    for lo, hi in cfg["pen_hue_ranges"]:
        ink |= (hue >= lo) & (hue <= hi) & (sat > cfg["pen_sat_thr"])
    return ink.mean() < 0.05
