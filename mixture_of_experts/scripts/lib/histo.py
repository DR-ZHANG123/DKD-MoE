"""Histology helpers: spot-centred patch extraction, interpretable H&E features, fixed encoder.

Patch geometry: a square field of `field_um` micrometres centred on the spot, cropped at native
resolution (um/px from spot_diameter_fullres = 55 um) and resized to `patch_px` pixels.
Interpretable features use Ruifrok-Johnston colour deconvolution (skimage.color.rgb2hed)."""
from __future__ import annotations
import numpy as np
import cv2
from skimage.color import rgb2hed

SPOT_UM = 55.0


def crop(img: np.ndarray, r: float, c: float, half: int) -> np.ndarray | None:
    r, c = int(round(r)), int(round(c))
    if r - half < 0 or c - half < 0 or r + half > img.shape[0] or c + half > img.shape[1]:
        return None
    p = np.asarray(img[r - half:r + half, c - half:c + half])
    return p[..., :3] if p.ndim == 3 else np.repeat(p[..., None], 3, -1)


def resize(p: np.ndarray, px: int) -> np.ndarray:
    return cv2.resize(p, (px, px), interpolation=cv2.INTER_AREA if p.shape[0] > px else cv2.INTER_CUBIC)


def background_rgb(img: np.ndarray, pts, half: int) -> np.ndarray:
    """Median RGB (0-1) of patches at off-tissue positions = the section's glass/background colour."""
    meds = [np.median(p.reshape(-1, 3), 0) for p in (crop(img, r, c, half) for r, c in pts) if p is not None]
    return (np.median(np.vstack(meds), 0) / 255.0) if meds else np.array([0.95, 0.95, 0.95])


def he_features(p: np.ndarray, cfg: dict, bg: np.ndarray | None = None) -> dict:
    """p uint8 RGB. Tissue = pixels whose colour departs from the section background by more than
    tissue_delta (Euclidean, RGB 0-1) and that are darker than the background; bg defaults to white."""
    f = p.astype(np.float32) / 255.0
    bg = np.array([0.95, 0.95, 0.95]) if bg is None else np.asarray(bg, np.float32)
    gray = f.mean(-1)
    dist = np.sqrt(((f - bg) ** 2).sum(-1))
    tissue = (dist > cfg["tissue_delta"]) & (gray < bg.mean() - cfg["tissue_min_darker"])
    hed = rgb2hed(np.clip(f, 1e-6, 1))
    h, e = hed[..., 0], hed[..., 1]
    nuc = tissue & (h > cfg["hema_thr"])
    n_lab, _, st, _ = cv2.connectedComponentsWithStats(nuc.astype(np.uint8), connectivity=8)
    areas = st[1:, cv2.CC_STAT_AREA] if n_lab > 1 else np.array([])
    min_a = cfg["nucleus_min_px"]
    t = max(int(tissue.sum()), 1)
    return {
        "tissue_frac": float(tissue.mean()),
        "white_frac": float((~tissue).mean()),                            # lumina, interstitial gaps
        "hema_frac": float(nuc.sum() / t),
        "nuclei_per_1e4px_tissue": float((areas >= min_a).sum() / t * 1e4),
        "eosin_mean": float(e[tissue].mean()) if tissue.any() else np.nan,
        "hema_mean": float(h[tissue].mean()) if tissue.any() else np.nan,
        "eosin_frac": float((tissue & (e > cfg["eosin_thr"])).sum() / t),
    }


class Encoder:
    """Fixed pathology foundation encoder (CLS token of a ViT, HF transformers)."""

    def __init__(self, name: str, revision: str, device: str = "cuda"):
        import torch
        from transformers import AutoModel
        self.torch = torch
        self.model = AutoModel.from_pretrained(name, revision=revision).eval().to(device)
        self.device = device
        self.mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
        self.std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    def __call__(self, patches: np.ndarray, bs: int = 256) -> np.ndarray:
        torch = self.torch
        out = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
            for s in range(0, len(patches), bs):
                x = torch.from_numpy(patches[s:s + bs]).to(self.device).permute(0, 3, 1, 2).float() / 255.0
                x = (x - self.mean) / self.std
                out.append(self.model(pixel_values=x).last_hidden_state[:, 0].float().cpu().numpy())
        return np.vstack(out) if out else np.zeros((0, self.model.config.hidden_size), np.float32)
