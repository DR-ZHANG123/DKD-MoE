"""Illustrated icons for Fig. 1, generated once with an image model and stored as transparent PNGs.

Only pictograms are generated (no text, no numbers); layout, arrows, the model diagram, labels and equations are
drawn in Fig1_framework.py, so all text in the figure stays vector and editable. The prompts and the files are kept
in figures/source_data/Fig1_icons/, so the figure is redrawn from them without calling the model again.

    IMAGE_API_BASE=... GPTIMAGE_KEY=... python fig1_icons_generate.py [--only kidney cell ...]
"""
from __future__ import annotations
import argparse, base64, json, os, urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

OUT = Path(__file__).resolve().parents[2] / "figures" / "source_data" / "Fig1_icons"
BASE = os.environ.get("IMAGE_API_BASE", "")          # OpenAI-compatible image endpoint
STYLE = ("Single scientific icon in BioRender style: clean illustration with soft flat shading and gentle highlights, "
         "thin dark outline, muted professional palette, centred with a small margin, no text, no letters, no numbers, "
         "no labels, no shadow on the ground, isolated on a fully transparent background.")
ICONS = {
    "kidney": "An anatomical human kidney, front view, muted red-pink, renal pelvis and ureter visible.",
    "glucose": "A single drop of blood with a small white glucose hexagon ring inside it, representing diabetes.",
    "bp_cuff": "A blood pressure cuff with a round analogue gauge, representing hypertension, steel blue and grey.",
    "pt_cell": "A single cuboidal proximal tubule epithelial cell with a brush border of microvilli on top and a round "
               "purple nucleus, soft pink cytoplasm.",
    "nucleus": "A single isolated cell nucleus, round, lavender purple, with faint chromatin texture and a nucleolus.",
    "chromatin": "A cell nucleus cut open showing DNA chromatin fibres, some wrapped around nucleosomes and some open "
                 "accessible stretches highlighted in teal.",
    "visium": "A glass microscope slide with a square capture area containing a regular grid of small round spots "
              "over a pink and purple stained kidney tissue section.",
    "he_patch": "A square microscopy image tile of haematoxylin and eosin stained kidney cortex: tubules in pink with "
                "purple nuclei, rounded corners.",
    "tubule": "A cross-section of a kidney proximal tubule: a ring of cuboidal epithelial cells with purple nuclei "
              "around a lumen with brush border.",
    "urine_cup": "A clear plastic urine specimen cup with a screw lid, pale yellow liquid, representing albuminuria.",
    "dna": "A short DNA double helix segment, blue and purple strands with coloured base pairs.",
    "slides_stack": "A small stack of four glass microscope slides with stained tissue sections, slightly fanned, "
                    "representing independent tissue cohorts.",
    "microscope": "A laboratory light microscope, side view, grey and blue.",
    "patients": "Three simple human bust silhouettes side by side, neutral grey-blue, representing study participants.",
    "neural_net": "A small stylised neural network: three columns of round nodes connected by thin lines, blue and purple.",
}


def generate(name: str, key: str) -> str:
    prompt = f"{STYLE} Subject: {ICONS[name]}"
    body = json.dumps({"model": "gpt-image-2", "prompt": prompt, "size": "1024x1024", "background": "transparent",
                       "quality": "high", "n": 1}).encode()
    req = urllib.request.Request(f"{BASE}/images/generations", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        d = json.load(r)
    (OUT / f"{name}.png").write_bytes(base64.b64decode(d["data"][0]["b64_json"]))
    return prompt


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--only", nargs="*")
    a = ap.parse_args(); key = os.environ["GPTIMAGE_KEY"]; OUT.mkdir(parents=True, exist_ok=True)
    names = a.only or list(ICONS)
    with ThreadPoolExecutor(6) as ex:
        prompts = dict(zip(names, ex.map(lambda n: generate(n, key), names)))
    rec = OUT / "prompts.json"
    old = json.loads(rec.read_text()) if rec.exists() else {}
    rec.write_text(json.dumps({**old, **prompts}, indent=1))
    print("generated:", ", ".join(names))


if __name__ == "__main__":
    main()
