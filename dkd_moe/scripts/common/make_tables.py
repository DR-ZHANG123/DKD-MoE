"""make_tables: supplementary-table inputs that are read from configs and result tables (no recomputation).

Outputs dkd_moe/results/dl_tables/{hyperparameters.tsv, participants.tsv, participant_summary.tsv} + PROVENANCE.json."""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cdl import CDL, ROOT, load_cfg, repro  # noqa: E402

OUT = CDL / "results" / "dl_tables"


def hyperparameters() -> pd.DataFrame:
    e, a = load_cfg("expr"), load_cfg("align")
    sel = json.loads((CDL / "results/expr_cv/selection.json").read_text())
    r2 = e["readout_stability"]["stability"]
    m, t, cv = e["model"], e["train"], e["cv"]
    rows = [
        ("Mixture-of-experts", "Genes", e["prep"]["n_hvg"], "fixed"),
        ("Mixture-of-experts", "Encoder layers x units", f"{m['n_backbone_layers']} x {m['n_hidden']}", "fixed"),
        ("Mixture-of-experts", "Expert hidden units", m["expert_hidden"], "fixed"),
        ("Mixture-of-experts", "Latent dimension", m["n_latent"], "fixed"),
        ("Mixture-of-experts", "Experts routed per cell (top-k)", m["top_k"], "fixed"),
        ("Mixture-of-experts", "Number of experts K", sel["K"], f"selected from {cv['K_grid']}"),
        ("Mixture-of-experts", "Condition weight", sel["w_cond"], f"selected from {cv['w_cond_grid']}"),
        ("Mixture-of-experts", "Load-balancing weight", t["w_balance"], "compared on held-out participants of one partition {10, 50}"),
        ("Mixture-of-experts", "Technology discriminator weight", t["w_adv"], "compared on held-out participants of one partition {5, 20}"),
        ("Mixture-of-experts", "Gradient-reversal factor", t["grl_lambda"], "fixed"),
        ("Mixture-of-experts", "KL weight (warm-up epochs)", f"{t['w_kl']} ({t['kl_warmup_epochs']})", "fixed"),
        ("Mixture-of-experts", "Dropout", m["dropout"], "fixed"),
        ("Mixture-of-experts", "Optimizer, learning rate, weight decay", f"AdamW, {t['lr']}, {t['weight_decay']}", "fixed"),
        ("Mixture-of-experts", "Batch size", t["batch_size"], "fixed"),
        ("Mixture-of-experts", "Cells per participant per epoch (maximum)", t["cells_per_participant_cap"], "fixed"),
        ("Mixture-of-experts", "Epochs: single-expert start / mixture", f"{r2['warm_epochs']} / {r2['moe_epochs']}", "fixed"),
        ("Mixture-of-experts", "Expert initialization noise", r2["expert_init_noise"], "fixed"),
        ("Mixture-of-experts", "Gate noise", "annealed to 0" if r2["anneal_gate_noise"] else "constant", "fixed"),
        ("Mixture-of-experts", "Random initializations", len(e["final"]["seeds"]), "fixed"),
        ("Mixture-of-experts", "Integrated-gradient steps", e["interpret"]["ig_steps"], "fixed"),
        ("Mixture-of-experts", "Signature size (up / down)", f"{e['transfer']['n_up']} / {e['transfer']['n_down']}", "fixed"),
        ("Histology alignment", "Image embedding", f"phikon-v2 (weights fixed), {a['model']['emb_dim']}", "fixed"),
        ("Histology alignment", "Image head hidden units", a["model"]["img_hidden"], "selected on inner validation"),
        ("Histology alignment", "Expression head hidden units", a["model"]["expr_hidden"], "selected on inner validation"),
        ("Histology alignment", "Projection dimension", a["model"]["proj_dim"], "fixed"),
        ("Histology alignment", "Initial temperature / maximum inverse temperature",
         f"{a['model']['init_temperature']} / {a['model']['max_logit_scale']}", "fixed"),
        ("Histology alignment", "Negatives", a["train"]["primary_batch_mode"], f"compared {a['train']['batch_modes']}"),
        ("Histology alignment", "Batch size, learning rate, weight decay",
         f"{a['train']['batch_size']}, {a['train']['lr']}, {a['train']['weight_decay']}", "fixed"),
        ("Histology alignment", "Maximum epochs (early-stopping patience)",
         f"{a['train']['epochs']} ({a['train']['patience']})", "fixed"),
        ("Histology alignment", "Random initializations", len(a["seeds"]), "fixed"),
    ]
    return pd.DataFrame(rows, columns=["component", "parameter", "value", "how chosen"])


def participants() -> tuple[pd.DataFrame, pd.DataFrame]:
    s = pd.read_csv(CDL / "config/splits.tsv", sep="\t", dtype={"participant": str})
    s = s.rename(columns={"fold": "partition"})
    flags = [c for c in s.columns if c.startswith("in_")]
    summ = s.groupby("group")[flags].sum().astype(int)
    summ.insert(0, "participants", s.groupby("group").size())
    return s, summ.reset_index()


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    hp = hyperparameters(); hp.to_csv(OUT / "hyperparameters.tsv", sep="\t", index=False)
    p, ps = participants()
    p.to_csv(OUT / "participants.tsv", sep="\t", index=False)
    ps.to_csv(OUT / "participant_summary.tsv", sep="\t", index=False)
    outs = [OUT / f for f in ["hyperparameters.tsv", "participants.tsv", "participant_summary.tsv"]]
    repro.write_provenance("../dkd_moe/results/dl_tables",
                           [CDL / "config/expr.yaml", CDL / "config/align.yaml", CDL / "config/splits.tsv",
                            CDL / "results/expr_cv/selection.json"], outs, seed=None)
    print(hp.to_string(index=False)); print(ps.to_string(index=False))


if __name__ == "__main__":
    main()
