"""Step 1 — layer-level attribution of the outlier-position phenomenon.

Goal: figure out which layers in Qwen2.5-0.5B contribute most to the
high-norm residual stream at positions 0–3. This is the cheap first pass
before head-level dissection (Step 2) and ablation (Step 3).

Method
------
- Run a held-out text batch through the base model with output_hidden_states=True.
- We get hidden_states as a tuple of length n_layers+1: hidden_states[0] is the
  embedding output (input to layer 0), hidden_states[i+1] is the residual stream
  after layer i.
- For each layer i, compute the per-position L2 norm of (hidden_states[i+1] −
  hidden_states[i]). This is the magnitude of what layer i writes into the
  residual stream at each position.
- Aggregate over many sequences; report mean norms at positions 0, 1, 2, 3,
  4, and a mid-sequence average (positions 16–127).

Output
------
A JSON report with the per-layer per-position-bucket norm contributions, and
a markdown summary table.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


@torch.no_grad()
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.out.mkdir(parents=True, exist_ok=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.float16
    ).to(device).eval()
    n_layers = model.config.num_hidden_layers
    d_model = model.config.hidden_size
    print(f"[setup] {args.model}: {n_layers} layers, d_model={d_model}")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts: list[str] = []
    for row in ds:
        if len(texts) >= args.n_sequences:
            break
        texts.append(row["text"])

    # Accumulators: for each layer index and each position, the squared norm
    # of the layer's contribution. We use squared norm so we can average then
    # sqrt — keeps the math clean.
    per_pos_contrib = torch.zeros(n_layers, args.seq_len, dtype=torch.float64)
    per_pos_resid = torch.zeros(n_layers + 1, args.seq_len, dtype=torch.float64)
    counts = torch.zeros(args.seq_len, dtype=torch.float64)

    for start in range(0, args.n_sequences, args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=args.seq_len).to(device)
        out = model(**enc, output_hidden_states=True)
        hs = out.hidden_states  # tuple of length n_layers+1, each (B, L, D)
        mask = enc.attention_mask  # (B, L)

        # Per-position residual stream norm at every layer boundary
        for i in range(n_layers + 1):
            n_sq = hs[i].float().pow(2).sum(dim=-1)  # (B, L)
            valid_n_sq = n_sq * mask.float()
            per_pos_resid[i] += valid_n_sq.sum(dim=0).double().cpu()

        # Per-position layer-write norm = norm(hs[i+1] - hs[i])
        for i in range(n_layers):
            contrib = hs[i + 1].float() - hs[i].float()
            n_sq = contrib.pow(2).sum(dim=-1)
            valid_n_sq = n_sq * mask.float()
            per_pos_contrib[i] += valid_n_sq.sum(dim=0).double().cpu()

        counts += mask.sum(dim=0).double().cpu()

    # Per-position normalize: mean of squared norm at each position
    eps = 1e-8
    counts = counts.clamp_min(eps)
    mean_sq_resid = per_pos_resid / counts.unsqueeze(0)   # (n_layers+1, L)
    mean_sq_contrib = per_pos_contrib / counts.unsqueeze(0)  # (n_layers, L)
    rms_resid = mean_sq_resid.sqrt()
    rms_contrib = mean_sq_contrib.sqrt()

    # Position buckets
    pos_buckets = {
        "pos_0": [0],
        "pos_1": [1],
        "pos_2": [2],
        "pos_3": [3],
        "pos_4_to_7": list(range(4, 8)),
        "pos_8_to_15": list(range(8, 16)),
        "pos_16_to_127": list(range(16, min(128, args.seq_len))),
    }

    def bucket_avg(t: torch.Tensor, idxs: list[int]) -> float:
        return float(t[..., idxs].mean(dim=-1).item()) if t.ndim == 1 \
            else float(t[..., idxs].mean(dim=-1).mean().item())

    # Build report
    report: dict = {
        "model": args.model,
        "n_sequences": args.n_sequences,
        "seq_len": args.seq_len,
        "n_layers": n_layers,
        "d_model": d_model,
        "residual_rms_norm": {},
        "layer_write_rms_norm": {},
    }
    for bname, idxs in pos_buckets.items():
        report["residual_rms_norm"][bname] = [
            float(rms_resid[i, idxs].mean().item()) for i in range(n_layers + 1)
        ]
        report["layer_write_rms_norm"][bname] = [
            float(rms_contrib[i, idxs].mean().item()) for i in range(n_layers)
        ]

    (args.out / "layer_attribution.json").write_text(json.dumps(report, indent=2))

    # Print summary table to console
    print("\n=== Residual stream RMS norm by layer (rows) and position (cols) ===")
    header = " layer "
    for b in pos_buckets:
        header += f"| {b:>14s} "
    print(header)
    for i in range(n_layers + 1):
        label = "in" if i == 0 else f"L{i-1}o"
        row = f"  {label:>4s} "
        for b, idxs in pos_buckets.items():
            v = rms_resid[i, idxs].mean().item()
            row += f"| {v:>14.2f} "
        print(row)

    print("\n=== Per-layer write RMS norm by layer (rows) and position (cols) ===")
    print(header)
    for i in range(n_layers):
        row = f"   L{i:>2d} "
        for b, idxs in pos_buckets.items():
            v = rms_contrib[i, idxs].mean().item()
            row += f"| {v:>14.2f} "
        print(row)

    # Headline metric: ratio of pos-0 residual RMS to mid-sequence RMS, at the final layer
    final_pos0 = rms_resid[-1, 0].item()
    final_mid = rms_resid[-1, list(range(16, min(128, args.seq_len)))].mean().item()
    ratio = final_pos0 / final_mid if final_mid > 0 else float("inf")
    print(f"\n[headline] final-layer pos-0 RMS / mid-sequence RMS = {ratio:.1f}x")

    print(f"[save] wrote {args.out}/layer_attribution.json")


if __name__ == "__main__":
    main()
