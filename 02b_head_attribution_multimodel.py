"""Multi-architecture per-head attribution at suspect layers.

Same intent as 02_head_attribution.py but dispatches on model architecture
so it works on Qwen2 (`model.model.layers[i].self_attn.o_proj`), GPT-2
(`model.transformer.h[i].attn.c_proj`), and Pythia / GPT-NeoX
(`model.gpt_neox.layers[i].attention.dense`).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_layer(model, idx):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return model.model.layers[idx]
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return model.transformer.h[idx]
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers[idx]
    raise ValueError(f"Unknown architecture: {type(model).__name__}")


def get_attn(layer):
    for name in ["self_attn", "attn", "attention"]:
        if hasattr(layer, name):
            return getattr(layer, name)
    raise ValueError("No attention submodule found")


def get_o_proj(attn):
    for name in ["o_proj", "dense", "c_proj"]:
        if hasattr(attn, name):
            return getattr(attn, name)
    raise ValueError("No output projection found")


def get_mlp(layer):
    return layer.mlp


@torch.no_grad()
def measure_pos0_residual_after_layer(
    model, enc, target_layer_idx, head_to_zero=None, mlp_to_zero=False,
):
    hooks = []
    if head_to_zero is not None:
        attn = get_attn(get_layer(model, target_layer_idx))
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads
        o_proj = get_o_proj(attn)

        def o_proj_pre_hook(module, args):
            x = args[0]
            x = x.clone()
            x[:, :, head_to_zero * head_dim : (head_to_zero + 1) * head_dim] = 0.0
            return (x,) + args[1:]

        hooks.append(o_proj.register_forward_pre_hook(o_proj_pre_hook))

    if mlp_to_zero:
        mlp = get_mlp(get_layer(model, target_layer_idx))
        hooks.append(mlp.register_forward_hook(
            lambda module, inputs, output: torch.zeros_like(output)
        ))

    try:
        out = model(**enc, output_hidden_states=True)
        resid = out.hidden_states[target_layer_idx + 1]
        pos0_norm = resid[:, 0, :].float().pow(2).sum(dim=-1).sqrt().mean().item()
    finally:
        for h in hooks:
            h.remove()
    return pos0_norm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--target-layers", type=int, nargs="+", required=True)
    ap.add_argument("--n-sequences", type=int, default=64)
    ap.add_argument("--seq-len", type=int, default=128)
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
    n_heads = model.config.num_attention_heads
    print(f"[setup] {args.model}: {model.config.num_hidden_layers} layers, "
          f"{n_heads} attention heads")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(args.n_sequences), ds)]
    enc = tok(texts, return_tensors="pt", padding="max_length",
               truncation=True, max_length=args.seq_len).to(device)

    baselines = {}
    for L in args.target_layers:
        baselines[L] = measure_pos0_residual_after_layer(model, enc, L)
        print(f"[baseline] layer {L}: pos-0 residual RMS = {baselines[L]:.2f}")

    results: dict = {
        "model": args.model, "n_sequences": args.n_sequences,
        "seq_len": args.seq_len, "baselines": baselines,
        "per_head_pos0_norm_after_ablation": {},
        "mlp_pos0_norm_after_ablation": {},
    }

    for L in args.target_layers:
        print(f"\n=== Layer {L} per-head ablation ===")
        per_head = []
        for h in range(n_heads):
            norm = measure_pos0_residual_after_layer(
                model, enc, target_layer_idx=L, head_to_zero=h,
            )
            per_head.append(norm)
        results["per_head_pos0_norm_after_ablation"][L] = per_head

        ranked = sorted(enumerate(per_head), key=lambda x: x[1])
        print(f"  baseline: {baselines[L]:.2f}")
        print("  top 5 heads by drop in pos-0 residual norm when ablated:")
        for h_idx, norm in ranked[:5]:
            drop = baselines[L] - norm
            pct = 100 * drop / baselines[L] if baselines[L] > 0 else 0.0
            print(f"    head {h_idx:>2d}: pos-0 RMS = {norm:.2f} (drop {drop:.2f}, {pct:.1f}%)")

        mlp_norm = measure_pos0_residual_after_layer(
            model, enc, target_layer_idx=L, mlp_to_zero=True,
        )
        results["mlp_pos0_norm_after_ablation"][L] = mlp_norm
        mlp_drop = baselines[L] - mlp_norm
        mlp_pct = 100 * mlp_drop / baselines[L] if baselines[L] > 0 else 0.0
        print(f"  MLP ablation: pos-0 RMS = {mlp_norm:.2f} "
              f"(drop {mlp_drop:.2f}, {mlp_pct:.1f}%)")

    (args.out / "head_attribution.json").write_text(json.dumps(results, indent=2))
    print(f"\n[save] wrote {args.out}/head_attribution.json")


if __name__ == "__main__":
    main()
