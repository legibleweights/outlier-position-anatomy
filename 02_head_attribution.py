"""Step 2 — per-head attribution at the smoking-gun layers.

Step 1 found that on Qwen2.5-0.5B:
  - Layer 2 writes ~750 RMS to position 0 (vs ~5 elsewhere)
  - Layer 3 writes ~900 RMS to position 0
  - Layer 21 writes ~1630 RMS to position 0 (the "erase" / cancellation)

Now we ask: which specific attention heads within those layers (and layer 5,
which also showed a smaller spike) are responsible? And does the MLP at each
layer contribute, or is it pure attention?

Method — per-head ablation at the source.
We hook into self_attn's value-projected outputs, zero out one head's
contribution at a time, and measure the resulting change in position-0
residual stream norm at the next layer. The head whose ablation reduces
position-0 norm the most is the responsible head.

We also separately ablate the entire MLP at each suspect layer to see how
much of the contribution is attention vs MLP.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def get_attention(layer):
    return layer.self_attn


def get_mlp(layer):
    return layer.mlp


@torch.no_grad()
def measure_pos0_residual_after_layer(
    model, enc, target_layer_idx, head_to_zero=None, mlp_to_zero=False,
):
    """Return RMS of residual at position 0 *after* layer target_layer_idx,
    optionally with one attention head zeroed or the MLP zeroed at that layer.
    """
    hooks = []
    if head_to_zero is not None:
        attn = get_attention(model.model.layers[target_layer_idx])
        n_heads = model.config.num_attention_heads
        head_dim = model.config.hidden_size // n_heads
        num_kv_heads = getattr(model.config, "num_key_value_heads", n_heads)
        gqa_groups = n_heads // num_kv_heads

        def attn_hook(module, inputs, output):
            # Qwen2's self_attn returns a tuple where output[0] is the merged
            # attention output: shape (B, L, hidden). To zero one head, we
            # know each head contributes a slice of `hidden` that gets
            # projected by W_O. The simplest local intervention: zero the
            # appropriate columns of `o_proj`'s input. We do that by
            # patching the o_proj input via a separate hook on o_proj.
            return output

        # Better approach: directly hook o_proj's input. attn_output is
        # `head_outputs @ W_O.T`, where head_outputs has shape (B, L, n_heads*head_dim).
        # The per-head slice is [:, :, h*head_dim:(h+1)*head_dim].
        o_proj = attn.o_proj

        def o_proj_pre_hook(module, args):
            # args = (x,) where x has shape (B, L, n_heads * head_dim)
            x = args[0]
            x = x.clone()
            x[:, :, head_to_zero * head_dim : (head_to_zero + 1) * head_dim] = 0.0
            return (x,) + args[1:]

        hooks.append(o_proj.register_forward_pre_hook(o_proj_pre_hook))

    if mlp_to_zero:
        mlp = get_mlp(model.model.layers[target_layer_idx])

        def mlp_hook(module, inputs, output):
            return torch.zeros_like(output)

        hooks.append(mlp.register_forward_hook(mlp_hook))

    try:
        out = model(**enc, output_hidden_states=True)
        # hidden_states[target_layer_idx + 1] = residual stream after target layer
        resid = out.hidden_states[target_layer_idx + 1]   # (B, L, D)
        pos0_norm = resid[:, 0, :].float().pow(2).sum(dim=-1).sqrt().mean().item()
    finally:
        for h in hooks:
            h.remove()
    return pos0_norm


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--target-layers", type=int, nargs="+", default=[2, 3, 5, 21])
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
          f"{n_heads} attention heads, d_model={model.config.hidden_size}")

    # Load one fixed batch of held-out text (held same across all ablations
    # so the contribution measurements are comparable).
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(args.n_sequences), ds)]
    enc = tok(texts, return_tensors="pt", padding="max_length",
               truncation=True, max_length=args.seq_len).to(device)

    # Baseline: position-0 residual norm after each target layer, with NO ablation
    baselines = {}
    for L in args.target_layers:
        baselines[L] = measure_pos0_residual_after_layer(model, enc, L)
        print(f"[baseline] layer {L}: pos-0 residual RMS = {baselines[L]:.2f}")

    results: dict = {
        "model": args.model,
        "n_sequences": args.n_sequences,
        "seq_len": args.seq_len,
        "baselines": baselines,
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

        # Rank
        ranked = sorted(enumerate(per_head), key=lambda x: x[1])  # ascending = "best ablation"
        print(f"  baseline: {baselines[L]:.2f}")
        print("  top 5 heads by drop in pos-0 residual norm when ablated:")
        for h_idx, norm in ranked[:5]:
            drop = baselines[L] - norm
            pct = 100 * drop / baselines[L] if baselines[L] > 0 else 0.0
            print(f"    head {h_idx:>2d}: pos-0 RMS = {norm:.2f} (drop {drop:.2f}, {pct:.1f}%)")

        # MLP ablation at the same layer
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
