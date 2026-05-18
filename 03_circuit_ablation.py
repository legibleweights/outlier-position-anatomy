"""Step 3 — circuit-level ablation + perplexity preservation test.

Step 2 identified the Qwen2.5-0.5B outlier-position-0 circuit:
  - Writers: L2 attention head 5, L2 MLP, L3 MLP
  - Eraser:  L21 MLP

This script verifies the circuit by jointly ablating the writers and
measuring two things:

1. **Did we kill the outlier?** Position-0 residual RMS at every layer
   boundary, with and without the writer ablation.
2. **Did we preserve the model?** Mean per-token cross-entropy loss on
   a held-out text slice, with and without the writer ablation.

If the writers are the right circuit, ablating them should:
- Eliminate the position-0 outlier (RMS at position 0 should match
  mid-sequence RMS at every layer)
- Leave perplexity essentially unchanged (within ~0.1 nats at most)

For comparison we also ablate just the writers individually and just the
eraser, to see what each does in isolation.
"""
from __future__ import annotations

import argparse
import json
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.nn.functional as F
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


def install_head_zero(model, layer_idx, head_idx):
    layer = model.model.layers[layer_idx]
    n_heads = model.config.num_attention_heads
    head_dim = model.config.hidden_size // n_heads
    o_proj = layer.self_attn.o_proj

    def pre_hook(module, args):
        x = args[0].clone()
        x[:, :, head_idx * head_dim : (head_idx + 1) * head_dim] = 0.0
        return (x,) + args[1:]

    return o_proj.register_forward_pre_hook(pre_hook)


def install_mlp_zero(model, layer_idx):
    layer = model.model.layers[layer_idx]
    mlp = layer.mlp

    def hook(module, inputs, output):
        return torch.zeros_like(output)

    return mlp.register_forward_hook(hook)


@contextmanager
def ablation(model, spec: list[tuple]):
    """spec: list of ('head', layer, head_idx) or ('mlp', layer, None)."""
    hooks = []
    try:
        for kind, L, h in spec:
            if kind == "head":
                hooks.append(install_head_zero(model, L, h))
            elif kind == "mlp":
                hooks.append(install_mlp_zero(model, L))
        yield
    finally:
        for h in hooks:
            h.remove()


@torch.no_grad()
def measure_residual_profile(model, enc):
    """Return (n_layers+1, seq_len) tensor of per-position RMS norm."""
    out = model(**enc, output_hidden_states=True)
    hs = out.hidden_states
    mask = enc.attention_mask
    profile = []
    for h in hs:
        n_sq = h.float().pow(2).sum(dim=-1)  # (B, L)
        valid_n_sq = (n_sq * mask.float()).sum(dim=0)
        count = mask.sum(dim=0).clamp_min(1)
        rms = (valid_n_sq / count).sqrt()
        profile.append(rms.cpu())
    return torch.stack(profile, dim=0)  # (n_layers+1, L)


@torch.no_grad()
def measure_perplexity(model, enc, chunk: int = 16):
    """Chunked perplexity: process `chunk` sequences at a time to avoid OOM
    on small GPUs (logits at full vocab × full seq_len gets large)."""
    total_loss = 0.0
    total_count = 0
    n = enc.input_ids.shape[0]
    for i in range(0, n, chunk):
        sub = {k: v[i:i + chunk] for k, v in enc.items()}
        out = model(**sub)
        # Compute CE in fp16 logits → cast small slices to float for stability
        logits = out.logits[:, :-1, :].contiguous()
        targets = sub["input_ids"][:, 1:].contiguous()
        mask = sub["attention_mask"][:, 1:].bool()
        loss = F.cross_entropy(
            logits.float().view(-1, logits.size(-1)),
            targets.view(-1),
            reduction="none",
        ).view(targets.shape)
        valid = loss[mask]
        total_loss += valid.sum().item()
        total_count += valid.numel()
    return total_loss / total_count if total_count > 0 else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=256)
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
    print(f"[setup] {args.model}: {model.config.num_hidden_layers} layers")

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(args.n_sequences), ds)]
    enc = tok(texts, return_tensors="pt", padding="max_length",
               truncation=True, max_length=args.seq_len).to(device)

    # Define the candidate ablations (based on Step 2 findings on Qwen2.5-0.5B)
    interventions = {
        "baseline": [],
        "ablate_L2H5": [("head", 2, 5)],
        "ablate_L2_mlp": [("mlp", 2, None)],
        "ablate_L3_mlp": [("mlp", 3, None)],
        "ablate_L21_mlp": [("mlp", 21, None)],
        "ablate_writers": [
            ("head", 2, 5), ("mlp", 2, None), ("mlp", 3, None),
        ],
    }

    results: dict = {"model": args.model, "n_sequences": args.n_sequences,
                      "seq_len": args.seq_len, "interventions": {}}

    for name, spec in interventions.items():
        print(f"\n=== intervention: {name} ===")
        with ablation(model, spec):
            profile = measure_residual_profile(model, enc)
            ce = measure_perplexity(model, enc)

        # Headline numbers from the profile
        n_layers = profile.shape[0] - 1
        mid = list(range(16, min(128, args.seq_len)))
        pos0_after_L2 = profile[3, 0].item()
        pos0_after_L21 = profile[22, 0].item()
        pos0_final = profile[-1, 0].item()
        mid_final = profile[-1, mid].mean().item()
        ratio = pos0_final / mid_final if mid_final > 0 else float("inf")

        results["interventions"][name] = {
            "ce_loss": ce,
            "pos0_residual_after_L2": pos0_after_L2,
            "pos0_residual_after_L21": pos0_after_L21,
            "pos0_residual_final": pos0_final,
            "mid_residual_final": mid_final,
            "pos0_to_mid_ratio_final": ratio,
            "profile": profile.tolist(),
        }
        print(f"  CE loss:                       {ce:.4f}")
        print(f"  pos-0 RMS after L2:            {pos0_after_L2:.2f}")
        print(f"  pos-0 RMS after L21:           {pos0_after_L21:.2f}")
        print(f"  pos-0 RMS at output:           {pos0_final:.2f}")
        print(f"  mid-sequence RMS at output:    {mid_final:.2f}")
        print(f"  pos-0 / mid ratio at output:   {ratio:.2f}")

    # Compute Δ-CE relative to baseline for each intervention
    base_ce = results["interventions"]["baseline"]["ce_loss"]
    print("\n=== summary: Δ-CE relative to baseline ===")
    print(f"  baseline CE: {base_ce:.4f}")
    for name, r in results["interventions"].items():
        if name == "baseline":
            continue
        d = r["ce_loss"] - base_ce
        pos0_ratio_change = (r["pos0_to_mid_ratio_final"] -
                              results["interventions"]["baseline"]["pos0_to_mid_ratio_final"])
        print(f"  {name:>20s}: ΔCE = {d:+.4f}  "
              f"pos0/mid ratio: {r['pos0_to_mid_ratio_final']:.2f}")

    (args.out / "circuit_ablation.json").write_text(json.dumps(results, indent=2))
    print(f"\n[save] wrote {args.out}/circuit_ablation.json")


if __name__ == "__main__":
    main()
