"""Step 5 — what direction does the constant register point in?

v0.2 established that the position-0 register at mid-network is essentially
a fixed vector across inputs (cosine ~ 0.9999 between any two inputs'
registers). Now we ask: **what direction is it?**

Two candidate alignments:
  A) Token-input-embedding space (`model.get_input_embeddings()`):
     is the register close to some specific token's input embedding?
     If yes → the model is essentially storing "the embedding of token X"
     at position 0.
  B) Token-output (unembedding) space (`model.get_output_embeddings()`,
     i.e. lm_head.weight):
     does the register, viewed as a logit-direction, push toward predicting
     some specific token? If yes → the register is encoding "the model
     wants to predict X at position 0" at the residual-stream level.

Plus a random-Gaussian baseline so we can tell whether any apparent
alignment is real or just dimensional-collapse noise.
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
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer", type=int, required=True,
                    help="Mid-network layer to read the register from")
    ap.add_argument("--n-sequences", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--top-k", type=int, default=15)
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
    d_model = model.config.hidden_size

    # 1) Collect the constant register
    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                      split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(args.n_sequences), ds)]
    regs = []
    for start in range(0, args.n_sequences, args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=args.seq_len).to(device)
        out = model(**enc, output_hidden_states=True)
        regs.append(out.hidden_states[args.layer + 1][:, 0, :].float().cpu())
    regs = torch.cat(regs, dim=0)
    mean_reg = regs.mean(dim=0)  # (d_model,)
    print(f"[setup] model={args.model} layer={args.layer} "
          f"|mean_reg|={mean_reg.norm().item():.2f}")

    # 2) Get token-embedding matrix and lm_head (unembedding) matrix
    embed_w = model.get_input_embeddings().weight.detach().float().cpu()   # (V, d_model)
    # Output embeddings: for some models lm_head shares weights with input embeddings,
    # for others (e.g., Pythia, GPT-2) they're separate.
    lm_head = model.get_output_embeddings()
    if lm_head is not None and hasattr(lm_head, "weight"):
        unembed_w = lm_head.weight.detach().float().cpu()
    else:
        unembed_w = embed_w  # tied
    print(f"[setup] embed: {tuple(embed_w.shape)}  unembed: {tuple(unembed_w.shape)}  "
          f"tied={torch.equal(embed_w, unembed_w)}")

    # Strip pad / unused tokens for cleaner top-K (some tokenizers have many)
    vocab_size = min(embed_w.shape[0], unembed_w.shape[0])

    def top_k_aligned(direction: torch.Tensor, matrix: torch.Tensor, k: int):
        """direction: (d_model,). matrix: (V, d_model). Returns top-k (token_id, cosine, dot)."""
        v = direction / direction.norm().clamp_min(1e-8)
        norms = matrix.norm(dim=-1).clamp_min(1e-8)
        m_normed = matrix / norms.unsqueeze(-1)
        cos = m_normed @ v          # (V,)
        dot = matrix @ direction    # (V,)
        # Sort by absolute cosine to catch both "high alignment" and "high anti-alignment"
        vals, idx = cos.abs().topk(k)
        out = []
        for token_id in idx.tolist():
            out.append({
                "token_id": int(token_id),
                "token_str": tok.decode([token_id]),
                "cosine": float(cos[token_id].item()),
                "dot": float(dot[token_id].item()),
            })
        return out

    print("\n=== Top-K tokens by cosine alignment with mean register ===")

    # Input embedding alignment
    print("\n[input embeddings]")
    top_embed = top_k_aligned(mean_reg, embed_w[:vocab_size], args.top_k)
    for r in top_embed:
        print(f"  id {r['token_id']:>6d}  cos={r['cosine']:+.4f}  "
              f"dot={r['dot']:+.2f}  {r['token_str']!r}")

    # Unembedding alignment
    print("\n[unembedding (lm_head)]")
    top_unembed = top_k_aligned(mean_reg, unembed_w[:vocab_size], args.top_k)
    for r in top_unembed:
        print(f"  id {r['token_id']:>6d}  cos={r['cosine']:+.4f}  "
              f"dot={r['dot']:+.2f}  {r['token_str']!r}")

    # Random baseline
    print("\n[random Gaussian baseline]")
    rng = torch.Generator().manual_seed(0)
    rand_vec = torch.randn(d_model, generator=rng)
    rand_top = top_k_aligned(rand_vec, embed_w[:vocab_size], 5)
    print("  Top-5 cosines for a *random* direction (sanity check that alignment is meaningful):")
    for r in rand_top:
        print(f"  id {r['token_id']:>6d}  cos={r['cosine']:+.4f}  {r['token_str']!r}")

    # Save
    report = {
        "model": args.model, "layer": args.layer,
        "mean_register_norm": float(mean_reg.norm().item()),
        "d_model": d_model,
        "vocab_size": vocab_size,
        "tied_embedding": bool(torch.equal(embed_w, unembed_w)),
        "top_aligned_input_embeddings": top_embed,
        "top_aligned_unembeddings": top_unembed,
        "random_baseline_top_cos": [r["cosine"] for r in rand_top],
    }
    (args.out / "register_direction.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] wrote {args.out}/register_direction.json")


if __name__ == "__main__":
    main()
