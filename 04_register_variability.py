"""Step 4 — is the position-0 register constant or input-dependent?

The previous steps established that the position-0 residual is high-norm
(~1680 RMS on Qwen2.5-0.5B at layers 4–20) and produced by a specific
write-and-erase circuit. We don't yet know whether the register content is:

  (a) essentially constant — a fixed "BOS marker" vector that the model
      always writes to position 0 regardless of input;
  (b) input-dependent — encoding something about the input itself.

This script collects the layer-N position-0 residual across many inputs
and measures:
  - mean pairwise cosine similarity (how aligned are the registers?)
  - per-dim variance to per-dim mean ratio (how much of the register is
    constant vs varying?)
  - PCA: rank-1 explained variance ratio (does one direction dominate?)
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
    ap.add_argument("--layer", type=int, default=10)
    ap.add_argument("--n-sequences", type=int, default=512)
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

    ds = load_dataset("HuggingFaceFW/fineweb-edu", name="sample-10BT",
                       split="train", streaming=True)
    texts = [row["text"] for _, row in zip(range(args.n_sequences), ds)]

    registers = []  # one (d_model,) vector per input — the pos-0 residual after target layer
    first_token_ids = []
    for start in range(0, args.n_sequences, args.batch_size):
        batch = texts[start:start + args.batch_size]
        enc = tok(batch, return_tensors="pt", padding="max_length",
                   truncation=True, max_length=args.seq_len).to(device)
        out = model(**enc, output_hidden_states=True)
        # hidden_states[layer+1] = residual after that layer
        reg = out.hidden_states[args.layer + 1][:, 0, :].float().cpu()  # (B, d_model)
        registers.append(reg)
        first_token_ids.append(enc.input_ids[:, 0].cpu())
    registers = torch.cat(registers, dim=0)         # (N, d_model)
    first_token_ids = torch.cat(first_token_ids, dim=0)
    n, d = registers.shape
    print(f"[setup] collected {n} pos-0 residuals at layer {args.layer}, d_model={d}")

    # 1) Constant component analysis
    mean_vec = registers.mean(dim=0)
    centered = registers - mean_vec
    constant_norm = mean_vec.pow(2).sum().sqrt().item()
    variable_rms = (centered.pow(2).sum(dim=-1).mean()).sqrt().item()
    total_rms = registers.pow(2).sum(dim=-1).sqrt().mean().item()
    print(f"\n[constant vs variable]")
    print(f"  RMS norm of registers (total):      {total_rms:.2f}")
    print(f"  RMS norm of mean register:          {constant_norm:.2f}  ('constant' component)")
    print(f"  RMS norm of variable component:     {variable_rms:.2f}  ('variable' component)")
    print(f"  constant / variable ratio:          {constant_norm / variable_rms:.2f}")
    print(f"  fraction of energy in constant:     "
          f"{constant_norm**2 / (constant_norm**2 + variable_rms**2) * 100:.1f}%")

    # 2) Pairwise cosine similarity between random pairs
    import random
    random.seed(0)
    pairs = [(random.randrange(n), random.randrange(n)) for _ in range(2000)]
    pairs = [(i, j) for i, j in pairs if i != j]
    a = registers[[i for i, _ in pairs]]
    b = registers[[j for _, j in pairs]]
    cos = torch.nn.functional.cosine_similarity(a.float(), b.float(), dim=-1)
    print(f"\n[pairwise cosine similarity]")
    print(f"  mean cos(reg_i, reg_j) over {len(pairs)} random pairs: {cos.mean().item():.4f}")
    print(f"  std:                                                    {cos.std().item():.4f}")
    print(f"  min:                                                    {cos.min().item():.4f}")
    print(f"  max:                                                    {cos.max().item():.4f}")

    # 3) PCA on centered registers — how much variance explained by top-K?
    U, S, V = torch.svd(centered.float())
    eigenvalues = (S ** 2) / (n - 1)
    total_var = eigenvalues.sum().item()
    print(f"\n[PCA on centered registers]")
    print(f"  total variance:               {total_var:.2f}")
    for k in [1, 2, 5, 10, 50, 100]:
        if k <= len(eigenvalues):
            ratio = eigenvalues[:k].sum().item() / total_var
            print(f"  top-{k:3d} explained variance: {ratio*100:.1f}%")

    # 4) Does the register correlate with the first token id?
    # Quick test: for the most common first-token id, compute the mean register;
    # for inputs starting with a different token, compute the mean register.
    from collections import Counter
    counts = Counter(first_token_ids.tolist())
    top_tok, top_count = counts.most_common(1)[0]
    print(f"\n[first-token effect]")
    print(f"  most common first-token id: {top_tok} (count {top_count}/{n})")
    if top_count >= 5 and n - top_count >= 5:
        mask = first_token_ids == top_tok
        mean_top = registers[mask].mean(dim=0).float()
        mean_other = registers[~mask].mean(dim=0).float()
        diff = (mean_top - mean_other).pow(2).sum().sqrt().item()
        cos_diff = torch.nn.functional.cosine_similarity(
            mean_top.unsqueeze(0), mean_other.unsqueeze(0), dim=-1
        ).item()
        print(f"  RMS difference between mean(top-tok) and mean(other-tok): {diff:.2f}")
        print(f"  cosine between those two means:                            {cos_diff:.4f}")

    # Save
    report = {
        "model": args.model, "layer": args.layer, "n_sequences": n, "d_model": d,
        "total_rms": total_rms,
        "constant_norm": constant_norm,
        "variable_rms": variable_rms,
        "constant_fraction_of_energy": constant_norm**2 / (constant_norm**2 + variable_rms**2),
        "pairwise_cos_mean": cos.mean().item(),
        "pairwise_cos_std": cos.std().item(),
        "pca_explained_top1": eigenvalues[0].item() / total_var,
        "pca_explained_top10": eigenvalues[:10].sum().item() / total_var,
        "pca_explained_top100": eigenvalues[:100].sum().item() / total_var if len(eigenvalues) >= 100 else None,
    }
    (args.out / "register_variability.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] wrote {args.out}/register_variability.json")


if __name__ == "__main__":
    main()
