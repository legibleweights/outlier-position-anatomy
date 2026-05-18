"""Step 6 — is the register direction the anti-direction of the unembedding's bulk?

v0.3 found that Pythia's register direction is strongly anti-aligned with
whitespace tokens (cosine −0.44 vs random baseline 0.09). The natural
interpretation has two competing hypotheses:

  Functional: the model has *learned* to use the register at position 0
              to suppress whitespace-token predictions.
  Geometric:  whitespace tokens dominate the unembedding matrix's "bulk
              direction" (they're frequent in the Pile, their unembedding
              rows cluster); the register direction is just orthogonal-
              or-anti-orthogonal to that bulk because it has to be
              orthogonal to most of the unembedding to avoid distorting
              token predictions when carried into the lm_head.

This script computes the top principal components of the (centered)
unembedding matrix and compares them to the register direction. If the
register strongly aligns (or anti-aligns) with the top-1 PC of the
unembedding, the geometric hypothesis explains the v0.3 finding without
needing a functional interpretation.
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
    ap.add_argument("--layer", type=int, required=True)
    ap.add_argument("--n-sequences", type=int, default=128)
    ap.add_argument("--seq-len", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--n-pcs", type=int, default=10)
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

    # 1) Compute the constant register
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
    mean_reg = torch.cat(regs, dim=0).mean(dim=0)
    mean_reg_unit = mean_reg / mean_reg.norm().clamp_min(1e-8)
    print(f"[reg] |mean_reg| = {mean_reg.norm().item():.2f}")

    # 2) Compute top PCs of the unembedding matrix
    lm_head = model.get_output_embeddings()
    unembed_w = lm_head.weight.detach().float().cpu()  # (V, d_model)
    V, d = unembed_w.shape
    print(f"[unembed] shape ({V}, {d})")

    # Center the unembedding rows
    unembed_mean = unembed_w.mean(dim=0)
    unembed_centered = unembed_w - unembed_mean

    # Top-K PCs via SVD on centered matrix
    print(f"[pca] running SVD on centered unembedding...")
    U, S, Vh = torch.linalg.svd(unembed_centered, full_matrices=False)
    # Vh has shape (d, d): rows are principal directions; eigenvalues are S^2/(V-1)
    eigenvalues = (S ** 2) / (V - 1)
    total_var = eigenvalues.sum().item()
    pcs = Vh  # (d, d) — first row is top PC direction

    print(f"\n[explained variance]")
    for k in [1, 2, 5, 10]:
        ratio = eigenvalues[:k].sum().item() / total_var
        print(f"  top-{k:>2d}: {ratio*100:.2f}%")

    # 3) Cosine between register direction and each top PC + the unembedding mean
    print(f"\n[register direction vs unembedding structure]")
    unembed_mean_unit = unembed_mean / unembed_mean.norm().clamp_min(1e-8)
    cos_mean = (mean_reg_unit @ unembed_mean_unit).item()
    print(f"  cos(register, unembedding-mean):           {cos_mean:+.4f}")

    pc_cosines = []
    for i in range(args.n_pcs):
        pc_unit = pcs[i] / pcs[i].norm().clamp_min(1e-8)
        c = (mean_reg_unit @ pc_unit).item()
        pc_cosines.append(c)
        print(f"  cos(register, PC{i+1:>2d})  "
              f"(explains {eigenvalues[i].item()/total_var*100:>5.2f}% var):  {c:+.4f}")

    # 4) Decompose register into PC basis: how much of the register's norm
    #    lies in the top-K PCs of the unembedding?
    print(f"\n[register decomposition in unembedding-PCA basis]")
    register_in_pc_basis = pcs @ mean_reg     # (d,)
    register_total_sq = register_in_pc_basis.pow(2).sum().item()
    cum_pct = 0.0
    for k in [1, 2, 5, 10, 50, 100]:
        if k <= d:
            frac = register_in_pc_basis[:k].pow(2).sum().item() / register_total_sq
            print(f"  top-{k:>3d} PCs capture {frac*100:>5.1f}% of register's norm")

    report = {
        "model": args.model, "layer": args.layer,
        "mean_register_norm": float(mean_reg.norm().item()),
        "cos_register_unembed_mean": cos_mean,
        "cos_register_top_pcs": pc_cosines,
        "top_pc_explained_var": [eigenvalues[i].item()/total_var for i in range(args.n_pcs)],
        "register_norm_in_top_K_pcs": {
            str(k): float(register_in_pc_basis[:k].pow(2).sum().item() / register_total_sq)
            for k in [1, 2, 5, 10, 50, 100] if k <= d
        },
    }
    (args.out / "register_vs_unembed_pca.json").write_text(json.dumps(report, indent=2))
    print(f"\n[save] wrote {args.out}/register_vs_unembed_pca.json")


if __name__ == "__main__":
    main()
