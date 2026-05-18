# outlier-position-anatomy

**Mechanistic dissection of the "attention sink" / position-0 outlier in
small open-weight transformers.** Per-head + per-MLP attribution and
ablation on Qwen2.5-0.5B, GPT-2 small, and Pythia-1.4B.

## Headline finding

**The position-0 outlier in small open transformers is a load-bearing
write-and-erase circuit that maintains a fixed scaffolding vector — not
a memory. Removing it breaks the model.**

### Where it is (circuit topology)

| model              | n_layers | writers (first 4 layers)      | eraser (last 1-2 layers) |
|--------------------|---------:|-------------------------------|--------------------------|
| **Qwen2.5-0.5B**   | 24       | L2 head 5 + L2 MLP + L3 MLP   | L21 MLP                  |
| **GPT-2 small**    | 12       | L1 MLP + L2 MLP (writes 2334) | L11 MLP                  |
| **Pythia-1.4B**    | 24       | L3 MLP (writes 812) + L4 MLP  | L23 MLP (final layer)    |

### What's in it (across 256–512 held-out inputs per model)

| model              | layer | constant component | variable component | constant share of energy | mean pairwise cosine |
|--------------------|------:|-------------------:|-------------------:|-------------------------:|---------------------:|
| **Qwen2.5-0.5B**   | 10    | 1,682 RMS          | 65 RMS             | **99.8 %**               | **0.9999**           |
| **GPT-2 small**    | 6     | 3,041 RMS          | 34 RMS             | **99.99 %**              | **0.9999**           |
| **Pythia-1.4B**    | 12    | 1,283 RMS          | 56 RMS             | **99.8 %**               | **0.9996**           |

In all three models the register is essentially a fixed vector — same
direction, same magnitude, across hundreds of inputs.

### Why removing it doesn't work

On Qwen2.5-0.5B, ablating the writers drops position-0 residual RMS from
**754 → 11** at layer 2 — the outlier is gone. But CE loss on FineWeb-Edu
rises by **+3.29 nats** (from 2.66 → 5.94). The circuit is **functionally
necessary**; the model has *learned* to use the high-norm position-0 as
a structural anchor (an attention-sink dump) and breaks if that anchor is
removed.

All three architectures — different tokenizers, training corpora, layer
counts — implement the **same write-and-erase topology** at position 0.
The exact layer indices differ but the shape is universal: write in the
first 4 layers, carry through middle, erase in the last 1–2 layers.

**The dominant components are MLPs, not attention heads.** The standard
"attention sink" framing centers on attention; in these small open models
the MLPs do most of the work.

## What's in this repo

- `01_layer_attribution.py` — per-layer per-position residual-write
  attribution. Identifies the writer/eraser layers.
- `02_head_attribution.py` — per-head ablation at the suspect layers.
  Identifies the specific responsible components.
- `03_circuit_ablation.py` — joint writer-circuit ablation + held-out CE
  loss measurement. Tests load-bearing-ness.
- `04_register_variability.py` — across hundreds of inputs, measures the
  constant vs variable components of the position-0 register at mid-
  network. Shows the register is essentially a fixed vector.
- `results/{qwen2.5-0.5b,gpt2-small,pythia-1.4b}/*.json` — per-model
  numerical reports.
- [`NOTES.md`](NOTES.md) — full writeup with methodology, all results,
  and honest limitations.

## Why this is a contribution

The attention-sink phenomenon is documented at large scale (Xiao et al.
2023, Guo et al. 2024 on Llama-2 7B, Sok et al. 2026 on Gemma-3 /
Llama-3.1 / Qwen3 at 2B+). At **sub-2B open-weight model scale, with
per-head + per-MLP attribution and cross-architecture comparison, this
hasn't been publicly characterized.** Specifically:

- The Sok et al. (2026) `arxiv:2601.06787` paper is closest — they ablate
  BOS sink heads on Gemma-3 / Llama-3.1 / Qwen3, but at ≥2B model size.
- Guo et al. (2024) `arxiv:2410.13835` give a mechanistic account
  ("active-dormant heads") on Llama-2 7B.
- The ICLR 2025 dormant-heads paper (`arxiv:2504.03889`) touched
  Llama-3.2-1B but studied dormant-head *pruning*, not sink attribution.

What's new here: **per-head + per-MLP attribution + ablation on three
different sub-2B open architectures with cross-model topology comparison.**

## Reproducing

```bash
pip install torch>=2.3 transformers>=4.40 datasets>=2.19 tqdm safetensors

# Step 1 — layer-level attribution on each model
python 01_layer_attribution.py --model Qwen/Qwen2.5-0.5B \
    --n-sequences 128 --seq-len 256 --batch-size 8 \
    --out results/qwen2.5-0.5b
python 01_layer_attribution.py --model openai-community/gpt2 \
    --n-sequences 128 --seq-len 256 --batch-size 8 \
    --out results/gpt2-small
python 01_layer_attribution.py --model EleutherAI/pythia-1.4b \
    --n-sequences 64 --seq-len 256 --batch-size 4 \
    --out results/pythia-1.4b

# Step 2 — per-head ablation at suspect layers (Qwen example)
python 02_head_attribution.py --model Qwen/Qwen2.5-0.5B \
    --target-layers 2 3 5 21 \
    --out results/qwen2.5-0.5b

# Step 3 — joint writer-circuit ablation + perplexity (Qwen example)
python 03_circuit_ablation.py --model Qwen/Qwen2.5-0.5B \
    --out results/qwen2.5-0.5b
```

Total compute: under an hour on a single RTX 4090.

## Honest limitations

- **Three models, but Llama-3.2-1B not tested** (gated repo, no access).
  Unsloth's open mirror would work and is the obvious replication.
- **Head-level dissection only on Qwen2.5-0.5B.** The layer-level pass on
  all three is done; replicating step 2 on GPT-2 and Pythia is v0.2.
- **We don't know what the register stores.** We characterized *that* it
  exists, *which* components produce it, and *that* it's load-bearing.
  *Why* the model wants this register — what information it carries — is
  the next question.
- **No attention-probability analysis.** The Sok et al. framing measures
  attention mass at position 0; we measured residual norm. Both are valid
  and complementary; we picked residual norm because it's what our
  earlier SAE work had identified as the load-bearing artifact.

## Part of

[legible-weights](https://github.com/legibleweights/legible-weights) — an
umbrella research thread on interpretability of small open-weight LLMs.

## License

MIT.
