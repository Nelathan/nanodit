# tiny-dit

A from-scratch toy text-to-image diffusion transformer, trained on one RTX 4070 Super (12 GB).
The question it answers: can a small DiT built on this season's promising pieces produce images at all?

- **Target:** Ostris's *x0 velocity* parameterization of rectified flow
- **Latents:** Ostris's **Primal VAE** (2x2 variant), patched by the DiT itself
- **Conditioning:** **EmbeddingGemma 2**, mean-pooled 768-d text embedding (per-token later, maybe)
- **Optimizer:** **UsuiTrack** (trust-region LR relative to weight norm)
- **Framework:** pure PyTorch, single GPU, no trainer framework

Nothing is built yet. This repo holds the agreed design so implementation starts from one picture.

## Layout

```
docs/SPEC.md          the design: data, model, init, training, eval
docs/PLAN.md          open questions, grouped, and how each would be settled
docs/FACTS.md         measured numbers with source and date
docs/archive/         closed decisions and questions, one note per session
docs/HANDOVER.md      where things stand, next steps, traps
docs/REFERENCES.md    sources (papers, model cards, datasets) and what each one establishes
docs/references/      source material kept verbatim (the x0 velocity thread)
AGENTS.md             working rules for agents implementing this (CLAUDE.md imports it)
src/tinydit/          model, objective, data, sampler
scripts/              precompute, train, sample (empty)
configs/              run configs (empty)
```

## Stages

1. Precompute Primal latents (128 px, 256 px) and EmbeddingGemma 2 caption embeddings for text-to-image-2M.
2. LR sweep at 128 px.
3. Train 128 px, then 256 px. 512 px is out of scope for now.

## Setup

`uv sync` from the repo root. torch 2.15 comes from the PyTorch cu132 index; usuitrack is an editable path dependency from `../usuitrack`.
Primal VAE loads via `diffusers` `AutoModel` with `trust_remote_code`; EmbeddingGemma 2 via `sentence-transformers` / `transformers`.
