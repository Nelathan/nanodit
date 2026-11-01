# tiny-dit

A from-scratch toy text-to-image diffusion transformer, trained on one RTX 4070 Super (12 GB).
The question it answers: can a small DiT built on this season's promising pieces produce images at all?

- **Target:** Ostris's *x0 velocity* parameterization of rectified flow
- **Latents:** Ostris's **Primal VAE** (2x2 variant), patched by the DiT itself
- **Conditioning:** **EmbeddingGemma 2**, mean-pooled 768-d text embedding (per-token later, maybe)
- **Optimizer:** **UsuiTrack** (its fixed step, sized by lr and the matrix shape)
- **Framework:** pure PyTorch, single GPU, no trainer framework

The model, objective, sampler and training harness are built and tested; 300-step smoke runs at 128 px are in `docs/FACTS.md`. `docs/HANDOVER.md` says where things stand.

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
src/tinydit/          model, objective, sampler, data, optimizer split
scripts/              precompute, probes, train
configs/              run configs, eval prompts
```

## Stages

1. Precompute Primal latents (128 px, 256 px) and EmbeddingGemma 2 caption embeddings for the first shards of text-to-image-2M.
2. LR sweep at 128 px.
3. Train 128 px, then 256 px. 512 px is out of scope for now.

## Setup

`uv sync` from the repo root. torch 2.15 comes from the PyTorch cu132 index; usuitrack is an editable path dependency (`AGENTS.md`, "Running things").
Primal VAE loads via `diffusers` `AutoModel` with `trust_remote_code`; EmbeddingGemma 2 via `sentence-transformers` / `transformers`.
