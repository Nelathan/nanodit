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
docs/DECISIONS.md     every decision with its reason, and what is still open
docs/REFERENCES.md    sources (papers, model cards, datasets) and what each one establishes
docs/references/      source material kept verbatim (the x0 velocity thread)
CLAUDE.md             working rules for agents implementing this
src/tinydit/          model, objective, data, sampler (empty)
scripts/              precompute, train, sample (empty)
configs/              run configs (empty)
```

## Stages

1. Precompute Primal latents (128 px, 256 px) and EmbeddingGemma 2 caption embeddings for text-to-image-2M.
2. LR sweep at 128 px.
3. Train 128 px, then 256 px. 512 px is out of scope for now.

## Dependencies (expected)

PyTorch (2.15 targeted), `diffusers` (Primal VAE loads via `AutoModel` with `trust_remote_code`),
`sentence-transformers` / `transformers` (EmbeddingGemma 2), `webdataset`, UsuiTrack (local).
