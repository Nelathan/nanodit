# Design session, 2026-10-31

Closed in the design session of 2026-10-31. Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | Goal: see whether a small DiT built on these pieces produces images. Keep everything simple | Exploration, not a benchmark |
| 2 | x0 velocity target | Ostris reports sharper results at high compression; drop-in for flow-matching samplers |
| 3 | Primal VAE, 2x2 variant | Card shows 2x2 matching or beating the Flux.2 VAE at the same budget; 2x1 trails on texture and its per-image std drifts |
| 4 | EmbeddingGemma 2, pooled 768-d only; per-token maybe later | Simplest conditioning path |
| 5 | Full captions | Daniel judges EmbeddingGemma 2 capable of embedding them; length mismatch noted as a risk (PLAN B3) |
| 6 | text-to-image-2M | Ready-to-stream WebDataset at 512 px; fastest path to images. Trade-off: largely synthetic images |
| 7 | Width 512, 12 layers (→24) | Small, fast to train; scale depth once it works |
| 8 | MHA 8×64 | Head dim 64. GQA saves nothing without a KV cache |
| 9 | Krea sigmoid output gate | Cheap modern touch |
| 10 | Partial RoPE (half the head dims) + 3×3 depthwise conv before QKV | No causal mask, so position must come from RoPE or conv. Conv = 2D twin of the LLM short-conv trick (CPVT, SANA Mix-FFN) |
| 11 | No linear attention | 256 tokens: softmax is ~a tenth of layer FLOPs. Directional 2D-masked linear attention parked as a later toy |
| 12 | SwiGLU hidden 1024 | Difference to 1365 judged small |
| 13 | adaLN-single | Per-block adaLN costs 6d² per block, more than attention |
| 14 | Patch 8 on the latent, 16-px stride | 64 tokens at 128 px, 256 at 256 px |
| 15 | 10% prompt dropout to a learned null embedding | Needed for classifier-free guidance |
| 16 | Euler 30 steps for eval | Deterministic, fewest confounds |
| 17 | Progressive 128 → 256 px; skip 512 | 512 storage (~512 GB) and time not needed yet |
| 18 | Precompute latents and embeddings | Encoders off the training loop; static shapes |
| 19 | UsuiTrack as optimizer | Be bold, have fun. No AdamW reference run |
| 20 | Init per role and target-norm overrides as in SPEC §5 | Trust-region LR leaves zero-norm params stuck without an override |
| 21 | Pure PyTorch, no HF Trainer, no ai-toolkit | Usui's layerwise autograd, checkpointing and regional compile sit exactly where frameworks take control; one GPU needs none of what they offer |
| 22 | LR sweep at 128 px: 1e-3, 3e-3, 1e-2 relative | LFM's 4e-5 is a fine-tuning rate; from scratch needs larger relative steps (estimate) |

## Understanding of x0 velocity (from discussion, untested)

Worked out in discussion, not empirically tested here.
x0 velocity is velocity prediction with a hardwired output skip, i.e. preconditioning with `c_skip = 1, c_out = -1`.
Same objective, same optimum. Any benefit comes from inductive bias. The "image-shaped at every t" claim holds only
partly: the target carries a `(1 - t)·ε` term, largest near the clean end.
