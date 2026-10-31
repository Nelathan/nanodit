# tiny-dit — Design Spec

Notation: rectified flow, `x_t = (1 - t)·x + t·ε`, `t = 1` is pure noise, velocity `v = ε - x`.

---

## 1. Data

| Item | Decision |
|---|---|
| Dataset | `jackyhate/text-to-image-2M`, the `data_512_2M` WebDataset shards (512×512 crops, ~2M pairs) |
| Subset | the first few shards (50k pairs each), sized for ~50-min runs; more only once short runs are trusted |
| Resolutions | 128 px and 256 px, resized from the 512 crops. 512 px skipped for now |
| Captions | full caption, no truncation |
| Text encoder | `google/embeddinggemma-2`, text path only (`vision_config=None`, `audio_config=None`), loaded through `sentence-transformers`, bf16 or fp32 (never fp16) |
| Text prefix | one fixed prefix for training and inference: `title: none \| text: {caption}` |
| Embedding | mean-pooled, projected 768-d, L2-normalized as the model returns it |
| VAE | `ostris/primal_vae`, variant `2x2`: 2 channels at 2× downsample, deterministic, unit-normal per image and channel, no scale/shift factors. Image sides must be multiples of 16 |
| Precompute | latents encoded separately per resolution (a 128 px latent is not a downsampled 256 px latent) and embeddings computed once, stored bf16. Streamed per source shard: download, encode, write one Parquet file per shard and resolution (key, caption, embedding, latent), delete the shard |
| Storage | latents 16 KB (128 px) + 64 KB (256 px) per image, ≈160 GB for 2M; embeddings 1.5 KB per image, ≈3 GB |
| Augmentation | none |
| Shapes | square crops per stage, so shapes are static (CUDA-graph friendly via a preallocated input buffer) |

## 2. Tokens

Primal 2x2 latent `[B, 2, H/2, W/2]` → patch 8 on the latent (`pixel_unshuffle(8)`) → 128-d tokens at a 16-px image stride.

| Resolution | Latent | Token grid | Tokens |
|---|---|---|---|
| 128 px | 2×64×64 | 8×8 | 64 |
| 256 px | 2×128×128 | 16×16 | 256 |

Patch size is fixed across stages, so the patch embedding carries over unchanged.

## 3. Model

Plain DiT: no text tokens in the sequence (pooled conditioning only), bidirectional self-attention over image tokens.

| Component | Decision |
|---|---|
| Width | 512 |
| Depth | 12 layers (24 later, once it produces images) |
| Attention | MHA, 8 heads × head dim 64. No GQA (no KV cache, nothing to save) |
| QK norm | RMSNorm on Q and K |
| Output gate | Krea-style sigmoid gate on the attention output: `o = attn(·) ⊙ σ(W_g h)` |
| Position | 2D axial RoPE on **half** of each head's dims (32 of 64: 16 for rows, 16 for columns). The other half is position-free, for content matching |
| Local mixing | 3×3 depthwise conv over the token grid, applied to the normed block input before QKV, as a zero-initialized residual (starts as identity). Carries local relative position; RoPE carries global layout |
| MLP | SwiGLU, hidden 1024 |
| Modulation | adaLN-single (PixArt): one shared projection from the conditioning vector to shift/scale/gate, plus a small learned per-block table |
| Conditioning vector | `c = MLP_t(sinusoid(t)) + Linear_text(√768 · e_text)` |
| Final layer | adaLN-modulated norm → Linear 512 → 128, unpatchify |
| Size | ≈37M parameters at 12 layers (FACTS) |

Not used: linear attention (256 tokens make softmax cheap), MLA (no KV cache), DeepSeek sparse attention (no long context).

## 4. Objective: x0 velocity

- Network output `F`. Target `T = x_t - v = x + (1 - t)(x - ε)`.
- Loss: `‖F - T‖²`, unweighted. This is identically `‖v̂ - v‖²` with `v̂ = x_t - F`.
- The sampler receives `v̂ = x_t - F`; any flow-matching sampler runs unchanged.
- Timestep sampling: logit-normal. Shift grows with resolution, scaled by √(token ratio) per stage.
- Prompt dropout: 10%, replaced by a learned null embedding → classifier-free guidance at sampling.

## 5. Initialization

Default: fan-in normal, `std = 1/√fan_in`.

| Role | Init | Usui target-norm override |
|---|---|---|
| Patch embed 128→512 | fan-in normal (inputs are already unit-normal) | no |
| Timestep MLP | fan-in normal | no |
| Text projection 768→512 | fan-in normal on input scaled by √768 (unit RMS over entries) | no |
| Null embedding | EmbeddingGemma 2's embedding of the empty caption (`title: none \| text: `), learned from there | no |
| Q, K, V | fan-in normal | no |
| Sigmoid gate projection | fan-in normal (σ ≈ 0.5 at start) | no |
| Attention out-proj | fan-in normal | no |
| SwiGLU up / gate / down | fan-in normal | no |
| adaLN shift / scale rows | small random | no |
| adaLN **gate** rows (shared projection and per-block table) | zero | **yes** |
| Depthwise conv | zero | **yes** |
| Final linear 512→128 | zero | **yes** |
| QK-RMSNorm gains, final norm gain | ones | n/a |

Override target for a zero-initialized role: the norm a fan-in init of the same shape would have.
Consequence of zero output with x0 velocity: `F = 0` gives `v̂ = x_t` at step 0 (plain velocity would give `v̂ = 0`).

## 6. Optimizer: UsuiTrack

- Trust-region LR relative to weight norm; per-role target-norm override for zero-init roles (§5).
- Weight gradients through Usui's `Molegrad` (no `G` formed for tracked matrices); batches are built by gradient accumulation over micro-batches.

## 7. Training schedule

1. 128 px — LR sweep, then the main run.
2. 256 px — continue from the 128 px checkpoint, timestep shift raised.

## 8. Sampling and eval

- Sampler for eval: Euler, 30 steps, CFG. Guidance scale tuned on the grid.
- dpmpp_2m with the beta scheduler is allowed for best-looking samples, separately from the eval grid.
- Eval grid: fixed prompts × fixed seeds, every N steps.
- Loss logged bucketed by t (averaged loss stays flat and says little).
- Logs: plain JSONL files, readable without wandb.
