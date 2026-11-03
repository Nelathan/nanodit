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
| Training data | the shard Parquets of a resolution shuffled into one Parquet, rows in uniformly random order, each image once (a repeated latent keeps its first row in shard order); adding shards means shuffling again. Training streams it in blocks of row groups, each pass visiting the blocks in a new random order and shuffling rows within a block |
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
| Position | 2D axial RoPE on all of each head's dims (32 for rows, 32 for columns) |
| MLP | SwiGLU, hidden 4 × width |
| Modulation | adaLN-single (PixArt): one shared projection from the conditioning vector to every block's shift/scale/gate, plus a small learned table per block |
| Conditioning vector | `c = MLP_t(sinusoid(t)) + Linear_text(rms_norm(e_text))`, the norm with no gain |
| Output | The first 128 channels of the residual stream after the last block, unpatchified: no final norm, modulation or linear. The patch embed starts as the identity on those channels, so the stream carries the working latent and each block adds a correction |
| Size | Parameter counts by width and depth in FACTS |

Not used: linear attention (256 tokens make softmax cheap), MLA (no KV cache), DeepSeek sparse attention (no long context).

## 4. Objective: x0 velocity

- Network output `F`. Target `T = x_t - v = x + (1 - t)(x - ε)`.
- Loss: `‖F - T‖²`, unweighted. This is identically `‖v̂ - v‖²` with `v̂ = x_t - F`.
- The sampler receives `v̂ = x_t - F`; any flow-matching sampler runs unchanged.
- Timestep sampling: logit-normal. Shift grows with resolution, scaled by √(token ratio) per stage.
- Prompt dropout: 10%, replaced by a learned null embedding → classifier-free guidance at sampling.

## 5. Initialization

Default: fan-in normal, `std = 1/√fan_in`.

| Role | Init | Optimizer (§6) |
|---|---|---|
| Patch embed 128→512 | identity on the first 128 output channels, fan-in normal on the rest, zero bias | Usui |
| Timestep MLP | fan-in normal | Usui |
| Text projection 768→512 | fan-in normal on its input `rms_norm(e_text)` (unit RMS over entries) | Usui |
| Null embedding | EmbeddingGemma 2's embedding of the empty caption (`title: none \| text: `), learned from there | fallback |
| Q, K, V | fan-in normal | Usui |
| Sigmoid gate projection | fan-in normal (σ ≈ 0.5 at start) | Usui |
| Attention out-proj | fan-in normal | Usui |
| SwiGLU up / gate / down | fan-in normal | Usui |
| adaLN shift / scale rows | small random | Usui, with the shared projection; tables on the fallback |
| adaLN **gate** rows (shared projection and per-block table) | zero | Usui, with the shared projection; tables on the fallback |
| QK-RMSNorm gains, biases | ones / zero | fallback |

Consequence of the identity patch embed with x0 velocity: the zero gates make every block the identity, so `F = x_t` and `v̂ = 0` at step 0, and the loss reaches the patch embed and the gate rows at once; the blocks behind the gates follow as the gates open.

## 6. Optimizer: UsuiTrack

- Usui's fixed step on every tracked matrix: sized by `lr` and the matrix's shape, not relative to its norm, since a model trained from scratch has zero-initialized matrices with no norm to be relative to.
- Frame size: one rank fraction for every tracked matrix.
- Frames drawn at random when the optimizer is built (Usui's `frame_init="random"`): behind the zero-initialized gates every upstream gradient starts at zero, so a first-gradient fit has nothing to find.
- Linear LR warmup on both optimizers.
- Everything that is not a 2-D Linear weight, and the adaLN tables, train on Usui's fallback, `StochasticAdamW`.
- Weight gradients through Usui's `Molegrad` (no `G` formed for tracked matrices); batches are built by gradient accumulation over micro-batches.
- Precision: the model's parameters and activations are bf16; both optimizers write with stochastic rounding.

## 7. Training schedule

1. 128 px — LR sweep, then the main run.
2. 256 px — continue from the 128 px checkpoint, timestep shift raised.

## 8. Sampling and eval

- Sampler for eval: Euler, 30 steps, CFG. Guidance scale tuned on the grid.
- dpmpp_2m with the beta scheduler is allowed for best-looking samples, separately from the eval grid.
- Eval grid: 16 fixed prompts, each with its own fixed noise, one 4×4 grid every N steps.
- Loss logged bucketed by t (averaged loss stays flat and says little).
- Score at every checkpoint: the loss on an eval set never trained on, with each row's caption and with the null embedding, every row at the same 10 t (the midpoints of equal-probability bins of the training t) with its own fixed noise. One number per condition, and its value at each t.
- Logs: plain JSONL files, the record: one line per `log_every` steps, the reads averaged over them on the GPU. The same values and the eval grid stream to wandb for live curves.
