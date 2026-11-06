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

Plain DiT: no text tokens in the sequence, bidirectional mixing over image tokens. Three of every four blocks mix by a depthwise 3×3 convolution over the token grid (conv-first), the fourth by self-attention. The caption enters through the pooled embedding in the adaLN conditioning; blocks with cross attention (an experiment switch, off by default) also read the 512-d token states.

| Component | Decision |
|---|---|
| Width | 1024 |
| Depth | 16 layers |
| Mixer | Every 4th block (the 4th, 8th, …) is self-attention, the others a conv block: depthwise 3×3 over all channels after modulation, an output Linear of width → width − 128 that writes the meta channels only, a branch norm and the gate |
| Attention | GQA: 16 query heads × head dim 64, a quarter as many key/value heads (a speed gain at every grid size; no KV cache to slim) |
| QK norm | RMSNorm on Q and K |
| Branch norm | Sandwich norm (LFM3): an RMSNorm with a learned gain on each branch's output, before its gate and the residual add, so a block adds at most the gate times a vector of unit RMS before the gains |
| Output gate | Krea-style sigmoid gate on the attention output: `o = attn(·) ⊙ σ(W_g h)` |
| Position | 2D axial RoPE on the first quarter of the frequencies per axis, the rest position-free; the convs carry local position |
| Cross attention (switch) | In attention blocks only, after self-attention and before the MLP: no modulation, no positional encoding, GQA, its own q/k norm and branch norm, a zero-initialised per-channel gate |
| MLP | SwiGLU, hidden 4 × width |
| Modulation | adaLN-single (PixArt): one shared projection from the conditioning vector to every block's six shift/scale/gate vectors (mixer, MLP), plus a learned 6×width table per block; per image, not per token |
| Conditioning vector | `c = MLP_t(sinusoid(t)) + Linear_text(rms_norm(e_text))`, the norm with no gain |
| Output | The first 128 channels of the residual stream after the last block, unpatchified: no final norm, modulation or linear. The patch embed concatenates: the raw 128-d patch fills the first 128 channels, a Linear 128 → width − 128 fills the rest, so the image channels stay pure and the read-out has no bottleneck; the stream carries the working latent and each block adds a correction to the meta channels |
| Size | Parameter counts by width and depth in FACTS |

Not used: linear attention (256 tokens make softmax cheap), MLA (no KV cache), DeepSeek sparse attention (no long context).

## 4. Objective: x0 velocity

- Network output `F`. Target `T = x_t - v = x + (1 - t)(x - ε)`.
- Loss: `‖F - T‖²`, unweighted. This is identically `‖v̂ - v‖²` with `v̂ = x_t - F`.
- The sampler receives `v̂ = x_t - F`; any flow-matching sampler runs unchanged.
- Timestep sampling: logit-normal with logit std 0.7, which puts more samples at mid t, where the loss is highest. Shift grows with resolution, scaled by √(token ratio) per stage.
- Prompt dropout: 10%, replaced by a learned null embedding → classifier-free guidance at sampling.

## 5. Initialization

Default: fan-in normal, `std = 1/√fan_in`.

| Role | Init | Optimizer (§6) |
|---|---|---|
| Patch embed 128→(width − 128) | fan-in normal, zero bias; the first 128 channels are the raw patch, no parameters | Usui |
| Timestep MLP | fan-in normal | Usui |
| Text projection 768→1024 | fan-in normal on its input `rms_norm(e_text)` (unit RMS over entries) | Usui |
| Null embedding | EmbeddingGemma 2's embedding of the empty caption (`title: none \| text: `), learned from there | fallback |
| Q, K, V (fused, GQA) and cross-attention q, kv | fan-in normal | Usui |
| Conv block depthwise 3×3 and output Linear | fan-in normal | conv on the fallback, Linear on Usui |
| Sigmoid gate projection | fan-in normal (σ ≈ 0.5 at start) | Usui |
| Attention out-proj | fan-in normal | Usui |
| SwiGLU up / gate / down | fan-in normal | Usui |
| adaLN shift / scale rows (and the cross-attention gate, a vector of zeros on the fallback) | small random | Usui, with the shared projection; tables on the fallback |
| adaLN **gate** rows (shared projection and per-block table) | zero | Usui, with the shared projection; tables on the fallback |
| Branch-norm gains, QK-RMSNorm gains, biases | ones / zero | fallback |

Consequence of the concatenated patch embed with x0 velocity: the zero gates make every block the identity, so `F = x_t` and `v̂ = 0` at step 0. The loss reaches the shared adaLN projection and the gate rows at once; the patch embed's meta channels are read by no gate yet, so it and the blocks behind the gates follow as the gates open.

## 6. Optimizer: UsuiTrack

- Every tracked matrix is pinned to the Frobenius norm it has at its first step (Usui's `pin_norm`), and takes Usui's trust step, so `lr` is a constant relative step for the whole run: the weights cannot grow, and the step does not shrink as they would have. No tracked matrix is zero-initialized (§5), so every one has a norm to keep.
- Frame size: one rank fraction for every tracked matrix.
- Frames drawn at random when the optimizer is built (Usui's `frame_init="random"`): behind the zero-initialized gates every upstream gradient starts at zero, so a first-gradient fit has nothing to find.
- Linear LR warmup on both optimizers, then a linear decay over the run to a configured fraction of the peak.
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
- Loss logged bucketed by t (averaged loss stays flat and says little), with R² = 1 − loss·t²/E, E the mean square of the clean latents. The F loss is the clean-latent error over t, so R² is the share of the latents' energy the prediction recovers from the noisy input: 0 is no better than predicting zero, 1 is exact. It falls with t by construction (more noise, less to recover), so it is read across steps at one t, not across t. Ten equal-width buckets in the records; wandb shows five (each the pooled pair).
- Score at every eval: the loss on an eval set never trained on, with each row's caption, every row at the same 10 t (the midpoints of equal-probability bins of the shift-1 logit-normal, whatever t the run trains on) with its own fixed noise. One loss and one R², and each value at each t.
- Logs: plain JSONL files, the record: one line per `log_every` steps, the reads averaged over them on the GPU. The same values and the eval grid stream to wandb for live curves.
