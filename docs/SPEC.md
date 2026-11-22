# nanodit — Design Spec

Notation: rectified flow, `x_t = (1 - t)·x + t·ε`, `t = 1` is pure noise, velocity `v = ε - x`.

---

## 1. Data

| Item | Decision |
|---|---|
| Dataset | `jackyhate/text-to-image-2M`, the `data_512_2M` WebDataset shards (512×512 crops, ~2M pairs), and `ma-xu/fine-t2i`, all five subsets (each image scaled to cover the nearest of seven aspect buckets and centre-cropped, each row captioned by its `prompt`) |
| Subset | all 47 shards and all of Fine-T2I; trials train on the same set as long runs |
| Resolutions | 128 px and 256 px, each a Lanczos resize of the 512 crop. 512 px skipped for now |
| Captions | full caption as encoded; in training and at sampling the token states are cut at 512, with a warning |
| Excluded rows | a set's `excluded_keys.txt` lists the keys of rows that are never streamed: captions that match a defect pattern (escaped bytes, a short unit repeated 12 times or more, a leaked chat template, any CJK character, empty). The shuffle writes it empty and `scripts/exclude_captions.py` fills it |
| Held-out rows | a set's `heldout_keys.txt` lists the keys of its held-out eval set, never streamed: 1,000 rows of Fine-T2I's square subset with enhanced prompts, generator FLUX.2-dev, the 1,000 with the smallest SHA-256 of their key, not among the excluded. The shuffle writes it empty and `scripts/holdout.py` fills it and writes their rows |
| Text encoder | `google/embeddinggemma-2`, text path only (`vision_config=None`, `audio_config=None`), loaded through `sentence-transformers`, bf16 or fp32 (never fp16) |
| Text prefix | none: the caption is encoded as it is, for training and inference |
| Embedding | mean-pooled, projected 768-d, L2-normalized as the model returns it; stored, not read by the model |
| Token states | the encoder's 512-d states of the caption's tokens after its final norm, before the projection to 768; `<bos>` and `<eos>` dropped, unpadded |
| VAE | `ostris/primal_vae`, variant `2x2`: 2 channels at 2× downsample, deterministic, unit-normal per image and channel, no scale/shift factors. Image sides must be multiples of 16 |
| Precompute | latents encoded separately per resolution (a 128 px latent is not a downsampled 256 px latent) and embeddings computed once, stored bf16. Streamed per source shard or tar: download, encode, write one Parquet file per shard (key, source, caption, embedding, token states, both latents; a Fine-T2I file also its bucket), delete the download and any intermediate copy |
| Training data | both datasets merged into one set per resolution: one Parquet per aspect bucket, its rows in uniformly random order, each image once across both datasets (a repeated latent keeps its first row, text-to-image-2M first); new data means shuffling again. Each micro-batch is one bucket, drawn with probability proportional to its rows; a batch's micro-batches weigh by their tokens. A bucket streams in blocks of row groups, dealt anew at random every pass, the blocks visited in random order and rows shuffled within a block. The order follows the run's seed |
| Augmentation | none |
| Shapes | seven aspect buckets per stage, one preallocated input buffer per bucket. At 128 px: 128×128, 112×144, 144×112, 112×160, 160×112, 96×176, 176×96 |

## 2. Tokens

Primal 2x2 latent `[B, 2, H/2, W/2]` → patch 8 on the latent (`pixel_unshuffle(8)`) → 128-d tokens at a 16-px image stride.

| Resolution | Latent | Token grid | Tokens |
|---|---|---|---|
| 128 px | 2×64×64 | 8×8 | 64 |
| 256 px | 2×128×128 | 16×16 | 256 |

Patch size is fixed across stages, so the patch embedding carries over unchanged.

## 3. Model

Architecture **a4**: §2, this section and §5 as they stand. A change to any of them is a new version, and run names carry it (`a4-ref-1ep`).

DiT over image tokens; the caption's tokens are keys, never queries or stream positions. Three of every four blocks mix over the 3×3 neighbourhood on the token grid by LFM2's gated short conv, writing all channels (conv-first), the third by joint attention. The adaLN conditioning takes the timestep alone. The caption enters only through the attention blocks, which read its 512-d token states after a caption refiner.

| Component | Decision |
|---|---|
| Width | 768 |
| Depth | 24 layers |
| Layout | A sequence of block types, repeated to the depth: conv, conv, attention, conv, so the 3rd, 7th, … block is attention and the last a conv. A block type names its mixer |
| Mixer | LFM2's gated short conv: one Linear width → 3 × width from the modulated input gives B, C and v; a depthwise 3×3 conv over the token grid reads B ⊙ v; its read, RMS-normed without a gain, times C, goes through an output Linear width → width and the gate into all channels. No branch norm, so a small C makes a small write |
| Attention | Joint: the image tokens' queries read, in one softmax, the image keys and the caption's keys and values (a Linear 512 → key/value heads, its own k norm, no RoPE; padding masked). GQA: query heads of dim 64 (12 at width 768), a quarter as many key/value heads (a speed gain at every grid size; no KV cache to slim). Each head's read is RMS-normed without a gain |
| Sink | A learned null key per key/value head with value 0, met by the query before RoPE: a head with logit b on it writes σ(lse − b) times its normed read, so it can write nothing |
| Caption refiner | Two blocks over the caption's token states alone, at their native 512, before any attention block reads them: self-attention (8 heads of 64, QK RMSNorm, no RoPE, padding masked) then SwiGLU (hidden 4 × 512), each branch with a branch norm and a zero-initialised per-channel gate; no t and no image tokens, so they run once per sample |
| QK norm | RMSNorm on Q and K |
| Branch norm | Sandwich norm (LFM3) on the MLP branch: an RMSNorm with a learned gain on its output, before its gate and the residual add, so it adds at most the gate times a vector of unit RMS before the gain. The mixers have their norm before the output Linear instead, so a small read makes a small write: the conv's on its read (Mixer), attention's per head (Attention) |
| Output gate | Krea-style sigmoid gate on the attention output: `o = attn(·) ⊙ σ(W_g h)` |
| Position | 2D axial RoPE on the first quarter of the frequencies per axis, the rest position-free; the convs carry local position |
| MLP | SwiGLU, hidden 4 × width |
| Modulation | adaLN-single (PixArt): one shared projection from the conditioning vector to every block's four scale/gate vectors (mixer, MLP), plus a learned 4×width table per block; per image, not per token. No shift: a branch reads `rms_norm(x) · (1 + scale)` |
| Conditioning vector | `c = MLP_t(sinusoid(t))` |
| Output | The first 128 channels of the residual stream after the last block, unpatchified: no final norm, modulation or linear. The patch embed concatenates: the raw 128-d patch fills the first 128 channels, a Linear 128 → width − 128 fills the rest, so the image channels stay pure and the read-out has no bottleneck; the stream carries the working latent and each block adds a correction to it |
| Size | Parameter counts by width and depth: `scripts/count_params.py` |

Not used: linear attention (256 tokens make softmax cheap), MLA (no KV cache), DeepSeek sparse attention (no long context).

## 4. Objective: x0 velocity

- Network output `F`. Target `T = x_t - v = x + (1 - t)(x - ε)`.
- Loss: `‖F - T‖²`, unweighted. This is identically `‖v̂ - v‖²` with `v̂ = x_t - F`.
- The sampler receives `v̂ = x_t - F`; any flow-matching sampler runs unchanged.
- Timestep sampling: logit-normal with logit std 0.7 and shift 1 at 128 px, which centres the samples near the midpoint of the noising line: it follows the objective's geometry (`topics/prediction-target.md`), not the data. Shift grows with resolution, scaled by √(token ratio) per stage.
- Prompt dropout: 10%, replaced by the learned null tokens → classifier-free guidance at sampling. A caption with no tokens is the null prompt, in training and at sampling; the encoder warns of it and stores no token states.

## 5. Initialization

Default: fan-in normal, `std = 1/√fan_in`.

| Role | Init | Optimizer (§6) |
|---|---|---|
| Patch embed 128→(width − 128) | fan-in normal, zero bias; the first 128 channels are the raw patch, no parameters | Usui |
| Timestep MLP | fan-in normal | Usui |
| Null tokens | EmbeddingGemma 2's `<bos>` and `<eos>` states of the empty caption, learned from there | fallback |
| Q, K, V (fused, GQA), the caption's k, v, the refiner's Linears | fan-in normal | Usui |
| Null keys | zero | fallback |
| Conv block input Linear (B, C, v), depthwise 3×3 and output Linear | fan-in normal | conv on the fallback, Linears on Usui |
| Sigmoid gate projection | fan-in normal (σ ≈ 0.5 at start) | Usui |
| Attention out-proj | fan-in normal | Usui |
| SwiGLU up / gate / down | fan-in normal | Usui |
| adaLN scale rows | small random | Usui, with the shared projection; tables on the fallback |
| adaLN **gate** rows (shared projection and per-block table) | zero | Usui, with the shared projection; tables on the fallback |
| Refiner gates | zero | fallback |
| Branch-norm gains, QK-RMSNorm gains, biases | ones / zero | fallback |

Consequence of the concatenated patch embed with x0 velocity: the zero gates make every block and every refiner block the identity, so `F = x_t` and `v̂ = 0` at step 0. The loss reaches the shared adaLN projection and the gate rows at once; the patch embed's meta channels are read by no gate yet, so it and the blocks behind the gates follow as the gates open.

## 6. Optimizer: UsuiTrack

- Every tracked matrix takes Usui's trust step, so `lr` is the relative step, and its Frobenius norm is clipped at the norm it has at its first step (Usui's `norm_bounds = (None, 1)`): each matrix lives in the ball of its init radius, set by its init (§5), with no floor and no weight decay. No tracked matrix is zero-initialized (§5), so every one has an init norm.
- Frame size: one rank fraction for every tracked matrix. Every matrix is framed on its input side (Usui's default `side = "right"`).
- Frames drawn at random when the optimizer is built (Usui's group key `frame_init = "random"`): behind the zero-initialized gates every upstream gradient starts at zero, so a first-gradient fit has nothing to find.
- Linear LR warmup on both optimizers, then a constant lr. An anneal is a short run of its own, continued from a checkpoint, for eval or as post-training: Usui's lr falls by a cosine to 1e-5, the fallback's by the same factor. No EMA of the weights, no merging of checkpoints.
- Everything that is not a 2-D Linear weight, and the adaLN tables, train on Usui's fallback, `StochasticAdamW`, at its default betas and without weight decay.
- Weight gradients through Usui's `Molegrad` (no `G` formed for tracked matrices); batches are built by gradient accumulation over micro-batches.
- Precision: the model's parameters and activations are bf16; both optimizers write with stochastic rounding.

## 7. Training schedule

About 50 image tokens per image parameter in all:

1. 128 px — eight epochs at batch 256.
2. 256 px — three epochs, continued from the 128 px checkpoint, timestep shift raised.

## 8. Sampling and eval

- Sampler for eval: Euler, 30 steps, guidance 1: the conditional prediction alone, the null branch not run. CFG is for post-training.
- dpmpp_2m with the beta scheduler is allowed for best-looking samples, separately from the eval grid.
- Eval grid: 16 fixed prompts, each with its own fixed noise, one 4×4 grid every N steps.
- Loss logged bucketed by t (averaged loss stays flat and says little), with R² = 1 − loss·t²/E, E the mean square of the clean latents. The F loss is the clean-latent error over t, so R² is the share of the latents' energy the prediction recovers from the noisy input: 0 is no better than predicting zero, 1 is exact. It falls with t by construction (more noise, less to recover), so it is read across steps at one t, not across t. Ten equal-width buckets in the records; wandb shows five (each the pooled pair).
- Score at every eval, on two sets never trained on, each set on each t grid into its own record: the set's held-out rows (§1, in the training distribution), the eval, at the training resolution; and Lunara (out of it) at 256 px, whatever the training resolution. The score is the loss with each row's caption, every row at the same 10 t (the midpoints of equal-probability bins of the shift-1 logit-normal, whatever t the run trains on) with its own fixed noise; and the same at 10 t evenly spaced (the midpoints of equal bins of t), the uniform score. Each set and grid gives one loss, one decoder-weighted loss (each spatial frequency of the error weighted by the decoder's gain, `objective.spectral_weight` at α 1) and one R², and each value at each t; and R² per band of spatial frequency on the latent, by the period in tokens: over three tokens (beyond a 3×3 conv's reach), one to three, inside one token, each its error's share over its share of the latents' energy, per t and over the grid.
- At every eval, of the tracked matrices: the global Frobenius norm and its ratio to the init norm, and per role (the parameter name without the block index, prefixed by the block's mixer) the stable rank `‖W‖²_F / σ²_max` over the shorter side, averaged over the role's blocks; and per role the norm of each matrix's step before the lift, `‖unit_polar(moment)‖`, averaged over the role's blocks.
- Logs: plain JSONL files, the record: one line per `log_every` steps, the reads averaged over them on the GPU. The same values and the eval grid stream to wandb for live curves.
