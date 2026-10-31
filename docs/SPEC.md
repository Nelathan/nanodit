# tiny-dit — Design Spec

Status: agreed design, nothing implemented. Every change to this file is a design change and goes to Daniel.
Notation: rectified flow, `x_t = (1 - t)·x + t·ε`, `t = 1` is pure noise, velocity `v = ε - x`.

---

## 1. Data

| Item | Decision |
|---|---|
| Dataset | `jackyhate/text-to-image-2M`, the `data_512_2M` WebDataset shards (512×512 crops, ~2M pairs) |
| Resolutions | 128 px and 256 px, resized from the 512 crops. 512 px skipped for now |
| Captions | full caption, no truncation |
| Text encoder | `google/embeddinggemma-2`, text path only (`vision_config=None`, `audio_config=None`), bf16 (never fp16) |
| Text prefix | one fixed prefix for training and inference: `title: none \| text: {caption}` |
| Embedding | mean-pooled, projected 768-d, L2-normalized as the model returns it |
| VAE | `ostris/primal_vae`, variant `2x2`: 2 channels at 2× downsample, deterministic, unit-normal per image and channel, no scale/shift factors. Image sides must be multiples of 16 |
| Precompute | latents encoded separately per resolution (a 128 px latent is not a downsampled 256 px latent) and embeddings computed once, stored bf16 |
| Storage | latents 16 KB (128 px) + 64 KB (256 px) per image, ≈160 GB for 2M; embeddings 1.5 KB per image, ≈3 GB |
| Augmentation | horizontal flip applied to the latent directly. Primal was trained with latent flip/rotation equivariance, so no re-encode |
| Shapes | square crops per stage, so shapes are static (CUDA-graph friendly via a preallocated input buffer) |

Known risk: training captions are paragraph-length, inference prompts are short. If short prompts get ignored,
add a second embedding per image from the first sentence and sample 50/50 (another ~3 GB). Not built now.

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
| Size | ≈30M parameters at 12 layers |

Not used: linear attention (256 tokens make softmax cheap), MLA (no KV cache), DeepSeek sparse attention (no long context).

## 4. Objective: x0 velocity

- Network output `F`. Target `T = x_t - v = x + (1 - t)(x - ε)`.
- Loss: `‖F - T‖²`, unweighted. This is identically `‖v̂ - v‖²` with `v̂ = x_t - F`.
- The sampler receives `v̂ = x_t - F`; any flow-matching sampler runs unchanged.
- Timestep sampling: logit-normal. Shift grows with resolution, scaled by √(token ratio) per stage.
- Prompt dropout: 10%, replaced by a learned null embedding → classifier-free guidance at sampling.

Understanding (worked out in discussion, not empirically tested here):
x0 velocity is velocity prediction with a hardwired output skip, i.e. preconditioning with `c_skip = 1, c_out = -1`.
Same objective, same optimum. Any benefit comes from inductive bias. The "image-shaped at every t" claim holds only
partly: the target carries a `(1 - t)·ε` term, largest near the clean end.

## 5. Initialization

Default: fan-in normal, `std = 1/√fan_in`.

| Role | Init | Usui target-norm override |
|---|---|---|
| Patch embed 128→512 | fan-in normal (inputs are already unit-normal) | no |
| Timestep MLP | fan-in normal | no |
| Text projection 768→512 | fan-in normal on input scaled by √768 (unit variance per dim) | no |
| Null embedding | random unit vector (same distribution as real embeddings) | no |
| Q, K, V | fan-in normal | no |
| Sigmoid gate projection | fan-in normal (σ ≈ 0.5 at start) | no |
| Attention out-proj | fan-in normal | no |
| SwiGLU up / gate / down | fan-in normal | no |
| adaLN shift / scale rows | small random | no |
| adaLN **gate** rows (shared projection and per-block table) | zero | **yes** |
| Depthwise conv | zero | **yes** |
| Final linear 512→128 | zero | **yes** |
| QK-RMSNorm gains, final norm gain | ones | n/a (assumed handled by Usui's 1-D fallback path; verify) |

Override target for a zero-initialized role: the norm a fan-in init of the same shape would have.
Consequence of zero output with x0 velocity: `F = 0` gives `v̂ = x_t` at step 0 (plain velocity would give `v̂ = 0`).

## 6. Optimizer: UsuiTrack

- Trust-region LR relative to weight norm; per-role target-norm override for zero-init roles (§5).
- Reference point: LFM runs use `lr 4e-5 × norm`, beta 0.95, at 16k tokens per step.
- **LR is the first experiment.** 4e-5 relative was right for adapting a pretrained LFM. From scratch, typical
  recipes imply per-step relative updates of roughly 5e-3 to 5e-2 (estimate, not measured on Usui). At 4e-5 a gate
  needs ≥25k aligned steps to reach its target norm; around 1e-2 it needs ~100.
- Sweep at 128 px: `1e-3, 3e-3, 1e-2`.
- At 128 px, batch 256 gives 64 × 256 ≈ 16k tokens per step, matching the LFM setup on tokens.
  Open whether tokens are the right noise unit for diffusion (t sampling adds its own gradient noise).

## 7. Training schedule

1. 128 px — LR sweep, then the main run.
2. 256 px — continue from the 128 px checkpoint, timestep shift raised.

Throughput estimate (≈30M params, 6·params·tokens FLOPs per image, ~10 TFLOPS effective, unbenchmarked):
~900 img/s at 128 px, ~200 img/s at 256 px. One epoch over 2M images: ~40 min at 128 px, ~3 h at 256 px.

## 8. Sampling and eval

- Sampler for eval: Euler, 30 steps, CFG. Guidance scale tuned on the grid.
- dpmpp_2m with the beta scheduler is allowed for best-looking samples, separately from the eval grid.
- Eval grid: fixed prompts × fixed seeds, every N steps.
- Loss logged bucketed by t (averaged loss stays flat and says little).
- Logs: plain JSONL files, readable without wandb.
