# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds measured numbers; `archive/` holds
everything closed. See `AGENTS.md` for how this file is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs or code. Each "how to settle" is a proposal, not an
agreed arm.

---

## Group A. Usui wiring

### A2. How does Usui treat 1-D params?

Norm gains (QK-RMSNorm, final norm), biases and the null embedding are not
matrices. Usui rejects any `ndim != 2` param (`optimizer.py` `add_param_group`); they
go to a caller-owned fallback (AdamW or `StochasticAdamW`, Usui SPEC). Settled by code
reading, and the harness runs that split (`src/nanodit/optim.py`). Ready to close.

---

## Group B. Data and conditioning

### B1. Are Primal 2x2 latents unit-normal per image and channel?

The model card says so, and SPEC §1 relies on it (no scale/shift factors, patch
embed init for unit-normal inputs). Settle: a script that encodes a sample of
text-to-image-2M at 128 px and 256 px and reports per-image, per-channel mean and
std. Evidence: FACTS "Primal 2x2" (statistics, flips, stability under small changes). Ready to close: whether the measured spread is close enough to the card's claim.

### B2. What is EmbeddingGemma 2's output norm?

SPEC §1 stores the embedding L2-normalized as the model returns it; §3 feeds it
to the text projection through `rms_norm`, which is ×√768 for a unit-norm
embedding, and §5 initializes the null embedding from the encoder's empty caption.
Unsure the model returns unit norm. Settle: a script that embeds a
sample of captions in bf16 and fp32 and reports the norm distribution. Evidence:
FACTS "EmbeddingGemma 2": unit norm comes from the Normalize module; per dim, the
√768-scaled embedding is far from unit variance. Ready to close.

### B3. Do short prompts get ignored?

Training captions are paragraph-length, inference prompts are short. Settle: judge
it on the eval grid with short prompts. If they are ignored, add a second
embedding per image from the first sentence and sample 50/50 (another ~3 GB).
Evidence: FACTS "EmbeddingGemma 2" (cosines of short prompts against captions); no generation tested.

### B4. Flip augmentation, if ever needed

Parked: no augmentation now. Primal's decoder is flip- and rot90-equivariant, its encoder
is not (FACTS, Primal 2x2), so a flipped latent decodes correctly but is not on the
encoder's distribution. Settle, if more data diversity is needed: re-encode flipped
images, or test whether training on flipped latents hurts. Evidence: FACTS only.

### B5. Which downsampler for any re-encode and for 256 px?

The 128 px latents were encoded from a torch bicubic antialiased resize of the 512 crops. Lanczos keeps more of the top two frequency bands (FACTS, Downsampler). Decided 2026-11-04: keep the current latents; Lanczos is the better resizer if the data are ever re-encoded or the 256 px stage is made, and a Lanczos arm comes later. Open:
- Whether Lanczos latents train or sample better than bicubic ones at equal steps, at 128 px (a re-encode of a shard subset and one paired arm would say).
- Whether the 256 px stage should resize from the 512 crops with Lanczos in the same precompute that makes its latents.

---

### B6. The data sets and the encode

Its own topic doc: `topics/datasets-and-encode.md` (the sets, the fused encode, the 512-d token states, Fine-T2I and its buckets). State: text-to-image-2M is encoded fused at 128 and 256 px; Fine-T2I is staged and encoded in part; the per-bucket shuffle and stream are not built.

### B7. Which caption prefix should the pooled embedding carry?

Every stored pooled embedding is encoded with the document prefix `title: none | text: ` (SPEC §1; checked against a fresh encode, FACTS). On image retrieval that prefix ranks the caption's own image first for 90.5% of 1000 eval images, no prefix for 94.4% (FACTS, caption prefix). The 2026-11-07 read concluded that the stored embeddings stay because they were taken to have no prefix; that premise was wrong, so the conclusion is open again.
- Whether retrieval of the image by the caption says anything about the embedding as conditioning: a paired arm on a subset re-encoded without the prefix would say (pooled embedding and null embedding both; the 512-d token states already drop the prefix tokens but are computed with it in context).
- The cost of changing it: every pooled embedding encoded again (text only, no latents), and SPEC §1 and §5.

## Group C. LR sweep and step budget

### C1. Batch size and step budget at 256 px

At 128 px the batch is 128 (archive 2026-11-02, batch 128), 64 × 128 ≈ 8k tokens per step, and the step budget
follows from the data, each image seen once (archive 2026-11-01, final frame and smoke settings). Open for 256 px: the batch, and whether tokens are the right noise
unit for diffusion (t sampling adds its own gradient noise). A smaller batch may want a lower lr, and a lower rank
fraction if `tangent_live_fraction` falls; it did not fall at batch 64 (FACTS, 1000-step ablations). 256 px
throughput is unmeasured. Settle: when the 256 px stage is planned. At 128 px, after the sweep, going back to batch 256 is
an option; about 140 ms of each step is fixed, so batch 256 runs at about 850 img/s against 580 (FACTS, Sweep throughput).

When to add 256 px is open again (2026-11-05). On 2026-11-02 it was too early, with still no hint of a
cat; `128_d1024x16_swpin_5ep` now follows prompts at class level (FACTS). It needs the images downloaded again and a
precompute (B5: Lanczos from the 512 crops), and it interacts with G2: a patch over 4× the latent area keeps 256 px at
the current 64 tokens. The same data may reach a limit where it stops generalizing; 128 px does not look saturated yet
(2026-11-05). The second image dataset (G1), when it comes: Fine-T2I is the candidate found (REFERENCES,
Data); it streams, so it would be encoded shard by shard as text-to-image-2M was.

### C2. Which LR from scratch, and how long a warmup?

The lr is now a constant relative step on pinned weights (SPEC §6, archive 2026-11-05); what follows was measured
under Usui's fixed step, where a moment that fills its rank writes `‖ΔW‖ = lr·√(max(1, out/in)·r)`. The smoke config takes 2e-3 with rank fraction 0.1
and beta 0.99, chosen for now after 2e-2 read too high (archive 2026-11-01, final frame and smoke settings; eval grid, beta 0.99). The
negative `grad_moment_cosine` followed beta more than lr (FACTS, Runs at usui lr 2e-3). Settle: the sweep at 128 px, usui lr 2e-4, 1e-3, 2e-3 and 4e-3, one pass each over the deduped shards 0–46
at batch 128, warmup 5% of the run, then without decay (archive 2026-11-02, LR sweep); the runs now decay linearly (SPEC §6). The 300-step reads at warmup 20, 50 and 100 are in FACTS (Fixed-step runs). The final linear, the conv, the adaLN tables and the null embedding
train on the fallback at `fallback_lr` with betas (0.9, 0.99), a second sweep axis; they move
faster than the Usui matrices, and the null embedding turns away from its init in the
first 1000 steps (FACTS, Runs at usui lr 2e-3). Suggested: lower the fallback lr, and probably its beta, unless it
evolves fine as it is. No AdamW reference run. The 1000-step ablation (FACTS,
1000-step ablations; one seed): fallback lr 3e-4 is worse than 1e-3 by 0.014 overall and
in every t bin (0.005 to 0.044), near the 0.011 yardstick there; betas (0.9, 0.999)
cannot be told from (0.9, 0.99). Sweep (FACTS, LR sweep and seed merge): loss falls with lr, 0.009 from 1e-3 to
2e-3 and 0.004 from 2e-3 to 4e-3; 2e-3 looks best even though 4e-3 has the better loss. 4e-3 also carries the
larger common mode in the residual stream (see the common-mode question below). 2e-4, rerun with the fix, ends 0.046 above 2e-3 on the Lunara score: too small, not too big. Reading, 2026-11-02: focus
on progress over aesthetics, so 4e-3 is the best and could even be higher. Suggested: decay epoch-wise, because 4e-3
probably had a head start and the model later needs softer touches. Running:
`128_4e-3_ep2`, the 4e-3 run's weights and optimizer state, one more pass at half lr (usui 2e-3, fallback 5e-4). Then
`128_4e-3_ep3`: the same lr at double the batch, one pass at batch 256, its records, evals and checkpoints
every 128k images as before.

The continuation of `swpin_5ep` runs at 1e-4 flat (archive 2026-11-05, continuation lr) and learns about 6× slower per
step than the end of `swpin_5ep` (FACTS, Continuation lr arms). It still progresses, just slower, so it runs on;
4e-4 is probably very good next time even if it does not look good at first (2026-11-05). A
2000-step arm after an lr change mostly shows the noise floor moving, so it favours the lowest lr. The read that would
rank lrs for continued learning, none agreed: flat arms long enough to pass the transient (about 5k steps), ranked
by the score slope after it; optionally a short anneal at the end of the best one, to check the ranking survives it.
Read: `128_d1024x16_swpin_slope_lr{4e-4,1e-3}`, 5000 steps each from the end of `swpin_cont_3ep`, against
`swpin_cont_3ep`'s own last stretch at 1e-4 (FACTS, Continuation lr arms): neither higher lr falls within 5000 steps.

No learning-rate sweep from scratch has been run since the weights were pinned; the sweep above predates the pin and
measured a different update rule. The aim (2026-11-06) is a constant lr for the whole run with an anneal only at the end. The spec's
schedule section still says a linear decay over the whole run, so it differs from this aim until the schedule is decided.

On beta and batch (2026-11-05): higher beta makes `grad_moment_cosine` less negative, with diminishing returns, and beta
cannot turn it positive; only a lower lr can, which stops proper learning. So beta is probably just fine. Increasing
the batch size is more common, but more progress is the aim, and a larger batch probably does not give lower loss per
wall-clock time.

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

### C4. Does an edge-of-stability prediction for momentum optimizers bear on the learning rate chosen?

Its own topic doc: `topics/edge-of-stability.md` (the bubbleforce prediction, the toy, the lr-jump read and what is open). State: the pinned lr is not a stability ceiling, the curvature adapts to it; that favours a constant lr with a late anneal (C2), untested at length.

---

## Group D. Sampling and eval

### D1. EMA weights

Leaning no. Option on the table: render the eval grid from both raw and EMA
weights and drop EMA if they match.

### D2. Guidance scale

Tuned on the eval grid.

### D4. Sampler step schedule

The sampler steps a uniform t grid; training samples a shifted logit-normal t. SD3
shifts the step schedule the same way, which matters once the 256 px stage raises
the shift. Settle: compare uniform and shifted grids on the eval grid.

### D5. FID, and what the lunara sets are for

Keep lunara for FID or for fine-tuning later (2026-11-02). `moonworks/lunara-art-eval` (1,000 art prompts, each
rendered by 8 models, with aesthetic, emotion and integrity scores) gives the checkpoint score's eval set: its 1,000
Moonworks Lunara renders (SPEC §8; `eval_set` in the configs, `scripts/precompute_lunara.py`). `moonworks/lunara-aesthetic` (2,000 Lunara images, refined prompts) is held back
for an aesthetic fine-tune or as FID references. Open: when FID starts, on which features (Inception at 128 px is
weak; FD-DINOv2 is the alternative), and against which reference set.

---

## Group F. Model choices SPEC leaves open

Taken as the standard DiT/PixArt choice in `src/nanodit/model.py`; each is
to be confirmed. Settle: by reading, or a short 128 px comparison once training runs.

### F1. RoPE frequency base

Base 10000 with 8 frequencies per axis over at most 16 positions: the slowest four
rotate under 0.15 rad across the whole grid, so about half the rotated dims carry
almost no position. A base near 100, or frequencies log-spaced from 1 to 1/16,
would use them.

### F2. adaLN shift/scale init scale

SPEC says "small random". Std 0.02 on the shared 512→4096 projection makes the
blocks' and the final layer's shift/scale far from zero at init (FACTS, init), so
`1 + scale` starts far from 1. A fan-in-scaled or 10× smaller init would start near
identity.

### F3. Cross attention to the token states

Every block gets a cross-attention sublayer between the self-attention and the MLP: queries
from the stream, keys and values from the encoder's 512-d token states (the pre-projection states of the sequence
columns, rms-normalised once), a per-head q/k norm, no RoPE, a post-norm like the other branches, padded tokens masked. Its shift,
scale and gate rows extend the adaLN table (nine rows now; the gate starts at zero like the others). The pooled embedding
keeps conditioning the adaLN. The null prompt is the pooled null embedding with the two tokens of the empty caption as
learned parameters (`null_tokens` in `null_embedding.safetensors`), padded to the batch's length. Prompt dropout and
guidance replace both. These details are *Not yet decided*. Settle by a paired trial at d768x12
(`128_d768x12_swpin_3k` pooled only, `..._cross_3k` with cross attention), then the choices: every block or some,
whether the pooled path stays, the token width.

Read 2026-11-06 (FACTS, cross attention and conv period): `..._cross_3k` ends 0.017 behind the pooled-only baseline and 28 % slower. Reading: this
decides nothing, since following a sequence is much harder and cannot be learned in 3000 steps; the pooled-only model followed prompts
oddly after five epochs, which is a young model or the pooled vector conflating concepts. Changes: the cross attention gets no
adaLN rows, and the cross, self and conv mixers may run in parallel into one output projection (topic doc `topics/block-design.md`, which
holds the arms and their status).

### F4. Block layout: mixers, parallel blocks, RoPE and head sharing

The list of 2026-11-06, in `topics/block-design.md` with the ledger: meta-only patch embed at one primal block per token, parallel
mixers with one meta-only output projection, a sparse layout of conv layers with an occasional cross-and-self layer, a PaLM-style parallel
MLP, RoPE against partial RoPE against NoPE, MHA against GQA for self-attention (MHA is not cheap in runtime and a 1-megapixel image
costs far more), GQA and NoPE for the cross attention. Base: conv in 3/4 of the blocks.

---

## Group G. Scale and signal propagation

### G1. Growing the model

State (2026-11-11): width 768 × depth 12 is the baseline for trials. Width 512 was inefficient; 1024 is efficient but not fast enough for trial arms. The size of the long run is open.

Width comes first, since d matters more than depth; the SwiGLU hidden stays at 2d until the step budget can fill it,
and once enough steps go in, the MLP can grow to 8. Grid on the table: d 512, 768, 1024 × depth 12,
24, 32; params, FLOPs and an activation-memory guess per cell were estimated, not measured. Settle: a 30-step
microbench per size for img/s and peak memory, then a run at the sweep's lr. Open with it: whether Usui's lr
carries over in width (its write has spectral norm lr·√max(1, out/in), the µP condition for out ≥ in), and
whether the fallback's final linear wants lr ∝ 1/width, as µP gives AdamW's readout.

On the guidance grid (2026-11-02): the model learns fine, it is just tiny, and 128 px at 64 tokens is very low. Nothing might
show at that setting, or from this model at all, since it may lack a bigger MLP, more width or depth. Training
should get faster before the model or the data scale up (the throughput group).

For the night of 2026-11-02: d768 × depth 12 first; the MLP stays at 2d until the big run is planned, with the second
image dataset added and 256 px prepared, and the MLP scales accordingly then. Each 4 epochs at batch 256 from the
start, usui lr 8e-3, 4e-3, 2e-3, 1e-3 by epoch, fallback a quarter of it, 5 % warmup in the first. On 2026-11-03 the
depth arm (d512 × 24) was replaced by d1024 × 12, same schedule, because it is more interesting to explore one axis
first. The depth arm waits (see the depth-scaling question below).

### G2. Patch size

Patch 8 gives 64 tokens at 128 px. Patch 4 gives 256: about 4× the FLOPs at the same parameters, the axis DiT
found as strong as model size (XL/2 against XL/8). Not measured here.

The other direction (2026-11-05): a patch embed over 4× the latent area (`pixel_unshuffle(16)`, 512-d
tokens), so the latent fills half of the 1024-wide stream and the slice readout is 512 channels; 16 tokens at 128 px, 64
at 256 px. With it, whether the patch embed writes only the meta channels (the stream beyond the latent slice, the
slice carrying the latent unchanged) or all channels as now (SPEC §3: identity on the slice, learned everywhere); meta only
is preferred. A model change, so a fresh run, not a continuation (archive 2026-11-05, continuation lr).

Decided 2026-11-05: try it, if the cost is not too bad. A patch embed that just fills in the missing columns as meta
channels should be an obvious success. The original idea was 128/512 and the stream is now 1024 wide, so to be bold try
512/1024: the current setup works well but may be overprovisioned, and the 2-channel VAE was picked to have more real
estate. Other models: DiT, PixArt, SD3
and Flux take 16 px per token (f8 VAE, patch 2), SANA 32 px (f32 DC-AE, patch 1); the 512/1024 arm takes 32 px.
Measured: `128_d1024x16_p16meta_3k` (branch `g2-patch16`, meta-only patch embed), FACTS (Architecture arms).

Reading, 2026-11-06: that arm packed four primal blocks into each token ("4 blocks per latent"), which was the bad part, and it combined the
token size with the meta-only write. The token stays one primal block (8 px, 128 dims; patch 12 does not divide the 64×64 latent). The
untested arm is the meta-only write alone at 8 px (`topics/block-design.md`).

### G3. The common mode in the residual stream

By mid-depth most of each image's stream energy is one vector shared by its tokens, written by attention in blocks
0–1; the token-specific part keeps its rank (FACTS, Token signal). Unsure whether it costs anything: each branch's
RMSNorm then sees the token-specific part at about a quarter of unit scale. Candidates that fit the architecture:
register tokens, which give global state its own slots; nothing else proposed yet. Settle: a register arm against the
base, read on loss, the eval grid and the same diagnostic. Background (Li's notes ch. 5–6, the shaped Transformer,
CompleteP): at init the zero gates make every block the identity, so their rank-collapse results do not apply there;
SwiGLU measured decorrelating, so shaped activations have no target here.

### G4. Depth scaling, if depth arms run

CompleteP scales each block's branch by 1/L for lr transfer across depth, derived for AdamW. Usui's fixed-norm write
and the per-sample gates may absorb it. Settle: only if a depth-24/32 arm is run, by lr transfer from depth 12.

### G5. What the width runs left open

From the d768 and d1024 renders (2026-11-03): concepts emerge there and not at d512; the warrior keeps the
prompted black and white, and the red accent is missing; the ethereal prompt misses its gold; the model cannot yet
draw a human form. The embedding has it; the model just cannot render it yet, and it is unknown whether the cause is
`ada`, the dataset or general rendering capability. Loss does not measure prompt following (the caption-versus-null gap stayed
near 0.002 at every width, FACTS).

Calls and candidates, in the order given:
- Constant usui lr until a final anneal, assumed rather than tested: each halving of the lr roughly halved the epoch's
  gain (FACTS, d768 and d1024 rows). The next best-guess long run is d1024 (throughput was good).
- Conditioning: one `ada` linear per block in place of the shared one (cheap in FLOPs, +6d² parameters per block); or a
  shared one with a LoRA per block (Anima), judged worse at this scale. A one-key attention head (v only, no q or k)
  adds `W_v c` to the packed heads before the out projection: a per-block, gated shift.
- The final linear wants about 0.1× the fallback lr, as LLM embeddings do, since it anchors the backbone. Epoch 1
  tries either a rank-64 frame or the lower fallback lr; later epochs default Usui.
- Conv off: epoch 1 of d768 without the depthwise conv, `128_d768x12_noconv_ep1`, against `128_d768x12_ep1`
  (1.4807 / 1.4811). It runs from branch `arm/noconv`, with the conv deleted; one seed, so the init draw differs from the baseline's.
- Open: whether 64 tokens limit the model; saturation at 128 px before moving to 256.

No per-block `ada`, since it is not scalable and that is the job of the table; a shared `ada` that sees one vector per sample may be starved or have stifled gradients, which would explain why it makes no eval-loss difference. SPEC was written from vague ideas and is open to redesign. Zero init stays, being principled, though Usui needs experiments. Usui caps a frame at half the shorter side, so the final linear (128 x 768) tops out at rank 64; a full-rank, no-tracking Usui mode is an option for such bottlenecks. Arms against `128_d768x12_ep1` (8e-3, conv kept), one change each: `128_d768x12_noconv_ep1` (1.4805 / 1.4807, done), `128_d768x12_lr1e-2_ep1` (usui lr 1e-2; a constant 1e-2 is proposed), `128_d768x12_finalr64_ep1` (final linear in Usui at rank 64, branch `arm/finalr64`). Staying at d768 (about 40 min per epoch).

Read 2026-11-03: conv off 1.4805 / 1.4807, equal to the baseline within noise and 2.5% faster; the conv stays until it is decided. `128_d768x12_lr1e-2_ep1` ended 1.4799 / 1.4797 (-0.0008 / -0.0014, never worse): the 1e-2 lr was good. `128_d768x12_finalr64_ep1` ended 1.4911 / 1.4914 (+0.0104; gap to baseline 0.051 at step 1000, 0.017 at 3000, 0.013 at 5000): the gap is shrinking, so the final linear stays on the fallback optimizer for now, to bank loss where possible. Open: whether a final linear is needed at all (readout as a fixed slice of the residual stream); a new run could start from the final layer of a d768 ep4 run. Cross-attention to Gemma token states (needs a re-embed from the stored captions, about 190 GB at ~65 tokens/caption, no redownload; 256 px latents need the images again) is parked, noted as a plan for later; the aim is rather to use a strong model to reach rendering capabilities. Samples show the text is used clearly although the caption/null loss gap is 0.002, so that gap says nothing about `ada` being starved. No `ada` / no text arm and a 4-epoch finalr64 run are not chosen.

Run the simple identity patch embed with no final layer: `128_d768x12_slice_ep1` (branch `arm/slice`), one change from `_ep1` at usui lr 8e-3. The output is the first 128 channels of the residual stream, with no final norm or final `ada` rows; `patch_embed.weight[:128]` starts as the identity, so at init the output equals the input latent (checked: max difference 0.0) and the network's output F = x_t - v starts at v = 0. Stacking the noise with `c` is judged a bolder step beyond this. No-`ada` arm: not worth a run, it probably looks just like cfg 0 now.

`128_d768x12_slice_ep1` ended 1.4745 / 1.4749 against 1.4807 / 1.4811 (-0.0062 / -0.0062; ahead from step 2000 on, by 0.005 to 0.007 at 3000 to 7521; one seed, the other one-change arms moved at most 0.0014). Reading: gradients flow freely through the slice. Next idea: a solver that iterates on the 768 residual until the 128-channel slice may pass; simplest form without conv.

A single linear now, then an MLP, identity at init, on Usui. `128_d768x12_solver1_ep1` (branch `arm/solver`): one change from `_slice_ep1`, a tied 768 x 768 bias-free linear on the rms-normed stream applied 4 times as a residual (`SOLVER_ITERATIONS`) before the readout, zero-initialised. Molegrad with a Linear called several times is fine under `frame_init="random"` (a scratch comparison against the dense-G path at K = 1, 2, 4, 8 differs by about 1e-4 in every case, no trend with K); under `"fit"` it refuses a second call before the first step (`tests/test_molegrad.py`). `128_d768x12_solver1_ep1` ended 1.4816 / 1.4817, +0.0071 against `_slice_ep1` (1.4745), back at the baseline; `tangent_live_fraction` fell from 0.99 to 0.96 and `projected_grad_norm` rose 4 to 5 times (read as near 10x; measured 0.0341 against 0.0078 at step 500, 0.0203 against 0.0043 at 5000 to 7500); The run was judged not that good. Cause not isolated: whether the tied matrix's summed gradient dominates the aggregate norm, or the four-fold reuse hurts the frame, is untested.

Build the full-stream SwiGLU first, as the most powerful form, which needs SwiGLU very much; if it is best, explore the cheaper forms (render-only patch, cached context) to match it. `128_d768x12_solvermlp_ep1` (branch `arm/solvermlp`): one change from `_slice_ep1`, a tied `Solver` of SwiGLU (768 -> 1536 -> 768, zero-init `w_down`) as residual on the rms-normed stream, 4 iterations, on Usui, identity at init (checked). The adaptive stop comes later.

`128_d768x12_solvermlp_ep1` ended 1.4781 / 1.4785: +0.0036 against `_slice_ep1` (1.4745), -0.0026 against the baseline, 8 % slower (771 against 838 img/s). Per checkpoint it trails the slice arm by 0.003 to 0.005 throughout. The slice stays the best arm; the solver arms did not beat it. Open: whether anything else is worth trying, or whether the original design with an identity init would match the slice.

The extras (deep supervision of the slice, an untied extra SwiGLU) are not necessary; do the main check now. `128_d768x12_idfinal_ep1` (branch `arm/idfinal`): one change from `_slice_ep1`, a learnable `final` Linear(768, 128) on the fallback optimizer initialised as the selection matrix [I | 0] (output equals the input latent at init, checked). It separates the identity init from removing the learnable readout. A slice seed repeat (noise estimate) is not queued.
`128_d768x12_idfinal_ep1` was stopped by hand at step ~2650 (killed, no final score). Eval caption loss at 1000 / 2000 / 2500: 1.5654 / 1.5242 / 1.5146, against baseline 1.5674 / 1.5236 / 1.5134 and slice 1.5679 / 1.5210 / 1.5087: best at step 1000, level with the baseline from 1500, 0.006 behind the slice at 2500. Fallback grad norm about 0.10 against 0.04 for the slice.

Decided 2026-11-03: the slice readout becomes the base model (commit `56b4515`, branch `slice-base`), with the conv dropped as useless, RoPE on every head dim, and a deeper MLP (SwiGLU hidden 4 x width; the earlier remark to increase the MLP to 8 in the model-growth question was not read as a ratio, so 4 is a default, not decided, and a single constant, `MLP_RATIO`). The run is `128_d1024x16_5ep` (its `config.toml`): d1024, 16 layers, 293.8M parameters, usui lr 1e-2 constant for 5 epochs with no anneal, to decide how to go on from there, fallback 2.5e-3, rank fraction 0.1 (0.05 would be possible again since there is no bottleneck, but it is not worth it), batch 256, 37,605 steps, 5 % warmup of the first epoch, checkpoint 2 x per epoch (every 3750 steps), eval every 1000. The first 250 steps ran at 330 img/s, so about 8 hours plus evals. Not decided and left open: the final anneal; F1 (RoPE base 10000 kept; with every dim rotated, the slowest frequencies move under 0.001 rad across a 16-position axis and act as position-free dims); a seed repeat of the slice arm.

### G7. Does the narrower timestep distribution hold at hero length and in the samples?

The training t is logit-normal with σ 0.7 (SPEC §4, archive 2026-11-04). The 3000-step arms at d768x12 gave it a small, consistent edge at t 0.26 to 0.66 and a loss at t 0.16 and 0.84 (FACTS, Timestep distribution arms). Open:
- Whether the edge survives a long run, and whether the extremes (t 0.16 and 0.84) fall behind in R² over it: the score grid carries both, every 1000 steps.
- Whether the narrower distribution costs sample quality at the ends (high t sets the composition, low t the sharpness); R² does not say, the eval grids do.
- `128_d1024x16_swpin_5ep` trained at σ 1 (its config predates the default), so the continuation at usui lr 1e-4 (archive 2026-11-05) is the first long run at σ 0.7.

### G8. What the pin leaves open

Every Usui matrix is pinned to its init Frobenius norm and takes a constant relative step (SPEC §6, archive 2026-11-05).
The sandwich norm was necessary against the loss spikes (2026-11-06); no arm checks it again.
Open:
- Whether a learned scalar gain per pinned matrix (direction pinned, gain on the fallback) helps: some scale is free in
  the gates (`out` and the v rows of `qkv`, the `w_up`·`w_down` product), some is not (`w_gate` sets the SiLU operating
  point; `ada`, `time_mlp` and `text_proj` the conditioning scale; `patch_embed` the input scale).
- A pin per output row instead of per matrix.
- Reading: EDM2's magnitude-preserving layers (REFERENCES, Objective).

### G9. Does Usui hold back the rank the weights grow into, and can it be helped?

Its own topic doc: `topics/weight-spectra.md` (the spectra of the pinned five-epoch run against pretrained models, the reading, the candidates). State: open, no arm agreed; the rank 0.5 arm and the dense-gradient probe are not wanted.

### G10. The depthwise conv, placed differently

The conv before attention made no difference at d768 (FACTS) and was dropped; that may have been a design mistake
rather than a verdict on the conv (2026-11-05). Candidates: the conv in place of
attention in some blocks, rather than before it; the conv writing only the meta channels (see the patch-size question); its output stacked into
the MLP input (SANA's Mix-FFN puts a depthwise conv inside the MLP). Settle: one arm per placement against the current
model, at a scale and length agreed when it runs.

Decided 2026-11-05: the conv replaces attention, writing to the meta channels only, the cleanest variant (that would be half the
layers of a 768x12 model); always conv first, attention next. If it succeeds, 2/3 or 3/4 of the blocks are the next thing to try. Of the two forms, (a) the depthwise conv over the meta channels only, (b) over all channels and then `out` into the meta
channels, form (b) was chosen. Writing image channels may be fine, but it could also write context into the render,
whereas meta only would put the SwiGLU gate before it; in earlier samples that compared the canon conv with none, the
image was softer, as if it just copied context into the render. It may take more time. Measured (FACTS, Architecture arms): `128_d1024x16_conv_3k` (branch
`conv-meta`): `swpin_3k` with blocks 0, 2, ..., 14 a depthwise 3×3 conv over all 1024 channels after the adaLN
modulation, then `out` (1024 → 896) into the meta channels, its sandwich norm and the attention gate; the MLP as before.

At d768x12 (FACTS, cross attention and conv period, 2026-11-06): conv in 1/2, 2/3 and 3/4 of the blocks all end level with the pooled-only baseline
at 3000 steps (−0.0003, +0.0004, +0.003) and run 1.20×, 1.28× and 1.32× as fast. 3/4 is preferred: a faster model allows more layers. The conv is
the basic form; next is how GDN2 does it, and writing all channels once the model is smart enough. adaLN stays for the conv layers.

---

## Group H. Harness throughput

### H1. How much of the fixed per-step cost can go?

About 140 ms of each step is fixed, independent of the batch, and the step is launch-bound: the Usui step is about
14,700 of 19,500 launches and syncs 9 times, once per `eigh` (FACTS, Sweep throughput and host syncs). The blocks
are compiled once every Molegrad frame has fit; that cut forward/backward from ~5,400 to ~3,400 launches, a modest
share of the step. Simple blockwise compile worked well in the past; CUDA graphs are also open.

On the Usui-step options (2026-11-02): batching same-shape matrices and one `eigh` for all rank groups was tried before,
but on anima the intermediates run out of memory (usuitrack is built for full finetunes on small VRAM); the diagnostics
could be disabled now, since they are already tuned, though one every 500 steps would be a cheap meter, so noisy it
might not be worth looking at; a sync-free solver for the aim is a longer investigation but still worth a shot, since
in the past only `eigh` gave a stable aim, while with a diagonal aim simpler solutions may work too. Decided
2026-11-02: fitting is the default and pretraining uses random frames; an option gives a pretraining run a random frame
at init with the standard `eta`, no schedule, no refit (calling fit is dead; a geometric `eta` decay helped very short
runs before, but was not worth keeping in the code, since more rank does the same); and the `eigh` calls are stacked.
No random-frame comparison run: by step 1k the fitted frame has fully caught up, with zero difference. Both are usuitrack changes, in the usuitrack `zero-init` checkout. Also open: the pinned, non-blocking copy in `Stream` (4 syncs per step). Block compile measured
on a free GPU: FACTS, Sweep throughput.

---

## Group I. Training schemes beyond flow matching

### I1. Priming: reconstruction pretraining, then distributional post-training

Its own topic doc: `topics/priming.md` (the source, the direction, the arms and their reads, what is open). State (2026-11-11): a side quest; probably not used at all, and if anything then REPA on precomputed targets.
