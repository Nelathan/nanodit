# Plan history: the lr sweeps, model growth, the width runs and cross attention

2026-11-12. The open-questions file had grown into a diary in four places. Decided with this move: width 768, with the
model growing in depth; cross attention in the attention blocks as design; usui lr 4e-3. The sections follow as they
stood, as the record of the reads behind those calls; PLAN keeps only what is still open.

## C2. Which LR from scratch, and how long a warmup?

The lr is now the relative step, on weights under a norm ceiling (SPEC §6, archive 2026-11-12); what follows was measured
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

## F3. Cross attention to the token states

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

## G1. Growing the model

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

## G5. What the width runs left open

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

A single linear now, then an MLP, identity at init, on Usui. `128_d768x12_solver1_ep1` (branch `arm/solver`): one change from `_slice_ep1`, a tied 768 x 768 bias-free linear on the rms-normed stream applied 4 times as a residual (`SOLVER_ITERATIONS`) before the readout, zero-initialised. Molegrad with a Linear called several times is fine under `frame_init="random"` (a scratch comparison against the dense-G path at K = 1, 2, 4, 8 differs by about 1e-4 in every case, no trend with K); under `"range_finder"` it refuses a second call before the first step (`tests/test_molegrad.py`). `128_d768x12_solver1_ep1` ended 1.4816 / 1.4817, +0.0071 against `_slice_ep1` (1.4745), back at the baseline; `tangent_live_fraction` fell from 0.99 to 0.96 and `projected_grad_norm` rose 4 to 5 times (read as near 10x; measured 0.0341 against 0.0078 at step 500, 0.0203 against 0.0043 at 5000 to 7500); The run was judged not that good. Cause not isolated: whether the tied matrix's summed gradient dominates the aggregate norm, or the four-fold reuse hurts the frame, is untested.

Build the full-stream SwiGLU first, as the most powerful form, which needs SwiGLU very much; if it is best, explore the cheaper forms (render-only patch, cached context) to match it. `128_d768x12_solvermlp_ep1` (branch `arm/solvermlp`): one change from `_slice_ep1`, a tied `Solver` of SwiGLU (768 -> 1536 -> 768, zero-init `w_down`) as residual on the rms-normed stream, 4 iterations, on Usui, identity at init (checked). The adaptive stop comes later.

`128_d768x12_solvermlp_ep1` ended 1.4781 / 1.4785: +0.0036 against `_slice_ep1` (1.4745), -0.0026 against the baseline, 8 % slower (771 against 838 img/s). Per checkpoint it trails the slice arm by 0.003 to 0.005 throughout. The slice stays the best arm; the solver arms did not beat it. Open: whether anything else is worth trying, or whether the original design with an identity init would match the slice.

The extras (deep supervision of the slice, an untied extra SwiGLU) are not necessary; do the main check now. `128_d768x12_idfinal_ep1` (branch `arm/idfinal`): one change from `_slice_ep1`, a learnable `final` Linear(768, 128) on the fallback optimizer initialised as the selection matrix [I | 0] (output equals the input latent at init, checked). It separates the identity init from removing the learnable readout. A slice seed repeat (noise estimate) is not queued.
`128_d768x12_idfinal_ep1` was stopped by hand at step ~2650 (killed, no final score). Eval caption loss at 1000 / 2000 / 2500: 1.5654 / 1.5242 / 1.5146, against baseline 1.5674 / 1.5236 / 1.5134 and slice 1.5679 / 1.5210 / 1.5087: best at step 1000, level with the baseline from 1500, 0.006 behind the slice at 2500. Fallback grad norm about 0.10 against 0.04 for the slice.

Decided 2026-11-03: the slice readout becomes the base model (commit `56b4515`, branch `slice-base`), with the conv dropped as useless, RoPE on every head dim, and a deeper MLP (SwiGLU hidden 4 x width; the earlier remark to increase the MLP to 8 in the model-growth question was not read as a ratio, so 4 is a default, not decided, and a single constant, `MLP_RATIO`). The run is `128_d1024x16_5ep` (its `config.toml`): d1024, 16 layers, 293.8M parameters, usui lr 1e-2 constant for 5 epochs with no anneal, to decide how to go on from there, fallback 2.5e-3, rank fraction 0.1 (0.05 would be possible again since there is no bottleneck, but it is not worth it), batch 256, 37,605 steps, 5 % warmup of the first epoch, checkpoint 2 x per epoch (every 3750 steps), eval every 1000. The first 250 steps ran at 330 img/s, so about 8 hours plus evals. Not decided and left open: the final anneal; F1 (RoPE base 10000 kept; with every dim rotated, the slowest frequencies move under 0.001 rad across a 16-position axis and act as position-free dims); a seed repeat of the slice arm.
