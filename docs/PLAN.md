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
reading, and the harness runs that split (`src/tinydit/optim.py`). Daniel to close.

---

## Group B. Data and conditioning

### B1. Are Primal 2x2 latents unit-normal per image and channel?

The model card says so, and SPEC §1 relies on it (no scale/shift factors, patch
embed init for unit-normal inputs). Settle: a script that encodes a sample of
text-to-image-2M at 128 px and 256 px and reports per-image, per-channel mean and
std. Evidence: FACTS "Primal 2x2" (statistics, flips, stability under small changes). Daniel to close: whether the measured spread is close enough to the card's claim.

### B2. What is EmbeddingGemma 2's output norm?

SPEC §1 stores the embedding L2-normalized as the model returns it; §3 feeds it
to the text projection through `rms_norm`, which is ×√768 for a unit-norm
embedding, and §5 initializes the null embedding from the encoder's empty caption.
Unsure the model returns unit norm. Settle: a script that embeds a
sample of captions in bf16 and fp32 and reports the norm distribution. Evidence:
FACTS "EmbeddingGemma 2": unit norm comes from the Normalize module; per dim, the
√768-scaled embedding is far from unit variance. Daniel to close.

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

---

## Group C. LR sweep and step budget

### C1. Batch size and step budget at 256 px

At 128 px the batch is 128 (archive 2026-11-02, batch 128), 64 × 128 ≈ 8k tokens per step, and the step budget
follows from the data, each image seen once (archive 2026-11-01, final frame and smoke settings). Open for 256 px: the batch, and whether tokens are the right noise
unit for diffusion (t sampling adds its own gradient noise). A smaller batch may want a lower lr, and a lower rank
fraction if `tangent_live_fraction` falls; it did not fall at batch 64 (FACTS, 1000-step ablations). 256 px
throughput is unmeasured. Settle: when the 256 px stage is planned. Daniel, at 128 px, after the sweep: "honestly we might go back
to bs256 again"; about 140 ms of each step is fixed, so batch 256 runs at about 850 img/s against 580 (FACTS, Sweep throughput).

### C2. Which LR from scratch, and how long a warmup?

The lr is Usui's fixed step (SPEC §6): a moment that fills its rank writes
`‖ΔW‖ = lr·√(max(1, out/in)·r)`. The smoke config takes 2e-3 with rank fraction 0.1
and beta 0.99, Daniel's "for now" after 2e-2 read too high (archive 2026-11-01, final frame and smoke settings; eval grid, beta 0.99). The
negative `grad_moment_cosine` followed beta more than lr (FACTS, Runs at usui lr 2e-3). Settle: the sweep at 128 px, usui lr 2e-4, 1e-3, 2e-3 and 4e-3, one pass each over the deduped shards 0–46
at batch 128, warmup 5% of the run, no decay for now (archive 2026-11-02, LR sweep). Usui's README recommends a
decay; whether to add one stays open after the sweep. The 300-step reads at warmup 20, 50 and 100 are in FACTS (Fixed-step runs). The final linear, the conv, the adaLN tables and the null embedding
train on the fallback at `fallback_lr` with betas (0.9, 0.99), a second sweep axis; they move
faster than the Usui matrices, and the null embedding turns away from its init in the
first 1000 steps (FACTS, Runs at usui lr 2e-3). Daniel: "we should also lower fallback lr i
think. unless you say it evolves fine. probably also its beta." No AdamW reference run. The 1000-step ablation (FACTS,
1000-step ablations; one seed): fallback lr 3e-4 is worse than 1e-3 by 0.014 overall and
in every t bin (0.005 to 0.044), near the 0.011 yardstick there; betas (0.9, 0.999)
cannot be told from (0.9, 0.99). Sweep (FACTS, LR sweep and seed merge): loss falls with lr, 0.009 from 1e-3 to
2e-3 and 0.004 from 2e-3 to 4e-3; Daniel: "2e-3 looks best even though [4e-3] has better loss". 4e-3 also carries the
larger common mode in the residual stream (G3). 2e-4, rerun with the fix, ends 0.046 above 2e-3 on the Lunara score: too small, not too big. Daniel, 2026-11-02:
"Even if 2e-3 was looking better aesthetically lets focus on progress, so 4 is best and could even be higher"; "Why no
decay? I think 4e-3 had a headstart but later the model needs softer touches. I would decay epoch wise". Running:
`128_4e-3_ep2`, the 4e-3 run's weights and optimizer state, one more pass at half lr (usui 2e-3, fallback 5e-4). Then
`128_4e-3_ep3`: Daniel, "Next epoch is same lr but double bs", one pass at batch 256, its records, evals and checkpoints
every 128k images as before.

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

---

## Group D. Sampling and eval

### D1. EMA weights

Daniel leans no. Option on the table: render the eval grid from both raw and EMA
weights and drop EMA if they match.

### D2. Guidance scale

Tuned on the eval grid.

### D4. Sampler step schedule

The sampler steps a uniform t grid; training samples a shifted logit-normal t. SD3
shifts the step schedule the same way, which matters once the 256 px stage raises
the shift. Settle: compare uniform and shifted grids on the eval grid.

### D5. FID, and what the lunara sets are for

Daniel (2026-11-02): "keep lunara for fid or finetune later". `moonworks/lunara-art-eval` (1,000 art prompts, each
rendered by 8 models, with aesthetic, emotion and integrity scores) gives the checkpoint score's eval set: its 1,000
Moonworks Lunara renders (SPEC §8; `eval_set` in the configs, `scripts/precompute_lunara.py`). `moonworks/lunara-aesthetic` (2,000 Lunara images, refined prompts) is held back
for an aesthetic fine-tune or as FID references. Open: when FID starts, on which features (Inception at 128 px is
weak; FD-DINOv2 is the alternative), and against which reference set.

---

## Group F. Model choices SPEC leaves open

Taken as the standard DiT/PixArt choice in `src/tinydit/model.py`; each is Daniel's
to confirm. Settle: by reading, or a short 128 px comparison once training runs.

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

---

## Group G. Scale and signal propagation

### G1. Growing the model

Daniel: width first ("d is more important than depth"); the SwiGLU hidden stays at 2d until the step budget can
fill it ("once we put enough steps in, we can increase mlp to 8"). Grid on the table: d 512, 768, 1024 × depth 12,
24, 32; params, FLOPs and an activation-memory guess per cell were estimated, not measured. Settle: a 30-step
microbench per size for img/s and peak memory, then a run at the sweep's lr. Open with it: whether Usui's lr
carries over in width (its write has spectral norm lr·√max(1, out/in), the µP condition for out ≥ in), and
whether the fallback's final linear wants lr ∝ 1/width, as µP gives AdamW's readout.

Daniel, 2026-11-02, on the guidance grid: "it learns fine, its just tiny and 128 at 64token is just verY low. We might
not see anything at rhat setting. We might not see anything from this model anyway, might lack bigger mlp, more width
or depth." And: "We should make it faster training before we scale up model or data" (H1).

Daniel, 2026-11-02, for the night: d768 × depth 12 first ("768 first"); MLP stays at 2d ("just when we plan the big
run and add the second image dataset and prepare 256px too, then we also scale mlp accordingly"). Each 4 epochs at
batch 256 from the start, usui lr 8e-3, 4e-3, 2e-3, 1e-3 by epoch, fallback a quarter of it, 5 % warmup in the first.
Daniel, 2026-11-03: the depth arm (d512 × 24) is replaced by d1024 × 12, same schedule: "i think its more interesting
to first explore one axis". The depth arm waits (G4).

### G2. Patch size

Patch 8 gives 64 tokens at 128 px. Patch 4 gives 256: about 4× the FLOPs at the same parameters, the axis DiT
found as strong as model size (XL/2 against XL/8). Not measured here.

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
and our per-sample gates may absorb it. Settle: only if a depth-24/32 arm is run, by lr transfer from depth 12.

### G5. What the width runs left open

Daniel, 2026-11-03, from the d768 and d1024 renders: concepts emerge there and not at d512; the warrior keeps the
prompted black and white, and the red accent is missing; the ethereal prompt misses its gold; the model cannot yet
draw a human form. "Trust me the embedding has it the model just cant render it yet. We dont know if its ada or
dataset or general rendering capabilities." Loss does not measure prompt following (the caption-versus-null gap stayed
near 0.002 at every width, FACTS).

Calls and candidates, in the order Daniel gave them:
- Constant usui lr until a final anneal, assumed rather than tested: each halving of the lr roughly halved the epoch's
  gain (FACTS, d768 and d1024 rows). The next best-guess long run is d1024 (throughput was good).
- Conditioning: one `ada` linear per block in place of the shared one (cheap in FLOPs, +6d² parameters per block); or a
  shared one with a LoRA per block (Anima), judged worse at this scale. A one-key attention head (v only, no q or k)
  adds `W_v c` to the packed heads before the out projection: a per-block, gated shift.
- The final linear wants about 0.1× the fallback lr, as LLM embeddings do, since it anchors the backbone. Epoch 1
  tries either a rank-64 frame or the lower fallback lr; later epochs default Usui.
- Conv off: epoch 1 of d768 without the depthwise conv, `128_d768x12_noconv_ep1`, against `128_d768x12_ep1`
  (1.4807 / 1.4811). It runs from a worktree with the conv deleted; one seed, so the init draw differs from the baseline's.
- Open: whether 64 tokens limit the model; saturation at 128 px before moving to 256.

Daniel (2026-11-03, after the conv-off run): no per-block `ada` ("not scalable, that's the job of the table"); a shared `ada` that sees one vector per sample may be starved or have stifled gradients, which would explain why it makes no eval-loss difference. SPEC was written by Claude from vague ideas and is open to redesign. Zero init stays ("principled; usui needs experiments"). Usui caps a frame at half the shorter side, so the final linear (128 x 768) tops out at rank 64; a full-rank, no-tracking Usui mode is an option for such bottlenecks. Arms against `128_d768x12_ep1` (8e-3, conv kept), one change each: `128_d768x12_noconv_ep1` (1.4805 / 1.4807, done), `128_d768x12_lr1e-2_ep1` (usui lr 1e-2; constant 1e-2 is his proposal), `128_d768x12_finalr64_ep1` (final linear in Usui at rank 64, worktree `../tiny-dit-finalr64`); queue `runs/queue_g5_2026-11-03.sh`. Staying at d768 (about 40 min per epoch).

Daniel (2026-11-03, after the three arms): conv off 1.4805 / 1.4807, equal to the baseline within noise and 2.5% faster; conv kept until he decides. `128_d768x12_lr1e-2_ep1` ended 1.4799 / 1.4797 (-0.0008 / -0.0014, never worse): "the 1e-2 lr was good". `128_d768x12_finalr64_ep1` ended 1.4911 / 1.4914 (+0.0104; gap to baseline 0.051 at step 1000, 0.017 at 3000, 0.013 at 5000): the gap is shrinking, so he keeps the final linear on the fallback optimizer for now ("bank loss where we can"). Open from him: whether a final linear is needed at all (readout as a fixed slice of the residual stream); a new run could start from the final layer of a d768 ep4 run. Cross-attention to Gemma token states (needs a re-embed from the stored captions, about 190 GB at ~65 tokens/caption, no redownload; 256 px latents need the images again) is parked: "we note that plan for later; we will rather try to use a strong model to reach rendering capabilities." Samples show the text is used clearly although the caption/null loss gap is 0.002, so that gap says nothing about `ada` being starved. No `ada` / no text arm and a 4-epoch finalr64 run were offered, not chosen.

Daniel (2026-11-03, later): run the simple identity patch embed with no final layer: `128_d768x12_slice_ep1` (worktree `../tiny-dit-slice`, `runs/queue_slice_2026-11-03.sh`), one change from `_ep1` at usui lr 8e-3. The output is the first 128 channels of the residual stream, with no final norm or final `ada` rows; `patch_embed.weight[:128]` starts as the identity, so at init the output equals the input latent (checked: max difference 0.0) and the network's output F = x_t - v starts at v = 0. He thinks stacking the noise with `c` is a bolder step beyond this. No-`ada` arm: not worth a run, "probably looks just like cfg 0 now".

`128_d768x12_slice_ep1` ended 1.4745 / 1.4749 against 1.4807 / 1.4811 (-0.0062 / -0.0062; ahead from step 2000 on, by 0.005 to 0.007 at 3000 to 7521; one seed, the other one-change arms moved at most 0.0014). Daniel's reading: gradients flow freely through the slice. Next idea from him: a solver that iterates on the 768 residual until the 128-channel slice may pass; simplest form without conv.

Daniel (2026-11-03, later): "Single linear now, then an mlp", identity at init, on Usui. `128_d768x12_solver1_ep1` (worktree `../tiny-dit-solver`, `runs/queue_solver1_2026-11-03.sh`): one change from `_slice_ep1`, a tied 768 x 768 bias-free linear on the rms-normed stream applied 4 times as a residual (`SOLVER_ITERATIONS`) before the readout, zero-initialised. Molegrad with a Linear called several times is fine under `frame_init="random"` (a scratch comparison against the dense-G path at K = 1, 2, 4, 8 differs by about 1e-4 in every case, no trend with K); under `"fit"` it refuses a second call before the first step (`tests/test_molegrad.py`). `128_d768x12_solver1_ep1` ended 1.4816 / 1.4817, +0.0071 against `_slice_ep1` (1.4745), back at the baseline; `tangent_live_fraction` fell from 0.99 to 0.96 and `projected_grad_norm` rose 4 to 5 times (Daniel read "near 10x"; measured 0.0341 against 0.0078 at step 500, 0.0203 against 0.0043 at 5000 to 7500), "this run wasn't that good". Cause not isolated: whether the tied matrix's summed gradient dominates the aggregate norm, or the four-fold reuse hurts the frame, is untested.

Daniel: build the full-stream SwiGLU first ("most powerful; it needs swiglu very much"); if it is best, explore the cheaper forms (render-only patch, cached context) to match it. `128_d768x12_solvermlp_ep1` (worktree `../tiny-dit-solvermlp`, `runs/queue_solvermlp_2026-11-03.sh`): one change from `_slice_ep1`, a tied `Solver` of SwiGLU (768 -> 1536 -> 768, zero-init `w_down`) as residual on the rms-normed stream, 4 iterations, on Usui, identity at init (checked). The adaptive stop comes later.

`128_d768x12_solvermlp_ep1` ended 1.4781 / 1.4785: +0.0036 against `_slice_ep1` (1.4745), -0.0026 against the baseline, 8 % slower (771 against 838 img/s). Per checkpoint it trails the slice arm by 0.003 to 0.005 throughout. The slice stays the best arm; the solver arms did not beat it. Daniel asks whether anything else is worth trying, or whether the original design with an identity init would match the slice.

Daniel: the extras (deep supervision of the slice, an untied extra SwiGLU) are not necessary; "we do the main check now". `128_d768x12_idfinal_ep1` (worktree `../tiny-dit-idfinal`, `runs/queue_idfinal_2026-11-03.sh`): one change from `_slice_ep1`, a learnable `final` Linear(768, 128) on the fallback optimizer initialised as the selection matrix [I | 0] (output equals the input latent at init, checked). It separates the identity init from removing the learnable readout. A slice seed repeat (noise estimate) was offered and not queued.
`128_d768x12_idfinal_ep1` was stopped by Daniel at step ~2650 (killed, no final score). Eval caption loss at 1000 / 2000 / 2500: 1.5654 / 1.5242 / 1.5146, against baseline 1.5674 / 1.5236 / 1.5134 and slice 1.5679 / 1.5210 / 1.5087: best at step 1000, level with the baseline from 1500, 0.006 behind the slice at 2500. Fallback grad norm about 0.10 against 0.04 for the slice.

Daniel (2026-11-03, going to bed, "tomorrow I wanna see the best model we can produce"): the slice readout becomes the base model (commit `56b4515`, branch `slice-base`), with the conv dropped ("useless"), RoPE on every head dim, and a deeper MLP (SwiGLU hidden 4 x width; his earlier "increase mlp to 8" at G1 was not read as a ratio, so 4 is Claude's pick and a single constant, `MLP_RATIO`). The run is `128_d1024x16_5ep` (`configs/128_d1024x16_5ep.toml`): d1024, 16 layers, 293.8M parameters, usui lr 1e-2 constant for 5 epochs with no anneal ("we will run 5 epochs with no anneal and decide how to do from there"), fallback 2.5e-3, rank fraction 0.1 ("we can do 0.05 again because we have no bottleneck, but I don't think it is worth it, so keep 0.1"), batch 256, 37,605 steps, 5 % warmup of the first epoch, checkpoint 2 x per epoch (every 3750 steps), eval every 1000. The first 250 steps ran at 330 img/s, so about 8 hours plus evals. Not decided and left open: the final anneal; F1 (RoPE base 10000 kept; with every dim rotated, the slowest frequencies move under 0.001 rad across a 16-position axis and act as position-free dims); a seed repeat of the slice arm.

---

## Group H. Harness throughput

### H1. How much of the fixed per-step cost can go?

About 140 ms of each step is fixed, independent of the batch, and the step is launch-bound: the Usui step is about
14,700 of 19,500 launches and syncs 9 times, once per `eigh` (FACTS, Sweep throughput and host syncs). The blocks
are compiled once every Molegrad frame has fit; that cut forward/backward from ~5,400 to ~3,400 launches, a modest
share of the step. Daniel: "Simple blockwise compile worked well in the past. Cuda graph is also open."

Daniel, 2026-11-02, on the Usui-step options: batching same-shape matrices and one `eigh` for all rank groups "we
had that before but on anima we get oom from the intermediates" (usuitrack is built for full finetunes on small
VRAM); diagnostics "or we disable it now. we already tuned it. but yes just every 500 would be a cheap meter but so
noisy it might not be worth looking at"; a sync-free solver for the aim "in the past only eigh was able to get a
stable aim. but with diagonal aim, mabye simpler solutions work too. thats a longer investigation, still worth a
shot". Decided, 2026-11-02 (Daniel): "fitting is default and pretraining random. end": an option gives a pretraining
run a random frame at init with the standard `eta`, no schedule, no refit ("calling fit is dead"; a geometric `eta`
decay helped very short runs before "but wasnt worth it keeping in the code. bc you can also just use more rank");
"right, do stack the eigh calls". No random-frame comparison run: "by step 1k it is caught up fully with zero
difference". Both are usuitrack changes, in the `zero-init` worktree. Also open: the pinned, non-blocking copy in `Stream` (4 syncs per step). Block compile measured
on a free GPU: FACTS, Sweep throughput.

---

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
