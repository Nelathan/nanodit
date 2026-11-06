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

### B5. Which downsampler for any re-encode and for 256 px?

The 128 px latents were encoded from a torch bicubic antialiased resize of the 512 crops. Lanczos keeps more of the top two frequency bands (FACTS, Downsampler). Daniel (2026-11-04): keep the current latents; Lanczos is the better resizer if the data are ever re-encoded or the 256 px stage is made, and a Lanczos arm comes later. Open:
- Whether Lanczos latents train or sample better than bicubic ones at equal steps, at 128 px (a re-encode of a shard subset and one paired arm would say).
- Whether the 256 px stage should resize from the 512 crops with Lanczos in the same precompute that makes its latents.

---

### B6. The data sets and the encode

Daniel (2026-11-05): "the one you forgot in the original plan was Spawning/PD12M then theres also jasperai/monet both
are rather world knowledge images, not so much t2i. lunara is too small for pretraining but an option for finetune
later. if we pull any set, we should create 128 and 256 and gemma class and its encoding sequence. luna has 800GB free
but its a hdd. we can use it for storage before putting the shuffled set on mars. you can do that for current data too.
thats a good job for overnight." Candidates in REFERENCES, Data: Fine-T2I, PD12M, MONET; `moonworks/lunara-aesthetic`
for a fine-tune (D5).

Per set: 128 and 256 px latents, the EmbeddingGemma 2 pooled embedding (the 768-d vector trained on now) and its token
sequence, encoded and staged on `/mnt/luna` (HDD, ~800 GB free), the shuffled set then copied to `/mnt/mars` (107 GB
free on 2026-11-05). First: text-to-image-2M again, overnight. Sizes per 2M rows, from the current set: 128 px latents
32 GB, 256 px latents 128 GB; the token sequence unpadded without the prompt prefix at 1,536 B per token (768-d bf16):
the first encoded rows average 58 tokens, 90 KB, so about 170 GB per 1.9M rows (the 30-token mean estimated from 3,000
captions was low); at the 512-d states (1,024 B per token) about 115 GB, the figure for the shuffled set.

Daniel (2026-11-05): "512 for cross attention, gets down projected to 64 anyway so keep it straightforward. drop the
prefix tokens. unpadded"; "yes lets build a fused dataset of both sources, dedup later too. both in 128 and 256 with
lanczos"; "you can delete anything stale or no longer needed"; "if you need space you can use vel-ssd too:
/run/media/djg/vel-ssd"; "hdd read could be enough, just plan the async delay". EmbeddingGemma 2's residual stream is
512-d (24 layers); its `embedding_projection` (768x512, after the final norm, per token) lifts the token states to 768
before the mean pooling and normalize. The encode stores the projected 768-d states, which span only 512 dimensions;
the pooled embedding is 768-d. The prefix `title: none | text: ` is `<bos>` and 6 tokens;
its trailing space merges into the caption's first token. Daniel (2026-11-05): "of it encodes to 768 thats fine." Running
since 2026-11-05 23:00: text-to-image-2M shards 0–46, then the Lunara eval set (`scripts/precompute.py`,
`scripts/precompute_lunara.py`), into `/mnt/luna/tiny-dit/precomputed/fused/`, log
`/mnt/luna/tiny-dit/precompute_2026-11-05.log`.

Daniel (2026-11-06), after the 512-d residual was re-verified: "but redoing just the sequence is cheaper. do it on
finish. so 2". `scripts/reencode_sequence.py` re-encodes the captions of every fused shard (no images) to the 512-d
token states before the projection, into `fused/sequence512_<shard>.parquet` (key, sequence); it starts when the encode
exits, log `/mnt/luna/tiny-dit/reencode_sequence_2026-11-06.log`. The shuffle then joins on `key` and drops the 768-d
`sequence`. Open:
- Fine-T2I's images are above 1K and partly not square: center crop or aspect buckets.
- Fine-T2I's 2.3 TB at the 39 MB/s of the last download is about 16 h of download alone: several nights.

## Group C. LR sweep and step budget

### C1. Batch size and step budget at 256 px

At 128 px the batch is 128 (archive 2026-11-02, batch 128), 64 × 128 ≈ 8k tokens per step, and the step budget
follows from the data, each image seen once (archive 2026-11-01, final frame and smoke settings). Open for 256 px: the batch, and whether tokens are the right noise
unit for diffusion (t sampling adds its own gradient noise). A smaller batch may want a lower lr, and a lower rank
fraction if `tangent_live_fraction` falls; it did not fall at batch 64 (FACTS, 1000-step ablations). 256 px
throughput is unmeasured. Settle: when the 256 px stage is planned. Daniel, at 128 px, after the sweep: "honestly we might go back
to bs256 again"; about 140 ms of each step is fixed, so batch 256 runs at about 850 img/s against 580 (FACTS, Sweep throughput).

When to add 256 px is open again (Daniel, 2026-11-05). On 2026-11-02 it was "too early … Theres still no hint of a
cat"; `128_d1024x16_swpin_5ep` now follows prompts at class level (FACTS). It needs the images downloaded again and a
precompute (B5: Lanczos from the 512 crops), and it interacts with G2: a patch over 4× the latent area keeps 256 px at
the current 64 tokens. Daniel (2026-11-05): "we might reach a limit where same data stops generalizing. 128px doesn't
seem saturated yet." The second image dataset (G1), when it comes: Fine-T2I is the candidate Daniel found (REFERENCES,
Data); it streams, so it would be encoded shard by shard as text-to-image-2M was.

### C2. Which LR from scratch, and how long a warmup?

The lr is now a constant relative step on pinned weights (SPEC §6, archive 2026-11-05); what follows was measured
under Usui's fixed step, where a moment that fills its rank writes `‖ΔW‖ = lr·√(max(1, out/in)·r)`. The smoke config takes 2e-3 with rank fraction 0.1
and beta 0.99, Daniel's "for now" after 2e-2 read too high (archive 2026-11-01, final frame and smoke settings; eval grid, beta 0.99). The
negative `grad_moment_cosine` followed beta more than lr (FACTS, Runs at usui lr 2e-3). Settle: the sweep at 128 px, usui lr 2e-4, 1e-3, 2e-3 and 4e-3, one pass each over the deduped shards 0–46
at batch 128, warmup 5% of the run, then without decay (archive 2026-11-02, LR sweep); the runs now decay linearly (SPEC §6). The 300-step reads at warmup 20, 50 and 100 are in FACTS (Fixed-step runs). The final linear, the conv, the adaLN tables and the null embedding
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

The continuation of `swpin_5ep` runs at 1e-4 flat (archive 2026-11-05, continuation lr) and learns about 6× slower per
step than the end of `swpin_5ep` (FACTS, Continuation lr arms). Daniel (2026-11-05): "We still progress just slower so
lets let it run. Just that maybe 4e-4 is probably very good next time even if it doesnt look good first." A
2000-step arm after an lr change mostly shows the noise floor moving, so it favours the lowest lr. The read that would
rank lrs for continued learning, none agreed: flat arms long enough to pass the transient (about 5k steps), ranked
by the score slope after it; optionally a short anneal at the end of the best one, to check the ranking survives it.
Read: `128_d1024x16_swpin_slope_lr{4e-4,1e-3}`, 5000 steps each from the end of `swpin_cont_3ep`, against
`swpin_cont_3ep`'s own last stretch at 1e-4 (FACTS, Continuation lr arms): neither higher lr falls within 5000 steps.

Daniel (2026-11-05) on beta and batch: "higher beta makes it [`grad_moment_cosine`] less negative but diminishing returns
we cant turn it positive. only lower lr can but that stops proper learning. probably beta is just fine. actually
increasing batchsize is more common. but we want more progress, well i think more bs doenst give us lower loss per
walltime."

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

The other direction, from Daniel (2026-11-05): a patch embed over 4× the latent area (`pixel_unshuffle(16)`, 512-d
tokens), so the latent fills half of the 1024-wide stream and the slice readout is 512 channels; 16 tokens at 128 px, 64
at 256 px (C1). With it, whether the patch embed writes only the meta channels (the stream beyond the latent slice, the
slice carrying the latent unchanged) or all channels as now (SPEC §3: identity on the slice, learned everywhere); Daniel
prefers meta only. A model change, so a fresh run, not a continuation (archive 2026-11-05, continuation lr).

Daniel (2026-11-05): "g2 yes to test if cost is not to bad. how do other t2i have it? i mean patchembed. it should be
obvious success that patchembed just fills in the missing columns as meta channels." And, later: "for patch size, i just
wanted to mind that the original idea was 128/512 and we now have 1024, so to be bold try 512/1024. current works well it
might just be overprovisioned, we already picked the 2 channel vae to have more real estate." Elsewhere: DiT, PixArt, SD3
and Flux take 16 px per token (f8 VAE, patch 2), SANA 32 px (f32 DC-AE, patch 1); the 512/1024 arm takes 32 px.
Measured: `128_d1024x16_p16meta_3k` (branch `g2-patch16`, meta-only patch embed), FACTS (Architecture arms).

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

### G7. Does the narrower timestep distribution hold at hero length and in the samples?

The training t is logit-normal with σ 0.7 (SPEC §4, archive 2026-11-04). The 3000-step arms at d768x12 gave it a small, consistent edge at t 0.26 to 0.66 and a loss at t 0.16 and 0.84 (FACTS, Timestep distribution arms). Open:
- Whether the edge survives a long run, and whether the extremes (t 0.16 and 0.84) fall behind in R² over it: the score grid carries both, every 1000 steps.
- Whether the narrower distribution costs sample quality at the ends (high t sets the composition, low t the sharpness); R² does not say, the eval grids do.
- `128_d1024x16_swpin_5ep` trained at σ 1 (its config predates the default), so the continuation at usui lr 1e-4 (archive 2026-11-05) is the first long run at σ 0.7.

### G8. What the pin leaves open

Every Usui matrix is pinned to its init Frobenius norm and takes a constant relative step (SPEC §6, archive 2026-11-05).
Open:
- Whether a learned scalar gain per pinned matrix (direction pinned, gain on the fallback) helps: some scale is free in
  the gates (`out` and the v rows of `qkv`, the `w_up`·`w_down` product), some is not (`w_gate` sets the SiLU operating
  point; `ada`, `time_mlp` and `text_proj` the conditioning scale; `patch_embed` the input scale).
- A pin per output row instead of per matrix.
- Which of the sandwich norm and the pin removed the spikes; Daniel reads it as the sandwich norm.
- Reading, from Daniel: EDM2's magnitude-preserving layers (REFERENCES, Objective).

### G9. Does Usui hold back the rank the weights grow into, and can it be helped?

The spectra (FACTS, Spectra of the pinned weights): the weights start full rank and concentrate; at the end of
`swpin_5ep` the MLP matrices put 90 % of their energy in about a sixth of their directions, against about 0.7 in the
pretrained LFM3 and EmbeddingGemma 2 MLPs; Usui's frame holds about the whole per-step gradient.

Daniel (2026-11-05): "gradient is low rank so even adamw has trouble using the full space. But i want to know if usui is
harming full rank evolution and if we can help it." "Pin is just a form of weight decay and everyone uses it to prevent
explosive weights." "Backwards can only put changes where activations sit and they tend to cluster." "Zero-order would
help here, but is inefficient." "Higher rank usui always helps but diminishing. Gradient wise its already optimal
subspace. But even tail contributes, its a power law."

Daniel (2026-11-05, later): "Dont trust the capture, its just probes. Its comparable but not true fixpoint." "It
could well be age. At least its more than 10%." "Dropping tails did harm the initial final layer bottleneck, but on most
matrices it doesnt make any difference. Im leaning onto the original paper subtrack. No idea how they argued subspace is
good enough. My variant is arguably better, i have proven tracking is better and the polar + row norm is principled
higher rank." "I know gradient is low rank, no need to probe." "Rank 0.5 will help spectrum and a little loss but the
point is having a small space is all you need."

His question: "Applying polar to weights would surely break them. How about hadamard rotation that keeps it intact but
maximizes rank? Even possible?" Not as stated: an orthogonal rotation on either side (W → HW, W → WH) leaves every
singular value unchanged, so the rank and spectrum are invariant; any change to the spectrum changes the function.
What a Hadamard rotation does change is the basis: it spreads energy evenly over coordinates (the incoherence QuaRot and
SpinQuant use), which would matter only to the basis-dependent parts of the optimizer, such as a row norm.

Open: whether the concentration is Usui's (the rank-0.1 frame drops the gradient's tail every step), the training length
(theirs is far longer; the 3-epoch continuation shows whether it keeps falling), or this task. Not wanted: the rank 0.5
arm and the dense-gradient probe. Remaining candidates, none agreed:
- A full-rank optimizer arm (AdamW on every matrix) of `swpin_3k`'s length: the reference for "even AdamW". Needs code.
- The same spectra on the continuation's checkpoints: whether the effective rank still falls with age.
- Ways to help, if Usui is the cause: a small random component in the frame each refresh to reach the tail (Daniel's
  zero-order direction is the extreme of this).

### G10. The depthwise conv, placed differently

The conv before attention made no difference at d768 (G5; FACTS) and was dropped. Daniel
(2026-11-05): that may have been a design mistake rather than a verdict on the conv. Candidates: the conv in place of
attention in some blocks, rather than before it; the conv writing only the meta channels (G2); its output stacked into
the MLP input (SANA's Mix-FFN puts a depthwise conv inside the MLP). Settle: one arm per placement against the current
model, at a scale and length agreed when it runs.

Daniel (2026-11-05): "conv instead of attn, writing to meta only, i think thats the cleanest variant. would be half the
layers of a 768x12 model. if success, even 2/3 or 3/4. always conv first, attn next." On the two forms offered, (a) the
depthwise conv over the meta channels only, (b) over all channels, then `out` into the meta channels: "for conv yes
design b. maybe its ok to write image channels, but it could also write context into the render. meta only would put
the swiglu gate before it. in earlier samples where we compared the canon conv vs without i saw a softer image, like it
just copied context into render." "its ok if it needs some more time". Measured (FACTS, Architecture arms): `128_d1024x16_conv_3k` (branch
`conv-meta`): `swpin_3k` with blocks 0, 2, ..., 14 a depthwise 3×3 conv over all 1024 channels after the adaLN
modulation, then `out` (1024 → 896) into the meta channels, its sandwich norm and the attention gate; the MLP as before.

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

## Group I. Training schemes beyond flow matching

### I1. Priming: reconstruction pretraining, then distributional post-training

Its own topic doc: `topics/priming.md` (the source, Daniel's direction, the mapping onto this project, what is open).

---

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
