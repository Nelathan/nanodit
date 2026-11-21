# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds principles and gotchas, `log/` the measurements; `archive/` holds
everything closed. See `AGENTS.md` for how this file is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs or code. Each "how to settle" is a proposal, not an
agreed arm.

---

## Group A. Usui wiring

### A3. Does Usui need `eigh` at all?

A first frame fitted to a low-rank first gradient is not worth an `eigh`: the trunk draws its frames at random, and
usuitrack's random range finder fits without one. The aim (Molegrad's too) is closed-form, a Rayleigh-quotient
projection. `eigh` remains once per step in the frame turn: the spectra of the [r, r] tangent Grams that `turn` rotates
by (usuitrack `frame.tangent_spectra`), one stacked, launch-bound call that syncs with the host (H1). Settle: on a
usuitrack branch, the turn without `eigh` against the current one, by loss and `grad_capture_probe`; merged only
if it holds.

### A4. The polar step over the writers' two geometries

The writers (`out`, `w_down`, `conv.out`) stack rows that write the 128 image channels on rows that
write the meta channels; the image rows read out as the prediction, the meta rows as workspace. The polar step
orthogonalises the stacked matrix as one, as Muon on stacked heads does across heads that each have their own
geometry: it trains, the alignment is wrong. *Hypothesis*: a polar step per row block (image, meta) fits each role.
Related: G8's bound per output row. Settle: first whether the image rows' and the meta rows' gradients differ in
spectrum on a checkpoint; then a usuitrack branch. State: proposed, no read.

---

## Group B. Data and conditioning

### B3. Do short prompts get ignored?

Training captions are paragraph-length, inference prompts are short. Settle: judge
it on the eval grid with short prompts. If they are ignored, add a second
embedding per image from the first sentence and sample 50/50 (another ~3 GB).
Evidence: log 2026-11-20-measurements-from-facts, "EmbeddingGemma 2" (cosines of short prompts against captions); no generation tested.

### B4. Flip augmentation, if ever needed

Parked: no augmentation now. Primal's decoder is flip- and rot90-equivariant, its encoder
is not (log 2026-11-20-measurements-from-facts, Primal 2x2), so a flipped latent decodes correctly but is not on the
encoder's distribution. Settle, if more data diversity is needed: re-encode flipped
images, or test whether training on flipped latents hurts. Evidence: log 2026-11-20-measurements-from-facts only.

### B5. Which downsampler for any re-encode and for 256 px?

The 128 px latents were encoded from a torch bicubic antialiased resize of the 512 crops. Lanczos keeps more of the top two frequency bands (log 2026-11-20-measurements-from-facts, Downsampler). Decided 2026-11-04: keep the current latents; Lanczos is the better resizer if the data are ever re-encoded or the 256 px stage is made, and a Lanczos arm comes later. Open:
- Whether Lanczos latents train or sample better than bicubic ones at equal steps, at 128 px (a re-encode of a shard subset and one paired arm would say).
- Whether the 256 px stage should resize from the 512 crops with Lanczos in the same precompute that makes its latents.

---

### B6. The data sets and the encode

Its own topic doc: `topics/datasets-and-encode.md` (the sets, the fused encode, the 512-d token states, Fine-T2I and its buckets). State: text-to-image-2M is encoded fused at 128 and 256 px; Fine-T2I's last two subsets are being encoded; the merged set per aspect bucket and its stream are built (archive 2026-11-13-merged-training-set), the 128 px set waits on the encode. The bucket set is open in the topic. The subsets differ in caption quality (the LLaVA-captioned ones are weaker); the handbook's data recipe (equal weight per dataset, long captions only, `topics/scaling.md`) is not tried here.

### B7. Defective captions in `set_128`

About 2,900 of 8.24M rows (log 2026-11-20-measurements-from-facts, Captions of `set_128`; `log/2026-11-17-caption-outliers-and-oom.md`): escaped
UTF-8 bytes (410, jackyhate), captioner loops and repeats (195 matched, more with longer units), chat-template leaks
(3), and captions that switch into Chinese mid-sentence (part of 2,297 with CJK; quoted text to render is the other
part and is wanted). The length tail above about 256 tokens is almost all defects. *Decided* (2026-11-17): every
row the defect patterns match is excluded from streaming (SPEC §1, `excluded_keys.txt`), quoted CJK included; no
rewrite now; a token cut at 512 as the guard. Open: salvage, if wanted later — the escapes decode mechanically and
need only the text re-encoded (`reencode_text.py` works per file); loops, leaks and code-switched captions would be
rewritten following Krea 2's rules (`github.com/krea-ai/krea-2`, `docs/expansion.txt`: keep every subject,
attribute and relation, add nothing unsupported, one paragraph, quote rendered text). Also open: whether the
enhanced prompts and the 256 px set get the same scan before they are built. State: rows excluded, salvage open.

## Group C. LR sweep and step budget

### C1. Batch size and step budget at 256 px

At 128 px every reference since 2026-11-13 runs batch 256 (the configs, the 2026-11-13 and later log notes), 64 × 256 ≈
16k tokens per step. The archive's batch 128 (2026-11-02) no longer describes what runs, and no closure records the
return to 256. The step budget follows from the data, each image seen once (archive 2026-11-01, final frame and smoke settings). Open for 256 px: the batch, and whether tokens are the right noise
unit for diffusion (t sampling adds its own gradient noise). A smaller batch may want a lower lr, and a lower rank
fraction if the frame's live share falls (it was read as `tangent_live_fraction`, which the current pin no longer logs); it did not fall at batch 64 (log 2026-11-20-measurements-from-facts, 1000-step ablations). 256 px
throughput is unmeasured. Settle: when the 256 px stage is planned.

At 128 px the batch is *Decided* (2026-11-18): 256 (archive 2026-11-18-batch-256). Open here: the batch at 256 px.

Batch 512 at 128 px (`references/handover-2026-11-12-residual-gating.md`, E4, recomputed): at d768x24 only about 59 ms of
the 430 ms step is fixed per step, the Usui and fallback steps (log 2026-11-20-measurements-from-facts, Step time at d768x24). Doubling the batch
projects to about 630–640 img/s against 596, +6–7 % (*Hypothesis*, arithmetic on that table, assuming forward/backward
and data scale linearly with the batch and the Usui step is per optimizer step, not per micro-batch; the latter is to
verify). The handover's +25 % rested on
the 140 ms fixed cost of usuitrack `zero-init`. Worth an arm only if loss per image holds at 512.

At the move to 256 px (the thread in `references/gemini-thread-adaln-and-architecture.md`): how usuitrack's moment
sees accumulated micro-batches (one gradient per optimizer step under Molegrad, or per micro-batch), and whether the
continuation wants a short warmup so the moments re-form on the new statistics.

When to add 256 px is open again (2026-11-05). On 2026-11-02 it was too early, with still no hint of a
cat; `128_d1024x16_swpin_5ep` now follows prompts at class level (log 2026-11-20-measurements-from-facts). It needs the images downloaded again and a
precompute (B5: Lanczos from the 512 crops), and it interacts with G2: a patch over 4× the latent area keeps 256 px at
the current 64 tokens. The same data may reach a limit where it stops generalizing; 128 px does not look saturated yet
(2026-11-05). The second image dataset (B6), when it comes: Fine-T2I is the candidate found (REFERENCES,
Data); it streams, so it would be encoded shard by shard as text-to-image-2M was.

### C2. Warmup and the fallback lr at the chosen Usui lr

*Decided* (2026-11-16): usui lr 2e-3. At 6000 constant-lr steps at d768x12 on the merged set it ends 0.0113 ahead of
4e-3 (`log/2026-11-14-reference-6k-lr2e-3.md`); the curve does not say 2e-3 is the optimum, and it is carried to
depth 24 untested there (G4). Open: the warmup length, and the fallback lr (2.5e-3 in the reference) at 2e-3. Settle:
config-only arms against the reference. 1e-3 over a first pass at depth 24, annealed, ends 0.017 behind held-out
and 0.011 on Lunara (`log/2026-11-21-lr1e3-one-pass.md`); *Decided* (2026-11-21): the first pass stays at 2e-3. *Hypothesis*: 1e-3 pays
from the second pass on; not run.

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

### C4. Does an edge-of-stability prediction for momentum optimizers bear on the learning rate chosen?

Its own topic doc: `topics/edge-of-stability.md` (the bubbleforce prediction, the toy, the lr-jump read and what is open). State: the pinned lr is not a stability ceiling, the curvature adapts to it; that favours a constant lr with a late anneal, now the schedule (archive 2026-11-12-constant-lr); at one pass a 6000-step anneal takes more off the held-out loss than the pass's second half at constant lr did (log 2026-11-21-anneal6k); whether a lower constant lr ends lower after its own anneal is C2.

---

## Group D. Sampling and eval

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

### D7. A read of prompt following

Loss does not measure prompt following: the caption-versus-null gap stayed near 0.002 at every width (log 2026-11-20-measurements-from-facts, G5).
Every token-conditioning question (J5, L2) and G5 waits on a read that does; the caption-use probe
(`scripts/probe_caption_use.py`, loss with the null prompt against the caption, and the caption's softmax mass per block) is a partial one. Candidates:
- **Role-swap binding prompts**, scored per checkpoint: "a bear building a cabin" against "a cabin shaped like a
  bear", "a red cube on a blue sphere" and its reverse (E5b). The eval grid's bear prompt is the case that motivated
  token conditioning.
- **Caption alignment:** EmbeddingGemma 2's image embedding of the sample against the pooled caption, by cosine
  (`topics/priming.md`, ideas ledger). The model maps images and text into one space.

Settle: which read, its prompt set, and whether it runs at every eval or on checkpoints.

### D8. Loss curves fitted against images seen

Observation, not recorded in the repo: under Usui, models of different size train in parallel, with a gap per size
that stays constant. If that holds, one equal-wall-clock comparison decides a size question with no crossing to wait
for (E5d). Settle: a log-log fit with an offset term to the score curves kept in the run directories, compared only within
one data set and eval set (the d768x16 lr arms against the d768x12 arms, the d1024x16 runs), and a check of
whether the gaps stay constant.

---

## Group F. Model choices SPEC leaves open

Taken as the standard DiT/PixArt choice in `src/nanodit/model.py`; each is
to be confirmed. Settle: by reading, or a short 128 px comparison once training runs.

### F1. RoPE: the share of frequencies that rotate

Base 10000, 16 frequencies per axis, of which only the fastest quarter rotate (SPEC §3): 1, 0.56, 0.32 and 0.18 rad
per token (`model.rope_angles`). The slowest turns 1.2 rad across the 8 tokens of a 128 px grid and 2.7 across 16 at
256 px, so every rotated dim carries position; a lower base would only make them wrap sooner. The question left is the
share: a quarter against full RoPE, read on Lunara at 256 px and on samples. State: no arm; the samples place and
compose without a sign of a position fault.

### F2. adaLN scale init scale

SPEC says "small random". Std 0.02 on the shared 512→4096 projection makes the
blocks' scale rows far from zero at init (log 2026-11-20-measurements-from-facts, init), so
`1 + scale` starts far from 1. A fan-in-scaled or 10× smaller init would start near
identity.

### F4. Block layout: mixers, parallel blocks, RoPE and head sharing

The list of 2026-11-06, in `topics/block-design.md` with the ledger: meta-only patch embed at one primal block per token, parallel
mixers with one meta-only output projection, a sparse layout of conv layers with an occasional cross-and-self layer, a PaLM-style parallel
MLP, RoPE against partial RoPE against NoPE, MHA against GQA for self-attention (MHA is not cheap in runtime and a 1-megapixel image
costs far more). *Decided* (2026-11-20): conv in 3/4 of the blocks, attention third of four (archive
2026-11-20-attention-third-of-four).

### F5. What ports from the modded-nanogpt speedrun?

Its own topic doc: `topics/speedrun-ports.md` (a ledger of the speedrun's changes against this model: in design, a
question, not applicable, to read). State: only the Gemini thread's list and recollection; the record history itself
is not read. Settle: read it and sort every record into the ledger.

### F6. Which MLP activation?

SwiGLU at 4 × width (SPEC §3) against ReLU² at equal parameters, and a pre-activation-capped SwiGLU (MOSS-Image's
SiTU-GLU). In `topics/speedrun-ports.md`. Settle: one arm each, read on loss and img/s.

---

## Group G. Scale and signal propagation

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
Measured: `128_d1024x16_p16meta_3k` (branch `g2-patch16`, meta-only patch embed), log 2026-11-20-measurements-from-facts (Architecture arms).

Reading, 2026-11-06: that arm packed four primal blocks into each token ("4 blocks per latent"), which was the bad part, and it combined the
token size with the meta-only write. The token stays one primal block (8 px, 128 dims; patch 12 does not divide the 64×64 latent). The
untested arm is the meta-only write alone at 8 px (`topics/block-design.md`).

### G3. The common mode in the residual stream

By mid-depth much of each image's stream energy is one vector shared by its tokens, written from the first blocks
on, conv and attention alike (logs 2026-11-20-measurements-from-facts, Token signal, and 2026-11-21-stream-scale); the
token-specific part keeps its rank. Unsure whether it costs anything: each branch's
RMSNorm then sees the token-specific part at about a quarter of unit scale. Candidates that fit the architecture:
register tokens, which give global state its own slots; nothing else proposed yet. Settle: a register arm against the
base, read on loss, the eval grid and the same diagnostic. Background (Li's notes ch. 5–6, the shaped Transformer,
CompleteP): at init the zero gates make every block the identity, so their rank-collapse results do not apply there;
SwiGLU measured decorrelating, so shaped activations have no target here.

### G4. Depth scaling, if depth arms run

CompleteP scales each block's branch by 1/L for lr transfer across depth, derived for AdamW. Usui's fixed-norm write
and the per-sample gates may absorb it. The depth-24 reference runs the lr chosen at depth 12, untested at 24. Settle: an lr
arm at depth 24 against it.

### G5. Does 128 px saturate before 256?

Open: whether 64 tokens at 128 px limit the model, and when 128 px stops paying before the move to 256 px. Loss does
not measure prompt following (the caption-versus-null gap stayed near 0.002 at every width, log 2026-11-20-measurements-from-facts); samples do.
Observation, not recorded in a log: objects, not only the mood, appear in samples after about 10M images (on an earlier
model).

### G8. What the norm ceiling leaves open

Every Usui matrix takes the trust step inside the ball of its init norm (SPEC §6, archive 2026-11-12); this differs
from the former pin only where a step shrinks a norm.
The sandwich norm was necessary against the loss spikes (2026-11-06); no arm checks it again.
Open:
- Whether a learned scalar gain per bounded matrix (direction bounded, gain on the fallback) helps: some scale is free in
  the gates (`out` and the v rows of `qkv`, the `w_up`·`w_down` product), some is not (`w_gate` sets the SiLU operating
  point; `ada` and `time_mlp` the conditioning scale; `patch_embed` the input scale).
- A bound per output row instead of per matrix. The rows of `out` and `w_down` that write the 128 image
  channels have another role than those writing meta, and share one bound with them now (the thread in
  `references/gemini-thread-adaln-and-architecture.md`).
- Bounds per role, and a zero-initialized matrix: Usui's bounds are multiples of the init norm, so a matrix that
  starts at zero cannot take them; none does here (§5).
- Reading: EDM2's magnitude-preserving layers (REFERENCES, Objective).
- The bound binds by the end of one pass of `a1-ref-1ep` (its `score.jsonl`, `weights.norm_ratio`); what it does
  over further epochs is not seen.

### G9. Does Usui hold back the rank the weights grow into, and can it be helped?

Its own topic doc: `topics/weight-spectra.md` (the spectra of the pinned five-epoch run against pretrained models, the reading, the candidates). State: open, no arm agreed; the rank 0.5 arm and the dense-gradient probe are not wanted. New candidates: activation rank against weight rank per layer, and stochastic rounding switched off for a stretch; the stable rank per role is now logged at every eval. Over one pass of `a1-ref-1ep` every block role's stable rank falls and none turns upward (log 2026-11-18-a1-ref-1ep); read again over the long run's further epochs.

### G10. The depthwise conv, placed differently

The conv replaces attention in three of every four blocks (SPEC §3).
*Decided* (2026-11-20, archive 2026-11-20-conv-block-lfm-norm-read): the conv block is LFM2's gated short conv with
the norm on its read, writing all channels (SPEC §3). Open: the placement and reach below.
*Measured* (log 2026-11-20-lfm-norm-read-1ep): at the end of one pass it loses R² on the band over three tokens at
t 0.66 to 0.74 against the basic conv; not yet read on samples.

Candidates (`references/handover-2026-11-12-residual-gating.md` E3, `references/gemini-thread-adaln-and-architecture.md`):
- **A 5×5 or 7×7 kernel.** Another way to reach (a 16 × 16 grid in 8 convs instead of 15) without depth; it mixes
  differently from two stacked 3×3s (one filter, no nonlinearity or gate between). Settle: 5×5 and 7×7 arms trained
  at 128 px against a4's first 6000 steps, read on the held-out loss, on Lunara at 256 px (the larger grid, where
  reach matters more) and on step time at 128 and 256 px (`scripts/bench_step.py` on the Lunara rows). *Measured*
  (log 2026-11-21-conv-kernel-and-tanh-gates): 5×5 ahead on both sets within the seed spread, at about 1 % of
  the step; 7×7 level, behind at high t on Lunara, slower. State: 5×5 waits for a decision.
- **The kernel by depth.** Larger early, 3×3 late, and where attention sits: by thirds 7×7 / 5×5 / 3×3 and 5×5 /
  3×3 / 3×3; phased, three (attention, 7×7), twelve 3×3, three (attention, 5×5); the period attention, 7×7, 5×5,
  3×3. *Measured* (log 2026-11-21-conv-taps-a4-ref): after a pass the first 3×3 kernels stay near or outward of the
  init share on the ring, the middle ones move to the centre, the last sit between. Settle: each at 6000 steps as
  the kernel arms, read on loss and on `scripts/probe_conv_taps.py`. State: `arm/conv-kernel` (the kernel a field
  of the block type, four configs) queued.
- **Padding.** The conv zero-pads at the grid's border. Replicate or reflect padding avoids a constant 0 around the
  edge. Read at 256 px and on the unseen grid, where border effects would show.

### G11. The stream's scale

Its own topic doc: `topics/stream-scale.md` (the two workspaces, dilution and drowning, the split read, a held meta
stream, attention over depth on the meta channels, the tables). State: no arm run; three arms proposed, in order.

---

## Group H. Harness throughput

### H1. How much of the step's time can go?

At d768x24 on usuitrack `92b23bc`, a 429.5 ms step breaks down as follows (log 2026-11-20-measurements-from-facts, Step time at d768x24):
- forward and backward: 363.6 ms;
- the Usui step: 57.6 ms for 131 matrices;
- the GPU is busy about 397 ms, so launch overhead can win at most about 33 ms.

Open:
- **The Usui step's time.** The usuitrack session reports 140 → 30 ms; the probe here measures 57.6 ms
  synchronised, and under the profiler 101 ms of host time against 67 ms of GPU time
  (`log/2026-11-16-step-profile-d24.md`). Settle: the same probe on the same matrices on both sides. Until then the
  30 ms is not a fact here.
- **Graphing at micro-batch 64.** It waits on the item above. Block-wise CUDA graphs gave nothing at micro-batch 64
  and ran out of memory at 128 (HANDOVER).
- **Stacking same-shape matrices** for `eigh` and the polar step in chunks sized to the memory free after backward,
  when the activations are gone (`references/handover-2026-11-12-residual-gating.md`, E7). At depth 24 every block
  type repeats six times. A usuitrack change.
- **Forward and backward are now 85 % of the step.** Further speed comes from the model's kernels more than from the
  optimizer: compile, attention, the depthwise conv, FP8 (`topics/speedrun-ports.md`).

The 2026-11-02 profile below is of d768x12 on usuitrack `zero-init`: about 140 ms of each step was fixed and
launch-bound, and the Usui step made about 14,700 of 19,500 launches and synced 9 times, once per `eigh` (log 2026-11-20-measurements-from-facts, Sweep
throughput and host syncs). The blocks are compiled once every Molegrad frame has fit; that cut forward/backward from
~5,400 to ~3,400 launches, a modest share of the step. Simple blockwise compile worked well in the past.

On the Usui-step options (2026-11-02): batching same-shape matrices and one `eigh` for all rank groups was tried before,
but on anima the intermediates run out of memory (usuitrack is built for full finetunes on small VRAM); the diagnostics
could be disabled now, since they are already tuned, though one every 500 steps would be a cheap meter, so noisy it
might not be worth looking at; a sync-free solver for the aim is a longer investigation but still worth a shot, since
in the past only `eigh` gave a stable aim, while with a diagonal aim simpler solutions may work too. Decided
2026-11-02: fitting is the default and pretraining uses random frames; an option gives a pretraining run a random frame
at init with the standard `eta`, no schedule, no refit (calling fit is dead; a geometric `eta` decay helped very short
runs before, but was not worth keeping in the code, since more rank does the same); and the `eigh` calls are stacked.
No random-frame comparison run: by step 1k the fitted frame has fully caught up, with zero difference. Both are usuitrack changes, on usuitrack `lab`. Also open: the pinned, non-blocking copy in `Stream` (4 syncs per step). Block compile measured
on a free GPU: log 2026-11-20-measurements-from-facts, Sweep throughput.

---

### H2. The caption's padding

`pad_sequences` pads a micro-batch to its longest caption. The measurements below are of the removed SDPA cross
attention, where the masked call ran the cutlass memory-efficient kernel with the mask expanded to a full bias in the
backward; a4's joint flex_attention masks the padding in its score_mod, and its cost on an outlier is not measured. One 2000-token caption costs about 8× the time and 6×
the memory of a cross-attention call, and with micro-batch 128 it ran the 1ep run out of memory (log
2026-11-17-caption-outliers-and-oom). Packed captions through `torch.nn.attention.varlen.varlen_attn` (flash, GQA
native, no mask) remove that cost and gain nothing on clean captions. *Decided* (2026-11-17): micro-batch 64, the same
speed at half the peak; captions cut at 512 token states (SPEC §1), which bounds an outlier at about +10 % of a step
(log, L 512 against 136). Open: whether packing is wanted with the tail excluded; it is the structural guard
against any caption length, within 0.35 ms per call either way on captions without an outlier;
to remove the outlier's whole cost the token states are packed before `cross_kv` too, not only in the attention call. A retry of a step that runs out of
memory (the same micro-batches, after freeing the cache) is *Decided* acceptable (2026-11-17) and not built; it would only
help against another job's memory on the shared GPU. Batching by caption length, the handbook's answer, is *Not
wanted* (2026-11-18): packing is the direction here. State: open.

---

## Group I. Training schemes beyond flow matching

### I1. Priming: reconstruction pretraining, then distributional post-training

Its own topic doc: `topics/priming.md` (the source, the direction, the arms and their reads, what is open). State (2026-11-12): a side quest; no priming code on the trunk. Next arm: clean context tokens (t = 0, no loss) at a 1 to 25 % share. REPA only on precomputed targets of one size.

---

## Group J. The conditioning path

Its own topic doc: `topics/conditioning.md` (the path as built, the questions, other models' conditioning). The adaLN
takes t alone, without a shift; the caption is read by joint attention with a sink, after a two-block refiner
(archives 2026-11-20-adaln-takes-t-only, -joint-attention, -attention-sink, -caption-refiner, -adaln-without-shift).

### J1. t at the input only, no adaLN

t supplied once, into the meta channels at the patch embed, possibly re-injected every k blocks; static zero-init gates
in place of the adaLN gates. *Hypothesis*: under x0 velocity the model takes the noisy latent and
ships it, so only the first blocks need t, to know how carefully to touch it; the noise is overwritten fast, and deeper
blocks clean, add coherence and the prompt. Free read first: how much the trained adaLN's six vectors vary with t
over depth, from a checkpoint. State: parked. Krea 2 reports that dropping t conditioning underperformed its adaLN
baseline and that t as tokens fell behind at 512 and 1024 px (`topics/conditioning.md`).

### J5. Several encoder layers instead of the last

Krea 2 is said to fuse several layers of its text encoder; the token states here are EmbeddingGemma 2's last hidden
state. Needs a re-encode. Settle: read Krea 2's report and code first. State: hypothesis.

### J7. tanh on the adaLN gates

Lumina 2 and Z-Image bound each gate with a tanh (diffusers code; `topics/conditioning.md`); here the gates are
linear. Lumina-Next, where it comes from, adds it "to prevent extremely large modulation values from contributing to
the residual branch" (arXiv 2406.18583 §2.1), with no ablation. Here Usui holds the shared projection inside its
init-norm ball; only the per-block tables (fallback) are unbounded. *Hypothesis*: the bound matters only where a
gate grows, which loss may not show. Settle: one arm on a4, read on loss and on the gates' size over training.
*Measured* (log 2026-11-21-gate-sizes): after one pass the gates grow with depth, through the per-block tables; in
the last blocks most exceed 1, so a tanh bends them. *Measured* (log 2026-11-21-conv-kernel-and-tanh-gates): level
at 6000 steps, where few gates exceed 1 yet. State: proposed as superseded by G11.

---

## Group K. What the network predicts

Its own topic doc: `topics/prediction-target.md` (the skip derived for this target, AnisoFlow, what of the thread does
not apply).

### K1. A t-dependent skip

The identity on x_t is the optimal linear skip only at t = 0.5; `c(t) = 2(1 − t)²/((1 − t)² + t²)` halves the residual
the trunk must write at both ends. Preferred form: EDM's `c_skip` at the readout, so the slice still carries x_t.
Settle: one arm, read on the loss by t. State: derived, not run.

### K2. An anisotropic target (AnisoFlow)

Per-direction mixes of x0 and velocity targets, claimed to help in pixel space and not on SD-VAE latents; Primal 2x2
tokens sit nearer pixel space. Settle: the spectrum of the 128-d token covariance first, then the paper. State: not
read.

---

## Group L. Attention dynamics

Its own topic doc: `topics/attention-lock-in.md` (the source's claim, the model checked against what the source
assumed, the sensor, the backward temperature and its tests, the parked ideas).

### L1. Does attention lock in early here?

A sensor of attention maps on a fixed probe batch at log-spaced steps, over image and caption keys. A partial read is possible from
the 1ep run's checkpoints at 4000–16000. State: proposed; this comes before L2.

### L2. Backward temperature on the attention scores

Forward unchanged, the score gradient taken through `softmax(s / tau_b)`, values through the true weights; `tau_b = 2`
proposed. Three equivalence tests before any run. State: proposed, waits on L1.

### L3. Parked follow-ups

Interleaved head attention as an early phase, competition across heads, a forward temperature schedule. State: parked
behind L2.

---

## Group M. The residual stream as lanes

Its own topic doc: `topics/residual-lanes.md` (the write map of the stream, the two-stream reading, Hyper-Connections
and SimpleHC, other scratchpads). Related: G3, G10.

### M1. Per-token write gates onto the slice

Learned per token, initialized at the current masks. Today only the per-image adaLN gate governs the MLP's and
attention's writes to the image channels. State: parked in its source.

### M2. Long skips

Block i into block L − 1 − i, gated near zero, into the meta channels. State: proposed.

### M3. The input re-injected

The patch embed's meta projection, or x_t, added back to the values or the stream at depth (the speedrun's value
embeddings, recalled). Shares its mechanism with J1's t re-injection. State: proposed.

### M4. Mixers write the context, a separate step prints it

Attention writes the meta channels only, so the meta lanes carry the context and only the
per-token MLP writes the slice; the conv block writes all channels (archive 2026-11-20-conv-block-lfm-norm-read). A dedicated print step at the end is *Not wanted* (2026-11-17): skips (K1) carry
x_t, not idle channels. State: proposed, no arm.

### M5. Separate lanes, the image lane apart

The stream split into lanes in the manner of DeepSeek's hyper-connections: one lane of 128 for the image, three of
768 or 512 for the workspace. No sublayer reads or writes the image lane directly; it is written only through the
learned lane mixing. Open: whether that keeps the activations bounded. HC with a learned, unconstrained mixing
matrix compounds its gain over depth; mHC bounds it by Sinkhorn and SimpleHC finds identity mixing as good
(`topics/residual-lanes.md`, Background). Lanes of unequal width need a projection in the mixing, which no
source here has. State: proposed, no arm.
