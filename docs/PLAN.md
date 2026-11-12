# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds measured numbers; `archive/` holds
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
usuitrack branch, the turn without `eigh` against the current one, by loss and `tangent_live_fraction`; merged only
if it holds.

---

## Group B. Data and conditioning

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
(2026-11-05). The second image dataset (B6), when it comes: Fine-T2I is the candidate found (REFERENCES,
Data); it streams, so it would be encoded shard by shard as text-to-image-2M was.

### C2. Warmup and the fallback lr at the chosen Usui lr

*Decided* (2026-11-12): usui lr 4e-3, the best of the sweep at d768x16 over 3000 steps under the trust rule with each
norm pinned at its init, before the ceiling (archive 2026-11-12-norm-ceiling;
`log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`: 5e-4 1.5332, 1e-3 1.5101, 2e-3 1.4933, 4e-3 1.4868, 8e-3 1.4950).
The reference config still runs 2e-3. Open: the warmup length, and the fallback lr (2.5e-3 in the reference) at
4e-3. Settle: config-only arms against a 4e-3 reference.

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

### C4. Does an edge-of-stability prediction for momentum optimizers bear on the learning rate chosen?

Its own topic doc: `topics/edge-of-stability.md` (the bubbleforce prediction, the toy, the lr-jump read and what is open). State: the pinned lr is not a stability ceiling, the curvature adapts to it; that favours a constant lr with a late anneal, now the schedule (archive 2026-11-12-constant-lr); whether the anneal recovers the loss is untested at length.

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

### F3. Does the pooled caption keep the adaLN alongside cross attention?

Cross attention in the attention blocks is *Decided* as design (2026-11-12); the conv blocks have none, so the caption
reaches them only through the adaLN, which is cheap in compute. Open: whether the pooled caption stays in the adaLN
(`pooled_ada`, SPEC §3), or the adaLN takes t alone; and if it stays, whether one shared `ada` is enough or each block
wants its own (or a per-block LoRA on the shared one). Settle: by reading how other text-to-image DiTs with cross
attention condition their adaLN, not by an arm. At 6000 steps, cross with or without the pooled adaLN ended within
0.002 of each other (`log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`).

### F4. Block layout: mixers, parallel blocks, RoPE and head sharing

The list of 2026-11-06, in `topics/block-design.md` with the ledger: meta-only patch embed at one primal block per token, parallel
mixers with one meta-only output projection, a sparse layout of conv layers with an occasional cross-and-self layer, a PaLM-style parallel
MLP, RoPE against partial RoPE against NoPE, MHA against GQA for self-attention (MHA is not cheap in runtime and a 1-megapixel image
costs far more), GQA and NoPE for the cross attention. Base: conv in 3/4 of the blocks.

Open: where the attention block sits in the period. The trunk puts it fourth of four, as Qwen does; an earlier ablation
on another model found the third position best of the 3/4 layouts. Settle: one config-only arm per position (`layout`,
SPEC §3).

---

## Group G. Scale and signal propagation

### G1. How deep the long run is

*Decided* (2026-11-12): width 768; the model grows in depth. Width 768 × depth 12 is the baseline for trials. Open: the
depth of the long run, and the MLP ratio at that length. Settle: a 30-step microbench per depth for img/s and peak
memory, then the step budget the run can afford.

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

### G5. Does 128 px saturate before 256?

Open: whether 64 tokens at 128 px limit the model, and when 128 px stops paying before the move to 256 px. Loss does
not measure prompt following (the caption-versus-null gap stayed near 0.002 at every width, FACTS); samples do.

### G7. Does the narrower timestep distribution hold at hero length and in the samples?

The training t is logit-normal with σ 0.7 (SPEC §4, archive 2026-11-04). The 3000-step arms at d768x12 gave it a small, consistent edge at t 0.26 to 0.66 and a loss at t 0.16 and 0.84 (FACTS, Timestep distribution arms). Open:
- Whether the edge survives a long run, and whether the extremes (t 0.16 and 0.84) fall behind in R² over it: the score grid carries both, every 1000 steps.
- Whether the narrower distribution costs sample quality at the ends (high t sets the composition, low t the sharpness); R² does not say, the eval grids do.
- `128_d1024x16_swpin_5ep` trained at σ 1 (its config predates the default), so the continuation at usui lr 1e-4 (archive 2026-11-05) is the first long run at σ 0.7.

### G8. What the norm ceiling leaves open

Every Usui matrix takes the trust step inside the ball of its init norm (SPEC §6, archive 2026-11-12); this differs
from the former pin only where a step shrinks a norm.
The sandwich norm was necessary against the loss spikes (2026-11-06); no arm checks it again.
Open:
- Whether a learned scalar gain per bounded matrix (direction bounded, gain on the fallback) helps: some scale is free in
  the gates (`out` and the v rows of `qkv`, the `w_up`·`w_down` product), some is not (`w_gate` sets the SiLU operating
  point; `ada`, `time_mlp` and `text_proj` the conditioning scale; `patch_embed` the input scale).
- A bound per output row instead of per matrix.
- Bounds per role, and a zero-initialized matrix: Usui's bounds are multiples of the init norm, so a matrix that
  starts at zero cannot take them; none does here (§5).
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
No random-frame comparison run: by step 1k the fitted frame has fully caught up, with zero difference. Both are usuitrack changes, on usuitrack `lab`. Also open: the pinned, non-blocking copy in `Stream` (4 syncs per step). Block compile measured
on a free GPU: FACTS, Sweep throughput.

---

## Group I. Training schemes beyond flow matching

### I1. Priming: reconstruction pretraining, then distributional post-training

Its own topic doc: `topics/priming.md` (the source, the direction, the arms and their reads, what is open). State (2026-11-12): a side quest; no priming code on the trunk. Next arm: clean context tokens (t = 0, no loss) at a 1 to 25 % share. REPA only on precomputed targets of one size.
