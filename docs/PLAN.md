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

---

## Group H. Harness throughput

### H1. How much of the fixed per-step cost can go?

About 140 ms of each step is fixed, independent of the batch, and the step is launch-bound: the Usui step is about
14,700 of 19,500 launches and syncs 9 times, once per `eigh` (FACTS, Sweep throughput and host syncs). The blocks
are compiled once every Molegrad frame has fit; that cut forward/backward from ~5,400 to ~3,400 launches, a modest
share of the step. Daniel: "Simple blockwise compile worked well in the past. Cuda graph is also open." Open, each a
usuitrack change and Daniel's to choose: batch the per-matrix loop over same-shape matrices (12 blocks per role); one
`eigh` call for all rank groups instead of 9; diagnostics only on record steps; a CUDA graph around the step, which
needs `eigh` either outside the graph or replaced by a sync-free solver. Also open: the pinned, non-blocking copy in
`Stream` (4 syncs per step). Settle: a clean img/s benchmark when the GPU is free, then per option.

---

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
