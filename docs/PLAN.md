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

### C1. Batch size and step budget per stage

Rides on Usui and the LR sweep. Is batch 256 needed? A smaller batch may want a lower lr,
and a lower rank fraction if `tangent_live_fraction` falls (0.993–0.996 at batch 256,
rank fraction 0.1). Open whether tokens are the right noise unit for
diffusion (t sampling adds its own gradient noise). Evidence: at 128 px, batch 256
gives 64 × 256 ≈ 16k tokens per step, matching the LFM setup on tokens. Throughput
at 128 px is measured (FACTS, GPU smoke runs); 256 px is unmeasured. The 1000-step
ablation at a constant number of images (FACTS, 1000-step ablations; one seed): batch 64,
at the same lr, is 0.127 lower per image and about a tie per training time against a
batch-256 run of the earlier model, at 353 img/s against about 850; `tangent_live_fraction`
does not fall. Settle: the sweep in C2.

### C2. Which LR from scratch, and how long a warmup?

The lr is Usui's fixed step (SPEC §6): a moment that fills its rank writes
`‖ΔW‖ = lr·√(max(1, out/in)·r)`. The smoke config takes 2e-3 with rank fraction 0.1
and beta 0.99, Daniel's "for now" after 2e-2 read too high (archive 2026-11-01, final frame and smoke settings; eval grid, beta 0.99). The
negative `grad_moment_cosine` followed beta more than lr (FACTS, Runs at usui lr 2e-3). Settle: a sweep around 2e-3 at 128 px once the run
length is set by the data (each image seen once, archive 2026-11-01, final frame and smoke settings). Warmup is 100 steps; the common rule,
Daniel's note, is 5% of the run. The 300-step reads at warmup 20, 50 and 100 are in FACTS (Fixed-step runs). The conv, the adaLN tables and the null embedding
train on the fallback at `fallback_lr` with betas (0.9, 0.99), a second sweep axis; they move
faster than the Usui matrices, and the null embedding turns away from its init in the
first 1000 steps (FACTS, Runs at usui lr 2e-3). Daniel: "we should also lower fallback lr i
think. unless you say it evolves fine. probably also its beta." No AdamW reference run. The
harness has no decay; Usui's README recommends one. The 1000-step ablation (FACTS,
1000-step ablations; one seed): fallback lr 3e-4 is worse than 1e-3 by 0.014 overall and
in every t bin (0.005 to 0.044), near the 0.011 yardstick there; betas (0.9, 0.999)
cannot be told from (0.9, 0.99).

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

### C4. The final linear: Usui's frame or the fallback?

SPEC §6 gives the final linear Usui's largest frame, r = 64 (archive 2026-11-01, final
frame and smoke settings: at 9000 steps r = 64 ended about 0.010 below r = 13, the low-t
gap closing). In the 1000-step ablation (FACTS, 1000-step ablations) the final linear on
the fallback is 0.076 lower over steps 501–1000 (one seed), mostly in t bins 1–3, tapering
to none by bin 8 (bin 1: 1.19 against 1.54). Unknown whether the lead holds or closes over a full run. Settle: the two arms
over one pass or more of the data. Daniel decides.

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

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
