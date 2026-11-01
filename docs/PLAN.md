# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds measured numbers; `archive/` holds
everything closed. See `AGENTS.md` for how this file is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs or code. Only 300-step smoke runs have trained; each
"how to settle" is a proposal, not an agreed arm.

---

## Group A. Usui wiring

### A2. How does Usui treat 1-D params?

Norm gains (QK-RMSNorm, final norm), biases and the null embedding are not
matrices. Usui rejects any `ndim != 2` param (`optimizer.py` `add_param_group`); they
go to a caller-owned fallback (AdamW or `StochasticAdamW`, Usui SPEC). Settled by code
reading, and the harness runs that split (`src/tinydit/optim.py`). Daniel to close.

### A3. Does the zero-initialized final linear get the rank it needs under Usui?

Usui writes each step inside its frame, r = 13 for the final linear `[128, 512]`, so the
matrix only gains rank as the frame rotates. At 300 steps its stable rank is near the
frame's rank, and its loss below t ≈ 0.7 lags; a rank-64 frame on the same code closes
most of the gap to the final linear on the fallback, at either step size. Early loss does not decide it: over 9000 steps (`128_fixed_2e-2_9k`)
the low-t loss catches up while the frame stops rotating after about step 1000 and the
weight concentrates in it (FACTS, Fixed-step runs). Open: whether a final linear held
near a 13-dimensional subspace costs anything the loss does not show (the eval grid's
streaks are one candidate, F1 another). Settle: Daniel decides between the default
frame, a larger frame for this matrix, or another route; a same-length rank-64 arm
would compare.

---

## Group B. Data and conditioning

### B1. Are Primal 2x2 latents unit-normal per image and channel?

The model card says so, and SPEC §1 relies on it (no scale/shift factors, patch
embed init for unit-normal inputs). Settle: a script that encodes a sample of
text-to-image-2M at 128 px and 256 px and reports per-image, per-channel mean and
std. Evidence: FACTS "Primal 2x2" (statistics, flips, stability under small changes). Daniel to close: whether the measured spread is close enough to the card's claim.

### B2. What is EmbeddingGemma 2's output norm?

SPEC §1 stores the embedding L2-normalized as the model returns it; §5 scales it
by √768 for the text projection and initializes the null embedding from the
encoder's empty caption. Unsure the model returns unit norm. Settle: a script that embeds a
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

Rides on Usui and the LR sweep. Open whether tokens are the right noise unit for
diffusion (t sampling adds its own gradient noise). Evidence: at 128 px, batch 256
gives 64 × 256 ≈ 16k tokens per step, matching the LFM setup on tokens. Throughput
at 128 px is measured (FACTS, GPU smoke runs); 256 px is unmeasured. Settle: the sweep in C2.

### C2. Which LR from scratch, and how long a warmup?

The first experiment. The lr is Usui's fixed step (SPEC §6): a moment that fills
its rank writes `‖ΔW‖ = lr·√(max(1, out/in)·r)`. At 2e-2 that is a nominal relative step of
0.45–0.64% for the fan-in matrices at init and 1.9% for the shared adaLN
projection (its rows start at std 0.02, F2), against 0.3% for the earlier relative
3e-3. Proposed sweep at 128 px: 1e-2, 2e-2, 5e-2. Warmup: the smoke config uses 50
steps; the common rule, Daniel's note, is 5% of the run, and the run length is not set. The 300-step
reads of 20, 50 and 100 are in FACTS (Fixed-step runs); a longer run reads it better. The conv, the adaLN tables and the
null embedding train on the fallback at `fallback_lr`, a second sweep axis; the null
embedding (elements ≈0.036) takes full-size Adam steps from the 10% dropped rows, so
it may leave the encoder's manifold fast unless its LR is set apart. Under the
relative step at 3e-3, constant, `grad_moment_cosine` turned negative after about
step 100 (FACTS, GPU smoke runs), which Usui reads as overshoot. No AdamW reference
run. The harness has no decay; Usui's README recommends one.

### C3. Timestep shift values per stage

The rule is in SPEC §4; the base value is not chosen. Settle: with the sweep or a short follow-up.

---

## Group D. Sampling and eval

### D1. EMA weights

Daniel leans no. Option on the table: render the eval grid from both raw and EMA
weights and drop EMA if they match.

### D2. Guidance scale

Tuned on the eval grid.

### D3. Eval prompt set

Not chosen.

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

SPEC says "small random". Std 0.02 on the shared 512→3072 projection makes the
blocks' shift/scale far from zero at init (FACTS, init), so `1 + scale` starts far
from 1. A fan-in-scaled or 10× smaller init would start near identity.

### F3. Final-layer modulation input

The final layer adds raw `c` to its table (PixArt); the blocks use `SiLU(c)` through
the shared projection. Raw `c` is of order 1 per element, for real and random texts
alike (FACTS, init), so the final `1 + scale` starts far from 1, of either sign. The
final linear is zero, so `F = 0` still holds; the scale sets how fast it leaves zero.

---

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
