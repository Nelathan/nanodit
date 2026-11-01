# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds measured numbers; `archive/` holds
everything closed. See `AGENTS.md` for how this file is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs or code. Only 300-step smoke runs have trained; each
"how to settle" is a proposal, not an agreed arm.

---

## Group A. Usui wiring

### A1. How do the zero-init roles train under Usui? (decided 2026-11-01, being built)

SPEC §5 assumes a per-role target-norm override. Usui has none, on any branch: the
step is `W -= lr·‖W‖·lift(O)` (`usuitrack/optimizer.py` `_write_weight`), and its SPEC
says a matrix with `‖W‖ = 0` never moves and needs a non-zero init or the fallback
optimizer. FACTS (Usui wiring) has what Usui does with each zero-init role: the
final linear never moves, the conv is rejected, the shared adaLN projection moves
through its non-zero rows, the per-block tables are accepted. The tables are put on
the fallback by analogy with Usui's README, which routes `nn.Embedding` tables there.
Options: (a) zero-init roles (final linear, conv, tables) on
the AdamW fallback, as Usui documents; (b) small non-zero init for them, giving up
`F = 0` at step 0; (c) an override in Usui. Decided by Daniel: (c), as a
per-group norm floor in Usui, with no frame fit from a zero gradient, on usuitrack
branch `zero-init`; this closes into the archive when built. Until then
`src/tinydit/optim.py` runs (a). Its costs: the zero-init roles grow only at
the absolute fallback LR, so `fallback_lr` becomes a second sweep axis beside C2; the
null embedding (elements ≈0.036) takes full-size Adam steps from the 10% dropped rows,
so it may leave the encoder's manifold fast unless its LR is set apart; and with the
final linear at zero, every tracked matrix has exactly zero gradient on step 1, so
Usui fits each frame from a zero `G` (coordinate axes; under accumulation from the
first micro-batch only) and turns it at most 0.02 rad per step after (Usui SPEC).
How many steps that costs is unmeasured; the wiring check in FACTS shows the frame
does turn.

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

### C2. Which relative LR from scratch?

The first experiment. Sweep at 128 px: 1e-3, 3e-3, 1e-2 relative. Evidence: LFM
runs use `lr 4e-5 × norm`, beta 0.95, at 16k tokens per step, a rate for adapting
a pretrained model; from-scratch recipes imply per-step relative updates of roughly
5e-3 to 5e-2 (estimate, not measured on Usui). Until A1 is built, the zero-init roles
train on the fallback at `fallback_lr`, not at this rate. At 3e-3 `grad_moment_cosine`
turns negative after about step 100 (FACTS, GPU smoke runs), which Usui reads as
overshoot. No AdamW reference run. The harness runs a constant LR, no warmup or
decay; Usui's README recommends a schedule, and SPEC says nothing yet.

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
