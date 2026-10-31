# Open design questions

What is unsure, how each would be settled, and the current evidence. `SPEC.md`
says what the design is; `FACTS.md` holds measured numbers; `archive/` holds
everything closed. See `AGENTS.md` for how this file is worked.

Questions are grouped so a session can take **one group** and leave the rest
alone. Inside a group the items share runs or code. Nothing has been run yet; each
"how to settle" is a proposal, not an agreed arm.

---

## Group A. Usui wiring

### A1. Does Usui's target-norm override work as SPEC §5 assumes?

Trust-region LR is relative to weight norm, so a zero-initialized role never moves
without an override. SPEC §5 marks the adaLN gate rows, the depthwise conv and the
final linear as needing one, with the target norm a fan-in init of the same shape
would have. Unsure whether Usui accepts and applies such an override as described.
Settle: a small script that builds the roles, applies the overrides and shows the
first steps moving the zero-init weights. Evidence: none yet.

### A2. How does Usui treat 1-D params?

Norm gains (QK-RMSNorm, final norm), biases and the null embedding are not
matrices. Assumed: Usui's 1-D fallback path handles them without a target-norm
override. Settle: read UsuiTrack for what it does with 1-D params, then confirm in
the same script as A1 that they update. Evidence: assumed, not verified.

---

## Group B. Data and conditioning

### B1. Are Primal 2x2 latents unit-normal per image and channel?

The model card says so, and SPEC §1 relies on it (no scale/shift factors, patch
embed init for unit-normal inputs). Settle: a script that encodes a sample of
text-to-image-2M at 128 px and 256 px and reports per-image, per-channel mean and
std. Evidence: the model card claim only.

### B2. What is EmbeddingGemma 2's output norm?

SPEC §1 stores the embedding L2-normalized as the model returns it; §5 scales it
by √768 for the text projection and initializes the null embedding as a random
unit vector. Unsure the model returns unit norm. Settle: a script that embeds a
sample of captions in bf16 and fp32 and reports the norm distribution. Evidence:
none yet.

### B3. Do short prompts get ignored?

Training captions are paragraph-length, inference prompts are short. Settle: judge
it on the eval grid with short prompts. If they are ignored, add a second
embedding per image from the first sentence and sample 50/50 (another ~3 GB).
Evidence: none; not built.

---

## Group C. LR sweep and step budget

### C1. Batch size and step budget per stage

Rides on Usui and the LR sweep. Open whether tokens are the right noise unit for
diffusion (t sampling adds its own gradient noise). Evidence: at 128 px, batch 256
gives 64 × 256 ≈ 16k tokens per step, matching the LFM setup on tokens. Throughput
estimate (≈30M params, 6·params·tokens FLOPs per image, ~10 TFLOPS effective,
unbenchmarked): ~900 img/s at 128 px, ~200 img/s at 256 px; one epoch over 2M
images ~40 min at 128 px, ~3 h at 256 px. Settle: the sweep in C2 and measured
throughput.

### C2. Which relative LR from scratch?

The first experiment. Sweep at 128 px: 1e-3, 3e-3, 1e-2 relative. Evidence: LFM
runs use `lr 4e-5 × norm`, beta 0.95, at 16k tokens per step, a rate for adapting
a pretrained model; from-scratch recipes imply per-step relative updates of roughly
5e-3 to 5e-2 (estimate, not measured on Usui). At 4e-5 a zero-init gate needs ≥25k
aligned steps to reach its target norm; around 1e-2 it needs ~100. No AdamW
reference run.

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

---

## Group E. Housekeeping

### E1. Repo name

`tiny-dit` is a placeholder.
