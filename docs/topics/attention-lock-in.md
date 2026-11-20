# Attention lock-in, and a backward temperature against it

Topic doc for PLAN Group L (L1–L3). Opened 2026-11-17 from `references/handover-2026-11-12-attention-lock-in.md`, a
chat session that never saw this repo. The parts it assumed about the model are checked against the code below.
Measured numbers go to `log/`, closures to `archive/`.

## Source

Eyler et al., *Stuck in a Loop: Runaway credit ratcheting in attention matrices*, Hyperstition, 2026-09-09,
https://hyperstition.cc/stuck-in-a-loop. The chat read it through a fetch tool that paraphrases, and it is **not read
here**. Read it before relying on a number.

The claim, for one head with `y_i = Σ_j a_ij v_j` and `δ_i = ∂L/∂y_i`:
- **Value gradient:** `a_ij · δ_i`.
- **Score gradient:** `a_ij · (u_ij − Σ_l a_il u_il)`, with `u_ij = δ_i · v_j`. The centred term is what the paper
  calls the advantage: how much the loss wants `y_i` moved toward `v_j`.
- **Why winners keep winning:** every term carries `a_ij`, so a key at attention ε gets ε of the signal however
  useful it is. Softmax weights are exponential in score gaps.
- **The reported result:** attention at step 1k of a 10k run is closer to its own final state than two runs' final
  states are to each other.
- **Their fix fails:** NATQ, inverse-Fisher preconditioning of each token's query gradient rescaled to its original
  norm, spreads attention and **worsens validation loss**.
- **Caveats:** one 124M model, 5k steps, no metric named for the lock-in result, no code.

Why it matters here: global routing rests on a few layers. At depth 24, 6 of 24 blocks have attention, each with
self-attention over 64 tokens at 128 px and cross attention over the caption. The aim is heads that keep exploring
through training without hardening early, while still exploiting the easy early signal.

## The model, checked against what the chat assumed (`12c2028`)

- **Attention blocks:** 3, 7, …, 23 at depth 24. 12 query heads of 64 and 3 KV heads, in self-attention and in cross
  attention alike.
- **Fused attention:** both use `F.scaled_dot_product_attention`. A backward temperature needs the explicit
  attention matrix on the layers it touches. That is cheap at 64 tokens, and at 256 tokens too.
- **Frame side:** every Usui matrix is framed on its input side (SPEC §6, `side = "right"`), `qkv`, `cross_q` and
  `cross_kv` included. The chat's condition holds: the projected gradient keeps the output side whole, which is where
  the new signal on `W_Q` and `W_K` lives. Its warning about a frame on the output side does not apply.
- **Scale erasure:** Usui takes the trust step (SPEC §6), a step relative to ‖W‖ from the polar factor of the moment.
  The gradient's overall scale is erased; each token's share of it survives. The chat verified this on usuitrack
  `lab` at `be3f298`; for the pin in use, `92b23bc`, it is assumed from SPEC §6, not read in its code.

## L1. Does lock-in happen here? The sensor

The premise is unverified. Usui's frame analysis does not answer it: the frame follows the leading eigenspace of the
gradient on the input side and settles by design, so it is not a read of q/k evolution.

The read:
- A fixed probe batch: a few fixed latents with fixed noise, at fixed t spanning high, middle and low noise.
- At log-spaced steps, for every attention layer and head, self and cross: mean row entropy, mean top-1 mass, and
  the maps themselves saved as fp16.
- Curves: the similarity of the maps at step s to the final maps, over s, and the entropy per head over s.
- Lock-in in the paper's sense: `sim(early, final of the same run) > sim(final A, final B)`. The right side needs a
  second seed.

Storage per dump, at depth 24 and 128 px:
- Self-attention: 6 layers × 12 heads × 64 × 64 × 2 bytes ≈ 0.59 MB per probe sample and t.
- Cross attention: about 1.8 MB per probe sample and t, at a 200-token caption.
- At 256 px: self-attention grows 16×, to about 9.4 MB.
- Disk is not the constraint (HANDOVER, disk), so the dump steps need not be few.
- At the move to 256 px, with 4× the keys per query, the Gemini thread warns of an entropy spike at the switch (QK
  norm is in the trunk). The sensor run across the switch would show it.

The 30k run's checkpoints were deleted, so no read existed when the source was written. A partial read now exists
without a new run. The crashed 1ep run left checkpoints at steps 4000, 8000, 12000 and 16000
(HANDOVER), which gives the `sim(s, 16000)` curve and the entropies, without the second-seed baseline. Whether those
checkpoints are kept for it is the maintainer's call (AGENTS, checkpoints).

Open choices:
- The similarity metric. The paper names none; the candidates are mean per-row Jensen–Shannon divergence and cosine,
  neither validated.
- Whether a second seed is worth its cost.
- How many probe samples.

Diagnostics should be readable from files or logs, not only from wandb. The same code serves the free read of J3
(`topics/conditioning.md`): whether cross maps concentrate on a de facto sink.

## L2. Backward temperature

*Proposed, not agreed.* The forward pass is unchanged. On the backward pass the score gradient uses a flattened copy
of the attention:

```python
a  = softmax(s, dim=-1)
ab = softmax(s / tau_b, dim=-1)
w  = ab + (a - ab).detach()   # value: a.  d/ds: through ab
y  = w @ v                    # value gradient still weighted by a
```

- **The range:** `tau_b = 1` is plain backprop. As `tau_b` grows, the score gradient tends to `u_ij` minus its plain
  mean, which is the natural gradient in score space up to a constant. `tau_b` slides between the two.
- **`tau_b = 2`:** a square root on every attention weight in the backward pass. A key buried at 1e-6 gets gradient as
  if it sat near 1e-3, and winners still lead.
- **Linearity:** it stays linear in δ, so zero-mean noise from single tokens cancels in the sum over tokens and steps.
  NATQ's per-row renormalisation broke that.

Constraints, as the chat session agreed them:
- **Scores only.** Values keep the true `a_ij`; otherwise the values of unread keys train on noise.
- **No norm compensation.** Usui erases gradient scale (above), so the 1/`tau_b` factor needs no handling.
- **A constant `tau_b`**, with no schedule, since exploration is meant to continue.
- **One knob.** No thresholds on which keys get gradient. The expectation that the spectrum over keys is a smooth
  power law with no signal/noise edge is not measured.
- **Division of labour:** backward temperature opens the gate before the sum over tokens, and Usui's polar step
  handles scale after it. The polar step alone cannot rescue a buried key: a signal scaled by ε is under the noise
  before polar sees it.

Tests before any run:
1. The forward output is bit-identical for any `tau_b`.
2. At `tau_b = 1` the gradients equal the baseline's at float tolerance.
3. At `tau_b > 1` the value gradient is unchanged; only the q/k gradients differ.

The straight-through form above is derived by hand and has never run; these three tests are its check.

Open choices:
- `tau_b`. The proposal is 2: derived, untested.
- Self-attention only, or cross attention too. Not discussed. Cross attention over text tokens has its own lock-in
  risk and far fewer keys.
- The arm: minimally, the reference against `tau_b = 2` with the same seed and images seen, and the L1 sensor on both.

*Prediction*, stated before any run: entropy rises modestly, far less than NATQ's jump, since the forward pass stays
sharp, and the `sim(s, final)` curve keeps moving later into the run. The effect on loss is unknown; NATQ lost on it.
The verdict is the samples. Kill criterion: no later attention movement than the baseline, or clearly worse samples
at equal images seen.

## L3. Parked, in case L2 earns a follow-up

- **Interleaved Head Attention as a priming phase** (Meta, arXiv 2602.21371; the chat read the abstract only).
  - What it is: pseudo-heads are learned linear mixes of all heads' q/k/v, giving up to P² patterns per head. The
    idea is to use it early to surface more distinct aspects, then keep a favourite subset.
  - The chat's conclusions:
    - The mixing coefficients can serve as the router (sparsify, then freeze), or the subset is chosen from a
      checkpoint.
    - Consolidation must be gradual, because representations co-adapt to the routing they trained under.
    - Selection by coverage alone repeats NATQ's mistake, so selection needs a usefulness term.
  - Expected, unmeasured: each q/k parameter gets gradient through many softmax gates, so a key buried in one pattern
    can be visible in another.
- **Competition across heads:** a softmax over heads per key, as in Slot Attention, to force distinct heads ("force q
  to be unique, like k-means"). Not taken further.
- **Start broad, sharpen later:** a forward temperature schedule. Judged blunt, because broad forward attention
  averages every value into a weak signal. The backward temperature is preferred because the forward pass keeps
  exploiting.

## Dropped

- **Consensus-gated boosting of buried keys.** A gate that requires head agreement rewards redundancy, and the aim
  is diversity.
- **NATQ as published.** It whitens per token before aggregation.
- **Out of scope for this repo:** GDN2 in place of softmax attention at layer 0, and block patterns such as
  softmax attention in the middle of three, or linear/soft/soft/linear. These are LLM work.
