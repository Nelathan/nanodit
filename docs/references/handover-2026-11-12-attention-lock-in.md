# Handover of 2026-11-12: attention lock-in and backward temperature

Kept verbatim as received on 2026-11-17: a handover written by a chat session that never saw this repo. `topics/attention-lock-in.md` carries what holds.

---

# Handover: attention lock-in and backward temperature

Written 2026-11-12 from a chat session between Daniel and Claude (claude.ai, not
Claude Code). For the next Claude Code session in the tiny-DiT repo. **The chat
session never saw this repo's code.** Everything about the model below comes
from Daniel's descriptions; check it against the code before building.

Status: one direction Daniel called promising (backward temperature), one
sensor that should come first, several parked ideas. No run is agreed. Each
design choice below is a proposal for Daniel, not a decision.

## Source

*Stuck in a Loop: Runaway credit ratcheting in attention matrices*, Eyler et
al., Hyperstition, 2026-09-09, https://hyperstition.cc/stuck-in-a-loop.
Read it yourself before relying on a number: the chat session saw it through a
fetch tool that paraphrases.

The claim, for one head with `y_i = sum_j a_ij v_j` and `delta_i = dL/dy_i`:

- value gradient: `a_ij * delta_i`
- score gradient: `a_ij * (u_ij - sum_l a_il u_il)`, with `u_ij = delta_i . v_j`
  (how much the loss wants `y_i` moved toward `v_j`; the paper calls the
  centered term the advantage)

Every term carries `a_ij`. A key at attention `eps` gets `eps` of the signal,
however useful it is. Since softmax weights are exponential in score gaps, early
winners keep winning. The paper reports that attention at step 1k of a 10k run
is closer to its own final state than two runs' final states are to each other.

Their fix (NATQ: inverse-Fisher preconditioning of each token's query gradient,
then rescaling to the original norm) spread attention and **made validation
loss worse**. Caveats they or the text carry: one 124M model, 5k steps, the
lock-in result names no metric, no code.

## Why it matters here

Daniel wants heads that keep exploring through training without hardening
early, while still exploiting the easy early signal. The model has attention in
about a quarter of its layers plus cross-attention, so a few attention layers
carry all the global routing.

## Step 0: the sensor (do this before any mechanism)

Unverified premise: that lock-in happens in this model at all. Daniel deleted
the intermediate model checkpoints of the 30k run for disk space, so there is
nothing to read retroactively. The frame analysis he did from optimizer
checkpoints does not answer this: the UsuiTrack frame follows the leading
eigenspace of the gradient on (most likely) the input side and settles by
design. It is not a read of q/k evolution.

Proposal for the read, to be logged during the next run:

- A fixed probe batch: a few fixed latents, fixed noise, and fixed timesteps
  spanning high, middle and low noise.
- At log-spaced steps, for every attention layer and head (self and cross):
  mean row entropy, mean top-1 mass, and the attention maps themselves saved as
  fp16.
- Reads: similarity of maps at step `t` to the final maps, as a curve over `t`;
  entropy per head over `t`. With a second seed: similarity between the two
  runs' final maps. Lock-in in the paper's sense is `sim(early, final_same_run)
  > sim(final_A, final_B)`.

Open choices for Daniel: the similarity metric (the paper does not name one;
mean per-row Jensen-Shannon divergence and cosine similarity are the obvious
candidates, neither validated); whether a second seed is worth its cost; how
many probe samples. Disk is tight. Map storage is
`layers * heads * samples * timesteps * N_query * N_key * 2 bytes` per dump;
compute it for the real token counts before choosing, and prefer few probe
samples over few dump steps. Daniel prefers diagnostics he can read from logs
or files over wandb.

## Step 1: backward temperature (proposed, awaiting Daniel's go)

Forward pass unchanged. On the backward pass the score gradient uses a
flattened copy of the attention:

```python
a  = softmax(s, dim=-1)
ab = softmax(s / tau_b, dim=-1)
w  = ab + (a - ab).detach()   # value: a.  d/ds: through ab
y  = w @ v                    # value gradient still weighted by a
```

What this does:

- `tau_b = 1` is plain backprop. As `tau_b` grows, the score gradient tends to
  `u_ij` minus its plain mean, which is the natural gradient in score space (up
  to a constant). `tau_b` slides between the two.
- `tau_b = 2` is a square root on every attention weight in the backward pass:
  a key buried at 1e-6 gets gradient as if it sat near 1e-3. Winners still lead.
- It stays linear in `delta`, so zero-mean noise from single tokens cancels in
  the sum over tokens and steps. NATQ's per-row renormalization broke that.

Agreed constraints from the session:

- **Scores only.** Values keep the true `a_ij`; otherwise values of unread keys
  are trained on noise.
- **No norm compensation.** UsuiTrack erases gradient scale (unit projected
  gradient into the moment, then the polar factor). Only each token's share of
  the projected gradient survives, and that stays loss-weighted. Daniel: the
  gradient norm is irrelevant for Usui. The `1/tau_b` factor needs no handling.
- **Constant `tau_b`,** no schedule: exploration is meant to continue.
- **One knob.** No thresholds on which keys get gradient. The tracker's tangent
  spectrum is a smooth power law with no signal/noise edge; the session expects
  the same over keys (expectation, not measured).

Division of labour, as the session understood it: backward temperature opens
the gate before the sum over tokens; UsuiTrack's polar step handles scale after
it. Polar alone cannot rescue a buried key, because an `eps`-scaled signal is
under the noise before polar sees it.

Implementation notes:

- Needs the explicit attention matrix, so no fused
  `scaled_dot_product_attention` on these layers. Fine at this token count;
  check memory at 256 px.
- Tests before any run: (1) forward output is bit-identical for any `tau_b`;
  (2) at `tau_b = 1` the gradients equal the baseline's at float tolerance;
  (3) at `tau_b > 1` the value gradient is unchanged and only q/k gradients
  differ.
- **Check the UsuiTrack side for the q and k matrices.** The new signal lives
  on the output side of `W_Q`/`W_K` (which keys pull the query). With the frame
  on the input side, the projected gradient keeps the output side whole and the
  signal passes. A frame on the output side would filter exactly the
  exploratory component. The chat session read `usuitrack` `lab` at `be3f298`;
  this repo's variant (hyperball pin, fixed per-matrix scalar, rank fraction
  0.1, lr 4e-3) was not read.

Open choices for Daniel:

- `tau_b`. Proposal: 2. Derived, untested.
- Self-attention only, or cross-attention too. Not discussed. Cross-attention
  over text tokens has its own lock-in risk and far fewer keys.
- Arm design. Minimal: baseline against `tau_b = 2`, same seed, same images
  seen, the Step 0 sensor on both.

Prediction, stated before any run: entropy rises modestly, far less than NATQ's
jump, since the forward pass stays sharp; the `sim(t, final)` curve keeps moving
later into the run. Loss is unknown; NATQ lost on it. The verdict here is the
samples, as usual for this project. Kill criterion: no later attention movement
than baseline, or clearly worse samples at equal images seen.

## Parked, in case Step 1 earns a follow-up

- **Interleaved Head Attention as a priming phase** (Meta, arXiv 2602.21371;
  the session read the abstract only). Pseudo-heads are learned linear mixes of
  all heads' q/k/v, giving up to `P^2` patterns per head. Daniel's idea: use it
  early to surface more distinct aspects, then keep a favourite subset. Session
  conclusions: the learned mixing coefficients can serve as the router
  (sparsify, then freeze), or decide from a checkpoint; consolidation must be
  gradual, because representations co-adapt to the routing they trained under;
  selecting by coverage alone repeats NATQ's mistake, so selection needs a
  usefulness term. Expected (unmeasured) benefit: each q/k parameter receives
  gradient through many softmax gates, so a key buried in one pattern can be
  visible in another.
- **Competition across heads** (a softmax over heads per key, as in Slot
  Attention) to force distinct heads. Raised by Claude in answer to Daniel's
  "force q to be unique, like k-means". Not taken further.
- **Start broad, sharpen later** (a forward temperature schedule). Judged blunt:
  broad forward attention averages every value into a weak signal. Backward
  temperature was preferred because the forward pass keeps exploiting.

Dropped:

- **Consensus-gated boosting** of buried keys. Daniel has no hopes for it; a
  gate that requires head agreement rewards redundancy, and he wants diversity.
- **NATQ as published.** Whitens per token before aggregation.

Belongs elsewhere (LLM work, not this repo): GDN2 in place of softmax attention
at layer 0, and the untested block patterns (softmax attention in the middle of
three, or linear/soft/soft/linear).

## Verified, assumed, open

- **Verified:** UsuiTrack's scale erasure, read in code on `lab`.
- **Derived by hand, never run:** the straight-through form above (value
  gradient through `a`, score gradient through `ab`). The three tests listed
  under Step 1 are its check.
- **Assumed:** this repo's attention layout, token counts, and that its
  UsuiTrack variant erases scale the same way and frames q/k on the input side.
- **Open:** whether lock-in occurs here at all (Step 0); every choice marked
  for Daniel above.
