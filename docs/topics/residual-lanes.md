# The residual stream as lanes: who writes where, long skips, re-injection

Topic doc for PLAN Group M (M1–M3). Opened 2026-11-17 from `references/handover-2026-11-12-residual-gating.md` §3.1
and §5 (the two-stream reading, SimpleHC) and `references/gemini-thread-adaln-and-architecture.md` (scratchpads, U-Net
skips). Related: G3 (the common mode, register tokens), G10 (the conv block), `topics/block-design.md`. Measured
numbers go to `log/`, closures to `archive/`.

## The stream as built (SPEC §3, `12c2028`)

768 channels: the first 128 are the **image slice** (one 8×8 patch of the 2-channel Primal latent), the other 640
are **meta**.

| Writer | Writes | Gate |
|---|---|---|
| patch embed | the slice: the raw patch, identity; meta: `Linear(128 → 640)` | none |
| conv block | all channels (architecture a2; a1 wrote meta only) | adaLN gate per image and channel; C per token and channel, after the read's norm |
| self-attention | all channels | adaLN gate per image; the sigmoid gate per token on the heads, before `out` |
| cross attention | all channels | static `cross_gate` per channel |
| MLP | all channels | adaLN gate per image and channel |

The readout is the slice after the last block: no norm, no linear.

**Reading** (handover §3.1): this is a two-stream residual with fixed, role-encoding write masks and an identity
transition.
- Spatial mixing happens in the meta workspace. A static conv is the same filter at every position, so what it wrote
  into the prediction directly would be a blur.
- Per-token translation may touch the prediction. The MLP is per token, so its writes to the slice can be sharp.

The 2026-11-16 decision for the meta-only conv (PLAN G10) rested on the same principle; it is reversed: a conv gated per
token by C writes all channels (archive 2026-11-19-conv-writes-all-channels, 2026-11-20-conv-block-lfm-norm-read).

## M1. Learned per-token write gates onto the slice

*Parked in the source, not adopted.* The proposal: learned soft write gates per token onto the slice, initialized at
the current masks.

The lesson from Hyper-Connections (below): learned gating is where HC's quality comes from, and fixed gates hurt in
LMs. Here the masks encode a function, a stronger prior than anything HC learns, so that lesson need not transfer.

What exists already:
- Per image and channel: the adaLN gate.
- Per token: only the self-attention's sigmoid gate, which acts on the heads before `out`, not on the slice channels.
- Not per token at all: the MLP's write to the slice, and the cross attention's.

The candidate is a per-token gate on the slice part of those two writes. Settle: one arm, read on loss and on samples
for sharpness.

## M2. Long skips (U-Net style)

*Proposal* (the thread, from the modded-nanogpt speedrun): block i's output added into block L − 1 − i's input
through a learned gate initialized near zero.
- The thread says the speedrun added these in records #9–#11 and gated them later (unverified).
- In diffusion, U-ViT and HDiT carry long skips, U-ViT as an essential part (recalled, not read here).
- Here the skip would write into the meta channels only, so the slice stays the working latent.
- *Hypothesis*: early blocks hold local structure from x_t, and handing it to the late blocks frees the middle ones.

Settle: one arm at depth 24 against the reference. That needs a code change: the block loop keeps the first half's
outputs.

## M3. The input re-injected

From the modded-nanogpt speedrun's value embeddings, recalled and unverified: the token embedding is mixed into the
attention values at some layers. The analogue here is the patch embed's meta projection, or x_t itself, added to the
values of each attention block or re-added to the meta channels every k blocks.

The t re-injection of J1 (c) is the same mechanism for the timestep. *Hypothesis*: the deep blocks then need not carry
the input forward through the stream. Settle: after M2, or with it if both change the block loop.

## M4. The mixers write context only; printing onto the slice is its own step

*Proposal*. Self-attention and cross attention write the meta channels only, so the meta lanes hold the context and the slice changes only where something
prints the context onto it. **The MLP prints:** it is per token, so it can write sharply. The write map becomes:
mixers → meta, MLP → all channels.

*Not wanted* (2026-11-17): a dedicated print step at the end with the slice left at x_t through the stack. The
residual is worked on continuously; keeping x_t reachable late is the job of a skip (K1), not of channels held idle.

What it tests: whether the slice writes of attention and cross attention help (they are global and per token, unlike
the conv's blur) or only add noise to the prediction. GDN2 writing the image channels added 0.0008 (log 2026-11-20-measurements-from-facts), which is
the one measured hint, from a conv block.

Interactions: M1 (per-token gates on the slice writes) is the soft form of the same question; K1 is where a skip
carries x_t forward. Settle: one arm against the reference.

## Background

**Hyper-Connections**, as digested in the handover §5 from SimpleHC (Zhang et al., arXiv 2611.05984). The handover's
session read the paper in full; it is not read here.
- **HC / mHC:**
  - n residual streams, a read gate h_pre, a write gate h_post and a mixing matrix H_res: `X' = H_res X + h_post oᵀ`.
  - mHC makes H_res doubly stochastic by ~20 Sinkhorn iterations.
  - Its cost is the memory traffic of the wide state: 18C per token per sublayer even fused, which is 7–27 % extra
    training time for under 1 % extra FLOPs.
  - At 230B and 10× lr, mHC spiked to a gradient norm of 1.9e6: Sinkhorn failed to converge, and the dH_res path
    dominated.
- **identityHC (H_res = I)** matches mHC: 55.4 against 55.0 at 15B, 62.8 against 62.9 at 30B.
- **SimpleHC's Table 2:**
  - Learned read/write gating and unconstrained mixing are two independently effective groups, each about equal
    to mHC, and best together (56.2).
  - Sinkhorn wipes out mixing's gain.
  - Fixed gates hurt badly (−1.8 to −2.6 average points).
  - Gates conditioned on the full state beat stream-local ones slightly.
- **SimpleHC's own contribution is execution:**
  - With identity transitions, sublayer j's input is the group-entry state plus scalar-weighted earlier outputs.
  - Gates for K = 4 sublayers are computed from the group-entry state.
  - Traffic halves, from 18C to 9C, and the overhead falls from 7.0 % to 2.4 % at C = 6144.
  - Ablations: reusing gates across the group costs 1.5 points, dropping the corrections costs 0.045 in loss, and
    K = 8 degrades.
- **What transfers here:**
  - Learned per-token gates (M1).
  - The overhead of any wide residual is memory traffic, which dominates at small width and with cheap mixers: this
    regime.
  - Gates conditioned on a slightly stale state are nearly free.
  - No multi-stream work is planned.

**Other scratchpads**, the thread's list (unverified):
- Register tokens: ViT registers, the sequence-axis form (G3).
- mHC and MUDD: dense cross-layer connections.
- Multi-stream MM-DiTs: Qwen-Image's image and text streams.
- The thread's reading: this model does it densely along the channel axis.
- The "four lanes" of a Qwen preview architecture, which came up as a comparison, are not identified here. Find the
  source before comparing.
