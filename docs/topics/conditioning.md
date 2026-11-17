# The conditioning path: t, the pooled caption, the token states

Topic doc for PLAN Group J (J1–J4) and for F3, which it shares a session with. Opened 2026-11-17 from
`references/gemini-thread-adaln-and-architecture.md` (the conditioning questions) and
`references/handover-2026-11-12-residual-gating.md` §3.3 (the sink derivation). Measured numbers go to FACTS, closures to
`archive/`.

## As built (SPEC §3, §5; `src/nanodit/model.py` at `12c2028`)

- **t**: `sinusoid(t)` (256-d, fp32) → `time_mlp` (Linear 256→768, SiLU, Linear 768→768) → `c`.
- **Pooled caption**: `text_proj(rms_norm(pooled))` added to `c` while `pooled_ada` is on (the F3 switch).
- **adaLN-single**: `SiLU → Linear(768 → 6·768)`, one projection for the whole network (3.5M parameters), plus a
  learned `table[6, 768]` per block. It gives shift, scale and gate for the mixer and for the MLP of every block,
  conv and attention alike. It is per image, not per token. The gate rows start at zero, so every block is the
  identity at step 0 (SPEC §5).
- **Where t reaches**: every mixer input and every MLP input, through shift and scale, and every branch write,
  through the gate. Self-attention's sigmoid gate reads the modulated input, so t reaches it too. Cross attention
  takes no t.
- **Token states**: the 512-d states are `rms_norm`ed once per forward. Each attention block's cross attention
  projects them with its own `cross_kv`: GQA, no RoPE, its own q/k norms and branch norm, and a static per-channel
  `cross_gate` initialized to zero. The token states are **already static**: no layer updates them, and no MLP or
  query runs on them.
- **Null prompt**: a learned null embedding plus learned `null_tokens[2, 512]`, started from EmbeddingGemma 2's
  `<bos>` and `<eos>` states of the empty caption. A caption with no tokens maps to it, in training and at sampling.
  A real caption carries no `<bos>` or `<eos>` (SPEC §1), so no null key sits in its key set.

## J1. t at the input only, no adaLN

*Proposal* (from the thread): supply t once, at the input projection, and drop the adaLN.

What goes and what has to replace it:
- **Gates.** The adaLN gate rows provide the zero-init identity (SPEC §5). Without adaLN each block needs static
  gates: a learned vector per branch, zero-initialized, on the fallback, as `cross_gate` is now. The identity at init
  is kept.
- **Shift and scale.** These go. The pre-norm becomes a plain `rms_norm`.
- **Where t enters**, as candidates:
  - (a) Added to the meta channels at the patch embed: `time_mlp` to width − 128.
  - (b) A t token among the attention keys. This only fits with J2, or as an extra key in cross attention.
  - (c) (a), plus a learned, gated re-injection into the meta channels every k blocks (k 6–8 in the thread). It is
    the thread's "insurance" against t diluting over depth, which it says is borrowed from the modded-nanogpt
    speedrun (`topics/speedrun-ports.md`). It has the same shape as M3.
- **The pooled caption.** J1 with `pooled_ada = false` means the caption enters only by cross attention. The switch
  exists (F3).

Two arguments for it, from the thread:
- t carried in the stream, unlike adaLN's per-image vectors, would allow a per-token t (the thread's case: editing,
  per-token noise schedules). Per-token t as a training scheme ran and was dropped (`topics/priming.md`).
- The thread claims that a Gemma-based encoder's pooled embedding smears the bindings the token states keep, unlike
  CLIP's. EmbeddingGemma 2 is itself a contrastively trained embedding model (mean pool, REFERENCES), so the claim is
  doubtful as given; that the token states hold more than one vector does stand. That argues for `pooled_ada = false` (F3) whatever happens to t.

*Hypothesis* (the reason J1 is the preferred direction): the model always takes a noisy latent and
ships it, so it needs t only at the start, to know how carefully to touch the latent. The noise is overwritten fast;
deeper blocks clean, improve coherence and add the prompt, and need no t. Even small, the adaLN may not be wanted by
this architecture under x0 velocity.

The counter-*hypothesis*: the gate's dependence on t is part of what adaLN buys under x0 velocity. The correction a block should
write varies with t (`topics/prediction-target.md`, K1: the optimal skip runs from 2 to 0 over t). A static gate can
scale a branch by noise level only through the stream's content.

**Free read, before any arm.** Run the trained `ada` over a t grid from a checkpoint (the 1ep run left step 16000).
For each block, take the six vectors and measure how much each varies with t against how much it varies across
blocks. The hypothesis predicts strong t-dependence in the first blocks and little in the deep ones. If gate and scale barely move with t, J1 costs little. If they move strongly, (c) is the variant to arm.

Settle: an arm against the reference at the agreed arm length, read on `loss_by_t` at the extremes and on the eval
grid. Paired with F3 if both are taken.

**Does the model need t at all?** The thread says yes, and cites "What Time Is It?" on time-unconditional flow
matching (unverified, not read). One fact specific to this data: the Primal latents are unit-normal per image and
channel (SPEC §1), so the per-element variance of x_t is (1 − t)² + t². That value is the same at t and at 1 − t.
x_t's variance alone therefore cannot tell t from 1 − t; only its structure can, meaning the correlation across a
token's 128 values and across tokens. *Hypothesis*: dropping t entirely is harder here than the papers suggest. No
arm is proposed.

## J2. Joint attention: the token states as keys of self-attention

*Proposal* (from the thread): one attention per attention block instead of self-attention followed by cross
attention.
- **Mechanism.** Image queries attend over [image keys, text keys]. Each block projects text K/V from the static
  token states, as `cross_kv` does now. Text tokens get no queries and no MLP.
- **The text projection.** Either each block projects the token states with its own K/V (as `cross_kv` now), or one
  global Linear 512 → width translates them once and each block's `qkv` reads them like image tokens. Open. The
  thread's form is the second, with an RMSNorm after the Linear so the text scale matches the image tokens.
- **Positions.** RoPE on the image keys only; the text keys stay position-free, as now. The thread offers the
  alternative of 1-D positions on the text keys along a separate axis (Lumina 2's mRoPE, unverified); EmbeddingGemma's
  states already carry their order, so this is a question only if J2's text keys are found to lose it. Each source gets its own k
  norm.
- **What it removes per attention block:** `cross_q`, `cross_out`, the two cross q/k norms, `cross_norm` and
  `cross_gate`. The sigmoid gate and `attn_norm` then cover both sources.
- **What it changes in function:** one softmax over both sets, so image and text keys compete for each query's mass.
  The self-match is an always-available sink, so a background patch can ignore the caption (J3).
- **Caching:** the text K/V are the same at every sampling step, so they can be computed once per sample, as they
  can today.
- **The conv blocks** see image tokens only, which holds by construction, since the text tokens are not in the
  stream.
- **Cost:** 64 + L keys in one call, against 64 and L keys in two calls now. L is the caption length, paragraph-long
  in training (the eval set allows 512). Speed is probably similar; unmeasured.

Where it sits: between today's cross-attention DiT and a single-stream model such as Lumina 2 or Z-Image. Those
update the text tokens in every layer; J2 keeps them static, with no text stream (background below).

Settle: an arm against the reference, read on the samples and on a binding probe (D7). Caption loss could not judge
token conditioning at 3000 steps (`topics/block-design.md`, lessons), and at 6000 cross attention on or off was level
in loss (FACTS).

## J3. Can a patch abstain from the caption? A null key and value in cross attention

Derivation (residual-gating handover §3.3). Softmax with a sink logit b rearranges to
`o = σ(log Z − b) · softmax(s) V`, where `Z = Σ_j exp(s_j)`. So a sink is a scalar output gate per head and per query,
and it sees log Z: how well this query matched any key, a property of the whole key set. The Krea gate sees only the
query token's own state. The two gates are complementary.

Today the softmax runs over the caption's tokens alone, so every patch, background included, absorbs a convex mix of
the text. The static `cross_gate` cannot abstain per patch, and no null key sits among a real caption's keys (above).

*Proposal*: one learned key and value per KV head, always in the cross key set. The null key's logit plays the
derivation's b; the null value gives a defined output when nothing in the caption applies. Whether `null_tokens` stays
for dropped prompts or the null key/value replace it is a choice.

**Free read first.** From a checkpoint with cross attention (the 1ep run's 16000), check whether the cross maps already
concentrate on one token as a de facto sink: the first or last caption token, or punctuation. One forward pass over
eval rows. It shares its code with the attention-map sensor (`topics/attention-lock-in.md`, L1).

Under J2 the self-match is always available, but it still writes its value; a sink with a learned (or zero) value
lets a head write nothing. J3 therefore stands under J2 too, as a sink in the joint attention.

Settle: the prediction is that background patches put their mass on the null key and that binding improves before
loss does. That needs D7 to be visible; at 6000 steps loss alone will likely sit within one seed's resolution.

## J4. If the adaLN stays: drop its shift

Lumina-Image 2.0 is said to modulate with scale and gate only, with no shift (the thread's claim, unverified). Here
that would drop 2 of the 6 rows (`shift_a`, `shift_m`) and keep 4. The thread's reason, that velocity targets are zero-mean, does not obviously carry
over: what the shift acts on is the normed stream, meta channels included, and that is not zero-mean by
construction. Settle: by reading Lumina 2's code, or one arm. Moot if J1 is taken.

## J5. Which encoder states: one layer, or several fused

The token states are one point of EmbeddingGemma 2: the input of its `embedding_projection`, the last hidden state
(`precompute.encode_states`). Krea 2 is said to fuse several layers of its Qwen3-VL encoder (the thread's claim,
unverified; its technical report, `krea.ai/blog/krea-2-technical-report`, is not read). Intermediate layers keep more
lexical and positional detail; the last keeps what the embedding objective pooled for. Cost here: a re-encode of the
set (one forward per caption, the hook moved) and a wider or learned-mixed sequence; disk grows with the layers kept
unless they are mixed at encode time with fixed weights. *Hypothesis*, no arm. Settle: read Krea 2's report and code
(`github.com/krea-ai/krea-2`, `encoder.py`) first.

## J6. A text refiner before the image reads the tokens

Z-Image runs a few unmodulated "context refiner" blocks over the text tokens before the main stack (the thread's
claim, unverified); Lumina 2 and the single-stream models update the text in every layer. Here the token states are
static. A middle form: two or three small self-attention blocks over the token states, run once per sample (cached
over sampling steps, as the K/V are), adapting the frozen encoder's states to the image model. *Hypothesis*, no arm;
it composes with J2 and J5.

## Background: conditioning in other open DiTs (the thread's claims, unverified)

- **Shared adaLN:** FLUX.2 (shared across double-stream blocks and across single-stream blocks); Wan 2.x (one base
  modulation MLP, per-layer biases). This is the adaLN-single pattern this model already has, from PixArt.
- **Per-layer adaLN:** Qwen-Image (text tokens modulated at a fixed t = 0), HiDream-I1 (t plus pooled CLIP).
- **AdaLN-LoRA:** Anima / Cosmos-Predict2 (rank 256).
- **AdaRMSNorm-Zero, no shift:** Lumina-Image 2.0 (J4).
- **Selective adaLN:** Z-Image modulates the main blocks and the noise refiner, not the context refiner.
- **No adaLN:** i1 (arXiv 2606.11289), with t as tokens.
- **Krea 2:** "lightweight timestep modulation", zero-centred RMSNorm, sigmoid-gated GQA.
- **MOSS-Image:** shared modulation, joint attention.
- **The thread's "parameter tax" (15–25 % of parameters in modulation) does not apply here.** It refers to dense
  per-layer adaLN; adaLN-single here is one 4608×768 projection plus a 6×768 table per block.

Four backbone topologies, as the thread groups them:
1. **Cross-attention DiT** (PixArt-Σ, Anima): image self-attention, then cross attention to frozen text.
2. **Dual-stream MM-DiT** (SD3, Qwen-Image): separate weights per modality, joined only inside attention.
3. **Hybrid, dual then single stream** (FLUX.1, FLUX.2).
4. **Single stream** (Lumina 2, Z-Image, PRX, i1): one stack over [t, text, image].

This model is (1), with conv mixers in three of four blocks (SANA-like) and a channel-split stream
(`topics/residual-lanes.md`). J2 would move it toward (4) without a text stream.

Noted, no question: the thread's warnings about LoRA fine-tuning (modulation layers left frozen) and quantized
inference (modulation linears kept in bf16). Neither is planned.

To read before relying on any of the above: i1, "What Time Is It?", Lumina-Image 2.0 and Z-Image (code), and the
Wan 2.1 report.
