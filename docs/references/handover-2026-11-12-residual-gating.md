# Handover of 2026-11-12: residual gating, the identity skip, optimizer rank, and what to run next

Kept verbatim as received on 2026-11-17: a handover written by another agent session on 2026-11-12 from a design conversation and a read-only audit of `main` at `30264af`. Much of its section 2 is stale against the code of 2026-11-17; PLAN and the topics carry what still holds (`PLAN.md`, the 2026-11-17 reorganisation).

---

# tiny-dit handover: residual gating, the identity skip, optimizer rank, and what to run next

For: the Fable agent taking over tiny-dit with Daniel.
Written: 2026-11-12, from a design conversation on 2026-11-09 and a read-only repo audit on 2026-11-12 (`main` at `30264af`, dirty tree).

## 0. How to use this document

**Authority order:** the code, then `docs/SPEC.md`, `docs/FACTS.md` and `docs/HANDOVER.md`, then this document, then anything recalled from chat. The design conversation ran partly on Daniel's verbal description of the model. The audit showed that several of those descriptions were stale or loose. Section 2 lists every mismatch. **Settle those with Daniel before you build on any of them.** He prefers one sharp question to a wrong assumption.

What follows was all **verified in code**, except where a line is marked *unverified* or *hypothesis*.

**Section map:**
1. The model and setup as they actually are.
2. Mismatches between the conversation and the code.
3. The mechanisms behind the design.
4. The experiment queue, with mechanism, prediction, test and cost for each item.
5. Hyper-Connections background, distilled.
6. Numbers and the noise floor.
7. Working with Daniel.

---

## 1. The model as it is

**Repo state**
- **Location:** `/home/djg/Projects/tiny-dit`, package `nanodit`, entry point `uv run python scripts/train.py CONFIG.toml`.
- **Config:** a frozen dataclass at `scripts/train.py:33`. All 17 configs are explicit, and there is no default config.
- **Git risk:** no git remote is configured. The working tree has 33 modified files, `src/nanodit/data.py` among them. UsuiTrack is an editable path dependency on `../usuitrack-zero-init`, branch `zero-init` at `bb21798`, and that branch **has no upstream**. Both are single-copy work. Raise this with Daniel; don't push or commit on your own.

**Trunk: 768×12, 122.8M parameters, no encoders inside the model**
- **Heads:** head dim 64, 12 query heads, 3 KV heads, SwiGLU MLP at 4×.
- **Blocks:** block `i` is conv when `i % 4 != 3`. That makes 9 conv blocks and 3 attention blocks (3, 7, 11).
- **Sublayers:** mixer, then optional cross-attention (attention blocks only, off by default), then MLP.

**Residual stream, 768 channels**
- **Layout:** channels 0–127 are the **image slice**: one 8×8 patch of the 2-channel Primal latent, so 128 values per token. Channels 128–767 are **meta**.
- **Patch embed:** writes the raw patch into the slice **as identity, unscaled**, and writes `Linear(128 → 640)` into meta.
- **Head:** reads the slice directly. There is no output linear and no final norm.
- **Writers:**

| Writer | Channels it writes |
|---|---|
| plain conv (`gdn=False`) | meta only (zero-padded slice, `model.py:97,137`) |
| GDN conv (`gdn=True`, **the default**) | **all** channels |
| self-attention, cross-attention, MLP | all channels |

**Mixers**
- **GDN conv** (default): `σ(write)⊙silu(dwconv3×3(value)) − σ(forget)⊙state`. The kernel is static depthwise 3×3. The write and forget gates are per-token and data-dependent.
- **Plain conv:** a static depthwise 3×3 on the normed input.
- **Content-dependent window mixer:** a 3×3 softmax over q·k. It lives only on branch `gdn-conv` and lost by 0.044 at 256 px.

**Gates and conditioning**
- **Self-attention gate:** `x += gate_a · attn_norm(out(attn ⊙ σ(Wy)))`. This is the Krea/Qwen-style elementwise gate, full width, **no bias**, computed from the modulated normed input. Init is `normal(0, in^-0.5)`, so σ starts near 0.5.
- **Cross-attention gate:** a static per-channel zero-init `cross_gate`. It is **not data-dependent**, so it can't abstain per patch.
- **Sink:** none anywhere, and no learned null key/value.
- **adaLN-single:** one shared `SiLU → Linear(768, 6·768)` plus a per-block `table[6,768]`. It produces shift, scale and gate for both the mixer and the MLP. The gate rows are zero-init. The conditioning is `c = time_mlp(sinusoid(t)) + text_proj(rms_norm(pooled))`.

**Norms:** sandwich. The pre-norm is a gainless `F.rms_norm` with adaLN modulation. The post-branch norm is an `RMSNorm` with a learned gain. QK-norm is applied per head. Partial RoPE (quarter of the frequencies per axis) is used in attention blocks only.

**Text**
- **Token states:** EmbeddingGemma 2's 512-d token states, taken after the final norm and before its projection to 768. The task prefix is dropped per SPEC §1, but this is *unverified in `precompute.py`*.
- **Pooled embedding:** the stored pooled embeddings **still carry the document prefix**. PLAN B7 says they retrieve images worse than no prefix. That's an open data bug feeding adaLN.
- **Dropped prompts:** dropout is 10%. A dropped prompt becomes the learned `null_embedding` (adaLN) plus learned `null_tokens[2,512]` (a space and `<eos>`) for cross-attention.

**Objective**
- **Interpolation:** `x_t = (1−t)·x + t·ε`, with `v = ε − x`.
- **Prediction:** the network output F is the image slice. The target is `T = x_t − v`, so `v̂ = x_t − F`. The loss is unweighted `‖F − T‖² = ‖v̂ − v‖²`.
- **What the trunk writes:** the slice starts at `x_t`, so the trunk must write **−v**. Under unit-variance latents that correction has variance 2 at every t.
- **t-sampling:** `sigmoid(σ·randn + ln shift)`. Every config uses `shift = 1.0` and `t_sigma = 1.0`. The code default of 0.7 is unused. This is a logit-normal with median t = 0.5.
- **Eval:** the scored loss uses 10 quantile midpoints of the same distribution, with fixed noise.

**Optimizer: UsuiTrack**
- **Rank:** `rank = round(0.1·√(mn))`, capped at half the shorter side.
- **Frame:** moved by an Oja tangent with geodesic turns of 0.02 rad, retracted by one Polar-Express step. The initial frame is random.
- **Update:** `ΔW = −lr · ‖W‖_F · lift(polar(moment))`, where `step_rule="trust"` and the polar step is Aurora row balance followed by 5-step quintic Newton–Schulz. `pin_norm` then rescales W back to its first-step Frobenius norm. This fixed-radius sphere is what Daniel calls the "hmuon hyperball" upgrade.
- **Molegrad:** replaces each tracked Linear's backward so that it returns the frame-projected gradient `G·Q`, an aim and a 16-probe sketch, without materializing G.
- **Stacked eigh:** the tangent-Gram eigh is **still stacked into one call** on `zero-init`. Branch `lab` chunks it by rank (`lab:frame.py:104`), but tiny-dit doesn't use `lab`.
- **Precision:** stochastic rounding on all bf16 weights and moments. The model is pure bf16 with no fp32 master weights. The frame is round-to-nearest on purpose.
- **Routing:** 2-D parameters go to UsuiTrack, except those named `table` or `null`. Everything else (norm gains, `cross_gate`, biases, the adaLN tables, the null embeddings, the 4-D depthwise kernels) goes to StochasticAdamW.
- **LR:** `usui_lr = 2e-3` with 376 warmup steps, then constant. No weight decay and no clipping.

**Setup**
- **Batch:** 256 = 4 × 64 by gradient accumulation. `block.compile()` per block. No CUDA graphs.
- **Data:** 128 px only, the first 16 of 47 text-to-image-2M shards, one shuffled Parquet.
- **Eval:** Lunara (2k synthetic art images) for loss and R² by t. Sampling every 1000 steps uses Euler, 30 steps, CFG 4.0, and 16 fixed prompts. Those include "a bear building a log cabin in the snow covered mountains".
- **Missing:** there is no in-distribution held-out split and no binding/role-swap probe set.

---

## 2. Mismatches between the conversation and the code (reconcile first)

| # | What the conversation assumed | What the code says | Why it matters |
|---|---|---|---|
| M1 | "The conv writes only meta, else it smudges; the MLP translates meta to image" | True for **plain** conv only. The **default GDN conv writes all 768 channels**, the slice included | The smudging rationale is the main structural principle of the design, and the default violates it. Either GDN's per-token write gate makes all-channel writes safe (a testable claim; GDN beat plain conv, see §6), or the mask was lost when GDN went in. **Ask Daniel which.** |
| M2 | "Trains well at 4e-3" | Configs use 2e-3. FACTS records a d768×16 sweep: **4e-3 scored 1.4868, 2e-3 scored 1.4933 (lower is better)** | 4e-3 won by 0.0065, about 2× the single-seed noise, yet the baselines stayed at 2e-3. Ask whether that was deliberate, or a winner that never got promoted |
| M3 | "Activations are checkpointed" | No activation checkpointing found in `model.py` or `train.py`. Usuitrack/molegrad were not searched | Every memory-bound argument (microbatch 128 OOM, 2048-width feasibility) depends on this. Verify before sizing anything |
| M4 | "Removed stacked eigh and polar; less memory, but slower" | tiny-dit pins `zero-init`, which **still stacks eigh**. Polar runs per matrix. `lab` chunks eigh | The "slower" observation may come from `lab` or local edits, not from the code tiny-dit runs. Find out which code produced the observation |
| M5 | "Stable rank grew to 20% at rank fraction 0.1; gradient rank 3–5%; LFM3 / EmbeddingGemma 2 at 70%" | No rank metric is logged in training. The only recorded spectra are in `docs/topics/weight-spectra.md`: **effective rank** (exp-entropy of σ²) of 0.08–0.19 of the short side, on `128_d1024x16_swpin_5ep`, a **pre-conv model** | The rank hypothesis in §4 (E8) rests on numbers that aren't in the repo. The 20% may be a different metric or model, or come from a scratch probe. Recompute with `scripts/probe_frame.py` on current checkpoints |
| M6 | "t sampled with sigma 0.7 around 0.5" | Median 0.5 is right. Sigma is **1.0** in every config | Wider t mass than assumed, so more weight at the ends, where the identity skip costs most (E1) |
| M7 | "Loss is worst at t 0.6–0.7; only R² is worst at high t" | Not checked. The data exists in W&B as eval loss and R² by t | Pull it before E1. The c_skip analysis was first argued under an x0-target assumption that turned out to be wrong (§3.2) |
| M8 | "Cross-attention was added because adaLN wasn't enough" | `cross=False` by default. The conv + cross, no-pooled-adaLN arm scored 1.4771, equal to the GDN trunk | Cross-attention isn't in the reference trunk, and its value is unproven at 6k steps. The bear-cabin binding failure motivated it, and nothing measures binding yet |
| M9 | "Prefix and EOS dropped deliberately" | Token states: prefix dropped per SPEC (*unverified*), BOS unknown. `null_tokens` contains `<eos>`. Pooled embeddings **carry the prefix** (PLAN B7) | It affects whether cross-attention needs a sink (E2), and adaLN conditioning quality |

The audit misread one premise: it reported that t is not centred at 0.5. In logit space it is centred at 0, so its median is t = 0.5. Only the sigma premise was wrong.

---

## 3. The mechanisms behind the design

### 3.1 The image/meta split is an implicit two-stream hyper-connection
Daniel never intended n generic streams. The 128-channel slice plus 640 meta channels, with a role-based write mask, is a two-stream residual with **fixed, role-encoding write gates** and an identity transition. The design gives a precise reading of that prior:
- **Spatial mixing happens in a latent workspace.** A static conv is the same linear filter at every position. Whatever it writes directly into the prediction is a blur, hence "smudging".
- **Per-token translation is allowed to touch the prediction.** The MLP is per-token, so its writes to the slice can be sharp.

The SimpleHC study (§5) found that fixed gates hurt in LMs (−1.8 to −2.6 avg points). Daniel's masks aren't arbitrary, though. They encode a function, a stronger prior than anything HC learns. A parked idea, which Daniel has not adopted: learned per-token soft write gates onto the slice, initialized at the current masks. M1 is effectively a live, uncontrolled version of that question.

### 3.2 The identity skip, recomputed for the actual target
The slice starts at x_t and the target is `T = x_t − v`. The minimum-variance linear skip `c(t)·x_t` for predicting T, with unit-variance latents, is

  **c(t) = 1 − (2t − 1) / ((1 − t)² + t²)**

That's **2 at t = 0, exactly 1 at t = 0.5, 0 at t = 1**. I verified it by Monte Carlo (2M samples, three-decimal agreement). The residual variance the trunk must still produce:

| t | 0 | 0.1 | 0.25 | 0.5 | 0.65 | 0.75 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|
| c(t) | 2.00 | 1.98 | 1.80 | 1.00 | 0.45 | 0.20 | 0.02 | 0 |
| residual with identity (c = 1) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| residual with optimal c(t) | 1.00 | 1.22 | 1.60 | 2.00 | 1.84 | 1.60 | 1.22 | 1.00 |

Reading the table:
- **The identity is the optimal skip at t = 0.5,** which is the median of the t distribution. That's why the design works without a skip coefficient.
- **The identity costs up to 2× residual variance at both ends,** symmetrically. With sigma 1.0 there is real mass out there.
- **c(t) departs fast past the middle.** At t = 0.65 the optimal coefficient is already 0.45, though the residual saving there is small (2.00 → 1.84).

These are linear-predictor residuals, a statement about how much work the skip leaves for the trunk. They are not irreducible error. Sandwich norm caps each branch output's RMS, so a large-amplitude correction has to be assembled from many coordinated writes. That makes residual amplitude a real capacity cost.

During the conversation I first derived c_skip assuming an x0 target (c = (1−t)/((1−t)² + t²)). That formula is **wrong for this code**. Use the one above.

### 3.3 Sink gates and query gates condition on different things
Rearranging softmax with a sink logit b gives `o = σ(log Z − b) · softmax(s)V`, where Z = Σ exp(s_j). So a sink is a scalar, per-head, per-query output gate. The two gate types differ in what they can see:
- **The sink gate sees log Z:** how well this query matched *any* key, which is a property of the whole key set.
- **The Krea/Qwen gate sees only the query token's own state:** it is elementwise, so it generalizes the sink in *dimensions*, but it can't see match quality.

They are complementary. What follows for this model:
- **Self-attention:** a matching key always exists, the token itself. Abstaining means "this layer shouldn't write here", a judgment from query state, so the Krea gate fits. It's already in the trunk.
- **Cross-attention:** abstaining means "nothing in this prompt concerns this patch", which depends on the keys. Today the softmax over text tokens alone forces every patch, background included, to absorb a convex mix of text. The static `cross_gate` can't abstain per patch. **A learned null key/value per head is the right fix:** the null key's logit is the derivation's b, and the null value gives the unconditional branch a defined output.
- **Conv:** no softmax, so no sink exists. Abstention has to be an output gate. The GDN conv's `σ(write)` already is a data-dependent per-token output gate on the conv path, so the GDN win over plain conv may be mostly a **gate effect** rather than a state effect (E3).

### 3.4 Optimizer rank dynamics
Agreed with Daniel; the figures are pending M5:
- **The gradient lies in the activation space.** After hmuon's norm pin decays the initial noise floor, W can only gain directions spanned by δ ⊗ x.
- **Weight rank above the per-step update rank** means W is integrating a rotating subspace over time. That's normal.
- **Stochastic rounding is the likely tail-keeper.** Small tail-direction updates fall below half a bf16 ulp. Round-to-nearest erases them deterministically; SR keeps them in expectation. Pin-norm rescaling is uniform, so it shrinks tails without selecting against them.

---

## 4. Experiment queue

Each item lists mechanism, prediction, test, cost and status. All are **not run** unless stated. Order is by value per hour. Every architecture change is a design change, so Daniel approves it first. These are proposals with committed predictions, not decisions.

**E0. Reconcile §2 (M1–M9).** Free. Blocks E1, E2, E6 and E8.

**E1. c(t)-scaled identity skip**
- **Mechanism:** §3.2.
- **Change:** multiply the patch embed's identity write into the slice by c(t) from §3.2. No parameters, and purity is preserved: still the identity, modulated by t.
- **Prediction:** loss falls in the outer t bins (t < 0.25 and t > 0.75). Little change at the middle.
- **Test:** 6k steps, 768×12 GDN trunk against `gdn_6k` (1.4771), compared by eval loss per t bin. Pull the existing per-t curves first (M7).
- **Caveat:** c(0) = 2 doubles the slice amplitude at low t. Check that sandwich norm and the slice readout tolerate it. If not, clamp c to [0, 1] as a second arm.

**E2. Learned null key/value in cross-attention**
- **Mechanism:** §3.3.
- **Change:** one learned key and one learned value per head, always present in the cross-attention key set. Keep `null_tokens` for dropped prompts, or replace them with the null key/value alone. That's a choice for Daniel.
- **Predictions:**
  - Before the change, the cross-attention maps already concentrate on some token as a de facto sink. The `<eos>` inside null captions, or the first text token, are candidates. Check from one forward pass on a cross-enabled checkpoint, at no cost.
  - After the change, background patches attend to the null token, and binding probes improve before loss does.
- **Test:** needs the binding probes (E5) to be visible at all. At 6k steps, loss alone will likely sit within noise.

**E3. Decompose the GDN win: gate vs state**
- **Mechanism:** §3.3. GDN adds a data-dependent write gate *and* a forget-state term.
- **Arms:**
  - plain conv + `σ(Wx) ⊙ conv(x)` output gate, writing meta only;
  - plain conv + the same gate, writing all channels;
  - the GDN reference.
- **Prediction:** the gated plain conv recovers most of GDN's 0.0017 (128 px) / 0.004 (256 px) lead, at close to plain-conv speed (1395 against 1169 img/s on `gdn-conv`). The meta-only variant won't lose to the all-channel one, which also settles M1 empirically.
- **Smudging:** no one has looked at samples of these arms (HANDOVER). Looking is part of the test.

**E4. Batch 512**
- **Mechanism:** FACTS records a ~140 ms per-step cost outside forward/backward. At 745 img/s and batch 256 a step takes 344 ms, so **about 41% of the step is fixed per-step overhead**, mostly the optimizer.
- **Prediction:** if that cost is per optimizer step, not per microbatch (verify), batch 512 gives a step of about 2 × 204 + 140 = 548 ms, so **about 935 img/s (+25%)**. The empirical rule so far: batch 128 drew with 256 per step, and 256 won on time. If loss per sample holds, 512 wins on time again.
- **Test:** an equal-wall-clock comparison.

**E5. Measurement infrastructure (cheap, high leverage)**
- (a) **In-distribution holdout:** about 2k held-out text-to-image-2M rows. Lunara is out-of-distribution synthetic art, so without a holdout a capacity limit can't be told from a style gap.
- (b) **Role-swap binding probes,** scored per checkpoint: "a bear building a cabin" / "a cabin shaped like a bear", "a red cube on a blue sphere" / the reverse, and so on. Binding is the likely small-scale ceiling, and loss doesn't measure it.
- (c) **Stable rank (Σs²/s₁²) logging** per matrix in training, or `probe_frame.py` run at every checkpoint.
- (d) **A log-log fit of held-out loss** against images seen, with an offset term, per config. On Daniel's observation that **under UsuiTrack, models of different sizes train in parallel with a constant loss gap per size**, a single equal-wall-clock comparison decides size questions, with no crossing to wait for.

**E6. Width vs depth at equal parameter count and equal wall time**
- **Arms:** 768×32 (about 302M, 640 meta channels) against 1024×16 (about 268M, 896 meta channels). Optional third arm: 2048×8 (about 540M).
- **Mechanisms:**
  - The optimizer's per-step call count scales with the number of matrices, which means depth.
  - With chunked stacking of same-shape matrices (E7), that cost becomes nearly depth-independent. Every layer has identical shapes.
  - With a 3:1 conv mixer, depth also buys spatial receptive field, which grows linearly with the number of conv layers.
- **Predictions:**
  - Once E7 lands, 768×32 beats 1024×16 on loss and on binding probes: 640 meta channels is ample workspace for a 128-d prediction, and the doubled conv reach is worth more than 256 extra channels. This puts Daniel's "32 layers is the sweet spot" intuition on a mechanism.
  - 2048×8 wins on loss per wall-time and loses on binding, since it has only 2 attention layers and almost no conv reach. If so, the fix is attention in every other block at 2048×8.
- **Memory:** sizing depends on M3.

**E7. Chunked stacked eigh and polar sized to the post-backward memory window**
- **Mechanism:** the optimizer step runs after backward, when activations are freed. Its free memory budget is the forward/backward peak minus parameters and gradients.
- **Change:** stack same-shape matrices in chunks of k, with k sized to fill that window. Also batch Newton–Schulz polar, which currently runs per matrix.
- **Prediction:** most of the ~140 ms fixed cost disappears, which feeds E4 and E6.
- **Where:** `lab` already chunks eigh by rank. Porting it to what tiny-dit runs means moving off `zero-init` or merging, and that is Daniel's call (M4).

**E8. Rank: data-limited or optimizer-limited?**
- **Hypothesis:** the ceiling of weight rank is set by the effective rank of each layer's input-activation covariance, not by UsuiTrack.
- **Test:** per layer, compare the stable rank of the input activations with the stable rank of W.
  - W tracks just below the activation rank: the data and training time are the limit, and the optimizer is fine (expected).
  - Activation rank sits well above W: the subspace rotates too slowly. Raise the rank fraction or the geodesic step.
- **Companion test, stochastic rounding off:** switch SR to round-to-nearest for a stretch and watch the tail spectrum. If the tails stall, SR is what keeps the edges alive, and full-rank noise isn't needed.
- **Depends on:** M5.

**E9. REPA: run only if precomputed**
- **Status:** ran (`okmpjpo6`). 1.4748 against 1.4771, a margin of 0.0023, which is **inside single-seed noise** (~0.003). At 256 px: 1.3715 against 1.3728.
- **Cost:** the 3000 aligned steps took 2.4 h against about 17 min unaligned, roughly 8.5× slower. The live pipeline (decode, ×3 bicubic upscale, Gemma vision tower) is the whole cost.
- **Only viable form:** PCA-fit once (top ~64 components per patch), store per chunk, align only early (`repa_steps` exists; HASTE 2025 shows REPA's benefit is front-loaded). Not worth building unless a multi-seed check shows the gain is real.

**E10. Fix the pooled-prefix bug (PLAN B7).** Re-encode the pooled embeddings without the document prefix. It's a data fix on the adaLN path. The cost is a re-precompute, which Daniel describes as days chunk by chunk; schedule it alongside any other re-precompute.

**Closed or not worth running:**
- **Masked priming:** −0.009. **Two-stage priming:** −0.013. **Per-token t spread:** removed. Daniel's reading: "no priming; maybe skip is all you need".
- **All-attention trunk:** 1.4884, clearly worse.
- **Window mixer:** −0.044 at 256 px.

---

## 5. Hyper-Connections background (SimpleHC, Zhang et al., arXiv 2611.05984)

Read the paper in full; the repo has no code yet.

- **HC / mHC:**
  - n residual streams, read gate h_pre, write gate h_post, mixing matrix H_res: `X' = H_res X + h_post oᵀ`.
  - mHC constrains H_res to be doubly stochastic via ~20 Sinkhorn iterations. Its cost is wide-state memory traffic: 18C per token per sublayer even when fused, which comes to 7–27% extra training time for under 1% extra FLOPs.
  - In their 230B stress test at 10× LR, mHC spiked to gradient norm 1.9e6. Sinkhorn failed to converge (row sums 2, 1, 0.5, 0.5), and the dH_res gradient path dominated.
- **identityHC (Guo et al.):** H_res = I. It matches mHC: 55.4 against 55.0 at 15B, 62.8 against 62.9 at 30B.
- **Their Table 2, the useful part:** learned read/write gating and unconstrained mixing are two independently effective groups, each about equal to mHC, and best together (56.2). Sinkhorn wipes out mixing's gain. Fixed gates hurt badly. Full-state gate conditioning beats stream-local conditioning slightly.
- **SimpleHC's contribution is execution:**
  - With identity transitions, updates are additive, so sublayer j's input is the base read of the group-entry state plus scalar-weighted earlier outputs, with weights `h_preⱼ·h_postₖ`.
  - Gates for K = 4 sublayers are computed from the group-entry state, with each sublayer keeping its own parameters. Reads are batched and written back once per group. That halves traffic, 18C → 9C, and takes overhead from 7.0% to 2.4% at C = 6144.
  - **Ablations:** reusing gates across the group costs −1.5 points; dropping the corrections costs loss +0.045; K = 8 starts to degrade.
- **Relevance here:** the trunk already is an identity-transition two-stream residual with fixed gates (§3.1). The transferable lessons:
  - learned per-token gates are where HC's quality comes from;
  - the overhead of any wide residual is memory traffic, which dominates at small width and with cheap mixers, exactly this regime;
  - per-token gate conditioning on a slightly stale state is nearly free.

  No multi-stream work is planned.

---

## 6. Numbers and the noise floor

All runs are 6k steps (1.536M images), 768×12, 128 px, batch 256, seed 41, LR 2e-3, `t_sigma` 1.0, eval = Lunara caption loss. **One seed each: treat |Δ| < 0.003 as noise.**

| Config | Eval loss |
|---|---|
| REPA (GDN, aligned 3000 steps) | 1.4748 |
| `arch_window_6k` (branch `gdn-conv`) | 1.4741 |
| GDN trunk `gdn_6k` (reference) | 1.4771 |
| conv + cross, no pooled adaLN | 1.4771 |
| `arch_windowall_6k` | 1.4787 |
| plain conv `arch_6k` | 1.4788 |
| all attention | 1.4884 |

**Throughput:** 745 img/s for the GDN trunk without REPA, 89 img/s with REPA on. On branch `gdn-conv`: 1395 plain conv, 1169 GDN.

**Fixed per-step cost:** about 140 ms outside forward/backward (FACTS).

**Not comparable:** the "conv-meta 6k reference" values of 1.4655–1.4680 were on different data and eval.

**Daniel's longer-run observations (not in the repo):**
- objects (not just mood) emerged after about 10M images;
- 512 width was too small, 1024 gave a slight lift;
- 768×16 and 1024×16 five-epoch runs exist;
- microbatch 128 OOMs.

---

## 7. Working with Daniel

`AGENTS.md` is the operating contract; read it. What this session added:
- **Commit to hypotheses.** State the mechanism, a falsifiable prediction and the test. Hedging cost real friction this session. Label speculation once, then commit.
- **Every design change goes to him.** An observation isn't a request to fix. Inside an agreed route, work without "continue?" stops.
- **For choices,** give the deciding facts first, then a recommendation. One question per decision, in prose.
- **Fail early:** check the cheap thing that would kill an idea before building it. E2's sink check and E4's per-step-cost check are each a single forward pass or log read.
- **He reads every word and thinks in mechanisms.** When he corrects you, he's usually right about his own system. Update cleanly and move on.
- **He and Claude are a team on a hard problem.** Bring curiosity and some joy; he has a newborn at home and runs on focused bursts.
