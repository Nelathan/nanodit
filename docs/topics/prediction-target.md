# What the network predicts: the skip and the target's geometry

Topic doc for PLAN Group K (K1, K2). Opened 2026-11-17 from `references/handover-2026-11-12-residual-gating.md` §3.2
(the skip) and `references/gemini-thread-adaln-and-architecture.md` (AnisoFlow). Measured numbers go to `log/`,
closures to `archive/`.

## As built (SPEC §3, §4)

- **Interpolation:** `x_t = (1 − t)·x + t·ε`, with `v = ε − x`.
- **Target:** `T = x_t − v`. The network output F is the stream's first 128 channels after the last block, and the
  loss is `‖F − T‖² = ‖v̂ − v‖²` with `v̂ = x_t − F`.
- **What the trunk must write:** the slice starts at x_t (concatenated patch embed, identity on the image channels),
  so the trunk writes `F − x_t = −v`. For unit-variance latents that correction has variance 2 at every t.

## K1. A t-dependent skip on the identity

**Derivation.** This is the minimum-variance linear skip `c(t)·x_t` for predicting T, with unit-variance latents
(handover §3.2, Monte-Carlo checked there). Re-derived here as `c = Cov(T, x_t) / Var(x_t)`:

  **c(t) = 2(1 − t)² / ((1 − t)² + t²)**,

the same function as the handover's `1 − (2t − 1)/((1 − t)² + t²)`. It is 2 at t = 0, exactly 1 at t = 0.5, and 0 at
t = 1. The variance the trunk still has to produce:

| t | 0 | 0.1 | 0.25 | 0.5 | 0.65 | 0.75 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|
| c(t) | 2.00 | 1.98 | 1.80 | 1.00 | 0.45 | 0.20 | 0.02 | 0 |
| residual with the identity (c = 1) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| residual with the optimal c(t) | 1.00 | 1.22 | 1.60 | 2.00 | 1.84 | 1.60 | 1.22 | 1.00 |

Reading:
- The identity is the optimal skip at the median t, 0.5. That is why the design works without a skip coefficient.
- At both ends it leaves up to twice the residual variance, symmetrically.
- These are linear-predictor residuals: a measure of how much work the skip leaves for the trunk, not irreducible
  error. Sandwich norm caps each branch's write, so a large correction has to be assembled from many writes.
  Residual amplitude is therefore a real capacity cost (*Hypothesis*).
- The handover first derived c for an x0 target, `(1 − t)/((1 − t)² + t²)`. That is wrong for this objective.

**Two forms:**
- (a) **Scale the identity write at the input:** the slice starts at `c(t)·x_t`. Parameter-free, but the image
  channels then no longer carry x_t, which changes what SPEC §3 says the stream carries. At t → 0 the slice amplitude
  doubles.
- (b) **The skip at the readout, EDM's `c_skip`:** the slice still starts at x_t, and `F = slice + (c(t) − 1)·x_t`.
  The stream keeps its meaning; only the correction the trunk has to write changes. Parameter-free.
  *Decided* for the arm (2026-11-21): (a). The output stays the raw slice and the slice is the prediction so far from
  the first block on, where under (b) every block reads a slice off by (c − 1)·x_t; the meta channels still embed x_t.
  *Measured*: a4's trunk rebuilds the rescale exactly, and at the ends it is most of its correction's energy (log
  2026-11-21-linear-skip-probe).
- In either form, a second arm clamps c to [0, 1] if the factor 2 at low t proves unstable.

*Prediction:* the loss falls in the outer t bins (t < 0.25 and t > 0.75), with little change at the middle.

Settle: one arm against the reference, read on `loss_by_t` (SPEC §8). First read the existing runs' loss by t; the
score grid holds it.

Interaction: the training t is logit-normal (SPEC §4), so the ends carry little mass. At σ 1, which the merged-set
references before `a1-ref-1ep` trained at, they carry more than at SPEC's σ 0.7 (archive 2026-11-18-timestep-sigma).

## K2. An anisotropic target (AnisoFlow)

Source: Zirvi, Chen et al., "Learning Anisotropic Prediction Targets for Flow Models", https://hanshengchen.com/anisoflow/.
Known here only through the thread's summary; **not read**.

As the thread gives it:
- **Target:** the network predicts `u_W = Wε − x`, with `W = B diag(w) Bᵀ` and `w_i ≥ 0`. B is a learned
  orthonormal basis (patch PCA or orthogonal rotations), and velocity is recovered by a closed form.
- **Limits:** all `w = 0` is x0 prediction; all `w = 1` is velocity prediction; `w ∈ {0, 1}` is AsymFlow: velocity on
  the top principal components, x0 on the tail.
- **Claimed cost:** W and B fold into the patch projection and the readout, so none at inference.
- **Claimed results:** on SD-VAE latents the learned w goes back to about 1, so no gain. In pixel space (JiT) FID
  goes from 1.90 to 1.55 over x0 prediction.

Relevance here:
- **This latent is nearer pixel space than SD-VAE.** A token is one 8×8 patch of the 2-channel Primal latent at 2×
  downsample: 128 spatially correlated values, against SD-VAE's 4–16 channels at 8×. *Hypothesis*: the 128-d token
  space is strongly anisotropic, dominated by its low spatial frequencies.
- **Our target is also isotropic, but t-dependent:** `T = (2 − t)·x − (1 − t)·ε`. How it maps onto their family is
  for the paper read.

Settle, cheapest first:
1. A probe: the eigenvalue spectrum of the 128-d token covariance over a sample of training latents. If the spectrum
   is nearly flat, K2 is moot here.
2. Read the paper.
3. Only then, an arm with a fixed PCA basis and a few w values before anything learned.

## Checked, not applicable

- **The thread's "1/t amplification" of x0 prediction** and its advice to clamp t near 0 at sampling. Here
  `v̂ = x_t − F` (SPEC §4): no division by t at training or at sampling, which is the point of Ostris's form
  (`references/ostris-x0-velocity-thread.md`). An output error passes one to one into v̂. No clamp is needed. The thread's patch-seam artifacts under FP8 inference
  rest on the same division and on quantized inference, which is not planned.
- **The thread's advice to decay the weight rows that write the slice, against "1/t overshoot".** It falls with the
  point above; the question of per-row bounds stays in G8.
