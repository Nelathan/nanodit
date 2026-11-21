# What the network predicts: the target, its geometry and the skip

Topic doc for PLAN Group K (K1, K2). It explains the objective from the ground up: what the network is asked for,
why the x0-velocity target behaves as it does, where the noisy input enters the prediction, and what the model
measurably does with it. Sources: Ostris's thread (`references/ostris-x0-velocity-thread.md`, the figure
`references/x0-velocity-targets.jpg`), the residual-gating handover §3.2 (the skip), EDM (Karras et al. 2022,
`c_skip` and `c_out`), the Gemini thread (AnisoFlow). Measured numbers live in `log/`, closures in `archive/`.

## The objective as built (SPEC §4)

- **Noising.** A clean latent x and a noise draw ε mix on a straight line: `x_t = (1 − t)·x + t·ε`, t = 0 clean,
  t = 1 pure noise. The velocity along the line is `v = ε − x`, the same at every t.
- **Target.** `T = x_t − v = x + (1 − t)(x − ε)`. The network's output F is the stream's first 128 channels after the
  last block (the slice), read out raw.
- **Loss.** `‖F − T‖²`, which is identically `‖v̂ − v‖²` with `v̂ = x_t − F`: the plain flow-matching velocity loss,
  unweighted. The sampler gets `v̂ = x_t − F` and runs unchanged.
- **The skip.** The slice starts at x_t: the patch embed concatenates the raw patch with a projection of it. With
  the gates at zero the model outputs `F = x_t`, so `v̂ = 0`, and each block adds a correction to the slice.

## Three targets, one line

The figure `references/x0-velocity-targets.jpg` shows one image under the three targets, at five noise levels, on
one colour scale. What changes between them is the correction the network must make, its target less its input,
`target − x_t`. For unit-variance latents:

| Target | Output at t = 1 | Output at t = 0 | Correction | Its variance, t = 1 → 0.5 → 0 |
|---|---|---|---|---|
| velocity (flow matching) | ε − x | ε − x | `(1 − t)·ε − (2 − t)·x` | 1 → 2.5 → 5 |
| x0 | x | x | `t·(x − ε)` | 2 → 0.5 → 0 |
| x0 velocity | x | 2x − ε | `−v` | 2 → 2 → 2 |

- **Velocity:** the output is a noise field. Its correction grows toward the clean end, where the network has to
  reproduce its input and invert it.
- **x0:** the output is an image. Its correction vanishes at the clean end, and its loss, `t²` times the velocity
  loss, weighs the fine-detail steps least.
- **x0 velocity:** the output is image-shaped at every t, and the correction is `−v`, of one size at every t, with
  the velocity loss unchanged. At t = 1 it is x0 prediction; below that it is x0 carried on along the line, so at
  t = 0 it asks for `2x − ε`: the clean image with the noise subtracted, the speckle in the figure's last column.

## The geometry: everything happens on one line

In high dimensions a noise draw is almost exactly orthogonal to any fixed image (cosine about `1/√D`), and both have
about the same length. So the whole objective lives in the plane of x and ε, and can be drawn with x = (1, 0),
ε = (0, 1), the origin O being the zero latent, the mean (taken here to decode to grey; not checked by a decode):

```
  ε ●  (0, 1)
    │ ╲
    │   ╲
    │     ●     x_t = (1 − t, t): from x at t = 0 to ε at t = 1
    │       ╲
    │         ╲
  O ●───────────● x  (1, 0)
                  ╲
                    ╲
                      ●     T = x_t − v = (2 − t, t − 1): from x at t = 1 to 2x − ε at t = 0
                        ╲
                          ╲
                            ● 2x − ε  (2, −1)
```

- **The line.** ε, x and 2x − ε lie on one line, `x + y = 1`. The noisy input x_t walks its segment from x to ε.
  The velocity `v = ε − x = (−1, 1)` is the line's direction, of squared length 2.
- **The target is the input moved back one chord.** `T = x_t − v` sits exactly one chord length, |v|, from x_t, toward x
  and past it: where the path would be at time `t − 1`. That fixed distance is the constant correction.
- **The best linear skip is T's shadow on x_t.** A prediction `c·x_t` can only move along the line through O and
  x_t. The best c is the projection of T onto that ray:

  **c(t) = 2(1 − t)² / ((1 − t)² + t²) = 1 − (2t − 1) / ((1 − t)² + t²)**

  (`c = Cov(T, x_t) / Var(x_t)`; the handover's form is the second). What is left, the part of T perpendicular to
  x_t, has length `1/|x_t|`, with `|x_t|² = (1 − t)² + t²`.
- **Why 0 to 2, not 0 to 1.** At t = 1, x_t = ε is orthogonal to T = x: the shadow is zero, c = 0. At t = 0,
  x_t = x and T = 2x − ε: the shadow is 2x, c = 2. The 2 is the target's overshoot past x. For the plain x0 target
  the shadow is `c = (1 − t)/((1 − t)² + t²)`, which runs from 1 to 0.
- **Why the linear error peaks at t = 0.5.** The line comes closest to O at its midpoint, `|x_t| = 1/√2`, and there
  x_t is perpendicular to v: the input says nothing linear about which way along the line to go, and all of v is
  left, `1/|x_t|² = 2`. Toward either end x_t turns toward the line, and part of v shows in it. With a latent
  variance s² (x at (s, 0) in the plane) the closest point is at `t = s²/(1 + s²)`; a4's latents (mean square
  1.114) put it at t 0.53.
- **Grey.** As t → 1, c → 0: the best linear guess is O, the mean latent, grey. Cancelling the noise without
  replacing it with structure leaves grey. *Hypothesis*: grey patches in samples are regions where the high-t
  predictions stayed near the conditional mean, which the low-t steps, mostly scaling what is there (c near 2),
  cannot fill.

This picture is exact for Gaussian latents. Real latents are not Gaussian, so part of the perpendicular remainder is
predictable from the image's structure and the caption: that part is what the trunk learns. The linear numbers
below measure how much work the skip leaves, not irreducible error.

| t | 0 | 0.1 | 0.25 | 0.5 | 0.65 | 0.75 | 0.9 | 1 |
|---|---|---|---|---|---|---|---|---|
| c(t), the linear skip | 2.00 | 1.98 | 1.80 | 1.00 | 0.45 | 0.20 | 0.02 | 0 |
| remainder variance with the identity (c = 1) | 2 | 2 | 2 | 2 | 2 | 2 | 2 | 2 |
| remainder variance with c(t), `1/|x_t|²` | 1.00 | 1.22 | 1.60 | 2.00 | 1.84 | 1.60 | 1.22 | 1.00 |
| c_out(t), its RMS `1/|x_t|` | 1.00 | 1.10 | 1.26 | 1.41 | 1.36 | 1.26 | 1.10 | 1.00 |

The identity is the best skip exactly at the line's midpoint, which is why the design works without a skip
coefficient; toward both ends it leaves up to twice the variance the best linear skip does.

## What a4 does with it

*Measured* on the annealed one-pass reference (log 2026-11-21-linear-skip-probe, `scripts/probe_linear_skip.py`):

- **The trunk learns the rescale exactly.** The correction it writes, regressed on x_t, has the slope `c(t) − 1` to
  within 0.007 at every t.
- **At the ends the rescale is most of what it writes:** 0.78 of its correction's energy at t 0.16, 0.69 at t 0.84;
  none at the midpoint. *Hypothesis*: this is a capacity cost. The branch norms bound each write, so a correction
  is assembled from many writes, and the rescale takes most of them where it is large.
- **The loss peaks at t 0.66 to 0.74, not at the midpoint.** The best linear predictor's loss peaks at t 0.53; a4's
  loss is lowest against it there (about 0.61 of it) and peaks later, in the upper-middle noise. *Hypothesis*: the
  weak band is where the image's layout has to be committed from little signal.

## K1. Where the noisy input enters the prediction

The question: should the prediction start from x_t, or from the best linear prediction `c(t)·x_t`, so that the trunk
writes only what a rescale of its input cannot give?

**Forms.**
- (a) **The skip at the input.** The slice starts at `c(t)·x_t`. The read-out stays the raw slice, and the slice is
  the prediction so far from the first block on; with the gates at zero the model is the best linear predictor. The
  meta channels still embed x_t itself, so no information is lost. At t → 0 the slice starts at twice x_t.
- (b) **The skip at the read-out, EDM's `c_skip`.** The slice starts at x_t and `F = slice + (c(t) − 1)·x_t` after
  the last block. The trunk's task is the same remainder, but the slice is never the prediction so far: every block
  reads a slice off by `(c − 1)·x_t`, corrected only at the end, and the read-out gains a term.
- **c_out on the slice's writes (EDM's `c_out`),** on (a). The gates on the slice are scaled by `c_out(t) = 1/|x_t|`,
  so what the trunk produces before the scale is of unit size at every t, as the correction `−v` is with the
  identity, and `c_out(t)` scales it to the remainder's RMS. The adaLN gates already depend on t and could learn
  the same scale; the arm reads whether the prior helps.

*Decided* for the arms (2026-11-21): (a), alone and with c_out. Branches: `arm/linear-skip`, the slice starts at
`c(t)·x_t` (`linear_skip` in the model); `arm/linear-skip-cout`, on it, a block's gates multiplied by a
`write_scale` that is `c_out(t)` on the slice and 1 on the meta channels (`skip_out`). Both take unit-variance
latents; against a4's 1.114 the slope is off by at most 0.03.

*Prediction:* the linear skip lowers the loss at the ends of t (below 0.25 and above 0.75), with little change at the
midpoint; c_out, against the linear skip, moves the middle and upper-middle t, where its writes are larger.

**Settle:** the linear skip against the reference, and c_out against the linear skip, on `loss_by_t` on both sets
(SPEC §8), and on samples for the grey patches. A further arm clamps c to [0, 1] if the factor 2 at low t proves
unstable.

**Interaction with the t distribution.** Training t is logit-normal (SPEC §4), centred near the midpoint, where a4 is
already strongest against the linear predictor; the ends carry little mass. If the skip frees the ends, the mass
could move: toward the upper-middle band by the shift, rather than spread by a wider σ (archive
2026-11-18-timestep-sigma).

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
