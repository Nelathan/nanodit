# Edge of stability and the gradient-moment cosine

The question and its reads, moved here whole from `PLAN.md` and `FACTS.md` on 2026-11-11 so the main ledgers stay short. PLAN keeps the question's heading and points here.

## The question (PLAN C4)

**C4. Does an edge-of-stability prediction for momentum optimizers bear on the learning rate chosen?**

Added 2026-11-06: https://jeffreycider.github.io/posts/bubbleforce/ (listed in REFERENCES).
The post (Cheng and Zheng, 2026) claims that for non-adaptive optimizers, Muon included, the equilibrium is
`lr · pathspring = bubbleforce`: pathspring is the top real eigenvalue of the optimizer step's Jacobian (the Hessian for
SGD), and bubbleforce is set by the momentum kernel alone (2 for SGD, `2(1+β)/(1-β)` for heavy ball). It predicts that
halving or doubling lr and lowering β move the sharpness to the predicted equilibrium, some within a few
iterations, on a nanoGPT Muon run; raising β does not follow it (about 10% off); the baseline run falls behind it near
the end under lr decay, for which the authors give no theory, and one scheduled-sharpness run ends at 2 to 10 times the
prediction. Adam-type optimizers are outside it by the authors' account (dead neurons spoil the eigenvalue
measurements). Code: https://github.com/jeffreycider/bubbleforce.
This is optimizer work that belongs to the usuitrack repo and starts here, because it may settle cleanly why the cosine
between the gradient and the momentum (`grad_moment_cosine`) is negative in the runs: more momentum makes it less
negative with diminishing returns, and only a lower lr turns it up, which stops proper learning (the learning-rate
question above). A negative cosine means the gradient points against the accumulated step, an oscillation, which is
what the post's edge-of-stability picture describes for momentum: the largest root of the step's characteristic
polynomial sits on the unit circle, and where it sits on it sets the oscillation (reading of the post, untested).
Decided 2026-11-06: the authors' code is a reference only; the implementation is the repo's own, for its harness.
First read, UsuiTrack's update (`usuitrack/optimizer.py` in the usuitrack repo, branch `zero-init`): the moment is an exponential average of unit projected
gradients, `M = beta M + (1 - beta) Z/||Z||`, which is the post's momentum kernel `c_j = (1 - beta) beta^j` (weights sum
to 1, no Nesterov) acting on the normalised gradient; the write is `lr · ||W||` times the Newton-Schulz polar factor of
`M` (the post's Muon case), inside a low-rank frame that itself turns each step. On a noisy quadratic with that kernel,
bubbleforce is `2(1 + beta)/(1 - beta)` and the cosine between gradient and previous moment falls from 0 to -1 as
the effective step times the sharpness approaches it (the effective step is not `lr`: the normalisation and the polar
map set it). It is consistent with the beta dependence of the real runs in one pair of readings, with one fitted
parameter (below, Momentum cosine on a noisy quadratic). The lr dependence cannot be read from the runs so far, since
the lr pair also differs in beta, warmup and rank fraction. The toy leaves out the normalisation, the polar map and the
turning frame, which change the Jacobian the post's pathspring is defined on.
Reading (2026-11-06): a higher lr usually does make the cosine more negative, and a tiny lr makes it positive;
the sign is a spectrum effect, set by how many channels oscillate against how many are stable. A scratch toy with many
directions of power-law sharpness and a descent signal goes the same way (positive at a tiny lr, negative as the lr
rises); it is not saved. In the LLM lane a good fit read about -0.01; the DiT adds the noise of denoising and of the
sampled t, and the belief is that this is the difference to the -0.4 here.
Cautious stepping (zero the entries where the moment and the projected gradient differ in sign) was tried in
usuitrack and did not work out nicely. A linear filter on the moments instead, writing the mean of the last and the
current moment, was never tried. In the toy it fails: the half step of extra lag moves the edge from `2(1+β)/(1-β)`
(398 at β 0.99) down to `lr · sharpness = 2`, for every β, so it is not worth an arm. A kernel with no response at
period 2 needs weight on the newest gradient to stay stable, which MaxEntSlop fixes by construction.
The lr-jump test (2026-11-06, below) bears on this: from a late checkpoint, a higher constant lr made the cosine more
negative (about -0.04, -0.08 and -0.11 at 1e-4, 4e-4 and 1e-3), and the network answered by flattening: the top Hessian
eigenvalue fell 35 % at 1e-3 within 150 steps and stayed put at the control lr, for about 2 % more loss and no spikes.
Reading: the cosine follows lr times sharpness, not the lr, and the pinned lr is not a stability ceiling, since the
curvature adapts to it. That favours a constant lr with a late anneal: the extra loss at a high lr looks like a higher
noise floor, which the anneal removes; whether it recovers the 2 % is untested. The cosine at 10x the lr is still far from
the -0.4 of the earlier runs, so the lr alone does not explain those. Next: lower the lr again after a high-lr stretch and
see whether the curvature rebounds (then the anneal is partly progressive sharpening and should stay short).
Open:
- Whether the normalised gradient, the polar map and the turning frame keep the update inside the post's non-adaptive class; the momentum kernel itself does.
- Whether pathspring (Arnoldi on Hessian-vector products) is measurable on this model at an affordable cost; the post
  reports 16 to 33 HVPs per point on 8×H100s for a 12-layer, 768-wide model.
- Whether it would explain what the 2026-11-02 lr sweep measured: loss kept falling as lr rose to 4e-3, and the common mode (the component every token's residual stream shares) grew with it.

## Measurements

### Momentum cosine on a noisy quadratic (2026-11-06)

`scripts/probe_momentum_cosine.py`, CPU, seed 0: one quadratic direction (sharpness 1), white gradient noise, UsuiTrack's moment `M = beta M + (1 - beta) g` written as `-lr M` (the polar map, the gradient normalisation and the frame are not modelled). The characteristic polynomial is `z^2 - (1 + beta - eta lam (1 - beta)) z + beta`; its roots reach -1 at `eta lam = 2 (1 + beta) / (1 - beta)` (38 at beta 0.9, 78 at 0.95, 398 at 0.99), and the root radius is `sqrt(beta)` below it.

| `eta lam` / bubbleforce | 0.01 | 0.1 | 0.3 | 0.6 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| `cos(g_t, M_{t-1})`, beta 0.9 | -0.04 | -0.26 | -0.52 | -0.76 | -0.95 | -0.995 |
| beta 0.95 | -0.06 | -0.29 | -0.53 | -0.77 | -0.95 | -0.995 |
| beta 0.99 | -0.08 | -0.31 | -0.55 | -0.77 | -0.95 | -0.995 |

The cosine is nearly a function of the fraction alone. The real `grad_moment_cosine` at step 1000 is -0.41 at beta 0.9 (`128_fixed_2e-3_9k`) and -0.10 at beta 0.99 (`128_fixed_2e-3_b099_1k`), same lr, other settings equal. The script fits the fraction at which beta 0.9 reads -0.41 (0.203) and scales it to beta 0.99, where bubbleforce is 10.5 times larger (0.0194): it reads -0.124. That is consistent with the observed -0.10 in one pair of readings with one fitted parameter. It assumes the same sharpness in both runs, which an edge-of-stability equilibrium would not give, the beta 0.99 reading is still drifting (+0.09, -0.01, -0.10 at steps 300, 500, 1000), and a real matrix has many directions of different sharpness. Any monotone curve would order the two runs the same way.

What the toy cannot say: the lr direction. The two lr readings (`128_fixed_2e-2_9k`, -0.33 to -0.36, against `128_fixed_2e-3_9k`, -0.39 to -0.45, steps 501 to 4300) also differ in beta (0.95 against 0.9), warmup and rank fraction, so no lr effect can be read from them. The early positive values of the smoke runs (+0.44 to +0.51 over steps 3 to 11) are not in the toy, which has no gradient signal.

### Learning-rate jump from `swpin_cont_3ep` (C4, 2026-11-06)

Source: `runs/128_d1024x16_shock_lr{1e-4,4e-4,1e-3}/` (`sharpness.jsonl`, `train.jsonl`; `scripts/probe_sharpness.py`; each run's `config.toml`). Each arm continues `128_d1024x16_swpin_cont_3ep` from step 62168 for 300 steps at a constant Usui relative step (fallback 1.25x it), with the optimizer state of the checkpoint. The top Hessian eigenvalue is read at 17 checkpoints on one fixed batch of the first 128 Lunara rows (one fixed noise and t each), fp32, central-difference Hessian-vector products, 12 power iterations; 40 iterations agree to four digits at the start and the three end checkpoints.

| usui lr after the jump | top eigenvalue at 62168 / at 62468 | change | mean `grad_moment_cosine`, steps 12 to 300 | mean loss, 7 logs to step 150 | mean loss, last 4 logs |
|---|---|---|---|---|---|
| 1e-4 (control) | 28.0 / 28.9 | +3 % | -0.04 | 1.132 | 1.129 |
| 4e-4 | 28.0 / 26.6 | -5 % | -0.08 | 1.136 | 1.137 |
| 1e-3 | 28.0 / 18.3 | -35 % | -0.11 (-0.15 at step 12) | 1.148 | 1.155 |

- The control stays between 27.9 and 29.1 at every checkpoint. The 4e-4 arm falls to 25.4 by step 62340 and sits at 26 to 27 after. The 1e-3 arm falls to 21.8 by step 62280, to 19 by 62340 and reads 18.3 to 19.7 from 62400, still drifting down slowly at the end.
- The cosine is more negative at the higher lr, and in the 1e-3 arm it eases from -0.15 to about -0.10 over the first 100 steps as the eigenvalue falls.
- No loss spike in any arm. The losses are the means of 16 logged micro-batch averages, so the 0.01 to 0.02 gaps are at the noise of that estimate; the sharpness change at 1e-3 is not.
- One seed and one fixed batch per arm; the batch is the eval rows (conditional loss, no prompt dropout).
- Not read from this: how the cosine depends on beta (the arms share beta 0.99), and whether the lower sharpness is kept when the lr is lowered again.
