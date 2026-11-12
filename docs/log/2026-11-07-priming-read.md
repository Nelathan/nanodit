# Priming read (2026-11-07)

What the 3000 + 3000 priming arms (`128_d768x12_arch_prime_s80_3k`, then `..._cont_3k`) show, against `128_d768x12_arch_6k` from scratch. Same model (plain conv in 3/4, d768x12), data, batch 256 and seed. Stage 1 is our own objective (F = x_t − v) at a single noise level, t = 0.8, with the time embedding kept; stage 2 is ordinary flow training from its checkpoint with no warmup. At t = 0.8 the target is F = 1.2 x − 0.2 ε, so stage 1 is nearly a clean-latent prediction from a heavily noised input (x̂ = (1 − t) x_t + t F, so x-MSE is t² times F-MSE: at fixed t our loss is x-prediction up to a constant).

The arms were set up from the threads before the repo was read, and are not a faithful stage 1 (`topics/priming.md`, What the code does): the repo's network has no time input, ours fed a constant t to a trained time embedding that stage 2 then had to adapt; the repo normalises each sample's loss by its own detached value and adds perceptual losses, ours is plain MSE; it trains 160 epochs, our 3000-step stage 1 0.96 of an epoch of the 798k-row trial set (768k rows seen), the 1000-step one 0.32; its s = 0.8 is tuned for one-step generation from pure noise, which we do not do. Nor is it the masked reconstruction planned on 2026-11-05.

## Speed

Median training throughput: scratch 881 img/s, stage 1 877, stage 2 877. Priming is not faster per image: same network, same batch, and the source's dropped time embedding would cost nothing here either. The paper's "10x" is compute to a one-step FID, not throughput.

## Stage 1 saturates by step 2000

Caption loss on the eval grid during stage 1 (t bins from low to high noise; the last three bins are the ones stage 1 trains):

| step | all t | t ≈ 0.16 | … 0.74 | 0.84 | 0.93 |
|---|---|---|---|---|---|
| 1000 | 2.5085 | 3.821 | 1.833 | 1.607 | 1.400 |
| 2000 | 2.3851 | 3.564 | 1.793 | 1.582 | 1.390 |
| 3000 | 2.3794 | 3.546 | 1.798 | 1.582 | 1.384 |
| scratch at 3000 | 1.5042 | 1.301 | 1.605 | 1.547 | 1.379 |

- The trained level stops improving between 2000 and 3000 steps: the third of stage 1 was spent on a converged task. Priming "until convergence" is what this run did.
- At the noise levels it trains, stage 1 with all its 3000 steps there is no better than scratch, which spent a tenth of its steps there (1.384 against 1.379 at the top bin, 1.582 against 1.547 below it). What is learnable at high noise (layout, colour) is learned fast either way.
- At low noise stage 1 is far worse than the identity init (3.55 at t ≈ 0.16 against 2.16 for F = x_t): having only seen t = 0.8, it overrides the input with a blurred guess at every t. The time embedding saw one value, so nothing in the network tells levels apart.

## Recovery

Eval caption loss after the same number of normal-t steps:

| normal-t steps | scratch | after 3000 primed steps |
|---|---|---|
| 1000 | 1.5633 | 1.5273 |
| 2000 | 1.5207 | 1.5042 |
| 3000 | 1.5042 | 1.4914 |
| 6000 | 1.4788 | not run |

- The low-t damage is undone within 50 steps (train loss at t < 0.1 is back to 2.14 at step 3050).
- Stage 2 then leads scratch at equal normal-t steps (0.036 at 1000, 0.013 at 3000), so the primed weights are a better start than random ones. But stage 2 at 2000 steps equals scratch at 3000: 3000 primed steps bought about 1000 normal ones. At equal total steps it is 0.013 behind (1.4914 against 1.4788).

## Was stage 1 a good init for x0 velocity?

The parametrisation is consistent (our objective at one t, the same output head), so nothing has to be re-learned in the head; that is why the recovery is fast and the start is better than random. What it is not is a good use of compute for a multi-step model: the loss that is left after a few thousand steps sits at low and middle t (fine detail), which stage 1 never trains, and at its own t it gains nothing over spreading the steps. The paper uses the primed network as the backbone of a one-step generator that maps t = 0.8-like input to an image, where all of stage 1's learning is on the path; that is not our setting.

## The 1000-step stage 1 (`queue_prime1k_2026-11-07.sh`)

Stage 1 cut to 1000 steps, then 5000 normal-t steps, against scratch:

| step (total) | scratch | prime 3000 + 3000 | prime 1000 + 5000 |
|---|---|---|---|
| 2000 | 1.5207 | | 1.5531 |
| 3000 | 1.5042 | | 1.5187 |
| 4000 | 1.4935 | 1.5273 | 1.5008 |
| 5000 | 1.4851 | 1.5042 | 1.4903 |
| 6000 | 1.4788 | 1.4914 | 1.4830 |
| 6000 at 256 px | 1.3769 | 1.3891 | 1.3814 |

At equal normal-t steps it starts 0.010 ahead (1.5531 after 1000 against scratch at 1000, 1.5633) and ends 0.002 ahead (1.4830 after 5000 against 1.4851): the primed start is worth less than the steps it took, and its lead fades. At equal total steps it is 0.004 behind at 128 px and 0.0045 at 256 px. Shorter priming loses less only because it spends less.

## Verdict on single-level denoising

Single-level denoising at t = 0.8 with a trained time embedding does not help a multi-step flow model at our scale: every step spent there buys less than an ordinary step. That says nothing yet about priming done as the source does it, nor about masked reconstruction. The per-t lesson stands: high-noise structure is learned fast from the ordinary t distribution anyway, and what is left to learn sits at low and middle t, which a single high noise level never trains. Hypothesis, untested: a corruption that teaches content from context (masking) could add something flow training does not already learn.
