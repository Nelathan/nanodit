# a4 at 6000 steps: the frequency weight at α 1, floored at 16 and 8 latent px

Runs (each `config.toml` in its run directory; a4 at `loss_alpha` 1, seed 41, 6000 steps), against `a4-base-6k`
(α 0) and the α arms floored at 64 px (log 2026-11-22-freq-weight-arms). The floor is `objective.GAIN_FLOOR`, the
period past which the decoder's gain is held flat.

| Run | Code | Floor |
|---|---|---|
| `a4-freq-alpha1-floor16-6k` | `arm/freq-floor-16` `b70d20a` | 1 / (2 · PATCH): a period of 16 latent px, twice a token |
| `a4-freq-alpha1-floor8-6k` | `arm/freq-floor-8` `21f2845` | 1 / PATCH: a period of 8 latent px, one token |

The decoder-weighted score is the loss at α 1 under the run's own floor, so it is not comparable across floors and is
not tabled. The plain scores' seed spread is about 0.001; under about 0.002 is not resolved (log 2026-11-18-seed42).

## Score at 6000

Plain MSE; Δ against the base, from the unrounded scores.

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| α 0.5, floor 64 | 1.3214 | +0.0144 | 1.3748 | +0.0140 | 1.2694 | +0.0123 | 1.3088 | +0.0117 |
| α 1, floor 64 | 1.3808 | +0.0738 | 1.4297 | +0.0689 | 1.3184 | +0.0613 | 1.3529 | +0.0558 |
| α 1, floor 16 | 1.3423 | +0.0353 | 1.3950 | +0.0342 | 1.2848 | +0.0277 | 1.3235 | +0.0264 |
| α 1, floor 8 | 1.3269 | +0.0199 | 1.3806 | +0.0198 | 1.2736 | +0.0165 | 1.3129 | +0.0158 |

Uniform grid, held-out: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1171 | 1.2121 | 1.2582 | 1.2857 | 1.3129 | 1.3448 | 1.3705 | 1.3516 | 1.2416 | 1.0767 |
| α 0.5, floor 64 | +0.0073 | +0.0114 | +0.0196 | +0.0238 | +0.0215 | +0.0162 | +0.0091 | +0.0034 | +0.0028 | +0.0077 |
| α 1, floor 64 | +0.0494 | +0.0487 | +0.0847 | +0.1071 | +0.1085 | +0.0901 | +0.0607 | +0.0299 | +0.0131 | +0.0206 |
| α 1, floor 16 | +0.0102 | +0.0268 | +0.0457 | +0.0561 | +0.0517 | +0.0404 | +0.0249 | +0.0120 | +0.0062 | +0.0027 |
| α 1, floor 8 | +0.0112 | +0.0170 | +0.0282 | +0.0326 | +0.0294 | +0.0212 | +0.0115 | +0.0048 | +0.0054 | +0.0037 |

Uniform grid, Lunara: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1214 | 1.2505 | 1.3206 | 1.3549 | 1.3782 | 1.4031 | 1.4226 | 1.3936 | 1.2561 | 1.0699 |
| α 0.5, floor 64 | +0.0050 | +0.0087 | +0.0181 | +0.0240 | +0.0219 | +0.0168 | +0.0087 | +0.0029 | +0.0049 | +0.0062 |
| α 1, floor 64 | +0.0430 | +0.0397 | +0.0776 | +0.1034 | +0.1062 | +0.0868 | +0.0548 | +0.0256 | +0.0099 | +0.0114 |
| α 1, floor 16 | +0.0079 | +0.0217 | +0.0433 | +0.0569 | +0.0525 | +0.0410 | +0.0247 | +0.0115 | +0.0041 | +0.0007 |
| α 1, floor 8 | +0.0084 | +0.0140 | +0.0268 | +0.0331 | +0.0307 | +0.0227 | +0.0123 | +0.0049 | +0.0045 | +0.0008 |

## Read

- **The floor sets most of α 1's plain cost.** Floored at 16 px α 1 costs about half what it costs at 64 px on every
  score, at 8 px 0.27 to 0.29 of it; at 8 px it is within 0.004 to 0.006 of α 0.5 at 64 px.
- **The cost keeps its place in t.** Every arm peaks at t 0.35 to 0.45; the floor scales the peak (held-out +0.107,
  +0.056, +0.033 at t 0.35) and the high end least: at t 0.85 the three floors of α 1 are within 0.008 of each other,
  and at t 0.95 floors 16 and 8 are level with the base or near it (+0.003 / +0.004 held-out, +0.001 Lunara).
- **The plain score cannot rank a frequency weight against the base.** A weight that moves error from coarse to fine
  periods costs on the unweighted MSE by construction; between floors there is no common weighted score. The sample
  grids decide.
