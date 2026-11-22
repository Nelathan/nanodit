# a4 at 6000 steps: the loss's frequency weight at α 0.5 and 1

Runs (each `config.toml` in its run directory; a4, seed 41, 6000 steps, from `main` at `ad1a7ad`), against
`a4-base-6k` (α 0). The weight is the decoder's gain to the power α, `loss_alpha` (PLAN K3,
`objective.spectral_weight`). The decoder-weighted score is the loss at α 1, whatever α a run trains on: a power-law model of the image's
error, floored at a period of 64 latent px, comparable within one resolution. The held-out set scores at 128 px, Lunara at 256 px. The base's
decoder-weighted scores come from `scripts/score_checkpoint.py` on its 6000-step checkpoint at `ad1a7ad`
(`score_128`, `score_256` and their uniform files in its run directory); its plain scores reproduce the ones it
trained with. The plain scores' seed spread is about 0.001 (log 2026-11-18-seed42); the decoder-weighted score's is
not measured.

## Score at 6000

Δ against the base in brackets.

| Run | Held-out | Lunara | Held-out uniform | Lunara uniform | Held-out decoder, uniform | Lunara decoder, uniform | Held-out decoder | Lunara decoder |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | 1.3608 | 1.2571 | 1.2971 | 1.5635 | 1.6458 | 1.5431 | 1.6500 |
| α 0.5 | 1.3214 (+0.0144) | 1.3748 (+0.0140) | 1.2694 (+0.0123) | 1.3088 (+0.0117) | 1.5572 (−0.0064) | 1.6400 (−0.0058) | 1.5328 (−0.0103) | 1.6409 (−0.0090) |
| α 1 | 1.3808 (+0.0738) | 1.4297 (+0.0689) | 1.3184 (+0.0613) | 1.3529 (+0.0558) | 1.5974 (+0.0339) | 1.6784 (+0.0326) | 1.5779 (+0.0348) | 1.6828 (+0.0329) |

Uniform grid, held-out, decoder-weighted: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1381 | 1.2212 | 1.2753 | 1.3264 | 1.4008 | 1.5201 | 1.7060 | 1.9430 | 2.0831 | 2.0215 |
| α 0.5 | −0.0053 | −0.0067 | −0.0055 | −0.0043 | −0.0082 | −0.0136 | −0.0187 | −0.0156 | −0.0022 | +0.0163 |
| α 1 | +0.0437 | +0.0176 | +0.0365 | +0.0503 | +0.0517 | +0.0428 | +0.0309 | +0.0127 | +0.0196 | +0.0328 |

Uniform grid, Lunara, decoder-weighted: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1455 | 1.2713 | 1.3506 | 1.4158 | 1.5005 | 1.6342 | 1.8461 | 2.1022 | 2.1769 | 2.0150 |
| α 0.5 | −0.0038 | −0.0126 | −0.0081 | −0.0011 | −0.0003 | −0.0034 | −0.0154 | −0.0220 | +0.0012 | +0.0073 |
| α 1 | +0.0410 | +0.0013 | +0.0241 | +0.0461 | +0.0550 | +0.0488 | +0.0249 | +0.0189 | +0.0340 | +0.0314 |

## Read

- **Both α lose on the plain score,** α 0.5 by +0.012 to +0.014, α 1 by +0.056 to +0.074, well past its 0.001
  spread.
- **α 1 also loses on the decoder-weighted score it trains on,** +0.033 to +0.035, positive at every t (smallest
  +0.001 at Lunara t 0.15). This is 2 % of the score and likely outside its spread, which is not measured.
  *Hypothesis*: the weight spans orders of magnitude over a few coarse coefficients and, normalised to mean 1, puts
  most of the grid below 1, so the step sees fewer effective samples, more gradient noise and a changed effective
  learning rate; whether the fine bands' gradient also teaches the coarse ones has no probe behind it.
- **α 0.5's gain on the decoder-weighted score is not resolved,** −0.006 / −0.009 to −0.010 (uniform / shaped grid).
  The score's spread is not measured and is likely wider than the plain score's, the few low-frequency coefficients
  carrying large weights. By t the gain sits at t 0.55–0.75 on held-out (−0.014 to −0.019) and at t 0.65–0.75 and
  0.15 on Lunara (−0.013 to −0.022); at t 0.95 it loses (+0.016 held-out, +0.007 Lunara).
- The decision is on the sample grids (PLAN K3): `eval/step_0006000.png` in each run directory.
