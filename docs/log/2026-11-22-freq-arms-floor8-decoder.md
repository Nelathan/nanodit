# a4 at 6000 steps: the frequency arms and the base under the decoder weight floored at 8 latent px

`uv run python scripts/rescore.py RUN_DIR 6000 floor8` from `arm/freq-floor-8` `21f2845` (`objective.GAIN_FLOOR` at
1 / PATCH) on `a4-base-6k` (α 0), `a4-freq-alpha05-6k` (α 0.5, floor 64), `a4-freq-alpha1-floor16-6k` and
`a4-freq-alpha1-floor8-6k` (log 2026-11-22-freq-weight-floors): each run's held-out and Lunara sets on both grids into
`score*_floor8.jsonl` in its run directory. The decoder-weighted loss is the loss at α 1 under the floor at 8 px, the
objective the floor-8 arm trains on. The rescore runs the blocks uncompiled; on the floor-8 run it repeats the
training's own scores within 0.0001. The seed spread is about 0.001; under about 0.002 is not resolved (log
2026-11-18-seed42).

## Score at 6000

Decoder-weighted loss, floor 8 px; Δ against the base, from the unrounded scores.

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.4605 | | 1.5470 | | 1.4247 | | 1.4911 | |
| α 0.5, floor 64 | 1.4667 | +0.0062 | 1.5544 | +0.0075 | 1.4309 | +0.0062 | 1.4977 | +0.0066 |
| α 1, floor 16 | 1.4794 | +0.0189 | 1.5679 | +0.0209 | 1.4396 | +0.0149 | 1.5076 | +0.0165 |
| α 1, floor 8 | 1.4663 | +0.0058 | 1.5563 | +0.0093 | 1.4300 | +0.0054 | 1.4988 | +0.0077 |

Uniform grid, Δ decoder-weighted loss by t against the base.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| α 0.5, held-out | +0.0058 | +0.0052 | +0.0096 | +0.0122 | +0.0105 | +0.0068 | +0.0022 | −0.0010 | +0.0011 | +0.0098 |
| α 1 floor 16, held-out | +0.0056 | +0.0138 | +0.0248 | +0.0317 | +0.0285 | +0.0213 | +0.0120 | +0.0055 | +0.0048 | +0.0011 |
| α 1 floor 8, held-out | +0.0081 | +0.0069 | +0.0116 | +0.0130 | +0.0100 | +0.0044 | −0.0009 | −0.0027 | +0.0022 | +0.0010 |
| α 0.5, Lunara | +0.0047 | +0.0043 | +0.0104 | +0.0146 | +0.0135 | +0.0097 | +0.0032 | −0.0018 | +0.0026 | +0.0052 |
| α 1 floor 16, Lunara | +0.0047 | +0.0119 | +0.0260 | +0.0357 | +0.0326 | +0.0252 | +0.0151 | +0.0072 | +0.0043 | +0.0026 |
| α 1 floor 8, Lunara | +0.0066 | +0.0069 | +0.0138 | +0.0173 | +0.0154 | +0.0103 | +0.0041 | +0.0005 | +0.0025 | −0.0003 |

## Read

- **The base scores best on the floor-8 arm's own objective.** α 1 at 8 px trains on this weight and is behind the
  base on all four scores, by 0.005 to 0.009; α 0.5 is level with it within the spread (0.0004 to 0.0019 apart), α 1
  at 16 px behind both.
- **On the uniform grid the cost sits at low and mid t**, as on the plain score (log 2026-11-22-freq-weight-floors).
  Floor 8 is behind the base from t 0.05 to 0.55 (most at t 0.35, +0.013 / +0.017) and within 0.003 of it from t
  0.75; held-out at t 0.75 (−0.003) is the one point where it is ahead of the base past the spread.
