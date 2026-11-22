# a4 at 6000 steps: a slice and a meta write gate per branch

Run (`config.toml` in its run directory; a4 on the arm's code, seed 41, 6000 steps), against
`a4-held-meta-skips-linear-skip-6k` (the tables, `e271f72`), `a4-held-meta-skips-linear-skip-scalar-gates-6k` (one
scalar per write, `887cf86`; log 2026-11-22-scalar-gates-linear-skip) and `a4-base-6k` (main):

| Run | Code | Change on the scalar gates |
|---|---|---|
| `a4-held-meta-skips-linear-skip-split-gates-6k` | `arm/held-meta-skips-linear-skip-split-gates` `7e5e088` | each write under two scalars through a tanh, from zero and independent of t: one on the slice, one on the meta channels |

It scores the uniform grid natively. The seed spread at 6000 steps is about 0.001; a difference under about 0.002 is
not resolved (log 2026-11-18-seed42).

## Score at 6000

Δ against the base, from the unrounded scores.

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| tables | 1.2974 | −0.0095 | 1.3513 | −0.0095 | 1.2481 | −0.0090 | 1.2880 | −0.0091 |
| one scalar per write | 1.3234 | +0.0164 | 1.3751 | +0.0143 | 1.2685 | +0.0114 | 1.3062 | +0.0091 |
| slice and meta gates | 1.3153 | +0.0083 | 1.3689 | +0.0081 | 1.2621 | +0.0050 | 1.3018 | +0.0046 |

Against the tables the split gates are +0.0178 / +0.0175 / +0.0140 / +0.0138.

Uniform grid, Δ loss by t against the base (its loss in log 2026-11-22-tables-on-held-stream).

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| tables, held-out | −0.0175 | −0.0094 | −0.0104 | −0.0114 | −0.0116 | −0.0111 | −0.0095 | −0.0062 | −0.0026 | −0.0007 |
| one scalar, held-out | −0.0096 | +0.0079 | +0.0204 | +0.0265 | +0.0261 | +0.0201 | +0.0116 | +0.0040 | +0.0010 | +0.0058 |
| split, held-out | −0.0094 | +0.0043 | +0.0156 | +0.0195 | +0.0163 | +0.0081 | −0.0002 | −0.0033 | −0.0024 | +0.0012 |
| tables, Lunara | −0.0183 | −0.0100 | −0.0102 | −0.0114 | −0.0122 | −0.0119 | −0.0092 | −0.0028 | −0.0019 | −0.0033 |
| one scalar, Lunara | −0.0121 | +0.0045 | +0.0184 | +0.0249 | +0.0239 | +0.0178 | +0.0098 | +0.0029 | −0.0005 | +0.0012 |
| split, Lunara | −0.0132 | +0.0016 | +0.0147 | +0.0190 | +0.0156 | +0.0071 | −0.0004 | +0.0001 | +0.0009 | +0.0012 |

## The gates at 6000

`uv run python scripts/probe_gates.py RUN_DIR 6000` from `7e5e088`: |tanh g| per write and part. Against the tables'
run, the same probe from `e271f72` (log 2026-11-22-scalar-gates-linear-skip): its gate rows as RMS over channels and
the quantile grid's 10 t, raw multipliers with no bound; the ratio is the split meta gate over the tables' meta RMS in
the same write. Ranges over the blocks of each kind; the last block's MLP meta gate, which nothing reads, is left out.

| Write | Slice gate | Meta gate | Meta gate over the tables' |
|---|---|---|---|
| conv mixer (18 blocks) | 0.001 – 0.055 | 0.015 – 0.143 | 0.11 – 0.63 |
| attention mixer (6 blocks) | 0.008 – 0.208 | 0.232 – 0.601 | 0.83 – 1.47 |
| MLP (23 blocks) | 0.006 – 0.204 | 0.092 – 0.434 | 0.45 – 1.63 |

## Read

- **The split takes back a quarter to a third of the scalar's cost.** Against the tables the cost falls from +0.026 / +0.024
  / +0.020 / +0.018 to +0.018 / +0.018 / +0.014 / +0.014; from t 0.65 the split gates are level with the base or
  better, and in mid-t they still cost +0.019 at t 0.35.
- **The conv mixers' meta gates stay below the tables' in every block.** Attention's meta gates are 0.8 to 1.5 times
  the tables', the MLP's 0.5 to 1.6 (below 1 in blocks 0 to 8 and 16 to 22); the conv mixers' are 0.1 to 0.6, and
  their slice gates 0.001 to 0.055 (*Hypothesis*: the remaining cost is in the small gates of the conv mixers' writes,
  and perhaps the early MLPs').
