# a4 at 6000 steps: scalar write gates with the linear skip, and the gates' size

Run (`config.toml` in its run directory; a4 on the arm's code, seed 41, 6000 steps), against
`a4-held-meta-skips-linear-skip-6k` (`arm/held-meta-skips-linear-skip` `e271f72`, log 2026-11-22-skip-arms),
`a4-held-meta-skips-scalar-gates-6k` (log 2026-11-22-tables-on-held-stream) and `a4-base-6k` (main):

| Run | Code | Change |
|---|---|---|
| `a4-held-meta-skips-linear-skip-scalar-gates-6k` | `arm/held-meta-skips-linear-skip-scalar-gates` `887cf86` | on held meta + long skips + linear skip, the scalar gates of `dcfc8e7`: the tables and the shared gate rows removed, each write gated by one per-block scalar through a tanh, from zero and independent of t; the MLP's output norm without a gain |

It scores the uniform grid natively. The seed spread at 6000 steps is about 0.001; a difference under about 0.002 is
not resolved (log 2026-11-18-seed42).

## Score at 6000

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| held meta + long skips + linear skip | 1.2974 | −0.0095 | 1.3513 | −0.0095 | 1.2481 | −0.0090 | 1.2880 | −0.0091 |
| the same, scalar gates | 1.3234 | +0.0164 | 1.3751 | +0.0143 | 1.2685 | +0.0114 | 1.3062 | +0.0091 |

Δ against the base, from the unrounded scores. Against the same run with the tables: +0.0259 / +0.0237 / +0.0204 /
+0.0182.

Uniform grid, Δ loss by t against the base (its loss in log 2026-11-22-tables-on-held-stream); the scalar gates
without the linear skip in log 2026-11-22-tables-on-held-stream.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| tables, held-out | −0.0175 | −0.0094 | −0.0104 | −0.0114 | −0.0116 | −0.0111 | −0.0095 | −0.0062 | −0.0026 | −0.0007 |
| scalar gates, held-out | −0.0096 | +0.0079 | +0.0204 | +0.0265 | +0.0261 | +0.0201 | +0.0116 | +0.0040 | +0.0010 | +0.0058 |
| tables, Lunara | −0.0183 | −0.0100 | −0.0102 | −0.0114 | −0.0122 | −0.0119 | −0.0092 | −0.0028 | −0.0019 | −0.0033 |
| scalar gates, Lunara | −0.0121 | +0.0045 | +0.0184 | +0.0249 | +0.0239 | +0.0178 | +0.0098 | +0.0029 | −0.0005 | +0.0012 |

## The gates at 6000

`uv run python scripts/probe_gates.py RUN_DIR 6000`, from `887cf86` on the scalar gates' run and from `e271f72` on
the tables' run. For the scalar gates |tanh g|, one value per write, the same for every channel and t. For the tables'
run the gate rows of `mod + table`, raw multipliers with no tanh and no bound, as RMS over channels and the 10 t of
the quantile score grid (0.16 to 0.84), on the slice (the first 128 channels, the output) and on the meta channels.
The two are magnitudes of different objects: a scalar against an RMS that weights the larger channels and t more.
Ranges over the blocks of each kind.

| Write | Scalar gates | Tables, slice | Tables, meta |
|---|---|---|---|
| conv mixer (18 blocks) | 0.010 – 0.055 | 0.010 – 0.149 | 0.128 – 0.311 |
| attention mixer (6 blocks) | 0.093 – 0.238 | 0.016 – 0.179 | 0.205 – 0.459 |
| MLP (24 blocks) | 0.059 – 0.191 | 0.028 – 0.173 | 0.167 – 0.378 |

## Read

- **The linear skip does not account for the scalar gates' cost.** It moves t 0.05 as it does with the tables (by
  −0.016 / −0.015 against the scalar gates without it) and t 0.15 by −0.005; from t 0.25 to 0.85 the scalar gates'
  cost is the same with it or without, within 0.003 (+0.026 / +0.025 at t 0.35). At t 0.95 held-out it is 0.007
  higher with the skip.
- **The scalar gates stay far inside their bound.** The largest is 0.24 against the tanh's 1; the conv mixers' are
  0.01 to 0.06.
- **With the tables the slice and the meta channels are written at different sizes.** The meta gates are 1.1 to 19
  times the slice's in the same write: 17 to 19 in the mixers of blocks 6 to 11 except block 10, 1.1 to 4.9 in the
  last six blocks. The scalar of a write is 0.3 to 6.8 times the tables' slice gate in the same write, and its meta gate
  is 1.3 to 17 times the scalar, in every write (*Hypothesis*: the slice, read by the loss directly, sets the scalar, and
  the meta writes may be too small). The scalar arm also lost the MLP output norm's gain; this run does not separate
  the two.
