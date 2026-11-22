# a4 at 6000 steps: what the adaLN tables hold, and whether the step carries the writers' row scales

Two probes at step 6000. On `a4-held-meta-skips-linear-skip-6k` (the tables, `e271f72`, log 2026-11-22-skip-arms) and
`a4-base-6k` (main): `uv run python scripts/probe_modulation_parts.py RUN_DIR 6000` and the block-similarity maps of
`scripts/plot_gates.py`, each run from the run's code. On `a4-held-meta-skips-linear-skip-split-gates-6k` (a slice and
a meta scalar gate per write, `7e5e088`, log 2026-11-22-split-gates): `uv run python scripts/probe_write_scales.py
RUN_DIR 6000 40` from `7e5e088`, 40 training steps at the run's constant lr from its checkpoint and optimizer state.

## The parts of the modulation

Each row M[block, t, channel] = mod(t) + table(block) is exactly μ(c) + A(block, c) + B(t, c): a per-channel
constant, the tables about their mean over blocks, the shared projection about its mean over t (20 evenly spaced t).
Shares of mean M²; of A, the share one scalar per block carries and the share its first singular pair carries.

| Row, part | Tables run: μ / A / B | of A: scalar / first pair | Base: μ / A / B | of A: scalar / first pair |
|---|---|---|---|---|
| mixer read scale, slice | 0.86 / 0.05 / 0.08 | 0.15 / 0.32 | 0.71 / 0.24 / 0.05 | 0.27 / 0.43 |
| mixer read scale, meta | 0.90 / 0.09 / 0.01 | 0.23 / 0.37 | 0.93 / 0.06 / 0.01 | 0.18 / 0.32 |
| mixer gate, slice | 0.13 / 0.86 / 0.01 | 0.01 / 0.25 | 0.14 / 0.85 / 0.01 | 0.01 / 0.27 |
| mixer gate, meta | 0.42 / 0.53 / 0.05 | 0.00 / 0.19 | 0.26 / 0.71 / 0.03 | 0.00 / 0.36 |
| MLP read scale, slice | 0.12 / 0.66 / 0.22 | 0.11 / 0.34 | 0.03 / 0.74 / 0.23 | 0.04 / 0.37 |
| MLP read scale, meta | 0.51 / 0.25 / 0.24 | 0.18 / 0.26 | 0.80 / 0.09 / 0.10 | 0.12 / 0.37 |
| MLP gate, slice | 0.21 / 0.75 / 0.05 | 0.01 / 0.23 | 0.30 / 0.64 / 0.07 | 0.01 / 0.19 |
| MLP gate, meta | 0.54 / 0.34 / 0.13 | 0.00 / 0.16 | 0.35 / 0.48 / 0.18 | 0.00 / 0.34 |

Block similarity on the tables run: the cosine between two blocks' table rows about their mean over blocks, averaged
over pairs of conv blocks, of attention blocks, of one of each, of neighbours, and of blocks 8 or more apart.

| Row, part | conv–conv | attn–attn | conv–attn | neighbours | ≥ 8 apart |
|---|---|---|---|---|---|
| mixer read scale, slice | +0.00 | +0.27 | −0.15 | +0.08 | −0.07 |
| mixer read scale, meta | +0.06 | +0.45 | −0.24 | +0.04 | −0.07 |
| mixer gate, slice | +0.18 | −0.06 | +0.01 | +0.34 | −0.10 |
| mixer gate, meta | +0.02 | −0.00 | −0.10 | +0.01 | −0.06 |
| MLP read scale, slice | −0.05 | −0.13 | −0.02 | +0.27 | −0.11 |
| MLP read scale, meta | −0.05 | −0.11 | −0.02 | +0.32 | −0.13 |
| MLP gate, slice | +0.01 | −0.04 | +0.04 | +0.37 | −0.17 |
| MLP gate, meta | −0.03 | −0.08 | −0.03 | +0.07 | −0.08 |

## The writers' row scales under the scalar gates

Each writer (`out` of a mixer, `w_down` of the MLP) wrapped as `W x · (1 + s)`, s zero and outside the optimizers:
-∂L/∂s_c is the demand on row c's scale; the step's row-scale component is <ΔW_c, W_c> / |W_c|². Per writer and part,
over channels: the share of rows with a persistent demand (|mean| over 3 standard errors over the 40 steps); the
demand's coherence |Σ_c mean| / Σ_c |mean| (for `w_down`, normed after, the demand sums to zero over the width); the
demand's split-half correlation; the correlation of the step's row-scale component with the demand over the same
steps, and across halves. Ranges over the blocks of each kind; the last block's MLP meta rows, which nothing reads,
are left out.

| Writer, part | persistent | coherence | split-half r | step–demand r, same steps | across halves |
|---|---|---|---|---|---|
| conv mixer, slice | 0.00 – 0.05 | 0.00 – 0.63 | −0.26 – +0.13 | +0.29 – +0.61 | +0.04 – +0.26 |
| conv mixer, meta | 0.00 | 0.00 – 0.11 | −0.27 – −0.09 | +0.39 – +0.59 | +0.05 – +0.15 |
| attention mixer, slice | 0.00 | 0.01 – 0.23 | −0.40 – −0.04 | +0.33 – +0.53 | −0.03 – +0.14 |
| attention mixer, meta | 0.00 | 0.02 – 0.08 | −0.41 – +0.01 | +0.12 – +0.42 | −0.01 – +0.07 |
| MLP, slice | 0.00 – 0.02 | 0.01 – 0.28 | −0.30 – +0.02 | +0.32 – +0.64 | −0.01 – +0.22 |
| MLP, meta | 0.00 | 0.00 – 0.24 | −0.45 – −0.07 | +0.24 – +0.47 | +0.01 – +0.19 |

Every writer's norm is its init norm, 1.000, the bound of its ball.

## Read

- **The tables vary by block, not by t.** The t part B holds 0.01 to 0.24 of a row's mean square, the block part A
  0.05 to 0.86; the gates' B is at most 0.18.
- **The block part is per channel.** One scalar per block and part carries 0.00 to 0.01 of the gates' A and 0.04 to
  0.27 of the read scales'; its first singular pair carries 0.16 to 0.43. Two scalars per write, as the split gates have,
  could hold at most 0.01 of how the tables' gates differ between blocks.
- **The blocks' patterns are their own.** Neighbours share +0.27 to +0.37 in the MLP read scales, the MLP's slice
  gate and the mixer's slice gate; blocks 8 apart or more are near zero or opposite. By kind, only the attention
  blocks' mixer read scales are alike (+0.27, +0.45); attention against conv is −0.15 / −0.24, conv among
  themselves near zero.
- **The step follows each row's scale demand, conv as attention, and no demand persists.** Per step the writers' row
  scales move with the demand (+0.12 to +0.64; the conv mixers +0.29 to +0.61); almost no row's demand persists over
  40 steps (0.00 to 0.05 of rows), the split halves are uncorrelated or opposite, and one half's step against the other half's demand is at
  +0.26 at most. Under the scalar gates no writer's row scale is held off a steady pull.
- **Every writer sits on its norm bound, and so does every tracked matrix.** The global norm ratio in `score.jsonl`
  at 6000 is 1.000 on both held-stream runs, and no matrix exceeds its bound of 1; every writer of the split gates'
  run is at 1.000 (the probe). A writer can move its rows' scales against
  each other but not grow, so a write's size is its gate's; a reading matrix cannot absorb an input gain either, so a
  per-block read gain, which the tables give and the scalar gates' arms lack, has no other place (*Hypothesis*: the
  cost of the tables' removal is the MLP's per-block read gain, its read having no norm after it).
