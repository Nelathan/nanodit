# a4 at 6000 steps: the conv period's phase

Run (`config.toml` in its run directory; a4 on `arm/conv-kernel` `54d476e`, seed 41, 6000 steps), against
`a4-conv-period-6k` (`4929af7`, log 2026-11-22-conv-kernel-layouts) and `a4-base-6k` (main, every conv 3×3):

| Run | Layout |
|---|---|
| `a4-conv-period-6k` | (attention, 7×7, 5×5, 3×3) repeated: attention at blocks 0, 4, …, 20, the last block a 3×3 |
| `a4-conv-period-3first-6k` | (3×3, attention, 7×7, 5×5) repeated: the same cycle a block later, attention at blocks 1, 5, …, 21, the last block a 5×5 |

The uniform-grid scores of both are backfilled by `scripts/score_uniform.py` with `9e7f3f7` cherry-picked. The seed
spread is about 0.001; under about 0.002 is not resolved (log 2026-11-18-seed42).

## Score at 6000

| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| period | 1.3048 | −0.0022 | 1.3578 | −0.0030 | 1.2556 | −0.0016 | 1.2947 | −0.0024 |
| period, 3×3 first | 1.3069 | −0.0001 | 1.3607 | −0.0001 | 1.2572 | +0.0001 | 1.2971 | −0.0000 |

Uniform grid, Δ loss by t against the base (its loss in log 2026-11-22-conv-kernel-layouts).

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| period, held-out | −0.0039 | −0.0002 | −0.0010 | −0.0018 | −0.0027 | −0.0030 | −0.0028 | −0.0025 | −0.0016 | +0.0041 |
| 3×3 first, held-out | −0.0024 | +0.0015 | +0.0021 | +0.0019 | +0.0007 | −0.0010 | −0.0017 | −0.0020 | −0.0010 | +0.0031 |
| period, Lunara | −0.0040 | −0.0014 | −0.0022 | −0.0031 | −0.0037 | −0.0035 | −0.0029 | −0.0024 | −0.0018 | +0.0009 |
| 3×3 first, Lunara | −0.0027 | +0.0007 | +0.0016 | +0.0014 | +0.0000 | −0.0014 | −0.0014 | +0.0005 | −0.0005 | +0.0014 |

## Read

- **The phase takes the period's gain away.** The same cycle one block later is level with the base on all four
  scores, 0.0017 to 0.0029 behind the period, past the spread on three of four (held-out uniform not resolved).
- **The loss is at low and mid t.** From t 0.15 to 0.45 the rotated layout is 0.002 to 0.005 behind the period on
  both sets; from t 0.55 the two are within about 0.002 on held-out, apart at t 0.75 on Lunara (0.003).
- **Which end of the stack carries it is not separated.** The rotation moves attention off block 0, where it reads
  the stream before any conv has mixed neighbours, and puts a 5×5 instead of a 3×3 last, before the readout; this run changes both.
