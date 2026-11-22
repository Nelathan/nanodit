# a4 at 6000 steps: conv kernels by depth

Runs (each `config.toml` in its run directory; a4 on `arm/conv-kernel` `4929af7`, where the block type names its conv
kernel, seed 41, 6000 steps), against `a4-base-6k` (main, every conv 3×3; it reproduces a4-ref-1ep's first 6000
steps). Six attention blocks of 24 in every layout.

| Run | Layout |
|---|---|
| `a4-conv-graded-6k` | a4's layout with the convs 7×7 in the first third, 5×5 in the middle, 3×3 in the last |
| `a4-conv-graded-soft-6k` | the same, 5×5 in the first third, 3×3 after |
| `a4-conv-phased-6k` | three (attention, 7×7), twelve 3×3, three (attention, 5×5) |
| `a4-conv-period-6k` | (attention, 7×7, 5×5, 3×3) repeated |

The uniform-grid scores are backfilled by `scripts/score_uniform.py` from `4929af7` with `9e7f3f7` cherry-picked. The
seed spread is about 0.001; under about 0.002 is not resolved (log 2026-11-18-seed42).

## Score at 6000


| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| graded 7/5/3 | 1.3053 | −0.0017 | 1.3602 | −0.0006 | 1.2559 | −0.0012 | 1.2964 | −0.0007 |
| graded-soft 5/3/3 | 1.3077 | +0.0007 | 1.3606 | −0.0002 | 1.2574 | +0.0003 | 1.2964 | −0.0008 |
| phased | 1.3093 | +0.0023 | 1.3640 | +0.0032 | 1.2584 | +0.0013 | 1.2993 | +0.0022 |
| period | 1.3048 | −0.0022 | 1.3578 | −0.0030 | 1.2556 | −0.0016 | 1.2947 | −0.0024 |

Uniform grid, held-out: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1171 | 1.2121 | 1.2582 | 1.2857 | 1.3129 | 1.3448 | 1.3705 | 1.3516 | 1.2416 | 1.0767 |
| graded 7/5/3 | −0.0010 | −0.0008 | −0.0016 | −0.0017 | −0.0023 | −0.0020 | −0.0017 | −0.0017 | −0.0007 | +0.0016 |
| graded-soft 5/3/3 | −0.0031 | +0.0007 | +0.0010 | +0.0012 | +0.0013 | +0.0013 | +0.0011 | +0.0005 | −0.0009 | −0.0001 |
| phased | −0.0034 | +0.0013 | +0.0030 | +0.0042 | +0.0040 | +0.0030 | +0.0016 | +0.0001 | −0.0008 | −0.0002 |
| period | −0.0039 | −0.0002 | −0.0010 | −0.0018 | −0.0027 | −0.0030 | −0.0028 | −0.0025 | −0.0016 | +0.0041 |

Uniform grid, Lunara: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1214 | 1.2505 | 1.3206 | 1.3549 | 1.3782 | 1.4031 | 1.4226 | 1.3936 | 1.2561 | 1.0699 |
| graded 7/5/3 | −0.0020 | −0.0015 | −0.0020 | −0.0016 | −0.0015 | +0.0008 | +0.0020 | +0.0008 | −0.0013 | −0.0008 |
| graded-soft 5/3/3 | −0.0034 | +0.0000 | +0.0004 | +0.0004 | +0.0002 | −0.0001 | −0.0002 | +0.0009 | −0.0005 | −0.0053 |
| phased | −0.0032 | +0.0004 | +0.0022 | +0.0034 | +0.0036 | +0.0036 | +0.0036 | +0.0056 | +0.0038 | −0.0010 |
| period | −0.0040 | −0.0014 | −0.0022 | −0.0031 | −0.0037 | −0.0035 | −0.0029 | −0.0024 | −0.0018 | +0.0009 |

## Read

- **The period gains and phased costs, each past the spread on three of the four scores:** the period −0.0022 /
  −0.0030 on the old grid and −0.0024 on Lunara uniform (−0.0016 held-out uniform); phased +0.0023 / +0.0032 and
  +0.0022 on Lunara uniform (+0.0013 held-out uniform), its cost in mid-t. The graded layouts are within the spread.
  *Hypothesis*: phased pays for twelve 3×3 blocks with no attention in the middle.
- **Every layout with a larger kernel gains at t 0.05,** −0.001 to −0.004. The period loses +0.0041 at t 0.95 on
  held-out, a single cell.
- **Samples (read by eye):** of the single-kernel arms (log 2026-11-21-conv-kernel-and-tanh-gates), 3×3 is the
  sharpest, 5×5 the best composed, 7×7 muddied and grey; of the layouts here, phased looks good and the rest do not
  separate.
