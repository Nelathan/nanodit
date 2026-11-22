# a4 at 6000 steps: the split read, the held meta stream, long skips, no tables

Runs (each `config.toml` in its run directory; a4 on the arm's code, seed 41, 6000 steps), against `a4-base-6k` (main,
the first 6000 steps of a4-ref-1ep, which it reproduces: 1.3070 held-out):

| Run | Code | Change |
|---|---|---|
| `a4-split-read-6k` | `arm/split-read` `4ec5c48` | a branch reads the slice raw and the meta channels normed on their own |
| `a4-held-meta-6k` | `arm/held-meta` `5a5409e` | on it, the meta channels normed back to unit RMS after every write |
| `a4-held-meta-skips-6k` | `arm/held-meta-skips` `d178c97` | on it, long skips: the meta channels after block i added before block 23 − i, by a per-channel weight from zero |
| `a4-no-tables-6k` | `arm/no-tables` `384dc22` | the per-block adaLN tables removed: every block gets the shared projection's scales and gates |
| `a4-held-meta-skips-linear-skip-6k` | `arm/held-meta-skips-linear-skip` `e271f72` | held meta + long skips + the linear skip (log 2026-11-22-skip-arms) |

The uniform-grid scores (t 0.05 … 0.95, evenly spaced) of all but the last run are backfilled post hoc by
`scripts/score_uniform.py` from the run's code with `9e7f3f7` cherry-picked; on the base the post-hoc score equals the
native one (1.2571). The seed spread at 6000 steps is about 0.001; a difference under about 0.002 is not resolved
(log 2026-11-18-seed42).

## Score at 6000


| Run | Held-out | Δ | Lunara | Δ | Held-out uniform | Δ | Lunara uniform | Δ |
|---|---|---|---|---|---|---|---|---|
| base | 1.3070 | | 1.3608 | | 1.2571 | | 1.2971 | |
| split read | 1.3051 | −0.0018 | 1.3576 | −0.0032 | 1.2562 | −0.0009 | 1.2950 | −0.0022 |
| held meta | 1.3029 | −0.0040 | 1.3570 | −0.0039 | 1.2536 | −0.0035 | 1.2935 | −0.0036 |
| held meta + long skips | 1.2999 | −0.0071 | 1.3544 | −0.0065 | 1.2520 | −0.0051 | 1.2922 | −0.0050 |
| no tables | 1.3221 | +0.0151 | 1.3747 | +0.0139 | 1.2691 | +0.0120 | 1.3079 | +0.0108 |
| held meta + long skips + linear skip | 1.2974 | −0.0095 | 1.3513 | −0.0095 | 1.2481 | −0.0090 | 1.2880 | −0.0091 |

Uniform grid, held-out: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1171 | 1.2121 | 1.2582 | 1.2857 | 1.3129 | 1.3448 | 1.3705 | 1.3516 | 1.2416 | 1.0767 |
| split read | +0.0057 | −0.0001 | −0.0010 | −0.0017 | −0.0023 | −0.0028 | −0.0025 | −0.0014 | +0.0003 | −0.0031 |
| held meta | +0.0010 | −0.0013 | −0.0030 | −0.0045 | −0.0053 | −0.0057 | −0.0051 | −0.0034 | −0.0018 | −0.0061 |
| held meta + long skips | +0.0019 | −0.0026 | −0.0052 | −0.0076 | −0.0091 | −0.0096 | −0.0086 | −0.0058 | −0.0033 | −0.0013 |
| no tables | +0.0050 | +0.0108 | +0.0196 | +0.0248 | +0.0234 | +0.0174 | +0.0101 | +0.0052 | +0.0001 | +0.0032 |
| held meta + long skips + linear skip | −0.0175 | −0.0094 | −0.0104 | −0.0114 | −0.0116 | −0.0111 | −0.0095 | −0.0062 | −0.0026 | −0.0007 |

Uniform grid, Lunara: Δ loss by t against the base, whose row is its loss.

| t | 0.05 | 0.15 | 0.25 | 0.35 | 0.45 | 0.55 | 0.65 | 0.75 | 0.85 | 0.95 |
|---|---|---|---|---|---|---|---|---|---|---|
| base | 1.1214 | 1.2505 | 1.3206 | 1.3549 | 1.3782 | 1.4031 | 1.4226 | 1.3936 | 1.2561 | 1.0699 |
| split read | +0.0041 | −0.0006 | −0.0014 | −0.0021 | −0.0026 | −0.0033 | −0.0042 | −0.0052 | −0.0039 | −0.0022 |
| held meta | −0.0001 | −0.0016 | −0.0029 | −0.0043 | −0.0050 | −0.0051 | −0.0039 | −0.0023 | −0.0021 | −0.0086 |
| held meta + long skips | +0.0006 | −0.0028 | −0.0050 | −0.0074 | −0.0088 | −0.0090 | −0.0068 | −0.0043 | −0.0027 | −0.0034 |
| no tables | +0.0041 | +0.0086 | +0.0181 | +0.0236 | +0.0218 | +0.0163 | +0.0088 | +0.0055 | +0.0008 | +0.0000 |
| held meta + long skips + linear skip | −0.0183 | −0.0100 | −0.0102 | −0.0114 | −0.0122 | −0.0119 | −0.0092 | −0.0028 | −0.0019 | −0.0033 |

## Read

- **The held stream is the gain.** The split read takes −0.0009 / −0.0022 on the uniform grid (held-out / Lunara),
  holding the meta channels on it −0.0035 / −0.0036. The gain is largest at t 0.35–0.65 and in the single cell at
  t 0.95, and absent at t 0.05.
- **Long skips on the held stream** move the uniform score by a further −0.0016 / −0.0014, not resolved; by t they
  gain −0.003 to −0.004 at t 0.35–0.65 on both sets and give back most of what holding gained at t 0.95.
- **The linear skip adds on top, at low t,** and the sum is close to additive: −0.0090 against −0.0051 and −0.0036
  apart. The two act on different t.
- **Removing the tables costs +0.012 on the old stream,** worst at t 0.25–0.45. That arm also removes every per-block
  part of the gates (in its code every block reads the shared projection's gates), so it measures the loss of
  per-block gating, not of the tables as such.
- **Samples (read by eye, the eval grids at 6000):** held meta alone is somewhat over contrasted and sharpened; with
  long skips and the linear skip, composition, prompt following and vibrancy are better and texture is weaker.
