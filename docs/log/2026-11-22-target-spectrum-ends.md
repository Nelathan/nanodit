# The decoder-weighted loss over the plain one at the ends of t, from the data

`scripts/probe_target_spectrum.py EVAL_PARQUET RESOLUTION` on the held-out set at 128 px and Lunara at 256 px, 1000
rows each, CPU, from `7b002a3` plus the script. At t 0 the error is ε's, white; at t 1 with no prompt it is x minus the set's mean latent, the data's
spectrum. At t 0 the ratio is 1 by the weight's mean of 1. Beside it, the base's ratio on the uniform grid (`a4-base-6k`, `score_uniform_128` and `score_uniform_256`,
log 2026-11-22-freq-weight-arms).

| Set | x − mean (t 1) | Base at t 0.05 | Base at t 0.95 | Energy over 3 tokens / 1–3 / inside 1 |
|---|---|---|---|---|
| held-out, 128 px | 2.096 | 1.019 | 1.877 | 0.021 / 0.102 / 0.877 |
| Lunara, 256 px | 1.910 | 1.022 | 1.883 | 0.017 / 0.108 / 0.875 |

## Read

- **The decoder-weighted loss rises with t because the error's spectrum does.** At low t the error is the noise's,
  white, and the weight's mean of 1 leaves it unchanged; at high t it is the image's own structure, whose coarse part,
  about 2 % of the energy, the weight multiplies most. The base follows that from 1.02 to 1.88, short of the data's
  limit.
- **The prompt removes coarse error first** (*Hypothesis*): at t 0.95 the base's plain loss, 1.077, is 2 % under the
  mean guess's 1.099 (0.9969 · 1.05²), its decoder-weighted loss, 2.022, 12 % under 2.304.
- **The limits differ with the sets' coarse energy,** 0.021 of it at 128 px held-out against 0.017 at 256 px Lunara,
  the band the weight multiplies most; set and size are confounded.
