# Every Usui matrix framed on its input side ("xside") (2026-11-15)

Branch `arm/xside` (commits `6781534`, `dfd3b1e`): `side = "right"` on both Usui groups, so `out`, `w_down`,
`patch_embed`, `cross_out` and `cross_kv` move from the output to the input side; the other matrices were on the input
side already under `"auto"`. Otherwise the reference: d768x12, merged set `set_128`, seed 41, constant lr after 376
warmup steps, 6000 steps. Each arm against the reference at the same usui lr. Run directories
`/mnt/mars/ai/tiny-dit/runs/128_d768x12_6k_xside` and `128_d768x12_6k_xside_lr2e-3`.

Caption eval loss on the Lunara eval set (`score.jsonl`):

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| `128_d768x12_6k` (4e-3) | 1.5499 | 1.5190 | 1.5056 | 1.4985 | 1.4956 | 1.4905 |
| `128_d768x12_6k_xside` (4e-3) | 1.5581 | 1.5184 | 1.5004 | 1.4910 | 1.4853 | 1.4816 |
| `128_d768x12_6k_lr2e-3` | 1.5622 | 1.5159 | 1.4997 | 1.4903 | 1.4840 | 1.4792 |
| `128_d768x12_6k_xside_lr2e-3` | 1.5732 | 1.5237 | 1.5033 | 1.4905 | 1.4827 | 1.4781 |

Usui diagnostics (`train.jsonl`, the reads `pop_diagnostics` returns), records 2–21 / the last 20:

| Run | `grad_capture` | `tangent_live_fraction` |
|---|---|---|
| 4e-3 | 0.993 / 1.004 | 0.979 / 0.967 |
| xside 4e-3 | 0.986 / 0.959 | 0.991 / 0.983 |
| 2e-3 | 0.988 / 0.994 | 0.991 / 0.978 |
| xside 2e-3 | 0.984 / 0.947 | 0.993 / 0.990 |

Median img/s 651–664 in all four.

Read: one seed each, a difference below about 0.003 is not resolved. At 4e-3 xside ends 0.0089 ahead; at 2e-3 it
ends 0.0011 ahead, level, after trailing by 0.011 at 1000 steps. *Hypothesis*: the gain at 4e-3 is the input side
tolerating an lr that is too hot for the output side; at the better lr the side makes no resolved difference.
