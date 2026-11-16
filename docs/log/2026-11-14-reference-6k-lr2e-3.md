# The reference at usui lr 2e-3 constant (2026-11-14)

At commit `7f3b8e2`, the reference `configs/128_d768x12_6k.toml` with usui lr 2e-3 instead of 4e-3 (the run
directory's `config.toml`), everything else the same (merged set `set_128`, seed 41, constant lr after 376 warmup steps,
fallback lr 2.5e-3). Run directory `/mnt/mars/ai/tiny-dit/runs/128_d768x12_6k_lr2e-3`.

Caption eval loss on the Lunara eval set (`score.jsonl`):

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| `128_d768x12_6k` (4e-3) | 1.5499 | 1.5190 | 1.5056 | 1.4985 | 1.4956 | 1.4905 |
| `128_d768x12_6k_lr2e-3` | 1.5622 | 1.5159 | 1.4997 | 1.4903 | 1.4840 | 1.4792 |

Median img/s: 651 and 664.

Read: one seed each, a difference below about 0.003 is not resolved. 4e-3 leads at 1000 steps by 0.012; 2e-3 is ahead
from 2000 on and by 0.0113 at 6000; the gap grows to 0.0116 at 5000 and holds. The 4e-3 of PLAN C2 came from 3000 steps at d768x16 with a
decaying lr; at 6000 constant steps at d768x12 the lower lr wins, and the curve does not say that 2e-3 is the optimum.
