# a1-attnfirst-6k: the attention block first in each period of four

Run: `/mnt/mars/ai/tiny-dit/runs/a1-attnfirst-6k` (its `config.toml`; config only, on the code of `9da2709`):
`a1-ref-1ep` with `layout = ["attn", "conv", "conv", "conv"]` for 6000 steps, so attention in blocks 0, 4, …, 20
instead of 3, 7, …, 23. Against `a1-ref-1ep` at the same step. One seed each.

| Step | Score | Δ |
|---|---|---|
| 1000 | 1.5575 | +0.0013 |
| 2000 | 1.5181 | −0.0004 |
| 3000 | 1.4960 | −0.0022 |
| 4000 | 1.4839 | −0.0017 |
| 5000 | 1.4749 | −0.0020 |
| 6000 | 1.4687 | −0.0007 |

Δ per t at 6000, the score's 10 t (0.16 … 0.84): +0.0002, −0.0014, −0.0018, −0.0017, −0.0012, −0.0009, −0.0004,
+0.0003, +0.0001, +0.0000.

Throughput: 588 img/s against 581 for the reference over its first 6000 steps (means of `imgs_per_s` after the first
five records).

## Read

Level with the reference: the largest gap, 0.0022, is below the 0.003 taken as one seed's resolution, which
`a1-seed42-6k` measures next.
