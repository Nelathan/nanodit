# a1-attnthird-6k: the attention block third in each period of four

Run: `/mnt/mars/ai/tiny-dit/runs/a1-attnthird-6k` (its `config.toml`; config only, on `40b375c`): `a1-ref-1ep` with
`layout = ["conv", "conv", "attn", "conv"]` for 6000 steps, so attention in blocks 2, 6, …, 22. Against `a1-ref-1ep`
(seed 41) and `a1-seed42-6k`; the two references differ by 0.0009 at 6000 (log 2026-11-18-seed42).

| Step | Score | Δ ref | Δ seed 42 |
|---|---|---|---|
| 1000 | 1.5553 | −0.0009 | −0.0002 |
| 2000 | 1.5165 | −0.0020 | −0.0007 |
| 3000 | 1.4959 | −0.0023 | −0.0014 |
| 4000 | 1.4846 | −0.0011 | −0.0001 |
| 5000 | 1.4754 | −0.0015 | −0.0005 |
| 6000 | 1.4683 | −0.0010 | −0.0001 |

Band R² at 6000 (over three tokens, one to three, inside one), Lunara: 0.808, 0.707, 0.614. Throughput 595 img/s
(median `imgs_per_s` after the first five records).

## Read

Level with both references. With attention first (log 2026-11-18-attnfirst) and last, every position of four ends
within the seed noise at 6000 steps on the basic conv.
