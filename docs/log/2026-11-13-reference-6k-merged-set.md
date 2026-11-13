# The reference on the merged training set (2026-11-13)

`configs/128_d768x12_6k.toml` at commit `a5d31f8`, the config of log 2026-11-13-reference-6k-constant-lr, trained on
the merged set `set_128` (text-to-image-2M and all five Fine-T2I subsets, deduplicated, one Parquet per aspect bucket;
`scripts/shuffle.py`). 6000 steps of batch 256, seed 41, a bucket per micro-batch, 40 minutes at about 650 images/s.
Run directory `/mnt/mars/ai/tiny-dit/runs/128_d768x12_6k`.

The set holds 8,239,111 rows; the build dropped 365,799 as repeated images. Rows per bucket:

| Bucket (h x w) | 128x128 | 112x144 | 144x112 | 96x176 | 176x96 | 112x160 | 160x112 |
|---|---|---|---|---|---|---|---|
| Rows | 5,556,652 | 748,368 | 702,803 | 342,889 | 291,766 | 312,746 | 283,887 |

Caption eval loss on the Lunara eval set (`score.jsonl`), against the same config on the trial set
(`128_d768x12_6k_trialset`):

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| `128_d768x12_6k_trialset` | 1.5468 | 1.5145 | 1.5038 | 1.4985 | 1.4923 | 1.4867 |
| `128_d768x12_6k` | 1.5499 | 1.5190 | 1.5056 | 1.4985 | 1.4956 | 1.4905 |

R² at 6000: 0.6209. No empty-caption warning in the encode log; no recompile limit hit.

Read: 0.0038 behind at 6000 steps, level at 4000. One seed resolves about 0.003, so the difference is at the edge of
what the comparison can say. The two runs differ in data (the trial set was text-to-image-2M shards 0–15, 128 px
square only; the merged set adds the Fine-T2I buckets and the other shards) and in batch composition (a bucket per
micro-batch), so no cause is attributed. This run is the baseline for the arms that follow.
