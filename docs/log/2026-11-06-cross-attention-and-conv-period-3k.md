# Cross attention and conv period at d768x12, 3000 steps (F3, G10, 2026-11-06)

Run log. The dense reading is in FACTS, Architecture at d768x12 (2026-11-06).


Lunara eval caption loss on the same rows in the same order, batch 256, seed 41, usui lr 2e-3, `configs/128_d768x12_*`; images per second is the median of the training log. The baseline and the conv arms train on the old 47-shard set, the cross arm on the 16-shard trial set (798,005 rows); no arm passes over its data once.

| arm | 1000 | 2000 | 3000 | img/s |
|---|---|---|---|---|
| `swpin_3k`, pooled embedding only | 1.5532 | 1.5023 | 1.4855 | 682 |
| `swpin_cross_3k`, cross attention in every block, pooled embedding in the adaLN | 1.5670 | 1.5159 | 1.5024 | 488 |
| `conv2_3k`, conv in blocks 0, 2, 4, … (1/2) | 1.5449 | 1.5003 | 1.4852 | 822 |
| `conv3_3k`, conv in 2/3 of the blocks | 1.5451 | 1.5015 | 1.4859 | 874 |
| `conv4_3k`, conv in 3/4 of the blocks | 1.5474 | 1.5037 | 1.4884 | 901 |

Against the baseline: cross attention +0.014 / +0.014 / +0.017 and 0.72× the speed, no recompile warnings in its log. Conv 1/2 is level (−0.0003 at 3000) at 1.20× the speed, conv 2/3 level (+0.0004) at 1.28×, conv 3/4 +0.003 at 1.32×. Conv has the lower loss at 1000 steps in all three (−0.006 to −0.008) and gives it back by 3000.
