# The score drops the null embedding

2026-11-05. The checkpoint score (archive 2026-11-02, checkpoint score) also scored every eval row with the null
embedding. Across every run that gap stayed between 0.002 and 0.004 while the samples showed the caption in use, so it
measured nothing about prompt following. Decided: the null eval is dropped.

The score is the caption condition only, with its loss and R² per t; SPEC §8 states it. Records keep the
`loss_caption` keys, so old and new `score.jsonl` files read the same.
