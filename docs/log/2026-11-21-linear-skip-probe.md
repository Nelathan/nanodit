# How much of a4's correction is a rescale of its input

`scripts/probe_linear_skip.py /mnt/mars/ai/tiny-dit/runs/a4-ref-1ep-anneal6k 38173 32`: the annealed one-pass
reference, the first 32 held-out rows with the score's noise, at the score's 10 t; CPU, fp32, on `main` `751616f`.
The latents' mean square over these rows is 1.114.

The correction is what the trunk writes on the slice, F − x_t; the true correction is T − x_t. Slope: either one
regressed on x_t, pooled over rows, tokens and channels; the true slope is c(t) − 1 for these latents. Linear
share: the slope's part of the correction's energy. Loss: the score's loss; linear, the loss of c(t)·x_t with the
true slope; identity, the loss of F = x_t (the model at init).

| t | slope true | slope model | linear share true | linear share model | loss model | linear | identity |
|---|---|---|---|---|---|---|---|
| 0.16 | +0.950 | +0.949 | 0.346 | 0.775 | 1.1792 | 1.3811 | 2.1128 |
| 0.26 | +0.825 | +0.828 | 0.218 | 0.510 | 1.2026 | 1.6521 | 2.1128 |
| 0.34 | +0.659 | +0.663 | 0.124 | 0.298 | 1.2181 | 1.8500 | 2.1128 |
| 0.40 | +0.457 | +0.461 | 0.055 | 0.137 | 1.2376 | 1.9956 | 2.1128 |
| 0.47 | +0.227 | +0.232 | 0.013 | 0.034 | 1.2624 | 2.0852 | 2.1128 |
| 0.53 | −0.021 | −0.015 | 0.000 | 0.000 | 1.2918 | 2.1126 | 2.1128 |
| 0.60 | −0.270 | −0.264 | 0.019 | 0.048 | 1.3235 | 2.0734 | 2.1128 |
| 0.66 | −0.507 | −0.500 | 0.069 | 0.186 | 1.3455 | 1.9669 | 2.1128 |
| 0.74 | −0.717 | −0.715 | 0.152 | 0.415 | 1.3390 | 1.7915 | 2.1128 |
| 0.84 | −0.897 | −0.896 | 0.280 | 0.690 | 1.2548 | 1.5216 | 2.1128 |

## Read

- The trunk learns the linear rescale exactly: its slope is the true slope to 0.007 at every t.
- At the ends the rescale is most of what it writes: 0.78 of its correction's energy at t 0.16, 0.69 at t 0.84,
  against 0.35 and 0.28 of the true correction's (the trunk does not write the part it cannot predict). At the middle
  it writes none.
- The loss peaks at t 0.66 to 0.74, later than the linear predictor's peak at t 0.53, where x_t is orthogonal to the
  velocity. The ratio of the model's loss to the linear loss is lowest, about 0.61, at t 0.47 to 0.53, and rises
  toward both ends.
- With unit variance assumed, c(t) − 1 differs from the true slope by at most 0.03 here.
- 32 rows, one draw. The loss column is the score's loss on a subset; the score itself (`score.jsonl` at 38173, all
  held-out rows) peaks at the same t, 0.74: 1.1646 at t 0.16, 1.2464 at 0.53, 1.3091 at 0.74, 1.2376 at 0.84.
