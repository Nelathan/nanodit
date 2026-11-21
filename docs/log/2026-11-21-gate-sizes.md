# The adaLN gates' size in a4 after one pass

`scripts/probe_gates.py` on `a4-ref-1ep` step 32173 and `a4-ref-1ep-anneal6k` step 38173: per block, the gate rows of
`mod + table` (before any bound) at the score's 10 t. Columns: RMS over channels and t, the largest |gate|, the share
of channels and t with |gate| > 1, for the mixer-branch gate and the MLP-branch gate.

| Block | Mixer gate, step 32173 | MLP gate, step 32173 |
|---|---|---|
| 0 conv | 0.358 / 1.17 / 0.002 | 0.240 / 1.78 / 0.003 |
| 1 conv | 0.306 / 1.41 / 0.004 | 0.342 / 1.28 / 0.004 |
| 2 attn | 0.360 / 1.93 / 0.012 | 0.315 / 1.38 / 0.001 |
| 3 conv | 0.375 / 1.23 / 0.005 | 0.411 / 1.61 / 0.010 |
| 4 conv | 0.447 / 1.34 / 0.017 | 0.458 / 1.46 / 0.010 |
| 5 conv | 0.359 / 1.47 / 0.008 | 0.449 / 1.45 / 0.013 |
| 6 attn | 0.380 / 1.66 / 0.024 | 0.447 / 1.53 / 0.016 |
| 7 conv | 0.354 / 1.62 / 0.018 | 0.430 / 1.49 / 0.009 |
| 8 conv | 0.334 / 2.00 / 0.021 | 0.547 / 1.48 / 0.038 |
| 9 conv | 0.386 / 2.08 / 0.037 | 0.510 / 2.21 / 0.035 |
| 10 attn | 0.603 / 2.18 / 0.107 | 0.580 / 2.11 / 0.074 |
| 11 conv | 0.484 / 3.05 / 0.040 | 0.536 / 1.93 / 0.046 |
| 12 conv | 0.501 / 3.16 / 0.049 | 0.563 / 1.98 / 0.064 |
| 13 conv | 0.512 / 3.18 / 0.053 | 0.577 / 1.69 / 0.080 |
| 14 attn | 0.915 / 4.68 / 0.243 | 0.636 / 2.10 / 0.080 |
| 15 conv | 0.607 / 2.89 / 0.109 | 0.683 / 2.06 / 0.125 |
| 16 conv | 0.603 / 2.40 / 0.067 | 0.712 / 2.27 / 0.154 |
| 17 conv | 0.558 / 2.08 / 0.070 | 0.797 / 2.75 / 0.196 |
| 18 attn | 1.026 / 3.41 / 0.329 | 0.968 / 2.43 / 0.362 |
| 19 conv | 0.676 / 2.10 / 0.137 | 0.899 / 2.61 / 0.278 |
| 20 conv | 0.951 / 3.19 / 0.295 | 1.169 / 3.61 / 0.446 |
| 21 conv | 0.897 / 3.38 / 0.240 | 1.075 / 4.25 / 0.276 |
| 22 attn | 2.353 / 5.27 / 0.771 | 1.618 / 3.84 / 0.610 |
| 23 conv | 2.035 / 6.59 / 0.726 | 0.365 / 1.41 / 0.004 |

After the anneal (step 38173) every block's gates are a little larger: RMS up 4 to 10 %, e.g. block 22's mixer gate
2.545 / 5.70 / 0.780.

The split, step 32173 (the same probe): the shared projection's gate rows
have RMS 0.226 (mixer) and 0.316 (MLP) over channels and t; the per-block tables' gate rows grow with depth, 0.29 /
0.22 at block 2, 0.50 / 0.42 at block 10, 0.94 / 0.81 at block 18, 2.24 / 1.45 at block 22.

## Read

- The gates grow with depth and keep growing with training: in the last attention block three quarters of its
  mixer gates exceed 1. The growth is in the per-block tables, which StochasticAdamW trains outside any norm ceiling;
  the shared projection, inside Usui's ball, stays small. With every matrix held in its init-norm ball, the tables are
  the one unbounded amplitude on a branch's write, and the deep blocks use it.
- A tanh on the gates is therefore not near the identity here: it caps exactly this growth. The tanh arm tests
  whether the deep blocks need writes larger than their bounded matrices give, not only a guard against outliers.
