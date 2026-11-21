# a4 at 6000 steps: a 5×5 and a 7×7 conv kernel, tanh on the gates

Runs (each `config.toml` in its run directory, all `a4-base-6k` on the arm's code), against `a4-ref-1ep` at the same
steps (the same seed, 41; the first 6000 steps of its pass, `usui_lr` the same, the schedule constant over them):

| Run | Code | Change |
|---|---|---|
| `a4-conv5-6k` | `arm/conv5` `b1986b3` | conv kernel 5×5, padding 2 |
| `a4-conv7-6k` | `arm/conv7` `250185d` | conv kernel 7×7, padding 3 |
| `a4-tanh-gates-6k` | `arm/tanh-gates` `b65251a` | both adaLN gates through a tanh after `mod + table` |

The seed spread at 6000 steps is one measured draw of about 0.001, and a difference under about 0.002 is not resolved
(log 2026-11-18-seed42).

## Score

Held-out loss, then Δ against the reference.

| Step | ref | 5×5 | Δ | 7×7 | Δ | tanh | Δ |
|---|---|---|---|---|---|---|---|
| 1000 | 1.4057 | 1.4083 | +0.0026 | 1.4083 | +0.0026 | 1.4055 | −0.0003 |
| 2000 | 1.3601 | 1.3597 | −0.0004 | 1.3607 | +0.0007 | 1.3596 | −0.0005 |
| 3000 | 1.3377 | 1.3377 | −0.0001 | 1.3387 | +0.0010 | 1.3376 | −0.0001 |
| 4000 | 1.3245 | 1.3234 | −0.0011 | 1.3244 | −0.0000 | 1.3242 | −0.0003 |
| 5000 | 1.3144 | 1.3129 | −0.0015 | 1.3143 | −0.0002 | 1.3149 | +0.0005 |
| 6000 | 1.3070 | 1.3054 | −0.0015 | 1.3073 | +0.0003 | 1.3067 | −0.0003 |

Lunara at 256 px (the larger grid).

| Step | ref | 5×5 | Δ | 7×7 | Δ | tanh | Δ |
|---|---|---|---|---|---|---|---|
| 1000 | 1.4501 | 1.4519 | +0.0018 | 1.4530 | +0.0029 | 1.4500 | −0.0001 |
| 2000 | 1.4087 | 1.4091 | +0.0003 | 1.4081 | −0.0007 | 1.4076 | −0.0011 |
| 3000 | 1.3878 | 1.3881 | +0.0003 | 1.3884 | +0.0006 | 1.3879 | +0.0001 |
| 4000 | 1.3772 | 1.3763 | −0.0009 | 1.3783 | +0.0010 | 1.3768 | −0.0005 |
| 5000 | 1.3668 | 1.3666 | −0.0002 | 1.3682 | +0.0014 | 1.3684 | +0.0016 |
| 6000 | 1.3607 | 1.3592 | −0.0015 | 1.3617 | +0.0010 | 1.3601 | −0.0006 |

R² at 6000, held-out / Lunara: ref 0.6544 / 0.6394, 5×5 0.6549 / 0.6397, 7×7 0.6544 / 0.6389, tanh 0.6545 / 0.6396.

Δ by t at 6000, the score's 10 t (0.16 … 0.84):

| Arm | Held-out | Lunara |
|---|---|---|
| 5×5 | −0.0004 −0.0009 −0.0014 −0.0016 −0.0017 −0.0016 −0.0020 −0.0026 −0.0018 −0.0014 | −0.0013 −0.0017 −0.0021 −0.0023 −0.0021 −0.0015 −0.0008 −0.0007 +0.0001 −0.0023 |
| 7×7 | +0.0004 +0.0008 +0.0009 +0.0007 +0.0007 +0.0007 +0.0002 −0.0007 −0.0001 −0.0003 | −0.0008 −0.0001 +0.0002 −0.0002 +0.0002 +0.0007 +0.0019 +0.0023 +0.0050 +0.0012 |
| tanh | +0.0001 −0.0000 −0.0001 −0.0004 −0.0005 −0.0006 −0.0009 −0.0011 −0.0004 +0.0009 | −0.0002 −0.0000 +0.0001 −0.0002 −0.0001 −0.0001 −0.0008 −0.0021 −0.0013 −0.0011 |

R² by band at t 0.84, Δ on the three bands (coarse to fine): 7×7 −0.0053 / +0.0003 / +0.0003 held-out and
−0.0064 / −0.0026 / −0.0004 on Lunara; 5×5 and tanh within ±0.0042 held-out and ±0.0018 on Lunara; at t 0.47 and
0.16 every arm within ±0.0006.

## Speed

`scripts/bench_step.py configs/a4-base-6k.toml` on the Lunara rows, on the trunk's and the two conv arms' code, batch 256; the trunk at
`191bed5`.

| Code | 128 px (4 × 64) | 256 px (16 × 16) | Peak |
|---|---|---|---|
| 3×3, trunk | 455.4 ms, 562.1 img/s | 1648.5 ms, 155.3 img/s | 4.3 / 4.2 GiB |
| 5×5 | 459.7 ms, 556.9 img/s | 1660.2 ms, 154.2 img/s | 4.3 / 4.2 GiB |
| 7×7 | 467.1 ms, 548.1 img/s | 1684.7 ms, 152.0 img/s | 4.3 / 4.2 GiB |

In training at 128 px, median `imgs_per_s` over the run: ref (its first 6000 steps) 521, 5×5 517, 7×7 509, tanh 522.

## The tanh arm's gates and stream

`scripts/probe_gates.py` and `scripts/probe_stream.py RUN 6000 32` on the arm's code at step 6000; the 5×5 arm
at step 6000 beside it, as the run without a tanh at the same length (it differs in the kernel too). The gates are
read before the tanh.

Share of gate values with |g| > 1, the blocks where it is above 0.001:

| Block | tanh: mixer / MLP | 5×5: mixer / MLP |
|---|---|---|
| 10 attn | 0.012 / 0 | 0.001 / 0 |
| 14 attn | 0.011 / 0 | 0.003 / 0 |
| 18 attn | 0.006 / 0.000 | 0.007 / 0.000 |
| 20 conv | 0 / 0.010 | 0 / 0.000 |
| 21 conv | 0 / 0.013 | 0 / 0.024 |
| 22 attn | 0.123 / 0.070 | 0.095 / 0.045 |
| 23 conv | 0.008 / 0 | 0.012 / 0 |

Gate RMS on the meta channels at block 22, mixer / MLP: tanh 0.804 / 0.660, 5×5 0.732 / 0.610; after the full pass
2.576 / 1.763 (log 2026-11-21-stream-scale).

The stream, slice / meta RMS:

| After | tanh, t 0.16 / 0.47 / 0.84 | 5×5, t 0.16 / 0.47 / 0.84 |
|---|---|---|
| embed | 0.90 / 0.98, 0.73 / 0.79, 0.86 / 0.88 | 0.90 / 0.97, 0.73 / 0.78, 0.86 / 0.88 |
| 10 attn | 0.99 / 1.36, 0.76 / 1.20, 0.71 / 1.63 | 1.00 / 1.33, 0.77 / 1.18, 0.71 / 1.60 |
| 18 attn | 1.44 / 1.92, 0.98 / 1.77, 0.45 / 2.17 | 1.46 / 1.98, 0.99 / 1.79, 0.45 / 2.21 |
| 22 attn | 1.71 / 2.80, 1.17 / 2.64, 0.48 / 2.72 | 1.70 / 3.02, 1.17 / 2.78, 0.49 / 2.92 |
| 23 conv | 1.80 / 2.88, 1.24 / 2.70, 0.51 / 2.67 | 1.80 / 3.07, 1.24 / 2.82, 0.52 / 2.90 |

After the full pass, after block 22, the meta part is 9.54 / 8.76 / 8.23 (log 2026-11-21-stream-scale).

## Read

- **5×5:** −0.0015 at 6000 on both sets; behind at 1000 on both, ahead from 2000 on held-out and from 4000 on Lunara; on held-out ahead at every t, on Lunara
  at all but one. Inside the seed spread, and the seed draw's own shape is also a mid-t offset of
  the same size, so the comparison does not resolve it. Its cost is 1 % of the step at both sizes.
- **7×7:** level on held-out, behind on Lunara at high t, where it loses coarse-band R² at t 0.84 on both sets; 2 to
  3 % slower. Nothing for it.
- **tanh:** level on both sets at every eval. At 6000 steps few gates reach 1 (at most 12 % of one gate's values, at
  block 22), so the bound has little to bend; the stream at 6000 is within 10 % of the run without it, and a third
  of the size it reaches after the pass. A 6000-step arm does not reach the regime the tanh is for.
- For the stream's question: at 6000 steps the meta part after block 22 is about 3 times its input, against 8 to 9
  times after the pass. Arms on the stream's scale read at 6000 see the start of the growth only.
