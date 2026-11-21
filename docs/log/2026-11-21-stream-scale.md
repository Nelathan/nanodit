# The residual stream's scale in a4: where the gates grow and what the stream carries

Two probes on CPU, fp32, on `a4-ref-1ep` (step 0 = the init of its config and seed, step 32173) and
`a4-ref-1ep-anneal6k` (step 38173):

- `scripts/probe_gates.py RUN_DIR STEP`: each block's gate rows of `mod + table` at the score's 10 t, RMS on the slice
  (channels 0–127: x_t at the input, the prediction at the read-out) and on the meta channels (128–767).
- `scripts/probe_stream.py RUN_DIR STEP 32`: the first 32 held-out rows with the score's noise, at the score's 10 t;
  after the patch embed and after each block, the RMS of the slice and of the meta channels, the meta part split into
  its mean over an image's 64 tokens (common mode) and the rest. Console outputs were not kept; the scripts reproduce them.

## The gates by channel group, step 32173

RMS over t and channels, slice / meta.

| Block | Mixer gate | MLP gate |
|---|---|---|
| 0 conv | 0.123 / 0.388 | 0.087 / 0.260 |
| 1 conv | 0.039 / 0.335 | 0.113 / 0.371 |
| 2 attn | 0.068 / 0.393 | 0.091 / 0.342 |
| 3 conv | 0.041 / 0.410 | 0.105 / 0.448 |
| 4 conv | 0.049 / 0.489 | 0.100 / 0.500 |
| 5 conv | 0.019 / 0.393 | 0.094 / 0.490 |
| 6 attn | 0.017 / 0.417 | 0.161 / 0.484 |
| 7 conv | 0.011 / 0.388 | 0.111 / 0.469 |
| 8 conv | 0.011 / 0.365 | 0.138 / 0.596 |
| 9 conv | 0.012 / 0.423 | 0.230 / 0.549 |
| 10 attn | 0.030 / 0.660 | 0.140 / 0.632 |
| 11 conv | 0.027 / 0.530 | 0.366 / 0.564 |
| 12 conv | 0.017 / 0.549 | 0.264 / 0.605 |
| 13 conv | 0.047 / 0.560 | 0.331 / 0.615 |
| 14 attn | 0.075 / 1.002 | 0.408 / 0.672 |
| 15 conv | 0.047 / 0.664 | 0.456 / 0.720 |
| 16 conv | 0.088 / 0.659 | 0.354 / 0.763 |
| 17 conv | 0.109 / 0.609 | 0.501 / 0.844 |
| 18 attn | 0.271 / 1.117 | 0.290 / 1.052 |
| 19 conv | 0.164 / 0.736 | 0.390 / 0.969 |
| 20 conv | 0.072 / 1.041 | 0.463 / 1.264 |
| 21 conv | 0.099 / 0.982 | 0.540 / 1.152 |
| 22 attn | 0.239 / 2.576 | 0.421 / 1.763 |
| 23 conv | 0.232 / 2.227 | 0.481 / 0.337 |

After the anneal, block 22: 0.261 / 2.785 and 0.466 / 1.887; block 23: 0.256 / 2.364 and 0.524 / 0.360.

## The stream through depth, step 32173

RMS of slice / meta / meta common mode / meta rest. The target's RMS at these t: 2.112, 1.701, 1.237.

| After | t 0.16 | t 0.47 | t 0.84 |
|---|---|---|---|
| embed | 0.90 / 1.15 / 0.78 / 0.86 | 0.73 / 0.97 / 0.66 / 0.71 | 0.86 / 1.02 / 0.56 / 0.85 |
| 0 conv | 0.95 / 1.41 / 1.10 / 0.89 | 0.74 / 1.18 / 0.92 / 0.73 | 0.78 / 1.23 / 0.89 / 0.85 |
| 1 conv | 0.96 / 1.47 / 1.16 / 0.90 | 0.74 / 1.23 / 0.97 / 0.75 | 0.77 / 1.33 / 1.00 / 0.87 |
| 2 attn | 0.97 / 1.74 / 1.48 / 0.93 | 0.74 / 1.45 / 1.23 / 0.77 | 0.73 / 1.59 / 1.34 / 0.86 |
| 3 conv | 0.97 / 1.88 / 1.61 / 0.97 | 0.74 / 1.57 / 1.34 / 0.82 | 0.73 / 1.67 / 1.41 / 0.89 |
| 4 conv | 0.97 / 1.94 / 1.64 / 1.04 | 0.74 / 1.62 / 1.36 / 0.89 | 0.73 / 1.73 / 1.45 / 0.95 |
| 5 conv | 0.97 / 1.97 / 1.64 / 1.08 | 0.74 / 1.64 / 1.35 / 0.94 | 0.73 / 1.74 / 1.44 / 0.98 |
| 6 attn | 0.97 / 2.00 / 1.65 / 1.14 | 0.74 / 1.69 / 1.37 / 0.99 | 0.72 / 1.83 / 1.52 / 1.01 |
| 7 conv | 0.97 / 1.97 / 1.59 / 1.15 | 0.74 / 1.68 / 1.34 / 1.02 | 0.72 / 1.90 / 1.60 / 1.03 |
| 8 conv | 0.97 / 1.95 / 1.57 / 1.16 | 0.74 / 1.70 / 1.35 / 1.04 | 0.72 / 1.99 / 1.69 / 1.05 |
| 9 conv | 0.98 / 2.02 / 1.61 / 1.21 | 0.74 / 1.77 / 1.38 / 1.10 | 0.70 / 2.06 / 1.76 / 1.08 |
| 10 attn | 0.98 / 2.19 / 1.76 / 1.31 | 0.75 / 1.97 / 1.53 / 1.24 | 0.70 / 2.41 / 2.11 / 1.16 |
| 11 conv | 1.01 / 2.27 / 1.83 / 1.34 | 0.76 / 2.06 / 1.59 / 1.31 | 0.66 / 2.60 / 2.30 / 1.21 |
| 12 conv | 1.02 / 2.35 / 1.86 / 1.42 | 0.77 / 2.15 / 1.62 / 1.41 | 0.62 / 2.65 / 2.34 / 1.25 |
| 13 conv | 1.03 / 2.48 / 1.98 / 1.50 | 0.77 / 2.21 / 1.63 / 1.49 | 0.60 / 2.70 / 2.37 / 1.28 |
| 14 attn | 1.10 / 2.90 / 2.37 / 1.67 | 0.80 / 2.61 / 1.99 / 1.70 | 0.56 / 3.18 / 2.87 / 1.37 |
| 15 conv | 1.18 / 3.18 / 2.62 / 1.79 | 0.84 / 2.82 / 2.15 / 1.82 | 0.52 / 3.41 / 3.08 / 1.46 |
| 16 conv | 1.23 / 3.38 / 2.73 / 2.00 | 0.86 / 3.05 / 2.26 / 2.04 | 0.49 / 3.48 / 3.12 / 1.55 |
| 17 conv | 1.33 / 3.59 / 2.89 / 2.12 | 0.91 / 3.21 / 2.36 / 2.17 | 0.49 / 3.54 / 3.15 / 1.61 |
| 18 attn | 1.40 / 4.58 / 3.67 / 2.73 | 0.98 / 4.02 / 2.97 / 2.72 | 0.48 / 4.18 / 3.77 / 1.82 |
| 19 conv | 1.43 / 5.13 / 4.14 / 3.04 | 1.00 / 4.49 / 3.30 / 3.04 | 0.47 / 4.68 / 4.27 / 1.92 |
| 20 conv | 1.49 / 6.00 / 4.65 / 3.79 | 1.03 / 5.35 / 3.66 / 3.90 | 0.46 / 5.35 / 4.81 / 2.35 |
| 21 conv | 1.60 / 6.45 / 4.83 / 4.28 | 1.10 / 5.69 / 3.70 / 4.33 | 0.45 / 5.26 / 4.63 / 2.50 |
| 22 attn | 1.67 / 9.54 / 7.70 / 5.63 | 1.17 / 8.76 / 6.67 / 5.68 | 0.49 / 8.23 / 7.71 / 2.88 |
| 23 conv | 1.82 / 9.05 / 6.73 / 6.05 | 1.27 / 7.76 / 4.80 / 6.10 | 0.52 / 7.15 / 6.45 / 3.09 |

At init every block is the identity and the meta channels are the patch embed's projection of x_t: slice and meta
equal (0.90, 0.73, 0.86 at the three t), common mode 0.34, 0.22, 0.12. After the anneal the meta part is larger
again: after block 22, 10.49, 9.73, 9.22; the slice is unchanged within 0.02.

The slice's share of the energy a block reads (every branch reads the stream through an RMS norm over all 768
channels), from the table: 17 % at init (128 of 768 channels at equal RMS); after the trained patch embed 11 % at t
0.16 and 12 % at t 0.84; into block 22, 1.2 % at t 0.16 and 0.15 % at t 0.84.

## Read

- The slice is anchored, the meta channels are not. The slice is read out raw (no final norm or projection), so its
  size is the prediction's and its gates stay small. The meta channels are only ever read through RMS norms, which
  are blind to their scale; nothing in the loss holds their size, and it grows with depth to eight or nine times its
  input, and grew further in the anneal.
- The growth coincides with the gates' meta channels, which the per-block tables carry (log 2026-11-21-gate-sizes):
  the stream grows most where those gates are large. A write's size also depends on the branch's output weights.
- Both parts of the meta channels grow. The common mode, one vector per image shared by its tokens, is most of the
  meta energy at high t (88 % after block 22 at t 0.84) and less than half of it at t 0.47 from block 19 on. It rises
  most at blocks 0, 2, 10, 14, 18, 19, 20 and 22, conv and attention, and falls at block 23. The token-specific rest
  grows through the deep blocks, attention and conv alike; the largest single step is block 22.
- As the meta part grows, x_t and the prediction so far take a shrinking share of what the blocks read: into the last
  blocks the slice is about a hundredth of the read's energy at low t and a thousandth at high t (amplitude ratios
  about 0.11 and 0.04). *Inference*: the deep blocks write the slice's correction from a read in which the slice is a
  small part; a linear read can still amplify it.
- *Hypothesis*: with pre-norm reads, a block's influence is its write relative to the stream it lands in; once the
  stream grows, a later block needs a larger write for the same change, so the gates grow with depth, and the
  stream with them. The tables, outside the norm ceiling, let this run. That the attention blocks' share of the
  common mode is the caption, which every token reads alike, is a hypothesis too.
