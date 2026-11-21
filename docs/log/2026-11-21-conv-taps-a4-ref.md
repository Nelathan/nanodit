# Where the 3×3 conv kernels put their weight in a4 after one pass

`scripts/probe_conv_taps.py /mnt/mars/ai/tiny-dit/runs/a4-ref-1ep 32173`: per conv block, the share of the depthwise
kernel's squared weight, summed over channels, on the centre tap and on the ring of 8 around it. A kernel drawn at
init (every tap the same variance) has 0.111 / 0.889.

| Block | Centre | Ring 1 |
|---|---|---|
| 0 | 0.100 | 0.900 |
| 1 | 0.055 | 0.945 |
| 3 | 0.062 | 0.938 |
| 4 | 0.117 | 0.883 |
| 5 | 0.127 | 0.873 |
| 7 | 0.191 | 0.809 |
| 8 | 0.203 | 0.797 |
| 9 | 0.260 | 0.740 |
| 11 | 0.278 | 0.722 |
| 12 | 0.393 | 0.607 |
| 13 | 0.313 | 0.687 |
| 15 | 0.370 | 0.630 |
| 16 | 0.234 | 0.766 |
| 17 | 0.225 | 0.775 |
| 19 | 0.203 | 0.797 |
| 20 | 0.146 | 0.854 |
| 21 | 0.194 | 0.806 |
| 23 | 0.251 | 0.749 |

## Read

The attention blocks (2, 6, 10, …) have no conv and are not listed. One checkpoint, one seed; the share is weighted
by each channel's kernel norm.

- The first blocks stay near the init share or move outward (blocks 0, 1, 3 at 0.90 to 0.95 on the ring; 4 and 5
  just under it). The middle blocks move weight to the centre, most at blocks 12 to 15 (0.61 to 0.69); the last
  blocks sit between (0.75 to 0.85).
- The centre tap mixes nothing across tokens, so a centre-heavy kernel is, as a proxy, a block doing less spatial
  mixing.
- The shape is not monotone in depth: the least reach is used in the middle third, not the last. A kernel graded
  "larger early, smaller late" follows a prior this probe supports for the first third only. The middle blocks'
  locality is read with attention at blocks 10 and 14 beside them.
