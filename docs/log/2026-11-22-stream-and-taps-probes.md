# The stream's size and the conv taps on the 6000-step arms

`scripts/probe_stream.py RUN_DIR 6000 16` on the stream arms and `scripts/probe_conv_taps.py RUN_DIR 6000` on the kernel
layouts, each on its arm's code (the runs of logs 2026-11-22-stream-arms, 2026-11-22-skip-arms and
2026-11-22-conv-kernel-layouts), CPU, the first 16 held-out rows with the score's noise. One checkpoint per arm, so
the reads below describe these checkpoints; what they say about the mechanism is *Hypothesis*.

| Run | Probe from |
|---|---|
| `a4-base-6k` | `298b2e0` |
| `a4-split-read-6k` | `4ec5c48` |
| `a4-held-meta-6k` | `5a5409e` |
| `a4-held-meta-skips-6k` | `d178c97` |
| `a4-held-meta-skips-linear-skip-6k` | `e271f72` |
| `a4-no-tables-6k` | `384dc22` |
| the four conv layouts | `54d476e` |

## The stream

Each cell: slice RMS / meta RMS / the meta part's common mode (its mean over an image's tokens) RMS / the rest RMS.
The target's RMS is 2.11, 1.70 and 1.24 at t 0.16, 0.47 and 0.84.

| Run | After | t 0.16 | t 0.47 | t 0.84 |
|---|---|---|---|---|
| base | patch embed | 0.90/0.98/0.47/0.85 | 0.73/0.79/0.34/0.71 | 0.86/0.88/0.22/0.85 |
| base | block 11 | 1.04/1.40/1.03/0.94 | 0.79/1.24/0.85/0.91 | 0.68/1.71/1.39/0.99 |
| base | block 23 | 1.80/2.98/2.33/1.86 | 1.24/2.79/1.88/2.06 | 0.51/2.72/2.31/1.44 |
| split read | patch embed | 0.90/0.99/0.47/0.86 | 0.73/0.79/0.34/0.71 | 0.86/0.88/0.23/0.85 |
| split read | block 11 | 1.04/1.57/1.22/0.99 | 0.80/1.74/1.37/1.08 | 0.75/2.69/2.31/1.38 |
| split read | block 23 | 1.80/3.15/2.49/1.94 | 1.24/4.69/3.97/2.50 | 0.51/13.50/13.26/2.52 |
| held meta | patch embed | 0.90/0.99/0.49/0.86 | 0.73/0.81/0.38/0.71 | 0.86/0.90/0.28/0.85 |
| held meta | block 11 | 1.02/1.00/0.86/0.50 | 0.79/1.00/0.83/0.55 | 0.75/1.00/0.91/0.40 |
| held meta | block 23 | 1.81/1.00/0.85/0.52 | 1.24/1.00/0.86/0.52 | 0.51/1.00/0.98/0.20 |
| held meta + long skips | patch embed | 0.90/0.99/0.49/0.86 | 0.73/0.81/0.38/0.71 | 0.86/0.90/0.28/0.85 |
| held meta + long skips | block 11 | 0.99/1.00/0.90/0.43 | 0.77/1.00/0.87/0.50 | 0.76/1.00/0.90/0.44 |
| held meta + long skips | block 23 | 1.81/1.00/0.82/0.57 | 1.24/1.00/0.86/0.51 | 0.51/1.00/0.98/0.19 |
| + linear skip | patch embed | 0.90/1.02/0.52/0.88 | 0.73/0.83/0.40/0.72 | 0.86/0.91/0.31/0.85 |
| + linear skip | block 11 | 1.74/1.00/0.81/0.59 | 0.85/1.00/0.85/0.53 | 0.09/1.00/0.98/0.19 |
| + linear skip | block 23 | 1.81/1.00/0.78/0.62 | 1.24/1.00/0.83/0.56 | 0.52/1.00/0.96/0.28 |
| no tables | patch embed | 0.90/0.99/0.52/0.84 | 0.73/0.80/0.39/0.70 | 0.86/0.89/0.27/0.85 |
| no tables | block 11 | 1.15/1.36/1.02/0.90 | 0.77/1.16/0.81/0.84 | 0.51/1.53/1.21/0.93 |
| no tables | block 23 | 1.80/1.99/1.62/1.15 | 1.23/1.87/1.44/1.20 | 0.52/2.27/1.99/1.10 |

## The conv taps

Per conv block, the share of the kernel's squared weight on each ring of taps (ring r at Chebyshev distance r from
the centre), with the share at init in brackets.

```
== a4-conv-graded-6k scripts/probe_conv_taps.py /mnt/mars/ai/tiny-dit/runs/a4-conv-graded-6k 6000 from 54d476e
a4-conv-graded-6k step 6000: per conv block, kernel, then per ring the share of squared weight (init share)
 0 7x7  0.031 (0.020)  0.289 (0.163)  0.285 (0.327)  0.395 (0.490)
 1 7x7  0.031 (0.020)  0.340 (0.163)  0.266 (0.327)  0.363 (0.490)
 3 7x7  0.031 (0.020)  0.522 (0.163)  0.189 (0.327)  0.258 (0.490)
 4 7x7  0.037 (0.020)  0.529 (0.163)  0.181 (0.327)  0.252 (0.490)
 5 7x7  0.044 (0.020)  0.533 (0.163)  0.175 (0.327)  0.249 (0.490)
 7 7x7  0.049 (0.020)  0.524 (0.163)  0.167 (0.327)  0.261 (0.490)
 8 5x5  0.078 (0.040)  0.769 (0.320)  0.153 (0.640)
 9 5x5  0.096 (0.040)  0.728 (0.320)  0.176 (0.640)
11 5x5  0.147 (0.040)  0.696 (0.320)  0.157 (0.640)
12 5x5  0.155 (0.040)  0.674 (0.320)  0.171 (0.640)
13 5x5  0.157 (0.040)  0.693 (0.320)  0.151 (0.640)
15 5x5  0.154 (0.040)  0.703 (0.320)  0.142 (0.640)
16 3x3  0.114 (0.111)  0.886 (0.889)
17 3x3  0.121 (0.111)  0.879 (0.889)
19 3x3  0.116 (0.111)  0.884 (0.889)
20 3x3  0.111 (0.111)  0.889 (0.889)
21 3x3  0.121 (0.111)  0.879 (0.889)
23 3x3  0.142 (0.111)  0.858 (0.889)
== a4-conv-graded-soft-6k scripts/probe_conv_taps.py /mnt/mars/ai/tiny-dit/runs/a4-conv-graded-soft-6k 6000 from 54d476e
a4-conv-graded-soft-6k step 6000: per conv block, kernel, then per ring the share of squared weight (init share)
 0 5x5  0.060 (0.040)  0.504 (0.320)  0.436 (0.640)
 1 5x5  0.047 (0.040)  0.568 (0.320)  0.385 (0.640)
 3 5x5  0.051 (0.040)  0.655 (0.320)  0.294 (0.640)
 4 5x5  0.056 (0.040)  0.690 (0.320)  0.254 (0.640)
 5 5x5  0.064 (0.040)  0.669 (0.320)  0.267 (0.640)
 7 5x5  0.079 (0.040)  0.591 (0.320)  0.330 (0.640)
 8 3x3  0.140 (0.111)  0.860 (0.889)
 9 3x3  0.155 (0.111)  0.845 (0.889)
11 3x3  0.203 (0.111)  0.797 (0.889)
12 3x3  0.194 (0.111)  0.806 (0.889)
13 3x3  0.152 (0.111)  0.848 (0.889)
15 3x3  0.208 (0.111)  0.792 (0.889)
16 3x3  0.161 (0.111)  0.839 (0.889)
17 3x3  0.134 (0.111)  0.866 (0.889)
19 3x3  0.144 (0.111)  0.856 (0.889)
20 3x3  0.122 (0.111)  0.878 (0.889)
21 3x3  0.135 (0.111)  0.865 (0.889)
23 3x3  0.142 (0.111)  0.858 (0.889)
== a4-conv-phased-6k scripts/probe_conv_taps.py /mnt/mars/ai/tiny-dit/runs/a4-conv-phased-6k 6000 from 54d476e
a4-conv-phased-6k step 6000: per conv block, kernel, then per ring the share of squared weight (init share)
 1 7x7  0.025 (0.020)  0.445 (0.163)  0.233 (0.327)  0.298 (0.490)
 3 7x7  0.041 (0.020)  0.442 (0.163)  0.225 (0.327)  0.292 (0.490)
 5 7x7  0.044 (0.020)  0.339 (0.163)  0.264 (0.327)  0.353 (0.490)
 6 3x3  0.109 (0.111)  0.891 (0.889)
 7 3x3  0.141 (0.111)  0.859 (0.889)
 8 3x3  0.129 (0.111)  0.871 (0.889)
 9 3x3  0.175 (0.111)  0.825 (0.889)
10 3x3  0.164 (0.111)  0.836 (0.889)
11 3x3  0.135 (0.111)  0.865 (0.889)
12 3x3  0.186 (0.111)  0.814 (0.889)
13 3x3  0.120 (0.111)  0.880 (0.889)
14 3x3  0.101 (0.111)  0.899 (0.889)
15 3x3  0.114 (0.111)  0.886 (0.889)
16 3x3  0.095 (0.111)  0.905 (0.889)
17 3x3  0.122 (0.111)  0.878 (0.889)
19 5x5  0.146 (0.040)  0.732 (0.320)  0.122 (0.640)
21 5x5  0.152 (0.040)  0.786 (0.320)  0.062 (0.640)
23 5x5  0.164 (0.040)  0.769 (0.320)  0.067 (0.640)
== a4-conv-period-6k scripts/probe_conv_taps.py /mnt/mars/ai/tiny-dit/runs/a4-conv-period-6k 6000 from 54d476e
a4-conv-period-6k step 6000: per conv block, kernel, then per ring the share of squared weight (init share)
 1 7x7  0.030 (0.020)  0.355 (0.163)  0.259 (0.327)  0.356 (0.490)
 2 5x5  0.046 (0.040)  0.603 (0.320)  0.351 (0.640)
 3 3x3  0.058 (0.111)  0.942 (0.889)
 5 7x7  0.044 (0.020)  0.442 (0.163)  0.212 (0.327)  0.302 (0.490)
 6 5x5  0.079 (0.040)  0.677 (0.320)  0.244 (0.640)
 7 3x3  0.144 (0.111)  0.856 (0.889)
 9 7x7  0.078 (0.020)  0.459 (0.163)  0.207 (0.327)  0.256 (0.490)
10 5x5  0.148 (0.040)  0.676 (0.320)  0.176 (0.640)
11 3x3  0.154 (0.111)  0.846 (0.889)
13 7x7  0.116 (0.020)  0.571 (0.163)  0.141 (0.327)  0.172 (0.490)
14 5x5  0.105 (0.040)  0.801 (0.320)  0.093 (0.640)
15 3x3  0.131 (0.111)  0.869 (0.889)
17 7x7  0.130 (0.020)  0.624 (0.163)  0.118 (0.327)  0.128 (0.490)
18 5x5  0.119 (0.040)  0.800 (0.320)  0.081 (0.640)
19 3x3  0.126 (0.111)  0.874 (0.889)
21 7x7  0.120 (0.020)  0.724 (0.163)  0.079 (0.327)  0.077 (0.490)
22 5x5  0.069 (0.040)  0.901 (0.320)  0.030 (0.640)
23 3x3  0.109 (0.111)  0.891 (0.889)
```

## Read

- **On the held stream the meta part is mostly its common mode.** Held at unit RMS, 0.78–0.98 of it is the mean over
  an image's tokens; what differs between tokens is 0.43–0.62 at low and mid t and 0.19–0.28 at t 0.84 after the last
  block. That is 18–38 % of the meta energy at low and mid t and 4–8 % at t 0.84 after block 23. On the base, after
  block 23, the token part is 1.4–2.1, half to three quarters of a meta RMS that grows with depth. Under the hold the
  common mode takes most of the fixed budget (*Hypothesis*: a per-image vector, from the text and t, that each
  block re-reads).
- **The common mode is what grows without the hold** (*Hypothesis*). With the split read alone the meta RMS reaches 13.5 at t 0.84
  after block 23, 13.3 of it the common mode.
- **Large kernels gather on ring 1.** In every layout the rings past 1 hold less than their init share; the first
  7×7 blocks keep the most (ring 3 at 0.29–0.40 of an init 0.49), the deep ones least (period block 21: 0.08). A
  5×5 late in the depth is close to a 3×3 (phased blocks 21, 23: ring 2 at 0.06–0.07 of 0.64).
- **3×3 kernels keep a centre share of 0.06–0.21 against an init 0.111,** drifting either way: graded stays
  within 0.111–0.142, graded-soft rises to 0.20–0.21 in blocks 11 and 15, and period block 3 falls to 0.058.
