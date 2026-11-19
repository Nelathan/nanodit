# The conv block's forms at d768x24: widths, gates, the norm's place and `out`

Runs, each `a1-ref-1ep` for 6000 steps with one change, under `/mnt/mars/ai/tiny-dit/runs/`. Every conv form below
writes all channels, except the basic conv blocks of `a1-attngout-6k` (meta only); the `mlpconv` arms have no mixer (archive 2026-11-19-conv-writes-all-channels); `d` is the block's read before `out`, `y` the
block's modulated input, and the write is `x += gate_a ⊙ norm(out(d))` unless stated.
- Branch `arm/conv-designs` (`5469328`):
  - `a1-mlpconv-6k`: no mixer sublayer; a depthwise 3×3 conv on the MLP's up branch, before the SwiGLU gate.
  - `a1-mlpconv-center3-6k`: the same, layout `conv, attn, conv`.
  - `a1-lfmhalf-6k`: LFM-all (`d = C ⊙ conv(B ⊙ v)`, B, C, v from one projection) at 384 channels.
  - `a1-lfmall-center3-6k`: LFM-all, layout `conv, attn, conv`.
- Branch `arm/slim-conv` (`5e9725d`): `a1-slimconv-6k`, `a1-slim512-6k`: `d = σ(G y) ⊙ conv(V y)` at 384 and 512
  channels.
- Branch `arm/gate-outside-norm` (`e96a30b`), the form's own write gate replaced by one gate on the normed write,
  `x += gate_a ⊙ σ(G y) ⊙ norm(out(d))`, at the same parameter count:
  - `a1-lfmall-gout-6k` (C replaced), `a1-gdn2all-gout-6k` (GDN2-all's σ(write) replaced), `a1-slim768-gout-6k`
    (σ(G y) moved).
  - `a1-slim768-6k`: slim at 768 channels, the baseline of `a1-slim768-gout-6k`.
  - `a1-attngout-6k`: the attention gate after `attn_norm` instead of before `out`, and the cross attention's write
    gated per token by `σ(W y)` beside its static gate; basic conv blocks.
- Branch `arm/lfm-forms` (`41e92d2`):
  - `a1-slimsend-6k`: `d = σ(G y) ⊙ conv(σ(S y) ⊙ V y)`, slim at 768 with a gate on what each token sends.
  - `a1-lfmnormread-6k`: `x += gate_a ⊙ out(C ⊙ norm(conv(B ⊙ v)))`, the norm on the read and no branch norm.
  - `a1-lfmnoout-6k`: `x += gate_a ⊙ norm(C ⊙ conv(B ⊙ v))`, no `out`.

The seed draw at 6000 is about 0.001 (log 2026-11-18-seed42); a gap under about 0.002 is not resolved. Comparisons
named below: LFM-all −0.0047 at 6000, GDN2-all −0.0072, attention in the middle of three with the basic conv −0.0044,
−0.0029 at the same wall-clock (log 2026-11-19-conv-forms-and-attn-middle-of-three).

## Score

Δ against `a1-ref-1ep` (seed 41, 593 img/s at the 220 W power limit) on the Lunara score; "same time" and throughput
as in log 2026-11-19-conv-forms-and-attn-middle-of-three (the median `imgs_per_s` of `train.jsonl`); these arms
trained at 198 W.

| Arm | img/s | Params | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | Same time | Held-out 6000 |
|---|---|---|---|---|---|---|---|---|---|---|
| conv in the MLP | 597 | 196.2M | −0.0004 | −0.0037 | −0.0008 | +0.0014 | +0.0026 | +0.0027 | +0.0024 | 1.3164 |
| conv in the MLP, middle of three | 568 | 203.0M | −0.0024 | −0.0042 | −0.0018 | +0.0000 | +0.0006 | +0.0011 | +0.0026 | 1.3143 |
| LFM-all at 384 | 561 | 217.0M | −0.0052 | −0.0053 | −0.0030 | −0.0025 | −0.0039 | −0.0013 | +0.0007 | 1.3111 |
| LFM-all, middle of three | 508 | 240.4M | −0.0072 | −0.0080 | −0.0063 | −0.0056 | −0.0070 | −0.0053 | +0.0003 | 1.3072 |
| slim at 384 | 563 | 211.7M | −0.0022 | −0.0030 | −0.0026 | −0.0016 | −0.0028 | −0.0024 | −0.0006 | 1.3102 |
| slim at 512 | 550 | 217.0M | −0.0027 | −0.0027 | −0.0029 | −0.0036 | −0.0045 | −0.0022 | +0.0004 | 1.3102 |
| slim at 768 | 534 | 227.7M | −0.0027 | −0.0037 | −0.0035 | −0.0028 | −0.0042 | −0.0023 | +0.0014 | 1.3097 |
| slim at 768, gate outside | 528 | 227.7M | −0.0025 | −0.0017 | −0.0014 | −0.0005 | −0.0022 | −0.0020 | +0.0022 | 1.3114 |
| slim at 768, sending gate | 506 | 238.3M | −0.0048 | −0.0043 | −0.0034 | −0.0030 | −0.0046 | −0.0022 | +0.0036 | 1.3100 |
| LFM-all, gate outside | 509 | 238.3M | −0.0061 | −0.0043 | −0.0037 | −0.0019 | −0.0023 | −0.0013 | +0.0042 | 1.3107 |
| LFM-all, norm on the read | 524 | 238.3M | −0.0074 | −0.0084 | −0.0078 | −0.0066 | −0.0070 | −0.0059 | −0.0014 | 1.3069 |
| LFM-all, no `out` | 543 | 227.7M | −0.0048 | −0.0017 | +0.0011 | +0.0031 | +0.0019 | +0.0033 | +0.0064 | 1.3163 |
| GDN2-all, gate outside | 490 | 248.9M | −0.0039 | −0.0048 | −0.0049 | −0.0031 | −0.0041 | −0.0043 | +0.0028 | 1.3086 |
| attention gate after the norm | 576 | 208.2M | +0.0008 | −0.0005 | −0.0009 | −0.0003 | −0.0017 | −0.0011 | −0.0001 | 1.3127 |

LFM-all with the norm on its read is ahead of LFM-all at every eval from 2000 (by 0.0002, 0.0014, 0.0004, 0.0014,
0.0012).

Δ per t at 6000, the score's 10 t:

| Arm | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| conv in the MLP | +0.0023 | +0.0064 | +0.0076 | +0.0073 | +0.0055 | +0.0027 | −0.0008 | −0.0019 | −0.0020 | +0.0000 |
| conv in the MLP, middle of three | +0.0030 | +0.0055 | +0.0059 | +0.0057 | +0.0033 | −0.0003 | −0.0033 | −0.0047 | −0.0038 | −0.0000 |
| LFM-all at 384 | −0.0010 | −0.0011 | −0.0009 | −0.0006 | −0.0009 | −0.0019 | −0.0034 | −0.0029 | −0.0014 | +0.0015 |
| LFM-all, middle of three | −0.0026 | −0.0047 | −0.0057 | −0.0060 | −0.0063 | −0.0069 | −0.0075 | −0.0063 | −0.0052 | −0.0023 |
| slim at 384 | −0.0005 | −0.0023 | −0.0032 | −0.0035 | −0.0035 | −0.0039 | −0.0036 | −0.0021 | −0.0018 | +0.0003 |
| slim at 512 | −0.0013 | −0.0031 | −0.0039 | −0.0039 | −0.0037 | −0.0040 | −0.0039 | −0.0024 | −0.0005 | +0.0041 |
| slim at 768 | −0.0014 | −0.0027 | −0.0034 | −0.0034 | −0.0034 | −0.0037 | −0.0039 | −0.0019 | −0.0011 | +0.0015 |
| slim at 768, gate outside | −0.0004 | −0.0015 | −0.0020 | −0.0024 | −0.0028 | −0.0037 | −0.0041 | −0.0030 | −0.0016 | +0.0016 |
| slim at 768, sending gate | −0.0018 | −0.0030 | −0.0034 | −0.0033 | −0.0028 | −0.0031 | −0.0034 | −0.0020 | −0.0005 | +0.0012 |
| LFM-all, gate outside | −0.0015 | −0.0022 | −0.0020 | −0.0012 | −0.0013 | −0.0022 | −0.0028 | −0.0018 | −0.0003 | +0.0020 |
| LFM-all, norm on the read | −0.0032 | −0.0058 | −0.0068 | −0.0071 | −0.0073 | −0.0078 | −0.0078 | −0.0063 | −0.0050 | −0.0015 |
| LFM-all, no `out` | +0.0008 | +0.0033 | +0.0047 | +0.0054 | +0.0053 | +0.0043 | +0.0029 | +0.0024 | +0.0019 | +0.0020 |
| GDN2-all, gate outside | −0.0033 | −0.0051 | −0.0061 | −0.0055 | −0.0056 | −0.0056 | −0.0060 | −0.0045 | −0.0028 | +0.0014 |
| attention gate after the norm | −0.0009 | −0.0016 | −0.0022 | −0.0020 | −0.0015 | −0.0011 | −0.0008 | +0.0002 | −0.0003 | −0.0007 |

## Bands

R² per band at 6000 against `a1-seed42-6k`'s 6000 checkpoint, as in log
2026-11-19-conv-forms-and-attn-middle-of-three (bands: over three tokens, one to three, inside one; noise at most
0.001 for t ≤ 0.66 and 0.004 on the coarse band at t 0.74).

| Arm, t | 0.53 | 0.60 | 0.66 | 0.74 |
|---|---|---|---|---|
| conv in the MLP | 0.000, −0.001, −0.001 | 0.000, 0.000, 0.000 | −0.001, 0.000, 0.000 | −0.006, +0.001, +0.001 |
| conv in the MLP, middle of three | −0.002, 0.000, 0.000 | −0.004, +0.001, +0.001 | −0.007, +0.002, +0.002 | −0.016, +0.002, +0.002 |
| LFM-all at 384 | −0.002, 0.000, 0.000 | −0.002, +0.001, +0.001 | −0.004, +0.001, +0.001 | −0.009, 0.000, +0.001 |
| LFM-all, middle of three | +0.004, +0.001, +0.001 | +0.005, +0.002, +0.002 | +0.005, +0.003, +0.002 | +0.005, +0.004, +0.002 |
| slim at 384 | +0.001, 0.000, +0.001 | +0.002, +0.001, +0.001 | +0.003, +0.001, 0.000 | +0.008, +0.002, +0.001 |
| slim at 512 | −0.001, 0.000, +0.001 | −0.003, +0.001, +0.001 | −0.006, +0.001, +0.001 | −0.017, +0.001, +0.001 |
| slim at 768 | −0.001, +0.001, +0.001 | 0.000, +0.001, +0.001 | 0.000, +0.001, 0.000 | +0.003, +0.001, 0.000 |
| slim at 768, gate outside | +0.002, 0.000, +0.001 | +0.004, +0.001, +0.001 | +0.006, +0.001, +0.001 | +0.011, +0.001, +0.001 |
| slim at 768, sending gate | 0.000, 0.000, 0.000 | −0.001, +0.001, +0.001 | −0.003, 0.000, 0.000 | −0.009, −0.001, +0.001 |
| LFM-all, gate outside | −0.001, 0.000, 0.000 | −0.002, 0.000, 0.000 | −0.006, 0.000, 0.000 | −0.016, 0.000, 0.000 |
| LFM-all, norm on the read | +0.003, +0.001, +0.002 | +0.004, +0.002, +0.002 | +0.004, +0.003, +0.002 | +0.003, +0.004, +0.002 |
| LFM-all, no `out` | +0.001, −0.001, −0.001 | +0.001, −0.001, −0.001 | −0.001, −0.001, −0.001 | −0.014, −0.001, −0.001 |
| GDN2-all, gate outside | +0.001, +0.001, +0.001 | +0.001, +0.002, +0.001 | +0.003, +0.002, +0.001 | +0.003, +0.001, +0.001 |
| attention gate after the norm | +0.001, 0.000, 0.000 | +0.002, 0.000, 0.000 | +0.004, 0.000, 0.000 | +0.004, 0.000, 0.000 |

## Step time

`scripts/probe_step.py` at 198 W (`runs/time_slim_forms_2026-11-19.txt`, `runs/time_gate_outside_2026-11-19.txt`,
`runs/time_lfm_forms_2026-11-19.txt`): basic 591–592 img/s, slim 384 566, slim 512 554, slim 768 536–537, LFM-all
525–526, LFM-all with the norm on its read 527, LFM-all without `out` 547, slim with the sending gate 508, LFM-all with
the gate outside 514, GDN2-all with the gate outside 493, slim 768 with the gate outside 531, the attention gate after
the norm 580. The sending gate's slim and LFM-all hold the same three 768 × 768 matrices per conv block; slim's are
three separate projections, LFM's one.

## Read

- LFM-all with the norm on its read is the best conv form: ahead of LFM-all at every eval from 2000, about 0.0012
  at 6000 (inside the seed draw, the same sign five times), at LFM-all's speed; it gains on all three bands like LFM-all, +0.003 to +0.004 on the coarse band and +0.001 to
  +0.004 on the fine bands at t 0.53 to 0.74 (LFM-all +0.001 to +0.003 on the fine bands); held-out 1.3069 against
  1.3081. At the same wall-clock −0.0014, level with GDN2-all and behind attention in the middle of three.
- The gain of the LFM form lives in the signed linear factors. Replacing C by a sigmoid gate outside the norm loses
  0.0034 of LFM-all's 0.0047. Slim's sigmoid gates leave it at −0.0020 to −0.0024 whatever the width (384, 512, 768), with
  a sending gate, or with its gate moved outside the norm: slim with the sending gate is LFM-all with sigmoids on B and
  C, and is 0.0025 behind it. *Hypothesis*: the LFM form works as a product of three projections, as SwiGLU does, not
  as gating that admits.
- `out` is needed: without it LFM-all leads at 1000 and is behind the reference from 3000 (+0.0033 at 6000, worse at
  every t). The conv in the MLP runs a similar course: ahead at 2000 (−0.0037), behind from 4000 (+0.0027 at 6000).
- The gate outside the norm costs where it replaces a signed factor (LFM-all, 0.0034) or the gate inside GDN2's update
  (GDN2-all, 0.0029), is level on slim, and adds nothing on attention (−0.0011, −0.0001 at the same wall-clock). The separate gate matrix
  costs 1 to 2.3 % of the step against the same parameters inside one projection (slim 768 537 → 531, attention 591 →
  580, LFM-all 526 → 514).
- Attention in the middle of three and LFM-all do not add: −0.0053 together against −0.0044 and −0.0047 alone. LFM-all
  at half width keeps −0.0013 of −0.0047 and loses the coarse band at t 0.74 (−0.009).
