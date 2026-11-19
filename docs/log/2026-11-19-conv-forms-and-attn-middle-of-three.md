# The conv block's forms at d768x24, and attention in the middle of three

Runs, each `a1-ref-1ep` for 6000 steps with one change, under `/mnt/mars/ai/tiny-dit/runs/`:
- `a1-attncenter3-6k` (`main`, config only): layout `conv, attn, conv`, so 8 attention blocks of 24, not 6; the basic
  conv.
- The conv blocks' forms, branch `arm/gdn2-lite` (`13937a6` for lite and head gates, `e29ede1`, `49cd2ba`, `19df5aa`; the
  same model code for the forms each ran), every conv block changed, `d` the block's write before `out`:
  - `a1-gdn2-lite-6k`: `d = σ(write) ⊙ silu(conv(y)) − σ(forget) ⊙ y`, one forget and one write gate per head.
  - `a1-gdn2-headgates-6k`: as lite, with a projected value `conv(W y)`.
  - `a1-gdn2-inputstate-6k`: value and per-channel gates from one 768 → 2304 projection, the state `y`.
  - `a1-lfmconv-6k`: LFM2's gated short conv, `d = C ⊙ conv(B ⊙ v)`, B, C, v from one 768 → 2304 projection.
  - `a1-gdn2-fullall-6k`: GDN2 as in `a1-gdn2meta-6k` (log 2026-11-18-conv-gate-and-gdn2), writing all channels.
  - `a1-lfmall-6k`: the LFM form, writing all channels.
  All but the two `all` arms write the meta channels only.

The seed draw at 6000 is about 0.001 (log 2026-11-18-seed42); a gap under about 0.002 is not resolved.

## Score

Δ against `a1-ref-1ep` (seed 41) on the Lunara score. "Same time": against the reference after the same wall-clock,
6000 × (593 / arm's img/s) steps, read off `a1-ref-1ep`'s pass by linear interpolation between its 1000-step evals.
Throughput: median `imgs_per_s` to 6000 after step 500, at the 220 W power limit.

| Arm | img/s | Params | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | Same time | Held-out 6000 |
|---|---|---|---|---|---|---|---|---|---|---|
| seed 42 | 599 | 204.6M | −0.0007 | −0.0013 | −0.0009 | −0.0010 | −0.0010 | −0.0009 | −0.0014 | |
| attention third of four | 595 | 204.6M | −0.0009 | −0.0020 | −0.0023 | −0.0011 | −0.0015 | −0.0010 | −0.0011 | 1.3124 |
| attention middle of three | 568 | 210.5M | −0.0019 | −0.0038 | −0.0051 | −0.0044 | −0.0050 | −0.0044 | −0.0029 | 1.3089 |
| gated | 559 | 215.2M | −0.0009 | −0.0041 | −0.0025 | −0.0028 | −0.0034 | −0.0019 | +0.0001 | 1.3106 |
| GDN2, meta only | 505 | 247.1M | −0.0049 | −0.0054 | −0.0052 | −0.0050 | −0.0061 | −0.0053 | +0.0006 | 1.3071 |
| lite | 580 | 204.9M | +0.0028 | +0.0028 | +0.0033 | +0.0031 | +0.0020 | +0.0031 | +0.0039 | 1.3170 |
| head gates | 555 | 215.6M | −0.0012 | −0.0015 | −0.0005 | −0.0008 | −0.0011 | +0.0004 | +0.0027 | 1.3134 |
| input state | 519 | 236.5M | −0.0022 | −0.0045 | −0.0034 | −0.0029 | −0.0040 | −0.0022 | +0.0026 | 1.3105 |
| LFM, meta only | 526 | 236.5M | −0.0052 | −0.0061 | −0.0051 | −0.0035 | −0.0042 | −0.0034 | +0.0009 | 1.3088 |
| LFM, all channels | 522 | 236.5M | −0.0074 | −0.0082 | −0.0064 | −0.0062 | −0.0056 | −0.0047 | −0.0001 | 1.3081 |
| GDN2, all channels | 504 | 248.9M | −0.0040 | −0.0071 | −0.0052 | −0.0064 | −0.0074 | −0.0072 | −0.0012 | 1.3054 |

The GDN2 meta-only count is 204.6M + 42.5M; gated adds a 768 × 768 gate and lite 768 × 24 per conv block. `a1-ref-1ep` falls 0.0057 from 6000 to 7000 and 0.0154 from 16000 to
32173 (log 2026-11-18-a1-ref-1ep).

Δ per t at 6000, the score's 10 t:

| Arm | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| middle of three | −0.0028 | −0.0054 | −0.0069 | −0.0072 | −0.0067 | −0.0060 | −0.0047 | −0.0023 | −0.0011 | −0.0006 |
| lite | +0.0003 | +0.0020 | +0.0032 | +0.0039 | +0.0044 | +0.0040 | +0.0032 | +0.0041 | +0.0040 | +0.0024 |
| head gates | −0.0013 | −0.0008 | −0.0002 | +0.0005 | +0.0010 | +0.0003 | −0.0006 | +0.0003 | +0.0013 | +0.0031 |
| input state | −0.0021 | −0.0028 | −0.0031 | −0.0027 | −0.0025 | −0.0028 | −0.0034 | −0.0018 | −0.0011 | +0.0002 |
| LFM, meta | −0.0014 | −0.0028 | −0.0035 | −0.0037 | −0.0041 | −0.0049 | −0.0060 | −0.0045 | −0.0022 | −0.0009 |
| LFM, all | −0.0022 | −0.0042 | −0.0050 | −0.0052 | −0.0054 | −0.0060 | −0.0067 | −0.0057 | −0.0050 | −0.0017 |
| GDN2, all | −0.0046 | −0.0075 | −0.0096 | −0.0098 | −0.0093 | −0.0089 | −0.0088 | −0.0061 | −0.0060 | −0.0015 |

## Bands

R² per band at 6000 against `a1-seed42-6k`'s 6000 checkpoint, scored after its run with the band score
(`scripts/score_checkpoint.py`, `a1-seed42-6k/score_128.jsonl`; caption loss 1.4684, the run's own). Bands: over three
tokens, one to three, inside one. The band noise, seed 42 against `a1-attnthird-6k` (level in score): at most 0.001 in
every band for t ≤ 0.66, 0.004 on the coarse band at t 0.74 and 0.024 at t 0.84, where the coarse band is not read.

| Arm, t | 0.53 | 0.60 | 0.66 | 0.74 |
|---|---|---|---|---|
| middle of three | +0.003, +0.001, +0.001 | +0.004, +0.001, +0.001 | +0.005, 0.000, 0.000 | +0.006, 0.000, +0.001 |
| LFM, meta | +0.002, +0.001, +0.001 | +0.004, +0.002, +0.001 | +0.005, +0.002, +0.001 | +0.006, +0.003, +0.001 |
| LFM, all | +0.003, +0.001, +0.001 | +0.004, +0.001, +0.002 | +0.004, +0.002, +0.002 | +0.003, +0.003, +0.002 |
| GDN2, all | 0.000, +0.002, +0.002 | −0.001, +0.002, +0.002 | −0.003, +0.002, +0.002 | −0.007, +0.004, +0.003 |

Under t 0.5 the changes are at most 0.003 in any band.

## Step time

`scripts/probe_step.py` (`runs/time_gdn2_forms_2026-11-18.txt`, 220 W): basic 598 img/s, full GDN2 507, input state
522, head gates 559, lite 583. At 198 W (`runs/time_power90_2026-11-18.txt`): basic 594, middle of three 567, LFM meta
525, LFM all 526, full GDN2 501. The profile (`runs/profile_conv_2026-11-18.txt`, basic against full GDN2): the
depthwise conv's kernels are not among the 25 largest by CUDA time in either; full GDN2's 77 ms more per step are its
projection's matmuls, its Usui step (+9 ms) and copies.

## Read

- What buys loss in a conv block is a projected value and per-channel gates. With per-head gates the projected value
  is level with the reference (head gates); without it the block is worse than the basic conv (lite). The projected
  state is worth about 0.003 more (GDN2 meta against input state).
- Writing all channels costs no time and gains 0.0013 (LFM) to 0.0019 (GDN2) at 6000; ahead at every eval but GDN2's
  first.
- The cost of a form is its projection's width: 768 × 768 per conv block costs about 4 % of the step
  (lite against head gates). The conv itself
  costs next to nothing.
- At the same wall-clock at 6000 steps only the middle-of-three layout is ahead beyond the seed draw (−0.0029). Every
  conv form is within ±0.0015 or behind: at 6000 the reference still falls 0.006 per 1000 steps, which prices a 15 %
  slower step at about the forms' lead. In the second half of the pass it falls 0.001 per 1000 steps; a lead that
  holds would then outweigh the step (*Hypothesis*). GDN2-all's lead does not shrink from 2000 to 6000 (0.0052
  to 0.0074, 0.0072 at 6000); LFM-all's does (0.0082 to 0.0047).
- The forms differ in where they gain. Attention density gains on the coarse band, at mid t. The LFM forms gain on all
  three bands. GDN2-all gains most on the fine bands and loses on the coarse band at t 0.66 to 0.74, where the forms are
  set (log 2026-11-18-residual-writes).
