# a1-lfmnormread-1ep: LFM's gated conv with the norm on its read, one pass

Run: `/mnt/mars/ai/tiny-dit/runs/a1-lfmnormread-1ep` (branch `arm/lfm-forms`, `5ad4c3e`): `a1-ref-1ep` with every conv
block `x += gate_a ⊙ out(C ⊙ norm(conv(B ⊙ v)))`, B, C, v from one 768 → 2304 projection, writing all channels, no
branch norm; 32,173 steps, seed 41, the same data order. 238.3M parameters against 204.6M. Its eval grid renders at
guidance 1, the reference's at 4, so the grids are not compared. One seed each; at 6000 steps the seed draw is about
0.001 (log 2026-11-18-seed42), unmeasured later in the pass.

Throughput: median `imgs_per_s` after step 500, 524 img/s at the 198 W power limit, against the reference's 594 at
220 W (591 at 198 W, FACTS). 4 h 32 min wall-clock. The lr is constant after warmup in both, so the reference read at
an earlier step is the reference after the same time.

## Score

Lunara caption loss. "Same time": against the reference after the same wall-clock, step × 524 / 594, read off
`a1-ref-1ep` by linear interpolation between its 1000-step evals.

| Step | Arm | `a1-ref-1ep` | Δ | Same time |
|---|---|---|---|---|
| 1000 | 1.5488 | 1.5562 | −0.0074 | |
| 2000 | 1.5100 | 1.5185 | −0.0084 | −0.0173 |
| 4000 | 1.4791 | 1.4857 | −0.0065 | −0.0124 |
| 6000 | 1.4643 | 1.4694 | −0.0051 | −0.0105 |
| 8000 | 1.4539 | 1.4582 | −0.0044 | −0.0096 |
| 12000 | 1.4429 | 1.4458 | −0.0029 | −0.0071 |
| 16000 | 1.4341 | 1.4396 | −0.0055 | −0.0079 |
| 20000 | 1.4288 | 1.4334 | −0.0046 | −0.0079 |
| 24000 | 1.4245 | 1.4296 | −0.0051 | −0.0075 |
| 28000 | 1.4221 | 1.4256 | −0.0035 | −0.0073 |
| 32000 | 1.4197 | 1.4236 | −0.0039 | −0.0061 |
| 32173 | 1.4199 | 1.4242 | −0.0043 | −0.0059 |

Ahead at all 33 evals. Mean Δ over the evals of each stretch: 1000–6000 −0.0070, 7000–12000 −0.0047, 13000–18000
−0.0049, 19000–24000 −0.0040, 25000–32000 −0.0042. From 24000 to 32000 the reference falls 0.0060, 0.00075 per 1000
steps.

Δ per t at 32173, the score's 10 t:

| 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|
| −0.0020 | −0.0045 | −0.0055 | −0.0060 | −0.0063 | −0.0064 | −0.0060 | −0.0042 | −0.0018 | −0.0003 |

Held-out (FLUX.2-dev rows): 1.4095 at 1000, 1.3070 at 6000, 1.2718 at 16000, 1.2536 at 32173. `a1-ref-1ep` predates the
held-out score; no Δ.

## Bands

R² per band at 32173 against `a1-ref-1ep`'s 32173, scored after its run (`scripts/score_checkpoint.py`,
`a1-ref-1ep/score_128.jsonl`; caption loss 1.4242, the run's own). Bands: over three tokens, one to three, inside one.
The band noise at 6000 (log 2026-11-19-conv-forms-and-attn-middle-of-three): at most 0.001 for t ≤ 0.66, 0.004 on the
coarse band at t 0.74, 0.024 at t 0.84.

| t | Over three | One to three | Inside one |
|---|---|---|---|
| 0.34 | +0.0003 | +0.0006 | +0.0005 |
| 0.47 | +0.0002 | +0.0014 | +0.0012 |
| 0.53 | −0.0003 | +0.0019 | +0.0015 |
| 0.60 | −0.0019 | +0.0026 | +0.0018 |
| 0.66 | −0.0054 | +0.0030 | +0.0016 |
| 0.74 | −0.0135 | +0.0022 | +0.0009 |
| 0.84 | −0.0291 | −0.0012 | +0.0008 |

Under t 0.3 every band is within 0.0003.

## Read

- The lead holds over the pass. Its mean shrinks from 0.0070 over the first 6000 steps to between 0.0040 and 0.0049
  over each later stretch; the same-time lead is 0.0059 at the end. With the reference falling 0.00075 per 1000 steps late in the pass, 0.0043 is
  about 5700 of its steps, 18 % more steps, against an 11.8 % lower throughput.
- The gain is at low and mid t (largest −0.0064 at t 0.53), level at t 0.84.
- The fine bands gain at every t from 0.34 to 0.74. The coarse band loses from t 0.60 up, −0.0135 at 0.74, beyond the
  band noise; at 6000 the LFM forms gained on all three bands. *Hypothesis*: composition, set at high t, falls behind
  while the conv blocks write texture into every channel. Not checked on samples: the grids differ in guidance.
