# A null key in joint attention, the sink

2026-11-20. Closes the question of a null key and value, letting a head write nothing (PLAN J3).

Decided: each key/value head has a learned null key, initialised at zero, with value zero, met by the query before
RoPE; a head writes σ(lse − b) times its RMS-normed read, b its logit on the null key. Architecture a4 (SPEC §3, §5).

Reasons:
- Joint attention without it cannot write nothing: the self-match still writes its value. Measured (2026-11-20,
  6000 steps, seed 41; log 2026-11-20-joint-attention-and-caption-refiner): the sink recovers the 0.0020 joint
  attention alone lost (score 1.4638 against 1.4658).
- The null key at zero against a randn init: within the spread (0.0008 score, 0.0011 held-out, `a3-jointsink-6k`
  against `a2-jointsink-nopool-6k`); zero is taken as the plain choice, the sink fully open at step 0.
- The lowest head's openness at t 0.84 is 0.90 or more in every block of every run: the sink is used little at
  6000 steps. Kept as the structural option for a head with nothing to read.
