# The training t keeps σ 0.7

2026-11-18. Closes PLAN G7.

Decided: the training t stays logit-normal with σ 0.7 (SPEC §4).

Reason: the held-out loss. Over one pass at depth 24, σ 0.7 is ahead of σ 1 at every one of 19 shared evals, by 0.0012
to 0.0034 after step 12000, behind at t 0.16 (and at 16000 slightly at both ends of the grid; log
2026-11-18-a1-ref-1ep). One seed per σ: the lead is near the 0.003 taken as one seed's resolution, not yet measured,
so the decision rests on the sign held over every eval. The eval grids of the two runs at 16000 show no visible
difference; the unfinished forms in the samples are taken as the model's state, not the t distribution's
(*Hypothesis*), though they are fixed between t 0.8 and 0.67 of the sampler (log 2026-11-18-residual-writes).
