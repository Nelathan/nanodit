# Timestep distribution: logit-normal σ 0.7

2026-11-04. Closes PLAN G7's choice of the training t.

Facts weighed (FACTS, Timestep distribution arms at d768x12): three paired same-seed runs of 3000 steps. σ 0.7 beat the
σ 1 baseline in the score-grid loss and R² at every scored step, by 0.004 in loss at the end, winning at t 0.26 to 0.66 and
losing at t 0.16 and 0.84. Shift 1.5 was worse at the grid mean at every step. This had been read as inside the earlier
ablation's 0.011 yardstick and not worth acting on; that yardstick is unpaired seed noise, and these runs share their seed.

Decided: σ 0.7 is the default. It is principled: it puts more learning where it is needed, since the centre t has higher
loss. Long runs are hero runs, so they take the best current evidence rather than
a significance test against the last run.

SPEC §4 states σ 0.7; `Config.t_sigma` defaults to 0.7. Every earlier config states `t_sigma = 1.0` so it still
describes the run it made. The score grid stays at shift 1, σ 1. The extremes and the sample quality stay open in PLAN G7.
