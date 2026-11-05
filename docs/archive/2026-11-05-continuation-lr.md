# Continuation at usui lr 1e-4

2026-11-05. Closes PLAN C4.

Training goes on from `128_d1024x16_swpin_5ep`'s final checkpoint rather than from scratch, unless the model changes
(Daniel). Another decay to the end is not affordable, so the lr is flat. Four 2000-step arms (FACTS, Continuation lr
arms): 2e-5 and 1e-4 ended level at 1.3831, 2e-4 (the run's last lr) at 1.3843, 4e-4 at 1.3884. Daniel: "For me 1e-4
looks best, considering that an anneal drops loss but decelerates learning": the lowest lr wins a short read by
annealing, not by learning faster, so the tie goes to the higher of the two.

The continuation trains at usui lr 1e-4 flat, fallback 1.25e-4, t σ 0.7.
