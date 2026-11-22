# The loss stays unweighted over spatial frequency

2026-11-22. Closes the question of weighting each spatial frequency of F − T by the decoder's gain (PLAN K3).

Decided: the loss stays `‖F − T‖²`, unweighted (SPEC §4); the `loss_alpha` switch leaves the code. The
decoder-weighted score (`objective.spectral_weight` at α 1) stays in the score records (SPEC §8).

Reasons:
- Arms are read on the held-out and uniform-grid scores, not on the sample grids (decided 2026-11-22): the model
  samples over every t evenly, and a judgement of samples does not settle a comparison.
- Measured (6000 steps, seed 41; logs 2026-11-22-freq-weight-arms, 2026-11-22-freq-weight-floors): every frequency
  arm (α 0.5 and 1 floored at 64 px, α 1 at 16 and 8 px) is behind the base on the plain scores, the uniform grid
  included, at every t or level with it.
- Measured (log 2026-11-22-freq-arms-floor8-decoder): under the decoder weight floored at 8 px, the objective the
  floor-8 arm trains on, the base is ahead of every frequency arm, the floor-8 arm by 0.005 to 0.009.

Not taken: the sample grids' read, on which α 0.5 beats the base and α 1 floored at 8 px is the best of the
frequency arms, its samples laying out forms first.

The arms stay on their branches: `arm/freq-floor-16` (`b70d20a`), `arm/freq-floor-8` (`21f2845`).
