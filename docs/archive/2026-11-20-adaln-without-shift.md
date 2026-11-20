# The adaLN without its shift

2026-11-20. Closes the question of dropping the adaLN's shift (PLAN J4).

Decided: each branch reads `rms_norm(x) · (1 + scale)`; the modulation is four vectors per block, scale and gate for
the mixer and the MLP. Architecture a4 (SPEC §3, §5).

Reasons:
- Lumina 2 and Z-Image modulate with scale and gate only (diffusers code); Qwen-Image-2.0 reports bias-free
  modulation.
- Measured (2026-11-20, 6000 steps, seed 41, over the refiner; log 2026-11-20-adaln-without-shift): score 0.0015
  lower, held-out 0.0008 higher, both within twice the seed spread; throughput level; 1.22M parameters fewer. Level
  on loss, the smaller form is taken.
