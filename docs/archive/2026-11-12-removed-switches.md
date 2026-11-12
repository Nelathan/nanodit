# Removed: the plain conv mixer, the fixed Usui step, masked priming and the online REPA

2026-11-12. The losing branches of questions that are settled, or of a side quest set aside, leave the trunk. Each is
in the tree at `a9a111a`.

| Removed | Why |
|---|---|
| `gdn = false`: the plain depthwise conv writing the meta channels | The gated update (SPEC §3) leads it by 0.0014 to 0.0022 at 128 px and 0.0039 to 0.0047 at 256 px after 6000 steps, at 84 % of its speed (`log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`) |
| `pin_weights = false`: Usui's fixed step with free weight norms | The pin is the design since archive 2026-11-05; how it generalizes to norm bounds is PLAN G8 |
| Masked priming as noise (`mask_start`, `prime_steps`, `objective.token_t`) | 0.009 behind scratch at 6000 steps with the share falling over 3000, 0.017 over 6000 (`log/2026-11-07-masked-priming.md`); tokens of pure noise also keep the conv blocks from mixing as intended. The arm worth running is the opposite, clean context tokens without loss (`topics/priming.md`) |
| REPA with online targets (`align.VisionTarget`, `repa_block`, `repa_weight`, `repa_steps`, `probe_repa_target.py`) | 0.0023 ahead of scratch at 8.4× the time per aligned step (`log/2026-11-08-repa-first-arm.md`). It returns only with precomputed targets of one size, which make it cheap to add again |

Configs that set these fields, and the earlier trunk arms whose model no longer exists, are deleted with them; each
run directory keeps the `config.toml` it ran with.
