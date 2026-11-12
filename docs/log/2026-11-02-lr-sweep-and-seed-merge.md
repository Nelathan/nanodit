# LR sweep and seed merge (2026-11-02)

Moved verbatim from `FACTS.md` on 2026-11-11. These reads are of the model before the 2026-11-06 architecture merge (width 512 at first, a final linear, no branch norm, no pin); the numbers are the record of that model, not of the current one.

## LR sweep and seed merge (2026-11-02)

The LR sweep (PLAN C2): `128_sweep_lr{1e-3,2e-3,4e-3}`, one pass over `128_shuffled_0-46.parquet` (15042 steps, batch 128, warmup 752, no decay; tiny-dit `78fded7`). `lr2e-4` stopped in its first steps: cuSOLVER's fp32 `eigh` in usuitrack `frame.tangent_spectra` returned non-orthonormal vectors for a tangent Gram of ~1e-22 entries (a matrix behind a gate still near zero), the turn left the manifold (QᵀQ − I 1.7, then ~1e25), and `eigh` raised; with each Gram divided by its mean diagonal for the `eigh` (usuitrack `zero-init`, uncommitted; 1.61 ms for 48 Grams against 1.52 unscaled), the same arm ran 300 steps clean (scratch `eighrepro/`). Scored with scratch `heldout/score.py`: the first 1024 rows of shards 12 and 40, trained on once in every arm, fixed noise (seed 1234), t at the 10 bin midpoints 0.05 … 0.95, mean over t.

| usui lr | shard 12 (dalle3), caption / null | shard 40 (llava flux), caption / null |
|---|---|---|
| 1e-3 | 1.3816 / 1.3840 | 1.1805 / 1.1852 |
| 2e-3 | 1.3726 / 1.3752 | 1.1718 / 1.1745 |
| 4e-3 | 1.3691 / 1.3709 | 1.1679 / 1.1712 |

- Loss falls with lr, by 0.009 from 1e-3 to 2e-3 and 0.004 from 2e-3 to 4e-3; the caption gap is 0.002–0.003 in every arm. On the samples, 2e-3 looks best even though 4e-3 has better loss (a visual judgment).

The same checkpoints on the checkpoint score's eval set (SPEC §8: the 1000 Lunara renders of `moonworks/lunara-art-eval`, never trained on), scored by `train.score` from scratch `heldout/score_lunara.py`: noise seed 1, t at `t_quantiles(10, 1.0)`, about 4 s per checkpoint on the 4070 Super.

| checkpoint | caption | null |
|---|---|---|
| `lr2e-4` final (rerun with the fix, its own `score.jsonl`) | 1.5584 | 1.5598 |
| `lr1e-3` final | 1.5207 | 1.5215 |
| `lr2e-3` step 1000 / 4000 / 8000 | 1.6442 / 1.5564 / 1.5299 | 1.6438 / 1.5570 / 1.5307 |
| `lr2e-3` final | 1.5122 | 1.5130 |
| `lr4e-3` final | 1.5106 | 1.5109 |
| `128_4e-3_ep2` final (epoch 2, usui lr 2e-3, batch 128; its own `score.jsonl`) | 1.4929 | 1.4951 |
| `128_4e-3_ep3` final (epoch 3, usui lr 2e-3, batch 256) | 1.4873 | 1.4895 |
| `128_4e-3_ep4` final (epoch 4, usui lr 1e-3, fallback 2.5e-4, batch 256) | 1.4821 | 1.4844 |
| `128_d768x12_ep1` final (d768, epoch 1, usui lr 8e-3, batch 256) | 1.4807 | 1.4811 |
| `128_d768x12_ep2` final (d768, epoch 2, usui lr 4e-3, batch 256) | 1.4631 | 1.4643 |
| `128_d768x12_ep3` final (d768, epoch 3, usui lr 2e-3, batch 256) | 1.4557 | 1.4573 |
| `128_d768x12_ep4` final (d768, epoch 4, usui lr 1e-3, batch 256) | 1.4518 | 1.4536 |
| `128_d1024x12_ep1` final (d1024, epoch 1, usui lr 8e-3, batch 256) | 1.4621 | 1.4613 |
| `128_d1024x12_ep2` final (d1024, epoch 2, usui lr 4e-3, batch 256) | 1.4428 | 1.4439 |
| `128_d1024x12_ep3` final (d1024, epoch 3, usui lr 2e-3, batch 256) | 1.4343 | 1.4359 |
| `128_d1024x12_ep4` final (d1024, epoch 4, usui lr 1e-3, batch 256) | 1.4302 | 1.4323 |
| `128_d768x12_noconv_ep1` final (d768, epoch 1, usui lr 8e-3, conv deleted; 854 img/s against 834) | 1.4805 | 1.4807 |
| `128_d768x12_lr1e-2_ep1` final (d768, epoch 1, usui lr 1e-2, one change from `_ep1`) | 1.4799 | 1.4797 |
| `128_d768x12_finalr64_ep1` final (d768, epoch 1, usui lr 8e-3, final linear in Usui at rank 64, one change from `_ep1`) | 1.4911 | 1.4914 |
| `128_d768x12_slice_ep1` final (d768, epoch 1, usui lr 8e-3, no final layer: first 128 stream channels are the output, identity patch embed; 838 img/s) | 1.4745 | 1.4749 |
| `128_d768x12_solver1_ep1` final (slice arm plus a tied zero-init 768 x 768 linear residual, 4 iterations, on Usui; `tangent_live_fraction` 0.96 against 0.99, `projected_grad_norm` 0.020 against 0.0043 from step 1000 on) | 1.4816 | 1.4817 |
| `128_d768x12_solvermlp_ep1` final (slice arm plus a tied zero-init SwiGLU residual, 768 -> 1536 -> 768, 4 iterations, on Usui; 771 img/s, `tangent_live_fraction` 0.95, `projected_grad_norm` 0.0058) | 1.4781 | 1.4785 |

Parameters of the slice-readout model (2026-11-03, MLP hidden 4 x width, counted from `DiT`): 125.4M at width 768 and depth 12, 222.5M at 1024 and 12, 293.8M at 1024 and 16. The `128_d768x12_*` rows above count the earlier architecture (final layer, conv, MLP at 2 x width) except where the arm says otherwise.

- Same lr order as on shards 12 and 40; 2e-4 ends 0.046 above 2e-3, its training loss (mean of the last 10 records) 1.396 against 1.326, so 2e-4 is not too big but too small. The caption gap is under 0.001, and negative at step 1000: the captions here are one short line each (PLAN B3).

Seed merge (scratch `seedmerge/`): the shuffled file split into three disjoint thirds, one 5000-step child on each at lr 2e-3 (warmup 250), their step-5000 weights merged by multislerp, then 5000 steps from the merged weights (fresh optimizer state) over the whole file in the sweep's order, warmup 752. Same scoring, shard 12 caption:

| | loss |
|---|---|
| `128_sweep_lr2e-3`, step 5000 | 1.4060 |
| children 0 / 1 / 2, step 5000 | 1.4059 / 1.4058 / 1.4036 |
| the merge | 2.5958 (null 2.5513) |
| 5000 steps from the merge | 1.3970 (step 1000: 1.4837) |

Sweep merge (scratch `sweepmerge/`, same scoring as the Lunara table above): the arms' final checkpoints share their init and data order (seed 0). Lunara caption / null:

| | caption | null |
|---|---|---|
| `lr4e-3` final | 1.5106 | 1.5109 |
| multislerp of 2e-3 + 4e-3 | 2.1471 | 2.1865 |
| multislerp of 1e-3 + 2e-3 + 4e-3 | 2.2553 | 2.2354 |
| multislerp of all four | 2.2702 | 2.2631 |
| mean of 1e-3 + 2e-3 + 4e-3 | 2.4683 | 2.4436 |
| mean of all four | 3.0589 | 3.0599 |

- Before: merging the 4 lr sweep checkpoints might be crazy again, but there was hope it would be good. Every merge is broken, like the seed merge's, though these arms share init and data order.

Weight norms of `128_sweep_lr4e-3` (scratch `norms.py`; Frobenius norm over all blocks for `blocks.*`, init rebuilt at seed 0):

| | init | 1k | 2k | 4k | 8k | 12k | 15042 |
|---|---|---|---|---|---|---|---|
| `qkv` | 135.8 | 151.8 | 178.6 | 224.0 | 293.6 | 348.5 | 384.8 |
| `w_up` | 110.8 | 120.0 | 134.9 | 156.6 | 195.5 | 228.2 | 249.7 |
| `w_down` | 78.4 | 85.2 | 96.5 | 117.5 | 146.1 | 166.7 | 179.7 |
| `out` | 78.4 | 83.4 | 92.2 | 107.4 | 131.6 | 150.8 | 163.4 |
| `final` (fallback) | 0 | 6.3 | 9.2 | 13.8 | 19.7 | 23.8 | 26.2 |
| `ada.1` (fallback) | 25.1 | 32.7 | 44.0 | 57.1 | 70.5 | 80.2 | 86.4 |
| null embedding | 1.00 | 1.19 | 1.39 | 1.70 | 2.38 | 2.95 | 3.36 |

- No norm has levelled off; the Usui matrices grow by about 2.5–3× their init, slowing (qkv +17 per 1k steps from 4k to 8k, +12 from 12k to 15042). Nothing sets a target norm: no weight decay on either optimizer, and Usui's fixed step does not depend on `‖W‖`. The same absolute step on a weight 2.8× larger is a 2.8× smaller relative step: an implicit decay with no schedule.

- The merge alone is broken, worse than any child, and better with the null embedding than with the caption; 5000 steps from it end 0.009 below the single run at step 5000, after 4× its compute. Decided: no merges, fresh init each time.
