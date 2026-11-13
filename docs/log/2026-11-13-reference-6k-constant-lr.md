# The reference at constant lr, captions without prefix (2026-11-13)

`configs/128_d768x12_6k.toml` at commit `ff00690`: d768x12, conv-conv-conv-attention, cross attention in every
attention block, usui lr 4e-3 constant after 376 warmup steps, every matrix under its init-norm ceiling, the trial set
(text-to-image-2M shards 0–15) re-encoded without the caption prefix. 6000 steps of batch 256, seed 41, 40 minutes.
Run directory `/mnt/mars/ai/tiny-dit/runs/128_d768x12_6k_trialset` (renamed from `128_d768x12_6k`, the name it ran and logged to wandb under).

Caption eval loss on the Lunara eval set (`score.jsonl`), against `128_d768x12_gdn_6k` (the earlier trunk reference:
lr 2e-3 decaying linearly over the 6000 steps, norms pinned, prefixed captions, no cross attention):

| Step | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| `128_d768x12_gdn_6k` | 1.5598 | 1.5127 | 1.4965 | 1.4891 | 1.4820 | 1.4771 |
| `128_d768x12_6k` | 1.5468 | 1.5145 | 1.5038 | 1.4985 | 1.4923 | 1.4867 |

R² at 6000: 0.6208. The earlier cross-attention arm `128_d768x12_arch_cross_6k` ends at 1.4780.

Read: ahead by 0.013 at 1000 steps, behind by 0.0096 at 6000; the earlier runs' lr decay lowers their late loss, and
the run without one is not annealed. Four things changed at once (schedule, lr, bound, captions), so the comparison
attributes nothing to any one of them. Accepted as the baseline unannealed (2026-11-13); it is superseded by the same
config on the merged training set (archive 2026-11-13-merged-training-set).
