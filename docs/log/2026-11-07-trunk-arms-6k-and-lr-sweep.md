# Trunk arms at d768x12 (6000 steps) and the d768x16 lr sweep (3000 steps), 2026-11-07

Trunk code (62b5727) except gdn (branch `gdn-conv`); trial set `trial_128_seq512_shuffled_0-15`, scored on `lunara_eval_128_seq512` (caption loss at 10 fixed t). These scores are on a different data and eval set than the conv-meta 6k references (`conv4_gqa3_rope25_meta` 1.4655, `conv4` 1.4680, `attn` 1.4679), so compare within a table only.

| Run | 3000 | 6000 |
|---|---|---|
| arch_6k (conv 3/4, pooled ada) | 1.5042 | 1.4788 |
| arch_cross_6k (ada + cross) | 1.5008 | 1.4780 |
| arch_crossonly_6k (cross, no pooled ada) | 1.5013 | 1.4771 |
| arch_gdn_6k (short-conv linear attention with output gate; NOT GDN2) | 1.5070 | 1.4849 |
| prime stage 1 (shift 4, t sigma 1e-4), 3000 steps | 2.3794 | |
| prime stage 2 (continued, normal t), 3000 more | | 1.4914 |

| d768x16, 3000 steps | 1000 | 2000 | 3000 |
|---|---|---|---|
| usui lr 2e-3 | 1.5596 | 1.5129 | 1.4933 |
| 1e-3 | 1.5846 | 1.5311 | 1.5101 |
| 5e-4 | 1.6163 | 1.5568 | 1.5332 |

Reads: cross attention with or without the pooled ada ends within 0.002 of plain ada at 6k; the pooled caption is not needed once cross attention exists, and cross adds nothing yet. The short-conv linear attention is 0.006 worse than the plain conv; it has none of GDN2's decay, erase or write gates or its delta rule, and writes the meta channels only, so it says nothing about GDN2. Priming then continuing is 0.013 worse than the same 6000 steps straight, but stage 2 here was ordinary flow training, whereas the paper's stage 2 is distributional post-training for a one-step generator (docs/references/priming-threads.md), so this does not test the paper's claim. Neither loss measures prompt following; that needs samples. The loss falls monotonically with lr up to 2e-3, so the peak lies at or above it; 4e-3 and a trunk all-attention baseline are queued.

## Scored at 256 px (the 128 px checkpoints on an unseen 16×16 grid, Lunara art eval, 6000 steps)

| Run | 256 px caption loss |
|---|---|
| arch_crossonly_6k | 1.3740 |
| arch_6k | 1.3769 |
| arch_cross_6k | 1.3769 |
| prime stage 2 | 1.3891 |
| arch_gdn_6k | 1.3918 |

The ordering at 256 px matches 128 px: the GDN-style mixer and priming lose, cross attention is level with ada (crossonly 0.003 ahead).
