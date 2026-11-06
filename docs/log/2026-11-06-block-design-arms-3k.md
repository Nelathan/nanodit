# Block-design arms on the conv-in-3/4 base, 3000 steps (F4, 2026-11-06)

Run log. The dense reading is in FACTS, Architecture at d768x12 (2026-11-06).


Branch `conv-meta`, base `128_d768x12_conv4_3k`; each arm adds one knob. Eval caption loss on Lunara: at 128 px the scores of the run, at 256 px the 128 px checkpoint scored on a 16×16 token grid it never trained on (`scripts/score_checkpoint.py`, 256 px Lunara from the fused shard). Losses at 256 px sit lower than at 128 px, so only the comparison between models at one resolution carries meaning.

| arm | 1000 | 2000 | 3000 | at 256 px | img/s |
|---|---|---|---|---|---|
| `swpin_3k`, all attention | 1.5532 | 1.5023 | 1.4855 | 1.4353 | 682 |
| `conv4_3k`, the base | 1.5474 | 1.5037 | 1.4884 | 1.4045 | 901 |
| `conv4_metaembed_3k`, patch embed writes the meta channels only (8 px token) | 1.5472 | 1.5037 | 1.4869 | 1.4030 | 899 |
| `conv4_pmlp_3k`, PaLM parallel MLP | 1.5491 | 1.5052 | 1.4908 | 1.4070 | 904 |
| `conv4_nope_3k`, no rotary position | 1.5488 | 1.5052 | 1.4895 | 1.4079 | 895 |
| `conv4_rope25_3k`, a quarter of the rotary frequencies | 1.5481 | 1.5033 | 1.4883 | 1.4035 | 902 |
| `conv4_gqa3_3k`, 3 key/value heads against 12 | 1.5477 | 1.5054 | 1.4888 | 1.4037 | 912 |

At 256 px the conv base is 0.031 below the all-attention baseline, against 0.003 above it at 128 px. Every knob sits within 0.003 of the base at 128 px; at 256 px NoPE is +0.0034 (from +0.0011), the parallel MLP +0.0025, meta-only embed −0.0015, quarter RoPE −0.0010, GQA −0.0008.
