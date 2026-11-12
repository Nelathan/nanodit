# REPA, first arm (2026-11-08)

REPA as built in `topics/priming.md` (commit 7f5bfb4): the output of block 3 (the first attention block) through
REPA's projector, aligned by cosine to EmbeddingGemma 2's vision-tower features of the clean image (decoded by Primal,
upsampled to 384 px, one pooled feature per token), weight 0.5, on for the first 3000 of 6000 steps, then off. Trunk
model (GDN mixer) at d768x12, batch 256, lr 2e-3, trial set, seed 41; caption loss on `lunara_eval`. Config
`configs/128_d768x12_repa_6k.toml`, queue `/mnt/mars/ai/tiny-dit/runs/queue_repa_2026-11-07.sh`. Scratch is the
trunk run of `2026-11-07-masked-priming.md`, same seed and data order; that config reproduced a separately seeded run
to 0.0006.

| step | scratch | REPA | REPA − scratch |
|---|---|---|---|
| 1000 | 1.5598 | 1.5648 | +0.0050 |
| 2000 | 1.5127 | 1.5140 | +0.0013 |
| 3000 | 1.4965 | 1.4979 | +0.0014 |
| 4000 | 1.4891 | 1.4860 | −0.0031 |
| 5000 | 1.4820 | 1.4805 | −0.0015 |
| 6000 | 1.4771 | 1.4748 | −0.0023 |
| 6000 at 256 px | 1.3728 | 1.3715 | −0.0013 |

REPA − scratch by t (t the noise share):

| step | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| 3000 | +0.002 | +0.004 | +0.004 | +0.004 | +0.002 | +0.001 | −0.001 | −0.003 | 0.000 | 0.000 |
| 6000 | −0.001 | −0.001 | −0.002 | −0.003 | −0.004 | −0.005 | −0.006 | −0.004 | +0.001 | +0.002 |
| 6000 at 256 px | 0.000 | 0.000 | −0.001 | −0.002 | −0.003 | −0.004 | −0.005 | −0.003 | +0.002 | +0.002 |

- Feature cosine (train, block 3 projected against the tower): 0.31 at step 200, 0.42 at 500, 0.48 at 1000, 0.51 at
  2000, 0.52 at 3000, still rising slowly when removed.
- Speed: 89 img/s while aligned against 745 without (8.4×). Not split by part; the projector is two small matrices,
  so the cost is the Primal decode, the 3× upsample and the tower on every sample. The 3000 aligned steps took 2.4 h,
  the rest 17 min. GPU memory 4.8 GB at step 250 (nvidia-smi).
- Removal: no step in the train loss. At 3000 REPA's train loss is 0.0025 above scratch's, at 3050 level (0.0003).

Reading:
- REPA costs loss while it is on (0.005 behind at 1000, 0.0014 at 3000) and wins after it is removed: 0.0023 ahead at
  6000, 0.0013 at 256 px; the reproduction gap was 0.0005 and 0.0006, so about four and two times it. The first
  priming arm to beat scratch.
- The gain sits at the middle of t (40 % to 66 % noise, up to 0.006) and is a small loss at the noisy end (74 % and
  84 %). Reading: the aligned block learns image content the flow loss reaches slowly at mid noise; at high noise
  the clean-image features say little about the target the network can still use, and the alignment competes with it.
  While on, the alignment pulls block 3 toward features the flow loss does not need yet, so it lags; once free, the
  representation it left pays.
- Per wall-clock it loses by far: 2.4 h for 3000 steps against about 17 min. Any longer use needs the features
  stored with the latents (768 per token, 64 tokens per 128 px image: 98 KB per image in bf16), or a cheaper target.
- Not measured: a shorter span (the cosine is near its plateau by 2000), other weights or blocks, and whether the gain
  holds or fades past 6000 steps.
