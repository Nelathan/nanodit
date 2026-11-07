# Per-token t priming (2026-11-07)

Per-token t as built in `topics/priming.md` (commit 6276f44): each 8×8 latent token noised at its own t, the sample's
logit-normal t (sigma 1, shift 1) moved in logit by `spread`·N(0, 1); the network is told the sample's t; no masked
tokens. The spread falls linearly to 0 over the first 3000 of 6000 steps. Trunk model (GDN mixer) at d768x12, batch
256, lr 2e-3, trial set, seed 41; caption loss on `lunara_eval`, never primed. Queue
`/mnt/mars/ai/tiny-dit/runs/queue_spread_2026-11-07.sh`. Scratch is the trunk run of
`2026-11-07-masked-priming.md`; its noise draws differ from these runs' (the per-token draw is new), and the same
config reproduced a separately seeded run to 0.0006.

| step | scratch | spread 1.0 | spread 0.5 |
|---|---|---|---|
| 1000 | 1.5598 | 1.5885 | 1.5673 |
| 2000 | 1.5127 | 1.5219 | 1.5134 |
| 3000 | 1.4965 | 1.4980 | 1.4954 |
| 4000 | 1.4891 | 1.4893 | 1.4870 |
| 5000 | 1.4820 | 1.4831 | 1.4812 |
| 6000 | 1.4771 | 1.4792 | 1.4770 |
| 6000 at 256 px | 1.3728 | 1.3757 | 1.3728 |

Caption loss by t at 6000 (t the noise share):

| run | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| scratch | 1.285 | 1.388 | 1.446 | 1.491 | 1.529 | 1.561 | 1.583 | 1.583 | 1.532 | 1.372 |
| spread 1.0 | 1.286 | 1.391 | 1.450 | 1.495 | 1.532 | 1.563 | 1.584 | 1.583 | 1.534 | 1.374 |
| spread 0.5 | 1.285 | 1.388 | 1.446 | 1.491 | 1.528 | 1.560 | 1.583 | 1.583 | 1.534 | 1.373 |

- Spread 0.5 ends level with scratch (1.4770 against 1.4771, 1.3728 at 256 px both). It is behind at 1000 steps
  (0.0075, while the spread is wide) and 2000 (0.0007), and 0.001 to 0.002 ahead at 3000 to 5000: level, no
  measurable gain.
- Spread 1.0 ends 0.002 behind (0.003 at 256 px), lost at every t but 66 % noise; it starts 0.029 behind at 1000.
- Unlike masking, per-token t costs nothing at the gentler spread, and gains nothing measurable by 6000 steps. My
  reading: spreading t moves training signal between noise levels but adds none; at 0.5 the tokens still sit near the
  sample's t, so the run is nearly ordinary flow training.
