# Masked priming, mode A (2026-11-07)

Masked priming as built in `topics/priming.md`: each 8×8 latent token pure noise with probability equal to a share that falls linearly from 90% to 0; the other tokens at the sample's logit-normal t (sigma 1, shift 1), which the network is told; loss on all tokens. Trunk model (GDN mixer) at d768x12, 6000 steps, batch 256, lr 2e-3, trial set, caption loss on `lunara_eval` (never masked). Queue `/mnt/mars/ai/tiny-dit/runs/queue_mask_2026-11-07.sh`, commit 77f75b0.

| step | scratch | 90% → 0 over 3000 | 90% → 0 over 6000 |
|---|---|---|---|
| 1000 | 1.5598 | 1.6404 | 1.6594 |
| 2000 | 1.5127 | 1.5532 | 1.5816 |
| 3000 | 1.4965 | 1.5168 | 1.5423 |
| 4000 | 1.4891 | 1.5008 | 1.5218 |
| 5000 | 1.4820 | 1.4925 | 1.5059 |
| 6000 | 1.4771 | 1.4860 | 1.4944 |
| 6000 at 256 px | 1.3728 | 1.3830 | 1.3927 |

Caption loss by t at 6000 (the score's 10 fixed t, t the noise share):

| run | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| scratch | 1.285 | 1.388 | 1.446 | 1.491 | 1.529 | 1.561 | 1.583 | 1.583 | 1.532 | 1.372 |
| over 3000 | 1.291 | 1.400 | 1.462 | 1.506 | 1.543 | 1.572 | 1.591 | 1.587 | 1.535 | 1.372 |
| over 6000 | 1.297 | 1.411 | 1.474 | 1.520 | 1.556 | 1.584 | 1.601 | 1.594 | 1.536 | 1.371 |

- The trunk from scratch reproduces the branch's `gdnall` arm (1.4766, 1.3722) to 0.0006, so the merged GDN default is that mixer.
- Masking at this setting costs steps. The 3000-step ramp's loss at 6000 is scratch's at about step 4430 (linear between 4000 and 5000). After the mask is off the gap closes slowly (0.0117 at 4000, 0.0105 at 5000, 0.0090 at 6000). The 6000-step ramp is worse throughout (0.017 behind at 6000). Throughput is unchanged (727 against 740 img/s at step 3000).
- Both ramps lose at low and middle t and are level at the highest (0.84); the same pattern as single-level denoising (`2026-11-07-priming-read.md`).
- My reading: a pure-noise token trains only t = 1, where the single-level test already found the network level with scratch after few steps, and each masked token is one not trained at the sample's t (at 90%, about 6 of 64 tokens carry the ordinary flow signal). Variants that keep every token at a useful t (per-token t; masked tokens strictly noisier but not pure noise) would keep that signal and still make the network read its neighbours; untested.
