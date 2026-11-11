# 1000-step ablations at 128 px (2026-11-02)

Moved verbatim from `FACTS.md` on 2026-11-11. These reads are of the model before the 2026-11-06 architecture merge (width 512 at first, a final linear, no branch norm, no pin); the numbers are the record of that model, not of the current one.

## 1000-step ablations at 128 px (2026-11-01 23:56 to 2026-11-02 00:29)

tiny-dit `8eb82be` (the final layer's rows in the shared projection, text through `rms_norm`; code unchanged to `050dffa`), usuitrack `619b6e8`, the shuffled shards 0–10, seed 0, so every arm sees the same rows in the same order. Each arm is the smoke settings (usui lr 2e-3, beta 0.99, rank fraction 0.1, fallback lr 1e-3, fallback betas (0.9, 0.99), warmup 100, batch 256 as 4 × 64) for 1000 steps with one change (batch 64 also takes 4000 steps and warmup 400). Configs in the run directories, the two launchers that patch `nanodit.optim` next to their runs, the queue at `runs/ablations_queue_2026-11-01.sh`; one seed per arm, queued in series. Loss and the other columns but img/s are means of the records over images 128k–256k (steps 501–1000; batch 64: 2001–4000); img/s spans all records. t bins 1 to 10, low t first.

| arm | change | loss | Δ base | bin 1 | bin 2 | bin 3 | bin 5 | bin 10 | `grad_moment_cosine` | img/s |
|---|---|---|---|---|---|---|---|---|---|---|
| `128_abl_base` | none | 1.643 | | 1.536 | 1.581 | 1.647 | 1.742 | 1.209 | −0.07 | 765–844 |
| `128_abl_final_fallback` | final linear on the fallback at 1e-3, not its r = 64 frame | 1.567 | −0.076 | 1.193 | 1.333 | 1.485 | 1.679 | 1.210 | −0.10 | 839–865 |
| `128_abl_fallback_lr3e-4` | fallback lr 3e-4 | 1.657 | +0.014 | 1.581 | 1.614 | 1.669 | 1.751 | 1.215 | −0.03 | 827–856 |
| `128_abl_fallback_b0999` | fallback betas (0.9, 0.999) | 1.641 | −0.002 | 1.533 | 1.578 | 1.645 | 1.741 | 1.208 | −0.07 | 834–855 |
| `128_abl_batch64` | batch 64 as 1 × 64, 4000 steps, warmup 400 | 1.516 | −0.127 | 1.258 | 1.363 | 1.469 | 1.587 | 1.210 | −0.08 | 350–355 |

- Noise: the run-to-run spread of the GPU smoke runs is at most 0.003 per window (one seed pair, 300 steps, an earlier build). A nearer yardstick is 0.011, the gap between the base arm and the 10-pass run over the same window (below), for a changed model and init. Against it the final linear on the fallback and batch 64 stand out, the lower fallback lr (+0.014) is borderline, and betas 0.999 cannot be told apart.
- The final linear on the fallback gains most at low t, tapering with t: bins 1–7 lower by 0.34, 0.25, 0.16, 0.10, 0.06, 0.04, 0.02, bins 8–10 equal. At the last record its bin 1 is 1.182, below the 10-pass run's 1.281 at step 2100 (FACTS, `128_2e-3_0-10_10ep`). Its `fallback_grad_norm`, which now includes the final linear, is 0.064 against 0.017 (window means; 0.055 against 0.020 at the last record).
- The lower fallback lr is worse in every bin, by 0.005 to 0.044, most at low t.
- Batch 64 is better per image in bins 1–9, by 0.28 at bin 1 down to 0.014 at bin 9, and equal in bin 10. It trains at 353 img/s against about 850. By `imgs_per_s` it took 723 s; its last record (steps 3601–4000), 1.496, is against 1.510 for the 10-pass run's record at step 2200 (713 s), a batch-256 run of the model before `8eb82be` that ran at about 790–800 img/s over that stretch. Less the 0.011 of the model change, about a tie per training time. `tangent_live_fraction` 0.992–0.993 in the window, as at batch 256.
- The same window of the 10-pass run (same batches, the model before `8eb82be`, its init drawn differently because `ada` changed shape) is 1.654, 0.011 above the base arm; bin 1 1.597 against 1.536.
- Eval grids (seed 42, by eye): at step 1000 the base arm is blotches of colour; in the final-fallback arm the cat (cell 0) is a dark figure on white and the studio shot (cell 13) white, most other cells as noisy as the base. Batch 64 at step 4000: the warrior (cell 2) is high-contrast black and white, the ice cliff (cell 1) nearly flat green.
