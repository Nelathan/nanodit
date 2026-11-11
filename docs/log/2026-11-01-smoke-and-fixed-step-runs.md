# Smoke, fixed-step and 2e-3 runs at 128 px (2026-11-01)

Moved verbatim from `FACTS.md` on 2026-11-11. These reads are of the model before the 2026-11-06 architecture merge (width 512 at first, a final linear, no branch norm, no pin); the numbers are the record of that model, not of the current one.

## GPU smoke runs at 128 px (2026-11-01)

RTX 4070 Super, `configs/128_smoke.toml` as it was then (300 steps, batch 256 as 4 × 64, usui lr 3e-3, beta 0.95, fallback lr 1e-3, shift 1.0, shards 0–3), Usui `lab` at `be3f298`, option (a) split. The fp32 runs are tiny-dit `61448e4`; the bf16 run is the commit that adds this section. Runs and JSONL logs in `/mnt/mars/ai/tiny-dit/runs/<name>/`; the summary script is not in the repo. Mean loss per step window, median img/s from step 11:

| run | precision | seed | 1–50 | 51–100 | 101–200 | 201–300 | img/s |
|---|---|---|---|---|---|---|---|
| `128_smoke` | fp32 + TF32 | 0 | 2.0918 | 1.7427 | 1.5986 | 1.5422 | 756 |
| `128_smoke_fp32_seed1` | fp32 + TF32 | 1 | 2.0888 | 1.7419 | 1.6002 | 1.5429 | 745 |
| `128_smoke_bf16` | bf16 model, stochastic rounding | 0 | 2.0981 | 1.7439 | 1.5949 | 1.5389 | 862 |
| `128_smoke_zero_init` | bf16; final linear on Usui with a norm floor √128, zero-gradient rule | 0 | 2.4039 | 2.1247 | 1.9701 | 1.8528 | 843 |
| `128_smoke_nofit_final_adam` | bf16; zero-gradient rule, final linear on the fallback | 0 | 2.0715 | 1.7203 | 1.5857 | 1.5340 | 849 |

The last two ran on usuitrack `zero-init` at `a32658b` from uncommitted tiny-dit code (the floor build of A1; the second with a scratch launcher putting the final linear back on the fallback); neither split is in the tree now.

- Run-to-run spread (the two fp32 seeds): at most 0.003 per window, 0.0007 over steps 201–300.
- bf16 against fp32 at seed 0: +0.006 over steps 1–50, −0.0033 over 201–300. Over the last 100 steps, per t-bucket, bf16 is within 0.012 of fp32 in every bucket but the lowest (t < 0.1: 1.283 against 1.307 and 1.313). One bf16 seed; a 100-step window mean has a standard error near 0.002, so the late difference is not resolved.
- Loss by t bucket, steps 201–300 (fp32 seed 0): 1.31 at t < 0.1, peaking at 1.65 around t = 0.5, 1.16 at t > 0.9.
- Usui frame after the step-1 zero gradient, `grad_capture` (mean over matrices, then over steps; fp32 seed 0, seed 1, bf16): 0 at step 1, 0.017 at step 2, 0.33 / 0.32 / 0.33 over steps 3–11, 0.58 / 0.57 / 0.59 over 21–30, 0.80 / 0.78 / 0.80 over 51–60, 0.96 / 0.96 / 0.95 over 101–300. Under Molegrad the raw gradient norm is a probed estimate, so single steps exceed 1 (37–51 of 300 per run, max 1.16).
- `grad_moment_cosine` (same windows): +0.44 to +0.51 over steps 3–11, −0.07 to −0.10 over 91–100, −0.23 to −0.26 over 101–150, −0.36 to −0.39 over 151–300, in all three runs. Usui reads negative as overshoot, to be read against lr; only 3e-3 has run.
- Final linear Frobenius norm at steps 100 / 200 / 300: 3.88 / 3.80 / 3.65 on the fallback (`128_smoke_bf16`), 3.56 / 3.54 / 3.50 (`128_smoke_nofit_final_adam`), 1.85 / 2.09 / 1.74 under the floor (`128_smoke_zero_init`), which it never reached.
- With the zero-gradient rule, `grad_capture` over steps 3–11 is 0.92 (final linear on the fallback) and 1.02 (on Usui), against 0.33 without; over steps 21–60 the first dips to 0.79–0.82, and from step 101 it reads 0.94–0.97 in all runs.
- Loss by t bucket under the floor, steps 201–300: 2.70 at t < 0.1 against 1.28 for `128_smoke_bf16`, the gap shrinking to 0.01 above t = 0.8.
- The eval grid at step 300 follows each prompt's colour and mood, no shapes yet; faint vertical streaks.
- Step time with random data, model only (scratch script, 40 steps): fp32 + TF32 792 img/s, 2.58 GiB peak; bf16 autocast (Molegrad fix uncommitted in the usuitrack worktree `molegrad-autocast`, not used) 879 img/s, 2.17 GiB.

## Fixed-step runs at 128 px (2026-11-01)

Uncommitted tiny-dit code, by its diff the same as `b3adad8` in `src/` and `scripts/` (every matrix on Usui's fixed step at lr 2e-2, final linear included, frame rank fraction 0.05; linear warmup), usuitrack `zero-init` at `e746901`, otherwise `configs/128_smoke.toml` as it was then (300 steps, batch 256, fallback lr 1e-3, bf16, seed 0). The `final_r64` runs used a scratch launcher putting the final linear in its own Usui group at rank fraction 1.0 (r = 64, the cap at half the shorter side), at that group's full fixed step and at its step scaled by √(13/64) to the r = 13 step norm (`eq`). Same windows as above:

| run | warmup | final linear | 1–50 | 51–100 | 101–200 | 201–300 | img/s |
|---|---|---|---|---|---|---|---|
| `128_fixed_2e-2_w20` | 20 | r = 13 | 2.4571 | 2.1345 | 1.9808 | 1.8644 | 778 |
| `128_fixed_2e-2_w50` | 50 | r = 13 | 2.5553 | 2.1587 | 1.9963 | 1.8747 | 808 |
| `128_fixed_2e-2_w100` | 100 | r = 13 | 2.6490 | 2.2037 | 2.0176 | 1.8919 | 749 |
| `128_fixed_2e-2_final_r64` | 50 | r = 64 | 2.4175 | 1.8252 | 1.7158 | 1.6454 | 841 |
| `128_fixed_2e-2_final_r64eq` | 50 | r = 64, r = 13 step norm | 2.5432 | 1.8422 | 1.7095 | 1.6417 | 799 |

- Loss by t bucket, steps 201–300, lowest to highest t: w50 2.77 / 2.54 / 2.30 / 2.10 / 1.94 / 1.76 / 1.63 / 1.48 / 1.32 / 1.17; r64 1.85 / 1.80 / 1.77 / 1.75 / 1.73 / 1.66 / 1.60 / 1.47 / 1.32 / 1.17; `128_smoke_nofit_final_adam` (final linear on the fallback; also the earlier relative step at 3e-3 and no warmup, so not a same-code control) 1.27 / 1.37 / 1.50 / 1.59 / 1.64 / 1.64 / 1.58 / 1.46 / 1.31 / 1.15. The gap shrinks with t: 1.50 in the lowest bucket, 0.06 at 0.6–0.7, under 0.02 above.
- Warmup length at 300 steps: 1.8644 / 1.8747 / 1.8919 over steps 201–300 (w20 / w50 / w100), spread 0.028, shorter lower.
- `grad_capture` over steps 101–300: 0.94 / 0.92 / 0.89 (w20 / w50 / w100), 0.94 on the fallback run. `grad_moment_cosine` over 251–300: −0.34 / −0.34 / −0.33, −0.37 on the fallback run.
- The final linear at step 300 (`[128, 512]`; stable rank Σs²/s₁², and singular values above 10% of s₁): w20 stable rank 12.9, 41 of 128; r64 26.9, 93; fallback 45.9, 128; `128_smoke_bf16` 41.7, 128.
- How the r = 13 frame on the final linear moves (`scripts/probe_frame.py`; w50 at 100 / 200 / 300): stable rank 9.5 / 9.7 / 11.3; directions above 10% 20 / 32 / 38; share of ‖W‖² outside the current frame 0.17 / 0.27 / 0.37; ‖F_prevᵀ F‖²/r against the checkpoint 100 steps earlier 0.80 / 0.89. Under r = 64 (the same reads, the final linear indexed in its own last group) that overlap is 0.98 / 0.99 and 3–10% of ‖W‖² lies outside the frame.

### `128_fixed_2e-2_9k` (2026-11-01)

The committed design at `b3adad8` plus the wandb unit as first written (its diff is next to the run, `128_fixed_2e-2_9k.uncommitted.diff`; it differs from `ac730aa` only in logging, tests and docs): `configs/128_smoke.toml` with 9000 steps, warmup 450, eval every 500, checkpoint every 1000. About 200k rows sampled with replacement, so each seen about 11 times. 13:42 to 14:29, median 846 img/s from step 11. wandb `pink-marker/tiny-dit`, run `w0j9lfhv`.

| steps | loss | t < 0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | > 0.9 | `grad_capture` | `grad_moment_cosine` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 201–300 | 1.9587 | 2.985 | 2.719 | 2.441 | 2.212 | 2.024 | 1.829 | 1.681 | 1.508 | 1.337 | 1.177 | 0.95 | −0.22 |
| 901–1000 | 1.6702 | 2.166 | 2.046 | 1.918 | 1.812 | 1.717 | 1.631 | 1.535 | 1.429 | 1.276 | 1.149 | 0.90 | −0.38 |
| 1901–2000 | 1.5275 | 1.671 | 1.636 | 1.623 | 1.609 | 1.569 | 1.543 | 1.493 | 1.413 | 1.277 | 1.163 | 0.90 | −0.37 |
| 2901–3000 | 1.4663 | 1.467 | 1.484 | 1.508 | 1.517 | 1.522 | 1.505 | 1.458 | 1.386 | 1.268 | 1.164 | 0.88 | −0.34 |
| 4901–5000 | 1.4197 | 1.307 | 1.373 | 1.425 | 1.460 | 1.472 | 1.455 | 1.438 | 1.387 | 1.267 | 1.136 | 0.87 | −0.32 |
| 6901–7000 | 1.3878 | 1.267 | 1.329 | 1.380 | 1.415 | 1.427 | 1.430 | 1.419 | 1.370 | 1.261 | 1.150 | 0.87 | −0.31 |
| 8901–9000 | 1.3695 | 1.247 | 1.304 | 1.356 | 1.394 | 1.398 | 1.408 | 1.402 | 1.365 | 1.265 | 1.132 | 0.85 | −0.29 |

- The low-t gap of the 300-step reads closes with time: the t < 0.1 bucket goes 2.99 → 1.25, the buckets above t = 0.7 barely move. Loss still falls at 9000 (−0.012 over the last 1000 steps); constant lr, no decay.
- The final linear's frame (`scripts/probe_frame.py`), checkpoints 1000 to 9000: overlap with the previous checkpoint 0.992 to 0.995 every 1000 steps; stable rank 14.0 / 12.7 / 11.4 / 10.8 / 9.6 / 8.7 / 9.0 / 8.3 / 7.8; directions above 10% of s₁ 48 / 65 / 70 / 73 / 67 / 65 / 68 / 66 / 63 of 128; share of ‖W‖² outside the current frame 0.44 / 0.42 / 0.40 / 0.38 / 0.36 / 0.35 / 0.34 / 0.33 / 0.33. After step 1000 the frame holds its subspace; the weight's energy concentrates in it, and what lies outside was written before step 1000.
- The eval grid at step 9000 (16 prompts × 2 seeds, guidance 4): colour, light and coarse layout per prompt, no recognisable objects; strong vertical streaks, mostly in the seed-0 column, and blocky horizontal banding.
- The log ends with `shutdown can only be called once` after wandb's sync; exit code 0, all 9000 steps and checkpoints written.

### `128_fixed_2e-2_final_r64_9k` (2026-11-01)

The same run with the final linear in its own Usui group at rank fraction 1.0 (r = 64), full fixed step: code `e9f156f` plus the scratch launcher copied next to the run (`128_fixed_2e-2_final_r64_9k.launcher.py`). 14:30 to 15:16, median 872 img/s from step 11. wandb run `h483k5gj`. Against `128_fixed_2e-2_9k`, loss and the t < 0.1 bucket:

| steps | r = 64 loss | r = 13 loss | r = 64, t < 0.1 | r = 13, t < 0.1 |
|---|---|---|---|---|
| 201–300 | 1.6985 | 1.9587 | 1.976 | 2.985 |
| 901–1000 | 1.5146 | 1.6702 | 1.474 | 2.166 |
| 2901–3000 | 1.4369 | 1.4663 | 1.310 | 1.467 |
| 4901–5000 | 1.4048 | 1.4197 | 1.257 | 1.307 |
| 6901–7000 | 1.3749 | 1.3878 | 1.238 | 1.267 |
| 8901–9000 | 1.3591 | 1.3695 | 1.226 | 1.247 |

- The gap narrows but holds at about 0.010 from step 7000 on, one seed each. Over steps 8901–9000 the r = 64 run is lower by 0.011 to 0.021 in every bucket below t = 0.6, by 0.007 at 0.6–0.7, and level above.
- Its final linear's frame also settles: overlap 0.981 to 0.993 per 1000 steps; stable rank 23.9 at 1000 to 15.7 at 9000; 121 to 126 of 128 directions above 10% of s₁; 11–14% of ‖W‖² outside the frame.
- The eval grid at step 9000 has more fine texture than the r = 13 grid; the vertical streaks and the horizontal banding are there in both, so the r = 13 frame does not cause them.

## Runs at usui lr 2e-3 (2026-11-01)

`configs/128_smoke.toml` at `e8ca690`: usui lr 2e-3, beta 0.9, rank fraction 0.1 (r = 64 on the final linear), warmup 100, batch 256 as 4 × 64, shards 0–3, records every 100 steps. Against the 2e-2 runs this changes four things at once (lr, beta 0.95 → 0.9, warmup 450 → 100, rank fraction 0.05 → 0.1 for every matrix but the final linear), so no single one is isolated. Configs and logs next to the runs.

### `128_fixed_2e-3_9k`, stopped at step 4300

Stopped by hand at 356–400 img/s per record, with the CPU precompute (about 10 cores) running alongside; the next run, alone, did 850 img/s. wandb run `jdixfisn`. Means per window, against the two 2e-2 runs:

| steps | loss 2e-3 | loss 2e-2 r64 | loss 2e-2 r13 | `grad_moment_cosine` 2e-3 | 2e-2 r64 | `projected_grad_norm` 2e-3 | 2e-2 r64 |
|---|---|---|---|---|---|---|---|
| 501–1000 | 1.544 | 1.545 | 1.726 | −0.39 | −0.36 | 0.029 | 0.018 |
| 1001–2000 | 1.461 | 1.481 | 1.577 | −0.45 | −0.33 | 0.042 | 0.013 |
| 3001–4000 | 1.388 | 1.426 | 1.448 | −0.43 | −0.34 | 0.044 | 0.012 |
| 4001–4300 | 1.377 | 1.414 | 1.431 | −0.42 | −0.34 | 0.042 | 0.012 |

- `grad_capture` ≈ 1.00 (0.87 at 2e-2); `relative_step` 0.0008 (window means 0.0031–0.0047); `tangent_live_fraction` 0.994–0.998 (1.000).
- The two-seed grid at step 4000 follows each prompt in colour and scene (the night street dark blue with vertical streaks, the frog green, the woman in red on green); the 2e-2 r64 grid at the same step has more high-frequency clutter and a shared window-like layout across prompts. The 4×4 grid of the new prompt set (seed 42), rendered from both step-4000 checkpoints into `<run>/eval_4x4/` by `runs/render_grid_4x4.py`: 2e-3 matches the monochrome warrior, the explosion in the park, the beach, the snowy mountains, the white studio backdrop and the desert; 2e-2 is busier, with blue clutter for the bear and the bulldog.

Fallback parameters, relative change ‖W_b − W_a‖/‖W_b‖ between checkpoints, mean over the parameters of a group:

| group | 1000→2000 | 2000→3000 | 3000→4000 |
|---|---|---|---|
| null embedding | 0.194 | 0.160 | 0.148 |
| final table | 0.358 | 0.139 | 0.092 |
| block tables | 0.353 | 0.236 | 0.191 |
| conv | 0.328 | 0.207 | 0.175 |
| biases | 0.253 | 0.156 | 0.119 |
| norm gains | 0.053 | 0.038 | 0.034 |
| Usui matrices, for scale | 0.161 | 0.125 | 0.108 |

- The null embedding's cosine to its init (the empty caption's embedding) is 0.763 at step 1000, then 0.778, 0.803, 0.819 at 2000–4000; its norm goes 1.058, 0.984, 0.927, 0.872 (init 1.000).

### `128_fixed_2e-3_b099_1k`: beta 0.99, 1000 steps

The same config at beta 0.99. wandb run `facr413t`, 850 img/s with nothing else running. Against `128_fixed_2e-3_9k` at the same step:

| step | loss β 0.99 | loss β 0.9 | `grad_moment_cosine` β 0.99 | β 0.9 | `projected_grad_norm` β 0.99 | β 0.9 |
|---|---|---|---|---|---|---|
| 300 | 1.769 | 1.720 | +0.09 | −0.19 | 0.011 | 0.019 |
| 500 | 1.652 | 1.613 | −0.01 | −0.35 | 0.012 | 0.022 |
| 1000 | 1.513 | 1.511 | −0.10 | −0.41 | 0.015 | 0.032 |

- Per t bucket at step 1000, beta 0.99 is lower by 0.024 below t = 0.1, by 0.013 at 0.1–0.2, and up to 0.007 higher at 0.3–0.7.
- `tangent_live_fraction` 0.997–0.999.
- The 4×4 grids at step 1000 (`eval_4x4/`) are close; the same compositions under the same noise.

### `128_2e-3_0-10_10ep`: 10 passes over shards 0–10 (2026-11-01)

Launched 2026-11-01 22:06 from the uncommitted tree that became tiny-dit `727d94c` and usuitrack `e7e5210` (the range-finder fit; wandb records the base `5fa9ce4`); edits between the launch and those commits touched comments, docstrings and tests. The DiT is the one before the final layer's rows moved into the shared projection (`8eb82be`). The smoke settings (usui lr 2e-3, beta 0.99, rank fraction 0.1, fallback lr 1e-3, warmup 100, batch 256 as 4 × 64), 21,482 steps = 10 passes of the 549,945-row shuffled file. Records average 100 steps; a pass's mean is over the records that end in it. wandb run `q49embk5`; config, checkpoints every 2000 steps and the eval grid every 1000 in the run directory.

| pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean loss | 1.692 | 1.479 | 1.441 | 1.419 | 1.405 | 1.396 | 1.390 | 1.385 | 1.380 | 1.377 |

- Loss at the last record 1.375. Per t bin (10 bins, low t first): 1.167, 1.250, 1.308, 1.355, 1.378, 1.416, 1.457, 1.443, 1.340, 1.177; the peak moved from bin 6 at step 2100 (1.593; ends 1.281 and 1.215) to bin 7, where it is in every record from step 5900 on.
- Usui, per record from step 1000 on: `grad_capture` 0.94–0.98, `grad_moment_cosine` −0.08 at step 1000, then −0.11 to −0.17; `tangent_live_fraction` 0.994–0.997; `relative_step` 0.0008 at step 1000 falling to 0.0005. No nonfinite gradients. `fallback_grad_norm` rises from 0.019 at step 1000 to 0.029–0.040 from step 10500.
- The 4×4 grid (seed 42), step 4000 against the last: the last is more saturated, the teal of step 4000 mostly replaced by blue. The ice-cliff landscape reads as blue sky and cloud, the warrior and the woman in the dark hall keep their monochrome layout, the beach is blue sea under a figure, the studio shot a dark figure on white, and the sign prompt an orange rectangular object. No coherent anatomy or faces at either step.
