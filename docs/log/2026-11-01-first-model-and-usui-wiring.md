# First model and Usui wiring (2026-10-31 to 2026-11-01)

Moved verbatim from `FACTS.md` on 2026-11-11. These reads are of the model before the 2026-11-06 architecture merge (width 512 at first, a final linear, no branch norm, no pin); the numbers are the record of that model, not of the current one.

## Model and Usui wiring (2026-10-31, `src/nanodit/model.py`, `scripts/probe_usui.py`)

- Parameters at depth 12: 37,719,936 (2026-11-01, with the final layer's rows in the shared projection). Blocks 34.70M (qkv 9.44M, SwiGLU 18.87M, sigmoid gate 3.15M, out-proj 3.15M, conv 55k, tables 37k); shared adaLN projection 2.10M; time MLP 394k; text projection 394k; patch embed 66k; final linear 66k. 2-D 37.66M, 1-D 9,088, 4-D (conv) 55,296.
- Usui on `lab`, 5 steps at lr 1e-2: a zero-init 128×512 matrix stays at norm 0.0; a (512,1,3,3) conv weight and a (64,) vector are rejected; a shared-projection-like matrix with zero gate rows and std-0.02 shift/scale rows grows its gate rows 0 → 0.074 (matrix norm 2.56); a [6, 512] table is accepted (frame [6, 3]). In the zero-init case the fan-in matrix feeding the zero one does not move either: its gradient is zero while the other is.
- Molegrad and accumulation, wiring check only (synthetic latents, depth 2, CPU, batch 32, usui lr 3e-3, fallback lr 1e-3, 60 steps, identical batches per arm; scratch script, not in the repo, so not reproducible from it): Molegrad 1×32 and 4×8 agree to 4 digits at every step (loss 2.5807 → 2.0579). Step 1 (final linear on the fallback): every tracked gradient exactly zero, nothing tracked moves. By step 10 `tangent_live_fraction` 0.91, `relative_step` ≈ 98% of lr. After 60 steps `blocks.0.qkv` shares 0.38 of its step-1 frame (G path without Molegrad: 0.50, loss 2.0544).

## First frame fit on the GPU (2026-11-01)

- `test_accumulated_batch_trains_logs_and_checkpoints` (depth 2, micro-batch 2), a probe patched around `usuitrack.frame.fit`: `ada.1.weight` [3072, 512], bf16 gradient of matrix rank 5, its normalized side Gram with top eigenvalue 0.99997 and 510 of 512 below 1e-6, rank 125 asked. `torch.linalg.eigh` in fp32 on CUDA raises "failed to converge"; fp32 on CPU and fp64 on CUDA return.
- The test failed on the GPU from `a984e20` until usuitrack `e7e5210`, and passed before it (bisected by worktree); the whole suite passes on the CPU. Why that commit tips it is not isolated.
- The fit runs once per matrix, at its first nonzero gradient, on one micro-batch's gradient (`usuitrack/molegrad.py`, `usuitrack/optimizer.py`). The 128 px runs (micro-batch 64) did not fail; at most 64 samples feed `ada`'s gradient there, so its rank is at most 64 of 512.
- Since usuitrack `e7e5210` the fit is a random range finder with no `eigh` (archive 2026-11-01, first frame fit); the suite passes on the GPU (50 tests, 2026-11-01). Its capture against the exact fit, on this gradient and on the 78 first fits of the depth-12 model, is in usuitrack `docs/REVIEW_zero-init.md`.

## adaLN at init (2026-11-01, `scripts/probe_init.py`)

Depth 12, seed 0, 16 timesteps spread over (0, 1), text = the null embedding or random unit vectors. `c` per-element std 1.05 / 1.09. Block 0 shift/scale rows (shared projection + table): std 0.29 / 0.30. Final layer shift/scale (shared projection + final table): std 0.31 / 0.30, max |scale| 1.25 / 1.13. Before the final layer's rows moved into the shared projection, it took `final_table + c`: std 1.07, max |scale| 4.0 to 4.1 (2026-10-31).
