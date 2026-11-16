# Where a training step goes, d768x12 (2026-11-14)

`scripts/probe_step.py configs/128_d768x12_6k.toml` at commit `7f3b8e2`, usuitrack `c13ed91`: the reference model and
data, batch 256 as 4 × 64, 40 warm-up steps, then 30 steps timed whole (`train.py`'s step without the lr schedule and the records) and 30 timed by phase with
a synchronize between phases. The GPU otherwise idle (desktop only).

| Phase | ms per step |
|---|---|
| data (`Bucket.next`, 4 micro-batches) | 9.4 |
| forward + backward | 224.4 |
| UsuiTrack step | 138.7 |
| AdamW fallback step | 12.3 |
| sum | 384.8 |
| whole step, unsynchronised | 422.9 (605 img/s) |

`torch.profiler` over 3 steps: the UsuiTrack step holds 818 ms of host time against 137 ms of CUDA time, about 52,000
`cudaLaunchKernel` calls and 54 `cudaStreamSynchronize` in the 3 steps; the 6 `eigh` calls take 12.6 ms of CUDA time.

Read: forward + backward runs at about 1140 img/s. The UsuiTrack step is host-bound: about 46 ms of GPU work per step
in 139 ms. The model has 77 tracked matrices (9 conv blocks × 5, 3 attention blocks × 9, 5 outside the blocks);
*Hypothesis*: the step's cost is per matrix, so it grows with depth. The runs on the same config log a median of
651 img/s (`train.jsonl`), above the probe's whole step; the gap is not explained. The whole step is 38 ms longer than the sum of the phases; the phase loop leaves out the token
weighting and per-sample reads of `accumulate_gradients` and the fallback gradient norm, and the 38 ms is not
attributed among them; those are small operations, so this is not an explanation.
