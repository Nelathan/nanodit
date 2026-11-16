# Where a training step goes, d768x24 on usuitrack 92b23bc (2026-11-16)

`scripts/probe_step.py configs/128_d768x24_1ep.toml` at the commit that adds this note, usuitrack `92b23bc`: depth 24,
the meta-only conv block, batch 256 as 2 × 128, 40 warm-up steps, then 30 steps timed whole and 30 timed by phase with
a synchronize between phases. The GPU otherwise idle (desktop only). Output:
`/mnt/mars/ai/tiny-dit/probes/step_d768x24_92b23bc.txt`.

| Phase | ms per step |
|---|---|
| data (`Bucket.next`, 2 micro-batches) | 12.1 |
| forward + backward | 363.6 |
| UsuiTrack step | 57.6 |
| AdamW fallback step | 0.9 |
| sum | 434.2 |
| whole step, unsynchronised | 429.5 (596 img/s) |

Peak memory 7.9 GiB. `torch.profiler` over 3 steps: 1.192 s of CUDA time, about 397 ms per step of the 430 ms step;
the UsuiTrack step holds 101 ms of host time per step against 67 ms of self CUDA time (host time under the profiler,
which inflates it: the synchronised phase is 57.6 ms); 18,165 `cudaLaunchKernel` calls in the 3 steps.

Read: the GPU is busy about 92 % of the step, so launch overhead can win at most about 33 ms per step. Against the
d768x12 profile at usuitrack `c13ed91` (`log/2026-11-14-step-profile.md`: Usui step 138.7 ms for 77 matrices) the
Usui step is 57.6 ms for 131 matrices, and under the profiler its host time exceeds its GPU time by about half.
