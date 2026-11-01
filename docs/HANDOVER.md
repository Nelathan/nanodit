# Handover

Written 2026-10-31, night; updated 2026-11-01. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

Built and tested (`uv run python -m pytest tests/`: 48 pass on the CPU; on the GPU `test_accumulated_batch_trains_logs_and_checkpoints` fails in Usui's first frame fit, PLAN A3): the DiT, the x0-velocity objective, the Euler/CFG sampler, the Parquet loader, the Usui (fixed step, every matrix) + StochasticAdamW split with Molegrad and gradient accumulation, linear LR warmup, `scripts/train.py` training a pure bf16 model and streaming its curves, averaged per `log_every` steps, to wandb, the precompute and the probes. 300-step smoke runs at 128 px are in FACTS: five under the earlier relative step (GPU smoke runs), five under the fixed step at 2e-2 (Fixed-step runs: three warmup lengths, two with a rank-64 frame on the final linear). A 9000-step run of the committed design, `128_fixed_2e-2_9k` (FACTS; wandb project `tiny-dit`), reached 1.37 with the low-t gap closed; its final linear's frame stopped rotating after about step 1000. A rank-64 arm of it, `128_fixed_2e-2_final_r64_9k`, ended about 0.010 lower. At usui lr 2e-3 (FACTS, Runs at usui lr 2e-3) a run stopped at step 4300 was 0.04 below that arm at the same step, and a 1000-step counter-test showed beta 0.99 lifts `grad_moment_cosine` from −0.4 to −0.1; the smoke config now takes beta 0.99 and the 4×4 eval grid (archive 2026-11-01, eval grid, beta 0.99).

Data: the 128 px precompute of shards 0–3 is done (2026-11-01 04:39, log `/mnt/mars/ai/tiny-dit/precomputed/precompute_128.log`): `/mnt/mars/ai/tiny-dit/precomputed/128/data_00000{0..3}.parquet`, the tars deleted. Two image sources, half each (FACTS, Sample and shard), row counts and sizes there too. Shard 0's parquet is checked against its tar (FACTS). The null embedding is at `precomputed/null_embedding.safetensors`. Shards 4–46 are encoding on the GPU with TF32 convolutions, the next tar downloading while one encodes (log `precomputed/precompute_128_4-46_gpu.log`; FACTS, GPU precompute).

## Waiting on Daniel

- **F1, F2.** Model choices SPEC leaves open (PLAN group F). F3 is the session's to settle (Next).
- **B1, B2, A2.** Evidence is in; each needs a close or a follow-up.
- **A3.** The GPU frame fit failure: fp64 `eigh` or a jitter in usuitrack, or leave it.
- **Not chosen yet:** C3 shift base, D4 sampler grid, LR decay and the warmup length (C2), `fallback_lr` as a sweep axis.

## Next

1. The data pipeline (archive 2026-11-01, eval grid, beta 0.99, entry 5): one shuffled Parquet from the shard Parquets, streamed block-wise in training, replacing `load_shards` and the with-replacement `Batcher`.
2. 1000-step ablations: the final linear on the fallback (a scratch launcher; `build_optimizers` has no switch for it), the fallback lr and betas (C2), batch 64 at a constant number of images (C1).
3. F3 (the final layer's shift/scale from the shared adaLN projection) and the text RMS norm.
4. The LR sweep (C2). A run is launched in the background: `cd /home/djg/Projects/tiny-dit && nohup uv run python scripts/train.py configs/<name>.toml > /mnt/mars/ai/tiny-dit/runs/<name>.log 2>&1 &`; a run directory that exists is refused.

## Traps

- Precompute: one bad sample or an OOM restarts the shard from row 0 (the tar survives); a malformed tar entry is dropped or raises; nothing checks for 50k rows. It runs on the GPU (about 137 rows/s, FACTS) and shares it with any training run.
- The 256 px pass re-downloads each shard, because the 128 px pass deletes the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- `load_shards` peaks at about 3× the data size (FACTS); by the code, the Arrow tables, the per-shard copies and the cat result are alive at once. Fine for shards 0–3 at 128 px; about ten shards at 128 px, or shards 0–3 at 256 px, would not fit in 31 GB. Filling one preallocated tensor shard by shard would bring it near 1×.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things".
