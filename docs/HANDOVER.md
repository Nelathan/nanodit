# Handover

Written 2026-10-31, night; updated 2026-11-01. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

Built and tested (`uv run python -m pytest tests/`: 52 pass on the CPU and the GPU): the DiT, the x0-velocity objective, the Euler/CFG sampler, the shuffle into one Parquet and its streamed loader, the Usui (fixed step, every matrix) + StochasticAdamW split with Molegrad and gradient accumulation, linear LR warmup, `scripts/train.py` training a pure bf16 model and streaming its curves, averaged per `log_every` steps, to wandb, the precompute and the probes. 300-step smoke runs at 128 px are in FACTS: five under the earlier relative step (GPU smoke runs), five under the fixed step at 2e-2 (Fixed-step runs: three warmup lengths, two with a rank-64 frame on the final linear). A 9000-step run of the committed design, `128_fixed_2e-2_9k` (FACTS; wandb project `tiny-dit`), reached 1.37 with the low-t gap closed; its final linear's frame stopped rotating after about step 1000. A rank-64 arm of it, `128_fixed_2e-2_final_r64_9k`, ended about 0.010 lower. At usui lr 2e-3 (FACTS, Runs at usui lr 2e-3) a run stopped at step 4300 was 0.04 below that arm at the same step, and a 1000-step counter-test showed beta 0.99 lifts `grad_moment_cosine` from −0.4 to −0.1; the smoke config now takes beta 0.99 and the 4×4 eval grid (archive 2026-11-01, eval grid, beta 0.99).

Data: the 128 px precompute of shards 0–3 is done (2026-11-01 04:39, log `/mnt/mars/ai/tiny-dit/precomputed/precompute_128.log`): `/mnt/mars/ai/tiny-dit/precomputed/128/data_00000{0..3}.parquet`, the tars deleted. Two image sources, half each (FACTS, Sample and shard), row counts and sizes there too. Shard 0's parquet is checked against its tar (FACTS). The null embedding is at `precomputed/null_embedding.safetensors`. Shards 4–10 were encoded on the GPU with TF32 convolutions, the next tar downloading while one encodes (log `precomputed/precompute_128_4-46_gpu.log`; FACTS, GPU precompute). Shards 0–10, 549,945 rows, are shuffled into `precomputed/128_shuffled_0-10.parquet` (FACTS, Shuffle and stream), which `configs/128_smoke.toml` reads. The precompute is paused at shard 11, its tar on disk, while `128_2e-3_0-10_10ep` trains 10 passes over that file; resume it with `uv run python scripts/precompute.py 128 $(seq 11 46)`, then shuffle all shards again.

## Waiting on Daniel

- **F1, F2.** Model choices SPEC leaves open (PLAN group F).
- **B1, B2, A2.** Evidence is in; each needs a close or a follow-up.
- **Not chosen yet:** C3 shift base, D4 sampler grid, LR decay and the warmup length (C2), `fallback_lr` as a sweep axis.

## Next

1. 1000-step ablations: the final linear on the fallback (a scratch launcher; `build_optimizers` has no switch for it), the fallback lr and betas (C2), batch 64 at a constant number of images (C1).
2. The LR sweep (C2). A run is launched in the background: `cd /home/djg/Projects/tiny-dit && nohup uv run python scripts/train.py configs/<name>.toml > /mnt/mars/ai/tiny-dit/runs/<name>.log 2>&1 &`; a run directory that exists is refused.

## Traps

- Precompute: one bad sample or an OOM restarts the shard from row 0 (the tar survives); a malformed tar entry is dropped or raises; nothing checks for 50k rows. It runs on the GPU (about 137 rows/s, FACTS) and shares it with any training run.
- Checkpoints from before the final layer's rows moved into the shared projection, every run up to and including `128_2e-3_0-10_10ep`, do not load into the current `DiT` (`ada` is 4096 rows, not 3072, and the text enters through `rms_norm`): render them from a checkout before that commit.
- The 256 px pass re-downloads each shard, because the 128 px pass deletes the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things".
