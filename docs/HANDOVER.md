# Handover

Written 2026-10-31, night; updated 2026-11-02. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

Built and tested (`uv run python -m pytest tests/`: 54 pass on the CPU): the DiT, the x0-velocity objective, the Euler/CFG sampler, the shuffle into one Parquet and its streamed loader, the Usui (fixed step, every matrix but the final linear) + StochasticAdamW split with Molegrad and gradient accumulation, linear LR warmup, `scripts/train.py` training a pure bf16 model and streaming its curves, averaged per `log_every` steps, to wandb, the precompute and the probes. 300-step smoke runs at 128 px are in FACTS: five under the earlier relative step (GPU smoke runs), five under the fixed step at 2e-2 (Fixed-step runs: three warmup lengths, two with a rank-64 frame on the final linear). A 9000-step run of the design then committed, `128_fixed_2e-2_9k` (FACTS; wandb project `tiny-dit`), reached 1.37 with the low-t gap closed; its final linear's frame stopped rotating after about step 1000. A rank-64 arm of it, `128_fixed_2e-2_final_r64_9k`, ended about 0.010 lower. At usui lr 2e-3 (FACTS, Runs at usui lr 2e-3) a run stopped at step 4300 was 0.04 below that arm at the same step, and a 1000-step counter-test showed beta 0.99 lifts `grad_moment_cosine` from −0.4 to −0.1; the smoke config now takes beta 0.99 and the 4×4 eval grid (archive 2026-11-01, eval grid, beta 0.99).

Data: the 128 px precompute of all 47 shards is done (FACTS, Sample and shard; GPU precompute): `/mnt/mars/ai/tiny-dit/precomputed/128/data_0000{00..46}.parquet`, four sources, the tars deleted. Logs in `precomputed/`: `precompute_128.log` (shards 0–3, CPU), `precompute_128_4-46_gpu.log` (4–10 and the start of 11, GPU with TF32 convolutions), `precompute_128_11-46_gpu.log` (11–46). Shard 0's parquet is checked against its tar (FACTS). The null embedding is at `precomputed/null_embedding.safetensors`. Shards 20–34 (`llava_next_captioned_700K`) repeat images heavily (FACTS, Sample and shard); the shuffle keeps each image once (archive 2026-11-02, repeated images). Shards 0–10 are shuffled into `precomputed/128_shuffled_0-10.parquet` (FACTS, Shuffle and stream); `128_2e-3_0-10_10ep` trained 10 passes over it, then the 1000-step ablations ran (FACTS). That file predates dropping repeated images and keeps 760 repeated latents among its 549,945 rows (scratch count, 2026-11-02). All 47 shards, each image once, are shuffled into `precomputed/128_shuffled_0-46.parquet` (1,925,416 rows, 34.8 GB; FACTS, Shuffle and stream), which `configs/128_smoke.toml` and the sweep configs read.

## Waiting on Daniel

- **F1, F2.** Model choices SPEC leaves open (PLAN group F).
- **B1, B2, A2.** Evidence is in; each needs a close or a follow-up.
- **Fallback lr and betas** (C2). The 1000-step reads are in (FACTS, 1000-step ablations; one seed each).
- **Not chosen yet:** C3 shift base, D4 sampler grid, `fallback_lr` as a sweep axis, LR decay after the sweep (C2).

## Next

Daniel (2026-11-02), on what comes after the sweep: the prompt-following read (render one checkpoint's eval grid at several guidance scales) and a long 128 px run are next; the 256 px precompute "is too early i think, if we cant even get basic concepts recognizable. Theres still no hint of a cat". On batch: "honestly we might go back to bs256 again" (PLAN C1).

1. The 2e-4 sweep arm, rerun with the usuitrack `eigh` fix (`bf7fef7`): `runs/128_sweep_lr2e-4`. A run is launched in the background: `cd /home/djg/Projects/tiny-dit && nohup uv run python scripts/train.py configs/<name>.toml > /mnt/mars/ai/tiny-dit/runs/<name>.log 2>&1 &`; a run directory that exists is refused.
2. A long 128 px run, scored at every checkpoint: `128_4e-3_ep2` continues the 4e-3 arm at half lr, epoch-wise decay. `128_4e-3_ep3` (same lr, batch 256) is queued behind it by `runs/queue_ep3_2026-11-02.sh`. Daniel: "Yes i do want to see how good a model could be."
2a. The width and depth arms (PLAN G1) run overnight from `runs/queue_g1_2026-11-02.sh` (log beside it): `128_d768x12_ep1..4`, then `128_d512x24_ep1..4`, one resumed run per epoch, stopping at the first failure; about 3 h and 4.5 h. Each checkpoint is scored on Lunara. `128_4e-3_ep4` ended at 1.4821 / 1.4844 (FACTS). The sessions's usuitrack commits to review later (Daniel: "usui later"): `2f377b0` (first frame from any gradient, `frame_init`, traceable range finder; SPEC step 2 and the refusal list) and `721191d` (one stacked `eigh`; SPEC step 6).
3. Faster training (PLAN H1) before scaling model or data. Block-wise `torch.compile` is in (Daniel: "easy lunch"); its img/s on a free GPU is not measured yet. The Usui step is most of the remaining launches; the CUDA-graph and batching options are in PLAN H1.

## Traps

- Precompute: one bad sample or an OOM restarts the shard from row 0 (the tar survives); a malformed tar entry is dropped or raises; nothing checks for 50k rows. It runs on the GPU (about 137 rows/s, FACTS) and shares it with any training run.
- Checkpoints from before the final layer's rows moved into the shared projection, every run up to and including `128_2e-3_0-10_10ep`, do not load into the current `DiT` (`ada` is 4096 rows, not 3072, and the text enters through `rms_norm`): render them from a checkout before that commit.
- The 256 px pass re-downloads each shard, because the 128 px pass deletes the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things".
