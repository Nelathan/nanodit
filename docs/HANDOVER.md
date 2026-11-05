# Handover

Written 2026-10-31, night; rewritten 2026-11-05. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

The model of the current runs (SPEC §3): a d1024 × 16 single-stream DiT with the slice readout (the output is the first 128 stream channels, the patch embed the identity on them), a sandwich norm on every branch output, SwiGLU 4×, no conv. Usui trains every matrix but the adaLN table, each pinned to its init Frobenius norm with a constant relative step; both lrs warm up 500 steps and decay linearly to a tenth. Training t is logit-normal σ 0.7. The checkpoint score is the caption loss and R² on the Lunara eval rows at 10 fixed t. `PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q`: 64 pass.

The best model is `runs/128_d1024x16_swpin_5ep` (5 epochs of the 47 deduped shards at 128 px, batch 256, 37,605 steps): score 1.3856, no spike, class-level prompt following in the samples, still improving at the end (FACTS). It trained at σ 1. The unpinned `128_d1024x16_5ep` before it spiked from step 4850 (FACTS, Spikes); G6 closed on that (archive 2026-11-05).

Data: `/mnt/mars/ai/tiny-dit/precomputed/128_shuffled_0-46.parquet` (1,925,416 rows, each image once), bicubic 128 px latents; the null embedding and the Lunara eval set beside it.

## Branches

Daniel merges and pushes. `lr-decay` (worktree `../tiny-dit-decay`) holds everything: it sits on `t-dist` (`../tiny-dit-tdist`), on `sandwich-norm` (`../tiny-dit`), on `slice-base`; `master` is behind all of them. The `.venv` in `../tiny-dit-decay` is a symlink to `../tiny-dit-tdist/.venv`, excluded in `info/exclude`. The other worktrees are throwaway arms. usuitrack commits `9abfa3c` and `bb21798` (opt-in `pin_norm`, the pinned norm kept as a float) wait for Daniel's review, as do `2f377b0` and `721191d` from before.

## Next

1. The continuation lr (PLAN C4): short constant-lr arms from `swpin_5ep/checkpoints/step_0037605` at σ 0.7, read with the short-horizon bias in mind. Daniel: continue from this checkpoint rather than restart, unless the model changes.
2. The rank study over the `swpin_5ep` checkpoints (PLAN G9).
3. Model changes, each a fresh run: the 4× patch embed writing the meta channels (G2), the conv placed differently (G10); with them, when to add 256 px (C1, B5).
4. Reading: Priming (PLAN I1), EDM2 (G8).

A run is launched in the background: `nohup .venv/bin/python scripts/train.py configs/<name>.toml > /mnt/mars/ai/tiny-dit/runs/<name>.log 2>&1 &` with `PYTHONPATH` at the worktree's `src`; a run directory that exists is refused. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.

## Traps

- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name, so a waiter on a script name can wait on itself. Find and kill runs by PID.
- Checkpoints load only into the model they were trained with: the unpinned `5ep` ones and everything before the sandwich norm do not load into the current `DiT`.
- Configs state every field that differs from the current defaults (`pin_weights = false`, `t_sigma = 1.0` in older ones), so a config still describes its run.
- wandb groups series on the first slash only; keep keys one level deep (`train/…`, `eval/…`, one group per bucketed topic).
- The 256 px pass re-downloads each shard: the 128 px pass deleted the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things". The GPU sits next to Daniel's family: long runs are coil whine for hours.
