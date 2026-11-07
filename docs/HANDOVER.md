# Handover

Morning report, 2026-11-06 (night run). Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `docs/log/` the run notes, `AGENTS.md` how to run things.

## Where it stands

**The architecture is merged on the trunk (`lr-decay`, commit 62b5727) and SPEC §3/§5 state it. Accept it or change it; it is not moving by itself.** d768 × 12 for trials, d768 × 16 with SwiGLU 4× planned for the long run. Every 4th block is attention (GQA with a quarter of the heads as key/value heads, RoPE on the first quarter of the frequencies); the other three are depthwise-3×3 conv blocks writing the meta channels only, with adaLN from the pooled caption kept. The patch embed concatenates the raw 128-d patch with `Linear(128 → width−128)`. Cross attention (attention blocks only, no ada, zero gate) and `pooled_ada = false` are experiment switches, off / on by default. PaLM-parallel MLP is dropped. Reads behind it: FACTS "Architecture at d768x12", `docs/topics/block-design.md` (ledger of every suggestion), `docs/log/2026-11-06-*`. 77 tests pass (`PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q`, about 90 s).

SPEC still says width 1024 in the §3 table; the trial and planned long run are d768. Not changed, since it is a size decision.

Still open (yours): which attention types sit in which layers under the 3/4 conv pattern, and whether self and cross attention are sequential or parallel into one meta-only out projection. I proposed a pattern-string layer layout (`conv conv conv self+cross`, `|` for parallel); unanswered. Cross attention versus pooled ada is to be tested with and without ada at a length where tokens can matter; the 3k arm decides nothing.

## Running on the GPU (one job at a time)

1. `queue_6k_2026-11-06.sh` (branch `conv-meta`, old pooled-only model): 6000-step references `conv4_gqa3_rope25_meta`, `conv4`, `attn`, each scored at 256 px into `score_256.jsonl` and `queue_6k_2026-11-06.log`. The first is done (256 px caption loss 1.3804).
2. `queue_trunk_2026-11-06.sh` waits for (1), then runs on the trunk: `arch_6k` (the reference; scored at 256 px), the d768×16 lr sweep 2e-3 / 1e-3 / 5e-4 (3000 steps each, constant lr, fallback 1.25×), `arch_cross_6k`, `arch_crossonly_6k`, then priming stage 1 (`arch_prime_s80_3k`: shift 4, t sigma 0.0001) and stage 2 (`arch_prime_s80_cont_3k`, from its checkpoint, normal t), the latter scored at 256 px. Log: `runs/queue_trunk_2026-11-06.log`. It ends with the line `queue done`.

Priming needs no new code: the first stage is the existing `shift` / `t_sigma` knobs, the second `init_from` with `warmup = 0`. Its verdict is the continuation against `arch_6k` at equal total steps.

Runs live under `/mnt/mars/ai/tiny-dit/runs/`; each finished run gets a dated note in `docs/log/` and a one-line insight in FACTS.

## Not done yet from the overnight brief

- **GDN2 as a conv mixer**: branch `gdn-conv` (`../tiny-dit-gdn`) now has the real recurrence (decay, decoupled erase and write gates, delta rule) scanned row/column, forward/backward by a quarter of the heads each; `gdn = "meta"` writes the meta channels, `"all"` the image channels too (gates start at zero). The reference kernel code is not read (only the Megatron-Core doc page). It is a Python loop under activation recomputation: 66.7 img/s against 1376 for plain conv at 128 px (20× slower), 14 min compile, and not feasible at 256 px and above without a chunkwise (WY) form. `queue_gdn2_2026-11-07.sh` runs `arch_gdnmeta_6k` (about 6.5 h) after the lr 8e-3 run; `arch_gdnall_6k` is configured, not queued. An earlier, simplified arm (short-conv linear attention, 1.4849) was not GDN2; its run directory is deleted, the numbers are in `docs/log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`.
- **Priming**: the run tested only stage 1 + ordinary flow training. The paper's stage 2 is distributional (Sinkhorn/OT) post-training of a one-step generator, so the 10x claim is about one-step generation, not about multi-step flow training. Not a test of the paper.
- **Fine-T2I staging** runs since 2026-11-07 (`scripts/stage_fine.py`, log `/mnt/luna/tiny-dit/fine/stage_2026-11-07.log`, output `/mnt/luna/tiny-dit/fine/staged/<subset>/train-NNNNNN.parquet`, resumable: rerun the same command).
- **256 px training readiness**: `shuffle.py` now reads each shard in 4096-row pieces (the 256 px shuffle was killed by the memory reaper on 3.3 GB shards); tests cover it. The 256 px trial shuffle is not restarted (an earlier instruction said not to without being asked); command was `scripts/shuffle.py OUT 256 SHARD…` on shards 0-15. Memory use of a 256 px training step is untested.
- **Second dataset (Fine-T2I) and cross-dedup**: not started. Plan: per-tar download → resize to cover and crop overflow into the 7 aspect buckets → small staged images, then a GPU stage for latents and caption embeddings; per-bucket files and a stream drawing one bucket per batch; dedup against dataset 1 by exact latent bytes and a pooled-latent signature. The raw tars (2.3 TB) do not fit on `/mnt/luna`, hence per tar. Download is about 16 h.
- **One-epoch readiness for d768×16, both datasets**: waits on the lr sweep (constant 1e-3 is the working guess) and on the two items above.

## Branches

Daniel merges and pushes. `lr-decay` (worktree `../tiny-dit-decay`) is the trunk, with the conv-meta merge committed. `conv-meta` (`../tiny-dit-conv`) holds the old pooled-only model with the experiment knobs and the 6k configs; the queue above still runs from it, so leave it alone until the queue ends, then it can be deleted. `gdn-conv` (`../tiny-dit-gdn`) is the GDN-style mixer arm. `g2-patch16` (`../tiny-dit-g2`) is the closed 16 px patch arm. The `.venv` of `../tiny-dit-decay` is the one used by all worktrees (`PYTHONPATH=<worktree>/src` selects the code).

The older configs (`128_d1024x16_*`, `128_d768x12_ep*`, sweeps) are unchanged but now build the new architecture, so they no longer reproduce their old runs; their checkpoints do not load.

## Traps

- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name, so a waiter on a script name can wait on itself. Find and kill runs by PID.
- Checkpoints load only into the model they were trained with: everything before the 2026-11-06 merge does not load into the current `DiT`.
- A run is launched in the background (`run_in_background`, not `&` in a foreground command, which is killed). A run directory that exists is refused; one whose `wandb.init` failed has to be deleted. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.
- zsh does not word-split unquoted variables; long argument lists go through a bash script file.
- pyarrow int32 offsets cap a chunk at 2 GB: take slices.
- wandb groups series on the first slash only; keep keys one level deep. The entity and key depth are in `AGENTS.md`.
- With the concat patch embed the patch embed gets zero gradient at step 0 (no gate reads the meta channels yet); only `ada.1.weight` moves on the first step. That is expected, not a bug.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things". The GPU sits next to Daniel's family: long runs mean hours of audible airflow.
- Cleanup when idle: shock checkpoints (about 31 GB on mars), `/mnt/luna/tiny-dit/probe_fine`.
