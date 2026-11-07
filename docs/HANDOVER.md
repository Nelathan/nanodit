# Handover

Morning report, 2026-11-07. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `docs/log/` the run notes, `AGENTS.md` how to run things.

## Where it stands

**The architecture is merged on the trunk (`lr-decay`, commit 62b5727) and SPEC §3/§5 state it. Accept it or change it; it is not moving by itself.** d768 × 12 for trials, d768 × 16 with SwiGLU 4× planned for the long run. Every 4th block is attention (GQA with a quarter of the heads as key/value heads, RoPE on the first quarter of the frequencies); the other three are depthwise-3×3 conv blocks writing the meta channels only, with adaLN from the pooled caption kept. The patch embed concatenates the raw 128-d patch with `Linear(128 → width−128)`. Cross attention (attention blocks only, no ada, zero gate) and `pooled_ada = false` are experiment switches, off / on by default. PaLM-parallel MLP is dropped. Reads behind it: FACTS "Architecture at d768x12", `docs/topics/block-design.md` (ledger of every suggestion), `docs/log/2026-11-06-*`. 77 tests pass (`PYTHONPATH=$PWD/src .venv/bin/python -m pytest tests -q`, about 90 s).

SPEC still says width 1024 in the §3 table; the trial and planned long run are d768. Not changed, since it is a size decision.

Still open (yours): which attention types sit in which layers under the 3/4 conv pattern, and whether self and cross attention are sequential or parallel into one meta-only out projection. I proposed a pattern-string layer layout (`conv conv conv self+cross`, `|` for parallel); unanswered. Cross attention versus pooled ada is to be tested with and without ada at a length where tokens can matter; the 3k arm decides nothing.

## On the GPU

The tail of `/mnt/mars/ai/tiny-dit/runs/queue_spread_2026-11-07.sh` (its per-token t arms are done): the Fine-T2I encode of everything staged (curated, both enhanced-prompt subsets), log `/mnt/luna/tiny-dit/fine/encode_overnight_2026-11-07.log`. Staging (`stage_2026-11-07.log`) runs alongside on the CPU.

## Results since the merge (full tables in `docs/log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`)

- Cross attention is level in loss and better in samples; cross-only (no pooled ada) is as good as ada plus cross. Conv in 3/4 beats all attention by 0.010.
- **GDN2-style local update**, now the trunk's conv mixer (`gdn = true`, SPEC §3): the diagonal update writing every channel beat plain conv by 0.0022 at 128 px and 0.005 at 256 px at 84 % of its speed; writing only the meta channels was 0.0008 behind that. Plain conv stays as `gdn = false`, which the earlier trial configs set. The content-dependent window read won at 128 px (0.005) and lost 0.044 at 256 px at 0.62x speed; it stays on branch `gdn-conv` and is not ported. The matrix-state GDN2 update (key-axis erase) is not built; the reference code is not read (only the Megatron-Core doc page).
- **lr** at d768x16, 3000 steps: 1e-3 1.5101, 2e-3 1.4933, 4e-3 1.4868, 8e-3 1.4950. The best is near 4e-3; 1e-3 is far off.
- Samples for the GDN arms were not looked at (loss is blind to prompt following).

## Not done yet from the overnight brief

- **Priming**: open. The first test, single-level denoising at t = 0.8, was set up before the repo was read and is neither the source's stage 1 nor the masked reconstruction Daniel directed: 3000 + 3000 is 0.013 behind scratch, 1000 + 5000 is 0.004 behind, same img/s (`docs/log/2026-11-07-priming-read.md`). The code's recipe, and masked priming as Daniel set it (our logit-normal t, a share of the tokens pure noise falling from 90% to 0), are in `docs/topics/priming.md`. That mode loses, 0.009 behind scratch at best (`docs/log/2026-11-07-masked-priming.md`). Per-token t at spread 0.5 ends level with scratch, at 1.0 0.002 behind (`docs/log/2026-11-07-per-token-t-priming.md`). The modes and ideas still open are in the topic's ledger. REPA is built (online, the tower at 384 px, weight 0.5 on the first attention block for 3000 of 6000 steps, then off; "As built" in the topic); its first arm, `configs/128_d768x12_repa_6k.toml`, runs after the encode (`/mnt/mars/ai/tiny-dit/runs/queue_repa_2026-11-07.sh`).
- **Fine-T2I staging** runs since 2026-11-07 (2294 of 6356 tars done by the evening, 395 GB free on luna; `scripts/stage_fine.py`, log `/mnt/luna/tiny-dit/fine/stage_2026-11-07.log`, output `/mnt/luna/tiny-dit/fine/staged/<subset>/train-NNNNNN.parquet`, resumable: rerun the same command).
- **256 px training readiness**: `shuffle.py` now reads each shard in 4096-row pieces (the 256 px shuffle was killed by the memory reaper on 3.3 GB shards); tests cover it. The 256 px trial shuffle is not restarted (an earlier instruction said not to without being asked); command was `scripts/shuffle.py OUT 256 SHARD…` on shards 0-15. Memory use of a 256 px training step is untested.
- **Fine-T2I encoding**: `scripts/encode_fine.py` (latents at 128 and 256 px per bucket, pooled embedding, 512-d sequence of the short `prompt`, whole, in every subset; Daniel's decision, about 0.8 TB for all) is built and reviewed. Curated is encoded but for two files whose prompts were rewritten, enhanced-prompt random resolution to file 471 of 1621 but for file 378 (same reason); the queue's last step encodes the rest, skipping what exists; the datasets are taken to share no image (no caption is shared), so no dedup. Three defective prompts were rewritten (`docs/log/2026-11-07-fine-t2i-encode-and-overlap.md`). The per-bucket shuffle and stream (data.py reads fixed-size latents) are not built.
- **One-epoch readiness for d768×16, both datasets**: the lr sweep points to 2e-3 to 4e-3 (the 1-epoch check sweeps 2e-3, 1e-3, 5e-4 as briefed, so it would not reach the best); waits on the two items above.

## Branches

Daniel merges and pushes. `lr-decay` (worktree `../tiny-dit-decay`) is the trunk, with the conv-meta merge committed. `conv-meta` (`../tiny-dit-conv`) holds the old pooled-only model with the experiment knobs and the 6k configs; its queue is finished, so it can be deleted. `gdn-conv` (`../tiny-dit-gdn`) holds the GDN-style mixer arms; its diagonal form is merged by hand into the trunk; the window form lives only there. `g2-patch16` (`../tiny-dit-g2`) is the closed 16 px patch arm. The `.venv` of `../tiny-dit-decay` is the one used by all worktrees (`PYTHONPATH=<worktree>/src` selects the code).

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
