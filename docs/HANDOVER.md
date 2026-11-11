# Handover

Written 2026-11-11 after a cleanup pass over the whole repo. Orientation and a guide to reviewing the work of
2026-11-03 to 11-08, which ran mostly unattended. `PLAN.md` carries the questions, `FACTS.md` the measurements,
`docs/log/` the run notes, `AGENTS.md` how to run things.

## Where it stands

`main` is the trunk and holds everything that was merged: the architecture of SPEC §3, the experiment switches
around it, and the docs. 88 tests pass (`uv run python -m pytest tests/`, about 100 s). Nothing runs on the GPU.

The model on `main`: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a
gated depthwise-conv block (GDN2-style, `gdn = true`); sandwich norm; adaLN-single from the pooled caption;
concatenating patch embed and slice read-out; every Usui matrix pinned to its init norm. Trials run at d768 × 12
(122.8M parameters) on the 16-shard trial set at 128 px.

No long run of this model exists. The longest run, `128_d1024x16_swpin_5ep` (2026-11-05, 37,605 steps), is of the
model before the conv blocks; its checkpoints do not load into the current `DiT`. Its last checkpoint is kept.

## What was built, in the order to review it

Each line is one unit: what changed, where the evidence is, and what it rests on. "Arm" means one paired run
against a reference with the same seed, data order and eval rows; none was repeated across seeds, so differences
below about 0.003 in caption loss are within what one seed can say (the two-seed spread measured on 2026-11-01
was at most 0.003 per window mean, on another model).

1. **Slice read-out, sandwich norm, pin, lr decay** (11-03 to 11-05; archive notes of those dates; FACTS "Branch
   norm and pin arms", "`128_d1024x16_swpin_5ep`"). Decided by you at the time. The five-epoch run is the only
   evidence at length.
2. **Continuation lr and the edge-of-stability reads** (11-05, 11-06; FACTS "Continuation lr arms",
   `topics/edge-of-stability.md`). Reads only; no design changed.
3. **The data pipeline rebuilt** (11-05, 11-06): `precompute.py` writes one fused Parquet per shard (128 and
   256 px Lanczos latents, pooled embedding, token states), `reencode_sequence.py` redoes the token states at
   512-d, `shuffle.py` and `join_sequences.py` make the trial and eval sets. The old 47-shard 128 px set
   (`precomputed/128_shuffled_0-46.parquet`, bicubic, no token states) no longer loads in `Stream`.
4. **Cross attention to the token states** (11-06; `log/2026-11-06-cross-attention-and-conv-period-3k.md`,
   `log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`). A switch, off by default. Level in loss, better prompt binding
   in the three sample grids in `log/2026-11-07-samples/`.
5. **The block layout** (11-06; `topics/block-design.md` is the ledger, `log/2026-11-06-*`): conv in three of four
   blocks, GQA, quarter RoPE, concat patch embed. Merged as 62b5727 and written into SPEC §3 and §5 by the agent.
   **This is the largest unreviewed design change**: SPEC was changed on the strength of 3000-step arms, and the
   handover of 11-07 said "accept it or change it".
6. **GDN2-style gated update in the conv blocks** (11-07; same log note). 0.0022 ahead of plain conv at 128 px,
   0.005 at 256 px, at 84% of the speed; default since d083e8f and in SPEC §3. Samples of these arms were never
   looked at. The reference implementation was not read, only a documentation page.
7. **Priming** (11-07, 11-08; `topics/priming.md`, four log notes). Four schemes tried, three lose or tie; REPA
   (aligning block 3 to EmbeddingGemma 2's vision features for the first half) ends 0.0023 ahead of scratch at
   8.4× the cost per aligned step. The first test was not what you asked for (the log says so). The switches for
   all of it are in `train.py` and `model.py` on `main`. Your call of 2026-11-11: a side quest, probably not used;
   if anything, REPA on precomputed targets.
8. **Fine-T2I** (11-06, 11-07; `log/2026-11-07-fine-t2i-encode-and-overlap.md`, `topics/datasets-and-encode.md`): `stage_fine.py` and
   `encode_fine.py`. Curated and both enhanced-prompt subsets were reported encoded; staging of the
   original-prompt subsets was running into a nearly full disk when the last session ended. The state on luna was
   not re-checked in this pass. The per-bucket shuffle and stream are not built, so none of it can be trained on.

## What this pass changed (2026-11-11)

- `master` is `main`, fast-forwarded to the old trunk branch `lr-decay`. Uncommitted arm code found in eight
  worktrees is committed on `arm/*` branches as it ran.
- The package is `nanodit` (archive 2026-11-11).
- One `build_model` builds the config's model for `train.py`, `score_checkpoint.py` and `probe_sharpness.py`; the
  last one built it without `repa_block` and would have failed on a REPA checkpoint.
- 41 configs removed: they pointed at the old dataset schema and could not load. Each run keeps its own `config.toml`.
- FACTS: the run diaries of 10-31 to 11-02 moved whole to `docs/log/`; parameter counts added.
- Side quests have their own docs in `docs/topics/` with their question, reads and numbers: priming, edge of
  stability (PLAN C4), weight spectra (G9), datasets and encode (B6). PLAN and FACTS point to them.
- **A wrong claim corrected**: the stored pooled embeddings carry the document prefix (checked by re-encoding);
  the 11-07 conclusion "encoded without a prefix, so they stay" was built on the opposite. Open again as PLAN B7.
- Checkpoints of `128_d1024x16_swpin_5ep` before the last one deleted, as asked.

## Waiting on you

- **SPEC §3 width and depth.** It says 1024 × 16; every arm since 11-06 is 768 × 12 and the planned long run was
  768 × 16. A size decision.
- **SPEC §6 lr schedule.** It says linear decay to a fraction; PLAN C2 aims at a constant lr with a final anneal.
- **Items 5 and 6 above**: accept the architecture as SPEC now states it, or name what to re-test.
- **PLAN B7**: the caption prefix.
- **PLAN A2, B1, B2** have been marked "Daniel to close" since the first week.
- **The experiment switches on the trunk.** `conv_period`, `pooled_ada`, `gdn = false`, `pin_weights = false`,
  masked priming (`mask_start`, `prime_steps`, `token_t`) and REPA are all still in `model.py` and `train.py`.
  Each lost or was superseded except cross attention and REPA; the code style rule says they go, but removing a
  switch closes its question, so it is yours.
- **PLAN is still part diary** (C2, G5, F3 run long; C4, B6 and G9 moved to topics). The diary parts belong in
  `docs/log/` like the FACTS sections that moved; not done because cutting them means deciding what is still open.
- **usuitrack**: `zero-init` is nine commits ahead of `lab` and is not behaviour-neutral for `lab`'s users; the
  review and a merge plan are in the session report of 2026-11-11.

## Branches

Merged into `main` and deletable: `lr-decay`, `conv-meta`, `sandwich-norm`, `slice-base`, `t-dist`.
Not merged, each the record of an arm: `gdn-conv` (the content-dependent window mixer, lost at 256 px),
`g2-patch16` (closed), `pinned-copy` (pinned-memory staging in `Stream`, PLAN H1, never measured on the trunk),
`arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}` (early arms, committed 2026-11-11 from
worktrees that had run them uncommitted).

## Traps

- Checkpoints load only into the model they were trained with: everything before the 2026-11-06 merge does not
  load into the current `DiT`.
- A run is launched in the background. A run directory that exists is refused; one whose `wandb.init` failed has
  to be deleted. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.
- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name, so a waiter on a script
  name can wait on itself. Find and kill runs by PID.
- pyarrow int32 offsets cap a chunk at 2 GB: take slices.
- wandb groups series on the first slash only; keep keys one level deep.
- With the concat patch embed the patch embed gets zero gradient at step 0 (no gate reads the meta channels yet);
  only `ada.1.weight` moves on the first step. That is expected.
- `Stream` reads and pads the token states of every row even when `cross = false`, about 59 KB a row beside a
  16 KB latent.
- Disk: about 140 GB of checkpoints under `/mnt/mars/ai/tiny-dit/runs`, nearly all of models that no longer
  exist in code; luna was at 99%.
