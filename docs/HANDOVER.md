# Handover

State on the night of 2026-11-12, with the re-encode running. `PLAN.md` carries the questions, `FACTS.md` the
measurements, `docs/log/` the run notes, `AGENTS.md` the rules.

## Where it stands

`main` is the trunk: the architecture of SPEC §3, the switches of the questions still open, and the docs. 79 tests
pass (`uv run python -m pytest tests/`, about 100 s). The GPU runs the re-encode (below) until it ends; start no
training before.

The model: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a gated
depthwise-conv block (GDN2-style); cross attention to the caption's token states in every attention block (GQA, no
RoPE); sandwich norm; adaLN-single from the pooled caption; concatenating patch embed and
slice read-out; every Usui matrix inside the ball of its init norm (a ceiling, no floor). The baseline for trials is
width 768 × depth 12 (122.8M parameters) on the 16-shard trial set at 128 px. Width 768 is decided; the depth of a
long run is open (PLAN G1).

No long run of this model exists. The longest run, `128_d1024x16_swpin_5ep` (2026-11-05, 37,605 steps), is of the
model before the conv blocks; its last checkpoint is kept and does not load into the current `DiT`.

## What the trunk rests on

Each line is one unit of work: what changed, where the evidence is, and how far it reaches. An arm is one paired run
against a reference with the same seed, data order and eval rows. None was repeated across seeds, so a difference
below about 0.003 in caption loss is within what one seed can say.

1. **Slice read-out, sandwich norm, pin, lr decay** (archive 2026-11-03 to 11-05; FACTS "Branch norm and pin arms",
   "`128_d1024x16_swpin_5ep`"). Decided. The five-epoch run is the only evidence at length.
2. **Continuation lr and the edge-of-stability reads** (FACTS "Continuation lr arms",
   `topics/edge-of-stability.md`). Reads only; no design changed.
3. **The data pipeline**: `precompute.py` writes one fused Parquet per shard (128 and 256 px Lanczos latents, pooled
   embedding, 512-d token states); `shuffle.py` makes the training set, `precompute_lunara.py` the eval set.
4. **Cross attention to the token states** (`log/2026-11-06-cross-attention-and-conv-period-3k.md`,
   `log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`). Decided as design, in every attention block. Level in loss,
   better prompt binding in the three sample grids in `log/2026-11-07-samples/`.
5. **The block layout** (`topics/block-design.md`, `log/2026-11-06-*`): conv in three of four blocks, GQA, quarter
   RoPE, concat patch embed. In SPEC §3 and §5 on the strength of 3000-step arms.
6. **The GDN2-style gated update in the conv blocks** (same log note): 0.0022 ahead of plain conv at 128 px, 0.005
   at 256 px, at 84% of the speed. Accepted as a step; faster forms are still to be tried. Samples of these arms
   were not looked at, and the reference implementation was not read, only a documentation page.
7. **Priming** (`topics/priming.md`): a side quest. Of four schemes only REPA ends ahead of scratch (0.0023), at 8.4×
   the cost per aligned step with online targets. No priming code is on the trunk (archive 2026-11-12).
8. **Fine-T2I** (`topics/datasets-and-encode.md`, `log/2026-11-07-fine-t2i-encode-and-overlap.md`):
   `stage_fine.py` and `encode_fine.py`. On disk (checked 2026-11-12): curated (192 tars) and both enhanced-prompt
   subsets (1621, 1544) staged and encoded, 372 GB; original-prompt random resolution staged (1688), square staged
   1150 of 1311, staging stopped on a full disk; neither encoded. The per-bucket shuffle and stream are not built, so
   none of it can be trained on.

## Open decisions

- **The re-encode without the prefix** (archive 2026-11-12-no-prefix), running since the night of 2026-11-12:
  `scripts/reencode_text.py` over the 3409 files in `/mnt/mars/ai/tiny-dit/reencode_2026-11-12.files` (the Lunara
  eval sets, the null embedding, the trial set, the 47 fused shards and Lunara art, the Fine-T2I encoded tars), one
  line per file in `reencode_2026-11-12.log` beside it, `exit N` at the end. About 6 h at 254 rows/s (fused) and
  391 rows/s (Fine-T2I), measured on copies. Each file is replaced whole by a rename, so a killed run is resumed by
  passing the files the log does not list. A caption with no tokens stops it on an assertion, naming no file: the
  next file after the log's last line holds it. When it ends: check row counts and a pooled embedding against a fresh
  encode, then correct FACTS (the caption-prefix line, the null embedding line) and `topics/priming.md`, which still
  describe the prefixed data.
- **Switches on the trunk**, each for an open question:
  - `pooled_ada = false` (F3): the pooled caption leaves the adaLN, so the caption enters by cross attention only.
    Level in loss with cross plus adaLN at 6000 steps.
  - `layout` and `block_types` (F4): named block types repeated to the depth; where the attention block sits in the
    period is a config-only arm.
- **The reference** `configs/128_d768x12_6k.toml` (cross attention, usui lr 4e-3) has not run; it waits for the
  re-encode of the captions without the prefix.
- **Disk**: luna is 97 % full (132 GB free). Candidates to free, not decided: the staged copies of the three encoded
  Fine-T2I subsets (about 73 GB), and the old pre-fused `128_shuffled_0-46.parquet` on mars (33 GB, a layout no
  code reads, and the prefixed pooled embedding, not re-encoded).
- **usuitrack**: a git dependency at `lab` `c13ed91` (pushed): per-group `step_rule`, `frame_init` (`range_finder`
  default, `random` here) and `norm_bounds`. Whether Usui needs `eigh` at all is PLAN A3 here and R2 in usuitrack's
  PLAN. Moving `lab` also moved ai-toolkit's Anima lane, which runs the `../usuitrack` checkout: its first frame is
  now the range finder, not eigh (usuitrack PLAN R1). `lab` was `be3f298` before.

## Branches

Not merged, each the record of an arm: `gdn-conv` (a content-dependent window mixer, lost at 256 px), `g2-patch16`
(closed), `pinned-copy` (pinned-memory staging in `Stream`, PLAN H1, never measured on the trunk),
`arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}` (early arms).

## Traps

- Checkpoints load only into the model they were trained with.
- A run is launched in the background. A run directory that exists is refused; one whose `wandb.init` failed has
  to be deleted. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.
- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name, so a waiter on a script
  name can wait on itself. Find and kill runs by PID.
- pyarrow int32 offsets cap a chunk at 2 GB: take slices.
- wandb groups series on the first slash only; keep keys one level deep.
- With the concat patch embed the patch embed gets zero gradient at step 0 (no gate reads the meta channels yet);
  only `ada.1.weight` moves on the first step. That is expected.
