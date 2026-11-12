# Handover

State on 2026-11-12, after a cleanup pass over the whole repo. `PLAN.md` carries the questions, `FACTS.md` the
measurements, `docs/log/` the run notes, `AGENTS.md` the rules.

## Where it stands

`main` is the trunk: the architecture of SPEC §3, the switches of the questions still open, and the docs. 77 tests
pass (`uv run python -m pytest tests/`, about 100 s). Nothing runs on the GPU.

The model: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a gated
depthwise-conv block (GDN2-style); sandwich norm; adaLN-single from the pooled caption; concatenating patch embed and
slice read-out; every Usui matrix pinned to its init norm. The baseline for trials is width 768 × depth 12 (122.8M
parameters) on the 16-shard trial set at 128 px. The size of a long run is open (PLAN G1).

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
   `log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`). A switch, off by default. Level in loss, better prompt binding
   in the three sample grids in `log/2026-11-07-samples/`.
5. **The block layout** (`topics/block-design.md`, `log/2026-11-06-*`): conv in three of four blocks, GQA, quarter
   RoPE, concat patch embed. In SPEC §3 and §5 on the strength of 3000-step arms.
6. **The GDN2-style gated update in the conv blocks** (same log note): 0.0022 ahead of plain conv at 128 px, 0.005
   at 256 px, at 84% of the speed. Accepted as a step; faster forms are still to be tried. Samples of these arms
   were not looked at, and the reference implementation was not read, only a documentation page.
7. **Priming** (`topics/priming.md`): a side quest. Of four schemes only REPA ends ahead of scratch (0.0023), at 8.4×
   the cost per aligned step with online targets. No priming code is on the trunk (archive 2026-11-12).
8. **Fine-T2I** (`topics/datasets-and-encode.md`, `log/2026-11-07-fine-t2i-encode-and-overlap.md`):
   `stage_fine.py` and `encode_fine.py`. Curated and both enhanced-prompt subsets were reported encoded; the staging
   of the original-prompt subsets stopped on a nearly full disk. The state on disk was not re-checked. The per-bucket
   shuffle and stream are not built, so none of it can be trained on.

## Open decisions

- **SPEC §6 lr schedule** says linear decay to a fraction; PLAN C2 aims at a constant lr with a late anneal.
- **PLAN B7**: the stored pooled embeddings carry the document prefix, which retrieves images worse than no prefix.
- **PLAN A2, B1, B2** are measured and ready to close.
- **Switches on the trunk**, each for an open question:
  - `cross` (F3): cross attention to the token states in the attention blocks.
  - `pooled_ada = false` (F3): the pooled caption leaves the adaLN, so the caption enters by cross attention only.
    Level in loss with cross plus adaLN at 6000 steps, and the simpler model if cross attention is adopted.
  - `conv_period` (F4): every n-th block is attention, the rest conv. To be replaced by a layout pattern of block
    types repeated to the depth, which also places the attention block within the period.
- **PLAN is still part diary** in C2, G5 and F3.
- **usuitrack**: the optimizer runs from the unpushed branch `zero-init`. The `lab` candidate is agreed in outline:
  the range finder stays, `frame_init` random as an opt-in, norm bounds in place of the pin flag (PLAN G8); whether
  Usui needs `eigh` at all is PLAN A3. Not built yet.

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
- `Stream` reads the token states only for a model with cross attention; whether that changes the step time of the
  pooled-only model is not measured.
