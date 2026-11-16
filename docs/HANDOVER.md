# Handover

State on 2026-11-16, at the start of the one-epoch reference at depth 24. `PLAN.md` carries the questions,
`FACTS.md` the measurements, `docs/log/` the run notes, `AGENTS.md` the rules.

## Where it stands

`main` is the trunk: the architecture of SPEC §3, the switches of the questions still open, and the docs. 79 tests
pass (`uv run python -m pytest tests/`, about 150 s on the CPU). The GPU runs `128_d768x24_1ep`.

The model: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a
depthwise-conv block that writes only the meta channels; cross attention to the caption's token states in every attention block (GQA, no
RoPE); sandwich norm; adaLN-single from the pooled caption; concatenating patch embed and
slice read-out; every Usui matrix framed on its input side and inside the ball of its init norm (a ceiling, no floor),
usui lr 2e-3. The reference is width 768 × depth 24 at 128 px, micro-batch 128. Width 768 is decided; the depth of a
long run is open (PLAN G1).

No long run of this model exists yet: `128_d768x24_1ep` (one pass of `set_128`, 32,184 steps, about 4.5 h) is the
first. Every earlier checkpoint is deleted (2026-11-16).

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
6. **The conv block** (PLAN G10): the GDN2-style gated update was 0.0022 ahead of plain conv at 128 px at 84% of the
   speed (same log note); it is replaced by the basic meta-only conv, the fastest form and one that cannot write the
   image channels. A better block that keeps the meta-only write is open.
7. **Priming** (`topics/priming.md`): a side quest. Of four schemes only REPA ends ahead of scratch (0.0023), at 8.4×
   the cost per aligned step with online targets. No priming code is on the trunk (archive 2026-11-12).
8. **Fine-T2I and the merged set** (`topics/datasets-and-encode.md`, archive 2026-11-13-merged-training-set):
   `stage_fine.py`, `encode_fine.py` (deletes each staged Parquet once encoded), `shuffle.py` (both datasets into one
   Parquet per aspect bucket, deduplicated), `Stream` (a bucket per micro-batch, token-weighted accumulation). All five Fine-T2I subsets are encoded; the
   merged set `set_128` (8.24M rows after dedup, seven buckets) is built on vel-ssd and the reference trains on it (log
   2026-11-13-reference-6k-merged-set).
9. **The captions without the prefix** (archive 2026-11-12-no-prefix): every stored text column and the null
   embedding re-encoded on 2026-11-13 (FACTS, the caption-prefix line).

## Open decisions

- **Switches on the trunk**, each for an open question:
  - `pooled_ada = false` (F3): the pooled caption leaves the adaLN, so the caption enters by cross attention only.
    Level in loss with cross plus adaLN at 6000 steps.
  - `layout` and `block_types` (F4): named block types repeated to the depth; where the attention block sits in the
    period is a config-only arm.
- **The reference** `configs/128_d768x24_1ep.toml`: depth 24, micro-batch 128, usui lr 2e-3, Usui's beta and the
  fallback's betas at their defaults, eval every 1000 steps, a checkpoint every 4000. Running. At depth 12 the last
  reference was 1.4792 at 6000 steps (lr 2e-3, log 2026-11-14-reference-6k-lr2e-3); no depth-24 number exists yet.
- **Step speed** (log 2026-11-16-step-profile-d24): with the usuitrack pin below the Usui step is no longer most of
  the step; what is left of its host time is usuitrack's to take (its fast-step work). Block-wise CUDA graphs were tried
  and gave nothing at micro-batch 64 and ran out of memory at 128; not kept. The usuitrack session's claim of 140 →
  30 ms for the Usui step and the step probe's figure do not agree yet; that is the next speed question.
- **Disk** (2026-11-13): luna 89 GB free, mars 318 GB, vel-ssd 198 GB. The merged 128 px set is 380 GB on vel-ssd. A
  256 px set needs its own build, and `shuffle.py` casts the latent to `binary`, which overflows 2 GB per chunk at
  256 px: it needs `large_binary` first. No off-site backup: a lost source is pulled again.
- **usuitrack**: a git dependency at `lab` `92b23bc`, was `c13ed91`. Its state per matrix is `moment`, `init_norm` and, unless the
  matrix runs in full space (4 × rank reaches its short side), `frame`. The training log reads its
  diagnostics (`grad_capture_probe`, `projected_grad_norm`) and the norm of each matrix's step before the lift
  (`step_norm/<role>`, at each eval).

## Branches

Not merged, each the record of an arm: `gdn-conv` (a content-dependent window mixer, lost at 256 px), `g2-patch16`
(closed), `pinned-copy` (pinned-memory staging in `Stream`, PLAN H1, never measured on the trunk),
`arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}` (early arms).

## Traps

- Checkpoints load only into the model they were trained with.
- Without a desktop login `vel-ssd` (LUKS) is not unlocked or mounted, and the training set is not there. The machine
  can suspend while a run trains: launch under `systemd-inhibit --what=sleep:idle`.
- `nvidia-smi` fails with a driver/library mismatch after a driver update until the reboot.
- A run is launched in the background. A run directory that exists is refused; one whose `wandb.init` failed has
  to be deleted. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.
- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name, so a waiter on a script
  name can wait on itself. Find and kill runs by PID.
- pyarrow int32 offsets cap a chunk at 2 GB: take slices.
- wandb groups series on the first slash only; keep keys one level deep.
- With the concat patch embed the patch embed gets zero gradient at step 0 (no gate reads the meta channels yet);
  only `ada.1.weight` moves on the first step. That is expected.
