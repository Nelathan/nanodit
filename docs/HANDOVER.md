# Handover

State on 2026-11-18: the reference `a1-ref-1ep` has finished its pass, and its cooldown `a1-ref-1ep-anneal3k` is the
best model (score 1.4106, the README's grid). The long run's size, the training t and the end-of-run weights are
decided (archive 2026-11-18). A queue of four config-only 6000-step-equivalent arms runs on the GPU. `PLAN.md` carries
the questions, `FACTS.md` the measurements, `docs/log/` the run notes, `AGENTS.md` the rules.

## Where it stands

`main` is the trunk: the architecture of SPEC §3, the switches of the questions still open, and the docs. 81 tests
pass (`uv run python -m pytest tests/`, about 150 s on the CPU; last run before the 2026-11-18 doc-only commits).

The model: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a depthwise-conv
block that writes only the meta channels; cross attention to the caption's token states in every attention block;
sandwich norm; adaLN-single from the pooled caption; concatenating patch embed and slice read-out; every Usui matrix
framed on its input side and inside the ball of its init norm, usui lr 2e-3. Width 768 × depth 24 at 128 px, batch 256
as 4 × 64.

Runs of 2026-11-18, all under `/mnt/mars/ai/tiny-dit/runs/`:
- `a1-ref-1ep` (from `768745b`): one pass, 32,173 steps, about 4 h, score 1.4242. σ 0.7 is ahead of the σ 1 run at
  every shared eval (log 2026-11-18-a1-ref-1ep). Its checkpoints 4000 to 32173 are kept: 32173 is the anneal's start
  and the long run may continue from it.
- `a1-ref-1ep-anneal3k`: a cosine cooldown over a tenth of a pass from 32173, score 1.4106 (log 2026-11-18-anneal3k).
  Its last checkpoint is kept as the best model.
- The residual-write probe on 32173 (`scripts/probe_residual_writes.py`, log 2026-11-18-residual-writes): no sublayer
  damages the prediction; the MLPs from block 8 on write it; the forms are fixed in the sampler at t 0.8 to 0.67; the
  detail inside a token is the least recovered. Its decoded strips are in
  `/mnt/mars/ai/tiny-dit/probes/residual_writes_a1-ref-1ep_32173/`.
- `128_d768x24_1ep` (σ 1, stopped by the out-of-memory fault at 19,600): the comparison for σ; checkpoints 4000 to
  16000 kept for the free reads below.

## The arm queue

`/tmp/arm_queue.sh` then `/tmp/arm_queue2.sh` run one arm after another in the background (outside the repo; gone on
reboot). Each is compared against `a1-ref-1ep` at the same images, each config in `configs/` until it is read (`a1-b128-6k`'s stays
while the lr 1.4e-3 arm, which compares against it, runs):
1. `a1-b128-6k`: read (log 2026-11-18-batch-128): batch 128 is 0.0104 ahead at equal images, the lead shrinking.
   Its checkpoint is deleted once the batch question is settled.
2. `a1-b128-lr14-6k`: the same at usui lr 1.4e-3 (fallback 1.75e-3). Running.
3. `a1-attnfirst-6k`: the attention block first in each period of four (F4), 6000 steps.
4. `a1-seed42-6k`: the reference at seed 42, 6000 steps: the seed noise that every arm's read needs.

## Waiting on a decision

- **The long run's batch** (PLAN C1), after arms 1 and 2, and with the seed noise from arm 4.
- **The in-distribution held-out set** (PLAN D6): a few thousand rows of the stronger Fine-T2I subsets kept out of
  training by key. It must exist before the long run starts. `exclude_captions.py` rewrites the defect list on every
  run, so the held-out keys need their own list, or that script changes.
- **Guidance off above t ≈ 0.8** on the anneal's checkpoint, a free sampling read of whether CFG at high noise makes
  the form defects. Proposed, not run.
- **256 px**: the set needs `shuffle.py`'s latent cast to `large_binary` (2 GB per chunk at `binary`), then the
  build; throughput at 256 px is unmeasured.

## Open decisions on the trunk

- **Switches**, each for an open question: `pooled_ada = false` (F3); `layout` and `block_types` (F4).
- **Step speed** (log 2026-11-16-step-profile-d24): the usuitrack session's 140 → 30 ms for the Usui step and the step
  probe's 57.6 ms do not agree yet.
- **Disk** (2026-11-13): luna 89 GB free, mars 318 GB, vel-ssd 198 GB. The merged 128 px set is 380 GB on vel-ssd.
- **usuitrack**: a git dependency at `lab` `92b23bc`.

## Free reads, no training run

On checkpoints of `128_d768x24_1ep` or `a1-ref-1ep`:
- J1: how much the adaLN's vectors vary with t.
- J3 with L1: the cross maps for a de facto sink, the attention maps' movement from 4000 to 16000.
- K2: the spectrum of the token covariance (data only).
- A4: whether the writers' image rows and meta rows differ in gradient spectrum.

## Branches

Not merged, each the record of an arm: `gdn-conv`, `g2-patch16` (closed), `pinned-copy` (PLAN H1, never measured on
the trunk), `arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}` (early arms).

## Traps

- Checkpoints load only into the model they were trained with.
- Without a desktop login `vel-ssd` (LUKS) is not unlocked or mounted, and the training set is not there. The machine
  can suspend while a run trains: launch under `systemd-inhibit --what=sleep:idle`.
- `nvidia-smi` fails with a driver/library mismatch after a driver update until the reboot.
- A run is launched in the background. A run directory that exists is refused; one whose `wandb.init` failed has
  to be deleted. A resumed run takes `init_from` at a checkpoint and `warmup = 0`.
- `pgrep -f <name>` matches the shell wrapper of any agent command that contains the name. Find and kill runs by PID.
- A run's console ends with `shutdown can only be called once` from wandb; it is not a fault.
- pyarrow int32 offsets cap a chunk at 2 GB: take slices.
- wandb groups series on the first slash only; keep keys one level deep.
- With the concat patch embed the patch embed gets zero gradient at step 0; only `ada.1.weight` moves on the first
  step. That is expected.
- The per-role `norm_ratio` in `score.jsonl` reads up to about 1.002 under the ceiling of 1; the mean is 1.0000 at
  the end of a pass. The cause is not read.
