# Handover

State on 2026-11-20: the trunk is architecture a3. Its conv blocks are LFM2's gated short conv with the norm on the
read (archive 2026-11-20-conv-block-lfm-norm-read), and its adaLN takes t alone, the caption entering only by cross
attention (archive 2026-11-20-adaln-takes-t-only). Its last one-pass reference is the a2 run `a1-lfmnormread-1ep`
(score 1.4199; log 2026-11-20-lfm-norm-read-1ep). Next, all on joint attention with the sink (`arm/joint-attn`):
a base rerun without the pooled embedding and with the null key at zero; a caption refiner of two unmodulated blocks
over the token states (PLAN J6); the adaLN without its shift (J4); then the merge of joint attention (J2). `PLAN.md` carries the questions, `FACTS.md` the measurements, `docs/log/` the run
notes, `AGENTS.md` the rules.

## Where it stands

`main` is the trunk: the architecture of SPEC §3, the switches of the questions still open, and the docs. 91 tests
pass (`uv run python -m pytest tests/`, about 120 s on the CPU).

The model: every 4th block attention (GQA, RoPE on a quarter of the frequencies), the other three a gated short-conv
block writing all channels; cross attention to the caption's token states in every attention block; sandwich norm
on the attention, cross-attention and MLP branches; adaLN-single from t alone; concatenating patch embed
and slice read-out; every Usui matrix framed on its input side and inside the ball of its init norm, usui lr 2e-3.
Width 768 × depth 24 at 128 px (237.7M parameters), batch 256 as 4 × 64.

Runs, under `/mnt/mars/ai/tiny-dit/runs/`:
- `a1-lfmnormread-1ep` (`arm/lfm-forms` `5ad4c3e`): the a2 reference, 32,173 steps, 4 h 32 min at 198 W. Its
  `config.toml` and last checkpoint (the only one kept) are migrated to a2's names and no longer load into a3;
  `configs/a3-ref-1ep.toml` is its config on the trunk, not yet run. It scores the held-out rows (1.2536 at the end).
- `a1-lfmnormread-6k`: the same form for 6000 steps, a2.
- `a2-joint-6k`, `a2-jointsink-6k`, `a2-jointsink-nopool-6k`, `a2-jointsink-captionread-6k`: the joint-attention
  arms (`arm/joint-attn`, `arm/caption-read`), 6000 steps; their log notes are not written yet.
- `a1-ref-1ep` (basic conv, from `768745b`) and its cooldown `a1-ref-1ep-anneal3k` (score 1.4106, the README's grid):
  architecture a1. Their checkpoints no longer load into the trunk; kept until the maintainer decides.
- `a1-seed42-6k`, `a1-gatedconv-6k`: a1 checkpoints at 6000, kept for band reads; no longer load into the trunk.

The power limit is 198 W (default 220 W, max 275 W); all arms since 2026-11-18 ran at 198 W.

## Waiting on a decision

- The a1 checkpoints above: AGENTS deletes a checkpoint that no longer loads into the model.
- **256 px**: the set needs `shuffle.py`'s latent cast to `large_binary` (2 GB per chunk at `binary`), then the
  build; throughput at 256 px is unmeasured.

## Open decisions on the trunk

- **Switches**, each for an open question: `layout` and `block_types` (F4).
- **Step speed** (log 2026-11-16-step-profile-d24): the usuitrack session's 140 → 30 ms for the Usui step and the step
  probe's 57.6 ms do not agree yet.
- **usuitrack**: a git dependency at `lab` `92b23bc`.

## Free reads, no training run

- J1 (parked): how much the adaLN's vectors vary with t.
- K2: the spectrum of the token covariance (data only).
- A4: whether the writers' image rows and meta rows differ in gradient spectrum.

## Branches

Not merged, each the record of an arm: `arm/joint-attn` (open, J2), `arm/joint-probe`, `arm/caption-read`,
`arm/caption-read-probe`, `arm/lfm-forms`, `arm/slim-conv`, `arm/gate-outside-norm`, `arm/gdn2-lite`,
`arm/gdn2-meta`, `arm/gated-conv`, `arm/conv-designs`, `arm/xside`, `gdn-conv`, `g2-patch16` (closed), `pinned-copy` (PLAN H1, never measured on the
trunk), `arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}` (early arms).

## Traps

- Checkpoints load only into the model they were trained with.
- Without a desktop login `vel-ssd` (LUKS) is not unlocked or mounted, and the training set is not there. The machine
  can suspend while a run trains: launch under `systemd-inhibit --what=sleep:idle`.
- `nvidia-smi -pl` needs `pkexec`, which opens a password dialog on the desktop; it cannot be confirmed remotely.
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
