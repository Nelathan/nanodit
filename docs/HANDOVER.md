# Handover

For the next agent session, starting cold. Read this, `FACTS.md` and `AGENTS.md` first; `SPEC.md` is the design,
`PLAN.md` the open questions.

## Where it stands (2026-11-21)

The trunk is architecture **a4** (SPEC §3), on `main`; `trunk/adaln-t-only` is the branch it was built on, merged. a4 in one line: d768x24 at 128 px, three gated short-conv blocks (LFM2's,
norm on the read) per period of four with joint attention third, the caption's EmbeddingGemma 2 token states as extra
keys after a two-block caption refiner, a zero-valued null key per KV head (the sink), adaLN-single from t alone with
scale and gate only, concatenating patch embed and slice read-out, UsuiTrack (lr 2e-3, each matrix in its init-norm
ball) plus StochasticAdamW, batch 256 as 4 × 64.

Done: `a4-ref-1ep` (one pass, log 2026-11-21-a4-ref-1ep: ahead of a2 by a small margin, all at high t) and its
6000-step cooldown `a4-ref-1ep-anneal6k` (log 2026-11-21-anneal6k; held-out 1.2322, Lunara at 256 px 1.3030; the
README's grids). The gates' size after one pass: log 2026-11-21-gate-sizes; the stream's size through depth: log
2026-11-21-stream-scale, now the topic `topics/stream-scale.md` (PLAN G11). The 5×5 / 7×7 conv and tanh-gate arms
at 6000 steps with the step bench: log 2026-11-21-conv-kernel-and-tanh-gates. The pass at half the lr, annealed:
log 2026-11-21-lr1e3-one-pass (behind by 0.017 / 0.011; the first pass stays at 2e-3).

`main` is pushed up to `191bed5`; the commits after it are local.

Tests: 95 (`uv run python -m pytest tests/`). The gradient tests are marked `cuda` (flex_attention has no CPU
backward) and need the GPU; with `CUDA_VISIBLE_DEVICES=""` they are skipped.

## Running now

`/tmp/nanodit/graded.sh` under `systemd-inhibit` (console `/tmp/nanodit/graded.out`), from 2026-11-21 17:51, about
55 min each: from one worktree of `arm/conv-kernel` (the conv kernel a field of the block type), four 6000-step layouts in turn: `a4-conv-graded-6k` (7/5/3 by thirds),
`a4-conv-graded-soft-6k` (5/3/3), `a4-conv-phased-6k` (three attention + 7×7, twelve 3×3, three attention + 5×5),
`a4-conv-period-6k` (attention, 7×7, 5×5, 3×3). PLAN G10. Read each on loss against `a4-ref-1ep`'s first 6000 and
on `scripts/probe_conv_taps.py` at step 6000 (run with the arm's code) before its checkpoint is deleted.

Then `/tmp/nanodit/stream.sh`, waiting on it under `systemd-inhibit` (console `/tmp/nanodit/stream.out`): the stream arms (`topics/stream-scale.md`, PLAN
G11), each from a detached worktree of its branch, removed after: `a4-split-read-6k` (`arm/split-read`),
`a4-held-meta-6k` (`arm/held-meta`, on split-read), `a4-held-meta-skips-6k` (`arm/held-meta-skips`, on held-meta),
`a4-no-tables-6k` (`arm/no-tables`, on `main`: the trunk, not on whichever stream arm holds, since those read after
it runs). Read on loss and on `scripts/probe_gates.py` and `scripts/probe_stream.py` at step 6000, with the arm's
code; keep each checkpoint until probed. `scripts/probe_residual_writes.py` asserts a block's output is the sum of
its writes, which the held meta stream breaks; it needs the hold re-applied before it reads those two arms. The
changed cuda tests of `arm/no-tables` (`test_optim`, `test_train`, the gradient-role test) are unverified; run them
on the GPU when it is free.

## Waiting on the maintainer

- **The t-dependent skip** (PLAN K1): K1 prefers the skip at the readout; proposed in its place, the skip at the
  input (the slice starts at c(t)·x_t), and a CPU probe first, the trunk's correction regressed on x_t by t on
  `a4-ref-1ep`. PLAN K1 changes once the form is chosen.
- **The 5×5 kernel** (PLAN G10): −0.0015 on both sets, inside the seed spread, at 1 % of the step. Merge or wait.
- **tanh on the gates** (PLAN J7): level at 6000 steps; proposed as superseded by G11.
- Whether `count_params.py`'s split is right for a4: the refiner and the null keys now count as conditioning, with the
  caption's keys and values; the scaling law's N is the image part.

## Next arms, proposed

- **Conv padding** (PLAN G10): replicate against zero, read on Lunara at 256 px; after the kernel is chosen.
- Parked: one refiner block against two (likely below one seed's resolution); t at the input without the adaLN.

## Code facts that are not obvious from SPEC

- The attention blocks' caption projections keep the names `cross_kv` and `cross_k_norm` from the cross-attention
  era; renaming them would break every a4 checkpoint, so they stay until a change breaks the checkpoints anyway.
- `Config.eval_set` is the held-out rows and must be out of the training stream (asserted at start);
  `lunara_set` at `lunara_resolution` is the out-of-distribution score. A run continuing one that trained on the
  held-out rows cannot start.
- `scripts/score_checkpoint.py` scores a checkpoint on one set at one resolution and writes `score_<res>.jsonl`; it
  does not know the Lunara/eval split of training.
- `scripts/probe_cross_attention.py` measures the removed SDPA cross attention; it stays as the source of the
  numbers in `log/2026-11-17-caption-outliers-and-oom.md`.

## Runs and checkpoints

Under `/mnt/mars/ai/tiny-dit/runs/`. No checkpoint of an earlier architecture is kept: none loads into a4. Each run
keeps its logs and `config.toml`. a4: `a4-ref-1ep` keeps step 32173, `a4-ref-1ep-anneal6k` step 38173; the 6000-step arms keep none. Earlier references: `a1-lfmnormread-1ep` (a2, score 1.4199 on Lunara at 128 px,
held-out 1.2536), `a1-ref-1ep` with its cooldown `a1-ref-1ep-anneal3k` (a1). The conditioning arms
of 2026-11-20 are in their two log notes. The power limit is 198 W (default 220 W) for every run since 2026-11-18;
it resets on reboot (`pkexec nvidia-smi -pl 198`).

## Branches

`main`: the trunk, a4, with `trunk/adaln-t-only` merged into it. Queued: `arm/conv-kernel`, `arm/split-read`,
`arm/held-meta`, `arm/held-meta-skips`, `arm/no-tables`. Read, waiting for a decision: `arm/conv5`, `arm/conv7`,
`arm/tanh-gates` (their checkpoints deleted). Merged into it and kept as records: `arm/joint-attn`,
`arm/caption-refiner`, `arm/no-shift`. Not merged, each the record of an arm: `arm/joint-probe` (the probes on a3),
`arm/joint-probe-a2`, `arm/caption-read`, `arm/caption-read-probe`, `arm/lfm-forms`, `arm/slim-conv`,
`arm/gate-outside-norm`, `arm/gdn2-lite`, `arm/gdn2-meta`, `arm/gated-conv`, `arm/conv-designs`, `arm/xside`,
`gdn-conv`, `g2-patch16`, `pinned-copy` (PLAN H1), and the early
`arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}`. usuitrack is pinned at `lab` `92b23bc`.

## Open on the harness

- Step speed (log 2026-11-16-step-profile-d24): the usuitrack session's 140 → 30 ms for the Usui step and the step
  probe's 57.6 ms do not agree yet.
- The per-role `norm_ratio` in `score.jsonl` reads up to about 1.002 under the ceiling of 1; the cause is not read.
