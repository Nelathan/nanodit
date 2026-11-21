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
README's grids). The gates' size after one pass: log 2026-11-21-gate-sizes.

Tests: 95 (`uv run python -m pytest tests/`). The gradient tests are marked `cuda` (flex_attention has no CPU
backward) and need the GPU; with `CUDA_VISIBLE_DEVICES=""` they are skipped.

## Running now

`/tmp/nanodit/queue.sh` under `systemd-inhibit`, console `/tmp/nanodit/queue.out`: the a5 arms in turn, each from a
detached worktree `/tmp/nanodit/wt-<name>` of its branch (removed when its run ends), against the first 6000 steps of
`a4-ref-1ep` (same seed, constant lr after warmup, nothing depends on `steps`; `configs/a4-base-6k.toml` is the
template and need not run). About an hour each, from 09:24:

1. `a4-conv5-6k` (`arm/conv5`): the depthwise conv 5×5. PLAN G10.
2. `a4-conv7-6k` (`arm/conv7`): 7×7. PLAN G10.
3. `a4-tanh-gates-6k` (`arm/tanh-gates`): tanh on both gates. PLAN J7.

Then `/tmp/nanodit/bench.sh` (console `/tmp/nanodit/bench.out`): `scripts/bench_step.py` on the trunk and both conv
arms, the 1,000 Lunara rows at 128 px (micro-batch 64) and 256 px (micro-batch 16, the same tokens).

When they end: for each arm, held-out and Lunara at 256 px against `a4-ref-1ep`'s evals at 1000…6000, by t and R² by
band; img/s from `train.jsonl` and the bench. For tanh, `scripts/probe_gates.py` on its last checkpoint against the
trunk's gates (the trunk has no checkpoint at 6000: compare the pre-tanh sizes, and how many saturate). One log note
per arm, PLAN's state, then bring the readings to the maintainer. Delete each arm's config after the read; its run
directory keeps it.

## Next arms, proposed

- **Conv padding** (PLAN G10): replicate against zero, read on Lunara at 256 px; after the kernel is chosen.
- Parked: one refiner block against two (likely below one seed's resolution); t at the input without the adaLN.

## Waiting on the maintainer

- Whether `count_params.py`'s split is right for a4: the refiner and the null keys now count as conditioning, with the
  caption's keys and values; the scaling law's N is the image part.

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
keeps its logs and `config.toml`. a4: `a4-ref-1ep` keeps step 32173, `a4-ref-1ep-anneal6k` step 38173. Earlier references: `a1-lfmnormread-1ep` (a2, score 1.4199 on Lunara at 128 px,
held-out 1.2536), `a1-ref-1ep` with its cooldown `a1-ref-1ep-anneal3k` (a1). The conditioning arms
of 2026-11-20 are in their two log notes. The power limit is 198 W (default 220 W) for every run since 2026-11-18;
it resets on reboot (`pkexec nvidia-smi -pl 198`).

## Branches

`main`: the trunk, a4, with `trunk/adaln-t-only` merged into it. Running: `arm/conv5`, `arm/conv7`, `arm/tanh-gates`. Merged into it and kept as records: `arm/joint-attn`, `arm/caption-refiner`,
`arm/no-shift`. Not merged, each the record of an arm: `arm/joint-probe` (the probes on a3), `arm/joint-probe-a2`,
`arm/caption-read`, `arm/caption-read-probe`, `arm/lfm-forms`, `arm/slim-conv`, `arm/gate-outside-norm`,
`arm/gdn2-lite`, `arm/gdn2-meta`, `arm/gated-conv`, `arm/conv-designs`, `arm/xside`, `gdn-conv`, `g2-patch16`,
`pinned-copy` (PLAN H1), and the early `arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}`.
usuitrack is pinned at `lab` `92b23bc`.

## Open on the harness

- Step speed (log 2026-11-16-step-profile-d24): the usuitrack session's 140 → 30 ms for the Usui step and the step
  probe's 57.6 ms do not agree yet.
- The per-role `norm_ratio` in `score.jsonl` reads up to about 1.002 under the ceiling of 1; the cause is not read.
