# Handover

For the next agent session, starting cold. Read this, `FACTS.md` and `AGENTS.md` first; `SPEC.md` is the design,
`PLAN.md` the open questions.

## Where it stands (2026-11-20)

The trunk is architecture **a4** (SPEC §3), on branch `trunk/adaln-t-only`, ahead of `main` (which is still
a2); the maintainer merges it into `main`. a4 in one line: d768x24 at 128 px, three gated short-conv blocks (LFM2's,
norm on the read) per period of four with joint attention third, the caption's EmbeddingGemma 2 token states as extra
keys after a two-block caption refiner, a zero-valued null key per KV head (the sink), adaLN-single from t alone with
scale and gate only, concatenating patch embed and slice read-out, UsuiTrack (lr 2e-3, each matrix in its init-norm
ball) plus StochasticAdamW, batch 256 as 4 × 64.

The closures of 2026-11-20, each with its archive note: joint attention, the sink, the caption refiner, the adaLN
without its shift, attention third of four; earlier the same day, the adaLN on t alone and the conv block's form.

Tests: 95 pass (`uv run python -m pytest tests/`, about 40 s). The gradient tests are marked `cuda` (flex_attention has
no CPU backward) and need the GPU; while a run trains they share it, which is fine for the suite.

## Running now

`a4-ref-1ep` (`configs/a4-ref-1ep.toml`, commit `91be26b`): the one-pass reference of a4, 32,173 steps, launched
2026-11-20 about 23:00 under `systemd-inhibit`, PID of the wrapper 420272, console `/tmp/nanodit/a4-ref-1ep.out`,
run directory `/mnt/mars/ai/tiny-dit/runs/a4-ref-1ep`, wandb project `tiny-dit`. About 514 img/s at 198 W, so about
4.5 h plus evals. It scores every 1000 steps into `score.jsonl` (the 1,000 FLUX.2-dev held-out rows at 128 px,
wandb `eval/…`) and `score_lunara.jsonl` (the 1,000 Lunara rows at 256 px, wandb `eval/lunara_…`); checkpoints every
4000, the last kept.

When it ends:
1. Write its log note (`docs/log/2026-11-21-a4-ref-1ep.md`): the score tables by step and by t, R² by band,
   throughput, against `a1-lfmnormread-1ep` (the a2 reference). The comparison has two breaks to state: Lunara is now
   scored at 256 px (a2's Lunara score was at 128 px and does not compare), and the held-out score compares directly
   (same rows, noise and t). a2's held-out at the end: 1.2536.
2. Run the caption-use probe on its last checkpoint (`uv run python scripts/probe_caption_use.py RUN_DIR STEP
   OUT_JSON`, CUDA only) and put the read in the log note.
3. Report to the maintainer, update the docs, commit after a subagent review.
4. Then `a4-base-6k` (`configs/a4-base-6k.toml`): the 6000-step base every a4 arm compares against. It is the
   first 6000 steps of the 1-epoch run (same seed, the lr constant after warmup, nothing depends on `steps`), so the
   1-epoch run's evals up to 6000 serve as the base and this config need not run; it is the template arms copy.

## Next arms, against a4 at 6000 steps (proposed, for the selection of a5)

- **tanh on the adaLN gates** (PLAN J7). The source gives only the motivation (Lumina-Next §2.1: keep large modulation
  values off the residual branch, no ablation). Read on loss and on the gates' size over training.
- **5×5 and 7×7 conv kernels at 256 px** (PLAN G10), read on loss and step time. Needs the 256 px training set:
  `shuffle.py` must cast the latent column to `large_binary` (a 256 px chunk passes pyarrow's 2 GB `binary` limit),
  then the build. Throughput at 256 px is unmeasured.
- **RoPE base** (PLAN F1): base 10000 leaves about half the rotated dims nearly position-free over 16 positions; a
  base near 100 would use them. A one-constant change (`ROPE_BASE` in `model.py`), one arm on a branch.
- Parked: one refiner block against two (likely below one seed's resolution); t at the input without the adaLN.

## Waiting on the maintainer

- Merging `trunk/adaln-t-only` into `main`.
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
keeps its logs and `config.toml`. The references: `a1-lfmnormread-1ep` (a2, score 1.4199 on Lunara at 128 px,
held-out 1.2536), `a1-ref-1ep` with its cooldown `a1-ref-1ep-anneal3k` (a1; the README's grid). The conditioning arms
of 2026-11-20 are in their two log notes. The power limit is 198 W (default 220 W) for every run since 2026-11-18;
it resets on reboot (`pkexec nvidia-smi -pl 198`).

## Branches

`trunk/adaln-t-only`: the trunk, a4. Merged into it and kept as records: `arm/joint-attn`, `arm/caption-refiner`,
`arm/no-shift`. Not merged, each the record of an arm: `arm/joint-probe` (the probes on a3), `arm/joint-probe-a2`,
`arm/caption-read`, `arm/caption-read-probe`, `arm/lfm-forms`, `arm/slim-conv`, `arm/gate-outside-norm`,
`arm/gdn2-lite`, `arm/gdn2-meta`, `arm/gated-conv`, `arm/conv-designs`, `arm/xside`, `gdn-conv`, `g2-patch16`,
`pinned-copy` (PLAN H1), and the early `arm/{base,finalnorm,finalr64,idfinal,noconv,slice,solver,solvermlp}`.
usuitrack is pinned at `lab` `92b23bc`.

## Open on the harness

- Step speed (log 2026-11-16-step-profile-d24): the usuitrack session's 140 → 30 ms for the Usui step and the step
  probe's 57.6 ms do not agree yet.
- The per-role `norm_ratio` in `score.jsonl` reads up to about 1.002 under the ceiling of 1; the cause is not read.
