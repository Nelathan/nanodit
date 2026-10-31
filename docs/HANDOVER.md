# Handover

Written 2026-10-31, night. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

Built and tested on CPU (`uv run python -m pytest tests/`, 43 pass): the DiT, the x0-velocity objective, the Euler/CFG sampler, the Parquet loader, the Usui + AdamW split with Molegrad and gradient accumulation, `scripts/train.py`, the precompute and the probes. Nothing has trained; the GPU was busy.

Data: the 128 px precompute of shards 0–3 runs in the background on CPU (`scripts/precompute.py 128 0 1 2 3`, log `/mnt/mars/ai/tiny-dit/precomputed/precompute_128.log`, rate in FACTS, done around 06:00 on 2026-11-01). Output: `/mnt/mars/ai/tiny-dit/precomputed/128/data_00000N.parquet`, 50k rows each; each tar is deleted after its parquet is written. Shard 0's tar is also hard-linked as `data/data_512_2M/keep_data_000000.tar`, kept until shard 0's parquet is checked, then removed. The null embedding is at `precomputed/null_embedding.safetensors`.

## Waiting on Daniel

- **A1, the optimizer split.** Built as option (a), provisional (`src/tinydit/optim.py`). Its costs, including the step-1 zero-gradient frame fit, are in PLAN A1.
- **C4, training precision.** Molegrad fails under bf16 autocast; `train.py` runs fp32 + TF32, provisional.
- **F1–F3.** Model choices SPEC leaves open. F3 is the sharpest: the final layer's `1 + scale` starts far from 1 (FACTS, adaLN at init).
- **B1, B2, A2.** Evidence is in; each needs a close or a follow-up.
- **SPEC §5 wording.** The text-projection row said "unit variance per dim", which FACTS contradicts; it now says "unit RMS over entries". The design is unchanged; revert if you meant something else.
- **Not chosen yet:** C3 shift base, D3 eval prompts (16 drafts in `configs/eval_prompts.txt`), D4 sampler grid, an LR schedule (C2), `fallback_lr` as a sweep axis.

## Next

1. When the GPU is free and shards exist: the smoke run, in the background:
   `cd /home/djg/Projects/tiny-dit && mkdir -p /mnt/mars/ai/tiny-dit/runs && nohup uv run python scripts/train.py configs/128_smoke.toml > /mnt/mars/ai/tiny-dit/runs/128_smoke.log 2>&1 &`
   It measures step time and memory under fp32 + TF32 (C4, C1); the config's LRs are placeholders.
2. The LR sweep (C2) once A1 and C4 are settled.

## Traps

- Precompute: one bad sample or an OOM restarts the shard from row 0 (the tar survives); a malformed tar entry is dropped or raises; nothing checks for 50k rows. Its peak RSS (FACTS) leaves little room to train alongside it.
- The 256 px pass re-downloads each shard, because the 128 px pass deletes the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things".
