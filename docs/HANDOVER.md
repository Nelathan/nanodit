# Handover

Written 2026-10-31, night; updated 2026-11-01. Orientation only; `PLAN.md` carries the questions, `FACTS.md` the measurements, `AGENTS.md` how to run things.

## Where it stands

Built and tested (`uv run python -m pytest tests/`, 46 pass): the DiT, the x0-velocity objective, the Euler/CFG sampler, the Parquet loader, the Usui + StochasticAdamW split with Molegrad and gradient accumulation, `scripts/train.py` training a pure bf16 model, the precompute and the probes. Three 300-step smoke runs at 128 px are in FACTS (GPU smoke runs): fp32 at two seeds, bf16 at one.

Data: the 128 px precompute of shards 0–3 is done (2026-11-01 04:39, log `/mnt/mars/ai/tiny-dit/precomputed/precompute_128.log`): `/mnt/mars/ai/tiny-dit/precomputed/128/data_00000{0..3}.parquet`, the tars deleted. Two image sources, half each (FACTS, Sample and shard), row counts and sizes there too. Shard 0's parquet is checked against its tar (FACTS). The null embedding is at `precomputed/null_embedding.safetensors`.

## Waiting on Daniel

- **F1, F2.** Model choices SPEC leaves open (PLAN group F). F3 is the session's to settle (Next).
- **B1, B2, A2.** Evidence is in; each needs a close or a follow-up.
- **Not chosen yet:** C3 shift base, D3 eval prompts (16 drafts in `configs/eval_prompts.txt`), D4 sampler grid, an LR schedule (C2), `fallback_lr` as a sweep axis.

## Next

1. A1, decided by Daniel 2026-11-01, being built: the final linear on Usui with a norm floor, and no frame fit from a zero gradient, both on usuitrack branch `zero-init`; then F3 and the text RMS norm.
2. The LR sweep (C2). A run is launched in the background: `cd /home/djg/Projects/tiny-dit && nohup uv run python scripts/train.py configs/<name>.toml > /mnt/mars/ai/tiny-dit/runs/<name>.log 2>&1 &`; a run directory that exists is refused.

## Traps

- Precompute: one bad sample or an OOM restarts the shard from row 0 (the tar survives); a malformed tar entry is dropped or raises; nothing checks for 50k rows. Its peak RSS (FACTS) leaves little room to train alongside it.
- The 256 px pass re-downloads each shard, because the 128 px pass deletes the tars (7.5 to 11 GB each, FACTS). Encoding both resolutions in one pass would avoid it.
- `load_shards` peaks at about 3× the data size (FACTS); by the code, the Arrow tables, the per-shard copies and the cat result are alive at once. Fine for shards 0–3 at 128 px; about ten shards at 128 px, or shards 0–3 at 256 px, would not fit in 31 GB. Filling one preallocated tensor shard by shard would bring it near 1×.
- GPU sharing, the data location and the editable usuitrack are in `AGENTS.md`, "Running things".
