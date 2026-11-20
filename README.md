# nanodit

A from-scratch text-to-image diffusion transformer, trained on one RTX 4070 Super (12 GB). Two aims: 1) a small
model that renders prompts, trained toward saturating its size on public data in days on one consumer GPU; 2) a
measured record of which of this season's pieces earn their place, each read by held-out loss in a paired arm, one
seed each.

- **Target:** Ostris's *x0 velocity* parameterization of rectified flow
- **Latents:** Ostris's **Primal VAE** (2x2 variant), patched 8×8 by the DiT itself into 128-d tokens
- **Conditioning:** the timestep through adaLN; the caption as extra keys in the attention blocks' joint attention,
  **EmbeddingGemma 2**'s 512-d token states after a two-block caption refiner
- **Blocks:** three gated short-conv blocks (LFM2's), writing all channels, for every attention block; sandwich
  norm on the MLP branch; a concatenating patch embed and a slice read-out, so the residual stream carries the working latent
- **Optimizer:** **UsuiTrack**: the learning rate is a relative step, and each matrix stays in the ball of its init norm
- **Framework:** pure PyTorch, bf16 parameters, single GPU, no trainer framework

It is a research log as much as a codebase: every design choice is in `docs/SPEC.md`, every number behind it in
`docs/log/`, every question still open in `docs/PLAN.md`, and what has to be kept in mind in `docs/FACTS.md`.

## Where it stands

![The 16 fixed eval prompts after 9.06M images at 128 px](docs/log/2026-11-18-samples/a1-ref-1ep-anneal3k_35373.png)

*The 16 fixed eval prompts, each from its own fixed noise: d768 × 24 (205M parameters) after 9.06M images at 128 px,
one pass over the 8.24M-pair training set plus a cooldown of a tenth of a pass, about 4.3 hours on the one GPU. Euler,
30 steps, guidance 4. Mood, palette, layout and lighting follow the prompt; the forms are not finished.*

One model is trained end to end: architecture **a1**, the trunk before its conv blocks were gated, 128 px only; the trunk is now **a4** (`docs/SPEC.md`). The 256 px stage is not trained yet.

| | |
|---|---|
| Training set | text-to-image-2M and Fine-T2I, deduplicated: 8.24M pairs in seven aspect buckets |
| Model | width 768, 24 blocks (18 conv, 6 attention), 191M image parameters + 13M conditioning |
| Optimizer | UsuiTrack on the matrices, relative step 2e-3, each matrix inside the ball of its init norm; StochasticAdamW on the rest |
| Batch | 256 images (16k tokens), constant lr after a short warmup; a cosine cooldown is its own short run |
| Timestep | logit-normal, σ 0.7 |
| Throughput | about 590 images/s at 128 px on an RTX 4070 Super |
| Score | loss on 1,000 Lunara pairs, never trained on (from a4 on, at 256 px, and the eval on 1,000 held-out rows of the training set), 10 fixed t with fixed noise: 1.4242 after one pass, 1.4106 after the cooldown (`docs/log/2026-11-18-*`) |

**How things are decided.** The held-out loss on fixed rows, fixed noise and a fixed t grid is the metric, as in
large-scale scaling work on text-to-image models: it is the training objective itself, comparable across runs, and
far less noisy than sample metrics. Samples are read beside it, for what loss does not measure. A design question is
settled by one paired arm against the reference: the same seed, data order and eval rows, one change. The next
steps (an in-distribution held-out set, 256 px, saturating this size at about 50 image tokens per parameter) are in
`docs/PLAN.md`.

## Layout

```
docs/SPEC.md          the design: data, model, init, optimizer, eval
docs/PLAN.md          open questions, grouped, and how each would be settled
docs/FACTS.md         principles and gotchas, no numbers
docs/log/             the logbook: one note per run or read, with the full tables
docs/topics/          one doc per side quest (block design, priming, edge of stability, ...)
docs/archive/         closed decisions and questions, one note per session
docs/HANDOVER.md      where things stand, what is undecided, traps
docs/REFERENCES.md    sources (papers, model cards, datasets) and what each one establishes
docs/references/      source material kept verbatim
AGENTS.md             working rules for the agents that write this code (CLAUDE.md imports it)
LICENSE               MIT
src/nanodit/          model, objective, sampler, data stream, optimizer split
scripts/              precompute, shuffle, train, score, probes
configs/              run configs that load on the current model, eval prompts
tests/                behaviour tests for all of the above
```

## Running it

```bash
uv sync
uv run python -m pytest tests/
uv run python scripts/train.py configs/a4-ref-1ep.toml
```

torch 2.15 comes from the PyTorch cu132 index. UsuiTrack is a git dependency pinned to a
commit of its `lab` branch. Primal VAE loads via `diffusers` `AutoModel` with
`trust_remote_code`; EmbeddingGemma 2 via `sentence-transformers`.

Training reads precomputed rows, not images. The pipeline, each step a script with its usage in its docstring:

1. `precompute.py`: text-to-image-2M shards to one Parquet per shard (Primal latents at 128 and 256 px, the pooled
   embedding, which the model does not read, the 512-d token states).
2. `stage_fine.py`, `encode_fine.py`: Fine-T2I into aspect-bucket crops, then into the same encodings.
3. `shuffle.py`: both datasets into one training set per resolution, one Parquet per aspect bucket in uniformly
   random row order, each image once.
4. `exclude_captions.py`: the set's `excluded_keys.txt`, the rows whose captions match a defect pattern.
5. `holdout.py`: the set's `heldout_keys.txt` and the held-out eval set, 1,000 rows of the set never streamed.
6. `precompute_lunara.py`: the out-of-distribution eval set, never trained on.
7. `train.py CONFIG.toml`: writes `train.jsonl`, `score.jsonl` (held-out rows), `score_lunara.jsonl` (Lunara at 256 px), eval grids and checkpoints to the run directory and
   mirrors them to wandb.

Scripts take their directories as arguments; the paths in `configs/` are the development machine's. The data is
not part of the repo.

## License

MIT.
