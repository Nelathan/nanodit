# nanodit

A from-scratch text-to-image diffusion transformer, trained on one RTX 4070 Super (12 GB).
The question it answers: can a small DiT built on this season's promising pieces produce images at all, and which
of those pieces earn their place when each is measured?

- **Target:** Ostris's *x0 velocity* parameterization of rectified flow
- **Latents:** Ostris's **Primal VAE** (2x2 variant), patched 8×8 by the DiT itself into 128-d tokens
- **Conditioning:** **EmbeddingGemma 2**: the pooled 768-d caption embedding through adaLN, and optionally
  cross attention to its 512-d token states
- **Blocks:** three gated depthwise-conv blocks (a GDN2-style local update) for every attention block; sandwich
  norm; a concatenating patch embed and a slice read-out, so the residual stream carries the working latent
- **Optimizer:** **UsuiTrack**, each matrix pinned to its init norm, so the learning rate is a constant relative step
- **Framework:** pure PyTorch, bf16 parameters, single GPU, no trainer framework

It is a research log as much as a codebase: every design choice is in `docs/SPEC.md`, every number behind it in
`docs/FACTS.md` and `docs/log/`, every question still open in `docs/PLAN.md`.

## Where it stands

128 px only so far. The model, objective, sampler, data stream and training harness are built and tested; the
architecture was settled by short paired arms (3000 to 6000 steps at width 768, depth 12, about 123M parameters). No
long run of the current architecture exists yet; the longest run, five epochs of an earlier one, is in FACTS. The
256 px stage and the second dataset (Fine-T2I) are encoded in part and not trained on.

![16 fixed prompts after 6000 steps at 128 px](docs/log/2026-11-07-samples/arch_cross_6k.png)

*The 16 fixed eval prompts after 6000 steps of batch 256 (about 35 minutes), 128 px, the cross-attention arm: mood and
palette follow the prompt, shapes are starting. This is a trial arm, not a result.*

## Layout

```
docs/SPEC.md          the design: data, model, init, optimizer, eval
docs/PLAN.md          open questions, grouped, and how each would be settled
docs/FACTS.md         measured numbers with source and date
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
uv run python scripts/train.py configs/128_d768x12_gdn_6k.toml
```

torch 2.15 comes from the PyTorch cu132 index. UsuiTrack is an editable path dependency on
a sibling checkout (`AGENTS.md`, "Running things"). Primal VAE loads via `diffusers` `AutoModel` with
`trust_remote_code`; EmbeddingGemma 2 via `sentence-transformers`.

Training reads precomputed rows, not images. The pipeline, each step a script with its usage in its docstring:

1. `precompute.py`: text-to-image-2M shards to one Parquet per shard (Primal latents at 128 and 256 px, the pooled
   embedding, the 512-d token states).
2. `shuffle.py`: the shards into one Parquet per resolution in uniformly random row order, each image once.
3. `precompute_lunara.py`: the eval set every run is scored on, never trained on.
4. `train.py CONFIG.toml`: writes `train.jsonl`, `score.jsonl`, eval grids and checkpoints to the run directory and
   mirrors them to wandb.

Scripts take their directories as arguments; the paths in `configs/` are the development machine's. The data is
about 160 GB and is not part of the repo.

## License

MIT.
