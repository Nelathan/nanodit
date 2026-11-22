# nanodit

A from-scratch text-to-image diffusion transformer, trained on one RTX 4070 Super (12 GB). Two aims: 1) a small
model that renders prompts, trained toward saturating its size on public data in days on one consumer GPU; 2) a
measured record of which of this season's pieces earn their place, each read by held-out loss in a paired arm, one
seed each.

- **Target:** Ostris's *x0 velocity* parameterization of rectified flow
- **Latents:** Ostris's **Primal VAE** (2x2 variant), patched 8×8 by the DiT itself into 128-d tokens
- **Conditioning:** the timestep through adaLN; the caption as extra keys in the attention blocks' joint attention,
  **EmbeddingGemma 2**'s 512-d token states after a two-block caption refiner
- **Blocks:** three gated short-conv blocks (LFM2's) for every joint-attention block; a concatenating patch embed and
  a slice read-out, so the residual stream carries the working latent
- **Optimizer:** **UsuiTrack**: the learning rate is a relative step, and each matrix stays in the ball of its init norm
- **Framework:** pure PyTorch, bf16 parameters, single GPU, no trainer framework

It is a research log as much as a codebase: every design choice is in `docs/SPEC.md`, every number behind it in
`docs/log/`, every question still open in `docs/PLAN.md`, and what has to be kept in mind in `docs/FACTS.md`.

## Where it stands

![The 16 fixed eval prompts after 9.77M images at 128 px, guidance 1](docs/log/2026-11-21-samples/a4-ref-1ep-anneal6k_38173_cfg1.png)

*The 16 fixed eval prompts, each from its own fixed noise: d768 × 24 (238M parameters) after 9.77M images at 128 px,
one pass over the 8.24M-pair training set plus a cooldown of a fifth of a pass, about 5.7 hours on the one GPU. Euler,
30 steps, no guidance (CFG 1). The same model and noise at guidance 4:
[`a4-ref-1ep-anneal6k_38173_cfg4.png`](docs/log/2026-11-21-samples/a4-ref-1ep-anneal6k_38173_cfg4.png); there the
named objects are present and placed, faces and hands are not formed.*

One model is trained end to end: the trunk architecture **a4** (`docs/SPEC.md`, below), 128 px only. The 256 px stage
is not trained yet.

| | |
|---|---|
| Training set | text-to-image-2M and Fine-T2I, deduplicated: 8.24M pairs in seven aspect buckets |
| Model | width 768, 24 blocks (18 conv, 6 attention), 225M image parameters + 13M conditioning |
| Optimizer | UsuiTrack on the matrices, relative step 2e-3, each matrix inside the ball of its init norm; StochasticAdamW on the rest |
| Batch | 256 images (16k tokens), constant lr after a short warmup; a cosine cooldown is its own short run |
| Timestep | logit-normal, σ 0.7 |
| Throughput | about 520 images/s at 128 px on an RTX 4070 Super at 198 W |
| Score | flow loss at 10 fixed t with fixed noise. Held-out: 1,000 FLUX.2-dev rows of the training sources kept out of training, at 128 px: 1.2527 after one pass, 1.2322 after the cooldown. Lunara: 1,000 pairs from another source, at 256 px, off the training resolution: 1.3164, then 1.3030 (`docs/log/2026-11-21-*`) |

**How things are decided.** The held-out loss on fixed rows, fixed noise and a fixed t grid is the metric, as in
large-scale scaling work on text-to-image models: it is the training objective itself, comparable across runs, and
far less noisy than sample metrics. Samples are read beside it, for what loss does not measure. A design question is
settled by one paired arm against the reference: the same seed, data order and eval rows, one change. The next
steps (256 px, saturating this size at about 50 image tokens per parameter) are in
`docs/PLAN.md`.

## Architecture

The model is a diffusion transformer over the latent's tokens, with three changes to the usual block stack: most
blocks mix by a short convolution instead of attention, the caption is read inside self-attention, and the residual
stream carries the latent itself.

- **Tokens and stream.** Primal VAE encodes an image into a 2-channel latent at half resolution; the DiT cuts it into
  8×8 patches, 128 numbers each (64 tokens at 128 px). The stream is 768 wide: the first 128 channels are the raw
  noisy patch, the other 640 a linear projection of it. The prediction is read straight off the first 128 channels
  after the last block, with no final layer. Each block adds a correction to a stream that already holds the input.
  Every block starts as the identity (its gates are zero), and under the *x0 velocity* target that untrained model
  predicts zero velocity: it starts by passing the noisy latent through and learns what to correct.
- **Blocks.** A period of four repeats six times: conv, conv, attention, conv. Each block is a mixer (across tokens)
  then a SwiGLU MLP (per token), each read through an RMS norm scaled per image by the timestep and written back
  through a per-image gate (adaLN-single, PixArt's: one projection shared by all blocks plus a small table per block).
  Each branch is normed before it writes, so a small read makes a small write: the MLP's output passes an RMS norm
  before its gate (sandwich norm), the conv's norm is on its read and attention's is per head.
  The conditioning vector is the timestep alone; there is no shift, only scale and gate.
- **The conv mixer** is LFM2's gated short conv: from the block's input, three projections B, C and v; a depthwise
  3×3 convolution over the token grid reads B ⊙ v; its result, normalized, times C goes through an output projection
  into all channels. It costs linearly in the number of tokens, so at large images it is cheap where attention is not,
  and it gives the network its sense of neighbourhood.
- **Joint attention** in every fourth block: the image tokens' queries read, in one softmax, the image tokens' keys
  (with 2D RoPE on a quarter of the frequencies) and the caption's keys (no positions). Grouped-query, 12 query heads
  over 3 key/value heads. Each head has a learned null key with value zero, a sink: it lets a head read nothing,
  which ordinary softmax cannot. The head outputs pass a sigmoid gate before the output projection.
- **The caption.** EmbeddingGemma 2 encodes the caption; its 512-d token states pass a two-block refiner
  (self-attention and SwiGLU over the caption alone, no timestep) and then serve as keys and values in every attention
  block, through a projection of each block's own. The caption reaches the image only through attention. An empty
  caption, and 10 % of training captions, are replaced by learned null tokens, which makes classifier-free guidance
  possible at sampling.
- **Training.** Rectified flow with the x0-velocity target, timesteps logit-normal (σ 0.7). UsuiTrack trains every
  matrix with a relative step and keeps each inside the ball of its initial norm; everything else trains with AdamW.
  The design in full, with every init and every choice, is `docs/SPEC.md`; why each piece is there is in
  `docs/archive/`.

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
7. `train.py CONFIG.toml`: writes `train.jsonl`, `score.jsonl` (held-out rows), `score_lunara.jsonl` (Lunara at 256 px), the same on the uniform t grid as `score_uniform.jsonl` and
   `score_lunara_uniform.jsonl`, eval grids and checkpoints to the run directory and
   mirrors them to wandb.

Scripts take their directories as arguments; the paths in `configs/` are the development machine's. The data is
not part of the repo.

## License

MIT.
