# Priming on the Primal latent

Topic doc for PLAN I1, opened 2026-11-05 at Daniel's request ("Create the doc now, plan has just a reference"). It holds
this line of work's plan, design and reads; measured numbers still go to FACTS, closed decisions to `archive/`. The code
gets its own branch when it starts.

## Source

Kim, Park et al., "Are We Overthinking One-Step Generation?" (Priming Models), arXiv 2610.20331; repo
https://github.com/JaeyeonKim01/priming; Moritz Piening's blog on Stage-1 choices,
https://moepien.github.io/blog/better-priming/. Known here only through Daniel's writeup of two X threads, kept verbatim
in `references/priming-threads.md`. Nothing below is checked against the paper or the code yet.

The method as the writeup gives it:
- Stage 1, reconstruction pretraining: one network learns to reconstruct clean images from corrupted inputs. The
  simplest form denoises at a single fixed noise level with no time embedding; masking-based reconstruction and
  reconstruction in a pretrained autoencoder's latent also work. At generation the same network takes pure Gaussian
  noise.
- Stage 2, distributional post-training: the one-step generator's samples are matched to real samples in a frozen
  feature space, by Sinkhorn divergence / optimal transport, or simpler pairwise-distance, Laplacian-kernel or FD
  losses. About 1,250 updates already take Priming-B to FID 2.26. Reconstruction pretraining is said to align the
  network's Jacobian with the correction directions the distributional loss supplies.
- Claimed: ImageNet 256 one-step FID 1.19 (Priming-L) at about a sixth of Drifting-L's compute; text-to-image GenEval
  0.82 one-step, 0.844 two-step; world models; one-step action policies near a 9-step flow-matching baseline, with the
  distributional loss on the raw normalized action chunks and no feature encoder.
- Works with very few real references per condition; text-to-image and world models have essentially one.
- Stage 1 is sensitive to its noise level s (Priming-B, 80 epochs + 12,500 OT updates, FID): 0.5 → 2.46, 0.7 → 1.76,
  0.8 → 1.65, 0.9 → 1.97, 1.0 → 65.08. Their intuition: close enough to pure Gaussian for generation, while keeping
  enough of the clean image to learn from.
- Open in the source: the dependence on a pretrained feature encoder for stage 2.

## Daniel's direction (2026-11-05)

- "I1 on the primal vae, like masked language modeling i assume." Stage 1 runs in the Primal latent, as a
  reconstruction of corrupted latent tokens, the image analogue of masked language modeling.
- "This is not trivial code."
- "V0 makes this quite honest i think." He meant x0 (there is no V0): the x0 velocity target we train with (SPEC §4, Ostris): the output is
  image-shaped at every t, so a reconstruction objective on corrupted latents fits it without a new head.
- Its own topic doc (this file) and branch.

## How it would map onto tiny-dit (my reading, untested)

- The current model is a stage-1 network at every t at once, conditioned on t. Priming's stage 1 fixes one corruption
  level and drops t. Two ways in: train stage 1 from scratch at one corruption, or take the hero checkpoint at one fixed
  t as the stage-1 network (the source names fine-tuning multi-step models into one-step generators as related work and
  asks how simple the preparation can be).
- Our identity slice readout starts every network at v̂ = 0, the output equal to its input, so a reconstruction starts
  from "return the corrupted latent" and every correction is learned.
- Masked reconstruction in the latent: mask a share of the 64 tokens (replace by noise or a learned mask vector),
  reconstruct the whole latent. How generation then starts from pure noise, and how a mask ratio relates to their s, is
  for the paper to say.
- How their s maps onto our x_t = (1 − t) x₀ + t ε is unchecked.
- Stage 2 feature space, cheapest first: the Primal latent itself with Euclidean distances (the action-chunk result
  works on raw data); EmbeddingGemma 2's own vision tower on decoded samples (the cached checkpoint carries
  `vision_tower` and `embed_vision`; our text path loads it with `vision_config=None`), a frozen image space that may
  share the embedding space our captions come from, at the cost of a VAE decode per sample. Whether its image and text
  embeddings are aligned is unchecked.
- One reference per prompt is our data's shape: each caption has one image.
- Eval: the score grid measures a flow model at 10 t and does not apply to a one-step generator. Samples on the eval
  prompts, and FID once it exists (PLAN D5).

## Code it needs (sketch)

- A corruption for stage 1 (fixed noise level, or a token mask) and a loss on the clean latent; t held fixed or the time
  path removed.
- A one-step sampler: pure noise and a caption in, a latent out.
- Stage 2: a batch of generated latents per step against real ones, a Sinkhorn or pairwise loss in the chosen feature
  space, gradients through the generator only.
- Which optimizer setup stage 2 takes (Usui at a small relative step, as a fine-tune) is open.

## Open

- Which start: from scratch at one corruption, or from the hero checkpoint.
- Masking or fixed-level noise for stage 1.
- The stage-2 feature space.
- Read the paper and the repo before any code.
