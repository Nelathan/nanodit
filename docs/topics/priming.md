# Priming on the Primal latent

Topic doc for PLAN I1, opened 2026-11-05 at Daniel's request ("Create the doc now, plan has just a reference"). It holds
this line of work's plan, design and reads; measured numbers still go to FACTS, closed decisions to `archive/`. The code
gets its own branch when it starts.

## Source

Kim, Park et al., "Are We Overthinking One-Step Generation?" (Priming Models), arXiv 2610.20331; repo
https://github.com/JaeyeonKim01/priming; Moritz Piening's blog on Stage-1 choices,
https://moepien.github.io/blog/better-priming/. Known here only through Daniel's writeup of two X threads, kept verbatim
in `references/priming-threads.md`. The repo's code is read (next section); the paper only through a summary.

## What the code does (repo read 2026-11-07: `pretrain.py`, `models/priming.py`, `losses/`, the configs)

- Stage 1 input `(1 − s) x + s ε` at s = 0.8, enforced in `pretrain.py`; target the clean image (x-prediction).
- No time input: the network has no timestep path at all. Class conditioning with 8 class tokens, no label dropout.
- Loss: per-sample pixel MSE divided by its own detached value plus eps (adaptive normalisation, in the code only),
  plus LPIPS (0.4) and a ConvNeXt feature loss (0.1) on random resized crops at 224.
- A masking corruption exists: a share s of the patches replaced by noise, the loss on all pixels. It is not the
  production recipe; the summary gives it FID 1.87, conditions unstated (single-level denoising at s = 0.8 is 1.65 in
  the s sweep below).
- 160 epochs of ImageNet (200,160 updates at batch 1024), Muon at lr 1e-3 constant, no warmup, weight decay 0, EMA
  0.9999; pixel space at 256 px, patch 16; OST-B is d768×16, OST-L d1024×32.
- Nothing in stage 1 is cheaper per step than flow training (the perceptual nets add cost); the "10×" is total compute
  to a one-step FID against Drifting. No experiment starts multi-step diffusion from a primed network.
- s = 0.8 is tuned for one-step generation, whose input is pure noise: close to it, with signal left to reconstruct.

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

## Masked priming (Daniel, 2026-11-07)

- No pinned t: the training t stays our logit-normal (sigma 1, centred on 0.5).
- Only part of the tokens are denoised, so the network must complete the rest from context, through attention.
- The masked share falls smoothly to 0 (from about 90%), ending in ordinary flow training; the source's objective is
  not taken as given, the aim is our own.

As built (`objective.token_t`, Config `prime_steps`, `spread_start`, `mask_start`): every 8×8 latent token is noised at
its own t. Its logit is the sample's plus `spread` times a normal draw, and with probability `mask_ratio` it is 1 instead
(pure ε, so under T = x_t − v its target is exactly the clean latent). `spread` and `mask_ratio` fall linearly from
their start values to 0 over `prime_steps` steps of the run (and restart with a run that resumes from `init_from`),
ending in ordinary flow training. The network is told the sample's t. There is no mask indicator: the network tells
masked tokens by their statistics. The loss covers all tokens. While priming is on, the train loss, R² and loss by t
mix tokens at other t than the sample's (R² assumes every token at the sample's t), so the train curves are comparable
to the reference only once priming has ended; the eval score is never primed.

## Ideas ledger (Daniel's answers, 2026-11-07)

- Per-token t: each token its own t around the sample's, annealed to one t per image, either by narrowing the spread or
  by shrinking the share of tokens that draw their own t. More signal per step, still leaning on attention. Wanted.
- The masked modes, pairing what the two groups of tokens see: (A) one group at the sampled t, the other pure noise,
  run (`log/2026-11-07-masked-priming.md`); (B) one group clean, the other pure noise, MAE-like; (C) one group clean, the other at the
  sampled t, denoising with clean context; (D) both at sampled t, the masked ones strictly noisier, which contains A.
  A clean token has no usable target under T = x_t − v (at t = 0 the x̂ is x_t whatever the output, and T = 2x − ε
  needs the unseen ε), so B and C take the loss on the other tokens only.
- Random scattered masks let a conv block fill a token from its visible neighbours; contiguous spans push the work to
  the attention blocks.
- Representation alignment, as Lodestone did with DINO for Chroma; EmbeddingGemma 2's vision tower is on disk. It is
  a Gemma 4 vision ViT: 16 layers, width 768, 16 px patches with axial RoPE at any resolution, pooled 3×3 into soft
  tokens for the language model. Its 16 px patch is our token (8 latent px of Primal 2×2), so the per-patch features
  before pooling land one 768-d vector per token on our grid. REPA aligns a trainable MLP projection of an early
  block's tokens to the frozen encoder's patch features of the clean image (cosine, weight 0.5); the best depth in
  REPA was block 8 of 28 in SiT-XL, about a third, which at d768x12 is the output of the first attention block.
  Unbuilt; features at 128 px are 98 KB per row if stored, or computed online from decoded latents.
- Caption alignment: the vision tower's embedding of the generated image against the pooled caption, by cosine;
  first as an eval of prompt following, whether its image and text embeddings share a space being unchecked. Daniel's
  reading: prompt following was poor for want of cross attention, and now waits on a long run on good captions.
- A 64 px first stage needs latents encoded again; noted, not planned: in Daniel's samples 128 px learns the mood within
  500 steps.

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
- Masking or fixed-level noise for stage 1. The first test (`log/2026-11-07-priming-read.md`, which lists how it
  differs from the code) is neither the masked reconstruction directed above nor a faithful stage 1. Masked
  priming in mode A, 90% falling to 0 over 3000 and over 6000 of 6000 steps, loses to the trunk from scratch by 0.009
  and 0.017 (`log/2026-11-07-masked-priming.md`). Open: which of per-token t, C and D runs next.
- The stage-2 feature space.
- Read the paper itself (the ablations are known only through a summary).
