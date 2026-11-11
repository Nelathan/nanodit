# Priming on the Primal latent

Topic doc for PLAN I1, opened 2026-11-05 at Daniel's request ("Create the doc now, plan has just a reference"). It holds
this line of work's plan, design and reads; measured numbers still go to FACTS, closed decisions to `archive/`. The code
gets its own branch when it starts.

## Status (Daniel, 2026-11-11)

A side quest, kept out of the main ledgers. Priming will probably not be used at all; if anything is, it is REPA, with its targets precomputed rather than computed online (the online tower costs 8.4× per aligned step, below). The switches for masked priming, per-token t and online REPA are still in the code.

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

As built (`objective.token_t`, Config `prime_steps`, `mask_start`): every 8×8 latent token is noised at the sample's t,
or with probability `mask_ratio` at 1 instead (pure ε, so under T = x_t − v its target is exactly the clean latent).
`mask_ratio` falls linearly from `mask_start` to 0 over `prime_steps` steps of the run (and restart with a run that resumes from `init_from`),
ending in ordinary flow training. The network is told the sample's t. There is no mask indicator: the network tells
masked tokens by their statistics. The loss covers all tokens. While priming is on, the train loss, R² and loss by t
mix tokens at other t than the sample's (R² assumes every token at the sample's t), so the train curves are comparable
to the reference only once priming has ended; the eval score is never primed.

## Ideas ledger (Daniel's answers, 2026-11-07)

- Per-token t: each token its own t around the sample's, annealed to one t per image, either by narrowing the spread or
  by shrinking the share of tokens that draw their own t. More signal per step, still leaning on attention. Run by
  narrowing the spread (`log/2026-11-07-per-token-t-priming.md`): no real benefit, dropped from the code (Daniel).
- The masked modes, pairing what the two groups of tokens see: (A) one group at the sampled t, the other pure noise,
  run (`log/2026-11-07-masked-priming.md`); (B) one group clean, the other pure noise, MAE-like; (C) one group clean, the other at the
  sampled t, denoising with clean context; (D) both at sampled t, the masked ones strictly noisier, which contains A.
  A clean token has no usable target under T = x_t − v (at t = 0 the x̂ is x_t whatever the output, and T = 2x − ε
  needs the unseen ε), so B and C take the loss on the other tokens only.
- Random scattered masks let a conv block fill a token from its visible neighbours; contiguous spans push the work to
  the attention blocks. Daniel's reasoning, unmeasured: a conv is most held back by large masked blocks, not by a masked row or column, which still
  leaves 6 of a token's 8 neighbours. Once the masked share is low, the attention blocks reach clean tokens anywhere
  and learn normally, so the share may want a saturating curve rather than a linear ramp.
- Representation alignment, as Lodestone did with DINO for Chroma; EmbeddingGemma 2's vision tower is on disk. It is
  a Gemma 4 vision ViT: 16 layers, width 768, 16 px patches with axial RoPE at any resolution, pooled 3×3 into soft
  tokens for the language model. Its 16 px patch is our token (8 latent px of Primal 2×2), so the per-patch features
  before pooling land one 768-d vector per token on our grid. REPA aligns a trainable MLP projection of an early
  block's tokens to the frozen encoder's patch features of the clean image (cosine, weight 0.5); the best depth in
  REPA was block 8 of 28 in SiT-XL, about a third, which at d768x12 is the output of the first attention block.
  To be built with the features computed online from decoded latents, not stored. The processor resizes every image to
  a budget of 280 soft tokens (about 2,500 patches, near 800 px), so an 8×8 patch grid at 128 px is far below what the
  tower sees in use. Its smallest budget, 70, takes a square image to 384 px, whose 24×24 patches pool 3×3 onto our
  8×8 tokens (280 gives 768 px and our 16×16 at 256 px); at 384 px it matches decoded images to their captions as well
  as at 768 (FACTS), so 384 px is the REPA input at 128 px. That probe decoded 256 px latents and scored the final
  embedding. On 128 px decodes upsampled 3×, the pooled per-token features follow their token's content under a
  horizontal flip (FACTS, `scripts/probe_repa_target.py`).
  As built (`align.VisionTarget`, `DiT(repa_block=…)`, Config `repa_block`, `repa_weight`): the clean latent is
  decoded by Primal, upsampled bicubic 3×, cut into the processor's 16 px patches and positions, and passed through
  the frozen tower (bf16), whose own 3×3 pooling hands back one 768-d feature per token in our row-major order. The
  output of block `repa_block` goes through REPA's projector (two hidden layers of 2048, SiLU, to 768); the loss adds
  `repa_weight`·(1 − mean cosine) to the flow loss for the first `repa_steps` steps of the run, then none (Daniel: a
  fixed span, so the removal shows). The train log carries the flow loss alone and the cosine as `repa_cosine`. A
  REPA checkpoint carries the projector, so it loads only into a config with `repa_block`. First arm
  `configs/128_d768x12_repa_6k.toml`: block 3 (the first attention block), weight 0.5, on for 3000 of 6000 steps.
  It ends 0.0023 ahead of scratch (0.0013 at 256 px) after trailing while on, at 8.4× the time per aligned step
  (`log/2026-11-08-repa-first-arm.md`); longer use needs the features stored with the latents.
- Caption alignment: the image embedding of the generated image against the pooled caption, by cosine, as an eval of
  prompt following. EmbeddingGemma 2 maps text, images, video and audio into one 768-d space; that is its core feature
  (model card). Against the real image's cosine to the same caption, it also marks captions that miss their image.
  Captions without a prefix retrieve their image best of the model's prefixes; the stored ones carry the document prefix (FACTS, PLAN B7).
  Daniel's reading: prompt following was poor for want of cross attention, and now waits on a long run on good
  captions.
- A 64 px first stage needs latents encoded again; noted, not planned: in Daniel's samples 128 px learns the mood within
  500 steps.

## How it would map onto nanodit (my reading, untested)

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
  `vision_tower` and `embed_vision`; our text path loads it with `vision_config=None`), a frozen image space shared
  with our captions' embeddings, at the cost of a VAE decode per sample.
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
  and 0.017 (`log/2026-11-07-masked-priming.md`). Per-token t, the spread falling from 0.5 or 1.0 to 0 over 3000 of
  6000 steps, ends level with scratch and 0.002 behind (`log/2026-11-07-per-token-t-priming.md`). REPA for 3000 of
  6000 steps ends 0.0023 ahead (`log/2026-11-08-repa-first-arm.md`). Open: which of
  C, D and spans runs next, and the shape of the ramp.
- The stage-2 feature space.
- Read the paper itself (the ablations are known only through a summary).

## Reads

Measured numbers of this topic, moved from `FACTS.md` on 2026-11-11; full tables in the `log/2026-11-07-*priming*` and `log/2026-11-08-repa-first-arm.md` notes.

- Single-level denoising before flow training (stage 1 = our objective at t = 0.8 only; not the source's stage 1, see `topics/priming.md`), at d768x12 for 6000 steps in total: 3000 + 3000 is 0.013 behind scratch, 1000 + 5000 is 0.004 behind (0.0045 at 256 px). Stage 1 plateaus by 2000 steps, is no better at its own t than scratch at 3000, and wrecks the low t (3.55 against 2.16 at init); recovery takes 50 steps, and the primed start's lead at equal normal-t steps fades from 0.010 to 0.002. Same throughput (877 against 881 img/s). `log/2026-11-07-priming-read.md`.
- Masked priming, mode A (a share of the tokens pure noise, falling from 90% to 0; the rest at the sampled t), d768x12, 6000 steps: the ramp over 3000 steps ends 0.009 behind scratch (0.010 at 256 px), its 6000 steps worth about 4430 of scratch; over 6000 steps 0.017 behind. Lost at low and middle t, level at the top: a pure-noise token trains only t = 1, which flow training learns fastest anyway. The trunk from scratch, 1.4771 / 1.3728, reproduces the branch GDN arm. `log/2026-11-07-masked-priming.md`.
- Per-token t priming (each token's t moved in logit by spread·N(0, 1) about the sample's, the spread falling to 0 over 3000 of 6000 steps), d768x12: spread 0.5 ends level with scratch (1.4770, 1.3728 at 256 px), spread 1.0 0.002 behind (0.003 at 256 px). It costs nothing at 0.5 and gains nothing by 6000 steps. `log/2026-11-07-per-token-t-priming.md`.
- EmbeddingGemma 2 image embeddings of 128 decoded eval images (256 px latents) against our stored pooled captions (encoded with the document prefix `title: none | text: `, as `precompute.py` writes them): the right caption ranks first among the 128 for 96.1% at the processor's 70-token budget (384 px square, 24×24 patches pooled 3×3 to 8×8), 93.8% at 140 (528 px), 94.5% at 280, the default (768 px); cosine to the own caption 0.797 against 0.669 to the others at 384 px. The retrieval query prefix on the captions gives the same 96.1%. CPU fp32, 1.37 img/s at 384 px, 0.24 at 768. The processor takes only budgets of 70, 140, 280, 560, 1120. `scripts/probe_vision.py`, 2026-11-07.
- REPA, block 3 (the first attention block) aligned to EmbeddingGemma 2's vision features at weight 0.5 for 3000 of 6000 steps, then off, d768x12: behind scratch while on (0.005 at 1000, 0.0014 at 3000), ahead after removal, 1.4748 against 1.4771 at 6000 (1.3715 against 1.3728 at 256 px). The gain is at 40–66 % noise (up to 0.006), a small loss at 74–84 %. The first priming arm to beat scratch. Online features cost 8.4× in img/s (89 against 745, not split by part; the decode, upsample and tower run per sample, the projector is small). `log/2026-11-08-repa-first-arm.md`.
- REPA's target (`align.VisionTarget`) on 32 eval latents at 128 px, decoded and upsampled to 384: under a horizontal flip of the latent, each token's feature has cosine 0.645 to its flipped-back counterpart, against 0.251 to the image's other tokens and 0.200 to the same token of other images; the counterpart is the best of the 64 for 40% of tokens (chance 1.6%). Feature norm about 20,400 (the tower's √768 scaling). CPU, `scripts/probe_repa_target.py`, 2026-11-07.
