# References

What each source establishes for this project.

## Objective

- **Ostris, "x0 velocity" thread on X, 2026-10-22.** Kept verbatim in `references/ostris-x0-velocity-thread.md`.
  Target `x_t - v`, conversion by one subtraction, constant residual magnitude, qualitative 1-day single-GPU results
  at 64× compression on MOSS-Image. No quantitative metrics.
- **Li & He, "Back to Basics: Let Denoising Generative Models Denoise" (JiT), arXiv 2511.13720.**
  Pure x0 prediction with large patches; the manifold/bottleneck argument for why the target matters with fat tokens.
- **Karras et al., EDM / EDM2.** Preconditioning (`c_skip`, `c_out`); EMA length as a major quality knob (EDM2).
  arXiv 2312.02696 ("Analyzing and Improving the Training Dynamics of Diffusion Models") is taken to be EDM2: magnitude-preserving
  layers; the pointer for the weight pin (PLAN G8), not read beyond the abstract.
- **Esser et al., SD3.** Rectified flow at scale, logit-normal timestep sampling, resolution-dependent shift.
- **Kim et al., "Are We Overthinking One-Step Generation?" (Priming), arXiv 2610.20331.** Repo
  https://github.com/JaeyeonKim01/priming, blog https://moepien.github.io/blog/better-priming/. Stage 1: reconstruction
  pretraining at one fixed noise level; stage 2: distributional post-training (Sinkhorn / optimal transport) in a frozen
  feature space, for one-step generation. Stage-1 noise-level ablation as the writeup quotes it (FID):
  s 0.5 → 2.46, 0.7 → 1.76, 0.8 → 1.65, 0.9 → 1.97, 1.0 → 65.08. Known here only through the writeup, kept verbatim in
  `references/priming-threads.md`; paper, repo and numbers unverified (PLAN I1).

## Components

- **Primal VAE** — https://huggingface.co/ostris/primal_vae
  2× spatial to 1 or 2 channels, the DiT does the patching; ~80M params; moment and Gaussian order-statistic losses
  keep the latent unit-normal per image and channel; trained with latent flip/rotation equivariance (measured: the decoder is, the encoder is not; FACTS) and noise-perturbed
  decode (σ ~ U(0, 0.5)); `patchify(·, compression)` is `pixel_unshuffle(compression / 2)`; sides multiple of 16.
- **EmbeddingGemma 2** — https://huggingface.co/google/embeddinggemma-2 (released 2026-10-06, Apache 2.0)
  Text path: 24 layers, model dim 512, 130M transformer + 140M embedder, mean pooling, 512→768 projection,
  8K context, task prefixes, bf16 or fp32 only (fp16 yields NaN). Attention direction not stated on the card.
- **UsuiTrack** — the project's optimizer (SubTrack++ + FlashOptim), local.

## Optimizer

- **bubbleforce: Predicting the Edge-of-Stability for Muon**, Cheng and Zheng, 2026 (the page gives no date).
  https://jeffreycider.github.io/posts/bubbleforce/ · code https://github.com/jeffreycider/bubbleforce.
  Extends the edge of stability to non-adaptive optimizers with momentum (Muon): `lr · pathspring = bubbleforce`,
  bubbleforce from the roots of the momentum kernel's characteristic polynomial. Tested on a 12-layer, 768-wide nanoGPT
  with Muon; the equilibrium claim for real training is a conjecture, the proofs assume a frozen Jacobian and a slowly
  moving Hessian; Adam outside its scope. PLAN C4.

## Architecture references

- **MOSS-Image Technical Report**, arXiv 2610.23374. 5B single-stream DiT, Qwen3.5-9B text encoder, FLUX.2 VAE,
  2×2 latent grouping → 16-px stride.
- **Krea 2.** Single-stream 12B DiT; GQA with gated sigmoid attention, QK-norm, SwiGLU 4×, 3D axial RoPE;
  progressive 256/512/1024 px.
- **Qwen-Image-2.0 Technical Report**, arXiv 2605.10730. MMDiT, MSRoPE, RMSNorm QK-norm, bias-free modulation, SwiGLU.
- **PixArt-α.** adaLN-single.
- **CPVT** (conditional positional encoding) and **SANA** (Mix-FFN with depthwise conv, linear attention).

## Data

- **text-to-image-2M** — https://huggingface.co/datasets/jackyhate/text-to-image-2M
  ~2M pairs as 512×512 WebDataset shards; mix of Flux-dev renders of LLaVA-pretrain prompts, DALL·E 3 images,
  ~700K real LLaVA-NeXT images recaptioned with Qwen2-VL; plus a 10K 1024 px set. ~423 GB.
- **Fine-T2I** — https://huggingface.co/datasets/ma-xu/fine-t2i
  About 6.15M synthetic pairs (Z-Image, FLUX2; 10 task combinations,
  32 prompt categories, 11 styles) plus 168k curated real photos (Pexels, Pixabay, Unsplash); ~2.3 TB WebDataset
  (jpg, txt with an original and an enhanced prompt, json), mostly above 1K, some subsets at random aspect ratios;
  Apache-2.0. Claims semantic dedup, safety and aesthetic filters and a VLM quality audit removing over 95 % of
  candidates (the authors' claim). A candidate for the second image dataset (PLAN G1).
- **MONET** — https://huggingface.co/datasets/jasperai/monet
  103.8M pairs distilled from 2.9B, 70.8 TB (Jasper Research, Apache-2.0). Mostly real web images (LAION-2B-en 46.3M,
  COYO 18.5M, Common-Catalog-CC-BY 11.0M, Megalith-10M 8.0M, CC12M 6.2M) plus 13.8M synthetic (Z-Image, FLUX.1-schnell,
  FLUX.2-klein-4B); alt-text plus up to four synthetic captions (Florence-2 short, ShareGPT4V, InternVL3, Gemini long);
  images below 512 px and LAION aesthetic below 5 dropped, pHash/SSCD dedup, NSFW and watermark filters. Parquet
  configs carry 384 px thumbnails, metadata, embeddings and VAE latents; WebDataset carries full resolution. Judged (with PD12M): "rather world knowledge images, not so much t2i".
- **PD12M** (not chosen) — https://huggingface.co/datasets/Spawning/PD12M
  12.4M public-domain/CC0 pairs, recaptioned, S3-hosted. Alternative if real-image aesthetics or licensing matter.
  The second set of the original plan, left out of the notes; world knowledge images rather than
  text-to-image.
