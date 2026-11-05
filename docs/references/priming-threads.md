# Priming Models: Daniel's writeup of two X threads by @Jaeyeon_Kim_0

Kept verbatim as Daniel pasted it on 2026-11-05 (a writeup of the threads of 15 Oct and 4 Nov 2026, not the threads themselves). Nothing in it is verified against the paper or repo here.

**Writeup for Claude: Full summary of the two X threads by @Jaeyeon_Kim_0 on Priming Models**

### Thread 1 — Original announcement (15 Oct 2026)
**Main post + long self-reply thread**

Jaeyeon (Jay) Kim (PhD student in CS at Harvard, born in Korea) and collaborators introduce **Priming Models**, a new paradigm for training high-quality *one-step* generative models from scratch.

**Core claim / slogan**  
Reconstruction pretraining → Distributional post-training.

**Key selling points**
- No diffusion teacher, no flow map, no classifier-free guidance (CFG), no Jacobian-vector products (JVP).
- State-of-the-art one-step results with far less compute:
  - ImageNet 256×256: **FID 1.19** (Priming-L). ~6× less compute than Drifting-L (FID 1.61), ~12× less than Drifting-B.
  - Also beats Improved/Pixel MeanFlow, Shortcut, W-Flow, Drifting, and even multi-step JiT (200 steps + CFG) on FID and on FDr5 (average normalized Fréchet distance across five feature spaces *not* used for training).
- Scales to text-to-image (GenEval 0.82 one-step; two-step version reaches 0.844) and world modeling (RT-1 / Bridge-V2, outperforming DriftWorld with less compute).

**Method (two stages)**
1. **Stage 1 – Reconstruction pretraining**  
   Train a network to reconstruct clean images from corrupted inputs. Simplest version: denoising at a *single fixed noise level*. No time embedding. The same network later receives pure Gaussian noise at inference/generation time.
2. **Stage 2 – Distributional post-training**  
   Refine the outputs by matching the distribution of generated samples to real samples in a *frozen* feature space (e.g., Sinkhorn divergence / optimal transport).  

The pretrained network still produces poor samples from pure noise, yet only ~1,250 post-training updates already bring Priming-B to FID 2.26. Full training is dramatically more efficient than prior from-scratch distributional methods (GANs historically, more recent Drifting / W-Flow, etc.).

**Flexibility shown**
- Pretraining can be pure denoising, masking-based reconstruction, or reconstruction of a pretrained RAE followed by frozen decoding.
- Post-training can use Sinkhorn/OT, simple pairwise distances, Laplacian kernel, or even FD-loss.
- Evidence is given that reconstruction pretraining improves the network’s Jacobian alignment with the useful correction directions provided by the distributional loss.

**Limitations / open questions noted**
- Still relies on pretrained feature encoders for the distributional stage (shared with Drifting). Reducing this dependence is left open.
- Related prior work that fine-tunes multi-step diffusion models into one-step generators is acknowledged; Priming asks how *simple* the initial preparation can be and turns it into a training-efficiency question.

**Releases**
- Paper: “Are We Overthinking One-Step Generation?” (arXiv:2610.20331)
- Blog post
- Code + final models + intermediate reconstruction checkpoints: https://github.com/JaeyeonKim01/priming  
  (Intermediate checkpoints let others try new distributional objectives without repeating Stage 1.)

**Authors**  
Equal contribution: Woobin Park (undergraduate at Seoul National University, applying to PhD programs) and Jaeyeon Kim. Advisors: Yilun Du, Sitan Chen, Bohyung Han.

The thread ends with tags of many people in the generative modeling community.

---

### Thread 2 — Follow-up with new experiments & release updates (4 Nov 2026)
**Quotes the original announcement and answers community questions**

**TL;DR of the follow-up**  
New results on diffusion-policy / action generation, ablation on number of real samples in OT, code & T2I checkpoint releases, and discussion of Stage-1 design choices (including a community blog post).

**[1] Can priming help with action generation / diffusion policies?**  
(Thanks to @alfie14x and @servo97)

- Small-scale ToolHang experiments with a 20 M-parameter U-Net, state-based setup adapted from “Much Ado About Noising”.
- Compared Priming (one-step) vs. flow matching (NFE=9) and MIP (NFE=2).
- One-step Priming policy approaches the 9-NFE flow-matching baseline over much of training and outperforms MIP.
- Distributional loss operates *directly* on normalized action chunks → no pretrained perceptual feature encoder needed (Euclidean distances + pairwise term).
- Notes that ESP work shows similar distributional losses can work well when training an action head from scratch on top of a frozen VLM backbone (LIBERO). ToolHang results are preliminary and do not yet prove scaling, but are encouraging.

**[2] Effect of OT batch size / number of real reference samples**  
(Thanks to @unixpickle)

- ImageNet ablation: varied number of real reference images per class per OT update from 1 → 64 (kept live-generated samples, stop-gradient references, and # classes fixed; global live batch = 1024).
- Priming works well even with very few (or effectively one) real reference samples — consistent with the fact that text-to-image and world-modeling settings are constrained to essentially one reference and still succeed.

**[3] Code and checkpoint updates**
- Text-to-image and world-modeling code now public in the same repository.
- Two text-to-image checkpoints released on Hugging Face.
- Repo: https://github.com/JaeyeonKim01/priming

**[4] Stage-1 design choices and open questions**  
References a new independent blog post by Moritz Piening (@moritz_piening) that explores Priming on CIFAR-10 / MNIST / Fashion-MNIST (https://moepien.github.io/blog/better-priming/).  
TL;DR of that blog: Stage-1 initialization is sensitive; optimal transport helps.

- Kim’s own ImageNet ablation on fixed noise level *s* used for Stage-1 pretraining of Priming-B (80 epochs + 12 500 OT updates):  
  FID results: s=0.5 → 2.46; s=0.7 → 1.76; s=0.8 → 1.65; s=0.9 → 1.97; s=1.0 → 65.08.
- Intuition: useful noise level balances staying close to pure Gaussian (used at generation time) while still retaining enough information about the clean image. Choosing the optimal noise level across settings remains an open empirical question.
- Clarification: different initializations can produce different samples for different noise inputs; scale and training efficiency effects may differ.

A short positive reply from @servo97 appears under the thread (“All of these results look fantastic! …”).

---

**Overall takeaway across both threads**  
Priming Models argue that a very simple reconstruction pretraining stage is sufficient to make subsequent pure distributional (OT-style) training of one-step generators both stable and highly compute-efficient. The work deliberately avoids the usual diffusion/flow machinery and challenges whether the field has been over-engineering one-step generation. Strong empirical results on ImageNet, T2I, world modeling, and early robotics/policy experiments, plus open-source code and checkpoints, are provided.

---
