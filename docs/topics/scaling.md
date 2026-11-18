# Scaling: model size, data, schedule and eval against a compute budget

Topic doc for the size and length of the long run (PLAN C1, C2, D8; archive 2026-11-18-long-run-size, 2026-11-18-no-ema-no-merging and
2026-11-18-heldout-set). Opened 2026-11-18 from W. Chai, "How
to Scale Text-to-image Models: a Handbook" (REFERENCES, Scaling), the first iso-FLOP study of text-to-image DiTs with
a recipe close to this one (a matrix optimizer, held-out loss, fixed noise). Measured numbers go to FACTS, closures to
`archive/`.

## What the handbook measures

Pixel-space DiT at 256 px on GPIC (100M permissive pairs), a ladder d9 to d20 (width = 128 × (image layers − 4), plus
2 to 3 text layers), Muon on the matrices and AdamW on the rest, batch 1024, 3.6e18 to 2.3e20 FLOPs, C = 6ND with N
the **image** parameters (no conditioning, adaLN or embeddings) and D the image tokens seen. The main metric is a
held-out loss: 3,072 validation images with their captions, 16 noise levels (the quantiles of the training t
distribution), one fixed noise per image and level. Its numbers, as stated there:

| Finding | Number | Bears on |
|---|---|---|
| Held-out loss tracks FD and FD-DINOv2 and is far less noisy: seed std 0.03 % against 2.0 % for FD; in 20 of 60 sweeps the gap beats twice the seed noise, against 1 for FD | 4 seeds of d9 | D5, the held-out set, the arm protocol |
| Compute-optimal split N* ∝ C^0.50, D* ∝ C^0.50: tokens per parameter (TPP) about 50, flat over 3.6e18 to 2.3e20 FLOPs | 38 runs, fit error 0.057 % | the long run's size |
| TPP depends on which noise levels the loss weighs: 40 to 72 over the weightings tried; the cleaner the level, the larger the optimal model | single-level fits | the long run's size, the score's t grid |
| Repetition: up to about 50 passes no measurable loss; d12 +0.18 % at 59 passes, +2.3 % at 206; larger models degrade sooner. FD keeps falling, so FD does not show it | d9, d12 | the epoch count |
| Shape at equal compute: best width ÷ image layers 28 to 34 (512 × 15, 640 × 23, 896 × 32); 768 × 44 (17) within 0.03 % at 670M; ratio 5 level with the ladder | 3 sizes | the shape, G4 |
| Learning rate: one Muon lr for all sizes per budget, falling as C^−0.5 (halved per 4× compute); the optimum is flat, ×0.5 to ×1 within 0.06 % | ×1/8 to ×4 | C2, the lr of a long run |
| Weight decay with Muon: lr × steps × decay = 5 is best for the loss; larger decay helps FD | d11, d12 | G8 (the norm ceiling stands in for decay here) |
| Batch at equal compute: smaller is better down to 512; doubling raises the loss up to 0.60 %; d12 512 → 1024 +0.005 %, 1024 → 2048 +0.10 %; a √2 or 2× lr does not recover 2048 | d9 to d16 | C1 |
| Cooldown: linear over the last 50 % beats 20 % by 0.22 to 0.24 %; 90 % and 1−√ no better | d11 | the anneal (SPEC §6) |
| Cooldown against EMA: cooldown 0.4 to 1.0 % lower held-out loss at every budget; EMA 7.5 % lower FD at 1.1e20. EMA in short runs still holds the init | equal compute | no EMA (archive) |
| Muon against AdamW: AdamW 0.7 to 2.1 % higher loss, 16 to 32 % higher FD (pixel space; a latent-space study cited there reports the opposite) | d9 to d12 | the optimizer choice |
| Text layers: 2 to 3 best; changing them moves the loss at most 0.10 % | d9 to d14 | J6 |
| Throughput: one caption-length bucket per step, fused residual + RMSNorm + adaLN and q/k norm + RoPE kernels | MFU 22.9 → 37.3 % | H1, H2 |
| Data (cited from i1, not measured there): a stronger captioner, long captions only, prompt rewriting at inference, equal sampling weight across datasets | | B, the merged set's mix |

Its t convention is the reverse of SPEC's (t = 1 clean); its training t is logit-normal(−0.8, 0.8), which in SPEC's
convention is a shift toward noise with σ 0.8, close to the σ 0.7 here plus a shift.

## What carries over, and what does not

- **Tokens.** Its token is a 16-px pixel patch; ours is a 16-px image stride through the Primal latent (SPEC §2), the
  same count per image. Whether TPP transfers across that change of what a token holds is a *Hypothesis*: the
  latent carries less to predict per token, which, by its own noise-level result, could move TPP either way.
- **N.** Its N excludes conditioning, adaLN and embeddings; `scripts/count_params.py` splits ours the same way, the
  cross attention counted as conditioning.
- **The lr.** Usui's lr is a relative step under a norm ceiling, not Muon's lr with decoupled decay, so no value
  transfers. The exponent may: a run 4× longer would want about half the lr (*Hypothesis*; the reference's 2e-3 was
  chosen on 3000- to 6000-step arms).
- **Batch.** Its smallest batch, 512, is twice ours. The direction (smaller wins at equal images until the step's
  noise dominates) held over 1.5M images at 128 px, at the full lr only, and 256 is kept for its speed (archive
  2026-11-18-batch-256).
- **The score's t.** Its held-out loss weighs the training distribution's quantiles; the score here weighs the shift-1
  logit-normal's (SPEC §8), whatever t the run trains on. Both are fixed grids, so both rank runs; they weigh the
  extremes differently, which by its result moves TPP.

## The budget, in this model's terms

Image parameters N from FACTS (Parameter counts). At TPP 50: d768 × 24 (191.3M) needs 9.6B tokens, d768 × 32
(255.1M) 12.8B, d1024 × 20 (284.1M) 14.2B, d1024 × 28 (397.6M) 19.9B.

`set_128` less its excluded rows is 8,236,215 rows: 527M image tokens per epoch at 128 px (64 per image), 2.11B at
256 px (256 per image). Eight epochs at 128 px and three at 256 px are 10.5B tokens, TPP 55 for d768 × 24.
Repetition stays far below the handbook's 50 passes.

At TPP 50 the compute is C = 6 · N · 50N = 300 N², so twice the compute buys √2 the parameters, not twice. A budget
twice that of d768 × 24 at TPP 50 is about 270M image parameters (d768 × 32 or d1024 × 20). d1024 × 28 (about twice
the parameters) at TPP 50 is four times the compute; trained in twice the time it sits at about TPP 23, under half of optimal
data. Where the iso-FLOP parabola is flat this costs little; how flat it is here is not measured.

The handbook's shape optimum (28 to 34) puts d768 × 24 (32) on it, d768 × 32 (24) on the deep side, between the
optimum and the 768 × 44 (17) it found within 0.03 % at 670M, and d1024 × 20 (51) on the wide side. The conv blocks here carry no attention, so a "layer" is cheaper than
the handbook's; the ratio is a guide, not a transfer.

Wall-clock is open: the 128 px rate is in log 2026-11-18-a1-ref-1ep; 256 px throughput is unmeasured (C1).

## Open here

- Scoring at 256 px, on Lunara and on the held-out rows (archive 2026-11-18-heldout-set), once the 256 px set exists.
- The seed noise of a 6000-step comparison (`a1-seed42-6k` against `a1-ref-1ep`): the handbook resolves 0.03 %; the
  bound used here, about 0.003 in caption loss (about 0.2 %), was never measured.
- Whether a longer run wants a lower lr (the C^−0.5 exponent), read by a 2× length pair once the long-run length is
  chosen.
