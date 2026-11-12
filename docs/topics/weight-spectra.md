# Weight spectra under Usui

The question and its reads, moved here whole from `PLAN.md` and `FACTS.md` on 2026-11-11 so the main ledgers stay short. PLAN keeps the question's heading and points here.

## The question (PLAN G9)

**G9. Does Usui hold back the rank the weights grow into, and can it be helped?**

The spectra (below): the weights start full rank and concentrate; at the end of
`swpin_5ep` the MLP matrices put 90 % of their energy in about a sixth of their directions, against about 0.7 in the
pretrained LFM3 and EmbeddingGemma 2 MLPs; Usui's frame holds about the whole per-step gradient.

The gradient is low rank, so even AdamW has trouble using the full space; the question is whether Usui harms full-rank
evolution and whether it can be helped (2026-11-05). The pin is just a form of weight decay, which everyone uses to
prevent explosive weights. Backward can only put changes where the activations sit, and they tend to cluster.
Zero-order would help here but is inefficient. A higher-rank Usui always helps, with diminishing returns: gradient-wise
the frame is already the optimal subspace, but even the tail contributes, being a power law.

Later the same day: do not trust the capture; it is just probes, comparable but not a true fixpoint. The concentration of the
spectrum could well be age, and the effect of age is at least more than 10%. Dropping tails did harm the initial final-layer bottleneck, but on
most matrices it makes no difference. The lean is toward the original SubTrack paper, though how they argued that the
subspace is good enough is unknown; this variant is arguably better, since tracking is proven better and the polar plus
row norm is a principled higher rank. That the gradient is low rank is known, with no need to probe. Rank 0.5 will help
the spectrum and the loss a little, but the point is that a small space is all you need.

The question was whether applying the polar map to the weights would surely break them, and whether a Hadamard rotation
could keep them intact while maximising rank. Not as stated: an orthogonal rotation on either side (W → HW, W → WH) leaves every
singular value unchanged, so the rank and spectrum are invariant; any change to the spectrum changes the function.
What a Hadamard rotation does change is the basis: it spreads energy evenly over coordinates (the incoherence QuaRot and
SpinQuant use), which would matter only to the basis-dependent parts of the optimizer, such as a row norm.

Open: whether the concentration is Usui's (the rank-0.1 frame drops the gradient's tail every step), the training length
(theirs is far longer; the 3-epoch continuation shows whether it keeps falling), or this task. Not wanted: the rank 0.5
arm and the dense-gradient probe. Remaining candidates, none agreed:
- A full-rank optimizer arm (AdamW on every matrix) of `swpin_3k`'s length: the reference for "even AdamW". Needs code.
- The same spectra on the continuation's checkpoints: whether the effective rank still falls with age.
- Ways to help, if Usui is the cause: a small random component in the frame each refresh to reach the tail (zero-order
  is the extreme of this).

## Measurements

### Spectra of the pinned weights (G9, 2026-11-05)

Source: scratch `spectra.py` on the CPU, `/mnt/mars/ai/tiny-dit/probes/spectra_swpin_5ep.json`: every Usui matrix of `128_d1024x16_swpin_5ep` at steps 7500, 15000, 22500, 30000, 37605 against its init, rebuilt from seed 41 (every pinned norm reads 1.000 of the rebuilt init's, which checks the rebuild). Effective rank is exp of the entropy of the normalized σ² spectrum, as a fraction of the shorter side k; Usui's frame rank is 0.1 k (102 for k = 1024).

| family (mean over blocks) | eff rank at init | 7500 | 37605 | cos(W, W₀) 7500 / 37605 | interval update ‖W_k − W_{k−1}‖/‖W‖, 7500→15000 / 30000→37605 | interval update r90 / k, same |
|---|---|---|---|---|---|---|
| `qkv` | 0.846 | 0.193 | 0.170 | 0.154 / 0.024 | 0.92 / 0.36 | 0.166 / 0.149 |
| `gate` | 0.606 | 0.100 | 0.082 | 0.148 / 0.021 | 0.80 / 0.29 | 0.102 / 0.089 |
| `out` | 0.606 | 0.106 | 0.093 | 0.165 / 0.029 | 0.75 / 0.25 | 0.101 / 0.089 |
| `w_gate` | 0.882 | 0.212 | 0.195 | 0.147 / 0.032 | 0.85 / 0.32 | 0.191 / 0.174 |
| `w_up` | 0.882 | 0.213 | 0.199 | 0.138 / 0.023 | 0.87 / 0.31 | 0.189 / 0.175 |
| `w_down` | 0.882 | 0.210 | 0.186 | 0.135 / 0.026 | 0.87 / 0.30 | 0.184 / 0.169 |
| `ada.1` | 0.882 | 0.361 | 0.264 | 0.234 / 0.029 | 1.02 / 0.37 | 0.395 / 0.251 |
| `time_mlp.0` | 0.882 | 0.345 | 0.108 | 0.471 / 0.211 | 0.88 / 0.33 | 0.223 / 0.184 |
| `time_mlp.2` | 0.606 | 0.286 | 0.239 | 0.288 / 0.066 | 1.05 / 0.37 | 0.284 / 0.126 |
| `text_proj` | 0.687 | 0.223 | 0.151 | 0.407 / 0.170 | 0.80 / 0.32 | 0.111 / 0.108 |
| `patch_embed` | 0.946 | 0.944 | 0.937 | 0.977 / 0.964 | 0.11 / 0.06 | 0.773 / 0.781 |

- The weights do not grow from a subspace to full rank; they start full rank (random init) and concentrate. By step 7500 the effective rank has fallen to about a quarter of the init's, and it barely moves after 15000. At 37605, block 8: 90 % of σ² sits in 148 of 1024 directions for `qkv`, 72 for `gate`, 80 for `out`, 170 for `w_up`, 160 for `w_down`; 277 for `ada.1`. The largest σ² stays 1–5 % of the total (`time_mlp.0` 42 %: one dominant direction from the scalar t).
- The init is forgotten, not kept: cos(W, W₀) ends at 0.02–0.03 for every block matrix, and the overlap of W's top-r left singular subspace with W₀'s is at chance (r over the side length) from step 7500 on. `patch_embed` is the exception (cos 0.96, mostly its identity rows).
- The late update (30000 → 37605) lies in W's own top-102 singular subspace on the stream side, the 1024-wide side the frame is on: right side for `qkv` 0.52, `gate` 0.89, `w_gate` 0.45, `w_up` 0.49, `text_proj` 0.77; left side for `out` 0.87, `w_down` 0.48 (chance 0.10). On the other side it is near chance (0.05–0.15 against 0.025–0.10). Block 8 except `out` (block 2) and `w_down` (block 14); `ada.1` 0.33 right.
- Uniform in depth: the effective ranks of `qkv`, `w_gate`, `w_down` differ by under 0.03 across blocks 0–15; block 15's MLP is lowest (0.175 / 0.166).
- Not separated: how much of the concentration is Usui's rank-0.1 frame, how much the pin's washing out of the init (each step rescales the init's share down), and how much the task. The unpinned `5ep` checkpoints and a different rank fraction would separate them; neither is measured.
