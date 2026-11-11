# Token signal and branch roles (2026-11-02)

Moved verbatim from `FACTS.md` on 2026-11-11. These reads are of the model before the 2026-11-06 architecture merge (width 512 at first, a final linear, no branch norm, no pin); the numbers are the record of that model, not of the current one.

## Token signal through the blocks (2026-11-02)

Source: scratch `heldout/token_signal.py` (re-implements `Block.forward`, checked equal to it), 256 rows of shard 12 with their captions (held out of the 0–10 runs, but trained on once in a sweep run; the sweep arms share them), fixed noise (seed 1234), at t = 0.05, 0.25, …, 0.95; on `128_sweep_lr2e-3` steps 1k, 2k, 4k, 8k, 15042 and `128_sweep_lr4e-3` step 15042. Per image over its 64 tokens: mean off-diagonal cosine, effective rank (exp entropy of the Gram's eigenvalues; 63 at most after centering), the share of the stream's energy in the token mean (`common`), then means over images.

| at the output of blocks 4–10 | lr 2e-3, 1k | lr 2e-3, 15042 | lr 4e-3, 15042 |
|---|---|---|---|
| `common`, t 0.05 / 0.45 / 0.95 | 0.44–0.57 / 0.39–0.50 / 0.55–0.85 | 0.90–0.94 / 0.81–0.84 / 0.92–0.95 | 0.94–0.95 / 0.88–0.89 / 0.96–0.97 |
| effective rank, raw | 9–15 / 9–18 / 1.4–10 | 1.5–1.8 / 2.7–3.2 / 1.4–1.7 | 1.3–1.4 / 2.0–2.2 / 1.2–1.3 |
| effective rank, centered | 39–40 / 24–39 / 3–40 | 26–30 / 35–37 / 24–33 | 21–23 / 31–34 / 18–21 |

- The embedding's `common` is 0.02–0.12. Each image's stream gains a vector shared by all its tokens, which grows with training and is larger at lr 4e-3; the token-specific part keeps a centered effective rank of 18–40, so it is a common mode, not a rank collapse.
- Attention writes it: blocks 0–1 raise the raw token cosine by +0.24 / +0.24 (lr 2e-3, t 0.05) up to +0.78 / +0.12 (lr 4e-3, t 0.95), later blocks by under +0.1. The MLP branch changes it by −0.04 to +0.12, mostly within ±0.02.
- SwiGLU decorrelates: the token cosine of the hidden `silu(W_g y) * W_u y` is below that of its input `y` in every block but the last (e.g. 0.67 → 0.61 down to 0.27 → 0.13), and its effective rank higher. The last block at t 0.95 is near rank one on both sides (input and hidden cosine 0.97–1.00) and its output RMS is 39–134 against 2–16 before it.
- Gate RMS (adaLN output) 0.14–0.35 at t ≤ 0.45, attention 0.44–0.75 and MLP 0.25–0.57 at t 0.95.

## What each branch does (2026-11-02)

Source: scratch `heldout/branch_role.py` (re-implements the forward, checked against `DiT.forward`), the same 256 rows of shard 12 and noise as Token signal, `128_sweep_lr2e-3` steps 1k, 4k, 15042. Per block: the loss rise when its attention or MLP branch is dropped, the branch's write RMS over the stream RMS, and |grad| / |W| per matrix under plain autograd (no Molegrad). t 0.05 / 0.45.

| | step 1k | step 15042 |
|---|---|---|
| drop attention, block 0 | +0.067 / +0.127 | +0.167 / +0.536 |
| drop attention, blocks 2–10 | +0.011–0.025 / +0.006–0.017 | +0.010–0.101 / +0.021–0.079 |
| drop MLP, block 0 | +0.003 / +0.021 | +0.029 / +0.121 |
| drop MLP, blocks 2–10 | +0.003–0.014 / +0.005–0.010 | +0.003–0.026 / +0.017–0.040 |
| drop MLP, block 11 | +0.004 / +0.025 | +0.250 / +0.323 (+0.664 at t 0.95) |
| MLP write / stream, blocks 2–10 | 0.04–0.10 / 0.12–0.41 | 0.02–0.10 / 0.06–0.16 |

- The middle MLPs cost about as much to drop as the middle attention branches, and their cost grew from step 1k to 15042. At t 0.95 they cost nothing (within ±0.003); there block 0's attention and block 11's MLP carry the prediction, the latter writing 3.4–5.3× the stream RMS.
- |grad| / |W| of `w_up` and `w_down` is the same order as `qkv`'s (1e-4 to 1e-3) in every block, with no decay toward the input. Usui's fixed step does not scale with the gradient's size anyway.
- Whole blocks skipped together (scratch `heldout/drop_many.py`, same rows), loss at t 0.05 / 0.45 / 0.95, step 1k → 15042. None skipped: 1.165 / 1.739 / 1.185 → 1.122 / 1.462 / 1.172. Blocks 2–10: 2.109 / 3.039 / 1.387 → 5.125 / 7.511 / 1.281. Blocks 4–8: 1.613 / 1.968 / 1.215 → 2.503 / 1.850 / 1.184. Blocks 0 and 11: 1.216 / 1.958 / 1.223 → 1.373 / 2.853 / 5.127. One middle block alone costs little; the nine together carry most of the prediction, more so at 15042 than at 1k.
