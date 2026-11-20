# A caption refiner of two blocks

2026-11-20. Closes the question of a small text refiner over the token states (PLAN J6).

Decided: two refiner blocks over the caption's token states alone, at 512, before any attention block reads them:
self-attention (8 heads of 64, QK norm, no RoPE, padding masked) then SwiGLU, each with a branch norm and a zero
per-channel gate; no t, no image tokens. Architecture a4 (SPEC §3, §5).

Reasons:
- Measured (2026-11-20, 6000 steps, seed 41; log 2026-11-20-joint-attention-and-caption-refiner): held-out 0.0022
  ahead at 6000 and ahead at every eval from 2000; score level (0.0006 behind, within the spread); nulling the caption
  costs half as much again (+0.0039 against +0.0026) and twice at t 0.84. 6.2 % lower throughput.
- The *Hypothesis* that EmbeddingGemma 2's final states need no refiner, unlike the decoder LLMs' merged layers of
  Lumina 2 and Z-Image, is not borne out: the refiner leads on the set with the longer captions, not on Lunara (one
  seed).
- The second block reads the first block's output (its residual), not the encoder's states again.

Parked: whether one block is enough. A one-block arm against two would likely differ by less than one seed resolves.
