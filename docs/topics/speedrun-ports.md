# What ports from the modded-nanogpt speedrun

Topic doc for PLAN F5 and F6. Opened 2026-11-17 from `references/gemini-thread-adaln-and-architecture.md` §5 and its
optimizer section. **The speedrun's record history is not read here.** The list below is the thread's, plus items
recalled from memory and marked as such, each needing a check against
https://github.com/KellerJordan/modded-nanogpt (the README's record table, and the PRs behind the recent records).
Measured numbers go to FACTS, closures to `archive/`.

## The task (F5)

Read the record table, recent and old, and sort each record's change into one of four states:
- **In design:** this model already has it.
- **A question:** a PLAN item, existing or new.
- **Not applicable:** with the reason.
- **To read:** not yet sorted.

The thread's headline figures, such as GPT-2 in "about 30.7 s on 8×H100" and record numbers "#9–#11" and "#32", are
unverified.

## Ledger

| Change | Source | State here |
|---|---|---|
| Muon: an orthogonalised update for 2-D matrices; Polar Express; NorMuon's per-neuron normalisation | thread | **In design**, in Usui's form: the polar factor of the moment by Aurora row balance and quintic Newton–Schulz, as a trust step (SPEC §6). Changes belong in usuitrack |
| RMS-matched lr, `lr·0.2·√max(out, in)` | thread | **Not applicable**: Usui's trust step is relative to ‖W‖ (SPEC §6) |
| QK-norm | thread | **In design** (SPEC §3) |
| Zero-init output projections | thread | **In design**, in another form: the adaLN gate rows are zero, so every block is the identity at init (SPEC §5) |
| Sigmoid-gated attention output | thread | **In design** (Krea gate, SPEC §3) |
| Untied input and output | thread | **In design**: identity patch embed, slice readout, no final norm |
| U-Net long skips, gated | thread | **Question**: M2 (`topics/residual-lanes.md`) |
| Value embeddings: the token embedding mixed into the attention values | recalled | **Question**: M3 |
| A gated re-injection of an input embedding every few blocks (the thread's source for t re-injection) | thread | **Question**: J1 (c), M3 |
| ReLU² MLP instead of SwiGLU | thread | **Question**: F6 |
| SiTU-GLU: SwiGLU with pre-activations capped by β·tanh(x/β) (MOSS-Image) | thread | **Question**: F6 |
| Zero-centred RMSNorm: gain `1 + w` with decay on w (Krea 2) | thread | **Not applicable without weight decay**: the fallback runs none (SPEC §6). It returns if decay does (G8) |
| Cautious weight decay: decay only where the gradient's sign agrees with the weight's | thread | **Not applicable**: no weight decay; the norm ceiling bounds the matrices (SPEC §6, G8) |
| A 0.5× lr on the input projections, which take gradient from every block | thread | **Not applicable**: the trust step erases gradient scale per matrix |
| Logit softcap, bigram hash embeddings, sliding-window warmup | thread | **Not applicable**: LLM-specific (no output logits, no vocabulary, 64 tokens) |
| Paired-head or sparse-gated attention | thread | **To read**: the per-head sigmoid gate exists; what "paired head" means in the records is unknown here |
| FP8 matmuls (the speedrun's lm_head) | recalled | **To read**: the RTX 4070 Super (Ada) has FP8 tensor cores; a throughput question for H1 if the step moves to forward/backward |
| Batch-size or sequence-length schedules | recalled | **To read**: bears on C1 |
| A trapezoidal lr schedule with a long cooldown | recalled | **To read**: bears on C4 and the anneal (SPEC §6) |
| Long and short attention windows across layers | recalled | **Not applicable** at 64 tokens; to read for 256 px and beyond |

## F6. Which MLP activation

The trunk has SwiGLU at hidden 4 × width: `w_down(silu(w_gate y) · w_up y)`, three matrices, 3 · 4 · 768² = 7.1M
parameters per block, the gate adaLN-modulated and the write sandwich-normed (SPEC §3).

**ReLU² as the sources have it** (read 2026-11-17, verbatim code):
- nanochat (`karpathy/nanochat`, `nanochat/gpt.py`): `c_fc` 768 → 4·768, `F.relu(x).square()`, `c_proj` back; two
  matrices, no bias; `c_fc` uniform at 0.4 × `√3/√n_embd`, `c_proj` zero-initialized. The repo states no reason for
  the choice.
- modded-nanogpt record 5 (2024-10-14, bundled with padded embeddings, zero-init projections and QK norm, 22.3 → 15.2
  min): the code comment is the only measurement, "~1-2% better than GELU", against GELU, not SwiGLU, citing
  arXiv 2109.08668 (Primer). Record 59 (2026-01-10) is a fused Triton kernel for the linear-ReLU²-linear step.
- The thread's "~79 % exact activation sparsity" saving H100 power is unverified and an LLM figure.

**What differs here:**
- *Equal parameters.* ReLU² with two matrices matches SwiGLU's three at hidden 6 × width (nanochat runs 4 ×, so it is
  smaller than its SwiGLU equivalent). At hidden 4 × the MLP is two thirds of SwiGLU's: −2.4M per block, about −57M
  at depth 24 of 204.6M.
- *Speed.* Two matmuls instead of three at equal hidden; at equal parameters the FLOPs are equal, and the win is only
  the fused elementwise and the sparsity, which dense tensor cores do not use.
- *The target is continuous.* An LM's MLP is a key-value lookup into a vocabulary; here it prints the per-token
  correction onto the slice (`topics/residual-lanes.md`). ReLU²'s output is non-negative before `c_proj` and grows
  quadratically, which with the sandwich norm on the write is bounded per token; without it, it is the spike risk
  SiTU-GLU addresses.
- *Optimizer.* Usui's trust step is scale-free per matrix, so ReLU²'s quadratic scale does not move the step size;
  nanochat's 0.4 × init does not carry over as a tuning.
- *Zero-init.* nanochat zero-inits `c_proj`; here the adaLN gate rows are zero instead (SPEC §5), so `w_down` keeps
  its init and the block is still the identity at step 0.

**Arms**, each against the reference at the agreed length, read on loss, loss by t and img/s:
- ReLU² at hidden 6 × (equal parameters).
- ReLU² at hidden 4 × (nanochat's shape, fewer parameters: a speed-per-quality read).
- SiTU-GLU (MOSS-Image, β·tanh(x/β) on the pre-activations), only if spikes return; the sandwich norm removed the
  ones seen before it (G8).

Open for the discussion: whether the MLP's gating (SwiGLU's product) matters for a block that writes a continuous
correction, which is what ReLU² removes.
