# The stream's scale: two workspaces, one read, and what each write is worth

Topic doc for PLAN G11. Related: G3 (the common mode, a part of the growth), J7 (tanh on the gates), K1 (what the
trunk can still read of x_t), Group M and `topics/residual-lanes.md` (the stream as lanes). Measured numbers go to
`log/`, closures to `archive/`.

## The question

The stream is two workspaces sharing one residual (SPEC §3): the **slice** (channels 0–127) starts as x_t and is read
out raw as the prediction, with no final norm and no projection; the **meta** channels (128–767) start as the patch
embed's projection of x_t and are never read out. Every branch reads the whole stream through one RMS norm over all
768 channels. Does the stream need its meta part held, its two parts read apart, or its residual replaced, so that
every layer takes an equal part and the slice stays visible to the deep blocks?

## What is measured

- *Measured* (log 2026-11-21-stream-scale): after one pass the slice stays at the prediction's scale, anchored by
  the raw readout; the meta part grows with depth to eight or nine times its input (1.15 to 9.54 after block 22 at
  t 0.16) and grew further in the anneal. Into the last blocks the slice is about a hundredth of the read's energy at
  low t and a thousandth at high t (17 % at init).
- *Measured* (log 2026-11-21-gate-sizes, probe split in log 2026-11-21-stream-scale): the gates on the slice stay
  small; the gates on the meta channels grow with depth, carried by the per-block tables, which the fallback trains
  outside the norm ceiling. The shared projection's gates stay small.
- *Measured* (log 2026-11-21-conv-kernel-and-tanh-gates): at 6000 steps the meta part after block 22 is about three
  times its input, a third of its size after the pass; few gates exceed 1 yet. tanh on the gates is level on loss at
  6000 and leaves the stream within 10 % of the run without it.

## The mechanism

*Hypothesis*, the one the AttnRes paper calls PreNorm dilution (below): with pre-norm reads a block's influence is the
size of its write against the stream it lands in. The plain sum makes the stream grow with depth, so a later block
needs a larger write for the same change of what the next block reads; the gates grow with depth and the stream with
them. On the slice the readout ties the size to the prediction, so it cannot run; on the meta channels nothing in the
loss sees their size, since they are only read through norms, and the tables let the gates run.

Two separate defects follow, and the candidates below address one or both:

1. **Dilution** (the meta stream): the size of a write no longer measures its weight; later writes must grow.
2. **Drowning** (the read): the slice shares the read's norm with a meta part that grows, so the deep blocks read
   x_t and the prediction so far as a vanishing part of their input.

*Decided* (2026-11-21), the direction for the design: every layer should take an equal part; a stream that grows is unstable and
needs a fix, not room for more magnitude; the readout has no norm and no projection, and stays so. The two
workspaces should be separated more cleanly.

## Candidates

### Read the two workspaces apart

Each branch reads `[slice, rms_norm(meta)] * (1 + scale)` instead of `rms_norm(stream) * (1 + scale)`: the slice raw,
at the prediction's scale, the meta part normed on its own. Addresses drowning only: the slice's share of the read
becomes `128 s² / (128 s² + 640)` for a slice RMS s, about 5 % at s 0.5 and 40 % at s 1.8, set by the slice and not
by the meta part's growth, and the slice's scale reaches the read. Dilution in the meta part stays. It is the base the
two candidates after it would sit on. Cost: none measurable; the norm moves to a slice of the tensor.

### Hold the meta stream at a fixed size

After each write the meta part is normed back: `meta ← ρ · rms_norm(meta + gate · w_meta)`, ρ fixed; the slice is
added as now. Addresses dilution: the stream cannot grow, and a write's weight against it is its gate. It does not
give equal participation: each renorm shrinks everything written before, so an early write fades geometrically and
the newest writes weigh most, a recency stream. With an unbounded gate the gate's size is a learned forget rate,
the same freedom as a per-channel forget; whether the write's weight is bounded to [0, 1] is a choice inside this
candidate. The recency is accepted (2026-11-21); early writes that fade are carried forward by long skips
(PLAN M2, the speedrun's U-net skips): the stream after an early block added back before its mirror block.
Post-norm streams are known to train worse than pre-norm in deep transformers; here the writes start at zero, so the renorm starts as the identity. Cost: one norm per sublayer on 640 channels.

### Attention over depth on the meta channels (Full AttnRes)

*Not wanted* (2026-11-21): the step time it costs here (below). Kept for its reasoning and the paper's numbers.

Each sublayer's meta input is a learned softmax mix of the meta writes before it instead of their sum (paper below):

    h_l = Σ_{i<l} α_{i→l} v_i,   α_{i→l} = softmax_i(w_l · rms_norm(v_i))

with v_0 the patch embed's meta projection, v_i the meta part of sublayer i's gated write, and w_l a learned
640-vector per sublayer, zero at init. The mixer and the MLP of each block are separate sublayers: 48 sublayers;
the last reads 48 sources. The slice stays the additive raw residual: the prediction is x_t plus corrections, which a convex mix
cannot build. The read is the split read above, `[slice, rms_norm(h_l)]`.

- **What it fixes.** Dilution: h_l is a convex mix, its size that of one source, whatever the depth. Equal
  participation: at init the weights are uniform over the sources, and they are learned per sublayer after.
- **What it leaves.** The keys are normed, the values are not: a source with a larger write still dominates the
  mix's direction, so a gate's size still buys weight among the sources, though no longer against depth.
- **The identity at init.** The gates start at zero, so every v_i but v_0 is zero, the keys are zero and α is
  uniform: h_l = v_0 / l, and the normed read is the patch embed's projection, as now. The gradient into a write is
  scaled by 1/l in the mix and by l in the read's norm, so it is O(1).
- **The tables** change role: a meta gate sets a source's size within the mix, not its weight against a growing sum.
- **The last sublayer's meta write** is read by nothing, as now: the readout is the slice.

Costs on a4 (d768×24, 238 M parameters, 226 M of them in `model.blocks`); *Estimate*, from the arithmetic shown, not
measured:

| | Full (48 sources per the last sublayer) | Block, 8 blocks of 6 sublayers |
|---|---|---|
| Parameters | 48 × 640 = 30 720 (0.013 %); the key norm has no gain, which the query would absorb | the same |
| Source visits per token, forward | Σ_{l=1}^{48} l = 1176 | 6 · Σ_{n=0}^{7} (n + 2) = 264 |
| FLOPs per token, forward | 1176 · 640 · 4 ≈ 3 M, against ≈ 450 M for the blocks' matmuls: under 1 % | ≈ 0.7 M |
| Memory traffic, forward | 1176 · 640 · 2 B = 1.5 MB per token | 0.34 MB per token |
| Step time, 128 px | 16 384 tokens per step: 25 GB forward, 75 to 100 GB with the backward, at the card's 504 GB/s 150 to 200 ms on the measured 455 ms: **+30 to 45 %** | +7 to 10 % |
| Step time, 256 px | 65 536 tokens: four times the traffic on a 1648 ms step: **+35 to 50 %** | +8 to 11 % |
| VRAM | the 48 sources, the keys if not recomputed, the weights: 252 + 252 + 19 MB per micro-batch of 4096 tokens, 0 to 0.5 GB on the 4.3 GB peak | the same for the sources; fewer keys |

- The work is elementwise, so its cost is memory traffic, not FLOPs; the step-time figures assume a fused kernel that
  reads each source once per consumer. An eager loop of adds is launch-bound and much slower. The step time is to
  be measured with `scripts/bench_step.py` on the arm before it trains.
- **Memory.** The paper's "no additional memory overhead" holds where each layer's output is a tensor autograd keeps anyway. The
  training loop does no activation checkpointing, so nothing is recomputed and nothing new has to be kept alive past
  what the backward needs; in a compiled block the partitioner decides whether a source is one of the saved
  tensors, hence 0 to 0.5 GB. The stacked sources must not be materialized per consumer: kept for the backward,
  1176 × 4096 × 640 × 2 B is 6.2 GB per micro-batch.
- **Compilation.** The unit compiled is the block (`compile_blocks`); the MLP's mix sits inside it, between the mixer
  and the MLP. Each block takes the list of sources before it and returns its two writes; each compiles for its own
  number of sources, as each block is compiled on its own already.
- **Block AttnRes.** The paper's 16-block model (32 sublayers): baseline 1.766, Full 1.737, Block at block sizes 2, 4
  and 8 sublayers about 1.746, at 16 1.753, at 32 1.757 (its Fig. 6); Block keeps 0.020 of Full's 0.029, about 70 %.
  Over its five scaling sizes Block keeps 69 to 96 % of Full's gain, the gap 0.001 at the largest (its Table 2).
  A window over the last 8 outputs and the embedding keeps almost nothing (1.764): distant sources matter.

### Without the per-block tables

The shared adaLN-single alone, a per-block modulation from t (Lumina 2's form) only if that loses. The tables carry
the gates' growth; *Not wanted* as a source of magnitude. Their role changes under each candidate above, so this
arm is read after them, or folded into the one adopted.

### Bounded gates (J7)

tanh on the gates: level at 6000 steps, where the gates have not yet grown. It bounds a write's multiplier, not the
sum, so the stream still grows with depth. Superseded by the candidates above unless they leave the gates large.

### Not wanted

- **A learned forget per channel on the stream** (2026-11-21): it would grow as the tables did.

## How it would be settled

Arms at 6000 steps from a4, read on loss by t (held-out and Lunara at 256 px) and on both probes (`probe_gates.py`,
`probe_stream.py`), against a4-ref-1ep's first 6000 steps and the conv arms' probes at 6000. At 6000 steps the stream
is a third of its size after the pass, so the probes show whether the growth starts; a loss difference may need the
full pass, about 4.4 h at 128 px.

Proposed order:

1. **The split read** alone: the cheapest, and the base the others are read against.
2. **The held meta stream**, with the split read.
3. **The held meta stream with long skips**, with the split read.
4. **Without the per-block tables**, on whichever of the above holds.

## Source: Attention Residuals

Kimi Team, "Attention Residuals", arXiv 2603.15031 (technical report). Read §2–5 from the PDF.

- **Form.** h_l = Σ_{i=0}^{l−1} α_{i→l} v_i, v_0 the embedding, v_i = f_i(h_i) the layer outputs; α a softmax over
  i of w_l^T RMSNorm(v_i), w_l ∈ R^d a learned pseudo-query per layer, independent of the input. Attention and MLP
  are separate layers. All pseudo-queries must start at zero, so α starts uniform: "prevents training volatility, as
  we validated empirically".
- **Motivation.** PreNorm dilution: "as hidden-state magnitudes grow monotonically with depth, deeper layers are
  compelled to learn increasingly large outputs from fixed-scale normalized inputs to remain influential." Their
  baseline shows it in output magnitude by block; Block AttnRes bounds it to a periodic pattern and makes the
  gradient size across depth more uniform (Fig. 5).
- **Block AttnRes.** Layers grouped into N blocks; within a block the outputs sum; attention runs over the block
  sums, the embedding and the partial sum of the current block. N = L is Full, and "N ≈ 8 recovers most of the
  benefit".
- **Cost.** Full needs O(L²d) compute and O(Ld) memory, the memory "overlaps entirely with the activations already
  retained for backpropagation"; with activation recomputation or pipeline parallelism the outputs must be kept or
  sent. Training overhead "negligible" without pipeline parallelism, under 4 % with it; inference under 2 %. These
  are for MoE language models with large layers and 4096 to 8192 tokens, where per-layer compute dwarfs the mix.
- **Results.** Language models of 194 M to 528 M active parameters (Table 2), hyperparameters tuned for the
  baseline: Full and Block below the baseline at every size, Block 1.25× compute-equivalent at the fitted scale.
  At 48 B / 3 B active over 1.4 T tokens, Block (9 blocks) better or level on every benchmark listed.
- **Ablations** (16-block model, Table 4): DenseFormer's fixed input-independent weights 1.767 (no gain);
  input-independent scalars 1.749; sigmoid for softmax 1.741; no RMSNorm on the keys 1.743 (Full), 1.750 (Block);
  per-head depth mixing 1.752 against 1.746 (worse); an input-dependent query from the hidden state 1.731, at a d × d
  projection per layer. The optimum under a fixed budget moves deeper and narrower with AttnRes (Fig. 7).
- **Not covered:** diffusion, image models, a stream whose part is read out raw. Applying it to the meta channels
  only, with the slice kept as a sum, has no source.
