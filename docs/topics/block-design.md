# Block design: mixers, layout, rope, heads

Topic doc for PLAN F3, F4 and G10, opened 2026-11-06 from Daniel's list. It holds every suggestion with its status, the
experiment ledger and the lessons drawn. Measured numbers go to FACTS, closures to `archive/`.

## Current design (2026-11-06)

- Patch embed: `pixel_unshuffle(8)` on the 2-channel latent, one primal block (2·8·8 = 128 dims) per token; one
  `Linear(128 → width)`, identity-initialised on the first 128 channels and trainable everywhere. No embed MLP. The time
  embedding is a 2-layer MLP and the pooled text embedding one `Linear(768 → width)`; their sum feeds one shared adaLN
  projection to the six (nine with cross attention) modulation rows, plus a per-block table.
- Block: adaLN-modulated self-attention (MHA, per-head q/k RMSNorm, RoPE, a sigmoid gate on the heads), then the SwiGLU
  MLP; every branch has a post-norm and a zero-initialised gate.
- Self-attention is MHA at 64 tokens; RoPE is on in every head dimension.

## Suggestions, in the order given

Each is Daniel's unless noted. Status: open, queued, running, read.

1. **Meta-only patch embed at one primal block per token** (2026-11-06). The patch is concatenated unchanged into the first 128 channels,
   `Linear(128 → width − 128)` fills the rest. The 16 px arm changed this and the token size together; the token size was the bad
   part ("4 blocks per latent"). Token size stays one block. Variants: the same with a small MLP in the embed. Open → arm `metaembed`.
2. **Cross attention gets no adaLN**: no shift, scale or gate rows from the pooled embedding or the timestep; it keeps
   a static learned gate per channel (my choice, to keep the zero-init identity). Open.
3. **The 3k cross-attention trial decides nothing**: following a sequence is much harder and cannot be learned in 3000 steps;
   after five epochs the pooled-only model followed prompts oddly, which is either a young model or the pooled vector
   conflating concepts. The pair needs a longer budget, or a prompt-following read rather than caption loss. Recorded.
4. **Parallel mixers with one output projection writing only the meta channels** (an experiment, not a rule): self-attention,
   cross attention and the conv all read the same modulated input and their outputs are summed through one `out`. Three mixer
   types: conv, self, cross. Open → arm `parallel_mixers`.
5. **Sparse layout**: n conv-only layers, then one layer with cross and self attention; if self-attention is cheap, more often.
   Open, after 4.
6. **Conv at 3/4 of the blocks is the edge and preferred** (a faster model allows more layers). Read: loss level at 3000, 1.32× faster.
   The conv is the basic form; look at how GDN2 does its short conv, and allow writing to all channels once the model is smart enough.
7. **adaLN stays for the conv layers.** Pooled-embedding ada is redundant once tokens arrive through cross attention but cheaper than
   the alternative; the conv blocks already take their modulation rows, so it works as is.
8. **PaLM-style parallel block**: the MLP in parallel with the attention (one norm, one modulation), read for speed and loss.
   Open → arm `parallel_mlp`.
9. **Cross attention: aggressive GQA and NoPE** — the keys come from an already-encoded sequence. NoPE already holds (no RoPE on
   the keys). Open → GQA arm after cross attention is back in.
10. **Self-attention ablations never run**: RoPE vs partial RoPE vs NoPE; MHA vs GQA. MHA was argued as cheap, but the runtime
    shows attention is not cheap even at 64 tokens (conv in its place gives 1.2–1.3×), and a 1-megapixel image costs far more.
    Open → arms `rope_partial`, `rope_none`, `gqa`.
11. **Fine-T2I: crop when it must be, never stretch** (not all to 1:1). Recorded in PLAN B6.

## Base for the next arms

Conv in 3/4 of the blocks, pooled-only (`128_d768x12_conv4_3k`, 1.4884 at 3000), branch `conv-meta`: it is the layout going forward, and the
attention ablations only mean something where attention is. Cross attention rejoins after its own longer test.

## Ledger

| arm | what changes vs. the base | status | 3000-step eval loss | read |
|---|---|---|---|---|
| `swpin_3k` | pooled-only, all attention | read | 1.4855 | baseline |
| `swpin_cross_3k` | + cross attention in every block | read | 1.5024 | behind by 0.017 at 3000, 28 % slower; decides nothing (3.) |
| `conv2_3k` | conv in 1/2 of the blocks | read | 1.4852 | level, 1.20× |
| `conv3_3k` | conv in 2/3 | read | 1.4859 | level, 1.28× |
| `conv4_3k` | conv in 3/4 | read | 1.4884 | level within 0.003, 1.32×; the new base |
| `metaembed` | meta-only patch embed, 8 px token (1.) | to build | | |
| `parallel_mlp` | PaLM block (8.) | to build | | |
| `rope_partial`, `rope_none` | rope on part / none of the head dims (10.) | to build | | |
| `gqa` | grouped self-attention heads (10.) | to build | | |

## Lessons

- Conv in place of attention does not cost loss at 1/2 to 3/4 of the blocks and gains 20–32 % speed; it leads early (−0.007 at 1000 steps) and
  the attention baseline catches up by 3000. Early lead plus late equality reads as the conv learning local mixing faster, with nothing
  more to learn at this length; a longer run is what shows whether attention's global mixing pays later.
- Caption loss at 3000 steps cannot judge token conditioning. It measures the class-level signal the pooled embedding already carries.
