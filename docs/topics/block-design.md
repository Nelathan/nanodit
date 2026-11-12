# Block design: mixers, layout, rope, heads

Topic doc for cross attention (archive 2026-11-12, plan history), PLAN F4 and G10, opened 2026-11-06 from a list of suggestions. It holds every suggestion with its status, the
experiment ledger and the lessons drawn. Measured numbers go to FACTS, closures to `archive/`.

## Current design (2026-11-06)

Shapes for d768x12 (width 768, 12 blocks, 12 heads of 64, B images, N tokens = 64 at 128 px).

**Tokens.** The 2-channel latent [B, 2, 64, 64] goes through `pixel_unshuffle(8)` to [B, 64, 128]: one primal block (2·8·8 values) per token.
The patch embed is one `Linear(128 → 768)` whose first 128 rows start as the identity, trainable everywhere (the meta-only arm replaces it,
see Decided). The stream is x [B, N, 768]. The output is read from its first 128 channels, un-patchified, with no final layer in between.

**The conditioning vector c [B, 768], one per image.** The timestep t becomes a 256-d sinusoid (fixed features), then a learned MLP
(`Linear 256→768`, SiLU, `Linear 768→768`), because the sinusoid is only a fixed basis and the MLP is the learned map from it into the
space the modulation reads (the DiT standard). The pooled caption embedding (768-d, rms-normalised) goes through one `Linear 768→768`. c is the
sum of the two.

**The adaLN projection.** One matrix for the whole network: `SiLU`, then `Linear 768 → 6·768` (a 4608×768 weight, 3.5 M parameters), applied to c
once per forward. Its output [B, 4608] is viewed as [B, 6, 768]: six vectors per image, shift_a, scale_a, gate_a (for the token mixer) and shift_m,
scale_m, gate_m (for the MLP). Each block adds its own learned table [6, 768] to those six vectors, so the images share one projection and
each block shifts it. Nothing here depends on the token, only on the image: every token of an image is steered by the same six vectors.

**Where it enters a block.** For the mixer (attention, or conv in conv blocks):
`y = rms_norm(x) · (1 + scale_a) + shift_a` (scale and shift broadcast over the N tokens, no learned norm weight);
the mixer reads y and its output goes through a learned RMSNorm (the post-norm); that is multiplied by `gate_a` and added to the stream:
`x = x + gate_a · post_norm(mixer(y))`. The MLP does the same with rows 3–5: `x = x + gate_m · post_norm(SwiGLU(y'))`, y' being x
modulated by scale_m, shift_m. The gate rows start at zero (projection rows and tables alike), so every block is the identity at init and the
output equals the input patch; shift and scale rows start small random (std 0.02).

**Blocks.** Self-attention: MHA, 12 heads, per-head q/k RMSNorm, RoPE on all head dims, a sigmoid gate on the heads before `out`. Conv blocks
(period 4: blocks 0–2, 4–6, 8–10 are conv, 3, 7, 11 attention): depthwise 3×3 conv over all 768 channels, `out` 768→640 into the meta channels
only (a zero pad stands in for the 128 image channels), then the same post-norm and gate. MLP: SwiGLU, hidden 4·768.

## Suggestions, in the order given

Each comes from that list unless noted. Status: open, queued, running, read.

1. **Meta-only patch embed at one primal block per token** (2026-11-06). The patch is concatenated unchanged into the first 128 channels,
   `Linear(128 → width − 128)` fills the rest. The 16 px arm changed this and the token size together; the token size was the bad
   part ("4 blocks per latent"). Token size stays one block. Variants: the same with a small MLP in the embed. Queued → `conv4_metaembed_3k`.
2. **Cross attention gets no adaLN**: no shift, scale or gate rows from the pooled embedding or the timestep; it keeps
   a static learned gate per channel (a default, not decided, to keep the zero-init identity). Open.
3. **The 3k cross-attention trial decides nothing**: following a sequence is much harder and cannot be learned in 3000 steps;
   after five epochs the pooled-only model followed prompts oddly, which is either a young model or the pooled vector
   conflating concepts. The pair needs a longer budget, or a prompt-following read rather than caption loss. Recorded.
4. **Parallel mixers with one output projection writing only the meta channels** (an experiment, not a rule): self-attention,
   cross attention and the conv all read the same modulated input and their outputs are summed through one `out`. Three mixer
   types: conv, self, cross. Open, needs cross attention back in the model.
5. **Sparse layout**: n conv-only layers, then one layer with cross and self attention; if self-attention is cheap, more often.
   Open, after 4.
6. **Conv at 3/4 of the blocks is the edge and preferred** (a faster model allows more layers). Read: loss level at 3000, 1.32× faster.
   The conv is the basic form; look at how GDN2 does its short conv, and allow writing to all channels once the model is smart enough.
7. **adaLN stays for the conv layers.** Pooled-embedding ada is redundant once tokens arrive through cross attention but cheaper than
   the alternative; the conv blocks already take their modulation rows, so it works as is.
8. **PaLM-style parallel block**: the MLP in parallel with the attention (one norm, one modulation), read for speed and loss.
   Queued → `conv4_pmlp_3k`.
9. **Cross attention: aggressive GQA and NoPE** — the keys come from an already-encoded sequence. NoPE already holds (no RoPE on
   the keys). Open → GQA arm after cross attention is back in.
10. **Self-attention ablations never run**: RoPE vs partial RoPE vs NoPE; MHA vs GQA. MHA was argued as cheap, but the runtime
    shows attention is not cheap even at 64 tokens (conv in its place gives 1.2–1.3×), and a 1-megapixel image costs far more.
    Queued → `conv4_nope_3k`, `conv4_rope25_3k`, `conv4_gqa3_3k`. Half the frequencies would be a null: the 8 slow ones turn under 0.07 rad across the 8×8 grid, so they are already position-free; a quarter keeps only the fast ones.
11. **256 px is available, so test position codes and head sharing where the grid is bigger, and interpolate** (2026-11-06). Two reads:
    the 128 px checkpoints scored on a 256 px Lunara set (a 16×16 grid the model never saw: how RoPE, partial RoPE and NoPE carry across
    resolution), then arms trained at 256 px (base, NoPE, quarter RoPE, GQA) where attention costs more and position has room to matter.
    Queued behind the 128 px arms; `scripts/score_checkpoint.py` (branch `conv-meta`) does the first.
12. **Fine-T2I: crop when it must be, never stretch** (not all to 1:1). Recorded in PLAN B6.

## Decided (2026-11-06)

- Patch embed: concat. The raw 128-d patch stays in the first 128 channels and `Linear(128 → width − 128)` fills the rest. It keeps the image
  channels pure, so the final read has no bottleneck, and each step then improves the image rather than only altering it, which seems to help
  gradient flow. Fewer degrees of freedom for nothing. SPEC §3 changes when the merged model carries it.
- Self-attention: a quarter of the rotary frequencies at least (RoPE is needed at 256 px and will be going up to 1024 px) and GQA (keep: it is faster; there
  is no cache to slim, so the case is speed alone). NoPE is not enough on the unseen grid; partial RoPE is the best of both.
- Cross attention: no positional encoding (the token states are already encoded) and likely GQA too: one 512-d vector needs few
  readings. No adaLN on it, a zero-initialised gate. Pooled-embedding adaLN beside cross attention may duplicate: test with and without at a length
  where tokens can matter.
- PaLM parallel MLP: dropped for good.
- Eval length: 3000 steps has barely any structure to read. Arms from here run 6000 steps (half a pass over the trial set, about 28 minutes at the
  conv speed), the balance between 3000 and 10000.
- Size generalisation is read from the 128 px checkpoints on 256 px; a 3000-step checkpoint does not yet hold proper positional readings,
  which is why conv dominated there. Speed is read by regression: step time at 128, 256, 512 and 1024 px per layout, projected to larger grids
  (`scripts/bench_speed.py`, FACTS).
- Branches are kept tidy.

## Open (2026-11-06)

Which attention types go in which layers once the 3/4 conv pattern stays, and whether sequentially or in parallel: the attention slot (every 4th
block) as self only, self then cross, self and cross in parallel into one meta-only `out`, or cross in a different slot than self.
Needs a layer notation in the config and a merged model that carries conv, GQA, partial RoPE and cross attention together.

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
| `conv4_gqa3_rope25_meta_6k`, `conv4_6k`, `attn_6k` | the decided combination, the conv base, the all-attention baseline at the 6000-step eval length; each also scored at 256 px | running | | the new references; later arms compare to these |
| `conv4_metaembed_3k` | meta-only patch embed, 8 px token (1.) | read | 1.4869 | level (−0.0015), same speed; the 16 px loss was the 4-blocks-per-token packing, not the meta-only write |
| `conv4_pmlp_3k` | PaLM block (8.) | read | 1.4908 | +0.0024 and no speed gain: dropped |
| `conv4_nope_3k` | no rotary position (10.) | read | 1.4895 | +0.0011 at 128 px, +0.0034 on the unseen 256 px grid |
| `conv4_rope25_3k` | a quarter of the rotary frequencies, the 4 fastest per axis (10.) | read | 1.4883 | level with full RoPE at both resolutions |
| `conv4_gqa3_3k` | 3 key/value heads against 12 (10.) | read | 1.4888 | level, 1.2 % faster in training; 2–3 % at every grid size in the benchmark (FACTS), a smaller projection and no cheaper attention |

## Lessons

- Conv in place of attention does not cost loss at 1/2 to 3/4 of the blocks and gains 20–32 % speed; it leads early (−0.007 at 1000 steps) and
  the attention baseline catches up by 3000. Early lead plus late equality reads as the conv learning local mixing faster, with nothing
  more to learn at this length; a longer run is what shows whether attention's global mixing pays later.
- Caption loss at 3000 steps cannot judge token conditioning. It measures the class-level signal the pooled embedding already carries.

Arms run in the order metaembed, pmlp, nope, gqa3, rope25, about 15 minutes each; code on branch `conv-meta`. Arms compare against `conv4_3k` (1.4884 at 3000, 1.5474 at 1000, 1.5037 at 2000).
- Rotary frequencies at 8×8 tokens: only the fastest 4 of 16 per axis turn more than 0.5 rad across the grid; partial RoPE contrasts must cut below half to differ from full RoPE. At 1 megapixel the grid is far larger and the slow frequencies matter, so a 64-token RoPE read does not carry over.
- Conv carries across resolution far better than attention: trained at 128 px, the 3/4-conv model scores 0.031 better than the all-attention one on the 256 px grid, though it is 0.003 worse at 128 px. Local mixing does not depend on how many tokens there are; global attention at 256 tokens meets position offsets it never saw.
- At 64 tokens none of NoPE, quarter RoPE or GQA costs anything, and the 256 px read keeps that order: full or partial RoPE best, NoPE worst by 0.003. Position helps a little when the grid grows, and a quarter of the frequencies carries all of it.
- The parallel MLP buys no speed (no fusion of the two branches under `torch.compile` at this size) and loses a little: dropped.
- Speed by regression (FACTS, step speed by layout): the quadratic term of attention is 1.78 ns per token per token of grid; conv in 3/4 has a quarter of it. Conv is 1.27× faster at 128 px and 1.66× at 1024 px; the fit projects 2.4× at 2048 px. GQA's 2–3 % does not grow with the grid: it saves projection work, not attention work. Earlier note that its saving grows was wrong.
- Meta-only embed at one primal block per token is free and slightly ahead at both resolutions; it keeps the first 128 channels as the raw latent. Keep.
