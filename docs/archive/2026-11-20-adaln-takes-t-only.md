# The adaLN takes t alone; the caption enters only by cross attention

2026-11-20. Closes the question of whether the pooled caption keeps the adaLN alongside cross attention (PLAN F3).

Decided: the conditioning vector is `c = MLP_t(sinusoid(t))`. The pooled caption embedding, its projection
`text_proj`, the learned null embedding and the `pooled_ada` switch leave the model. The caption reaches the network
only through the token states its attention blocks read, so a layout holds at least one attention block; the null
prompt is the null tokens alone. Architecture a3 (SPEC §3, §5). The pooled embedding stays in the data and in the
precompute, unread, and the null file keeps its `null` entry, unread. A t-dependent summary of the caption in the
adaLN (the caption read, branch `arm/caption-read`) goes with it.

Reasons:
- None of the current text-to-image DiTs read for this feeds a pooled caption into its modulation. Qwen-Image,
  Lumina 2 and Z-Image modulate from t alone (diffusers code); the reports of Qwen-Image-2.0, Krea 2 and MOSS-Image
  describe no pooled input. The caption reaches each of them as tokens (`topics/conditioning.md`).
- A pooled summary lets the network follow a prompt's class without attending to its tokens; prompt following needs
  the attention to carry the caption. *Hypothesis*: no prompt-following read exists yet to measure it.
- The decision is taken against the measured loss. With joint attention and a sink, at 6000 steps, seed 41 (runs
  `a2-jointsink-6k`, `a2-jointsink-nopool-6k`, `a2-jointsink-captionread-6k`; probe `scripts/probe_caption_use.py`
  on `arm/joint-probe-a2` `7d8fa8f`, its form for a2):

| Arm | Loss | Held-out | Nulling the caption, mean | at t 0.84 |
|---|---|---|---|---|
| Pooled embedding in the adaLN | 1.4638 | 1.3076 | +0.0030 | +0.0076 |
| No pooled embedding | 1.4655 | 1.3082 | +0.0028 | +0.0067 |
| t-dependent caption read in the adaLN | 1.4639 | 1.3067 | +0.0040 | +0.0108 |

  The pooled embedding is worth 0.0017 in loss, about twice the one measured seed-pair spread (0.0009, FACTS), and
  removing it does not raise what nulling the caption costs. What moves is where the caption is read: block 3's
  caption mass at t 0.16 is 0.106 without the pooled embedding and 0.025 with it. The caption read recovers the
  loss, as a summary path the current models do not have. With cross attention at d768x12, with and without the
  pooled adaLN ended within 0.002 (log 2026-11-07-trunk-arms-6k-and-lr-sweep).

Not taken: removing the adaLN and giving t at the input (PLAN J1, parked).
