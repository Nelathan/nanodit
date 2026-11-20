# Attention third of four blocks

2026-11-20. Closes the position of the attention block in a period of four (PLAN F4, the position only).

Decided: the layout is conv, conv, attention, conv, so the last block is a conv. Architecture a4 (SPEC §3).

Reasons:
- Measured (2026-11-18, 6000 steps, seed 41; logs 2026-11-18-attnfirst, 2026-11-18-attnthird): first, third and
  fourth of four are level within the seed spread.
- With the loss level, a prior decides: an earlier ablation on another model found the third position
  best of the 3/4 layouts, and attention third reads after two convs and is followed by one.
- One attention block in four is kept against one in three (log 2026-11-19-conv-forms-and-attn-middle-of-three:
  0.0044 ahead of fourth of four at 6000, about 0.003 ahead of third of four): attention's cost grows with the square of the tokens, and at 256 px and above one in four
  is taken as the knee of quality against cost, a *Decided* judgment, not a measurement.
