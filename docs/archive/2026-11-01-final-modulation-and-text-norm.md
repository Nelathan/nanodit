# Final-layer modulation and the text norm, 2026-11-01

Closes PLAN F3 and builds the text norm agreed with it. Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | The final layer's shift/scale are two more rows of the shared adaLN projection, plus its table, like the blocks (PLAN F3) | PixArt adds raw `c` to the final table; `c` is of order 1 per element, so the final `1 + scale` started anywhere from about −3 to 5, of either sign, and the gradient into the whole network passed through that gain. Through the projection all modulation comes from one place under one init rule (F2): std 0.30, max \|scale\| 1.25 at init (FACTS, adaLN at init). Anima gives its final layer its own projection of the timestep; Krea2 does what PixArt does. Daniel left it to the session: "youre probably right, its yours to own" |
| 2 | The text embedding enters as `rms_norm(e_text)` with no learned gain, in place of `√768 · e_text` | Identical for the encoder's unit-norm embeddings; it also holds the learned null embedding's input to `text_proj` at unit RMS, so its AdamW steps can turn that input but not inflate it. A learned gain would be a second parametrization of `text_proj`, trained by AdamW while `text_proj` trains under Usui. Daniel: "Yes rms is good, or rather a learned norm?", then "Good" to `rms_norm` with no gain |
