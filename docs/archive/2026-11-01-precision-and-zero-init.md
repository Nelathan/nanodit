# Precision and zero init, 2026-11-01

Decided by Daniel after the first GPU smoke runs (`FACTS.md`, GPU smoke runs). Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | Training precision (PLAN C4): a pure bf16 model with stochastic rounding, not fp32 + TF32 and not autocast | Daniel: "it should be bf16 … we have stochastic rounding for that and it works". It is the path Usui was built and measured on (its LFM and Anima runs load a bf16 model; its README lists autocast as untested). Measured at 128 px, one bf16 seed: a slower start (+0.006 over steps 1–50), no difference beyond noise from step 100, 14% faster (FACTS) |
