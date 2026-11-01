# Interval logging, 2026-11-01

Decided by Daniel: "on logging, we should accumulate on gpu like we do on ai-toolkit and report at interval, here 100 steps."

| # | Decision | Reason |
|---|---|---|
| 1 | One train.jsonl/wandb record per `log_every` steps (100 in the smoke config), the reads averaged over the interval on the GPU; Usui's diagnostics are drained at the same record | The step loop then has no device-to-host transfer, as in ai-toolkit |
| 2 | The non-finite check runs at the record and names the first non-finite step | Consequence of 1: UsuiTrack zeroes non-finite gradients, but the AdamW fallback can step on them for up to `log_every − 1` steps before the run stops. Eval and checkpoint cadences must be multiples of `log_every`, so nothing is evaluated or saved from an unchecked step |
