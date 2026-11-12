# Constant lr; the anneal is a run of its own

2026-11-12. Replaces the linear decay over the run (archive 2026-11-05, sandwich norm, pin and decay).

Decided: after warmup the lr is constant (SPEC §6). An anneal, when one is wanted, continues from a checkpoint and
takes Usui's lr down a cosine to 1e-5 over at most one epoch, the fallback's by the same factor (`anneal_to` in a
run's config).

Reasons: the slow linear decay over the whole run moved every metric with it, so a run's curve mixed learning with
the schedule, and a decayed run could not be continued well. A constant lr leaves a run continuable at any
checkpoint; the anneal banks the loss when a result is wanted.
