# No EMA weights and no weight merging; the cooldown is a short run

2026-11-18. Closes PLAN D1.

Decided: no EMA of the weights and no merging of checkpoints or runs. The noise in the weights at the end of a run is
taken out by a cooldown (SPEC §6), a short run of its own from a checkpoint, for eval or as post-training on the
data of the moment. A long cooldown over a large share of the main run is not wanted.

Reasons: a cooldown of a tenth of a pass buys 0.95 % in held-out loss after one pass, as much as the pass's second half
did at constant lr (log 2026-11-18-anneal3k), in the range the text-to-image handbook measures; there the cooldown is
ahead of EMA in held-out loss at every budget, and EMA in a short run still holds part of the init
(`topics/scaling.md`). One more set of weights and its lag are not needed when the anneal can be taken from any
checkpoint.
