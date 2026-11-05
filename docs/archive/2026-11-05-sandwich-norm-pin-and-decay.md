# Sandwich norm, init-norm pin and linear decay

2026-11-05. Closes PLAN G6 (bounding the stream and the weights).

Facts weighed (FACTS: Spikes in `128_d1024x16_5ep`; Branch norm and pin arms; `128_d1024x16_swpin_5ep`). The unpinned
d1024x16 run spiked from step 4850: one out-of-distribution token amplified by the SwiGLU branches of blocks 10–13 into a
stream the slice reads raw. Halving the lr made spikes rarer, not milder. The weights grew 10–15× with no target norm,
and the function came to depend on the grown norms; the growth was also an unscheduled anneal of the relative step.

Decided by Daniel (2026-11-04): the sandwich norm on every branch output, for new inits. The 3000-step arms then showed
the pin to the init norm, at a constant relative step, costing nothing (best of the arms at 2000 and 3000). Daniel
chose it for the long run, with a linear decay of both lrs from their peak to a tenth over 5 epochs and a 500-step
warmup. `128_d1024x16_swpin_5ep` ran 37,605 steps without a spike, ended at 1.3856 against the unpinned run's best of
1.4209, and is the first run whose samples follow prompts at class level. Daniel: "Its a clear success for both the arch
and usui." His reading is that the sandwich norm carried the stability; the run does not separate the two.

SPEC §3 states the sandwich norm, §6 the pin with the trust step and the decay; `Config.pin_weights` defaults to true and
every earlier config states `pin_weights = false`. The unpinned `5ep` checkpoints do not load into the sandwich-norm model.

Not tested and moved to PLAN G8: learned gains on the pinned matrices, a per-row pin, the EDM2 reading. Single seeds
throughout; Daniel: "we dont have the luxury to burn 4x the compute".
