# The final linear on the fallback

2026-11-02. Closes PLAN C4.

Facts weighed (FACTS, 1000-step ablations; one seed): with the final linear on `StochasticAdamW` at the fallback lr
1e-3 instead of Usui's r = 64 frame, loss over steps 501–1000 was 0.076 lower, against a noise yardstick of 0.011,
mostly in t bins 1–3 (bin 1: 1.19 against 1.54) and tapering to none by bin 8. C4 had asked for the two arms over
one pass or more of the data; that run was not made, so whether the lead holds over a full run is unmeasured.

Decided: the final linear goes to the fallback. It is a bottleneck and Usui means slower adaptation for it; it works but
is not efficient.

SPEC §6 and the init table state it; `build_optimizers` excludes `final.weight` from Usui. Every tracked matrix now
takes one rank fraction. The fallback's lr and betas are unchanged; they stay open in PLAN C2.
