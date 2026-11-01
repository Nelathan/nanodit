# Final linear's frame and the smoke settings, 2026-11-01

Decided by Daniel after the 9000-step runs (`FACTS.md`, Fixed-step runs). Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | The final linear (PLAN A3) takes Usui's largest frame, r = 64 for `[128, 512]`; every other matrix takes the rank fraction | Daniel: "rank 64 is the maximum for usui because a tangent always needs an outside. and the gap closes, this is a strong sign for subspace evolution!" and "i think we can actually keep rank 64". The zero-initialized final linear builds its whole map from Usui's steps; at 9000 steps the rank-64 frame ended about 0.010 lower than r = 13, all below t ≈ 0.7, with both frames settling after about step 1000. Daniel's read: the bottleneck would want a full-rank update, which Usui has no option for; a run with the final linear on the fallback is to be compared "for science". The 300-step gap was early loss and closes with time ("a red herring") |
| 2 | Smoke settings: usui lr 2e-3 (a tenth of 2e-2), beta 0.9, rank fraction 0.1, warmup 100 | Daniel: "lr is too high, lets take 0.1 of it for now. lets do beta 0.9 and rank fraction 0.1", "set warmup to 100". Under 2e-2 `grad_moment_cosine` stayed near −0.3 through 9000 steps. The rank fraction becomes a config field |
| 3 | The vertical streaks and banding on the eval grids are not treated as a defect | Daniel: "the streaks are just first signs of structure the model learns. its not a bad artifact". They appear under both final-linear frames |
| 4 | The run length is set by the data: each image seen once | Daniel: "we should fetch more images so we can do 10k steps with each image seen once." The 9000-step runs saw each of the 200k rows about 11 times |
