# The long run's size: d768 × 24, saturated at about 50 image tokens per parameter

2026-11-18. Closes PLAN G1.

Decided: the long run is the reference model as it stands, width 768 and depth 24 (191.3M image parameters, FACTS),
the MLP ratio unchanged. It trains toward saturation at about 50 image tokens per image parameter, the text-to-image
handbook's compute-optimal ratio, counted on the image parameters as there (`topics/scaling.md`): eight epochs of the
128 px set and three of the 256 px set, 10.5B image tokens, about 55 per image parameter. The next size, when one is trained, is d1024 × 32 (454.4M image parameters, width ÷ depth 32).

Reasons: width ÷ depth 32 sits in the handbook's equal-compute optimum (28 to 34); at a fixed ratio twice the compute
buys only √2 the parameters, so a larger model at the same wall-clock would be short of data; the samples of one pass
read as an unfinished model rather than a capacity limit (*Hypothesis*). Depth 24 carries 18 3×3 convs, enough to
span the 16 × 16 token grid of 256 px corner to corner (15 needed), so the conv's reach does not ask for more depth.
The MLP ratio is not reopened. Repetition at eight and three epochs is far
below the handbook's 50 passes. The equal-parameter shape arms of G1 (768 × 32 against 1024 × 16 and 2048 × 8) are
not run.

Not settled here: whether the ratio carries over from pixel space to these latents, and the lr at that length (C2).
