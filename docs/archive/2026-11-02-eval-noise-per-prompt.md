# Eval grid: a fixed noise per prompt

2026-11-02.

The eval grid gave all 16 prompts one noise (seed 42, repeated). The structure of the samples repeated
across prompts in the grids of `128_2e-3_0-10_10ep`. Reading, not measured as such: the caption barely
moves the samples (the held-out caption-against-null gap at that run's last step was 0.0015 in loss, scratch
scoring, not in FACTS), so one noise draws one layout.

Decided: seeds 1 to 16 for the samples, so the noise shape does not show on every image, while each cell still evolves
stably.

Prompt i now takes the noise of seed `eval_seed + i`; the smoke config sets `eval_seed = 1`, so seeds 1 to 16.
Each cell keeps its noise across steps. SPEC §8 states it.
