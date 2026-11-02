# Eval grid: a fixed noise per prompt

2026-11-02.

The eval grid gave all 16 prompts one noise (seed 42, repeated). Daniel saw the structure of the samples repeat
across prompts in the grids of `128_2e-3_0-10_10ep`. The agent's reading, not measured as such: the caption barely
moves the samples (the held-out caption-against-null gap at that run's last step was 0.0015 in loss, scratch
scoring, not in FACTS), so one noise draws one layout.

Daniel: "maybe we should have seed 1 to 16 for the samples so we dont see the noise shape on every image. still
stable evolution".

Prompt i now takes the noise of seed `eval_seed + i`; the smoke config sets `eval_seed = 1`, so seeds 1 to 16.
Each cell keeps its noise across steps. SPEC §8 states it.
