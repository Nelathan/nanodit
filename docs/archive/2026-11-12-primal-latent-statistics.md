# Closed: Primal 2x2 latents are close enough to unit-normal

2026-11-12. Decided on the measurement in FACTS, "Primal 2x2" (`scripts/probe_primal.py`, 2026-10-31). The model card
claims unit-normal latents per image and channel, and SPEC §1 and §5 rely on it with no scale or shift. Measured at
128 px: global std 1.056 and 1.021, global means 0.005 and 0.02, per-image std 0.80 to 1.20, 0.34 to 0.44 % beyond
3σ against a Gaussian's 0.27 %; 256 px is closer still. The spread is close enough to the claim that no rescale is
added.
