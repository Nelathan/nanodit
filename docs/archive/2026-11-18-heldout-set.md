# A held-out set from the training distribution; the eval grid at guidance 1

2026-11-18. Closes PLAN D6 and D2.

Decided: a run on a new set scores two sets at every eval, Lunara (out of the training distribution) and 1,000 rows
of the training set held out of it (SPEC §1, §8). The held-out rows are Fine-T2I's square subset with enhanced prompts,
generator FLUX.2-dev, picked by the smallest SHA-256 of their key, so a set built again from the same data, with the same
exclusion list, holds out the same rows. `scripts/holdout.py` writes their rows and the set's `heldout_keys.txt`, a list beside the exclusion
list; the stream draws neither, and a run refuses a held-out set whose rows it would draw. A run that continues one
trained on those rows, such as a continuation of `a1-ref-1ep`, scores Lunara only.

Reasons: without an in-distribution holdout a capacity limit cannot be told from a style gap; the handbook's case for
held-out loss over FD rests on a held-out set from the training distribution (`topics/scaling.md`). Of the synthetic
subsets, FLUX.2-dev is the full model of the two generators (Z-Image-Turbo is distilled); the stored captions are of
about the same length in every synthetic subset. The square subset keeps every held-out row in one bucket, so the set
reads as one tensor as Lunara does. Keys in a list rather than a reshuffle keep the set's row order, so every earlier
run's data order stands.

Decided: the eval grid renders at guidance 1, where CFG reduces to the conditional prediction (SPEC §8). At guidance 1 a model follows the
prompt worse but should render in distribution, and the grid shows whether it does. CFG is for post-training.
