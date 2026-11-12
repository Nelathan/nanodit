# A score at every checkpoint

2026-11-02.

Checkpoints were saved with no number to compare them by: the training loss is noisy in t and data, and the eval
grid is read by eye. Decided: one held-out score, kept to the caption and null conditions rather than a collection of
metrics, at a picked t distribution rather than random t. The eval set is `moonworks/lunara-art-eval`, which PLAN D5
also reserves for later FID or finetuning.

At every checkpoint the run scores the flow loss on the eval set with each row's caption and with the null embedding,
every row at the same 10 t, the midpoints of equal-probability bins of the training t, each row with its own fixed
noise. Two scalars go to wandb, the per-t values to `score.jsonl`. SPEC §8 states it. The eval set is the 1000
Moonworks Lunara renders, encoded by `scripts/precompute_lunara.py`; the configs name it as `eval_set`.
