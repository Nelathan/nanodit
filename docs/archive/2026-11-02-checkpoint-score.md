# A score at every checkpoint

2026-11-02.

Checkpoints were saved with no number to compare them by: the training loss is noisy in t and data, and the eval
grid is read by eye. Daniel: "Yes do 1 the held out score but not a garden of metrics, just with caption and without.
And not random t but picked distribution." On the eval set: "Use the [lunara] art eval as eval set", the
`moonworks/lunara-art-eval` he had named earlier for later use ("keep lunara for fid or finetune later", PLAN D5).

At every checkpoint the run scores the flow loss on the eval set with each row's caption and with the null embedding,
every row at the same 10 t, the midpoints of equal-probability bins of the training t, each row with its own fixed
noise. Two scalars go to wandb, the per-t values to `score.jsonl`. SPEC §8 states it. The eval set is the 1000
Moonworks Lunara renders, encoded by `scripts/precompute_lunara.py`; the configs name it as `eval_set`.
