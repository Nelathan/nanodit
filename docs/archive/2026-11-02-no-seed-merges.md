# No seed merges

2026-11-02.

Three children trained on disjoint thirds of the data and merged by multislerp gave a broken model (loss 2.60
against 1.41 for each child), and 5000 more steps from the merge ended 0.009 below the single run at the same step
after 4× its compute (FACTS, LR sweep and seed merge). Decided: no merges, a fresh init each time.

Every run starts from a fresh init. The merge's shuffled part files are deleted.
