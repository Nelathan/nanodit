# Working rules for agents on tiny-dit

- `docs/SPEC.md` is the source of truth. Implement what it says. Any deviation, however small, is a design change:
  stop and bring it to Daniel with the facts that decide it. Do not "improve" the design silently.
- Record each new decision in `docs/DECISIONS.md` with its reason, once Daniel has made it.
- An observation is not a fix request. Report it; change nothing until asked.
- Fail early: verify the risky assumption before building on it (e.g. Primal latent stats, EmbeddingGemma output norm,
  Usui's handling of 1-D params and target-norm overrides) with a small script and report the numbers.
- Never discard a working state without isolating the cause of a problem.
- Pure PyTorch. No HF Trainer, no Accelerate, no ai-toolkit, no wandb.
- Logs are JSONL in the run directory, one object per line, so they can be read directly.
- EmbeddingGemma 2 runs in bf16 or fp32, never fp16.
- Reports say what was verified, what was assumed, and what is still open.
