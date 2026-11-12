# Data and conditioning, 2026-10-31

Decided after the CPU encoder probes (`FACTS.md`, Primal 2x2 and EmbeddingGemma 2). Each entry: what, and why.

| # | Decision | Reason |
|---|---|---|
| 1 | Train on a few shards, sized for ~50-min runs; the full set only once short runs are trusted | Longer only once short runs are trusted. Precompute for 2M would be days of CPU encoder time (FACTS) |
| 2 | Precompute streams per shard into Parquet | The latents must stream. The raw set (~423 GB) does not fit the free disk, and one file per shard appends as shards arrive |
| 3 | Embeddings stay as the encoder returns them: no re-centering or per-dim standardization | The per-dim anisotropy (FACTS, EmbeddingGemma 2) is left as the encoder gives it |
| 4 | Null embedding starts from the encoder's embedding of the empty caption, not a random unit vector | An authentic null vector is wanted. A random unit vector is far from the real embeddings, which share a common direction (FACTS) |
| 5 | Load EmbeddingGemma 2 through `sentence-transformers`, adding `torchvision` | The reference pipeline (pooling, normalize) instead of a hand-copied one; torchvision only because its processors import it |
| 6 | No augmentation for now; latent flip parked (PLAN B4) | Flips are not needed now. Primal's encoder is not flip-equivariant (FACTS), so a flipped latent is not one the encoder would give |
| 7 | Usui with `Molegrad`, batches by gradient accumulation | Works well with accumulation (exact, its leaves are token sums), skips the `dW` GEMMs (usuitrack `docs/SPEC.md` and `docs/FACTS.md`) |
| 8 | EmbeddingGemma 2 runs in fp32 for the CPU precompute; SPEC allows bf16 or fp32 | Within the AGENTS rule (bf16 or fp32, never fp16), which SPEC now states: bf16 is 5× slower on CPU (FACTS) |
