# Closed: EmbeddingGemma 2's pooled output is unit norm

2026-11-12. Decided on the measurement in FACTS, "EmbeddingGemma 2" (`scripts/probe_embeddinggemma.py`, 2026-10-31).
The raw mean-pooled norm is 314 to 1577; the sentence-transformers Normalize module makes it 1.0000 ± 2e-7, and
0.9991 to 1.0009 after rounding to bf16, the same in fp32 and bf16. SPEC §1 stores it as the model returns it, and §3
feeds it to the text projection through `rms_norm`.
