# Training-step speed by layout and resolution (F4, 2026-11-06, `scripts/bench_speed.py` on branch `conv-meta`)

Run log. The dense reading is in FACTS, Architecture at d768x12 (2026-11-06).


d768x12, bf16, blocks compiled, forward + backward (no optimizer), 4096 tokens per batch, RTX 4070 Super. Images per second at 128 / 256 / 512 / 1024 px (64 / 256 / 1024 / 4096 tokens):

| layout | 128 | 256 | 512 | 1024 | tokens/s at 64 → 4096 tokens |
|---|---|---|---|---|---|
| all attention | 1071.5 | 260.6 | 59.2 | 11.2 | 68.6k → 45.9k |
| conv in 1/2 | 1230.9 | 312.1 | 72.2 | 15.4 | 78.8k → 63.1k |
| conv in 3/4 | 1361.3 | 340.9 | 80.8 | 18.6 | 87.1k → 76.2k |
| conv in 3/4, 3 KV heads | 1400.7 | 350.8 | 82.1 | 19.0 | 89.6k → 77.8k |
| all attention, 3 KV heads | 1153.0 | 284.8 | 62.3 | 12.2 | 73.8k → 50.0k |

Per-token time fits a + b·N (N tokens per image): all attention a = 14.6 µs, b = 1.78 ns; conv in 3/4 a = 11.5 µs, b = 0.41 ns (about a quarter of the attention layers, a quarter of b). Conv in 3/4 over all attention: 1.27× at 128 px, 1.31× at 256, 1.36× at 512, 1.66× at 1024, and the fit gives 2.4× at 2048 px (16,384 tokens: 3.3 against 1.4 images per second). GQA with 3 key/value heads adds 2–3 % on conv in 3/4 and 5–9 % on all attention at every size; it leaves b unchanged (1.78 → 1.60 ns all attention, 0.41 → 0.42 conv), so the gain is the smaller key/value projection, not cheaper attention.
