# Joint attention in place of self then cross attention

2026-11-20. Closes the question of whether the caption's token states join self-attention as extra keys (PLAN J2).

Decided: every attention block runs one softmax over the image keys and the caption's keys, with the caption's own
key/value Linear per block (512 → key/value heads) and k norm, no RoPE on the caption; the cross-attention sublayer,
its gate and branch norm leave the model. Architecture a4 (SPEC §3).

Reasons:
- Lumina 2, Z-Image and Qwen-Image read the caption in the same softmax as the image (`topics/conditioning.md`);
  every head can choose between image and caption per query.
- Measured (2026-11-20, 6000 steps, seed 41; log 2026-11-20-joint-attention-and-caption-refiner): joint attention
  with the sink is level with self then cross on loss (score 1.4638 against 1.4635, held-out 1.3076 against 1.3069,
  within the seed spread of about 0.001) at 5 % higher throughput (552 against 524 img/s).
- Without the sink (`a2-joint-6k`) it is 0.0020 behind joint attention with the sink; the sink is closed with it
  (archive 2026-11-20-attention-sink).

Not taken: one global Linear 512 → width translating the token states once for every block's `qkv`.
