# Caption outliers, the 1ep run's out-of-memory, and the cross attention's kernel (2026-11-17)

## The out-of-memory, reproduced

`128_d768x24_1ep` stopped after step 19,600 with `torch.OutOfMemoryError`; the failing allocation was
`(128, 3, L, 64)` bf16 of 94 MiB, a cross attention K/V with L about 2000 (`runs/128_d768x24_1ep.out`).
`pad_sequences` pads each micro-batch to its longest caption, so one long caption sizes the cross attention of all
its rows.

`scripts/probe_caption_memory.py configs/128_d768x24_1ep.toml MICRO_BATCH L…` at `12c2028`, usuitrack `92b23bc`, the
config as in the working tree (σ 0.7; now `configs/a1-ref-1ep.toml`): the training step with one row of every micro-batch given L random token states, 10 warm-up and 5
timed steps per L, GPU otherwise idle. Output `/mnt/mars/ai/tiny-dit/probes/caption_memory_d768x24.txt`.

| longest caption L | micro-batch 64: ms/step | peak GiB | micro-batch 128: ms/step | peak GiB |
|---|---|---|---|---|
| 136 | 450.9 | 4.10 | 453.7 | 7.89 |
| 512 | 496.1 | 4.51 | 505.8 | 8.43 |
| 1024 | 613.2 | 4.93 | 610.6 | 9.26 |
| 2048 | 896.9 | 5.77 | out of memory (384 MiB more asked, in the cross attention's backward) | — |

Read:
- The out-of-memory is the long caption: micro-batch 128 fails at L 2048, as the run did at about 2000.
- Micro-batch 64 measures no slower at ordinary captions (450.9 against 453.7 ms, one run each; the spread is not
  measured) and halves the peak.
- A long caption costs time at either micro-batch: a step whose micro-batches each carry one 2048-token row takes
  about twice as long (the probe puts the row in every micro-batch; a real step has it in one).
- The probe's failing request is 384 MiB, which equals `(128, 12, 64, 2048)` bf16, the size of the boolean mask
  expanded to a full bias over heads and queries (*inference* from the size; the run's own failing request was the
  94 MiB K/V). The failing frame is `_scaled_dot_product_efficient_attention_backward` (full traceback in the
  probe's console, not kept).

## The caption lengths of `set_128`

`scripts/probe_caption_lengths.py /run/media/djg/vel-ssd/tiny-dit/set_128 …`, the `sequence` column's bytes / 1024,
all 8,239,111 rows. Output `/mnt/mars/ai/tiny-dit/probes/caption_lengths_set128.json` (with the 200 longest rows).

| source | rows | mean | p50 | p99 | p99.9 | max | over 256 | over 512 |
|---|---|---|---|---|---|---|---|---|
| all | 8,239,111 | 30.5 | 28 | 88 | 128 | 1996 | 17 | 5 |
| Z-Image-Turbo | 4,585,332 | 30.8 | 29 | 84 | 125 | 1996 | 3 | 1 |
| FLUX.2-dev | 1,560,361 | 31.0 | 29 | 85 | 128 | 233 | 0 | 0 |
| jackyhate/text-to-image-2M | 1,925,032 | 27.0 | 26 | 82 | 124 | 831 | 8 | 2 |
| pexels | 117,376 | 58.3 | 53 | 131 | 165 | 514 | 5 | 2 |

(pixabay and the other sources are in the JSON.)

The longest row, 1996 tokens, is `synthetic_enhanced_prompt_square_resolution/…`: the ASCII-art caption that
`log/2026-11-07-fine-t2i-encode-and-overlap.md` rewrote in the random-resolution subset. The same prompt also sits in
the square subset, which that pass (2294 of 6356 tars) had not staged. The other rows above 256 tokens include
captioner loops (`复眼复眼…`, `伞伞伞…` in pexels rows), escaped bytes (`\xce\xa0…` in a jackyhate row) and binary strings
meant as rendered text, besides ordinary long descriptions.

## Defect patterns over the whole set

`scripts/exclude_captions.py /run/media/djg/vel-ssd/tiny-dit/set_128 …`, the `caption` column of all
8,239,111 rows against four patterns; a row can match more than one. Output
`/mnt/mars/ai/tiny-dit/probes/caption_defects_set128.json` (counts per source, 40 examples per pattern, the keys of
every flagged row).

| pattern | rows | where | what the examples are |
|---|---|---|---|
| CJK in the caption | 2,297 | pexels 954, jackyhate 407, pixabay 345, Z-Image-Turbo 293, unsplash_lite 199, FLUX.2-dev 99 | two kinds: text to be rendered, quoted (`"春"`, a proverb), which is wanted; and the captioner switching into Chinese mid-caption (`…blue sky.远处有几座低矮的山脉。`, `sand粒`), in the examples mostly the curated photo sources (not counted apart from quoted text) |
| a unit repeated 12+ times | 195 | jackyhate 186, pexels 5, Z-Image-Turbo 3, FLUX.2-dev 1 | jackyhate's short web-alt-text captions (`granny granny granny…`, `rico rico rico…`), captioner loops (`复眼复眼…`); the pattern also catches some legitimate runs (digits, dashes), not separated here |
| escaped bytes `\xNN` | 410 | jackyhate 410 | UTF-8 written as literal escapes: `saut\xc3\xa9`, `Pok\xc3\xa9mon`, `\xc2\xa32.7`; in the examples each escape run is valid UTF-8 (not checked over all rows) |
| a chat template | 3 | pexels 3 | `A high-resolution,�\nuser\nDescribe this image.` and two like it |
| empty | 0 | | |

Read: the defects are rare (about 2,900 rows, 0.035 %, before removing overlaps and the quoted-text CJK, which is
wanted), and of four kinds with different fixes: escapes decode mechanically, chat-template leaks and loops need the
caption rewritten or cut at the defect, and mid-caption Chinese needs a translation or a rewrite. The 16 rows longest after
the ASCII art, read at their end by hand (the JSON holds them), are all defects except the two binary-string prompts (279 tokens, text meant to be
rendered): the jackyhate DALL·E 3 rows (831, 705) are captioner loops (`'BUUUU…'`, `I'm a witch, …` decaying to
`w w w`), as are `RRRR…`, `0000…` and a pixabay tag list repeated in Chinese; the pexels rows stop at 512–514 tokens in
repeated CJK (`复眼复眼…`, `留留留…`, `模糊的模糊的…`), the captioner's length limit reached in a loop; three jackyhate rows
end in escaped bytes of Greek, Cyrillic and Tamil text. The length tail is the defect tail.

2,896 distinct rows match at least one pattern; their keys are `set_128/excluded_keys.txt`. Against the 200 longest
rows of the length probe: every row above 256 tokens is excluded except the two 279-token binary-string prompts, so
the longest caption streamed is 279 tokens and the 512 cut does not fire on this set.

## The cross attention's kernel

`scripts/probe_cross_attention.py MICRO_BATCH`, torch 2.15.0+cu132: one block's cross attention (12 query heads, 3 KV
heads, 64 queries) forward + backward, 5 warm-up and 20 timed repeats, as the trunk runs it (SDPA, captions padded,
boolean key mask, `enable_gqa`) and packed through `torch.nn.attention.varlen.varlen_attn` (the gather into the packed
layout included). Outputs `/mnt/mars/ai/tiny-dit/probes/cross_attention_varlen_mb{128,64}.txt`.

| captions in the micro-batch | micro-batch | padded ms | padded peak MiB | varlen ms | varlen peak MiB |
|---|---|---|---|---|---|
| 20–140 tokens | 128 | 1.26 | 126 | 1.48 | 166 |
| one row at 2000 | 128 | 20.75 | 1001 | 2.51 | 174 |
| 150–330 | 128 | 2.36 | 216 | 2.06 | 246 |
| 20–140 | 64 | 0.65 | 63 | 0.99 | 84 |
| one row at 2000 | 64 | 10.16 | 501 | 1.42 | 91 |
| 150–330 | 64 | 1.08 | 108 | 1.25 | 123 |

Outputs agree to 0.008 on random inputs (outputs of order 0.1–1). The masked call runs the cutlass memory-efficient
kernel (`fmha_cutlassF_…_sm80`); `varlen_attn` runs the flash kernel, GQA native, no mask. The "typical" case is
longer than `set_128`'s captions (mean 85 against 30.5).

Read:
- Per call, packing is within 0.35 ms either way on captions without an outlier, and at one 2000-token row it is
  8.3× faster and 5.8× smaller at micro-batch 128 (7.2× and 5.5× at 64).
- Per step, cross attention runs in 6 blocks per micro-batch: about 6 × 4 × 0.65 ≈ 16 ms of the 450 ms step at
  micro-batch 64 on ordinary captions.
- The attention call accounts for about half of the step-time cost of the long row (6 × 4 × 10.2 ≈ 245 ms of the 446 ms
  added at micro-batch 64). The rest is presumably the padded positions' share of `cross_kv` and the norms, which
  packing the attention call alone would not remove; packing the token states before `cross_kv` would.
