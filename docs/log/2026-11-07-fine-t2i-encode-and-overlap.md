# Fine-T2I: encoder, storage per row, caption overlap with dataset 1 (2026-11-07)

## Encoder

`scripts/encode_fine.py` turns each staged Parquet (`stage_fine.py`: bucket crops at the 256 px level, JPEG q90) into `/mnt/luna/tiny-dit/fine/encoded/<subset>/<tar>.parquet`: latents at 128 px (Lanczos halving of the crop) and 256 px as variable-size binary, the pooled 768-d embedding and the 512-d sequence of one caption, both captions as text. Smoke test on 48 curated rows: byte lengths match the bucket shapes, the pooled embedding matches a plain encode (max 9e-4), the sequence matches the hooked single-caption encode to bf16 rounding (0.08 on entries up to 51), and both latents decode to the source crop the right way round. 25.6 rows/s with the GPU shared with a training run; dataset 1 encoded at about 135 rows/s alone.

## Storage per row

Measured 145 KB per curated row with its `prompt` (54 tokens on average): about 88 KB of latents at an average bucket and 1 KB per token of sequence. Caption lengths in EmbeddingGemma tokens (first 300 rows of a tar):

| subset | prompt | enhanced_prompt |
|---|---|---|
| curated | 54 (max 132) | 208 (max 326) |
| synthetic, enhanced prompt | 33 (max 148) | 205 (max 325) |

The synthetic enhanced-prompt rows were rendered from the enhanced prompt, so their own caption is about 205 tokens, about 300 KB per row. With about 6.3 M rows (curated 168k, four synthetic subsets of about 1.2 to 1.5 M each), encoding each image's own caption is about 1.3 TB; the short prompt everywhere about 0.8 TB; no sequences (pooled only) about 0.55 TB. Free on 2026-11-07: luna 479 GB (about 110 GB of it still to be taken by staging), vel-ssd 535 GB, mars 110 GB.

## Caption overlap with dataset 1

Exact match after lower-casing and collapsing whitespace, against all 2,290,793 rows of dataset 1 (1,993,819 unique captions):

| subset (staged so far) | rows | prompt in dataset 1 | enhanced in dataset 1 | repeated prompts inside |
|---|---|---|---|---|
| curated | 168,424 | 0 | 0 | none |
| synthetic, enhanced prompt, random resolution | 1,179,729 | 0 | 0 | none |

No caption is shared, so there is no prompt-level duplicate to drop; an image-level duplicate (the same photograph in both) can only be found from the images. Dataset 1 itself repeats captions: 297k of its rows share a caption with another row.
