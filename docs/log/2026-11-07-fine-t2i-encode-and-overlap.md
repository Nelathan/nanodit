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

## Decisions (Daniel, 2026-11-07)

- Every subset encodes its short `prompt`, whole (the encoder has no token limit); the enhanced prompts stay as text and can be encoded later without the images. About 0.8 TB for all subsets.
- The exact caption check being clean, the two datasets are taken to share no image; no image-level dedup.
- If space runs short: old checkpoints can go, and more disk may come from elsewhere.

## Defective prompts rewritten (2026-11-07)

Of the prompts staged by 2294 of 6356 tars (curated, enhanced-prompt random resolution, part of enhanced-prompt square), 20 are longer than 1000 characters; 17 are ordinary long descriptions and stay. Three were defects and were rewritten in the staged Parquet, the encoded file deleted to be encoded again (the source text is still in the Fine-T2I dataset):

| key | was | now |
|---|---|---|
| `synthetic_enhanced_prompt_random_resolution/371c6a0c-030c-48bd-b9a1-45829100abea` (train-000378, row 308) | 4148 chars: a sentence, then 4000 chars of ASCII-art underscores and slashes (1997 tokens; it ran the text encoder out of memory at batch 32) | the sentence alone: "A digital art piece of a fantasy world with mountains and valleys displaying long lines of ASCII art." |
| `curated/4421493` (train-000108, row 721) | 2476 chars: one caption and near-copies joined by a leaked chat-template `assistant\n` | the first caption |
| `curated/4421498` (train-000189, row 704) | 2372 chars: the same leak | the first caption |

Text batches are now bounded by padded tokens (8192), so a long prompt no longer needs cutting to fit; nothing else is truncated.
