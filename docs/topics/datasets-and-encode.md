# Datasets and the encode

The question and its reads, moved here whole from `PLAN.md` and `FACTS.md` on 2026-11-11 so the main ledgers stay short. PLAN keeps the question's heading and points here.

## The question (PLAN B6)

**B6. The data sets and the encode**

Two sets were missing from the original plan: Spawning/PD12M and jasperai/monet, both world-knowledge images more than
text-to-image. Lunara is too small for pretraining but an option for fine-tuning later. Any set we pull gets 128 and
256 px latents plus the Gemma-class embedding and its token sequence. `/mnt/luna` has about 800 GB free but is an HDD,
so it holds the staging, and the shuffled set goes to `/mnt/mars` afterwards. The current data gets the same treatment,
as an overnight job (2026-11-05). Candidates in REFERENCES, Data: Fine-T2I, PD12M, MONET; `moonworks/lunara-aesthetic`
for the fine-tune set (see the FID and Lunara question in the sampling group).

Per set: 128 and 256 px latents, the EmbeddingGemma 2 pooled embedding (the 768-d vector trained on now) and its token
sequence, encoded and staged on `/mnt/luna` (HDD, ~800 GB free), the shuffled set then copied to `/mnt/mars` (107 GB
free on 2026-11-05). First: text-to-image-2M again, overnight. Sizes per 2M rows, from the current set: 128 px latents
32 GB, 256 px latents 128 GB; the token sequence unpadded without the prompt prefix at 1,536 B per token (768-d bf16):
the first encoded rows average 58 tokens, 90 KB, so about 170 GB per 1.9M rows (the 30-token mean estimated from 3,000
captions was low); at the 512-d states (1,024 B per token) about 115 GB, the figure for the shuffled set.

Decided 2026-11-05: a fused dataset of both sources, deduplicated later, both at 128 and 256 px with Lanczos; the
sequence for cross attention at 512, since it is projected down to 64 anyway, kept straightforward, with the prompt
prefix tokens dropped and unpadded. Permitted: deleting anything stale or no longer needed, and using
`/run/media/djg/vel-ssd` for space. The HDD read may be enough, so plan the async delay. EmbeddingGemma 2's residual stream is
512-d (24 layers); its `embedding_projection` (768x512, after the final norm, per token) lifts the token states to 768
before the mean pooling and normalize. The encode stores the projected 768-d states, which span only 512 dimensions;
the pooled embedding is 768-d. The prefix `title: none | text: ` is `<bos>` and 6 tokens;
its trailing space merges into the caption's first token. If the encoder gives 768, that is fine (2026-11-05). Running
since 2026-11-05 23:00: text-to-image-2M shards 0–46, then the Lunara eval set (`scripts/precompute.py`,
`scripts/precompute_lunara.py`), into `/mnt/luna/tiny-dit/precomputed/fused/`, log
`/mnt/luna/tiny-dit/precompute_2026-11-05.log`.

After the 512-d residual was re-verified (2026-11-06), the decision: redo only the sequence when the encode finishes,
since that is cheaper than restarting the encode. `scripts/reencode_sequence.py` re-encodes the captions of every fused shard (no images) to the 512-d
token states before the projection, into `fused/sequence512_<shard>.parquet` (key, sequence); it starts when the encode
exits, log `/mnt/luna/tiny-dit/reencode_sequence_2026-11-06.log`. The shuffle then joins on `key` and drops the 768-d
`sequence`. Open:
- Fine-T2I's images are above 1K and partly not square. Decided: no square crop; the images go into aspect buckets (token grids of about
  the same token count, the image resized to the grid's exact size), and the data is fused, deduplicated and shuffled in one
  pass once it is encoded. The sizes are good (below, Fine-T2I image sizes): nine aspect ratios, at most 5 % stretch on the
  synthetic sets at 128 px, more on the curated photos. Revised (2026-11-06): stretch is worse than a crop, so each image is scaled to cover its bucket with its aspect kept and only the overflow is cropped; the point was not to crop everything to 1:1. Open: the bucket set (a larger token count for less stretch), how the
  stream draws a bucket per batch, and one file per bucket (the latent width differs). The model already takes any grid.
- Fine-T2I's 2.3 TB at the 39 MB/s of the last download is about 16 h of download alone: several nights.

## Measurements

### Fine-T2I image sizes (B6, 2026-11-06)

Source: Hugging Face `ma-xu/fine-t2i`, file listing with sizes and the first tar of each of the five groups, read on `/mnt/luna/tiny-dit/probe_fine/` (about 1 GB; image headers only, sizes counted, nothing encoded). The groups: `curated` 192 tars, 279 GB, 1,024 images per tar; `synthetic_{enhanced,original}_prompt_{random,square}_resolution` 1,621 / 1,688 / 1,544 / 1,311 tars, 510 / 513 / 554 / 467 GB, 1,000 images per tar. In all, 6,363 files and 2.3 TB, about 6.16M synthetic pairs and 168k curated photos.

- The two square groups have six sizes: 512, 768, 1024, 1536, 2048 and 2560 px, all 1:1.
- The two random-resolution groups have 26 distinct sizes in 1,000 images, 9 aspect ratios: 1:1, 5:4, 4:5, 4:3, 3:4, 3:2, 2:3, 16:9 and 9:16. The json carries `image_aspect_ratio` and `image_resolution`.
- The curated photos have 589 distinct sizes in 1,024 images, camera sizes up to 6000 x 4000; 3:2 and 2:3 dominate (433 and 298 of 1,024 at the nearest 64-token grid), 4:3, 3:4 and 16:9 follow.
- Token grids (8 x 8 tokens is a 128 px square) of about 64 tokens with the best aspect, resized to the grid's exact size without a crop: the synthetic sets stretch by at most 5 % (1:1 0, 4:3 and 3:4 3.6 % and 3.7 %, 5:4 and 4:5 2.9 %, 16:9 and 9:16 3.1 % and 3.0 %, 3:2 and 2:3 4.8 % and 5.0 %); the sample of curated photos stretches by a median 4.8 %, at most 7.6 %.
