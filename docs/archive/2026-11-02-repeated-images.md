# Repeated images: each image once

2026-11-02. Closes the HANDOVER question on the repeated images in shards 20–34.

Facts weighed (FACTS, Sample and shard, Repeats): shards 20–34 (`llava_next_captioned_700K`) hold 738,524 rows but
378,362 distinct latents, about two rows per image. An image comes in runs of consecutive keys, 1 to 7 long, with the
same or a slightly different caption, and recurs later in the source. The other 32 shards hold 1,552,269 rows and
1,547,055 distinct latents. The options were to keep every row, to drop repeated (latent, caption)
rows, or to keep each image once.

Decided: repeated images are dropped.

`scripts/shuffle.py` drops every row whose latent repeats an earlier row's, in shard order, so an image keeps the
caption of its first row. Over shards 0–46 that leaves 1,925,416 of 2,290,793 rows. SPEC, Training data, states the
rule.
