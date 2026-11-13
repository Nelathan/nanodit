# One training set of both datasets, per aspect bucket; a micro-batch is one bucket

2026-11-13. Closes how Fine-T2I joins training, how its buckets are streamed, and the trial set.

Decided: text-to-image-2M and all five Fine-T2I subsets are merged into one training set per resolution
(`scripts/shuffle.py`): one Parquet per aspect bucket, its rows in uniformly random order, each image once across both
datasets (an exact repeat of the latent is dropped, the first row kept, text-to-image-2M first). Trials and long runs
train on the same set; the separate 16-shard trial set is deleted once the set exists, and the reference is run again
on it. Each micro-batch is one bucket, drawn with probability proportional to its rows, so a batch mixes buckets; the
micro-batches of a batch weigh by their tokens, so the batch loss is the mean over every latent element. Within a
bucket the row groups are dealt into new random blocks every pass. The order follows the run's seed (SPEC §1).
No intermediate copy is kept: a staged Fine-T2I tar is deleted once encoded. No off-site backup: a lost source is
pulled again.

Reasons: a uniformly shuffled file makes any block a uniform sample of every partition, so no batch is drawn from
one partition; the per-pass dealing of row groups makes a later epoch differ from the first in which rows share a
micro-batch, while the seed keeps runs comparable. Shapes differ between buckets, so a bucket per micro-batch is the
finest mix the batch can hold; weighing by tokens keeps every latent element of the batch at the same weight. A
separate trial set duplicates the training data on a full disk and makes trials read other data than long runs.
Only byte-identical latents are found as repeats: the two datasets share no caption (log 2026-11-07, Fine-T2I
overlap), and the same picture cropped differently would need an image-embedding comparison, not wanted for now.
