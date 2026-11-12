"""Shuffle precomputed shard Parquets into one Parquet, rows in uniformly random order (SPEC Precompute).

usage: shuffle.py OUT RESOLUTION SHARD [SHARD ...]

SHARD is a fused shard of `precompute.py`. OUT has its columns key, source, caption, embedding, sequence and
latent_<RESOLUTION>; the other latent stays behind.

Two passes, so no more than a few thousand rows of a shard or one bucket are in memory. Pass 1 deals each shard's rows into BUCKETS
temporary files, each row to a uniformly random one; pass 2 permutes each bucket in memory and appends it to
OUT. A random bucket per row, then a random order inside each bucket, is a uniformly random permutation of all
rows. OUT is written as OUT.tmp and renamed when complete. It needs about the shards' size again in free disk: the
buckets, then OUT as each bucket is appended and deleted. A new shard means running it again over all shards.
Each image is kept once: pass 1 drops a row whose latent bytes repeat an earlier row's, in shard order, so an image
keeps the caption of its first row.
"""
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

BUCKETS = 128  # about 150 GB of rows at 128 px make buckets of about 1.2 GB, under the 2 GB a binary column holds in one chunk
ROW_GROUP = 1024
TAKE_ROWS = 4096
SEED = 0


def first_sightings(latents, seen: set[bytes]) -> np.ndarray:
    """Mask of the rows whose latent bytes are not in `seen`, adding them to it."""
    array = latents.combine_chunks()
    width = array.type.byte_width
    raw = np.frombuffer(array.buffers()[1], dtype=np.uint8, count=len(array) * width, offset=array.offset * width)
    keep = np.zeros(len(array), dtype=bool)
    for i, row in enumerate(raw.reshape(len(array), width)):
        digest = hashlib.blake2b(row, digest_size=16).digest()
        keep[i] = digest not in seen
        seen.add(digest)
    return keep


def shuffle(shards: list[str], resolution: int, out: Path) -> None:
    rng = np.random.default_rng(SEED)
    latent = f"latent_{resolution}"
    columns = ["key", "source", "caption", "embedding", latent, "sequence"]
    fused = pq.read_schema(shards[0])
    schema = pa.schema([fused.field(c) for c in columns])
    parts = [out.with_name(f"{out.name}.bucket{b:03d}") for b in range(BUCKETS)]
    writers = [pq.ParquetWriter(part, schema, compression="none") for part in parts]
    seen: set[bytes] = set()
    rows = 0
    for shard in shards:
        dealt = dropped = 0
        for batch in pq.ParquetFile(shard).iter_batches(TAKE_ROWS, columns=columns):  # a whole 256 px shard is 3.3 GB of latents: read it in pieces
            part = pa.Table.from_batches([batch])
            assert part.schema.equals(schema), shard
            keep = first_sightings(part[latent], seen)
            bucket = rng.integers(BUCKETS, size=len(part))
            for b, writer in enumerate(writers):
                writer.write_table(part.take(np.flatnonzero(keep & (bucket == b))))
            dealt, dropped = dealt + keep.sum(), dropped + len(part) - keep.sum()
        rows += dealt
        print(f"{shard}: {dealt} rows dealt, {dropped} repeated images dropped", flush=True)
    for writer in writers:
        writer.close()
    tmp = out.with_name(out.name + ".tmp")
    with pq.ParquetWriter(tmp, schema, compression="none") as writer:
        for part in parts:
            table = pq.read_table(part)
            writer.write_table(table.take(rng.permutation(len(table))), row_group_size=ROW_GROUP)
            part.unlink()
    assert pq.ParquetFile(tmp).metadata.num_rows == rows
    os.replace(tmp, out)
    print(f"{out}: {rows} rows", flush=True)


if __name__ == "__main__":
    shuffle(sys.argv[3:], int(sys.argv[2]), Path(sys.argv[1]))
