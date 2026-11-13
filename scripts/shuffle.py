"""Shuffle the precomputed Parquets of both datasets into one training set per resolution: one Parquet per aspect
bucket, its rows in uniformly random order (SPEC §1).

usage: shuffle.py OUT_DIR RESOLUTION FILE [FILE ...]

FILE is a fused shard of `precompute.py` (square images) or an encoded Fine-T2I Parquet of `encode_fine.py` (a
`bucket` column into `stage_fine.BUCKETS`, sizes at 256 px). OUT_DIR/<h>x<w>.parquet holds every row whose image is
h x w pixels at RESOLUTION, with the columns key, source, caption, embedding, sequence and latent_<RESOLUTION>, the
latent of fixed size within the file; the other latent and Fine-T2I's spare captions stay behind.

Each image is kept once: a row whose latent bytes repeat an earlier row's, in FILE order, is dropped, so an image
keeps the caption of its first row.

Two passes, so fewer than 2 x DEAL_ROWS input rows and one part are in memory. Pass 1 counts each bucket's rows and
gives it parts of about PART_ROWS rows, then deals every row to a uniformly random part of its bucket; pass 2
permutes each part in memory and appends it to its bucket's file. A random part per row, then a random order inside
each part, is a uniformly random permutation of the bucket's rows. Each file is written as .tmp and renamed when
complete; a bucket left without rows by the dedup gets no file. It needs the output's size in free disk, plus one
part. New data means running it again over all FILEs.
"""
import hashlib
import math
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from stage_fine import BUCKETS as FINE_BUCKETS

PART_ROWS = 16384  # about 1 GB of sequences and latents at 128 px, the part pass 2 holds in memory
DEAL_ROWS = 32768  # input rows dealt at once, so a part's row groups hold tens of rows, not a few
ROW_GROUP = 1024
SEED = 0
LATENT_CHANNELS = 2
COLUMNS = ["key", "source", "caption", "embedding", "sequence"]


def image_sizes(path: str, resolution: int, rows: pa.Table | None = None) -> np.ndarray:
    """[n, 2] (height, width) in pixels at `resolution` of the rows of `path` (or of `rows`, read from it): square
    for a fused shard, the bucket's for a Fine-T2I Parquet."""
    if "bucket" not in pq.read_schema(path).names:
        n = pq.ParquetFile(path).metadata.num_rows if rows is None else len(rows)
        return np.full((n, 2), resolution)
    buckets = (pq.read_table(path, columns=["bucket"]) if rows is None else rows)["bucket"].to_numpy()
    return np.array(FINE_BUCKETS)[buckets] * resolution // 256


def first_sightings(latents: pa.ChunkedArray, seen: set[bytes]) -> np.ndarray:
    """Mask of the rows whose latent bytes are not in `seen`, adding them to it."""
    keep = np.zeros(len(latents), dtype=bool)
    for i, row in enumerate(latents.to_pylist()):
        digest = hashlib.blake2b(row, digest_size=16).digest()
        keep[i] = digest not in seen
        seen.add(digest)
    return keep


def shuffle(files: list[str], resolution: int, out_dir: Path) -> None:
    rng = np.random.default_rng(SEED)
    latent = f"latent_{resolution}"
    counts = Counter()
    for f in files:
        counts.update(map(tuple, image_sizes(f, resolution).tolist()))
    fields = pq.read_schema(files[0])
    out_dir.mkdir(parents=True, exist_ok=True)
    writers, schemas = {}, {}
    for (h, w), n in sorted(counts.items()):
        schema = pa.schema([fields.field(c) for c in COLUMNS] + [pa.field(latent, pa.binary(LATENT_CHANNELS * (h // 2) * (w // 2) * 2))])
        schemas[h, w] = schema
        part_schema = schema.set(schema.get_field_index("sequence"), pa.field("sequence", pa.large_binary()))  # a part's sequences pass 2 GB
        writers[h, w] = [pq.ParquetWriter(out_dir / f"{h}x{w}.part{p:04d}", part_schema, compression="none") for p in range(math.ceil(n / PART_ROWS))]
    seen: set[bytes] = set()
    dealt = Counter()

    def deal(tables: list[pa.Table], sizes: np.ndarray) -> None:
        table = pa.concat_tables(tables).combine_chunks()  # take on a chunked column copies the whole column per call
        keep = first_sightings(table[latent], seen)
        for size, size_writers in writers.items():
            mine = keep & (sizes == size).all(1)
            part = rng.integers(len(size_writers), size=len(table))
            for p, writer in enumerate(size_writers):
                rows = table.take(np.flatnonzero(mine & (part == p)))
                writer.write_table(pa.table([rows[c] for c in COLUMNS] + [pc.cast(rows[latent], writer.schema.field(latent).type)], schema=writer.schema))
            dealt[size] += int(mine.sum())
        print(f"{sum(dealt.values())} rows dealt; {len(table) - int(keep.sum())} of {len(table)} dropped as repeated images", flush=True)

    tables, sizes = [], []
    for f in files:
        source = pq.ParquetFile(f)
        extra = ["bucket"] if "bucket" in source.schema_arrow.names else []
        for batch in source.iter_batches(DEAL_ROWS, columns=COLUMNS + [latent] + extra):
            table = pa.Table.from_batches([batch])
            sizes.append(image_sizes(f, resolution, table))
            tables.append(pa.table([pc.cast(table[c], pa.large_binary()) if c == "sequence" else table[c] for c in COLUMNS] + [pc.cast(table[latent], pa.binary())], names=COLUMNS + [latent]))  # one schema for both datasets; large_binary, as 32768 rows of sequences pass 2 GB
            if sum(map(len, tables)) >= DEAL_ROWS:
                deal(tables, np.concatenate(sizes))
                tables, sizes = [], []
    if tables:
        deal(tables, np.concatenate(sizes))
    for size_writers in writers.values():
        for writer in size_writers:
            writer.close()

    for (h, w), size_writers in writers.items():
        out = out_dir / f"{h}x{w}.parquet"
        tmp = out.with_name(out.name + ".tmp")
        with pq.ParquetWriter(tmp, schemas[h, w], compression="none") as writer:
            for p in range(len(size_writers)):
                part = out_dir / f"{h}x{w}.part{p:04d}"
                table = pq.read_table(part)
                table = table.take(rng.permutation(len(table)))
                for start in range(0, len(table), ROW_GROUP):
                    group = table.slice(start, ROW_GROUP)
                    writer.write_table(pa.table([pc.cast(group[c], schemas[h, w].field(c).type) for c in group.column_names], schema=schemas[h, w]))
                part.unlink()
        assert pq.ParquetFile(tmp).metadata.num_rows == dealt[h, w]
        if not dealt[h, w]:
            tmp.unlink()
            continue
        os.replace(tmp, out)
        print(f"{out}: {dealt[h, w]} rows", flush=True)


if __name__ == "__main__":
    shuffle(sys.argv[3:], int(sys.argv[2]), Path(sys.argv[1]))
