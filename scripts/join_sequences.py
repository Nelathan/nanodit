"""Join a fused shard with its 512-d token sequences, rows in the shard's order: an eval set in `shuffle.py`'s schema.

usage: join_sequences.py OUT RESOLUTION SHARD
"""
import sys

import pyarrow.parquet as pq

from shuffle import read_shard


def join(shard: str, resolution: int, out: str) -> None:
    table = read_shard(shard, resolution, ["key", "source", "caption", "embedding", f"latent_{resolution}"])
    pq.write_table(table, out, compression="none")
    print(f"{out}: {len(table)} rows", flush=True)


if __name__ == "__main__":
    join(sys.argv[3], int(sys.argv[2]), sys.argv[1])
