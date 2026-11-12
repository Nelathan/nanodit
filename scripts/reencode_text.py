"""Re-encode the `embedding` and `sequence` columns of precomputed Parquets from their `caption` column (GPU).

usage: reencode_text.py FILE [FILE ...]

A `.safetensors` FILE is the null embedding, written again by `precompute.write_null_embedding`. A Parquet is rewritten
in place: every other column and the row groups are kept as they are, the two text columns are
encoded again by `precompute.encode_text` (no prompt prefix; the token states without <bos> and <eos>). The new file
is written next to the old one and renamed over it, so a killed run leaves every file whole, in one encoding or the
other.
"""
import sys
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

from encode_fine import encode_captions
from precompute import DEVICE, TEXT_DIM, bf16_bytes, bf16_column, load_text_encoder, write_null_embedding


def reencode(path: Path, st) -> int:
    source = pq.ParquetFile(path)
    schema = source.schema_arrow
    tmp = path.with_name(path.name + ".tmp")
    rows = 0
    with pq.ParquetWriter(tmp, schema, compression="none") as writer:
        for group in range(source.num_row_groups):
            table = source.read_row_group(group)
            embedding, sequences = encode_captions(st, table["caption"].to_pylist())
            table = table.set_column(schema.get_field_index("embedding"), schema.field("embedding"), bf16_column(embedding, TEXT_DIM * 2))
            table = table.set_column(schema.get_field_index("sequence"), schema.field("sequence"), pa.array([bf16_bytes(s) for s in sequences], pa.binary()))
            writer.write_table(table, row_group_size=len(table))
            rows += len(table)
    tmp.rename(path)
    return rows


def main(paths: list[str]):
    st = load_text_encoder(DEVICE)
    for path in map(Path, paths):
        t0 = time.time()
        if path.suffix == ".safetensors":
            write_null_embedding(st, path)
            print(f"{path}: the null embedding", flush=True)
            continue
        rows = reencode(path, st)
        print(f"{path}: {rows} rows in {time.time() - t0:.0f} s ({rows / (time.time() - t0):.0f} rows/s)", flush=True)


if __name__ == "__main__":
    with torch.inference_mode():
        main(sys.argv[1:])
