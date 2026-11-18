"""The held-out eval set of a training set (SPEC §1, §8): N rows of one Fine-T2I subset and generator, the N with the
smallest SHA-256 of their key, so the same rows are picked from any set built from the same data. Their rows go to
OUT.parquet in the set's columns, scored like the Lunara eval set; their keys go to the set's held-out list,
`SET_DIR/heldout_keys.txt` (`nanodit.data.HELDOUT`), which it replaces and which the stream never draws. Rows in the
exclusion list are not picked. The subset is square, so every row is in one bucket file and the set reads as one
tensor. Run it after every shuffle.

usage: holdout.py SET_DIR N OUT.parquet
"""
import hashlib
import heapq
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from nanodit.data import EXCLUDED, HELDOUT

SUBSET = "synthetic_enhanced_prompt_square_resolution/"  # key prefix: Fine-T2I's square images, LLM-enhanced prompts
SOURCE = "FLUX.2-dev"  # the full model of the two generators; the other, Z-Image-Turbo, is distilled


def main(set_dir: str, n: str, out: str) -> None:
    excluded = frozenset((Path(set_dir) / EXCLUDED).read_text().split())
    candidates = []  # (digest, file, row group, row)
    for path in sorted(Path(set_dir).glob("*.parquet")):
        f = pq.ParquetFile(path)
        for group in range(f.metadata.num_row_groups):
            table = f.read_row_group(group, columns=["key", "source"])
            for row, (key, source) in enumerate(zip(table["key"].to_pylist(), table["source"].to_pylist())):
                if key.startswith(SUBSET) and source == SOURCE and key not in excluded:
                    candidates.append((hashlib.sha256(key.encode()).digest(), path, group, row))
        print(path.name, len(candidates), flush=True)
    picked = heapq.nsmallest(int(n), candidates)
    assert len(picked) == int(n), f"{len(picked)} candidate rows"
    assert len({path for _, path, _, _ in picked}) == 1, "picked rows in more than one bucket"
    path = picked[0][1]
    f = pq.ParquetFile(path)
    tables = []
    for group in sorted({g for _, _, g, _ in picked}):
        rows = [r for _, _, g, r in picked if g == group]
        tables.append(f.read_row_group(group).take(rows))
    table = pa.concat_tables(tables)
    order = sorted(range(len(table)), key=lambda i: hashlib.sha256(table["key"][i].as_py().encode()).digest())
    pq.write_table(table.take(order), out)
    (Path(set_dir) / HELDOUT).write_text("".join(f"{k}\n" for k in sorted(table["key"].to_pylist())))
    print(len(table), "rows from", path.name, "to", out)


if __name__ == "__main__":
    main(*sys.argv[1:])
