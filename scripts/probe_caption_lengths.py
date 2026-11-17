"""Caption lengths of a training set in token states (the `sequence` column's bytes / 1024, bf16 × 512): the
distribution per source and bucket, and the longest rows with their captions.

usage: probe_caption_lengths.py SET_DIR OUT.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.compute as pc
import pyarrow.parquet as pq

BYTES_PER_TOKEN = 512 * 2
THRESHOLDS = (128, 256, 384, 512, 768, 1024, 2048)
LONGEST = 200


def main(set_dir: str, out: str) -> None:
    lengths, sources, where = [], [], []
    for path in sorted(Path(set_dir).glob("*.parquet")):
        f = pq.ParquetFile(path)
        for group in range(f.metadata.num_row_groups):
            table = f.read_row_group(group, columns=["sequence", "source"])
            n = pc.binary_length(table["sequence"]).to_numpy() // BYTES_PER_TOKEN
            lengths.append(n)
            sources += table["source"].to_pylist()
            where += [(path.name, group, row) for row in range(len(n))]
        print(path.name, flush=True)
    lengths = np.concatenate(lengths)
    sources = np.array(sources)

    def stats(n: np.ndarray) -> dict:
        return dict(rows=len(n), mean=float(n.mean()), p50=int(np.percentile(n, 50)), p99=int(np.percentile(n, 99)), p999=int(np.percentile(n, 99.9)), max=int(n.max()), over={t: int((n > t).sum()) for t in THRESHOLDS})

    report = dict(all=stats(lengths), by_source={s: stats(lengths[sources == s]) for s in np.unique(sources)})
    longest = []
    for i in np.argsort(lengths)[::-1][:LONGEST]:
        name, group, row = where[i]
        r = pq.ParquetFile(Path(set_dir) / name).read_row_group(group, columns=["key", "source", "caption"]).slice(row, 1).to_pylist()[0]
        longest.append(dict(tokens=int(lengths[i]), file=name) | r)
    report["longest"] = longest
    Path(out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
    print(json.dumps({k: report[k] for k in ("all", "by_source")}, indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:])
