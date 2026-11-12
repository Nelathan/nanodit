"""Precompute the eval set (SPEC §8) from moonworks/lunara-art-eval: the 1000 images Moonworks Lunara rendered, one per prompt.

usage: precompute_lunara.py DIR

Downloads the 19 Parquet shards to DIR/data/lunara-art-eval/ and writes
DIR/precomputed/fused/lunara_art_eval.parquet in precompute.py's schema, keyed "<shard>/<row>".
"""
import io
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from PIL import Image

from precompute import load_encoders, write_shard

REPO = "moonworks/lunara-art-eval"
SHARDS = 19
MODEL = "Moonworks Lunara"


def lunara_rows(data_dir):
    """(key, image bytes, prompt) of MODEL's rows, in shard order."""
    for shard in range(SHARDS):
        path = hf_hub_download(REPO, f"data/train-{shard:05d}-of-{SHARDS:05d}.parquet", repo_type="dataset", local_dir=data_dir / "lunara-art-eval")
        rows = [row for row, model in enumerate(pq.read_table(path, columns=["model"])["model"].to_pylist()) if model == MODEL]
        table = pq.read_table(path, columns=["image", "prompt"]).take(rows)
        for row, image, prompt in zip(rows, table["image"].to_pylist(), table["prompt"].to_pylist()):
            size = Image.open(io.BytesIO(image["bytes"])).size
            assert size[0] == size[1], f"shard {shard} row {row} is {size}, not square"  # write_shard resizes without cropping
            yield f"{shard}/{row}", image["bytes"], prompt


if __name__ == "__main__":
    root = Path(sys.argv[1])
    vae, st = load_encoders()
    out_dir = root / "precomputed" / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    write_shard(lunara_rows(root / "data"), REPO, out_dir / "lunara_art_eval.parquet", vae, st)
