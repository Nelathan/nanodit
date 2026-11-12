"""Stage Fine-T2I (ma-xu/fine-t2i) for encoding: one small Parquet per source tar, images cropped to aspect buckets.

usage: stage_fine.py DIR SUBSET [SUBSET ...]

SUBSET is a top-level folder of the dataset repo (`synthetic_original_prompt_square_resolution`, `curated`, ...). Each
tar (0.3 to 3.2 GB) downloads to DIR/raw, becomes DIR/staged/<subset>/<tar name>.parquet and is deleted; the next tar downloads
while one is processed. A tar whose Parquet exists is skipped, so a killed run resumes.

An image goes to the aspect bucket nearest its own (log ratio), is scaled until it covers the bucket and is cropped at
the centre: only overflow is cut, nothing is stretched. Bucket sizes are those of the 256 px stage (the 128 px stage
halves them); the staged JPEG keeps them, so the encoder can make both resolutions from it. Rows keep both captions.

Parquet columns: key, bucket (index into BUCKETS), jpg, prompt, enhanced_prompt, source, aesthetic.
"""
import io
import json
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from math import log
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
from PIL import Image

REPO = "ma-xu/fine-t2i"
BUCKETS = [(256, 256), (224, 288), (288, 224), (224, 320), (320, 224), (192, 352), (352, 192)]  # (height, width) in image pixels
QUALITY = 90
WORKERS = 12


def bucket_of(width: int, height: int) -> int:
    return min(range(len(BUCKETS)), key=lambda i: abs(log(height / width) - log(BUCKETS[i][0] / BUCKETS[i][1])))


def crop(jpg: bytes) -> tuple[int, bytes]:
    image = Image.open(io.BytesIO(jpg))
    b = bucket_of(*image.size)
    h, w = BUCKETS[b]
    scale = max(h / image.height, w / image.width)
    image.draft("RGB", (round(image.width * scale), round(image.height * scale)))  # the JPEG decoder skips ahead to a size at or above the target
    image = image.convert("RGB")
    scale = max(h / image.height, w / image.width)
    image = image.resize((max(w, round(image.width * scale)), max(h, round(image.height * scale))), Image.LANCZOS)
    left, top = (image.width - w) // 2, (image.height - h) // 2
    out = io.BytesIO()
    image.crop((left, top, left + w, top + h)).save(out, "JPEG", quality=QUALITY)
    return b, out.getvalue()


def samples(path: Path):
    """Yields (key, jpg bytes, metadata) in tar order."""
    pending: dict[str, dict] = {}
    with tarfile.open(path, "r|") as tar:
        for m in tar:
            key, ext = m.name.rsplit(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if "jpg" in pending[key] and "json" in pending[key]:
                part = pending.pop(key)
                yield key, part["jpg"], json.loads(part["json"])


def stage(path: Path, out: Path, pool: ProcessPoolExecutor, subset: str) -> int:
    rows = list(samples(path))
    cropped = list(pool.map(crop, [jpg for _, jpg, _ in rows], chunksize=16))
    table = pa.table({
        "key": [f"{subset}/{key}" for key, _, _ in rows],
        "bucket": pa.array([b for b, _ in cropped], pa.int8()),
        "jpg": [jpg for _, jpg in cropped],
        "prompt": [meta["prompt"] for _, _, meta in rows],
        "enhanced_prompt": [meta.get("enhanced_prompt") for _, _, meta in rows],
        "source": [meta.get("image_generator") for _, _, meta in rows],
        "aesthetic": pa.array([meta.get("aesthetic_predictor_v_2_5_score") for _, _, meta in rows], pa.float32()),
    })
    pq.write_table(table, out.with_suffix(".tmp"), compression="none")
    out.with_suffix(".tmp").rename(out)
    return len(rows)


def main(root: Path, subsets: list[str]) -> None:
    files = sorted(f for f in HfApi().list_repo_files(REPO, repo_type="dataset") if f.endswith(".tar"))
    todo = [(f, root / "staged" / Path(f).parent / (Path(f).stem + ".parquet")) for f in files if Path(f).parts[0] in subsets]
    todo = [(f, out) for f, out in todo if not out.exists()]
    for _, out in todo:
        out.parent.mkdir(parents=True, exist_ok=True)

    def fetch(name: str) -> Path:
        return Path(hf_hub_download(REPO, name, repo_type="dataset", local_dir=root / "raw"))

    with ThreadPoolExecutor(1) as downloads, ProcessPoolExecutor(WORKERS) as pool:
        pending = downloads.submit(fetch, todo[0][0]) if todo else None
        for i, (name, out) in enumerate(todo):
            path = pending.result()
            pending = downloads.submit(fetch, todo[i + 1][0]) if i + 1 < len(todo) else None
            n = stage(path, out, pool, Path(name).parts[0])
            path.unlink()
            print(f"{name}: {n} rows ({i + 1}/{len(todo)})", flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]), sys.argv[2:])
