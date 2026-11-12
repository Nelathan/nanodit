"""Encode the staged Fine-T2I Parquets (`stage_fine.py`) into Primal 2x2 latents and EmbeddingGemma 2 encodings (GPU).

usage: encode_fine.py DIR SUBSET [SUBSET ...]

For every DIR/staged/<subset>/<tar>.parquet (`stage_fine.py DIR`) writes DIR/encoded/<subset>/<tar>.parquet with columns
  key              string
  source           string                   the image's generator (`unsplash_lite` for the curated photographs)
  bucket           int8                     index into stage_fine.BUCKETS
  caption          string                   the caption encoded below: the dataset's `prompt`, whole (no token limit)
  prompt           string                   both captions of the dataset, so the enhanced one can be encoded later
  enhanced_prompt  string
  aesthetic        float32
  embedding        fixed_size_binary(1536)  768 bf16, the normalized pooled embedding, as precompute.py
  sequence         binary                   bf16 [n, 512], C-contiguous: the token states before the encoder's
                                            768x512 projection, prompt prefix dropped, unpadded, as precompute.py
  latent_128       binary                   bf16 [2, h/4, w/4], C-contiguous, from a Lanczos halving of the staged crop
  latent_256       binary                   bf16 [2, h/2, w/2], C-contiguous, from the staged crop (h, w = BUCKETS[bucket])
The latents are variable-size binary because the buckets differ in shape; the shuffle splits rows by bucket. A
Parquet whose output exists is skipped, so a killed run resumes.
"""
import io
import os
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
from PIL import Image

from precompute import DEVICE, TEXT_DIM, bf16_bytes, bf16_column, encode_text, load_encoders

ROW_GROUP = 1024
VAE_BATCH = 32
TEXT_TOKENS = 8192  # padded tokens per text batch: a few captions run to thousands of tokens, and 32 of them overflow the GPU
SCHEMA = pa.schema([
    ("key", pa.string()), ("source", pa.string()), ("bucket", pa.int8()), ("caption", pa.string()),
    ("prompt", pa.string()), ("enhanced_prompt", pa.string()), ("aesthetic", pa.float32()),
    ("embedding", pa.binary(TEXT_DIM * 2)), ("sequence", pa.binary()), ("latent_128", pa.binary()), ("latent_256", pa.binary()),
])


def pixels(jpg: bytes, half: bool) -> torch.Tensor:
    """[3, h, w] in [-1, 1]; `half` makes it a Lanczos resize to half the size."""
    image = Image.open(io.BytesIO(jpg)).convert("RGB")
    if half:
        image = image.resize((image.width // 2, image.height // 2), Image.LANCZOS)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1


def latents(vae, jpgs: list[bytes], buckets: list[int], half: bool) -> list[bytes]:
    """Each image's latent as bf16 bytes, encoded in batches of one bucket."""
    out: list[bytes | None] = [None] * len(jpgs)
    for b in set(buckets):
        rows = [i for i, x in enumerate(buckets) if x == b]
        for start in range(0, len(rows), VAE_BATCH):
            batch = rows[start : start + VAE_BATCH]
            z = vae.encode(torch.stack([pixels(jpgs[i], half) for i in batch]).to(DEVICE))
            for i, zi in zip(batch, z):
                out[i] = bf16_bytes(zi)
    return out


def encode_captions(st, captions: list[str]):
    """encode_text on batches of similar length holding at most TEXT_TOKENS padded tokens, returned in caption order."""
    lengths = [len(ids) for ids in st.tokenizer(captions)["input_ids"]]
    batches, batch = [], []
    for i in sorted(range(len(captions)), key=lengths.__getitem__):
        if batch and (len(batch) + 1) * lengths[i] > TEXT_TOKENS:
            batches.append(batch)
            batch = []
        batch.append(i)
    batches.append(batch)
    pooled, sequences = [None] * len(captions), [None] * len(captions)
    for batch in batches:
        embedding, tokens = encode_text(st, [captions[i] for i in batch])
        for i, e, s in zip(batch, embedding, tokens):
            pooled[i], sequences[i] = e, s
    return torch.stack(pooled), sequences


def encode(src: Path, out: Path, subset: str, vae, st) -> int:
    table = pq.read_table(src)
    tmp, t0 = out.with_name(out.name + ".tmp"), time.time()
    with torch.inference_mode(), pq.ParquetWriter(tmp, SCHEMA, compression="none") as writer:
        for start in range(0, len(table), ROW_GROUP):
            chunk = table.slice(start, ROW_GROUP)
            jpgs, buckets, captions = chunk["jpg"].to_pylist(), chunk["bucket"].to_pylist(), chunk["prompt"].to_pylist()
            embedding, sequences = encode_captions(st, captions)
            writer.write_table(pa.table({
                "key": chunk["key"], "source": chunk["source"], "bucket": chunk["bucket"], "caption": captions,
                "prompt": chunk["prompt"], "enhanced_prompt": chunk["enhanced_prompt"], "aesthetic": chunk["aesthetic"],
                "embedding": bf16_column(embedding, TEXT_DIM * 2), "sequence": [bf16_bytes(s) for s in sequences],
                "latent_128": latents(vae, jpgs, buckets, half=True), "latent_256": latents(vae, jpgs, buckets, half=False),
            }, schema=SCHEMA))
            print(f"{subset}/{out.name} {start + len(chunk)} rows, {(start + len(chunk)) / (time.time() - t0):.1f} rows/s", flush=True)
    os.replace(tmp, out)
    return len(table)


if __name__ == "__main__":
    vae, st = load_encoders()
    root = Path(sys.argv[1])
    for subset in sys.argv[2:]:
        (root / "encoded" / subset).mkdir(parents=True, exist_ok=True)
        for src in sorted((root / "staged" / subset).glob("*.parquet")):
            out = root / "encoded" / subset / src.name
            if not out.exists():
                encode(src, out, subset, vae, st)
