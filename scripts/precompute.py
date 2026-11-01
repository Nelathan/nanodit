"""Precompute Primal 2x2 latents and EmbeddingGemma 2 embeddings for text-to-image-2M data_512_2M shards (GPU).

usage: precompute.py RES SHARD [SHARD ...]

Writes /mnt/mars/ai/tiny-dit/precomputed/<RES>/data_<SHARD>.parquet with columns
  key        string
  caption    string
  embedding  fixed_size_binary(1536)   768 bf16, L2-normalized fp32 embedding rounded to bf16
  latent     fixed_size_binary(...)    bf16 [2, RES/2, RES/2], C-contiguous
then deletes the shard tar; the next shard's tar downloads while one encodes. A shard whose parquet exists
is skipped. Embeddings are copied from another resolution's parquet of the same shard when there is one.
Also writes precomputed/null_embedding.safetensors (key `null`, fp32 [768]), the embedding of the empty
caption, when it is missing.
"""
import io
import json
import os
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from diffusers import AutoModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

ROOT = Path("/mnt/mars/ai/tiny-dit")
REPO = "jackyhate/text-to-image-2M"
PREFIX = "title: none | text: "
ROW_GROUP = 1024
DEVICE = torch.device("cuda")  # cuDNN's TF32 convolutions stay on: 2x the rate, 0.25% on a latent (FACTS, GPU precompute)


def samples(tar_path):
    """Yields (key, jpg bytes, caption) in tar order."""
    pending = {}
    with tarfile.open(tar_path, "r|") as tar:
        for m in tar:
            key, ext = m.name.rsplit(".", 1)
            pending.setdefault(key, {})[ext] = tar.extractfile(m).read()
            if len(pending[key]) == 2:
                s = pending.pop(key)
                yield key, s["jpg"], json.loads(s["json"])["prompt"]


def fetch(name):
    """The shard's tar path, downloading it unless it is on disk."""
    tar_path = ROOT / "data" / "data_512_2M" / f"{name}.tar"
    if not tar_path.exists():
        hf_hub_download(REPO, f"data_512_2M/{name}.tar", repo_type="dataset", local_dir=ROOT / "data")
    return tar_path


def write_null_embedding(st, path):
    """The null embedding's init (SPEC §5): the encoder's embedding of the empty caption, under the same prefix."""
    save_file({"null": st.encode([""], prompt=PREFIX, convert_to_tensor=True)[0].contiguous()}, path)


def bf16_column(t, width):
    """[N, ...] float tensor -> fixed_size_binary column of the rows as bf16."""
    raw = t.bfloat16().cpu().contiguous().view(torch.int16).numpy().tobytes()
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(width), len(t), [None, pa.py_buffer(raw)])


def write_shard(tar_path, out_path, res, vae, st, cached_path):
    schema = pa.schema([
        ("key", pa.string()), ("caption", pa.string()),
        ("embedding", pa.binary(768 * 2)), ("latent", pa.binary(2 * (res // 2) ** 2 * 2)),
    ])
    cached = pq.read_table(cached_path, columns=["key", "caption", "embedding"]) if cached_path else None
    tmp = out_path.with_name(out_path.name + ".tmp")
    rows, t0 = 0, time.time()
    chunk = []

    def flush():
        nonlocal rows
        keys, jpgs, captions = zip(*chunk)
        x = torch.from_numpy(np.stack([np.array(Image.open(io.BytesIO(j)).convert("RGB")) for j in jpgs]))
        x = x.permute(0, 3, 1, 2).float() / 127.5 - 1
        x = F.interpolate(x, size=(res, res), mode="bicubic", antialias=True).clamp(-1, 1)
        z = torch.cat([vae.encode(x[i : i + 32].to(DEVICE)) for i in range(0, len(x), 32)])
        if cached is None:
            e = st.encode(list(captions), prompt=PREFIX, batch_size=32, convert_to_tensor=True)
            embedding = bf16_column(e, 768 * 2)
        else:
            c = cached.slice(rows, len(chunk))
            assert c["key"].to_pylist() == list(keys) and c["caption"].to_pylist() == list(captions)
            embedding = c["embedding"].combine_chunks()
        writer.write_table(pa.table({"key": list(keys), "caption": list(captions), "embedding": embedding, "latent": bf16_column(z, schema.field("latent").type.byte_width)}, schema=schema))
        rows += len(chunk)
        chunk.clear()
        print(f"{out_path.name} {rows} rows, {rows / (time.time() - t0):.2f} rows/s", flush=True)

    with torch.inference_mode(), pq.ParquetWriter(tmp, schema, compression="none") as writer:
        for s in samples(tar_path):
            chunk.append(s)
            if len(chunk) == ROW_GROUP:
                flush()
        if chunk:
            flush()
    os.replace(tmp, out_path)


if __name__ == "__main__":
    res, shards = int(sys.argv[1]), [int(s) for s in sys.argv[2:]]
    assert res % 16 == 0
    vae = AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32).to(DEVICE)
    st = SentenceTransformer("google/embeddinggemma-2", device=DEVICE, config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"dtype": torch.float32})
    null_path = ROOT / "precomputed" / "null_embedding.safetensors"
    if not null_path.exists():
        null_path.parent.mkdir(parents=True, exist_ok=True)
        write_null_embedding(st, null_path)
    out_dir = ROOT / "precomputed" / str(res)
    out_dir.mkdir(parents=True, exist_ok=True)
    todo = []
    for shard in shards:
        name = f"data_{shard:06d}"
        if (out_dir / f"{name}.parquet").exists():
            print(f"{name}: exists, skipping", flush=True)
        else:
            todo.append(name)
    with ThreadPoolExecutor(1) as downloads:
        pending = downloads.submit(fetch, todo[0]) if todo else None
        for i, name in enumerate(todo):
            tar_path = pending.result()
            if i + 1 < len(todo):
                pending = downloads.submit(fetch, todo[i + 1])
            others = sorted((ROOT / "precomputed").glob(f"*/{name}.parquet"))
            write_shard(tar_path, out_dir / f"{name}.parquet", res, vae, st, others[0] if others else None)
            tar_path.unlink()
