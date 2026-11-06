"""Precompute Primal 2x2 latents and EmbeddingGemma 2 encodings for text-to-image-2M data_512_2M shards (GPU).

usage: precompute.py SHARD [SHARD ...]

Writes /mnt/luna/tiny-dit/precomputed/fused/data_<SHARD>.parquet with columns
  key         string
  source      string                    the Hugging Face dataset the row comes from
  caption     string
  embedding   fixed_size_binary(1536)   768 bf16, L2-normalized fp32 embedding rounded to bf16
  sequence    binary                    bf16 [n, 768], C-contiguous: the encoder's token states of the caption, the
                                        prompt prefix's tokens dropped, unpadded (n = bytes / 1536)
  latent_128  fixed_size_binary(...)    bf16 [2, 64, 64], C-contiguous
  latent_256  fixed_size_binary(...)    bf16 [2, 128, 128], C-contiguous
Both latents are encoded from a Lanczos resize of the 512 px crop (PLAN B5). The shard tar is downloaded to
/mnt/luna and deleted once encoded; the next shard's tar downloads while one encodes. A shard whose parquet exists
is skipped. Also writes /mnt/mars/ai/tiny-dit/precomputed/null_embedding.safetensors (key `null`, fp32 [768]), the embedding of the
empty caption, when it is missing; reencode_sequence.py adds its tokens as `null_tokens`.
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
from diffusers import AutoModel
from huggingface_hub import hf_hub_download
from PIL import Image
from safetensors.torch import save_file
from sentence_transformers import SentenceTransformer

ROOT = Path("/mnt/mars/ai/tiny-dit")
STAGE = Path("/mnt/luna/tiny-dit")  # HDD: the tars and the encoded shards before the shuffle
REPO = "jackyhate/text-to-image-2M"
PREFIX = "title: none | text: "
PREFIX_TOKENS = 7  # <bos> title : ▁none ▁| ▁text : -- the prefix's trailing space merges into the caption's first token
RESOLUTIONS = (128, 256)
TEXT_DIM = 768
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
    tar_path = STAGE / "data" / "data_512_2M" / f"{name}.tar"
    if not tar_path.exists():
        hf_hub_download(REPO, f"data_512_2M/{name}.tar", repo_type="dataset", local_dir=STAGE / "data")
    return tar_path


def load_encoders():
    """Primal 2x2 and EmbeddingGemma 2, fp32, on the GPU."""
    vae = AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32).to(DEVICE)
    st = SentenceTransformer("google/embeddinggemma-2", device=DEVICE, config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"dtype": torch.float32})
    return vae, st


def write_null_embedding(st, path):
    """The null embedding's init (SPEC §5): the encoder's embedding of the empty caption, under the same prefix."""
    save_file({"null": st.encode([""], prompt=PREFIX, convert_to_tensor=True)[0].contiguous()}, path)


def bf16_bytes(t):
    return t.bfloat16().cpu().contiguous().view(torch.int16).numpy().tobytes()


def bf16_column(t, width):
    """[N, ...] float tensor -> fixed_size_binary column of the rows as bf16."""
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(width), len(t), [None, pa.py_buffer(bf16_bytes(t))])


def resize(image_bytes, res):
    """A square image as [1, 3, res, res] in [-1, 1], PIL Lanczos."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((res, res), Image.LANCZOS)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 127.5 - 1


def encode_text(st, captions):
    """The normalized pooled embeddings [N, 768] and each caption's token states [n_i, 768] after the prefix."""
    outputs = st.encode(captions, prompt=PREFIX, batch_size=32, output_value=None)
    prefix_ids = st.tokenizer(PREFIX.rstrip())["input_ids"][:PREFIX_TOKENS]
    sequences = []
    for o in outputs:
        live = o["attention_mask"].bool()
        assert o["input_ids"][live][:PREFIX_TOKENS].tolist() == prefix_ids
        sequences.append(o["token_embeddings"][live][PREFIX_TOKENS:])
    return torch.stack([o["sentence_embedding"] for o in outputs]), sequences


def write_shard(rows_in, source, out_path, vae, st):
    """Encodes (key, image bytes, caption) rows, square images, from the dataset `source` into one Parquet at `out_path`."""
    schema = pa.schema([
        ("key", pa.string()), ("source", pa.string()), ("caption", pa.string()),
        ("embedding", pa.binary(TEXT_DIM * 2)), ("sequence", pa.binary()),
        *[(f"latent_{res}", pa.binary(2 * (res // 2) ** 2 * 2)) for res in RESOLUTIONS],
    ])
    tmp = out_path.with_name(out_path.name + ".tmp")
    rows, t0 = 0, time.time()
    chunk = []

    def flush():
        nonlocal rows
        keys, jpgs, captions = zip(*chunk)
        latents = {}
        for res in RESOLUTIONS:
            x = torch.cat([resize(j, res) for j in jpgs])
            z = torch.cat([vae.encode(x[i : i + 32].to(DEVICE)) for i in range(0, len(x), 32)])
            latents[f"latent_{res}"] = bf16_column(z, schema.field(f"latent_{res}").type.byte_width)
        embedding, sequences = encode_text(st, list(captions))
        writer.write_table(pa.table({
            "key": list(keys), "source": [source] * len(keys), "caption": list(captions),
            "embedding": bf16_column(embedding, TEXT_DIM * 2), "sequence": [bf16_bytes(s) for s in sequences], **latents,
        }, schema=schema))
        rows += len(chunk)
        chunk.clear()
        print(f"{out_path.name} {rows} rows, {rows / (time.time() - t0):.2f} rows/s", flush=True)

    with torch.inference_mode(), pq.ParquetWriter(tmp, schema, compression="none") as writer:
        for s in rows_in:
            chunk.append(s)
            if len(chunk) == ROW_GROUP:
                flush()
        if chunk:
            flush()
    os.replace(tmp, out_path)


if __name__ == "__main__":
    shards = [int(s) for s in sys.argv[1:]]
    vae, st = load_encoders()
    null_path = ROOT / "precomputed" / "null_embedding.safetensors"
    if not null_path.exists():
        null_path.parent.mkdir(parents=True, exist_ok=True)
        write_null_embedding(st, null_path)
    out_dir = STAGE / "precomputed" / "fused"
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
            write_shard(samples(tar_path), REPO, out_dir / f"{name}.parquet", vae, st)
            tar_path.unlink()
