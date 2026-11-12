"""Precompute Primal 2x2 latents and EmbeddingGemma 2 encodings for text-to-image-2M data_512_2M shards (GPU).

usage: precompute.py DIR SHARD [SHARD ...]

Writes DIR/precomputed/fused/data_<SHARD>.parquet with columns
  key         string
  source      string                    the Hugging Face dataset the row comes from
  caption     string
  embedding   fixed_size_binary(1536)   768 bf16, L2-normalized fp32 embedding rounded to bf16
  sequence    binary                    bf16 [n, 512], C-contiguous: the encoder's token states of the caption after its
                                        final norm, before the 768x512 `embedding_projection`; <bos> and <eos>
                                        dropped, unpadded (n = bytes / 1024)
  latent_128  fixed_size_binary(...)    bf16 [2, 64, 64], C-contiguous
  latent_256  fixed_size_binary(...)    bf16 [2, 128, 128], C-contiguous
Both latents are encoded from a Lanczos resize of the 512 px crop (PLAN B5). The shard tar is downloaded to
DIR/data and deleted once encoded; the next shard's tar downloads while one encodes. A shard whose parquet exists
is skipped. Also writes DIR/precomputed/null_embedding.safetensors when it is missing: the empty caption's pooled
embedding (key `null`, fp32 [768]) and its <bos> and <eos> states (`null_tokens`, fp32 [2, 512]).
The captions are encoded with no prompt prefix (SPEC §1).
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

REPO = "jackyhate/text-to-image-2M"
RESOLUTIONS = (128, 256)
TEXT_DIM = 768
ROW_GROUP = 1024
TEXT_BATCH = 32
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


def fetch(data_dir, name):
    """The shard's tar path under `data_dir`, downloading it unless it is on disk."""
    tar_path = data_dir / "data_512_2M" / f"{name}.tar"
    if not tar_path.exists():
        hf_hub_download(REPO, f"data_512_2M/{name}.tar", repo_type="dataset", local_dir=data_dir)
    return tar_path


def load_text_encoder(device):
    """EmbeddingGemma 2's text path, fp32."""
    return SentenceTransformer("google/embeddinggemma-2", device=device, config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"dtype": torch.float32})


def load_encoders():
    """Primal 2x2 and EmbeddingGemma 2, fp32, on the GPU."""
    vae = AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32).to(DEVICE)
    return vae, load_text_encoder(DEVICE)


def write_null_embedding(st, path):
    """The null prompt's init (SPEC §5): the encoder's pooled embedding of the empty caption and its two states, <bos>
    and <eos>, the only tokens it has."""
    pooled, states = encode_states(st, [""])
    save_file({"null": pooled[0].cpu().contiguous(), "null_tokens": states[0].cpu().contiguous()}, path)


def bf16_bytes(t):
    return t.bfloat16().cpu().contiguous().view(torch.int16).numpy().tobytes()


def bf16_column(t, width):
    """[N, ...] float tensor -> fixed_size_binary column of the rows as bf16."""
    return pa.FixedSizeBinaryArray.from_buffers(pa.binary(width), len(t), [None, pa.py_buffer(bf16_bytes(t))])


def resize(image_bytes, res):
    """A square image as [1, 3, res, res] in [-1, 1], PIL Lanczos."""
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB").resize((res, res), Image.LANCZOS)
    return torch.from_numpy(np.array(image)).permute(2, 0, 1)[None].float() / 127.5 - 1


@torch.inference_mode()
def encode_states(st, captions):
    """The normalized pooled embeddings [N, 768] and each caption's token states [n_i, 512], <bos> to <eos>, from one
    forward with no prompt prefix: the states are what the encoder's `embedding_projection` (768x512, per token, before
    the pooling) is given. The pooling averages all of them, <bos> and <eos> included."""
    bos, eos = st.tokenizer.bos_token_id, st.tokenizer.eos_token_id
    states = []
    hook = st[0].auto_model.language_model.embedding_projection.register_forward_hook(lambda module, args, output: states.append(args[0]))
    pooled, sequences = [], []
    try:
        for start in range(0, len(captions), TEXT_BATCH):
            features = st.preprocess(captions[start : start + TEXT_BATCH])
            out = st({k: v.to(st.device) if isinstance(v, torch.Tensor) else v for k, v in features.items()})
            lengths = out["attention_mask"].sum(1).tolist()  # padded on the right
            assert all(ids[0] == bos and ids[n - 1] == eos for ids, n in zip(out["input_ids"].tolist(), lengths))
            pooled.append(out["sentence_embedding"])
            sequences += [s[:n] for s, n in zip(states.pop(), lengths)]
    finally:
        hook.remove()
    return torch.cat(pooled), sequences


def encode_text(st, captions):
    """The normalized pooled embeddings [N, 768] and each caption's token states [n_i, 512] without <bos> and <eos>,
    which carry nothing of the caption (SPEC §1)."""
    pooled, states = encode_states(st, captions)
    assert min(len(s) for s in states) > 2, "a caption with no tokens: cross attention would see a fully masked row"
    return pooled, [s[1:-1] for s in states]


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
    root, shards = Path(sys.argv[1]), [int(s) for s in sys.argv[2:]]
    vae, st = load_encoders()
    out_dir = root / "precomputed" / "fused"
    out_dir.mkdir(parents=True, exist_ok=True)
    null_path = root / "precomputed" / "null_embedding.safetensors"
    if not null_path.exists():
        write_null_embedding(st, null_path)
    todo = []
    for shard in shards:
        name = f"data_{shard:06d}"
        if (out_dir / f"{name}.parquet").exists():
            print(f"{name}: exists, skipping", flush=True)
        else:
            todo.append(name)
    with ThreadPoolExecutor(1) as downloads:
        pending = downloads.submit(fetch, root / "data", todo[0]) if todo else None
        for i, name in enumerate(todo):
            tar_path = pending.result()
            if i + 1 < len(todo):
                pending = downloads.submit(fetch, root / "data", todo[i + 1])
            write_shard(samples(tar_path), REPO, out_dir / f"{name}.parquet", vae, st)
            tar_path.unlink()
