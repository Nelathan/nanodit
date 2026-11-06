"""Re-encode the token sequences of the fused shards at the encoder's residual width, 512 (PLAN B6).

usage: reencode_sequence.py

For every fused/*.parquet writes fused/sequence512_<name>.parquet with columns
  key       string
  sequence  binary    bf16 [n, 512], C-contiguous: EmbeddingGemma 2's token states after the final norm, before its
                      768x512 `embedding_projection`; prompt prefix dropped, unpadded (n = bytes / 1024)
The 768-d `sequence` of precompute.py is that projection of these states, so it carries no more than 512 dimensions.
Captions are read from the shard; no images are touched. A shard whose output exists is skipped.
Also adds the key `null_tokens` (fp32 [2, 512]), the token states of the empty caption, to null_embedding.safetensors when it is missing.
"""
import time

import pyarrow as pa
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from sentence_transformers import SentenceTransformer

from precompute import DEVICE, ROOT, STAGE, bf16_bytes, encode_text

ROW_GROUP = 1024
SCHEMA = pa.schema([("key", pa.string()), ("sequence", pa.binary())])


def encoder(device):
    """EmbeddingGemma 2, fp32, whose `embedding_projection` passes its input through, so token states stay at 512."""
    st = SentenceTransformer("google/embeddinggemma-2", device=device, config_kwargs={"vision_config": None, "audio_config": None}, model_kwargs={"dtype": torch.float32})
    st[0].auto_model.language_model.embedding_projection.register_forward_hook(lambda module, args, out: args[0])
    return st


def write_null_tokens(st, path):
    """The null prompt's tokens (SPEC §5): the encoder's states of the empty caption, stored beside its pooled embedding."""
    null = load_file(path)
    null["null_tokens"] = encode_text(st, [""])[1][0].float().contiguous()
    save_file(null, path)


def reencode(src, out_path, st):
    table = pq.read_table(src, columns=["key", "caption"])
    tmp = out_path.with_name(out_path.name + ".tmp")
    t0 = time.time()
    with torch.inference_mode(), pq.ParquetWriter(tmp, SCHEMA, compression="none") as writer:
        for start in range(0, len(table), ROW_GROUP):
            chunk = table.slice(start, ROW_GROUP)
            _, sequences = encode_text(st, chunk["caption"].to_pylist())  # the pooled embeddings are not needed
            writer.write_table(pa.table({"key": chunk["key"], "sequence": [bf16_bytes(s) for s in sequences]}, schema=SCHEMA))
            print(f"{out_path.name} {start + len(chunk)} rows, {(start + len(chunk)) / (time.time() - t0):.1f} rows/s", flush=True)
    tmp.replace(out_path)


if __name__ == "__main__":
    st = encoder(DEVICE)
    null_path = ROOT / "precomputed" / "null_embedding.safetensors"
    if "null_tokens" not in load_file(null_path):
        write_null_tokens(st, null_path)
    fused = STAGE / "precomputed" / "fused"
    for src in sorted(fused.glob("*.parquet")):
        if src.name.startswith("sequence512_"):
            continue
        out_path = fused / f"sequence512_{src.name}"
        if out_path.exists():
            print(f"{out_path.name}: exists, skipping", flush=True)
        else:
            reencode(src, out_path, st)
