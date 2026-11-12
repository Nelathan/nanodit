import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import reencode_text  # noqa: E402
from precompute import bf16_bytes, bf16_column  # noqa: E402

from nanodit.data import _bf16, _sequences  # noqa: E402


def fake_encode(st, captions):
    """A pooled vector and len(caption) token states, each filled with the caption's length."""
    return torch.stack([torch.full((768,), float(len(c))) for c in captions]), [torch.full((len(c), 512), float(len(c))) for c in captions]


def test_the_text_columns_are_encoded_again_and_everything_else_is_kept(tmp_path, monkeypatch):
    monkeypatch.setattr(reencode_text, "encode_captions", fake_encode)
    captions = ["a", "bb", "ccc", "dddd", "eeeee"]
    latents = torch.randn(5, 4)
    path = tmp_path / "shard.parquet"
    table = pa.table({
        "key": [f"k{i}" for i in range(5)], "caption": captions,
        "embedding": bf16_column(torch.zeros(5, 768), 1536), "sequence": [bf16_bytes(torch.zeros(1, 512))] * 5,
        "latent_128": bf16_column(latents, 8),
    })
    pq.write_table(table, path, row_group_size=3)
    assert reencode_text.reencode(path, st=None) == 5
    out = pq.ParquetFile(path)
    assert out.schema_arrow == table.schema and out.metadata.num_row_groups == 2 and not path.with_name("shard.parquet.tmp").exists()
    new = out.read()
    for column in ("key", "caption", "latent_128"):
        assert new[column].equals(table[column])
    assert _bf16(new["embedding"], 768)[:, 0].tolist() == [1, 2, 3, 4, 5]
    assert [len(s) for s in _sequences(new["sequence"])] == [1, 2, 3, 4, 5]
