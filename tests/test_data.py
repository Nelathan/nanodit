import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tinydit.data import Batcher, load_shards


def write_shard(path, latents, embeddings):
    """A shard in the precompute schema; row groups of 3 so the loader sees several chunks."""
    n = len(latents)
    table = pa.table({
        "key": [f"k{i}" for i in range(n)],
        "caption": ["c"] * n,
        "embedding": pa.array([e.contiguous().view(torch.uint8).numpy().tobytes() for e in embeddings], pa.binary(1536)),
        "latent": pa.array([x.contiguous().view(torch.uint8).numpy().tobytes() for x in latents], pa.binary(2 * 16 * 16 * 2)),
    })
    pq.write_table(table, path, row_group_size=3)


@pytest.fixture
def shards(tmp_path):
    torch.manual_seed(0)
    latents = torch.randn(20, 2, 16, 16).bfloat16()
    embeddings = torch.randn(20, 768).bfloat16()
    paths = [str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")]
    write_shard(paths[0], latents[:12], embeddings[:12])
    write_shard(paths[1], latents[12:], embeddings[12:])
    return paths, latents, embeddings


def test_load_shards_round_trips_rows_in_order(shards):
    paths, latents, embeddings = shards
    got_latents, got_embeddings = load_shards(paths, resolution=32)
    assert got_latents.dtype == got_embeddings.dtype == torch.bfloat16
    assert got_latents.shape == (20, 2, 16, 16) and got_embeddings.shape == (20, 768)
    assert torch.equal(got_latents, latents) and torch.equal(got_embeddings, embeddings)


def test_batches_are_real_rows_cast_to_model_dtype_in_a_static_buffer(shards):
    paths, latents, embeddings = shards
    batcher = Batcher(*load_shards(paths, 32), batch=4, device=torch.device("cpu"), dtype=torch.float32)
    x, text = batcher.next()
    assert x.dtype == text.dtype == torch.float32 and x.shape == (4, 2, 16, 16) and text.shape == (4, 768)
    # each row of the batch is one dataset row with its own embedding
    for row, emb in zip(x, text):
        match = (latents.float() == row).flatten(1).all(1).nonzero()
        assert len(match) == 1 and torch.equal(embeddings[match.item()].float(), emb)


def test_batches_are_random_and_the_buffer_is_reused(shards):
    paths, _, _ = shards
    batcher = Batcher(*load_shards(paths, 32), batch=4, device=torch.device("cpu"), dtype=torch.float32)
    x1, _ = batcher.next()
    first = x1.clone()
    x2, _ = batcher.next()
    assert x1.data_ptr() == x2.data_ptr()
    assert not torch.equal(first, x2)
    seen = {tuple(batcher.next()[0][:, 0, 0, 0].tolist()) for _ in range(30)}
    assert len(seen) > 20
