import importlib.util
import time
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tinydit import data
from tinydit.data import Stream

ROOT = Path(__file__).parent.parent


def write_shard(path, latents, embeddings, keys=None):
    """A shard in the precompute schema; row groups of 3 so the readers see several chunks."""
    n = len(latents)
    table = pa.table({
        "key": keys or [f"k{i}" for i in range(n)],
        "caption": ["c"] * n,
        "embedding": pa.array([e.contiguous().view(torch.uint8).numpy().tobytes() for e in embeddings], pa.binary(1536)),
        "latent": pa.array([x.contiguous().view(torch.uint8).numpy().tobytes() for x in latents], pa.binary(2 * 16 * 16 * 2)),
    })
    pq.write_table(table, path, row_group_size=3)


@pytest.fixture
def rows():
    torch.manual_seed(0)
    return torch.randn(20, 2, 16, 16).bfloat16(), torch.randn(20, 768).bfloat16()


def shuffle_module(buckets):
    spec = importlib.util.spec_from_file_location("shuffle", ROOT / "scripts" / "shuffle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BUCKETS = buckets  # 20 rows would leave most of 64 buckets empty, which 50k-row shards never do
    module.ROW_GROUP = 3  # several row groups per bucket, so the stream sees several per block
    return module


def shuffle(tmp_path, rows, buckets):
    latents, embeddings = rows
    module = shuffle_module(buckets)
    write_shard(tmp_path / "a.parquet", latents[:12], embeddings[:12], [f"k{i}" for i in range(12)])
    write_shard(tmp_path / "b.parquet", latents[12:], embeddings[12:], [f"k{i}" for i in range(12, 20)])
    out = tmp_path / "shuffled.parquet"
    module.shuffle([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")], out)
    return out


@pytest.fixture
def shuffled(tmp_path, rows):
    return shuffle(tmp_path, rows, buckets=4)


def test_shuffle_keeps_every_row_once_in_a_new_order(shuffled, rows):
    latents, embeddings = rows
    table = pq.read_table(shuffled)
    keys = table["key"].to_pylist()
    assert sorted(keys, key=lambda k: int(k[1:])) == [f"k{i}" for i in range(20)] and keys != sorted(keys, key=lambda k: int(k[1:]))
    got_latents = data._bf16(table["latent"], 2, 16, 16)
    got_embeddings = data._bf16(table["embedding"], 768)
    for key, latent, embedding in zip(keys, got_latents, got_embeddings):  # each row stays whole
        i = int(key[1:])
        assert torch.equal(latent, latents[i]) and torch.equal(embedding, embeddings[i])
    assert not list(shuffled.parent.glob("shuffled.parquet.*"))  # the buckets and the .tmp are gone


def test_shuffle_keeps_each_image_once_with_its_first_caption(tmp_path, rows):
    latents, embeddings = rows
    write_shard(tmp_path / "a.parquet", latents[[0, 1, 1, 2]], embeddings[[0, 1, 2, 3]], ["k0", "k1", "k2", "k3"])  # k2 repeats k1's image
    write_shard(tmp_path / "b.parquet", latents[[0, 4]], embeddings[[10, 11]], ["k10", "k11"])  # k10 repeats k0's image
    out = tmp_path / "shuffled.parquet"
    shuffle_module(buckets=4).shuffle([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")], out)
    table = pq.read_table(out)
    by_key = dict(zip(table["key"].to_pylist(), data._bf16(table["embedding"], 768)))
    assert sorted(by_key) == ["k0", "k1", "k11", "k3"]
    assert torch.equal(by_key["k0"], embeddings[0]) and torch.equal(by_key["k1"], embeddings[1])


def test_rows_are_shuffled_inside_a_bucket(tmp_path, rows):
    keys = pq.read_table(shuffle(tmp_path, rows, buckets=1))["key"].to_pylist()
    assert keys != [f"k{i}" for i in range(20)]


def epoch(stream, micro_batches):
    return [tuple(t.clone() for t in stream.next()) for _ in range(micro_batches)]


def test_each_pass_streams_every_row_once_and_the_next_pass_in_another_order(shuffled, rows, monkeypatch):
    latents, embeddings = rows
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 2)  # blocks of up to 6 rows: batches of 4 cross blocks
    stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    passes = []
    for _ in range(2):
        seen = []
        for x, text in epoch(stream, 5):
            assert x.dtype == text.dtype == torch.float32 and x.shape == (4, 2, 16, 16) and text.shape == (4, 768)
            for row, emb in zip(x, text):
                match = (latents.float() == row).flatten(1).all(1).nonzero()
                assert len(match) == 1 and torch.equal(embeddings[match.item()].float(), emb)  # its own embedding
                seen.append(match.item())
        assert sorted(seen) == list(range(20))
        passes.append(seen)
    assert passes[0] != passes[1]


def test_rows_are_shuffled_inside_a_block(shuffled, rows, monkeypatch):
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 100)  # the whole file is one block, so only rows can move
    stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    first, second = (torch.cat([x for x, _ in epoch(stream, 5)]) for _ in range(2))
    assert not torch.equal(first, second) and torch.equal(first.sort(0).values, second.sort(0).values)


def test_the_buffers_are_reused_and_the_order_follows_the_seed(shuffled):
    def first_rows(seed):
        stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cpu"), dtype=torch.float32, seed=seed)
        x1, _ = stream.next()
        pointer = x1.data_ptr()
        rows = [stream.next()[0][:, 0, 0, 0].tolist() for _ in range(4)]
        assert stream.next()[0].data_ptr() == pointer
        return rows

    assert first_rows(0) == first_rows(0) != first_rows(1)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_a_micro_batch_does_not_wait_for_queued_gpu_work_and_arrives_whole(shuffled, rows):
    latents, _ = rows
    stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cuda"), dtype=torch.float32, seed=0)
    stream.next()  # the first call reads the first block
    waited = []
    for _ in range(3):  # later calls wait on the previous copy's event, already passed here
        torch.cuda.synchronize()
        torch.cuda._sleep(100_000_000)  # about 50 ms of queued GPU work
        start = time.perf_counter()
        x, _ = stream.next()
        waited.append(time.perf_counter() - start)
    assert max(waited) < 0.02
    for row in x.cpu():  # read after the queue: the copy landed whole and cast
        assert (latents.float() == row).flatten(1).all(1).sum() == 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
def test_back_to_back_micro_batches_behind_queued_work_each_arrive_whole(shuffled, rows):
    latents, embeddings = rows
    stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cuda"), dtype=torch.float32, seed=0)
    stream.next()
    torch.cuda.synchronize()
    torch.cuda._sleep(100_000_000)  # the first copy waits behind this, so the second refill must wait for it
    batches = [tuple(t.clone() for t in stream.next()) for _ in range(2)]  # clones queue on the stream, no sync
    for x, text in batches:
        for row, emb in zip(x.cpu(), text.cpu()):
            match = (latents.float() == row).flatten(1).all(1).nonzero()
            assert len(match) == 1 and torch.equal(embeddings[match.item()].float(), emb)
    assert not torch.equal(batches[0][0], batches[1][0])
