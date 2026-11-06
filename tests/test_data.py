import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tinydit import data
from tinydit.data import Stream, pad_sequences
from tinydit.model import NULL_TOKENS, SEQ_DIM, Prompt

ROOT = Path(__file__).parent.parent


def bf16_bytes(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def write_shard(path, latents, embeddings, sequences, keys=None):
    """A fused shard in the precompute schema (the 768-d sequence and the 64 px latent are stand-ins the shuffle must leave
    behind) and its sibling sequence512 file; row groups of 3 so the readers see several chunks."""
    n = len(latents)
    keys = keys or [f"k{i}" for i in range(n)]
    table = pa.table({
        "key": keys,
        "source": ["s"] * n,
        "caption": ["c"] * n,
        "embedding": pa.array([bf16_bytes(e) for e in embeddings], pa.binary(1536)),
        "sequence": [b"768"] * n,
        "latent_32": pa.array([bf16_bytes(x) for x in latents], pa.binary(2 * 16 * 16 * 2)),
        "latent_64": pa.array([b"x" * 4] * n, pa.binary(4)),
    })
    pq.write_table(table, path, row_group_size=3)
    path = Path(path)
    pq.write_table(pa.table({"key": keys, "sequence": [bf16_bytes(q) for q in sequences]}), path.with_name(f"sequence512_{path.name}"), row_group_size=3)


@pytest.fixture
def rows():
    torch.manual_seed(0)
    sequences = [torch.randn(2 + i % 5, SEQ_DIM).bfloat16() for i in range(20)]
    return torch.randn(20, 2, 16, 16).bfloat16(), torch.randn(20, 768).bfloat16(), sequences


def shuffle_module(buckets):
    spec = importlib.util.spec_from_file_location("shuffle", ROOT / "scripts" / "shuffle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BUCKETS = buckets  # 20 rows would leave most of 64 buckets empty, which 50k-row shards never do
    module.ROW_GROUP = 3  # several row groups per bucket, so the stream sees several per block
    return module


def shuffle(tmp_path, rows, buckets):
    latents, embeddings, sequences = rows
    module = shuffle_module(buckets)
    write_shard(tmp_path / "a.parquet", latents[:12], embeddings[:12], sequences[:12], [f"k{i}" for i in range(12)])
    write_shard(tmp_path / "b.parquet", latents[12:], embeddings[12:], sequences[12:], [f"k{i}" for i in range(12, 20)])
    out = tmp_path / "shuffled.parquet"
    module.shuffle([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")], 32, out)
    return out


@pytest.fixture
def shuffled(tmp_path, rows):
    return shuffle(tmp_path, rows, buckets=4)


def test_shuffle_keeps_every_row_once_in_a_new_order(shuffled, rows):
    latents, embeddings, sequences = rows
    table = pq.read_table(shuffled)
    assert table.column_names == ["key", "source", "caption", "embedding", "latent_32", "sequence"]  # the other latent and the 768-d sequence stay behind
    keys = table["key"].to_pylist()
    assert sorted(keys, key=lambda k: int(k[1:])) == [f"k{i}" for i in range(20)] and keys != sorted(keys, key=lambda k: int(k[1:]))
    got_latents = data._bf16(table["latent_32"], 2, 16, 16)
    got_embeddings = data._bf16(table["embedding"], 768)
    got_sequences = data._sequences(table["sequence"])
    for key, latent, embedding, sequence in zip(keys, got_latents, got_embeddings, got_sequences):  # each row stays whole
        i = int(key[1:])
        assert torch.equal(latent, latents[i]) and torch.equal(embedding, embeddings[i]) and torch.equal(sequence, sequences[i])
    assert not list(shuffled.parent.glob("shuffled.parquet.*"))  # the buckets and the .tmp are gone


def test_shuffle_keeps_each_image_once_with_its_first_caption(tmp_path, rows):
    latents, embeddings, sequences = rows
    write_shard(tmp_path / "a.parquet", latents[[0, 1, 1, 2]], embeddings[[0, 1, 2, 3]], sequences[:4], ["k0", "k1", "k2", "k3"])  # k2 repeats k1's image
    write_shard(tmp_path / "b.parquet", latents[[0, 4]], embeddings[[10, 11]], sequences[10:12], ["k10", "k11"])  # k10 repeats k0's image
    out = tmp_path / "shuffled.parquet"
    shuffle_module(buckets=4).shuffle([str(tmp_path / "a.parquet"), str(tmp_path / "b.parquet")], 32, out)
    table = pq.read_table(out)
    by_key = dict(zip(table["key"].to_pylist(), data._bf16(table["embedding"], 768)))
    assert sorted(by_key) == ["k0", "k1", "k11", "k3"]
    assert torch.equal(by_key["k0"], embeddings[0]) and torch.equal(by_key["k1"], embeddings[1])


def test_rows_are_shuffled_inside_a_bucket(tmp_path, rows):
    keys = pq.read_table(shuffle(tmp_path, rows, buckets=1))["key"].to_pylist()
    assert keys != [f"k{i}" for i in range(20)]


def epoch(stream, micro_batches):
    """The micro-batches as copies, since the stream refills its buffers."""
    batches = []
    for _ in range(micro_batches):
        x, prompt = stream.next()
        batches.append((x.clone(), Prompt(*(a.clone() for a in prompt))))
    return batches


def test_each_pass_streams_every_row_once_and_the_next_pass_in_another_order(shuffled, rows, monkeypatch):
    latents, embeddings, sequences = rows
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 2)  # blocks of up to 6 rows: batches of 4 cross blocks
    stream = Stream(str(shuffled), resolution=32, batch=4, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    passes = []
    for _ in range(2):
        seen = []
        for x, prompt in epoch(stream, 5):
            assert x.dtype == prompt.pooled.dtype == prompt.tokens.dtype == torch.float32 and x.shape == (4, 2, 16, 16) and prompt.pooled.shape == (4, 768)
            for row, emb, tokens, mask in zip(x, *prompt):
                match = (latents.float() == row).flatten(1).all(1).nonzero()
                i = match.item()
                assert len(match) == 1 and torch.equal(embeddings[i].float(), emb)  # its own embedding
                assert torch.equal(tokens[mask], sequences[i].float()) and not tokens[~mask].any()  # its own tokens, zero past them
                seen.append(i)
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


def test_sequences_pad_to_the_longest_caption_and_at_least_the_null_prompt():
    tokens, mask = pad_sequences([torch.ones(3, SEQ_DIM), torch.ones(5, SEQ_DIM)], torch.float32)
    assert tokens.shape == (2, 5, SEQ_DIM) and mask.sum(1).tolist() == [3, 5]
    assert not tokens[0, 3:].any() and tokens[0, :3].eq(1).all()
    assert pad_sequences([torch.ones(1, SEQ_DIM)])[0].shape == (1, NULL_TOKENS, SEQ_DIM)
