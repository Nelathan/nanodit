import hashlib
import importlib.util
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from nanodit import data
from nanodit.data import Stream, pad_sequences
from nanodit.model import NULL_TOKENS, SEQ_DIM

ROOT = Path(__file__).parent.parent


def bf16_bytes(t):
    return t.contiguous().view(torch.uint8).numpy().tobytes()


def write_shard(path, latents, embeddings, sequences, keys=None):
    """A fused shard in the precompute schema (the 64 px latent is a stand-in the shuffle must leave behind); row groups
    of 3 so the readers see several chunks."""
    n = len(latents)
    keys = keys or [f"k{i}" for i in range(n)]
    table = pa.table({
        "key": keys,
        "source": ["s"] * n,
        "caption": ["c"] * n,
        "embedding": pa.array([bf16_bytes(e) for e in embeddings], pa.binary(1536)),
        "sequence": [bf16_bytes(q) for q in sequences],
        "latent_32": pa.array([bf16_bytes(x) for x in latents], pa.binary(2 * 16 * 16 * 2)),
        "latent_64": pa.array([b"x" * 4] * n, pa.binary(4)),
    })
    pq.write_table(table, path, row_group_size=3)


@pytest.fixture
def rows():
    torch.manual_seed(0)
    sequences = [torch.randn(2 + i % 5, SEQ_DIM).bfloat16() for i in range(20)]
    return torch.randn(20, 2, 16, 16).bfloat16(), torch.randn(20, 768).bfloat16(), sequences


def write_fine(path, latents, buckets, embeddings, sequences, keys):
    """An encoded Fine-T2I Parquet in the encode_fine schema: a bucket column and variable-size latents."""
    n = len(latents)
    table = pa.table({
        "key": keys, "source": ["f"] * n, "bucket": pa.array(buckets, pa.int8()), "caption": ["c"] * n,
        "prompt": ["p"] * n, "enhanced_prompt": ["e"] * n, "aesthetic": pa.array([5.0] * n, pa.float32()),
        "embedding": pa.array([bf16_bytes(e) for e in embeddings], pa.binary(1536)),
        "sequence": [bf16_bytes(q) for q in sequences],
        "latent_32": [bf16_bytes(x) for x in latents], "latent_64": [b"x" * 4] * n,
    })
    pq.write_table(table, path, row_group_size=3)


def shuffle_module():
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location("shuffle", ROOT / "scripts" / "shuffle.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.PART_ROWS = 4  # several parts per bucket, as 16k-row parts of millions of rows are
    module.DEAL_ROWS = 5  # dealt in several pieces, across file boundaries
    module.ROW_GROUP = 3  # several row groups per bucket, so the stream sees several per block
    return module


WIDE = (28, 36)  # Fine-T2I's bucket 1 (224 x 288 at 256 px) at 32 px


@pytest.fixture
def mixed(tmp_path, rows):
    """12 + 8 square rows from two fused shards and 6 square plus 6 wide rows from a Fine-T2I file."""
    latents, embeddings, sequences = rows
    write_shard(tmp_path / "a.parquet", latents[:12], embeddings[:12], sequences[:12], [f"k{i}" for i in range(12)])
    write_shard(tmp_path / "b.parquet", latents[12:], embeddings[12:], sequences[12:], [f"k{i}" for i in range(12, 20)])
    torch.manual_seed(1)
    fine = [torch.randn(2, 16, 16).bfloat16() if i % 2 else torch.randn(2, WIDE[0] // 2, WIDE[1] // 2).bfloat16() for i in range(12)]
    write_fine(tmp_path / "f.parquet", fine, [0 if i % 2 else 1 for i in range(12)], embeddings[:12], sequences[:12], [f"f{i}" for i in range(12)])
    out = tmp_path / "set"
    shuffle_module().shuffle([str(tmp_path / n) for n in ("a.parquet", "b.parquet", "f.parquet")], 32, out)
    return out, fine


def test_shuffle_writes_each_bucket_once_whole_and_in_a_new_order(mixed, rows):
    out, fine = mixed
    latents, embeddings, sequences = rows
    assert sorted(p.name for p in out.iterdir()) == ["28x36.parquet", "32x32.parquet", data.EXCLUDED, data.HELDOUT]  # parts and .tmp are gone
    assert (out / data.EXCLUDED).read_text() == (out / data.HELDOUT).read_text() == ""
    square, wide = pq.read_table(out / "32x32.parquet"), pq.read_table(out / "28x36.parquet")
    assert square.column_names == ["key", "source", "caption", "embedding", "sequence", "latent_32"]  # the other latent and Fine's spare columns stay behind
    assert square.schema.field("sequence").type == wide.schema.field("sequence").type == pa.binary()  # data._sequences reads int32 offsets
    keys = square["key"].to_pylist()
    expected = [f"k{i}" for i in range(20)] + [f"f{i}" for i in range(1, 12, 2)]
    assert sorted(keys) == sorted(expected) and keys != expected
    assert sorted(wide["key"].to_pylist()) == sorted(f"f{i}" for i in range(0, 12, 2))
    for key, latent, embedding, sequence in zip(keys, data._bf16(square["latent_32"], 2, 16, 16), data._bf16(square["embedding"], 768), data._sequences(square["sequence"])):
        i = int(key[1:])  # each row stays whole
        assert torch.equal(latent, (latents if key[0] == "k" else fine)[i]) and torch.equal(embedding, embeddings[i]) and torch.equal(sequence, sequences[i])
    for key, latent in zip(wide["key"].to_pylist(), data._bf16(wide["latent_32"], 2, 14, 18)):
        assert torch.equal(latent, fine[int(key[1:])])


def test_shuffle_keeps_each_image_once_with_its_first_caption_across_datasets(tmp_path, rows):
    latents, embeddings, sequences = rows
    write_shard(tmp_path / "a.parquet", latents[[0, 1, 1, 2]], embeddings[[0, 1, 2, 3]], sequences[:4], ["k0", "k1", "k2", "k3"])  # k2 repeats k1's image
    write_fine(tmp_path / "f.parquet", latents[[0, 4]], [0, 0], embeddings[[10, 11]], sequences[10:12], ["f10", "f11"])  # f10 repeats k0's image
    out = tmp_path / "set"
    shuffle_module().shuffle([str(tmp_path / "a.parquet"), str(tmp_path / "f.parquet")], 32, out)
    table = pq.read_table(out / "32x32.parquet")
    by_key = dict(zip(table["key"].to_pylist(), data._bf16(table["embedding"], 768)))
    assert sorted(by_key) == ["f11", "k0", "k1", "k3"]
    assert torch.equal(by_key["k0"], embeddings[0]) and torch.equal(by_key["k1"], embeddings[1])


def keys_of(out):
    """Row identity by latent: {latent bytes: key} over the set."""
    found = {}
    for f in out.glob("*.parquet"):
        t = pq.read_table(f)
        found.update(zip(t["latent_32"].to_pylist(), t["key"].to_pylist()))
    return found


def stream_keys(stream, micro_batches, by_latent):
    """The keys of the next micro-batches, each micro-batch from one bucket."""
    batches = []
    for bucket in stream.draw(micro_batches):
        x, prompt = bucket.next()
        assert x.dtype == prompt.pooled.dtype == prompt.tokens.dtype == torch.float32 and x.shape[1:] == bucket.shape
        batches.append([by_latent[bf16_bytes(row.bfloat16())] for row in x])
    return batches


def test_a_bucket_streams_every_row_once_per_pass_and_the_next_pass_in_other_blocks(mixed, monkeypatch):
    out, _ = mixed
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 2)  # blocks of up to 6 rows: micro-batches of 2 cross blocks
    stream = Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    by_latent = keys_of(out)
    square = next(b for b in stream.buckets if b.shape == (2, 16, 16))
    passes = []
    for _ in range(2):
        seen = []
        for _ in range(13):  # 26 rows
            x, prompt = square.next()
            for row, emb, tokens, mask in zip(x, *prompt):
                seen.append(by_latent[bf16_bytes(row.bfloat16())])
                assert mask.sum() >= 2 and not tokens[~mask].any()  # its tokens, zero past them
        assert sorted(seen) == sorted(k for k in by_latent.values() if k[0] == "k" or int(k[1:]) % 2)
        passes.append(seen)
    assert passes[0] != passes[1]


def square_keys(out):
    return pq.read_table(out / "32x32.parquet", columns=["key"])["key"].to_pylist()


def test_an_excluded_row_is_never_streamed(mixed, monkeypatch):
    out, _ = mixed
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 2)
    (out / data.EXCLUDED).write_text("k3\nk7\n")
    by_latent = keys_of(out)
    square = next(b for b in Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=0).buckets if b.shape == (2, 16, 16))
    seen = [by_latent[bf16_bytes(row.bfloat16())] for _ in range(24) for row in square.next()[0]]  # two passes of 24 rows
    assert sorted(seen) == sorted(2 * [k for k in square_keys(out) if k not in ("k3", "k7")])


def test_a_held_out_row_is_never_streamed(mixed, monkeypatch):
    out, _ = mixed
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 2)
    (out / data.EXCLUDED).write_text("k3\n")
    (out / data.HELDOUT).write_text("k7\nk9\n")
    by_latent = keys_of(out)
    square = next(b for b in Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=0).buckets if b.shape == (2, 16, 16))
    seen = [by_latent[bf16_bytes(row.bfloat16())] for _ in range(23) for row in square.next()[0]]  # two passes of 23 rows
    assert sorted(seen) == sorted(2 * [k for k in square_keys(out) if k not in ("k3", "k7", "k9")])


def test_holdout_takes_the_rows_of_its_subset_with_the_smallest_key_hash(rows, tmp_path):
    sys.path.insert(0, str(ROOT / "scripts"))
    import holdout

    latents, embeddings, sequences = rows
    keys = [f"{holdout.SUBSET}r{i}" if i % 4 else f"other/r{i}" for i in range(20)]
    write_shard(tmp_path / "32x32.parquet", latents, embeddings, sequences, keys)
    table = pq.read_table(tmp_path / "32x32.parquet")
    sources = [holdout.SOURCE if i % 5 else "Z-Image-Turbo" for i in range(20)]
    pq.write_table(table.set_column(1, "source", pa.array(sources)), tmp_path / "32x32.parquet", row_group_size=3)
    (tmp_path / data.EXCLUDED).write_text(keys[1] + "\n")
    holdout.main(str(tmp_path), "4", str(tmp_path / "held.parquet"))
    eligible = [k for i, k in enumerate(keys) if i % 4 and i % 5 and i != 1]
    expected = sorted(eligible, key=lambda k: hashlib.sha256(k.encode()).digest())[:4]
    held = pq.read_table(tmp_path / "held.parquet")
    assert held["key"].to_pylist() == expected
    assert (tmp_path / data.HELDOUT).read_text().split() == sorted(expected)
    x, _ = data.read_rows(str(tmp_path / "held.parquet"), 32)
    assert torch.equal(x, latents[[keys.index(k) for k in expected]])


def test_a_bucket_streams_on_when_a_whole_block_is_excluded(mixed, monkeypatch):
    out, _ = mixed
    monkeypatch.setattr(data, "BLOCK_ROW_GROUPS", 1)  # a block is one row group of 3 rows
    keys = square_keys(out)
    (out / data.EXCLUDED).write_text("\n".join(keys[:3]))  # the first row group
    by_latent = keys_of(out)
    square = next(b for b in Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=0).buckets if b.shape == (2, 16, 16))
    seen = [by_latent[bf16_bytes(row.bfloat16())] for _ in range(23) for row in square.next()[0]]  # two passes of 23 rows
    assert sorted(seen) == sorted(2 * keys[3:])


def test_a_caption_past_the_limit_is_cut_with_a_warning(monkeypatch):
    monkeypatch.setattr(data, "MAX_TOKENS", 4)
    with pytest.warns(UserWarning, match="cut to 4"):
        tokens, mask = pad_sequences([torch.ones(2, SEQ_DIM), torch.ones(6, SEQ_DIM)], torch.float32)
    assert tokens.shape == (2, 4, SEQ_DIM) and mask.sum(1).tolist() == [2, 4]


def test_buckets_are_drawn_by_their_rows_and_a_micro_batch_is_one_bucket(mixed):
    out, _ = mixed
    stream = Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=0)
    assert sorted(b.rows for b in stream.buckets) == [6, 26]
    draws = [b.shape for b in stream.draw(4000)]
    assert abs(draws.count((2, 14, 18)) / 4000 - 6 / 32) < 0.02
    for keys in stream_keys(stream, 20, keys_of(out)):
        assert len({k[0] == "k" or int(k[1:]) % 2 for k in keys}) == 1  # square or wide, never both


def test_the_buffers_are_reused_and_the_order_follows_the_seed(mixed):
    out, _ = mixed
    by_latent = keys_of(out)

    def first_keys(seed):
        stream = Stream(str(out), resolution=32, batch=2, device=torch.device("cpu"), dtype=torch.float32, seed=seed)
        pointers = {id(b): b.next()[0].data_ptr() for b in stream.buckets}
        keys = stream_keys(stream, 12, by_latent)
        assert all(b.next()[0].data_ptr() == pointers[id(b)] for b in stream.buckets)
        return keys

    assert first_keys(0) == first_keys(0) != first_keys(1)


def test_sequences_pad_to_the_longest_caption_and_at_least_the_null_prompt():
    tokens, mask = pad_sequences([torch.ones(3, SEQ_DIM), torch.ones(5, SEQ_DIM)], torch.float32)
    assert tokens.shape == (2, 5, SEQ_DIM) and mask.sum(1).tolist() == [3, 5]
    assert not tokens[0, 3:].any() and tokens[0, :3].eq(1).all()
    assert pad_sequences([torch.ones(1, SEQ_DIM)])[0].shape == (1, NULL_TOKENS, SEQ_DIM)
