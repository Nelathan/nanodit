"""Micro-batches streamed from a training set of `scripts/shuffle.py`: one shuffled Parquet per aspect bucket (SPEC §1)."""

import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .model import LATENT_CHANNELS, NULL_TOKENS, SEQ_DIM, TEXT_DIM, Prompt

EXCLUDED = "excluded_keys.txt"  # in a set's directory: the keys of rows never streamed, one per line (SPEC §1)
MAX_TOKENS = 512  # token states a caption keeps; a longer one is cut, with a warning (SPEC §1)
BLOCK_ROW_GROUPS = 4  # rows read and reshuffled together: 4 row groups of up to 1024 rows, about 180 MB at 128 px; two per bucket in memory


def _bf16(column, *shape: int) -> Tensor:
    """A fixed_size_binary column of bf16 rows as one [rows, *shape] bf16 tensor."""
    array = column.combine_chunks()
    width = array.type.byte_width
    raw = np.frombuffer(array.buffers()[1], dtype=np.uint8, count=len(array) * width, offset=array.offset * width)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).view(len(array), *shape)


def _sequences(column) -> list[Tensor]:
    """A binary column of bf16 [n_i, 512] rows as one [n_i, 512] bf16 view per row, onto one buffer per chunk."""
    rows = []
    for chunk in column.chunks:
        offsets = np.frombuffer(chunk.buffers()[1], dtype=np.int32, count=len(chunk) + 1, offset=chunk.offset * 4)
        assert not ((offsets - offsets[0]) % (2 * SEQ_DIM)).any(), "a sequence column of another width"
        raw = np.frombuffer(chunk.buffers()[2], dtype=np.uint8, count=offsets[-1] - offsets[0], offset=offsets[0])
        states = torch.from_numpy(raw.copy()).view(torch.bfloat16).view(-1, SEQ_DIM)
        bounds = ((offsets - offsets[0]) // (2 * SEQ_DIM)).tolist()
        rows += [states[a:b] for a, b in zip(bounds[:-1], bounds[1:])]
    return rows


def pad_sequences(sequences: list[Tensor], dtype: torch.dtype = torch.bfloat16, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """Token states [B, L, 512], zero past each row's end, and the mask [B, L] of the real tokens; L is the longest row,
    at least NULL_TOKENS, at most MAX_TOKENS."""
    if any(len(s) > MAX_TOKENS for s in sequences):
        warnings.warn(f"a caption of {max(map(len, sequences))} token states is cut to {MAX_TOKENS}", stacklevel=2)
        sequences = [s[:MAX_TOKENS] for s in sequences]
    length = max(NULL_TOKENS, *(len(s) for s in sequences))
    tokens = torch.zeros(len(sequences), length, SEQ_DIM, dtype=dtype, device=device)
    for row, states in zip(tokens, sequences):
        row[: len(states)] = states
    mask = torch.arange(length, device=device) < torch.tensor([len(s) for s in sequences], device=device)[:, None]
    return tokens, mask


def read_rows(path: str, resolution: int) -> tuple[Tensor, Prompt]:
    """Every row of a precomputed Parquet, in file order: bf16 latents [N, 2, res/2, res/2] and their prompts."""
    table = pq.read_table(path, columns=[f"latent_{resolution}", "embedding", "sequence"])
    side = resolution // 2
    tokens, mask = pad_sequences(_sequences(table["sequence"]))
    return _bf16(table[f"latent_{resolution}"], LATENT_CHANNELS, side, side), Prompt(_bf16(table["embedding"], TEXT_DIM), tokens, mask)


class Bucket:
    """The rows of one aspect bucket's file as micro-batches, copied into buffers allocated once and cast to `dtype`.
    Each pass over the file deals its row groups into new random blocks and visits them in random order, shuffling
    the rows inside each block, so rows meet different rows in every pass. The reader thread reads the next block
    while the current one is consumed; a micro-batch that runs past a block's end continues into the next."""

    def __init__(self, path: Path, resolution: int, batch: int, device: torch.device, dtype: torch.dtype, rng: np.random.Generator, reader: ThreadPoolExecutor, excluded: frozenset[str]):
        self.file = pq.ParquetFile(path)
        self.excluded = excluded
        self.column = f"latent_{resolution}"
        height, width = map(int, path.stem.split("x"))
        self.shape = (LATENT_CHANNELS, height // 2, width // 2)
        self.rows = self.file.metadata.num_rows
        self.rng, self.reader = rng, reader  # drawn on the caller's thread only, so the order follows the seed
        self.blocks: list[list[int]] = []
        self.pending = self._read_next()
        self.latents = torch.empty(0, *self.shape, dtype=torch.bfloat16)
        self.embeddings = torch.empty(0, TEXT_DIM, dtype=torch.bfloat16)
        self.sequences: list[Tensor] = []
        self.position = 0
        self.dtype, self.device = dtype, device
        self.x = torch.empty(batch, *self.shape, dtype=dtype, device=device)
        self.pooled = torch.empty(batch, TEXT_DIM, dtype=dtype, device=device)

    def _read_next(self):
        if not self.blocks:  # a new pass
            groups = self.rng.permutation(self.file.num_row_groups).tolist()
            self.blocks = [groups[i : i + BLOCK_ROW_GROUPS] for i in range(0, len(groups), BLOCK_ROW_GROUPS)]
        groups = self.blocks.pop()
        rows = sum(self.file.metadata.row_group(g).num_rows for g in groups)
        return self.reader.submit(self._read, groups, torch.from_numpy(self.rng.permutation(rows)))

    def _read(self, groups: list[int], shuffle: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        table = self.file.read_row_groups(groups, columns=[self.column, "embedding", "sequence", "key"])
        keys = table["key"].to_pylist()
        # the permutation is drawn over every row, so the order follows the seed whatever is excluded
        order = torch.tensor([i for i in shuffle.tolist() if keys[i] not in self.excluded], dtype=torch.long)
        sequences = _sequences(table["sequence"])
        return _bf16(table[self.column], *self.shape)[order], _bf16(table["embedding"], TEXT_DIM)[order], [sequences[i] for i in order.tolist()]

    def next(self) -> tuple[Tensor, Prompt]:
        n = len(self.x)
        while len(self.latents) - self.position < n:
            latents, embeddings, sequences = self.pending.result()
            self.latents = torch.cat([self.latents[self.position :], latents])
            self.embeddings = torch.cat([self.embeddings[self.position :], embeddings])
            self.sequences = self.sequences[self.position :] + sequences
            self.position = 0
            self.pending = self._read_next()  # after the cat, so the reader's copies and the cat's never coexist
        rows = slice(self.position, self.position + n)
        self.x.copy_(self.latents[rows])
        self.pooled.copy_(self.embeddings[rows])
        self.position += n
        tokens, mask = pad_sequences(self.sequences[rows])  # on the CPU, so the GPU gets one copy of each
        return self.x, Prompt(self.pooled, tokens.to(self.device, self.dtype), mask.to(self.device))


class Stream:
    """Micro-batches from a training set directory, each from one aspect bucket, the bucket drawn with probability
    proportional to its rows, so over a pass every bucket is visited as often as it has rows (excluded rows counted:
    they are a few in millions). The order follows the seed."""

    def __init__(self, path: str, resolution: int, batch: int, device: torch.device, dtype: torch.dtype, seed: int):
        self.rng = np.random.default_rng(seed)
        reader = ThreadPoolExecutor(1)
        files = sorted(Path(path).glob("*.parquet"))
        assert files, f"no bucket files in {path}"
        excluded = frozenset((Path(path) / EXCLUDED).read_text().split())
        self.buckets = [Bucket(f, resolution, batch, device, dtype, self.rng, reader, excluded) for f in files]
        rows = np.array([b.rows for b in self.buckets], dtype=float)
        self.p = rows / rows.sum()

    def draw(self, micro_batches: int) -> list[Bucket]:
        """The buckets of the next `micro_batches` micro-batches, to be read in this order with `Bucket.next`."""
        return [self.buckets[i] for i in self.rng.choice(len(self.buckets), size=micro_batches, p=self.p)]
