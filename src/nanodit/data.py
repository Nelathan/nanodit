"""Micro-batches streamed from one shuffled Parquet file of precomputed rows (SPEC §1, `scripts/shuffle.py`)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .model import LATENT_CHANNELS, NULL_TOKENS, SEQ_DIM, TEXT_DIM, Prompt

BLOCK_ROW_GROUPS = 16  # rows read and reshuffled together: 16 row groups of up to 1024 rows, about 290 MB at 128 px


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
    """Token states [B, L, 512], zero past each row's end, and the mask [B, L] of the real tokens; L is the longest row, at least NULL_TOKENS."""
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


class Stream:
    """Micro-batches of latents and prompts, read from the file in blocks of row groups and copied into
    buffers allocated once, cast to `dtype` on the way. Each pass over the file visits the blocks in a fresh
    random order and shuffles the rows inside each block, so batches differ between epochs while the file is
    read a block at a time. A thread reads the next block while the current one is consumed; a micro-batch
    that runs past a block's end continues into the next. The token states are several times the latent's bytes
    at 128 px, so they are read only with `sequences`, for a model that attends to them; without it every prompt
    carries NULL_TOKENS masked-out zero tokens."""

    def __init__(self, path: str, resolution: int, batch: int, device: torch.device, dtype: torch.dtype, seed: int, sequences: bool):
        self.file = pq.ParquetFile(path)
        self.read_sequences = sequences
        self.column = f"latent_{resolution}"
        self.side = resolution // 2
        groups = self.file.num_row_groups
        self.blocks = [list(range(i, min(i + BLOCK_ROW_GROUPS, groups))) for i in range(0, groups, BLOCK_ROW_GROUPS)]
        self.rows = [sum(self.file.metadata.row_group(g).num_rows for g in block) for block in self.blocks]
        self.rng = np.random.default_rng(seed)  # drawn on this thread only, so the order follows the seed
        self.order: list[int] = []
        self.reader = ThreadPoolExecutor(1)
        self.pending = self._read_next()
        self.latents = torch.empty(0, LATENT_CHANNELS, self.side, self.side, dtype=torch.bfloat16)
        self.embeddings = torch.empty(0, TEXT_DIM, dtype=torch.bfloat16)
        self.sequences: list[Tensor] = []
        self.position = 0
        self.dtype, self.device = dtype, device
        self.x = torch.empty(batch, LATENT_CHANNELS, self.side, self.side, dtype=dtype, device=device)
        self.pooled = torch.empty(batch, TEXT_DIM, dtype=dtype, device=device)
        self.no_tokens = torch.zeros(batch, NULL_TOKENS, SEQ_DIM, dtype=dtype, device=device)
        self.no_mask = torch.zeros(batch, NULL_TOKENS, dtype=torch.bool, device=device)

    def _read_next(self):
        if not self.order:
            self.order = self.rng.permutation(len(self.blocks)).tolist()
        block = self.order.pop()
        return self.reader.submit(self._read, self.blocks[block], torch.from_numpy(self.rng.permutation(self.rows[block])))

    def _read(self, groups: list[int], shuffle: Tensor) -> tuple[Tensor, Tensor, list[Tensor]]:
        table = self.file.read_row_groups(groups, columns=[self.column, "embedding"] + ["sequence"] * self.read_sequences)
        sequences = _sequences(table["sequence"]) if self.read_sequences else []
        return (
            _bf16(table[self.column], LATENT_CHANNELS, self.side, self.side)[shuffle],
            _bf16(table["embedding"], TEXT_DIM)[shuffle],
            [sequences[i] for i in shuffle.tolist()] if self.read_sequences else [],
        )

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
        if not self.read_sequences:
            return self.x, Prompt(self.pooled, self.no_tokens, self.no_mask)
        tokens, mask = pad_sequences(self.sequences[rows])  # on the CPU, so the GPU gets one copy of each
        return self.x, Prompt(self.pooled, tokens.to(self.device, self.dtype), mask.to(self.device))
