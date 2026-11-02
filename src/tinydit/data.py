"""Micro-batches streamed from one shuffled Parquet file of precomputed rows (SPEC §1, `scripts/shuffle.py`)."""

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .model import LATENT_CHANNELS, TEXT_DIM

BLOCK_ROW_GROUPS = 16  # rows read and reshuffled together: 16 row groups of up to 1024 rows, about 290 MB at 128 px


def _bf16(column, *shape: int) -> Tensor:
    """A fixed_size_binary column of bf16 rows as one [rows, *shape] bf16 tensor."""
    array = column.combine_chunks()
    width = array.type.byte_width
    raw = np.frombuffer(array.buffers()[1], dtype=np.uint8, count=len(array) * width, offset=array.offset * width)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).view(len(array), *shape)


def read_rows(path: str, resolution: int) -> tuple[Tensor, Tensor]:
    """Every row of a precomputed Parquet, in file order: bf16 latents [N, 2, res/2, res/2] and embeddings [N, 768]."""
    table = pq.read_table(path, columns=["latent", "embedding"])
    side = resolution // 2
    return _bf16(table["latent"], LATENT_CHANNELS, side, side), _bf16(table["embedding"], TEXT_DIM)


class Stream:
    """Micro-batches of latents and embeddings, read from the file in blocks of row groups and copied into
    buffers allocated once, cast to `dtype` on the way. Each pass over the file visits the blocks in a fresh
    random order and shuffles the rows inside each block, so batches differ between epochs while the file is
    read a block at a time. A thread reads the next block while the current one is consumed; a micro-batch
    that runs past a block's end continues into the next."""

    def __init__(self, path: str, resolution: int, batch: int, device: torch.device, dtype: torch.dtype, seed: int):
        self.file = pq.ParquetFile(path)
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
        self.position = 0
        self.x = torch.empty(batch, LATENT_CHANNELS, self.side, self.side, dtype=dtype, device=device)
        self.text = torch.empty(batch, TEXT_DIM, dtype=dtype, device=device)

    def _read_next(self):
        if not self.order:
            self.order = self.rng.permutation(len(self.blocks)).tolist()
        block = self.order.pop()
        return self.reader.submit(self._read, self.blocks[block], torch.from_numpy(self.rng.permutation(self.rows[block])))

    def _read(self, groups: list[int], shuffle: Tensor) -> tuple[Tensor, Tensor]:
        table = self.file.read_row_groups(groups, columns=["latent", "embedding"])
        return _bf16(table["latent"], LATENT_CHANNELS, self.side, self.side)[shuffle], _bf16(table["embedding"], TEXT_DIM)[shuffle]

    def next(self) -> tuple[Tensor, Tensor]:
        n = len(self.x)
        while len(self.latents) - self.position < n:
            latents, embeddings = self.pending.result()
            self.latents = torch.cat([self.latents[self.position :], latents])
            self.embeddings = torch.cat([self.embeddings[self.position :], embeddings])
            self.position = 0
            self.pending = self._read_next()  # after the cat, so the reader's copies and the cat's never coexist
        self.x.copy_(self.latents[self.position : self.position + n])
        self.text.copy_(self.embeddings[self.position : self.position + n])
        self.position += n
        return self.x, self.text
