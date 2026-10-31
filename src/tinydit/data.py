"""Precomputed shards in RAM and random batches from them (SPEC §1)."""

import numpy as np
import pyarrow.parquet as pq
import torch
from torch import Tensor

from .model import LATENT_CHANNELS, TEXT_DIM


def _bf16(column, *shape: int) -> Tensor:
    """A fixed_size_binary column of bf16 rows as one [rows, *shape] bf16 tensor."""
    array = column.combine_chunks()
    width = array.type.byte_width
    raw = np.frombuffer(array.buffers()[1], dtype=np.uint8, count=len(array) * width, offset=array.offset * width)
    return torch.from_numpy(raw.copy()).view(torch.bfloat16).view(len(array), *shape)


def load_shards(paths: list[str], resolution: int) -> tuple[Tensor, Tensor]:
    """All rows of the shards: latents bf16 [N, 2, res/2, res/2] and embeddings bf16 [N, 768]."""
    side = resolution // 2
    tables = [pq.read_table(path, columns=["latent", "embedding"]) for path in paths]
    latents = torch.cat([_bf16(t["latent"], LATENT_CHANNELS, side, side) for t in tables])
    embeddings = torch.cat([_bf16(t["embedding"], TEXT_DIM) for t in tables])
    return latents, embeddings


class Batcher:
    """Random batches (with replacement) copied into buffers allocated once, so every batch has the same
    tensors and shapes. Rows are gathered into a CPU staging buffer (pinned for CUDA), then copied across,
    casting to `dtype` on the way."""

    def __init__(self, latents: Tensor, embeddings: Tensor, batch: int, device: torch.device, dtype: torch.dtype):
        pin = device.type == "cuda"
        self.latents, self.embeddings = latents, embeddings
        self.stage_latents = torch.empty(batch, *latents.shape[1:], dtype=latents.dtype, pin_memory=pin)
        self.stage_embeddings = torch.empty(batch, *embeddings.shape[1:], dtype=embeddings.dtype, pin_memory=pin)
        self.x = torch.empty(batch, *latents.shape[1:], dtype=dtype, device=device)
        self.text = torch.empty(batch, *embeddings.shape[1:], dtype=dtype, device=device)

    def next(self) -> tuple[Tensor, Tensor]:
        index = torch.randint(len(self.latents), (len(self.x),))
        # The staging buffers are rewritten here while the previous non_blocking copy may still read them:
        # safe only because every step synchronises (loss.item() in the train loop) before the next batch.
        torch.index_select(self.latents, 0, index, out=self.stage_latents)
        torch.index_select(self.embeddings, 0, index, out=self.stage_embeddings)
        self.x.copy_(self.stage_latents, non_blocking=True)
        self.text.copy_(self.stage_embeddings, non_blocking=True)
        return self.x, self.text
