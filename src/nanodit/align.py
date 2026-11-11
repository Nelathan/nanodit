"""REPA's target (topics/priming.md): EmbeddingGemma 2's frozen vision tower on the decoded clean image, one feature
per token of our grid."""

import torch
import torch.nn.functional as F
from torch import Tensor

from .model import PATCH

IMAGE_PX = 2  # image pixels per latent pixel, Primal 2x2
VISION_PATCH = 16
VISION_POOL = 3  # the tower pools 3x3 of its patches into one soft token
UPSAMPLE = VISION_POOL * VISION_PATCH // (PATCH * IMAGE_PX)  # 3: a token's 16 px become 3x3 patches, 128 px become 384


class VisionTarget:
    """The tower's pooled soft tokens of the decoded image, upsampled so each pools exactly one of our tokens: at
    128 px the image goes in at 384, the processor's smallest budget for a square (FACTS)."""

    def __init__(self, decoder, device: torch.device):
        from transformers import AutoModel

        full = AutoModel.from_pretrained("google/embeddinggemma-2", audio_config=None, dtype=torch.bfloat16)
        self.tower = full.vision_tower.to(device).eval().requires_grad_(False)
        self.decoder = decoder

    @torch.no_grad()
    def __call__(self, x: Tensor) -> Tensor:
        """[B, 2, H, W] clean latents -> [B, H*W / PATCH², 768] features, row-major as `patchify`'s tokens."""
        images = self.decoder.decode(x.float()).clamp(-1, 1)
        images = F.interpolate((images + 1) / 2, scale_factor=UPSAMPLE, mode="bicubic").clamp(0, 1)
        B, C, H, W = images.shape
        rows, cols = H // VISION_PATCH, W // VISION_PATCH
        # the processor's patch layout: patches row-major, each flattened as (row, col, channel)
        patches = images.view(B, C, rows, VISION_PATCH, cols, VISION_PATCH).permute(0, 2, 4, 3, 5, 1).reshape(B, rows * cols, -1)
        row, col = torch.meshgrid(torch.arange(rows, device=x.device), torch.arange(cols, device=x.device), indexing="ij")
        positions = torch.stack([col, row], dim=-1).view(1, rows * cols, 2).expand(B, -1, -1)  # (x, y) per patch
        pooled = self.tower(pixel_values=patches.to(torch.bfloat16), pixel_position_ids=positions).last_hidden_state
        return pooled.view(B, (rows // VISION_POOL) * (cols // VISION_POOL), -1)
