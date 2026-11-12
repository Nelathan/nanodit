"""Training-step speed of the block layouts across resolutions: forward + backward images per second at a fixed
token budget per batch, to regress cost against the token count and project it to 1024 px.

usage: bench_speed.py
"""
import time

import torch

from nanodit.data import pad_sequences
from nanodit.model import NULL_TOKENS, SEQ_DIM, BlockType, DiT, Prompt

TOKENS_PER_BATCH = 4096  # 64 images at 128 px
RESOLUTIONS = (128, 256, 512, 1024)
CONV, ATTN = BlockType("conv"), BlockType("attn")
ARMS = {  # name: DiT keywords
    "attn": dict(layout=(ATTN,)),
    "conv2": dict(layout=(CONV, ATTN)),
    "conv4": dict(layout=(CONV, CONV, CONV, ATTN)),
}
WARMUP, STEPS = 3, 8


def bench(kw: dict, resolution: int, device: torch.device) -> float:
    torch.manual_seed(0)
    model = DiT(torch.randn(768), torch.randn(NULL_TOKENS, SEQ_DIM), depth=12, width=768, **kw).to(device, torch.bfloat16)
    for block in model.blocks:
        block.compile()
    tokens = (resolution // 16) ** 2
    batch = max(1, TOKENS_PER_BATCH // tokens)
    x = torch.randn(batch, 2, resolution // 2, resolution // 2, device=device)
    tokens, mask = pad_sequences([torch.randn(40, SEQ_DIM)] * batch, torch.bfloat16, device)
    text = Prompt(torch.randn(batch, 768, device=device, dtype=torch.bfloat16), tokens, mask)
    t = torch.rand(batch, device=device)
    for i in range(WARMUP + STEPS):
        if i == WARMUP:
            torch.cuda.synchronize()
            start = time.perf_counter()
        model(x, t, text).square().mean().backward()
    torch.cuda.synchronize()
    return STEPS * batch / (time.perf_counter() - start)


if __name__ == "__main__":
    device = torch.device("cuda")
    for name, kw in ARMS.items():
        row = {}
        for resolution in RESOLUTIONS:
            try:
                row[resolution] = bench(kw, resolution, device)
            except torch.OutOfMemoryError:
                row[resolution] = float("nan")
            torch.cuda.empty_cache()
        print(name, {r: round(v, 1) for r, v in row.items()}, flush=True)
