"""Training-step speed of the block layouts across resolutions: forward + backward images per second at a fixed
token budget per batch, to regress cost against the token count and project it to 1024 px.

usage: bench_speed.py
"""
import time

import torch

from tinydit.model import DiT, HEAD_DIM

TOKENS_PER_BATCH = 4096  # 64 images at 128 px
RESOLUTIONS = (128, 256, 512, 1024)
ARMS = {  # name: DiT keywords
    "attn": dict(conv_period=1),
    "conv2": dict(conv_period=2),
    "conv4": dict(conv_period=4),
    "conv4_gqa3": dict(conv_period=4, kv_heads=3),
    "attn_gqa3": dict(conv_period=1, kv_heads=3),
}
WARMUP, STEPS = 3, 8


def bench(kw: dict, resolution: int, device: torch.device) -> float:
    torch.manual_seed(0)
    model = DiT(torch.randn(768), depth=12, width=768, **kw).to(device, torch.bfloat16)
    for block in model.blocks:
        block.compile()
    tokens = (resolution // 16) ** 2
    batch = max(1, TOKENS_PER_BATCH // tokens)
    x, text = torch.randn(batch, 2, resolution // 2, resolution // 2, device=device), torch.randn(batch, 768, device=device)
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
