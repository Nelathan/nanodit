"""Cross attention as the trunk runs it (SDPA over captions padded to the longest row, a boolean key mask, GQA)
against packed captions through `torch.nn.attention.varlen.varlen_attn`: time and peak memory of forward + backward
for one attention block's cross attention, the backend SDPA dispatches to, and the largest output difference.

usage: probe_cross_attention.py MICRO_BATCH
"""
import random
import sys
import time

import torch
import torch.nn.functional as F
from torch.nn.attention.varlen import varlen_attn
from torch.profiler import ProfilerActivity, profile

HEADS, KV_HEADS, HEAD_DIM, QUERIES = 12, 3, 64, 64  # d768, 128 px
WARMUP, REPEATS = 5, 20


def cases(batch: int) -> dict[str, list[int]]:
    rng = random.Random(0)
    typical = [rng.randint(20, 140) for _ in range(batch)]
    return {"typical (20-140)": typical, "one row at 2000": typical[:-1] + [2000], "long (150-330)": [rng.randint(150, 330) for _ in range(batch)]}


def main(micro_batch: str) -> None:
    batch, device = int(micro_batch), torch.device("cuda")
    torch.manual_seed(0)
    for name, lengths in cases(batch).items():
        longest, lens = max(lengths), torch.tensor(lengths, device=device)
        mask = torch.arange(longest, device=device)[None] < lens[:, None]  # [B, L]
        q = torch.randn(batch, HEADS, QUERIES, HEAD_DIM, device=device, dtype=torch.bfloat16, requires_grad=True)
        kv = torch.randn(batch, KV_HEADS, longest, HEAD_DIM, device=device, dtype=torch.bfloat16) * mask[:, None, :, None]
        kv.requires_grad_()
        cu_q = torch.arange(0, batch * QUERIES + 1, QUERIES, device=device, dtype=torch.int32)
        cu_k = F.pad(lens.cumsum(0), (1, 0)).int()

        def padded() -> torch.Tensor:
            return F.scaled_dot_product_attention(q, kv, kv, attn_mask=mask[:, None, None], enable_gqa=True)

        def packed() -> torch.Tensor:
            keys = kv.transpose(1, 2)[mask]  # [total tokens, KV_HEADS, HEAD_DIM]
            out = varlen_attn(q.transpose(1, 2).reshape(batch * QUERIES, HEADS, HEAD_DIM), keys, keys, cu_q, cu_k, QUERIES, longest, enable_gqa=True)
            return out.view(batch, QUERIES, HEADS, HEAD_DIM).transpose(1, 2)

        outputs, row = {}, [f"{name}, mean {sum(lengths) / batch:.0f}, max {longest}:"]
        for label, attend in (("padded", padded), ("varlen", packed)):
            for _ in range(WARMUP):
                attend().float().square().sum().backward()
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            base = torch.cuda.memory_allocated()
            start = time.perf_counter()
            for _ in range(REPEATS):
                attend().float().square().sum().backward()
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - start) / REPEATS
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                attend().float().square().sum().backward()
            kernel = next((e.key for e in prof.key_averages() if "fmha" in e.key or "flash" in e.key.lower() or "cudnn" in e.key.lower()), "?")
            outputs[label] = attend().detach().float()
            row.append(f"{label} {ms:.2f} ms, peak +{(torch.cuda.max_memory_allocated() - base) / 2**20:.0f} MiB, {kernel.split('(')[0][:60]}")
            q.grad = kv.grad = None
        row.append(f"max |diff| {(outputs['padded'] - outputs['varlen']).abs().max().item():.3g}")
        print(" | ".join(row), flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
