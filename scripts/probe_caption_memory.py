"""Peak memory and time of a training step against the longest caption in a micro-batch: the step of `train.py` on a
config's model, with one row of each micro-batch given a caption of L token states (random states, the other rows as
drawn). `pad_sequences` pads a micro-batch to its longest row, so this is the cost of one outlier. It
bypasses the cut at `data.MAX_TOKENS`, to measure lengths past it.

usage: probe_caption_memory.py CONFIG.toml MICRO_BATCH L [L ...]
"""
import sys
import time
import tomllib

import torch

from nanodit.data import Stream
from nanodit.model import Prompt
from nanodit.objective import flow_loss
from nanodit.optim import build_optimizers
from train import Config, build_model, compile_blocks

WARMUP, STEPS = 10, 5


def with_long_row(prompt: Prompt, length: int) -> Prompt:
    """The prompt padded to `length`, its first row a caption of `length` random token states."""
    batch, current, dim = prompt.tokens.shape
    tokens = prompt.tokens.new_zeros(batch, max(length, current), dim)
    mask = prompt.mask.new_zeros(batch, tokens.shape[1])
    tokens[:, :current], mask[:, :current] = prompt.tokens, prompt.mask
    tokens[0, :length], mask[0], mask[0, :length] = torch.randn(length, dim, dtype=tokens.dtype, device=tokens.device), False, True
    return Prompt(tokens, mask)


def main(config_path: str, micro_batch: str, *lengths: str) -> None:
    with open(config_path, "rb") as f:
        cfg = Config(**tomllib.load(f))
    torch.manual_seed(cfg.seed)
    device = torch.device("cuda")
    micro_batch = int(micro_batch)
    model = build_model(cfg, device, torch.bfloat16)
    stream = Stream(cfg.dataset, cfg.resolution, micro_batch, device, torch.float32, cfg.seed)
    optimizers = build_optimizers(model, cfg.usui_lr, cfg.beta, cfg.rank_fraction, cfg.fallback_lr)
    compile_blocks(model)
    accumulate = cfg.batch // micro_batch

    def step(length: int) -> None:
        for bucket in stream.draw(accumulate):
            x, prompt = bucket.next()
            loss, _, _ = flow_loss(model, x, with_long_row(prompt, length), cfg.shift, cfg.t_sigma)
            (loss / accumulate).backward()
        for optimizer in optimizers:
            optimizer.step()
            optimizer.zero_grad()

    for length in map(int, lengths):
        for _ in range(WARMUP):
            step(length)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start = time.perf_counter()
        for _ in range(STEPS):
            step(length)
        torch.cuda.synchronize()
        ms = 1000 * (time.perf_counter() - start) / STEPS
        print(f"micro_batch {micro_batch} L {length}: {ms:.1f} ms/step, peak {torch.cuda.max_memory_allocated() / 2**30:.2f} GiB", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
