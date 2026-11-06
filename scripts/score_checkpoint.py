"""Score a finished run's checkpoint on a fused eval set at another resolution: how a position code, trained at 128 px,
carries to a bigger token grid. Same caption loss and noise rule as the training score; appends to RUN_DIR/score_<res>.jsonl.

usage: score_checkpoint.py RUN_DIR STEP EVAL_SET RESOLUTION
"""
import sys
import tomllib
from pathlib import Path

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from tinydit.data import LATENT_CHANNELS, TEXT_DIM, _bf16
from tinydit.model import DiT
from tinydit.objective import t_quantiles
from train import SCORE_SHIFT, SCORE_TS, Config, append_jsonl, score

BATCH = 16


def main(run_dir: str, step: int, eval_set: str, resolution: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = DiT(load_file(cfg.null_embedding)["null"], cfg.depth, cfg.width, conv_period=cfg.conv_period, meta_embed=cfg.meta_embed,
                parallel_mlp=cfg.parallel_mlp, rope_fraction=cfg.rope_fraction, kv_heads=cfg.kv_heads).to(device, torch.bfloat16)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    side = resolution // 2
    table = pq.read_table(eval_set, columns=[f"latent_{resolution}", "embedding"])
    x, text = _bf16(table[f"latent_{resolution}"], LATENT_CHANNELS, side, side), _bf16(table["embedding"], TEXT_DIM)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))
    ts = t_quantiles(SCORE_TS, SCORE_SHIFT)
    scores = score(model, x, text, noise, ts, BATCH, device)
    append_jsonl(run / f"score_{resolution}.jsonl", dict(step=step, t=ts.tolist()) | scores)
    print(f"{run.name} step {step} at {resolution}: caption loss {scores['loss_caption']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
