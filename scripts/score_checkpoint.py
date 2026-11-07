"""Score a finished run's checkpoint on a joined eval set at another resolution: how a position code, trained at 128 px,
carries to a bigger token grid. Same caption loss and noise rule as the training score; appends to RUN_DIR/score_<res>.jsonl.
The eval set comes from `join_sequences.py OUT RESOLUTION SHARD`.

usage: score_checkpoint.py RUN_DIR STEP EVAL_SET RESOLUTION
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from tinydit.data import read_rows
from tinydit.model import DiT
from tinydit.objective import t_quantiles
from train import SCORE_SHIFT, SCORE_TS, Config, append_jsonl, score

BATCH = 16


def main(run_dir: str, step: int, eval_set: str, resolution: int) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    null = load_file(cfg.null_embedding)
    model = DiT(null["null"], null["null_tokens"], cfg.depth, cfg.width, cfg.conv_period, cfg.cross, cfg.pooled_ada, cfg.gdn, cfg.repa_block).to(device, torch.bfloat16)
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    x, prompt = read_rows(eval_set, resolution)
    noise = torch.randn(x.shape, generator=torch.Generator().manual_seed(cfg.eval_seed))
    ts = t_quantiles(SCORE_TS, SCORE_SHIFT)
    scores = score(model, x, prompt, noise, ts, BATCH, device)
    append_jsonl(run / f"score_{resolution}.jsonl", dict(step=step, t=ts.tolist()) | scores)
    print(f"{run.name} step {step} at {resolution}: caption loss {scores['loss_caption']:.4f}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), sys.argv[3], int(sys.argv[4]))
