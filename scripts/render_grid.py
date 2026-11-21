"""The eval grid of a run's checkpoint at a chosen guidance: the run's prompts (its `eval_prompts.txt`), each with its
fixed noise, the run's sampler steps. For grids at another guidance than the run's eval, e.g. CFG for a README image.

usage: render_grid.py RUN_DIR STEP GUIDANCE OUT_PNG
"""
import sys
import tomllib
from pathlib import Path

import torch
from safetensors.torch import load_file

from train import Config, build_model, embed_prompts, load_decoder, render_grid


def main(run_dir: str, step: int, guidance: float, out_png: str) -> None:
    run = Path(run_dir)
    with open(run / "config.toml", "rb") as f:
        cfg = Config(**tomllib.load(f))
    device = torch.device("cuda")
    model = build_model(cfg, device, torch.bfloat16).eval()
    model.load_state_dict(load_file(run / "checkpoints" / f"step_{step:07d}" / "model.safetensors"))
    prompt = embed_prompts((run / "eval_prompts.txt").read_text().splitlines())
    render_grid(model, load_decoder(device), prompt, cfg, guidance, device).save(out_png)
    print(f"wrote {out_png}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), sys.argv[4])
