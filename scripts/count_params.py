"""Parameter counts of the DiT per role, for shapes of the current model. `image` is what a scaling law counts as N
(the blocks' mixers and MLPs and the patch embed); `conditioning` is the timestep MLP, the shared adaLN projection,
the per-block tables, the caption refiner, the caption's keys and values in each attention block, the null keys
and the null prompt. Built on the meta device.

usage: count_params.py WIDTHxDEPTH [WIDTHxDEPTH ...]
"""
import sys

import torch

from nanodit.model import NULL_TOKENS, SEQ_DIM, DiT

CONDITIONING = ("time_mlp", "ada", "null_", ".table", ".cross_", "refiner.")


def counts(width: int, depth: int) -> dict[str, int]:
    with torch.device("meta"):
        model = DiT(torch.zeros(NULL_TOKENS, SEQ_DIM), depth, width)
    split = dict(image=0, conditioning=0)
    for name, p in model.named_parameters():
        split["conditioning" if any(c in name for c in CONDITIONING) else "image"] += p.numel()
    return split | dict(total=sum(split.values()))


if __name__ == "__main__":
    for shape in sys.argv[1:]:
        width, depth = map(int, shape.split("x"))
        c = counts(width, depth)
        print(f"d{width}x{depth}: total {c['total'] / 1e6:.1f}M, image {c['image'] / 1e6:.1f}M, conditioning {c['conditioning'] / 1e6:.1f}M, width/depth {width / depth:.0f}")
