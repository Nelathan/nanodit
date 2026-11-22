"""The decoder-weighted loss over the plain loss at the two ends of t, from the data alone.

At t 0 the input is x and the target (2 - t)·x - (1 - t)·ε leaves ε to guess: the error is white, the ratio 1 by the
weight's mean. At t 1 the input is ε and the target is x: with no prompt the best guess is the set's mean latent, the
error x - mean, its spectrum the data's. The ratio there is the data's spectrum seen through the decoder's gain (objective.spectral_weight
at alpha 1); a prompt removes some of the error, so the model's ratio need not reach it.

Usage: uv run python scripts/probe_target_spectrum.py EVAL_PARQUET RESOLUTION
"""

import sys

import torch

from nanodit.data import read_rows
from nanodit.objective import band_energy, BAND_NAMES, spectral_weight


def main(path: str, resolution: int):
    x = read_rows(path, resolution)[0].float()
    e = x - x.mean(0, keepdim=True)
    power = torch.fft.fft2(e, norm="ortho").abs().square()
    plain = power.mean()
    weighted = (power * spectral_weight(*power.shape[-2:], 1.0)).mean()
    shares = band_energy(e).mean(0) / band_energy(e).mean(0).sum()
    print(f"{path} at {resolution}: {len(x)} rows")
    print(f"t 1, x - mean: plain {plain:.4f}, decoder-weighted {weighted:.4f}, ratio {weighted / plain:.3f}")
    print("energy share of x - mean by band: " + ", ".join(f"{n} {s:.3f}" for n, s in zip(BAND_NAMES, shares.tolist())))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]))
