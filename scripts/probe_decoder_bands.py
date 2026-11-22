"""How the decoder weighs a latent error by spatial frequency: whether the latent MSE the loss counts is the image MSE
it costs. Each row's latent gets a Gaussian error confined to one radial band of latent frequency, of a fixed RMS; the
gain is the decoded image's mean-square change over the error's (pixels in [-1, 1]). Also each band's share of the
latents' energy, and where in the image's spectrum the change lands. CPU, fp32.

usage: probe_decoder_bands.py EVAL_SET RESOLUTION ROWS
"""
import sys

import torch

from nanodit.data import read_rows
from train import load_decoder

EDGES = (1 / 64, 1 / 32, 1 / 16, 1 / 8, 1 / 4, 1 / 2)  # cycles per latent pixel; a token is 8 latent pixels wide
RMS = (0.05, 0.3)  # against the latents' RMS of about 1


def radius(h: int, w: int) -> torch.Tensor:
    fy, fx = torch.fft.fftfreq(h), torch.fft.fftfreq(w)
    return (fy[:, None].square() + fx[None, :].square()).sqrt()


def band_masks(h: int, w: int) -> list[torch.Tensor]:
    band = torch.bucketize(radius(h, w), torch.tensor(EDGES))
    return [band == i for i in range(len(EDGES) + 1)]


def band_shares(e: torch.Tensor, masks: list[torch.Tensor]) -> list[float]:
    """Each band's share of e's energy, e [B, C, H, W] on the grid the masks were built for."""
    spectrum = torch.fft.fft2(e).abs().square().sum((0, 1))
    return [(spectrum[m].sum() / spectrum.sum()).item() for m in masks]


@torch.no_grad()
def main(eval_set: str, resolution: int, rows: int) -> None:
    torch.manual_seed(0)
    x, _ = read_rows(eval_set, resolution)
    x = x[:rows].float()
    decoder = load_decoder(torch.device("cpu"))
    h, w = x.shape[-2:]
    masks = band_masks(h, w)
    base = decoder.decode(x)
    image_masks = band_masks(*base.shape[-2:])  # cycles per image pixel: half the latent frequency at the 2x2 VAE
    print(f"{rows} rows, latent {tuple(x.shape[1:])}, latent RMS {x.square().mean().sqrt():.3f}")
    print("band (cycles/latent px) | latent energy share | gain at RMS " + " | ".join(str(r) for r in RMS) + " | where the image change lands (image bands, same edges / 2)")
    shares = band_shares(x, masks)
    lows = (0.0,) + EDGES
    highs = EDGES + (0.71,)
    for i, mask in enumerate(masks):
        gains, lands = [], None
        for rms in RMS:
            noise = torch.fft.ifft2(torch.fft.fft2(torch.randn_like(x)) * mask).real
            noise = noise * rms / noise.square().mean().sqrt()
            change = decoder.decode(x + noise) - base
            gains.append((change.square().mean() / noise.square().mean()).item())
            if lands is None:
                lands = band_shares(change, image_masks)
        print(f"{lows[i]:.4f}-{highs[i]:.4f} | {shares[i]:.4f} | " + " | ".join(f"{g:.4f}" for g in gains) + " | " + " ".join(f"{v:.2f}" for v in lands), flush=True)


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
