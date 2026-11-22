# What a latent error costs in the image, by spatial frequency

`scripts/probe_decoder_bands.py heldout_128_fine_flux2.parquet 128 16`, `main` after the uniform score: 16 held-out
latents (2 × 64 × 64, RMS 1.056), primal_vae 2x2 decoder, CPU fp32. Each row's latent gets a Gaussian error confined
to one radial band of latent frequency; the gain is the decoded image's mean-square change (pixels in [-1, 1]) over
the error's. A token is 8 latent pixels wide: the score's bands put "over three tokens" below 1/24 and "inside one
token" above 1/8.

| Band, cycles per latent px | Period, latent px | Share of the latents' energy | Gain at error RMS 0.05 | at 0.3 | Image change in the matching image band |
|---|---|---|---|---|---|
| 0 – 1/64 | > 64 | 0.010 | 2.57 | 2.04 | 0.84 |
| 1/64 – 1/32 | 32 – 64 | 0.006 | 1.16 | 0.93 | 0.76 |
| 1/32 – 1/16 | 16 – 32 | 0.022 | 0.48 | 0.39 | 0.74 |
| 1/16 – 1/8 | 8 – 16 | 0.071 | 0.119 | 0.116 | 0.71 |
| 1/8 – 1/4 | 4 – 8 | 0.181 | 0.043 | 0.047 | 0.64 |
| 1/4 – 1/2 | 2 – 4 | 0.459 | 0.023 | 0.024 | 0.49 |
| > 1/2 (the corners) | < 2, diagonal | 0.251 | 0.072 | 0.066 | 0.22 (scattered; 0.26 in the lowest) |

The image bands use the same edges halved (the decoder doubles the side); the first two latent bands both fall in the
lowest image band.

## Read

- A unit of latent error costs about 110 times more image error at periods over 64 latent pixels than at 2 to 4; the
  gain falls by about 2 to 4 per octave, steepest above a token's width.
- The latents' energy is the reverse: 71 % sits at periods under 4 latent pixels, inside half a token, and under 4 %
  above two tokens. The latent MSE, which weights every coordinate alike, is dominated by the bands the decoder
  weights least.
- The decoder is close to frequency-preserving: a band's error lands mostly in the matching image band, except the
  diagonal corners, which scatter. The gains at the two error sizes agree to within 20 %, so the read is the
  decoder's linear response; the lowest bands lose a fifth at the larger error.
- One noise draw per band and row over 16 rows; the gains span two orders of magnitude, far beyond that resolution.
  Image MSE is not perception: the eye weights edges, which pixel MSE weights less than the layout. The table
  says what latent MSE buys in pixel MSE, not what it buys in looks.
