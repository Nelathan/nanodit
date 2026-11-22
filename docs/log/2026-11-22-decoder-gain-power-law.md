# The decoder's gain as a power law of latent frequency

`scripts/probe_decoder_bands.py heldout_128_fine_flux2.parquet 128 16`, the probe of log 2026-11-22-decoder-bands with a
fit added; the same rows and seed, the same table. The fit is least squares of log gain (error RMS 0.05) on log
frequency, each band at its bins' mean radius, over the five bands between the lowest (which holds the DC) and the
diagonal corners.

| Fit | β in g ∝ f^-β | Worst band off the line |
|---|---|---|
| bands 1/64 – 1/2 cycles per latent px | 1.518 | factor 1.27 |

## Read

- The gain falls as f^-1.52 from periods of 64 latent pixels down to 2, within 27 % at every band.
- The corners (above 1/2, diagonal) leave the line: their gain, 0.072, is above that of the band below them,
  0.023, where the line keeps falling.
