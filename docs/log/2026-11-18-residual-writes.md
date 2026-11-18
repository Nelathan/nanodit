# Where the prediction forms: the image channels after every write, a1-ref-1ep at 32173

Probe: `scripts/probe_residual_writes.py /mnt/mars/ai/tiny-dit/runs/a1-ref-1ep 32173
/mnt/mars/ai/tiny-dit/probes/residual_writes_a1-ref-1ep_32173` (256 Lunara eval rows, the score's noise and 10 t; the
eval sampler on prompts 1, 6, 9, 10 and 14). The read-out takes the first 128 channels of the stream (SPEC §3), so
the image channels after write k are a partial prediction F_k, and x0_k = (1 − t)·x_t + t·F_k is its clean estimate.
The writes are re-added in bf16 in the model's order and match each block's output exactly.

## Which writes form the prediction

Caption loss of F_k at t 0.47 (the input's F_0 = x_t gives |v|², 2.1588 at every t), and the RMS of each write on the
image channels; the conv mixers and the cross attention are left out of the table, as they write nothing there
(below):

| After | loss | Δ | write RMS | | After | loss | Δ | write RMS |
|---|---|---|---|---|---|---|---|---|
| 0–7 (all) | 2.1416 | −0.0172 | ≤ 0.024 | | 15.attn | 1.9387 | −0.0206 | 0.055 |
| 8.mlp | 2.1198 | −0.0218 | 0.049 | | 15.mlp | 1.8888 | −0.0499 | 0.154 |
| 9.mlp | 2.1092 | −0.0106 | 0.029 | | 16.mlp | 1.8508 | −0.0380 | 0.115 |
| 10.mlp | 2.1007 | −0.0085 | 0.033 | | 17.mlp | 1.8184 | −0.0325 | 0.091 |
| 11.attn | 2.0988 | −0.0019 | 0.011 | | 18.mlp | 1.7697 | −0.0486 | 0.174 |
| 11.mlp | 2.0410 | −0.0579 | 0.137 | | 19.attn | 1.7468 | −0.0229 | 0.058 |
| 12.mlp | 2.0048 | −0.0361 | 0.100 | | 19.mlp | 1.6788 | −0.0680 | 0.213 |
| 13.mlp | 1.9947 | −0.0101 | 0.030 | | 20.mlp | 1.6669 | −0.0119 | 0.052 |
| 14.mlp | 1.9593 | −0.0354 | 0.108 | | 21.mlp | 1.5965 | −0.0704 | 0.227 |
| | | | | | 22.mlp | 1.5447 | −0.0519 | 0.150 |
| | | | | | 23.attn | 1.5064 | −0.0383 | 0.154 |
| | | | | | 23.mlp | 1.4673 | −0.0390 | 0.196 |

- The first two periods (blocks 0–7) write almost nothing to the image channels at mid t: 2.5 % of the loss drop.
  At t 0.84 block 0's MLP alone takes 0.17 off (the largest single write there); at t 0.16 it takes 0.04.
- From block 8 on, the MLPs write most of the prediction; by loss drop at t 0.47 the largest are 21, 19, 11, 22, 15
  and 18. 13.mlp and 20.mlp are small at every t.
- The cross attention writes RMS 0.0001 to 0.004 on the image channels at every t: it writes the meta channels only,
  in effect. Self-attention's image writes are a minority except in the last block.
- No write raises the loss by more than 0.0002 at any t on average (the largest: 0.mlp +0.0001 at t 0.40, 7.attn
  +0.0001 at t 0.60 to 0.84). Per band, one write raises one band's error by more than 0.5 % of that band's energy:
  23.mlp, coarse band, t 0.84, +0.5 %.

## What is left unrecovered, by spatial frequency

The error x0_k − x split into three radial bands of the 64 × 64 latent: coarser than two tokens (period > 16 latent
px), one to two tokens (8 to 16 px), inside one token (< 8 px). The clean latents' energy per band: 0.032, 0.088,
1.036 (90 % of it inside one token). Share of each band's energy not recovered after the last block:

| t | coarse | token | inside-token |
|---|---|---|---|
| 0.16 | 0.01 | 0.01 | 0.03 |
| 0.34 | 0.05 | 0.07 | 0.14 |
| 0.47 | 0.11 | 0.16 | 0.29 |
| 0.60 | 0.22 | 0.32 | 0.49 |
| 0.74 | 0.47 | 0.63 | 0.73 |
| 0.84 | 0.68 | 0.85 | 0.84 |

At every t below 0.74 the inside-token band is recovered worst, about twice the coarse band's share, and the 8–16 px
band sits between them. At t ≤ 0.6 the first two periods leave every band as the input left it; at t 0.74 and 0.84
they recover 6 % and 13 % of the inside-token band.

## In the samples

The decoded files in the probe directory (written by the probe as committed; columns in `columns.txt`):

- `prompt*_trajectory.png`, the whole network's clean estimate at each of the 30 sampler steps. At step 0 (t 1) the
  estimate already carries colour masses from the noise (the chess prompt's two red blobs); steps 1 to 3 wash them
  out; the forms come back and are fixed by step 10 (t 0.67), defects included: the chess prompt's doubled head, the
  cabin prompt's warped roof, and no bear in "a bear building a log cabin". After t 0.67 the estimate sharpens and
  changes no form.
- `prompt*_step08_t0.733.png`, the writes at t 0.73: the cabin and its warped roof are drawn over periods 4 and 5
  (blocks 12 to 19) by a run of MLP writes, none of which alone makes the form; the last period sharpens and colours.
- `prompt*_step03_t0.900.png`: the estimate goes from noise to a blurred layout across the periods; the last attention
  block's write (23.attn) makes the largest visible change. `prompt*_step15_t0.500.png`: the forms are in the input's
  estimate; the writes sharpen and colour, and under CFG 4 the last block pushes saturation (the chess prompt's grass).

## Read

- No sublayer makes the prediction worse on average beyond 0.0002 in loss: the smear is not a write that damages.
  The conv mixers cannot write the image channels (SPEC §3), and the cross attention, in effect, does not either.
- The unrecovered detail is inside the token: the 8 × 8 latent block each token's 128 channels hold.
- The forms, and their defects, are fixed in the sampler between t 0.8 and 0.67, and drawn there by the MLP writes of
  the middle periods over several blocks. Five prompts, one checkpoint, one noise each.
