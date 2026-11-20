# The adaLN without its shift, 2026-11-20

d768x24 at 128 px, 6000 steps, seed 41, batch 256, at 198 W, both runs with joint attention, the sink and two caption
refiner blocks (log 2026-11-20-joint-attention-and-caption-refiner). Score: caption loss on the Lunara eval set at 10
fixed t; held-out: the same on the FLUX.2-dev held-out rows. One seed; the measured seed spread is 0.0009.

| Run | Commit | Form |
|---|---|---|
| `a3-jointsink-refiner-6k` | `828ff25` | six modulation rows per block: shift, scale and gate for the mixer, then for the MLP |
| `a3-refiner-noshift-6k` | `30b7af9` | four rows: scale and gate per branch, y = rms_norm(x) · (1 + scale); 1.22M parameters fewer |

## Score and held-out by step

| Run | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | img/s |
|---|---|---|---|---|---|---|---|
| `a3-jointsink-refiner-6k` score | 1.5475 | 1.5100 | 1.4910 | 1.4794 | 1.4711 | 1.4653 | 521 |
| `a3-refiner-noshift-6k` score | 1.5469 | 1.5091 | 1.4909 | 1.4793 | 1.4702 | 1.4638 | 523 |
| `a3-jointsink-refiner-6k` held-out | 1.4062 | 1.3581 | 1.3359 | 1.3225 | 1.3126 | 1.3049 | |
| `a3-refiner-noshift-6k` held-out | 1.4063 | 1.3587 | 1.3375 | 1.3233 | 1.3134 | 1.3057 | |

## By t at 6000 (t 0.16 … 0.84)

| Run | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| `a3-jointsink-refiner-6k` score | 1.2826 | 1.3806 | 1.4351 | 1.4763 | 1.5115 | 1.5416 | 1.5632 | 1.5665 | 1.5248 | 1.3704 |
| `a3-refiner-noshift-6k` score | 1.2821 | 1.3795 | 1.4341 | 1.4750 | 1.5100 | 1.5398 | 1.5613 | 1.5640 | 1.5218 | 1.3701 |
| `a3-jointsink-refiner-6k` held-out | 1.2192 | 1.2613 | 1.2813 | 1.2984 | 1.3164 | 1.3359 | 1.3555 | 1.3687 | 1.3553 | 1.2567 |
| `a3-refiner-noshift-6k` held-out | 1.2190 | 1.2611 | 1.2819 | 1.2988 | 1.3171 | 1.3369 | 1.3572 | 1.3703 | 1.3568 | 1.2581 |

The caption-use probe was not run: its form on `arm/joint-probe` reads the six-row layout.

## Read

The two sets move in opposite directions. The score loss is 0.0015 lower (better) without the shift at 6000, no worse at any
step and lower at every t, most at t 0.53–0.74 (0.0018–0.0031). Held-out loss is 0.0009 higher (worse) at 6000 and 0.0006–0.0015
higher from step 2000; by t the gap grows from t 0.34 to 0.0011–0.0017 at t 0.53 and above. Each difference is within twice the seed spread. Throughput is level.
Without the shift the model trains no worse on its own distribution; whether the held-out gap is real is not resolved
by one seed.
