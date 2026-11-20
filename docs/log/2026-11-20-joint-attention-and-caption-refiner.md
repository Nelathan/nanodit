# Joint attention, its sink, the adaLN without the caption, and the caption refiner, 2026-11-20

d768x24 at 128 px, 6000 steps, seed 41, batch 256, at 198 W. Score: caption loss on the Lunara eval set at 10 fixed t;
held-out: the same on the FLUX.2-dev held-out rows, enhanced prompts, longer than the Lunara captions. Caption use:
`scripts/probe_caption_use.py` over the first 256 eval rows (the a2 runs with its form on `arm/joint-probe-a2`
`7d8fa8f`, the a3 runs on `arm/joint-probe` `fda6c99`), output in each run's `probes/caption_use.json`.

| Run | Commit | Form |
|---|---|---|
| `a1-lfmnormread-6k` | `arm/lfm-forms` | the trunk's self then cross attention, pooled adaLN |
| `a2-joint-6k` | `e672310` | joint attention over image and caption keys, no cross sublayer, pooled adaLN |
| `a2-jointsink-6k` | `22b0423` | joint, a null key per KV head (randn init), pooled adaLN |
| `a2-jointsink-nopool-6k` | `22b0423` | as `a2-jointsink-6k`, the adaLN on t alone |
| `a2-jointsink-captionread-6k` | `720741d` | as `a2-jointsink-6k`, a t-queried caption read in place of the pooled embedding |
| `a3-jointsink-6k` | `2ba3c6d` | a3 (adaLN on t alone), joint with the sink, the null key at zero |
| `a3-jointsink-refiner-6k` | `828ff25` | as `a3-jointsink-6k`, two caption refiner blocks at 512 over the token states |

## Score and held-out by step

| Run | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 | held-out 6000 | img/s |
|---|---|---|---|---|---|---|---|---|
| `a1-lfmnormread-6k` | 1.5488 | 1.5101 | 1.4903 | 1.4791 | 1.4699 | 1.4635 | 1.3069 | 524 |
| `a2-joint-6k` | 1.5476 | 1.5102 | 1.4910 | 1.4808 | 1.4714 | 1.4658 | 1.3087 | 554 |
| `a2-jointsink-6k` | 1.5488 | 1.5109 | 1.4912 | 1.4801 | 1.4708 | 1.4638 | 1.3076 | 552 |
| `a2-jointsink-nopool-6k` | 1.5466 | 1.5110 | 1.4920 | 1.4808 | 1.4711 | 1.4655 | 1.3082 | 553 |
| `a2-jointsink-captionread-6k` | 1.5484 | 1.5106 | 1.4911 | 1.4790 | 1.4700 | 1.4639 | 1.3067 | 547 |
| `a3-jointsink-6k` | 1.5471 | 1.5088 | 1.4900 | 1.4796 | 1.4704 | 1.4647 | 1.3071 | 556 |
| `a3-jointsink-refiner-6k` | 1.5475 | 1.5100 | 1.4910 | 1.4794 | 1.4711 | 1.4653 | 1.3049 | 521 |

Held-out by step for the a3 pair:

| Run | 1000 | 2000 | 3000 | 4000 | 5000 | 6000 |
|---|---|---|---|---|---|---|
| `a3-jointsink-6k` | 1.4062 | 1.3591 | 1.3376 | 1.3242 | 1.3150 | 1.3071 |
| `a3-jointsink-refiner-6k` | 1.4062 | 1.3581 | 1.3359 | 1.3225 | 1.3126 | 1.3049 |

Score by t at 6000 for the a3 pair (t 0.16 … 0.84):

| Run | 0.16 | 0.26 | 0.34 | 0.40 | 0.47 | 0.53 | 0.60 | 0.66 | 0.74 | 0.84 |
|---|---|---|---|---|---|---|---|---|---|---|
| `a3-jointsink-6k` | 1.2820 | 1.3794 | 1.4340 | 1.4750 | 1.5102 | 1.5404 | 1.5630 | 1.5658 | 1.5241 | 1.3729 |
| `a3-jointsink-refiner-6k` | 1.2826 | 1.3806 | 1.4351 | 1.4763 | 1.5115 | 1.5416 | 1.5632 | 1.5665 | 1.5241 | 1.3704 |

## Caption use at 6000

Loss with the null prompt minus loss with the caption; caption mass: the softmax mass on the caption's keys per joint
block, a mean over heads and queries.

| Run | null − caption, mean over t | at t 0.84 | mass at t 0.16, blocks 3 / 7 / 11 / 15 / 19 / 23 | mass at t 0.84 |
|---|---|---|---|---|
| `a1-lfmnormread-6k` | +0.0031 | +0.0098 | | |
| `a2-joint-6k` | +0.0023 | +0.0062 | .056 .015 .066 .008 .006 .008 | .184 .129 .088 .031 .029 .012 |
| `a2-jointsink-6k` | +0.0030 | +0.0076 | .025 .037 .012 .011 .005 .007 | .154 .182 .078 .030 .017 .009 |
| `a2-jointsink-nopool-6k` | +0.0028 | +0.0067 | .106 .042 .015 .005 .011 .007 | .281 .224 .105 .059 .092 .026 |
| `a2-jointsink-captionread-6k` | +0.0040 | +0.0108 | .030 .045 .004 .012 .006 .008 | .123 .172 .071 .040 .032 .011 |
| `a3-jointsink-6k` | +0.0026 | +0.0064 | .096 .038 .028 .008 .008 .013 | .285 .209 .083 .083 .029 .041 |
| `a3-jointsink-refiner-6k` | +0.0039 | +0.0125 | .055 .106 .033 .011 .014 .014 | .154 .274 .097 .065 .022 .034 |

The lowest head's openness σ(lse − b) at t 0.84 is 0.90 or more in every sink block of every run. The refiner's gates
at 6000, RMS over channels: block 0 attention 0.19, MLP 0.46; block 1 attention 0.34, MLP 0.72.

## Read

One seed each; the one measured seed-pair spread at this length is 0.0009 (FACTS), so differences below about 0.001
are not resolved.

- The null key at zero against randn (`a3-jointsink-6k` against `a2-jointsink-nopool-6k`): 0.0008 on the score,
  0.0011 held-out, at the spread.
- The caption refiner against the base: held-out 0.0022 ahead at 6000, ahead at every eval from 2000 (0.0010 to
  0.0024); score 0.0006 behind, within the spread; behind by about 0.001 at t 0.26 to 0.53 and ahead by 0.0025 at
  t 0.84. Nulling the caption costs half as much again (+0.0039 against +0.0026), twice as much at t 0.84. The mass
  moves from block 3 to block 7. 6.2 % lower throughput.
- The refiner leads on the set with the longer captions and not on the other. That is consistent with *Hypothesis* J6 failing: EmbeddingGemma 2's final states gain from a
  refiner too. One seed, at 6000 steps, and loss is not prompt following.
