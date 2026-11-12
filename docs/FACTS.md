# Facts

Measured numbers only: statistics, throughput, memory, run results. Each entry gives the number, its source (script or run) and the date; a new read replaces the old entry rather than being appended. Estimates stay with the question they inform in `PLAN.md`.

## Sample and shard

Probe sample = first 256 images/captions of `data_512_2M/data_000000.tar` (not random), 128 px uses all 256, 256 px the first 128. Source: `scripts/probe_primal.py`, `scripts/probe_embeddinggemma.py`, 2026-10-31.

- Shard 0: 7,475,189,760 bytes, 50,000 samples (50,000 `.jpg` + 50,000 `.json`), images 512×512, caption in `{"prompt": "..."}`, keys `flux_512_100k_<8 digits>`. Other shards are 7.5 to 11.1 GB (repo listing).
- Shards 0–3, precomputed 2026-11-01 (`precompute.py` log, parquet keys): 0 and 1 are `flux_512_100k_*`, 50,000 pairs each; 2 and 3 are `dalle3_1m_*`, 49,985 and 49,982 pairs, keys numbered 0–49999 with gaps. Shard 2's tar (11.1 GB) was re-downloaded and listed: 49,985 complete jpg+json pairs, all in the parquet, so the gaps are in the source. Shard 3's tar was not re-checked.
- Shards 0–3 at 128 px, every 97th row (2026-11-01, read back from the parquets): latent std 1.06, embedding norm 1.000.
- Shards 0–46 at 128 px, all precomputed by 2026-11-02 04:18 (parquet metadata and keys): 2,290,793 rows. Sources by shard: 0–1 `flux_512_100k`, 2–19 `dalle3_1m`, 20–34 `llava_next_captioned_700K`, 35–46 `llava_generated_by_flux_500K`. Keys restart at 0 in every shard, so they repeat across shards of one source. Below 50,000 rows: shard 19 44,251, 34 38,524, 46 (the last) 8,128, and 2–18 by up to 18; 19 and 34 not checked against their tars.
- Repeats in shards 0–46 at 128 px (2026-11-02, scratch scripts hashing each row's latent bytes, with its caption, not kept): 1,925,416 distinct latents and 2,021,723 distinct (latent, caption) rows in 2,290,793. Shards 20–34 (`llava_next_captioned_700K`, 738,524 rows) hold 378,362 distinct latents and 472,515 distinct (latent, caption) rows; the other shards (1,552,269 rows) 1,547,055 distinct latents. Within that source an image comes in runs of consecutive keys, 1 to 7 long and most often 4 (shard 20), with the same or a slightly different caption (shard 20, keys 11137–11140: three identical captions, one with "a pitcher of water"), and recurs at later keys and in other shards of the source, up to 11 times in one shard. Distinct latents per 50,000 rows rise, with dips, from 19,653 in shard 20 to 49,331 in shard 33. `128_shuffled_0-10.parquet` has none of these shards.
- The same caption does not always give the same embedding bytes: in shard 20, 1033 captions that occur more than once have embeddings differing by at most 4.9e-4 per entry, cosine ≥ 0.9999999, bf16 rounding (2026-11-02).

## Primal 2x2

Source: `scripts/probe_primal.py`, `/mnt/mars/ai/tiny-dit/probes/primal.json`, 2026-10-31. fp32, CPU. Resize: torch bicubic, antialias=True, from the 512 crop. Latent [N,2,H/2,W/2]. 80.35M params.

| | 128 px (N=256) | 256 px (N=128) |
|---|---|---|
| per-image mean, ch0 (min/med/max) | -0.137 / 0.012 / 0.108 | -0.054 / 0.005 / 0.057 |
| per-image mean, ch1 | -0.151 / 0.033 / 0.215 | -0.143 / 0.013 / 0.139 |
| per-image std, ch0 | 0.927 / 1.051 / 1.200 | 0.914 / 1.031 / 1.120 |
| per-image std, ch1 | 0.801 / 1.022 / 1.141 | 0.911 / 0.999 / 1.087 |
| global mean (ch0, ch1) | 0.0053, 0.0215 | 0.0041, 0.0063 |
| global std (ch0, ch1) | 1.056, 1.021 | 1.031, 1.003 |
| excess kurtosis, pooled standardized (ch0, ch1) | 0.07, 0.27 | 0.05, 0.23 |
| fraction beyond 3σ (ch0, ch1; Gaussian 0.27%) | 0.34%, 0.44% | 0.33%, 0.43% |
| recon PSNR, clamped [0,1] (mean / min / max dB) | 34.85 / 25.4 / 55.0 | 38.48 / 30.3 / 51.1 |
| encode / decode, CPU, loaded machine | 20.5 / 20.1 img/s | 5.1 / 4.5 img/s |

Flip, vflip and 90° rotation (the latent gets the same spatial op; `M` is a channel map applied to it). Target is enc(op(img)). rel err is mean over images of ‖M·op(z) − enc(op(img))‖/‖enc(op(img))‖.

- Encoder is not equivariant. Bare spatial op, M = identity (best of the 8 signed maps for all three ops at both sizes): hflip rel err 0.994 / cosine 0.496 (128 px), 1.076 / 0.412 (256 px); vflip 0.995 / 0.494, 1.078 / 0.409; rot90 1.020 / 0.466, 1.131 / 0.344.
- Other signed maps are worse. 128 px hflip, rel err / cosine: +c0,-c1 1.516 / -0.151; -c0,+c1 1.301 / 0.151; -c0,-c1 1.728 / -0.496; swapped (±c1, ±c0) 1.24 to 1.56 with |cosine| ≤ 0.22. 256 px and the other ops show the same pattern (`signed_maps` in the JSON).
- Least-squares 2×2 map, hflip: [[0.305, 0.158], [0.060, 0.662]] at 128 px (rel err 0.840, cosine 0.536); [[0.256, 0.168], [0.105, 0.529]] at 256 px (0.889, 0.453). vflip is within 0.003 of hflip. rot90 128 px [[0.278, 0.192], [0.095, 0.618]] (0.853, 0.513); 256 px [[0.185, 0.247], [0.187, 0.434]] (0.904, 0.425). Shrunk, not a signed permutation.
- Decoder is equivariant: PSNR of decode(op(z)) against op(img), no channel map: hflip 34.62 / 38.22 dB, vflip 34.61 / 38.23, rot90 34.55 / 38.18, against plain reconstruction 34.85 / 38.48 (128 / 256 px).
- Reference in the JSON (`baseline_rel_err_z_vs_op_z_no_map`): ‖z − op(z)‖/‖op(z)‖ for the same image's own latent, no re-encode; 1.43 (hflip, 128 px), about √2 as expected for uncorrelated equal-norm fields.

Stability under small input changes (same samples; `stability` in the JSON). cosine / rel err of enc(img') against enc(img), 128 px | 256 px:

- Gaussian pixel noise, std 1/255 (0..255 scale): 0.966 / 0.219 | 0.939 / 0.316. Std 4/255: 0.872 / 0.475 | 0.803 / 0.608.
- JPEG q95 round trip of the 8-bit image: 0.964 / 0.256 | 0.950 / 0.308. Control, 8-bit rounding alone: 0.997 / 0.064 | 0.994 / 0.097.
- Circular 2 px shift against the latent rolled by 1: 0.417 / 1.071 | 0.329 / 1.149 (interior only, borders cropped 8: 0.459 / 1.029 | 0.353 / 1.128).
- Determinism: same batch twice and batch 16 against single images give max abs diff 0.0 (CPU, fp32).
- Frequency split (latent avg-pooled by k then nearest-upsampled = low band; remainder = high band; mean per-image cosine; k = 2 / 4 / 8). hflip case, enc(flip img) against flip(enc img), 128 px: low 0.988 / 0.991 / 0.990, high 0.266 / 0.427 / 0.475; 256 px: low 0.987 / 0.992 / 0.991, high 0.177 / 0.336 / 0.389. 1/255-noise case, 128 px: low 0.979 / 0.989 / 0.990, high 0.960 / 0.963 / 0.965; 256 px: low 0.962 / 0.983 / 0.988, high 0.930 / 0.934 / 0.937. The low band holds 33% / 12% / 4% of the energy (k = 2 / 4 / 8, 128 px).

## Downsampler (B5, 2026-11-04)

Source: `scripts/probe_resize.py`, CPU, fp32. First 256 square 512 px jpgs of `data_512_2M/data_000000.tar`, resized to 128 px. Reference: the source cropped in the Fourier domain to the 128 px Nyquist. Band energy is the share of the reference's energy in the band, at 0 to .125 / .125 to .25 / .25 to .375 / .375 to .5 cycles per pixel of the 128 px grid. Primal round trip on the first 64 images; "detail" is the image minus its 5×5 box blur, and the error is the detail error over the detail energy.

| resize | PSNR vs reference | band energy | Primal round trip PSNR | detail error / energy |
|---|---|---|---|---|
| torch bicubic, antialias (the precompute's) | 26.43 | 1.00 / 0.97 / 0.81 / 0.57 | 35.59 | 0.082 |
| PIL Lanczos | 26.25 | 1.00 / 1.00 / 0.96 / 0.70 | 34.74 | 0.090 |
| PIL box (area) | 26.16 | 1.00 / 0.92 / 0.79 / 0.72 | 34.67 | 0.099 |
| PIL bicubic | 26.41 | 1.00 / 0.96 / 0.81 / 0.57 | 35.57 | 0.082 |
| torch bilinear, antialias | 26.59 | 1.00 / 0.82 / 0.56 / 0.35 | 37.40 | 0.068 |
| torch bicubic, no antialias | 24.04 | 1.00 / 1.08 / 1.29 / 1.75 | 30.28 | 0.175 |

- The precompute's resize loses about 10% of the amplitude in the .25 to .375 band and about 25% in the top band against Lanczos, which keeps 98% and 84%. It is soft, not mushy; bilinear is mushy; without antialiasing the energy above the reference is aliasing.
- Round trip PSNR rises with smoothness, so it does not rank the resizers. Latent std is 1.03 to 1.04 for all.
- 512 px jpgs, 256 images: the sources are already compressed, and nothing here says what the model does with the extra detail.

## EmbeddingGemma 2 (B3)

Source: `scripts/probe_embeddinggemma.py`, `/mnt/mars/ai/tiny-dit/probes/embeddinggemma.json`, 2026-10-31. Text-only, 271.0M params, via sentence-transformers 6.2.0 (`config_kwargs=dict(vision_config=None, audio_config=None)`, `model_kwargs={"dtype": ...}`): Transformer, mean Pooling over all tokens (bos and eos included), Normalize. Raw norms come from the first two modules. Prefix `title: none | text: `. 256 captions + 16 short prompts.

- Output 768-d. Raw mean-pooled norm (min / median / max): 314 / 498 / 1577 (fp32), 314 / 498 / 1577 (bf16). Not unit; unit norm comes only from the Normalize module (modules.json entry 2). After normalize: 1.0000 ± 2e-7. After rounding to bf16: 0.9991 to 1.0009. No NaN/inf.
- After ×√768 (256 captions, fp32): RMS over all entries 1.000; per-dim std across captions 0.347 / 0.603 / 0.980 (min / median / max); per-dim |mean| across captions 0.0002 / 0.26 / 8.5. bf16 matches to 3 digits.
- bf16 against fp32 cosine: min 0.99995, median 0.99998.
- Token lengths with prefix, bos, eos: captions 17 / 66 / 136 (mean 67.1), short prompts 12 / 14 / 17. None over 8192 (model card context; `config.json` text_config `max_position_embeddings` is 262144).
- Cosine, fp32 unit embeddings: caption-caption, all pairs mean 0.581 (median 0.579, 0.418 to 0.850); each caption to its nearest other caption mean 0.719 (min 0.628, median 0.712). Short prompt to nearest of the 256 captions mean 0.743 (min 0.707, median 0.734, max 0.800); short prompt to all captions mean 0.613; short-short mean 0.695 (0.641 to 0.771). Nearest long captions are often loosely related (e.g. "a cat sleeping on a sofa" to a leopard caption, 0.720); pairs listed in the JSON.
- sentence-transformers loads with torchvision 0.29.2 installed (without it: `ImportError: EmbeddingGemma2VideoProcessor requires the Torchvision library`). Without `model_kwargs={"dtype": torch.float32}` it loads in bf16 (the checkpoint dtype), 5× slower on CPU. `prompt=PREFIX` and `prompt_name="Document"` agree (max abs diff 9e-8). `max_seq_length` is unset (1e30), so nothing is truncated.
- Cross-check, 256 captions, fp32, ST output against a hand-written transformers `AutoModel` + masked mean + normalize: max abs diff 1.2e-7, min cosine 0.9999997. The probe now uses the ST path only (`probe_embeddinggemma.py` rerun 2026-10-31 reproduces every number above).
- Null embedding (`/mnt/mars/ai/tiny-dit/precomputed/null_embedding.safetensors`, key `null`, fp32 [768], norm 1.0, from ST on `title: none | text: `, 9 tokens): cosine to the normalized mean caption embedding 0.866; to the 256 captions min 0.585 / median 0.661 / max 0.734 (mean 0.661); to "a red apple on a table" 0.772 and "a cat sleeping on a sofa" 0.735. For scale, caption-caption mean 0.581.

## CPU throughput

Source: the two probe scripts, 2026-10-31, 16-core CPU with other processes running. EmbeddingGemma 2 text-only via ST, batch 16, 272 texts: fp32 15.2 captions/s, bf16 3.2 captions/s. Primal encode: see the Primal table. Extrapolated to 2M images: Primal encode ≈ 27 h at 128 px, ≈ 108 h at 256 px; embeddings ≈ 36 h in fp32.

`scripts/precompute.py 128`, shards 0–3, 2026-10-31 to 11-01, 12 threads, machine shared: 8.95 / 9.51 / 12.40 / 12.74 rows/s per shard (the log's rate is cumulative), about 93 / 88 / 67 / 65 min, the later ones on a less loaded machine. Shard 0 (load ≈ 15): cumulative rate 8.11 at its low after 4k rows, 8.95 at the end (Primal encode and ST in series; 1/20.5 + 1/15.2 s per row would give 8.7); output 912 MB. Read back against the source tar: 50,000 rows, unique keys in tar order; 9 sampled rows match their caption, re-encode to the stored latent (cosine 1.00000), embedding cosine 0.999999 to a fresh encode, decode PSNR 30.0 to 39.4 dB; peak RSS 9.9 GB (a 1024-row chunk held as fp32 at 512 px).

## Parameter counts (2026-11-11)

`DiT` at its defaults (cross attention off), counted from the built model: d768 × 12 122.8M, d768 × 16 162.1M, d1024 × 16 287.7M.

## Numbers kept with their topic

- `topics/priming.md`, Reads: single-level denoising, masked priming, per-token t, REPA and its target, the vision tower's input size.
- `topics/edge-of-stability.md`: momentum cosine on a noisy quadratic; the learning-rate jump from `swpin_cont_3ep` (top Hessian eigenvalue).
- `topics/weight-spectra.md`: spectra of the pinned weights of `swpin_5ep`.
- `topics/datasets-and-encode.md`: Fine-T2I image sizes.

## Before the 2026-11-06 architecture

Reads of the earlier models (width 512 with a final linear, then the slice readout, before branch norm, the pin and the conv blocks). They do not describe the current model; the sections moved to the logbook whole:

- `log/2026-11-01-first-model-and-usui-wiring.md`: Model and Usui wiring (2026-10-31, `src/nanodit/model.py`, `scripts/probe_usui.py`); First frame fit on the GPU (2026-11-01); adaLN at init (2026-11-01, `scripts/probe_init.py`).
- `log/2026-11-01-smoke-and-fixed-step-runs.md`: GPU smoke runs at 128 px (2026-11-01); Fixed-step runs at 128 px (2026-11-01); `128_fixed_2e-2_9k` (2026-11-01); `128_fixed_2e-2_final_r64_9k` (2026-11-01); Runs at usui lr 2e-3 (2026-11-01); `128_fixed_2e-3_9k`, stopped at step 4300; `128_fixed_2e-3_b099_1k`: beta 0.99, 1000 steps; `128_2e-3_0-10_10ep`: 10 passes over shards 0–10 (2026-11-01).
- `log/2026-11-02-1000-step-ablations.md`: 1000-step ablations at 128 px (2026-11-01 23:56 to 2026-11-02 00:29).
- `log/2026-11-02-token-signal-and-branch-roles.md`: Token signal through the blocks (2026-11-02); What each branch does (2026-11-02).
- `log/2026-11-02-lr-sweep-and-seed-merge.md`: LR sweep and seed merge (2026-11-02).

## GPU precompute (2026-11-01)

- `scripts/precompute.py 128` on the RTX 4070 Super, both encoders fp32, TF32 convolutions on (the cuDNN default), shard 4: about 137 rows/s, the GPU at 99%, about 6 min per 50k-row shard plus its download (4 min 45 s for shard 4's 11.1 GB, about 39 MB/s). With the next download overlapping: 141 rows/s over shard 4's first 5k rows (2026-11-01).
- 32 latents of shard 0 decoded by Primal and encoded again: GPU with TF32 against CPU fp32 differ by 0.25% relative per latent (max 0.73%), against 0.17% for rounding the CPU latent to bf16; GPU with TF32 off matches CPU to 2e-6, at about 75 rows/s over shard 4's first 15k rows (2026-11-01).
- Shards 11–46 on the GPU, 2026-11-02 00:29–04:18: 1,740,848 rows in 3 h 48 min with the downloads, about 127 rows/s; the progress lines read 90–170 rows/s, median 132 (log `precomputed/precompute_128_11-46_gpu.log`).

## Shuffle and stream (2026-11-01)

- `scripts/shuffle.py` over shards 0–10 at 128 px: 549,945 rows, 9.98 GB out, 164 s, peak RSS 4.4 GB (log `precomputed/shuffle_128_0-10.log`).
- `scripts/shuffle.py` over shards 0–46 at 128 px, keeping each image once (`8bcffbb`), 2026-11-02 08:17 to 08:33: 1,925,416 rows of 2,290,793 read, 34.8 GB out (log `precomputed/shuffle_128_0-46.log`); peak memory not measured.
- `nanodit.data.Stream` on the CPU over the 0–10 file, micro-batch 64 cast to fp32, with a training run alongside: 48k rows/s over 2000 micro-batches after one, peak RSS 2.4 GB for the whole process (a scratch probe under `/usr/bin/time`, not kept).
- `128_2e-3_0-10_10ep`, the first run on the stream, trained at 746–846 img/s per 100-step record up to step 3700 and 847–862 from step 3800 to the end, against about 850 for `128_fixed_2e-3_b099_1k`. The slow records run past the first pass (step 2148), so a cold page cache does not explain them. The agent session's CPU test runs overlapped part of them (the suite at 22:25, about step 3400); what else ran is not recorded.
- Every run above it drew its batches with replacement from shards 0–3 (about 200k rows); a run on the stream sees each of 550k rows once per pass. Losses do not compare across that line without saying so.

## Sweep throughput and host syncs (2026-11-02)

`128_sweep_lr2e-3` and `lr4e-3` (tiny-dit `78fded7`, usuitrack `zero-init` worktree): 560–585 img/s at batch 128 as 2 × 64, so about 222 ms per step. d512/depth 12 trains at about 13.7 GFLOP per image (analytic, 3 × forward), so about 7.9 TFLOP/s; the RTX 4070 Super's bf16 tensor peak with fp32 accumulate is about 35.5 TFLOP/s (spec figure, not measured). `nvidia-smi` during `lr4e-3`: 99 W, utilization sampled at 40–84 %.

- `Stream.next` copies pageable CPU bf16 into the GPU fp32 buffers with a blocking `copy_`: the host waits until all queued GPU work has run. Probe, 2026-11-02, a 64-row micro-batch behind a ~71 ms spin kernel: the latent and the text copy each blocked the host for the whole queue (71.0 of 71.0 ms, 76.8 of 76.8 ms); a pinned source copied `non_blocking` and cast on the GPU blocked it for 0.0 of 73.0 ms. So each micro-batch drains the queue twice.
- Per-step cost against batch (median `imgs_per_s` of every run, same model, micro-batch 64): batch 64 about 354 img/s (`128_abl_batch64`), 128 about 580 (sweep arms, seed-merge runs, `lr2e-4` rerun), 256 about 850 (`128_2e-3_0-10_10ep`, ablations). A fit of step time = a + b·batch gives a ≈ 140 ms fixed per step and b ≈ 0.63 ms per image, so at batch 128 about 64 % of the step does not scale with the batch: the optimizer step and per-step launches, not forward/backward. The drop from ~850 to ~580 is the move to batch 128 (archive 2026-11-02, batch 128), not a slowdown.
- Not syncs per step: the `.any()` reads in usuitrack `molegrad.py` and `optimizer.py` run only while a matrix has no frame (step 1); `get_total_norm` stays on the device; `Interval` reads once per record.
- Profile, 2026-11-02 (scratch `profile/profile_step2.py`, `torch.profiler` over 10 steps at batch 128 as 2 × 64, fresh init, after 3 warm-up steps; the GPU shared with `128_4e-3_ep2`, so wall times are indicative only). Launches and host syncs per step:

| phase | launches, eager | launches, blocks compiled | host syncs |
|---|---|---|---|
| forward/backward (2 micro-batches) | ~5,400 | ~3,400 | 4, the `Stream` copies |
| Usui step | ~14,700 | ~14,700 | 9, one `eigh` per rank group |
| fallback AdamW | ~1,400 | ~1,400 | 0 |

  Wall per step 275 ms eager, 249 ms with the blocks compiled. The GPU is busy about 141 ms of a ~220 ms step, so the step is launch-bound, and the Usui step's per-matrix loop over 77 matrices (about 190 launches each, of which its diagnostics about 2,500 per step) is most of the launches. `torch.linalg.eigh` reads back to the host on CUDA (its info check), which also blocks capturing the step as one CUDA graph.
- Block compile against fp32 (scratch `profile/equiv.py`, one batch): compiled bf16 gradients are closer to fp32 than eager bf16 (median relative error 4.0 % against 6.4 %, max 21 % against 33 %; Molegrad `z` 4.7 % against 8.3 %).
- Compiled while a Molegrad frame waits for its fit (until step 3 of a fresh run: the zero-gated blocks' matrices see a zero gradient before), each block breaks its graph at Molegrad's zero-gradient branch and hits dynamo's recompile limit (8), and forward/backward stays at ~3,600 launches. `scripts/train.py` compiles once no frame waits; a 20-step fresh run then had no graph break and one recompile (grad mode, for eval and score).
- Block compile on a free GPU: `128_4e-3_ep3` (batch 256, compiled from its first step, tiny-dit `80c3cb3`) ran at a median 1057 img/s, against about 850 for the eager batch-256 runs: +24 %.
- Width and depth arms (the depth arm was dropped, PLAN G1), 40-step microbench (2026-11-02, batch 256 as micro-batch 64, compiled, random frames, a GPU shared with nothing): d768 × 12 runs at 765 img/s (741, 762, 772 over the last three logs), d512 × 24 at 510 (507, 513, 507). Epochs of 7521 steps take about 42 and 63 minutes. Neither ran out of memory at micro-batch 64; peak memory is not logged.

## Timestep distribution arms at d768x12 (2026-11-04)

Source: `/mnt/mars/ai/tiny-dit/runs/128_d768x12_{t1,tsig07,tshift15}_3k/score.jsonl` and `train.jsonl`. Three runs, same seed (same init, data order and eval noise), 3000 steps, batch 256, usui lr 2e-3 fixed rule, the sandwich-norm model. Only the training t differs: `t1` logit-normal σ 1, shift 1; `tsig07` σ 0.7; `tshift15` shift 1.5 (median t 0.6). All scored on the same shift-1 σ-1 grid of 10 t (0.16, 0.26, 0.34, 0.40, 0.47, 0.53, 0.60, 0.66, 0.74, 0.84), every 500 steps.

| arm | caption loss at 500 / 1500 / 3000 | R² at 500 / 1500 / 3000 |
|---|---|---|
| `t1` | 1.6167 / 1.5179 / 1.4886 | 0.5939 / 0.6154 / 0.6213 |
| `tsig07` | 1.6105 / 1.5130 / 1.4847 | 0.5950 / 0.6166 / 0.6220 |
| `tshift15` | 1.6304 / 1.5283 / 1.4956 | 0.5921 / 0.6139 / 0.6206 |

- `tsig07` has a lower loss and higher R² than `t1` at all six scored steps (mean over the grid). Per t, over the six steps, it wins at all six at t 0.26 to 0.66, at three of six at t 0.74, and at none at t 0.16 and 0.84. At step 3000 its R² against `t1` per t: 0.0 / 0.0 / +0.001 / +0.001 / +0.002 / +0.002 / +0.002 / +0.001 / 0.0 / -0.002.
- `tshift15` is worse than `t1` at the grid mean at all six steps. Per t it wins at none of the six steps for t up to 0.60, and at 2 / 3 / 4 of six at t 0.66 / 0.74 / 0.84.
- Train-side R² by 0.1 bucket, mean of the last 1000 steps, against `t1` (buckets 0.0 to 0.9): `tsig07` -0.0016 / -0.0036 / -0.0025 / -0.0007 / +0.0016 / +0.0046 / +0.0080 / +0.0097 / +0.0059 / +0.0005; `tshift15` -0.0004 / -0.0014 / -0.0022 / -0.0034 / -0.0045 / -0.0048 / -0.0022 / -0.0012 / -0.0006 / +0.0027.
- Score R² per t of `t1` at step 3000: 0.971 / 0.916 / 0.855 / 0.785 / 0.706 / 0.616 / 0.514 / 0.400 / 0.281 / 0.170 (the null condition's mean R² differs by under 0.001 from the caption's).
- One run per arm: the six steps share a trajectory and are not six confirmations. No seed-to-seed spread was measured at this pairing. Sample grids at step 3000 were not compared beyond a look at `t1` and `tsig07`: both are blurred compositions.


## Spikes in `128_d1024x16_5ep` (2026-11-04)

Source: `/mnt/mars/ai/tiny-dit/runs/128_d1024x16_5ep/` (d1024x16, slice readout, no branch norm, usui lr 1e-2 constant, fixed step), a replay from its step-15000 checkpoint with a trap on any micro-batch loss above 4, and scratch probes on its checkpoints.

- Single-batch loss spikes, from 10 to 2.6e6, begin at step 4850 and recur about once per 260 steps, more often and larger as the run goes on (79 spike records after warmup); the fallback grad norm reaches 8.6e3 to 5e6. Each recovers. `tangent_live_fraction` falls from 0.94 at step 2500 to 0.64 at 37.5k (no earlier run went below 0.7). No d768 or d1024x12 run spiked.
- Scores: 1.4617 (3750), 1.4445 (7500), 1.4385 (11250), 1.4295 (15000), 1.4266 (18750), 1.4233 (22500), 1.4266 (26250), 1.4209 (30000, the best), 1.4268 (33750), 1.8257 (37500, saved straight after a spike), 1.4914 (37605). Flat from about step 15000.
- Residual stream rms at step 33750: 30–110, single entries 3300–4400 (about 1 at init; the d768 slice arm at 7.5k: rms 5–12, max ~490). 41k training rows through the step-33750 checkpoint: worst per-sample loss 2.25. On the eval rows the largest SwiGLU hidden value stays 25–67 at checkpoints 3750 to 33750, with no creep.
- The trapped spike, step 16222: sample 60 (t 0.81), loss 15378 against a batch median of 1.2. In blocks 10–13 its SwiGLU hidden reached ~1e4 and the MLP output ~3e4; the stream rms rose from 20 to 200. Attention and `ada` outputs stayed normal. Scaling `w_up` by 0.9 / 0.8 on that snapshot cuts the sample's loss to 7912 / 3653: smooth amplification, no threshold. σ₁²/‖W‖²_F is 0.010–0.022 in every MLP matrix: no rank-1 outlier direction.
- lr A/B, replay of steps 15000–16500 from the same checkpoint: lr 1e-2 about 6 spikes, lr 5e-3 one (step 16254, t 0.16, loss 15596). The lr sets how often, not how hard.
- Weight norm against init, no weight decay: `ada` 5× at 3750 and 15.5× at 37500; `qkv` 3.3× / 10.8×; `w_gate`, `w_up`, `w_down` 3.3–3.5× / 9.4–10.9×; `time_mlp` 2.6× / 10×; `patch_embed` 1.1× / 2.2×; the `table` gate rows 14×. Growth decelerates; the update over the last 3750 steps is ~28 % of the norm, cosine ~0.1 to W. The relative step falls by growth alone (3.4e-3 at step 500, 1.3e-3 at 3750, about 4.7e-4 at 30000).
- The function is tied to the grown norms. Checkpoint 30000, 256 eval rows (1.4287 as is): every Usui matrix projected to its init norm 3.12, to its step-3750 norm 2.92; one family ×0.7: `ada` 1.51, gate 1.58, `out` or `qkv` 1.78, `w_gate` 1.55, `w_up` or `w_down` 1.50, `patch_embed` 1.92.
- Projection test, 1500 steps from checkpoint 30000 with the projection after every step (Usui fixed step at 1e-2): no pin 1.4286 → 1.4272 (one spike, loss 374); pinned to the step-3750 norms 2.92 → 1.4374, no spike; pinned to the init norms 3.12 → 1.4709, no spike. The relative steps were about 4.7e-4, 1e-3 and 3e-3: the plateau follows the relative step.

## Branch norm and pin arms at d1024x16, 3000 steps (2026-11-04)

Source: `runs/128_d1024x16_{base,sw,sw_b,swpin,idfinal}_3k/score.jsonl`. Same seed (41), data order and eval rows; t logit-normal σ 1; caption loss on the score grid.

| arm | change from `base` | 1000 | 2000 | 3000 |
|---|---|---|---|---|
| `base_3k` | the `5ep` model and step (usui 1e-2, fixed step) | 1.5266 | 1.4861 | 1.4686 |
| `sw_3k_b` | sandwich norm | 1.5309 | 1.4839 | 1.4680 |
| `swpin_3k` | sandwich norm, pinned to init norm, relative step 2e-3 | 1.5329 | 1.4796 | 1.4624 |
| `idfinal_3k` | sandwich norm, learnable `Linear(1024, 128)` readout on the fallback, init [I \| 0], in place of the slice | 1.5306 | 1.4869 | 1.4710 |

- `sw_3k` (scored at 3000 only) ended at 1.4680, the same as its repeat `sw_3k_b`: a same-seed rerun reproduces the score.
- `swpin_3k` trails at 1000 and leads from 2000. The learned readout ends 0.003 behind the slice. One run per arm; 3000 steps is before the `5ep` run's first spike, so these arms measure cost, not stability.

## Architecture arms at d1024x16, 3000 steps (2026-11-05)

Source: `runs/128_d1024x16_{p16meta,conv}_3k/{score,train}.jsonl`, against `swpin_3k` above (same config, seed 41, data order and eval rows); throughput is the median `imgs_per_s` after the first 250 steps.

| arm | change from `swpin_3k` | 1000 | 2000 | 3000 | img/s |
|---|---|---|---|---|---|
| `swpin_3k` | | 1.5329 | 1.4796 | 1.4624 | 376 |
| `p16meta_3k` | patch 16 (512-d tokens, 16 tokens at 128 px) carried in the first 512 channels; patch embed `Linear(512, 512)` into the meta channels only; slice readout of 512 | 1.6407 | 1.5732 | 1.5428 | 767 |
| `conv_3k` | blocks 0, 2, ..., 14: depthwise 3×3 conv over all channels in place of attention, `out` (1024 → 896) into the meta channels only | 1.5261 | 1.4804 | 1.4623 | 451 |

- `p16meta_3k` is 0.080 behind per step at 3000 and 2.04× as fast. Per wall-clock its 3000 steps buy what `swpin_3k` has at about step 1470, linearly interpolated 1.508: 0.035 behind at equal time.
- `conv_3k` is ahead of `swpin_3k` at 1000 (−0.0068) and level at 2000 (+0.0008) and 3000 (−0.0001) and is 1.20× as fast. Per wall-clock its 3000 steps match `swpin_3k`'s step 2501, interpolated 1.471: 0.009 ahead at equal time. One seed.

## `128_d1024x16_swpin_5ep` (2026-11-05)

Source: `runs/128_d1024x16_swpin_5ep/` (its `config.toml`): `swpin_3k` for 5 epochs, 37,605 steps of 256; usui relative step 2e-3 and fallback lr 2.5e-3, warmup 500, both decaying linearly to a tenth; t σ 1. Checkpoints at 7500, 15000, 22500, 30000, 37500, 37605.

| step | 3000 | 6000 | 9000 | 12000 | 15000 | 18000 | 21000 | 24000 | 27000 | 30000 | 33000 | 36000 | 37605 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| caption loss | 1.4626 | 1.4400 | 1.4298 | 1.4242 | 1.4180 | 1.4126 | 1.4082 | 1.4043 | 1.3990 | 1.3958 | 1.3919 | 1.3873 | 1.3856 |
| R² | 0.6262 | 0.6305 | 0.6325 | 0.6334 | 0.6346 | 0.6358 | 0.6366 | 0.6373 | 0.6385 | 0.6390 | 0.6398 | 0.6407 | 0.6411 |

- Against the unpinned `5ep` (section Spikes): level at 3000–3750, ahead from 7500 (0.011 at 15000, 0.025 at 30000); the end is 0.035 below the unpinned run's best.
- No spike record after warmup (none more than 0.08 above the rolling median of 50-step means); fallback grad norm at most 0.023 after warmup.
- The gain per 3000 steps grows as the lr falls: −0.0032 (27k→30k), −0.0039 (30k→33k), −0.0046 (33k→36k). Train loss by eighth of the run: 1.328, 1.217, 1.200, 1.188, 1.178, 1.168, 1.159, 1.150.
- R² per score-grid t (0.16 … 0.84) at 6000: 0.971 / 0.920 / 0.862 / 0.796 / 0.720 / 0.631 / 0.529 / 0.413 / 0.289 / 0.173; at the end: 0.972 / 0.924 / 0.870 / 0.807 / 0.734 / 0.648 / 0.547 / 0.430 / 0.301 / 0.179. The late gain is at mid t (about +0.017 at t 0.47–0.66), +0.001 at t 0.16, +0.006 at t 0.84.
- The null-embedding score at the end is 1.3891, 0.0035 above the caption's (the last run to score it).
- Samples, final grid against the unpinned run's at 30k, by eye: the unpinned grid is mush; the final grid has coherent figures, lighting and backgrounds, with class-level prompt following (the red-haired woman, the man at the beach, a cabin with a bear-like shape, text-like marks on the sign). Faces melt; no shark, no chess.
- Which of the sandwich norm and the pin removed the spikes is not separated: neither ran this long alone.

## Continuation lr arms from `swpin_5ep` (C4, 2026-11-05)

Source: `runs/128_d1024x16_swpin_cont_lr{2e-5,1e-4,2e-4,4e-4}/score.jsonl`. 2000 steps each from `swpin_5ep/checkpoints/step_0037605` (score 1.3856), flat usui lr as named, fallback 1.25× it, t σ 0.7, no warmup, seed 42 shared.

| usui lr | 38000 | 38500 | 39000 | 39500 | 39605 | R² at 39605 |
|---|---|---|---|---|---|---|
| 2e-5 | 1.3840 | 1.3834 | 1.3832 | 1.3829 | 1.3831 | 0.6416 |
| 1e-4 | 1.3838 | 1.3838 | 1.3832 | 1.3832 | 1.3831 | 0.6416 |
| 2e-4 (where the run ended) | 1.3851 | 1.3851 | 1.3844 | 1.3845 | 1.3843 | 0.6414 |
| 4e-4 | 1.3884 | 1.3883 | 1.3882 | 1.3884 | 1.3884 | 0.6406 |

- 2e-5 and 1e-4 end level; 2e-4 is 0.0012 behind and still falling; 4e-4 jumps 0.003 above the start at once and stays flat. Per t, 1e-4 and 2e-5 differ by at most 0.0005 in R². Train loss by quarter (σ 0.7, not comparable with the 5-epoch run's): 1e-4 1.1403 / 1.1386 / 1.1367 / 1.1381, 2e-5 within 0.0002 of it.
- Score slope per 1000 steps (`score.jsonl`, read 2026-11-05): `swpin_5ep` from 33000 to 37000, usui lr decaying from about 4e-4 to 2e-4, −1.4e-3; `128_d1024x16_swpin_cont_3ep` (1e-4 flat from the 1e-4 arm's 39605, seed 43) from 40000 to 49000, −2.3e-4, from 50000 to 62168 −1.5e-4, ending at 1.3789; the 4e-4 arm from 38000 to 39605, 0.
- Slope arms from `swpin_cont_3ep`'s 62168, 5000 steps flat, seed 44 (`runs/128_d1024x16_swpin_slope_lr{4e-4,1e-3}/score.jsonl`, read 2026-11-05): 4e-4 jumps to 1.3850 at 62500 and ends at 1.3859 (+0.0070 on 1.3789), slope +0.3e-4 per 1000; 1e-3 jumps to 1.3977 and ends at 1.4009 (+0.0220), slope +2.0e-4 (linear fits over the scores from 63000; from 64000, −0.4e-4 and +0.1e-4). 1e-4 is the only lr of the three still falling.

Reference spectra, the same measures on cached pretrained models (scratch `ref_spectra.py` on the GPU, `/mnt/mars/ai/tiny-dit/probes/spectra_reference.json`; mean over layers; effective rank and r90 as fractions of the shorter side). Their optimizers and token counts are not known here; both trained far longer than this run's ~0.6B tokens.

| model, matrix | eff rank | r90 |
|---|---|---|
| LFM3-1.7B-Base, FFN w1 / w3 / w2 (7680 × 2048) | 0.746 / 0.827 / 0.802 | 0.699 / 0.724 / 0.721 |
| LFM3, attention q / k / v / o | 0.373 / 0.579 / 0.797 / 0.459 | 0.361 / 0.528 / 0.687 / 0.420 |
| LFM3, conv in / out proj | 0.705 / 0.451 | 0.666 / 0.459 |
| EmbeddingGemma 2 text, MLP gate / up / down (2048 × 512) | 0.766 / 0.830 / 0.783 | 0.711 / 0.728 / 0.726 |
| EmbeddingGemma 2 text, attention q / k / v / o | 0.630 / 0.473 / 0.543 / 0.576 | 0.614 / 0.462 / 0.532 / 0.607 |
| EmbeddingGemma 2 vision tower, MLP gate / up / down | 0.626 / 0.683 / 0.513 | 0.620 / 0.630 / 0.608 |
| EmbeddingGemma 2 vision tower, attention q / k / v / o | 0.340 / 0.323 / 0.364 / 0.312 | 0.367 / 0.363 / 0.335 / 0.338 |
| ours, `swpin_5ep` at 37605: `w_gate` / `w_up` / `w_down` | 0.195 / 0.199 / 0.186 | block 8: 0.17 / 0.17 / 0.16 |
| ours: `qkv` / `gate` / `out` | 0.170 / 0.082 / 0.093 | block 8: 0.14 / 0.07 / 0.08 |
| ours, `swpin_3k` at 3000 (constant relative step 2e-3): MLP / `qkv` / `gate`, `out` | 0.33 / 0.32 / 0.18–0.19 | 0.26–0.29 / 0.32 / 0.20 |

- Our MLP matrices hold 90 % of their energy in about a sixth of the directions; the pretrained MLPs need about 0.7. Attention is lower in every model, ours lowest.
- Usui's `grad_capture` (‖G Q‖/‖G‖, the gradient share its rank-0.1 frame holds) reads 0.98–1.03 over the whole `swpin_5ep` run (`train.jsonl`; above 1 under Molegrad, where the full gradient is not formed). It is a probed estimate: comparable across runs and steps, not the true capture (2026-11-05).

## Architecture at d768x12 (F3, F4, G2, G10, 2026-11-06)

Caption loss on the Lunara rows, same rows and noise per arm; full tables in `log/2026-11-06-cross-attention-and-conv-period-3k.md`, `log/2026-11-06-block-design-arms-3k.md`, `log/2026-11-06-step-speed-by-layout.md`.

- Conv replacing attention in 1/2, 2/3 and 3/4 of the blocks ends level with the all-attention baseline at 3000 steps (1.4855 against 1.4852, 1.4859, 1.4884) and trains 1.20×, 1.28×, 1.32× faster; it leads by 0.007 at step 1000.
- Trained at 128 px and scored on the unseen 256 px grid, conv in 3/4 is 0.031 better than all attention (1.4045 against 1.4353), though 0.003 worse at 128 px.
- Cross attention in every block, pooled embedding in the adaLN, ends 0.017 behind pooled-only at 3000 steps and runs 0.72× as fast. 3000 steps cannot judge token conditioning.
- On the conv-in-3/4 base at 3000 steps, within 0.003 at 128 px: meta-only patch embed at 8 px token −0.0015, NoPE +0.0011, quarter RoPE −0.0001, 3 KV heads +0.0004, PaLM parallel MLP +0.0024. At 256 px NoPE is +0.0034, the others within 0.0025 of the base; quarter RoPE and meta-only embed are the best.
- Step speed (forward + backward, d768x12, bf16, compiled): per-token time 14.6 µs + 1.78 ns·N for all attention, 11.5 µs + 0.41 ns·N for conv in 3/4; conv is 1.27× at 128 px, 1.66× at 1024 px, 2.4× projected at 2048 px. GQA gives a flat 2–3 % (conv base), 5–9 % (all attention).
- Trained at 128 px, 6000 steps, on the trunk data: cross attention in the attention blocks (with or without the pooled adaLN) leaves the caption loss level (1.4780, 1.4771 against 1.4788) and binds more of the prompt in samples; conv in 3/4 beats all attention by 0.010 (1.4788 against 1.4884). `log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`.
- Local GDN2-style gated update in the conv blocks (diagonal, reading through the 3x3): 0.0014 better at 128 px and 0.004 at 256 px, writing the image channels adds 0.0008, 84 % of conv speed. A content-dependent 9-neighbour read is 0.005 better at 128 px but 0.044 worse at 256 px and 0.62x as fast. A global linear attention in the conv slot (1.4849) loses to plain conv. `log/2026-11-07-trunk-arms-6k-and-lr-sweep.md`.
- d768x16, 3000 steps, usui lr: 5e-4 1.5332, 1e-3 1.5101, 2e-3 1.4933, 4e-3 1.4868, 8e-3 1.4950; the best is near 4e-3 and 1e-3 is far below it.
- Caption prefix for image retrieval, 1000 decoded eval images (256 px latents) at the 384 px budget, each caption ranking its image among the 1000 (top-1 / top-5 / cosine own − others): no prefix 94.4% / 99.7% / 0.183, retrieval query 93.2% / 99.7% / 0.189, document 90.5% / 99.3% / 0.150, sentence similarity 90.2% / 98.4% / 0.126, classification 83.1% / 97.1% / 0.116, clustering 77.2% / 95.5% / 0.102. No prefix retrieves best; the retrieval query is level within noise, and every other prefix pulls captions together (the others' cosine rises more than the own). CPU fp32, `scripts/probe_prefix.py`, 2026-11-07. The stored embeddings carry the document prefix, not none: 8 Lunara eval captions re-encoded have cosine 0.999999 to the stored embedding with `title: none | text: ` and 0.83 to 0.86 without (2026-11-11). So the stored conditioning is the 90.5% row, 3.9 points below no prefix (PLAN B7).
