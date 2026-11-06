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

## Primal 2x2 (B1)

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

## EmbeddingGemma 2 (B2, B3)

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

## Model and Usui wiring (2026-10-31, `src/tinydit/model.py`, `scripts/probe_usui.py`)

- Parameters at depth 12: 37,719,936 (2026-11-01, with the final layer's rows in the shared projection). Blocks 34.70M (qkv 9.44M, SwiGLU 18.87M, sigmoid gate 3.15M, out-proj 3.15M, conv 55k, tables 37k); shared adaLN projection 2.10M; time MLP 394k; text projection 394k; patch embed 66k; final linear 66k. 2-D 37.66M, 1-D 9,088, 4-D (conv) 55,296.
- Usui on `lab`, 5 steps at lr 1e-2: a zero-init 128×512 matrix stays at norm 0.0; a (512,1,3,3) conv weight and a (64,) vector are rejected; a shared-projection-like matrix with zero gate rows and std-0.02 shift/scale rows grows its gate rows 0 → 0.074 (matrix norm 2.56); a [6, 512] table is accepted (frame [6, 3]). In the zero-init case the fan-in matrix feeding the zero one does not move either: its gradient is zero while the other is.
- Molegrad and accumulation, wiring check only (synthetic latents, depth 2, CPU, batch 32, usui lr 3e-3, fallback lr 1e-3, 60 steps, identical batches per arm; scratch script, not in the repo, so not reproducible from it): Molegrad 1×32 and 4×8 agree to 4 digits at every step (loss 2.5807 → 2.0579). Step 1 (final linear on the fallback): every tracked gradient exactly zero, nothing tracked moves. By step 10 `tangent_live_fraction` 0.91, `relative_step` ≈ 98% of lr. After 60 steps `blocks.0.qkv` shares 0.38 of its step-1 frame (G path without Molegrad: 0.50, loss 2.0544).

## First frame fit on the GPU (2026-11-01)

- `test_accumulated_batch_trains_logs_and_checkpoints` (depth 2, micro-batch 2), a probe patched around `usuitrack.frame.fit`: `ada.1.weight` [3072, 512], bf16 gradient of matrix rank 5, its normalized side Gram with top eigenvalue 0.99997 and 510 of 512 below 1e-6, rank 125 asked. `torch.linalg.eigh` in fp32 on CUDA raises "failed to converge"; fp32 on CPU and fp64 on CUDA return.
- The test failed on the GPU from `a984e20` until usuitrack `e7e5210`, and passed before it (bisected by worktree); the whole suite passes on the CPU. Why that commit tips it is not isolated.
- The fit runs once per matrix, at its first nonzero gradient, on one micro-batch's gradient (`usuitrack/molegrad.py`, `usuitrack/optimizer.py`). The 128 px runs (micro-batch 64) did not fail; at most 64 samples feed `ada`'s gradient there, so its rank is at most 64 of 512.
- Since usuitrack `e7e5210` the fit is a random range finder with no `eigh` (archive 2026-11-01, first frame fit); the suite passes on the GPU (50 tests, 2026-11-01). Its capture against the exact fit, on this gradient and on the 78 first fits of the depth-12 model, is in usuitrack `docs/REVIEW_zero-init.md`.

## adaLN at init (2026-11-01, `scripts/probe_init.py`)

Depth 12, seed 0, 16 timesteps spread over (0, 1), text = the null embedding or random unit vectors. `c` per-element std 1.05 / 1.09. Block 0 shift/scale rows (shared projection + table): std 0.29 / 0.30. Final layer shift/scale (shared projection + final table): std 0.31 / 0.30, max |scale| 1.25 / 1.13. Before the final layer's rows moved into the shared projection, it took `final_table + c`: std 1.07, max |scale| 4.0 to 4.1 (2026-10-31).

## GPU smoke runs at 128 px (2026-11-01)

RTX 4070 Super, `configs/128_smoke.toml` as it was then (300 steps, batch 256 as 4 × 64, usui lr 3e-3, beta 0.95, fallback lr 1e-3, shift 1.0, shards 0–3), Usui `lab` at `be3f298`, option (a) split. The fp32 runs are tiny-dit `61448e4`; the bf16 run is the commit that adds this section. Runs and JSONL logs in `/mnt/mars/ai/tiny-dit/runs/<name>/`; the summary script is not in the repo. Mean loss per step window, median img/s from step 11:

| run | precision | seed | 1–50 | 51–100 | 101–200 | 201–300 | img/s |
|---|---|---|---|---|---|---|---|
| `128_smoke` | fp32 + TF32 | 0 | 2.0918 | 1.7427 | 1.5986 | 1.5422 | 756 |
| `128_smoke_fp32_seed1` | fp32 + TF32 | 1 | 2.0888 | 1.7419 | 1.6002 | 1.5429 | 745 |
| `128_smoke_bf16` | bf16 model, stochastic rounding | 0 | 2.0981 | 1.7439 | 1.5949 | 1.5389 | 862 |
| `128_smoke_zero_init` | bf16; final linear on Usui with a norm floor √128, zero-gradient rule | 0 | 2.4039 | 2.1247 | 1.9701 | 1.8528 | 843 |
| `128_smoke_nofit_final_adam` | bf16; zero-gradient rule, final linear on the fallback | 0 | 2.0715 | 1.7203 | 1.5857 | 1.5340 | 849 |

The last two ran on usuitrack `zero-init` at `a32658b` from uncommitted tiny-dit code (the floor build of A1; the second with a scratch launcher putting the final linear back on the fallback); neither split is in the tree now.

- Run-to-run spread (the two fp32 seeds): at most 0.003 per window, 0.0007 over steps 201–300.
- bf16 against fp32 at seed 0: +0.006 over steps 1–50, −0.0033 over 201–300. Over the last 100 steps, per t-bucket, bf16 is within 0.012 of fp32 in every bucket but the lowest (t < 0.1: 1.283 against 1.307 and 1.313). One bf16 seed; a 100-step window mean has a standard error near 0.002, so the late difference is not resolved.
- Loss by t bucket, steps 201–300 (fp32 seed 0): 1.31 at t < 0.1, peaking at 1.65 around t = 0.5, 1.16 at t > 0.9.
- Usui frame after the step-1 zero gradient, `grad_capture` (mean over matrices, then over steps; fp32 seed 0, seed 1, bf16): 0 at step 1, 0.017 at step 2, 0.33 / 0.32 / 0.33 over steps 3–11, 0.58 / 0.57 / 0.59 over 21–30, 0.80 / 0.78 / 0.80 over 51–60, 0.96 / 0.96 / 0.95 over 101–300. Under Molegrad the raw gradient norm is a probed estimate, so single steps exceed 1 (37–51 of 300 per run, max 1.16).
- `grad_moment_cosine` (same windows): +0.44 to +0.51 over steps 3–11, −0.07 to −0.10 over 91–100, −0.23 to −0.26 over 101–150, −0.36 to −0.39 over 151–300, in all three runs. Usui reads negative as overshoot, to be read against lr; only 3e-3 has run.
- Final linear Frobenius norm at steps 100 / 200 / 300: 3.88 / 3.80 / 3.65 on the fallback (`128_smoke_bf16`), 3.56 / 3.54 / 3.50 (`128_smoke_nofit_final_adam`), 1.85 / 2.09 / 1.74 under the floor (`128_smoke_zero_init`), which it never reached.
- With the zero-gradient rule, `grad_capture` over steps 3–11 is 0.92 (final linear on the fallback) and 1.02 (on Usui), against 0.33 without; over steps 21–60 the first dips to 0.79–0.82, and from step 101 it reads 0.94–0.97 in all runs.
- Loss by t bucket under the floor, steps 201–300: 2.70 at t < 0.1 against 1.28 for `128_smoke_bf16`, the gap shrinking to 0.01 above t = 0.8.
- The eval grid at step 300 follows each prompt's colour and mood, no shapes yet; faint vertical streaks.
- Step time with random data, model only (scratch script, 40 steps): fp32 + TF32 792 img/s, 2.58 GiB peak; bf16 autocast (Molegrad fix uncommitted in the usuitrack worktree `molegrad-autocast`, not used) 879 img/s, 2.17 GiB.

## Fixed-step runs at 128 px (2026-11-01)

Uncommitted tiny-dit code, by its diff the same as `b3adad8` in `src/` and `scripts/` (every matrix on Usui's fixed step at lr 2e-2, final linear included, frame rank fraction 0.05; linear warmup), usuitrack `zero-init` at `e746901`, otherwise `configs/128_smoke.toml` as it was then (300 steps, batch 256, fallback lr 1e-3, bf16, seed 0). The `final_r64` runs used a scratch launcher putting the final linear in its own Usui group at rank fraction 1.0 (r = 64, the cap at half the shorter side), at that group's full fixed step and at its step scaled by √(13/64) to the r = 13 step norm (`eq`). Same windows as above:

| run | warmup | final linear | 1–50 | 51–100 | 101–200 | 201–300 | img/s |
|---|---|---|---|---|---|---|---|
| `128_fixed_2e-2_w20` | 20 | r = 13 | 2.4571 | 2.1345 | 1.9808 | 1.8644 | 778 |
| `128_fixed_2e-2_w50` | 50 | r = 13 | 2.5553 | 2.1587 | 1.9963 | 1.8747 | 808 |
| `128_fixed_2e-2_w100` | 100 | r = 13 | 2.6490 | 2.2037 | 2.0176 | 1.8919 | 749 |
| `128_fixed_2e-2_final_r64` | 50 | r = 64 | 2.4175 | 1.8252 | 1.7158 | 1.6454 | 841 |
| `128_fixed_2e-2_final_r64eq` | 50 | r = 64, r = 13 step norm | 2.5432 | 1.8422 | 1.7095 | 1.6417 | 799 |

- Loss by t bucket, steps 201–300, lowest to highest t: w50 2.77 / 2.54 / 2.30 / 2.10 / 1.94 / 1.76 / 1.63 / 1.48 / 1.32 / 1.17; r64 1.85 / 1.80 / 1.77 / 1.75 / 1.73 / 1.66 / 1.60 / 1.47 / 1.32 / 1.17; `128_smoke_nofit_final_adam` (final linear on the fallback; also the earlier relative step at 3e-3 and no warmup, so not a same-code control) 1.27 / 1.37 / 1.50 / 1.59 / 1.64 / 1.64 / 1.58 / 1.46 / 1.31 / 1.15. The gap shrinks with t: 1.50 in the lowest bucket, 0.06 at 0.6–0.7, under 0.02 above.
- Warmup length at 300 steps: 1.8644 / 1.8747 / 1.8919 over steps 201–300 (w20 / w50 / w100), spread 0.028, shorter lower.
- `grad_capture` over steps 101–300: 0.94 / 0.92 / 0.89 (w20 / w50 / w100), 0.94 on the fallback run. `grad_moment_cosine` over 251–300: −0.34 / −0.34 / −0.33, −0.37 on the fallback run.
- The final linear at step 300 (`[128, 512]`; stable rank Σs²/s₁², and singular values above 10% of s₁): w20 stable rank 12.9, 41 of 128; r64 26.9, 93; fallback 45.9, 128; `128_smoke_bf16` 41.7, 128.
- How the r = 13 frame on the final linear moves (`scripts/probe_frame.py`; w50 at 100 / 200 / 300): stable rank 9.5 / 9.7 / 11.3; directions above 10% 20 / 32 / 38; share of ‖W‖² outside the current frame 0.17 / 0.27 / 0.37; ‖F_prevᵀ F‖²/r against the checkpoint 100 steps earlier 0.80 / 0.89. Under r = 64 (the same reads, the final linear indexed in its own last group) that overlap is 0.98 / 0.99 and 3–10% of ‖W‖² lies outside the frame.

### `128_fixed_2e-2_9k` (2026-11-01)

The committed design at `b3adad8` plus the wandb unit as first written (its diff is next to the run, `128_fixed_2e-2_9k.uncommitted.diff`; it differs from `ac730aa` only in logging, tests and docs): `configs/128_smoke.toml` with 9000 steps, warmup 450, eval every 500, checkpoint every 1000. About 200k rows sampled with replacement, so each seen about 11 times. 13:42 to 14:29, median 846 img/s from step 11. wandb `pink-marker/tiny-dit`, run `w0j9lfhv`.

| steps | loss | t < 0.1 | 0.1–0.2 | 0.2–0.3 | 0.3–0.4 | 0.4–0.5 | 0.5–0.6 | 0.6–0.7 | 0.7–0.8 | 0.8–0.9 | > 0.9 | `grad_capture` | `grad_moment_cosine` |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 201–300 | 1.9587 | 2.985 | 2.719 | 2.441 | 2.212 | 2.024 | 1.829 | 1.681 | 1.508 | 1.337 | 1.177 | 0.95 | −0.22 |
| 901–1000 | 1.6702 | 2.166 | 2.046 | 1.918 | 1.812 | 1.717 | 1.631 | 1.535 | 1.429 | 1.276 | 1.149 | 0.90 | −0.38 |
| 1901–2000 | 1.5275 | 1.671 | 1.636 | 1.623 | 1.609 | 1.569 | 1.543 | 1.493 | 1.413 | 1.277 | 1.163 | 0.90 | −0.37 |
| 2901–3000 | 1.4663 | 1.467 | 1.484 | 1.508 | 1.517 | 1.522 | 1.505 | 1.458 | 1.386 | 1.268 | 1.164 | 0.88 | −0.34 |
| 4901–5000 | 1.4197 | 1.307 | 1.373 | 1.425 | 1.460 | 1.472 | 1.455 | 1.438 | 1.387 | 1.267 | 1.136 | 0.87 | −0.32 |
| 6901–7000 | 1.3878 | 1.267 | 1.329 | 1.380 | 1.415 | 1.427 | 1.430 | 1.419 | 1.370 | 1.261 | 1.150 | 0.87 | −0.31 |
| 8901–9000 | 1.3695 | 1.247 | 1.304 | 1.356 | 1.394 | 1.398 | 1.408 | 1.402 | 1.365 | 1.265 | 1.132 | 0.85 | −0.29 |

- The low-t gap of the 300-step reads closes with time: the t < 0.1 bucket goes 2.99 → 1.25, the buckets above t = 0.7 barely move. Loss still falls at 9000 (−0.012 over the last 1000 steps); constant lr, no decay.
- The final linear's frame (`scripts/probe_frame.py`), checkpoints 1000 to 9000: overlap with the previous checkpoint 0.992 to 0.995 every 1000 steps; stable rank 14.0 / 12.7 / 11.4 / 10.8 / 9.6 / 8.7 / 9.0 / 8.3 / 7.8; directions above 10% of s₁ 48 / 65 / 70 / 73 / 67 / 65 / 68 / 66 / 63 of 128; share of ‖W‖² outside the current frame 0.44 / 0.42 / 0.40 / 0.38 / 0.36 / 0.35 / 0.34 / 0.33 / 0.33. After step 1000 the frame holds its subspace; the weight's energy concentrates in it, and what lies outside was written before step 1000.
- The eval grid at step 9000 (16 prompts × 2 seeds, guidance 4): colour, light and coarse layout per prompt, no recognisable objects; strong vertical streaks, mostly in the seed-0 column, and blocky horizontal banding.
- The log ends with `shutdown can only be called once` after wandb's sync; exit code 0, all 9000 steps and checkpoints written.

### `128_fixed_2e-2_final_r64_9k` (2026-11-01)

The same run with the final linear in its own Usui group at rank fraction 1.0 (r = 64), full fixed step: code `e9f156f` plus the scratch launcher copied next to the run (`128_fixed_2e-2_final_r64_9k.launcher.py`). 14:30 to 15:16, median 872 img/s from step 11. wandb run `h483k5gj`. Against `128_fixed_2e-2_9k`, loss and the t < 0.1 bucket:

| steps | r = 64 loss | r = 13 loss | r = 64, t < 0.1 | r = 13, t < 0.1 |
|---|---|---|---|---|
| 201–300 | 1.6985 | 1.9587 | 1.976 | 2.985 |
| 901–1000 | 1.5146 | 1.6702 | 1.474 | 2.166 |
| 2901–3000 | 1.4369 | 1.4663 | 1.310 | 1.467 |
| 4901–5000 | 1.4048 | 1.4197 | 1.257 | 1.307 |
| 6901–7000 | 1.3749 | 1.3878 | 1.238 | 1.267 |
| 8901–9000 | 1.3591 | 1.3695 | 1.226 | 1.247 |

- The gap narrows but holds at about 0.010 from step 7000 on, one seed each. Over steps 8901–9000 the r = 64 run is lower by 0.011 to 0.021 in every bucket below t = 0.6, by 0.007 at 0.6–0.7, and level above.
- Its final linear's frame also settles: overlap 0.981 to 0.993 per 1000 steps; stable rank 23.9 at 1000 to 15.7 at 9000; 121 to 126 of 128 directions above 10% of s₁; 11–14% of ‖W‖² outside the frame.
- The eval grid at step 9000 has more fine texture than the r = 13 grid; the vertical streaks and the horizontal banding are there in both, so the r = 13 frame does not cause them.

## Runs at usui lr 2e-3 (2026-11-01)

`configs/128_smoke.toml` at `e8ca690`: usui lr 2e-3, beta 0.9, rank fraction 0.1 (r = 64 on the final linear), warmup 100, batch 256 as 4 × 64, shards 0–3, records every 100 steps. Against the 2e-2 runs this changes four things at once (lr, beta 0.95 → 0.9, warmup 450 → 100, rank fraction 0.05 → 0.1 for every matrix but the final linear), so no single one is isolated. Configs and logs next to the runs.

### `128_fixed_2e-3_9k`, stopped at step 4300

Stopped by hand at 356–400 img/s per record, with the CPU precompute (about 10 cores) running alongside; the next run, alone, did 850 img/s. wandb run `jdixfisn`. Means per window, against the two 2e-2 runs:

| steps | loss 2e-3 | loss 2e-2 r64 | loss 2e-2 r13 | `grad_moment_cosine` 2e-3 | 2e-2 r64 | `projected_grad_norm` 2e-3 | 2e-2 r64 |
|---|---|---|---|---|---|---|---|
| 501–1000 | 1.544 | 1.545 | 1.726 | −0.39 | −0.36 | 0.029 | 0.018 |
| 1001–2000 | 1.461 | 1.481 | 1.577 | −0.45 | −0.33 | 0.042 | 0.013 |
| 3001–4000 | 1.388 | 1.426 | 1.448 | −0.43 | −0.34 | 0.044 | 0.012 |
| 4001–4300 | 1.377 | 1.414 | 1.431 | −0.42 | −0.34 | 0.042 | 0.012 |

- `grad_capture` ≈ 1.00 (0.87 at 2e-2); `relative_step` 0.0008 (window means 0.0031–0.0047); `tangent_live_fraction` 0.994–0.998 (1.000).
- The two-seed grid at step 4000 follows each prompt in colour and scene (the night street dark blue with vertical streaks, the frog green, the woman in red on green); the 2e-2 r64 grid at the same step has more high-frequency clutter and a shared window-like layout across prompts. The 4×4 grid of the new prompt set (seed 42), rendered from both step-4000 checkpoints into `<run>/eval_4x4/` by `runs/render_grid_4x4.py`: 2e-3 matches the monochrome warrior, the explosion in the park, the beach, the snowy mountains, the white studio backdrop and the desert; 2e-2 is busier, with blue clutter for the bear and the bulldog.

Fallback parameters, relative change ‖W_b − W_a‖/‖W_b‖ between checkpoints, mean over the parameters of a group:

| group | 1000→2000 | 2000→3000 | 3000→4000 |
|---|---|---|---|
| null embedding | 0.194 | 0.160 | 0.148 |
| final table | 0.358 | 0.139 | 0.092 |
| block tables | 0.353 | 0.236 | 0.191 |
| conv | 0.328 | 0.207 | 0.175 |
| biases | 0.253 | 0.156 | 0.119 |
| norm gains | 0.053 | 0.038 | 0.034 |
| Usui matrices, for scale | 0.161 | 0.125 | 0.108 |

- The null embedding's cosine to its init (the empty caption's embedding) is 0.763 at step 1000, then 0.778, 0.803, 0.819 at 2000–4000; its norm goes 1.058, 0.984, 0.927, 0.872 (init 1.000).

### `128_fixed_2e-3_b099_1k`: beta 0.99, 1000 steps

The same config at beta 0.99. wandb run `facr413t`, 850 img/s with nothing else running. Against `128_fixed_2e-3_9k` at the same step:

| step | loss β 0.99 | loss β 0.9 | `grad_moment_cosine` β 0.99 | β 0.9 | `projected_grad_norm` β 0.99 | β 0.9 |
|---|---|---|---|---|---|---|
| 300 | 1.769 | 1.720 | +0.09 | −0.19 | 0.011 | 0.019 |
| 500 | 1.652 | 1.613 | −0.01 | −0.35 | 0.012 | 0.022 |
| 1000 | 1.513 | 1.511 | −0.10 | −0.41 | 0.015 | 0.032 |

- Per t bucket at step 1000, beta 0.99 is lower by 0.024 below t = 0.1, by 0.013 at 0.1–0.2, and up to 0.007 higher at 0.3–0.7.
- `tangent_live_fraction` 0.997–0.999.
- The 4×4 grids at step 1000 (`eval_4x4/`) are close; the same compositions under the same noise.

### GPU precompute

- `scripts/precompute.py 128` on the RTX 4070 Super, both encoders fp32, TF32 convolutions on (the cuDNN default), shard 4: about 137 rows/s, the GPU at 99%, about 6 min per 50k-row shard plus its download (4 min 45 s for shard 4's 11.1 GB, about 39 MB/s). With the next download overlapping: 141 rows/s over shard 4's first 5k rows (2026-11-01).
- 32 latents of shard 0 decoded by Primal and encoded again: GPU with TF32 against CPU fp32 differ by 0.25% relative per latent (max 0.73%), against 0.17% for rounding the CPU latent to bf16; GPU with TF32 off matches CPU to 2e-6, at about 75 rows/s over shard 4's first 15k rows (2026-11-01).
- Shards 11–46 on the GPU, 2026-11-02 00:29–04:18: 1,740,848 rows in 3 h 48 min with the downloads, about 127 rows/s; the progress lines read 90–170 rows/s, median 132 (log `precomputed/precompute_128_11-46_gpu.log`).

### Shuffle and stream (2026-11-01)

- `scripts/shuffle.py` over shards 0–10 at 128 px: 549,945 rows, 9.98 GB out, 164 s, peak RSS 4.4 GB (log `precomputed/shuffle_128_0-10.log`).
- `scripts/shuffle.py` over shards 0–46 at 128 px, keeping each image once (`8bcffbb`), 2026-11-02 08:17 to 08:33: 1,925,416 rows of 2,290,793 read, 34.8 GB out (log `precomputed/shuffle_128_0-46.log`); peak memory not measured.
- `tinydit.data.Stream` on the CPU over the 0–10 file, micro-batch 64 cast to fp32, with a training run alongside: 48k rows/s over 2000 micro-batches after one, peak RSS 2.4 GB for the whole process (a scratch probe under `/usr/bin/time`, not kept).
- `128_2e-3_0-10_10ep`, the first run on the stream, trained at 746–846 img/s per 100-step record up to step 3700 and 847–862 from step 3800 to the end, against about 850 for `128_fixed_2e-3_b099_1k`. The slow records run past the first pass (step 2148), so a cold page cache does not explain them. The agent session's CPU test runs overlapped part of them (the suite at 22:25, about step 3400); what else ran is not recorded.
- Every run above it drew its batches with replacement from shards 0–3 (about 200k rows); a run on the stream sees each of 550k rows once per pass. Losses do not compare across that line without saying so.

### `128_2e-3_0-10_10ep`: 10 passes over shards 0–10 (2026-11-01)

Launched 2026-11-01 22:06 from the uncommitted tree that became tiny-dit `727d94c` and usuitrack `e7e5210` (the range-finder fit; wandb records the base `5fa9ce4`); edits between the launch and those commits touched comments, docstrings and tests. The DiT is the one before the final layer's rows moved into the shared projection (`8eb82be`). The smoke settings (usui lr 2e-3, beta 0.99, rank fraction 0.1, fallback lr 1e-3, warmup 100, batch 256 as 4 × 64), 21,482 steps = 10 passes of the 549,945-row shuffled file. Records average 100 steps; a pass's mean is over the records that end in it. wandb run `q49embk5`; config, checkpoints every 2000 steps and the eval grid every 1000 in the run directory.

| pass | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| mean loss | 1.692 | 1.479 | 1.441 | 1.419 | 1.405 | 1.396 | 1.390 | 1.385 | 1.380 | 1.377 |

- Loss at the last record 1.375. Per t bin (10 bins, low t first): 1.167, 1.250, 1.308, 1.355, 1.378, 1.416, 1.457, 1.443, 1.340, 1.177; the peak moved from bin 6 at step 2100 (1.593; ends 1.281 and 1.215) to bin 7, where it is in every record from step 5900 on.
- Usui, per record from step 1000 on: `grad_capture` 0.94–0.98, `grad_moment_cosine` −0.08 at step 1000, then −0.11 to −0.17; `tangent_live_fraction` 0.994–0.997; `relative_step` 0.0008 at step 1000 falling to 0.0005. No nonfinite gradients. `fallback_grad_norm` rises from 0.019 at step 1000 to 0.029–0.040 from step 10500.
- The 4×4 grid (seed 42), step 4000 against the last: the last is more saturated, the teal of step 4000 mostly replaced by blue. The ice-cliff landscape reads as blue sky and cloud, the warrior and the woman in the dark hall keep their monochrome layout, the beach is blue sea under a figure, the studio shot a dark figure on white, and the sign prompt an orange rectangular object. No coherent anatomy or faces at either step.

## 1000-step ablations at 128 px (2026-11-01 23:56 to 2026-11-02 00:29)

tiny-dit `8eb82be` (the final layer's rows in the shared projection, text through `rms_norm`; code unchanged to `050dffa`), usuitrack `619b6e8`, the shuffled shards 0–10, seed 0, so every arm sees the same rows in the same order. Each arm is the smoke settings (usui lr 2e-3, beta 0.99, rank fraction 0.1, fallback lr 1e-3, fallback betas (0.9, 0.99), warmup 100, batch 256 as 4 × 64) for 1000 steps with one change (batch 64 also takes 4000 steps and warmup 400). Configs in the run directories, the two launchers that patch `tinydit.optim` next to their runs, the queue at `runs/ablations_queue_2026-11-01.sh`; one seed per arm, queued in series. Loss and the other columns but img/s are means of the records over images 128k–256k (steps 501–1000; batch 64: 2001–4000); img/s spans all records. t bins 1 to 10, low t first.

| arm | change | loss | Δ base | bin 1 | bin 2 | bin 3 | bin 5 | bin 10 | `grad_moment_cosine` | img/s |
|---|---|---|---|---|---|---|---|---|---|---|
| `128_abl_base` | none | 1.643 | | 1.536 | 1.581 | 1.647 | 1.742 | 1.209 | −0.07 | 765–844 |
| `128_abl_final_fallback` | final linear on the fallback at 1e-3, not its r = 64 frame | 1.567 | −0.076 | 1.193 | 1.333 | 1.485 | 1.679 | 1.210 | −0.10 | 839–865 |
| `128_abl_fallback_lr3e-4` | fallback lr 3e-4 | 1.657 | +0.014 | 1.581 | 1.614 | 1.669 | 1.751 | 1.215 | −0.03 | 827–856 |
| `128_abl_fallback_b0999` | fallback betas (0.9, 0.999) | 1.641 | −0.002 | 1.533 | 1.578 | 1.645 | 1.741 | 1.208 | −0.07 | 834–855 |
| `128_abl_batch64` | batch 64 as 1 × 64, 4000 steps, warmup 400 | 1.516 | −0.127 | 1.258 | 1.363 | 1.469 | 1.587 | 1.210 | −0.08 | 350–355 |

- Noise: the run-to-run spread of the GPU smoke runs is at most 0.003 per window (one seed pair, 300 steps, an earlier build). A nearer yardstick is 0.011, the gap between the base arm and the 10-pass run over the same window (below), for a changed model and init. Against it the final linear on the fallback and batch 64 stand out, the lower fallback lr (+0.014) is borderline, and betas 0.999 cannot be told apart.
- The final linear on the fallback gains most at low t, tapering with t: bins 1–7 lower by 0.34, 0.25, 0.16, 0.10, 0.06, 0.04, 0.02, bins 8–10 equal. At the last record its bin 1 is 1.182, below the 10-pass run's 1.281 at step 2100 (FACTS, `128_2e-3_0-10_10ep`). Its `fallback_grad_norm`, which now includes the final linear, is 0.064 against 0.017 (window means; 0.055 against 0.020 at the last record).
- The lower fallback lr is worse in every bin, by 0.005 to 0.044, most at low t.
- Batch 64 is better per image in bins 1–9, by 0.28 at bin 1 down to 0.014 at bin 9, and equal in bin 10. It trains at 353 img/s against about 850. By `imgs_per_s` it took 723 s; its last record (steps 3601–4000), 1.496, is against 1.510 for the 10-pass run's record at step 2200 (713 s), a batch-256 run of the model before `8eb82be` that ran at about 790–800 img/s over that stretch. Less the 0.011 of the model change, about a tie per training time. `tangent_live_fraction` 0.992–0.993 in the window, as at batch 256.
- The same window of the 10-pass run (same batches, the model before `8eb82be`, its init drawn differently because `ada` changed shape) is 1.654, 0.011 above the base arm; bin 1 1.597 against 1.536.
- Eval grids (seed 42, by eye): at step 1000 the base arm is blotches of colour; in the final-fallback arm the cat (cell 0) is a dark figure on white and the studio shot (cell 13) white, most other cells as noisy as the base. Batch 64 at step 4000: the warrior (cell 2) is high-contrast black and white, the ice cliff (cell 1) nearly flat green.


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

## Token signal through the blocks (2026-11-02)

Source: scratch `heldout/token_signal.py` (re-implements `Block.forward`, checked equal to it), 256 rows of shard 12 with their captions (held out of the 0–10 runs, but trained on once in a sweep run; the sweep arms share them), fixed noise (seed 1234), at t = 0.05, 0.25, …, 0.95; on `128_sweep_lr2e-3` steps 1k, 2k, 4k, 8k, 15042 and `128_sweep_lr4e-3` step 15042. Per image over its 64 tokens: mean off-diagonal cosine, effective rank (exp entropy of the Gram's eigenvalues; 63 at most after centering), the share of the stream's energy in the token mean (`common`), then means over images.

| at the output of blocks 4–10 | lr 2e-3, 1k | lr 2e-3, 15042 | lr 4e-3, 15042 |
|---|---|---|---|
| `common`, t 0.05 / 0.45 / 0.95 | 0.44–0.57 / 0.39–0.50 / 0.55–0.85 | 0.90–0.94 / 0.81–0.84 / 0.92–0.95 | 0.94–0.95 / 0.88–0.89 / 0.96–0.97 |
| effective rank, raw | 9–15 / 9–18 / 1.4–10 | 1.5–1.8 / 2.7–3.2 / 1.4–1.7 | 1.3–1.4 / 2.0–2.2 / 1.2–1.3 |
| effective rank, centered | 39–40 / 24–39 / 3–40 | 26–30 / 35–37 / 24–33 | 21–23 / 31–34 / 18–21 |

- The embedding's `common` is 0.02–0.12. Each image's stream gains a vector shared by all its tokens, which grows with training and is larger at lr 4e-3; the token-specific part keeps a centered effective rank of 18–40, so it is a common mode, not a rank collapse.
- Attention writes it: blocks 0–1 raise the raw token cosine by +0.24 / +0.24 (lr 2e-3, t 0.05) up to +0.78 / +0.12 (lr 4e-3, t 0.95), later blocks by under +0.1. The MLP branch changes it by −0.04 to +0.12, mostly within ±0.02.
- SwiGLU decorrelates: the token cosine of the hidden `silu(W_g y) * W_u y` is below that of its input `y` in every block but the last (e.g. 0.67 → 0.61 down to 0.27 → 0.13), and its effective rank higher. The last block at t 0.95 is near rank one on both sides (input and hidden cosine 0.97–1.00) and its output RMS is 39–134 against 2–16 before it.
- Gate RMS (adaLN output) 0.14–0.35 at t ≤ 0.45, attention 0.44–0.75 and MLP 0.25–0.57 at t 0.95.

## What each branch does (2026-11-02)

Source: scratch `heldout/branch_role.py` (re-implements the forward, checked against `DiT.forward`), the same 256 rows of shard 12 and noise as Token signal, `128_sweep_lr2e-3` steps 1k, 4k, 15042. Per block: the loss rise when its attention or MLP branch is dropped, the branch's write RMS over the stream RMS, and |grad| / |W| per matrix under plain autograd (no Molegrad). t 0.05 / 0.45.

| | step 1k | step 15042 |
|---|---|---|
| drop attention, block 0 | +0.067 / +0.127 | +0.167 / +0.536 |
| drop attention, blocks 2–10 | +0.011–0.025 / +0.006–0.017 | +0.010–0.101 / +0.021–0.079 |
| drop MLP, block 0 | +0.003 / +0.021 | +0.029 / +0.121 |
| drop MLP, blocks 2–10 | +0.003–0.014 / +0.005–0.010 | +0.003–0.026 / +0.017–0.040 |
| drop MLP, block 11 | +0.004 / +0.025 | +0.250 / +0.323 (+0.664 at t 0.95) |
| MLP write / stream, blocks 2–10 | 0.04–0.10 / 0.12–0.41 | 0.02–0.10 / 0.06–0.16 |

- The middle MLPs cost about as much to drop as the middle attention branches, and their cost grew from step 1k to 15042. At t 0.95 they cost nothing (within ±0.003); there block 0's attention and block 11's MLP carry the prediction, the latter writing 3.4–5.3× the stream RMS.
- |grad| / |W| of `w_up` and `w_down` is the same order as `qkv`'s (1e-4 to 1e-3) in every block, with no decay toward the input. Usui's fixed step does not scale with the gradient's size anyway.
- Whole blocks skipped together (scratch `heldout/drop_many.py`, same rows), loss at t 0.05 / 0.45 / 0.95, step 1k → 15042. None skipped: 1.165 / 1.739 / 1.185 → 1.122 / 1.462 / 1.172. Blocks 2–10: 2.109 / 3.039 / 1.387 → 5.125 / 7.511 / 1.281. Blocks 4–8: 1.613 / 1.968 / 1.215 → 2.503 / 1.850 / 1.184. Blocks 0 and 11: 1.216 / 1.958 / 1.223 → 1.373 / 2.853 / 5.127. One middle block alone costs little; the nine together carry most of the prediction, more so at 15042 than at 1k.

## LR sweep and seed merge (2026-11-02)

The LR sweep (PLAN C2): `128_sweep_lr{1e-3,2e-3,4e-3}`, one pass over `128_shuffled_0-46.parquet` (15042 steps, batch 128, warmup 752, no decay; tiny-dit `78fded7`). `lr2e-4` stopped in its first steps: cuSOLVER's fp32 `eigh` in usuitrack `frame.tangent_spectra` returned non-orthonormal vectors for a tangent Gram of ~1e-22 entries (a matrix behind a gate still near zero), the turn left the manifold (QᵀQ − I 1.7, then ~1e25), and `eigh` raised; with each Gram divided by its mean diagonal for the `eigh` (usuitrack `zero-init`, uncommitted; 1.61 ms for 48 Grams against 1.52 unscaled), the same arm ran 300 steps clean (scratch `eighrepro/`). Scored with scratch `heldout/score.py`: the first 1024 rows of shards 12 and 40, trained on once in every arm, fixed noise (seed 1234), t at the 10 bin midpoints 0.05 … 0.95, mean over t.

| usui lr | shard 12 (dalle3), caption / null | shard 40 (llava flux), caption / null |
|---|---|---|
| 1e-3 | 1.3816 / 1.3840 | 1.1805 / 1.1852 |
| 2e-3 | 1.3726 / 1.3752 | 1.1718 / 1.1745 |
| 4e-3 | 1.3691 / 1.3709 | 1.1679 / 1.1712 |

- Loss falls with lr, by 0.009 from 1e-3 to 2e-3 and 0.004 from 2e-3 to 4e-3; the caption gap is 0.002–0.003 in every arm. Daniel, on the samples: "2e-3 looks best even though [4e-3] has better loss".

The same checkpoints on the checkpoint score's eval set (SPEC §8: the 1000 Lunara renders of `moonworks/lunara-art-eval`, never trained on), scored by `train.score` from scratch `heldout/score_lunara.py`: noise seed 1, t at `t_quantiles(10, 1.0)`, about 4 s per checkpoint on the 4070 Super.

| checkpoint | caption | null |
|---|---|---|
| `lr2e-4` final (rerun with the fix, its own `score.jsonl`) | 1.5584 | 1.5598 |
| `lr1e-3` final | 1.5207 | 1.5215 |
| `lr2e-3` step 1000 / 4000 / 8000 | 1.6442 / 1.5564 / 1.5299 | 1.6438 / 1.5570 / 1.5307 |
| `lr2e-3` final | 1.5122 | 1.5130 |
| `lr4e-3` final | 1.5106 | 1.5109 |
| `128_4e-3_ep2` final (epoch 2, usui lr 2e-3, batch 128; its own `score.jsonl`) | 1.4929 | 1.4951 |
| `128_4e-3_ep3` final (epoch 3, usui lr 2e-3, batch 256) | 1.4873 | 1.4895 |
| `128_4e-3_ep4` final (epoch 4, usui lr 1e-3, fallback 2.5e-4, batch 256) | 1.4821 | 1.4844 |
| `128_d768x12_ep1` final (d768, epoch 1, usui lr 8e-3, batch 256) | 1.4807 | 1.4811 |
| `128_d768x12_ep2` final (d768, epoch 2, usui lr 4e-3, batch 256) | 1.4631 | 1.4643 |
| `128_d768x12_ep3` final (d768, epoch 3, usui lr 2e-3, batch 256) | 1.4557 | 1.4573 |
| `128_d768x12_ep4` final (d768, epoch 4, usui lr 1e-3, batch 256) | 1.4518 | 1.4536 |
| `128_d1024x12_ep1` final (d1024, epoch 1, usui lr 8e-3, batch 256) | 1.4621 | 1.4613 |
| `128_d1024x12_ep2` final (d1024, epoch 2, usui lr 4e-3, batch 256) | 1.4428 | 1.4439 |
| `128_d1024x12_ep3` final (d1024, epoch 3, usui lr 2e-3, batch 256) | 1.4343 | 1.4359 |
| `128_d1024x12_ep4` final (d1024, epoch 4, usui lr 1e-3, batch 256) | 1.4302 | 1.4323 |
| `128_d768x12_noconv_ep1` final (d768, epoch 1, usui lr 8e-3, conv deleted; 854 img/s against 834) | 1.4805 | 1.4807 |
| `128_d768x12_lr1e-2_ep1` final (d768, epoch 1, usui lr 1e-2, one change from `_ep1`) | 1.4799 | 1.4797 |
| `128_d768x12_finalr64_ep1` final (d768, epoch 1, usui lr 8e-3, final linear in Usui at rank 64, one change from `_ep1`) | 1.4911 | 1.4914 |
| `128_d768x12_slice_ep1` final (d768, epoch 1, usui lr 8e-3, no final layer: first 128 stream channels are the output, identity patch embed; 838 img/s) | 1.4745 | 1.4749 |
| `128_d768x12_solver1_ep1` final (slice arm plus a tied zero-init 768 x 768 linear residual, 4 iterations, on Usui; `tangent_live_fraction` 0.96 against 0.99, `projected_grad_norm` 0.020 against 0.0043 from step 1000 on) | 1.4816 | 1.4817 |
| `128_d768x12_solvermlp_ep1` final (slice arm plus a tied zero-init SwiGLU residual, 768 -> 1536 -> 768, 4 iterations, on Usui; 771 img/s, `tangent_live_fraction` 0.95, `projected_grad_norm` 0.0058) | 1.4781 | 1.4785 |

Parameters of the slice-readout model (2026-11-03, MLP hidden 4 x width, counted from `DiT`): 125.4M at width 768 and depth 12, 222.5M at 1024 and 12, 293.8M at 1024 and 16. The `128_d768x12_*` rows above count the earlier architecture (final layer, conv, MLP at 2 x width) except where the arm says otherwise.

- Same lr order as on shards 12 and 40; 2e-4 ends 0.046 above 2e-3, its training loss (mean of the last 10 records) 1.396 against 1.326, so 2e-4 is not too big but too small. The caption gap is under 0.001, and negative at step 1000: the captions here are one short line each (PLAN B3).

Seed merge (scratch `seedmerge/`): the shuffled file split into three disjoint thirds, one 5000-step child on each at lr 2e-3 (warmup 250), their step-5000 weights merged by multislerp, then 5000 steps from the merged weights (fresh optimizer state) over the whole file in the sweep's order, warmup 752. Same scoring, shard 12 caption:

| | loss |
|---|---|
| `128_sweep_lr2e-3`, step 5000 | 1.4060 |
| children 0 / 1 / 2, step 5000 | 1.4059 / 1.4058 / 1.4036 |
| the merge | 2.5958 (null 2.5513) |
| 5000 steps from the merge | 1.3970 (step 1000: 1.4837) |

Sweep merge (scratch `sweepmerge/`, same scoring as the Lunara table above): the arms' final checkpoints share their init and data order (seed 0). Lunara caption / null:

| | caption | null |
|---|---|---|
| `lr4e-3` final | 1.5106 | 1.5109 |
| multislerp of 2e-3 + 4e-3 | 2.1471 | 2.1865 |
| multislerp of 1e-3 + 2e-3 + 4e-3 | 2.2553 | 2.2354 |
| multislerp of all four | 2.2702 | 2.2631 |
| mean of 1e-3 + 2e-3 + 4e-3 | 2.4683 | 2.4436 |
| mean of all four | 3.0589 | 3.0599 |

- Daniel, before: "Merging our 4 lr sweep checkpoints may be crazy again but maybe its good too, i do have hope". Every merge is broken, like the seed merge's, though these arms share init and data order.

Weight norms of `128_sweep_lr4e-3` (scratch `norms.py`; Frobenius norm over all blocks for `blocks.*`, init rebuilt at seed 0):

| | init | 1k | 2k | 4k | 8k | 12k | 15042 |
|---|---|---|---|---|---|---|---|
| `qkv` | 135.8 | 151.8 | 178.6 | 224.0 | 293.6 | 348.5 | 384.8 |
| `w_up` | 110.8 | 120.0 | 134.9 | 156.6 | 195.5 | 228.2 | 249.7 |
| `w_down` | 78.4 | 85.2 | 96.5 | 117.5 | 146.1 | 166.7 | 179.7 |
| `out` | 78.4 | 83.4 | 92.2 | 107.4 | 131.6 | 150.8 | 163.4 |
| `final` (fallback) | 0 | 6.3 | 9.2 | 13.8 | 19.7 | 23.8 | 26.2 |
| `ada.1` (fallback) | 25.1 | 32.7 | 44.0 | 57.1 | 70.5 | 80.2 | 86.4 |
| null embedding | 1.00 | 1.19 | 1.39 | 1.70 | 2.38 | 2.95 | 3.36 |

- No norm has levelled off; the Usui matrices grow by about 2.5–3× their init, slowing (qkv +17 per 1k steps from 4k to 8k, +12 from 12k to 15042). Nothing sets a target norm: no weight decay on either optimizer, and Usui's fixed step does not depend on `‖W‖`. The same absolute step on a weight 2.8× larger is a 2.8× smaller relative step: an implicit decay with no schedule.

- The merge alone is broken, worse than any child, and better with the null embedding than with the caption; 5000 steps from it end 0.009 below the single run at step 5000, after 4× its compute. Daniel: "no merges just fresh init each time".

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

Source: `runs/128_d1024x16_swpin_5ep/` (`configs/128_d1024x16_swpin_5ep.toml`): `swpin_3k` for 5 epochs, 37,605 steps of 256; usui relative step 2e-3 and fallback lr 2.5e-3, warmup 500, both decaying linearly to a tenth; t σ 1. Checkpoints at 7500, 15000, 22500, 30000, 37500, 37605.

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

## Spectra of the pinned weights (G9, 2026-11-05)

Source: scratch `spectra.py` on the CPU, `/mnt/mars/ai/tiny-dit/probes/spectra_swpin_5ep.json`: every Usui matrix of `128_d1024x16_swpin_5ep` at steps 7500, 15000, 22500, 30000, 37605 against its init, rebuilt from seed 41 (every pinned norm reads 1.000 of the rebuilt init's, which checks the rebuild). Effective rank is exp of the entropy of the normalized σ² spectrum, as a fraction of the shorter side k; Usui's frame rank is 0.1 k (102 for k = 1024).

| family (mean over blocks) | eff rank at init | 7500 | 37605 | cos(W, W₀) 7500 / 37605 | interval update ‖W_k − W_{k−1}‖/‖W‖, 7500→15000 / 30000→37605 | interval update r90 / k, same |
|---|---|---|---|---|---|---|
| `qkv` | 0.846 | 0.193 | 0.170 | 0.154 / 0.024 | 0.92 / 0.36 | 0.166 / 0.149 |
| `gate` | 0.606 | 0.100 | 0.082 | 0.148 / 0.021 | 0.80 / 0.29 | 0.102 / 0.089 |
| `out` | 0.606 | 0.106 | 0.093 | 0.165 / 0.029 | 0.75 / 0.25 | 0.101 / 0.089 |
| `w_gate` | 0.882 | 0.212 | 0.195 | 0.147 / 0.032 | 0.85 / 0.32 | 0.191 / 0.174 |
| `w_up` | 0.882 | 0.213 | 0.199 | 0.138 / 0.023 | 0.87 / 0.31 | 0.189 / 0.175 |
| `w_down` | 0.882 | 0.210 | 0.186 | 0.135 / 0.026 | 0.87 / 0.30 | 0.184 / 0.169 |
| `ada.1` | 0.882 | 0.361 | 0.264 | 0.234 / 0.029 | 1.02 / 0.37 | 0.395 / 0.251 |
| `time_mlp.0` | 0.882 | 0.345 | 0.108 | 0.471 / 0.211 | 0.88 / 0.33 | 0.223 / 0.184 |
| `time_mlp.2` | 0.606 | 0.286 | 0.239 | 0.288 / 0.066 | 1.05 / 0.37 | 0.284 / 0.126 |
| `text_proj` | 0.687 | 0.223 | 0.151 | 0.407 / 0.170 | 0.80 / 0.32 | 0.111 / 0.108 |
| `patch_embed` | 0.946 | 0.944 | 0.937 | 0.977 / 0.964 | 0.11 / 0.06 | 0.773 / 0.781 |

- The weights do not grow from a subspace to full rank; they start full rank (random init) and concentrate. By step 7500 the effective rank has fallen to about a quarter of the init's, and it barely moves after 15000. At 37605, block 8: 90 % of σ² sits in 148 of 1024 directions for `qkv`, 72 for `gate`, 80 for `out`, 170 for `w_up`, 160 for `w_down`; 277 for `ada.1`. The largest σ² stays 1–5 % of the total (`time_mlp.0` 42 %: one dominant direction from the scalar t).
- The init is forgotten, not kept: cos(W, W₀) ends at 0.02–0.03 for every block matrix, and the overlap of W's top-r left singular subspace with W₀'s is at chance (r over the side length) from step 7500 on. `patch_embed` is the exception (cos 0.96, mostly its identity rows).
- The late update (30000 → 37605) lies in W's own top-102 singular subspace on the stream side, the 1024-wide side the frame is on: right side for `qkv` 0.52, `gate` 0.89, `w_gate` 0.45, `w_up` 0.49, `text_proj` 0.77; left side for `out` 0.87, `w_down` 0.48 (chance 0.10). On the other side it is near chance (0.05–0.15 against 0.025–0.10). Block 8 except `out` (block 2) and `w_down` (block 14); `ada.1` 0.33 right.
- Uniform in depth: the effective ranks of `qkv`, `w_gate`, `w_down` differ by under 0.03 across blocks 0–15; block 15's MLP is lowest (0.175 / 0.166).
- Not separated: how much of the concentration is Usui's rank-0.1 frame, how much the pin's washing out of the init (each step rescales the init's share down), and how much the task. The unpinned `5ep` checkpoints and a different rank fraction would separate them; neither is measured.

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
- Usui's `grad_capture` (‖G Q‖/‖G‖, the gradient share its rank-0.1 frame holds) reads 0.98–1.03 over the whole `swpin_5ep` run (`train.jsonl`; above 1 under Molegrad, where the full gradient is not formed). It is a probed estimate: comparable across runs and steps, not the true capture (Daniel, 2026-11-05).

## Momentum cosine on a noisy quadratic (2026-11-06)

`scripts/probe_momentum_cosine.py`, CPU, seed 0: one quadratic direction (sharpness 1), white gradient noise, UsuiTrack's moment `M = beta M + (1 - beta) g` written as `-lr M` (the polar map, the gradient normalisation and the frame are not modelled). The characteristic polynomial is `z^2 - (1 + beta - eta lam (1 - beta)) z + beta`; its roots reach -1 at `eta lam = 2 (1 + beta) / (1 - beta)` (38 at beta 0.9, 78 at 0.95, 398 at 0.99), and the root radius is `sqrt(beta)` below it.

| `eta lam` / bubbleforce | 0.01 | 0.1 | 0.3 | 0.6 | 0.9 | 0.99 |
|---|---|---|---|---|---|---|
| `cos(g_t, M_{t-1})`, beta 0.9 | -0.04 | -0.26 | -0.52 | -0.76 | -0.95 | -0.995 |
| beta 0.95 | -0.06 | -0.29 | -0.53 | -0.77 | -0.95 | -0.995 |
| beta 0.99 | -0.08 | -0.31 | -0.55 | -0.77 | -0.95 | -0.995 |

The cosine is nearly a function of the fraction alone. The real `grad_moment_cosine` at step 1000 is -0.41 at beta 0.9 (`128_fixed_2e-3_9k`) and -0.10 at beta 0.99 (`128_fixed_2e-3_b099_1k`), same lr, other settings equal. The script fits the fraction at which beta 0.9 reads -0.41 (0.203) and scales it to beta 0.99, where bubbleforce is 10.5 times larger (0.0194): it reads -0.124. That is consistent with the observed -0.10 in one pair of readings with one fitted parameter. It assumes the same sharpness in both runs, which an edge-of-stability equilibrium would not give, the beta 0.99 reading is still drifting (+0.09, -0.01, -0.10 at steps 300, 500, 1000), and a real matrix has many directions of different sharpness. Any monotone curve would order the two runs the same way.

What the toy cannot say: the lr direction. The two lr readings (`128_fixed_2e-2_9k`, -0.33 to -0.36, against `128_fixed_2e-3_9k`, -0.39 to -0.45, steps 501 to 4300) also differ in beta (0.95 against 0.9), warmup and rank fraction, so no lr effect can be read from them. The early positive values of the smoke runs (+0.44 to +0.51 over steps 3 to 11) are not in the toy, which has no gradient signal.

## Learning-rate jump from `swpin_cont_3ep` (C4, 2026-11-06)

Source: `runs/128_d1024x16_shock_lr{1e-4,4e-4,1e-3}/` (`sharpness.jsonl`, `train.jsonl`; `scripts/probe_sharpness.py`; configs `configs/128_d1024x16_shock_*.toml`). Each arm continues `128_d1024x16_swpin_cont_3ep` from step 62168 for 300 steps at a constant Usui relative step (fallback 1.25x it), with the optimizer state of the checkpoint. The top Hessian eigenvalue is read at 17 checkpoints on one fixed batch of the first 128 Lunara rows (one fixed noise and t each), fp32, central-difference Hessian-vector products, 12 power iterations; 40 iterations agree to four digits at the start and the three end checkpoints.

| usui lr after the jump | top eigenvalue at 62168 / at 62468 | change | mean `grad_moment_cosine`, steps 12 to 300 | mean loss, 7 logs to step 150 | mean loss, last 4 logs |
|---|---|---|---|---|---|
| 1e-4 (control) | 28.0 / 28.9 | +3 % | -0.04 | 1.132 | 1.129 |
| 4e-4 | 28.0 / 26.6 | -5 % | -0.08 | 1.136 | 1.137 |
| 1e-3 | 28.0 / 18.3 | -35 % | -0.11 (-0.15 at step 12) | 1.148 | 1.155 |

- The control stays between 27.9 and 29.1 at every checkpoint. The 4e-4 arm falls to 25.4 by step 62340 and sits at 26 to 27 after. The 1e-3 arm falls to 21.8 by step 62280, to 19 by 62340 and reads 18.3 to 19.7 from 62400, still drifting down slowly at the end.
- The cosine is more negative at the higher lr, and in the 1e-3 arm it eases from -0.15 to about -0.10 over the first 100 steps as the eigenvalue falls.
- No loss spike in any arm. The losses are the means of 16 logged micro-batch averages, so the 0.01 to 0.02 gaps are at the noise of that estimate; the sharpness change at 1e-3 is not.
- One seed and one fixed batch per arm; the batch is the eval rows (conditional loss, no prompt dropout).
- Not read from this: how the cosine depends on beta (the arms share beta 0.99), and whether the lower sharpness is kept when the lr is lowered again.

## Fine-T2I image sizes (B6, 2026-11-06)

Source: Hugging Face `ma-xu/fine-t2i`, file listing with sizes and the first tar of each of the five groups, read on `/mnt/luna/tiny-dit/probe_fine/` (about 1 GB; image headers only, sizes counted, nothing encoded). The groups: `curated` 192 tars, 279 GB, 1,024 images per tar; `synthetic_{enhanced,original}_prompt_{random,square}_resolution` 1,621 / 1,688 / 1,544 / 1,311 tars, 510 / 513 / 554 / 467 GB, 1,000 images per tar. In all, 6,363 files and 2.3 TB, about 6.16M synthetic pairs and 168k curated photos.

- The two square groups have six sizes: 512, 768, 1024, 1536, 2048 and 2560 px, all 1:1.
- The two random-resolution groups have 26 distinct sizes in 1,000 images, 9 aspect ratios: 1:1, 5:4, 4:5, 4:3, 3:4, 3:2, 2:3, 16:9 and 9:16. The json carries `image_aspect_ratio` and `image_resolution`.
- The curated photos have 589 distinct sizes in 1,024 images, camera sizes up to 6000 x 4000; 3:2 and 2:3 dominate (433 and 298 of 1,024 at the nearest 64-token grid), 4:3, 3:4 and 16:9 follow.
- Token grids (8 x 8 tokens is a 128 px square) of about 64 tokens with the best aspect, resized to the grid's exact size without a crop: the synthetic sets stretch by at most 5 % (1:1 0, 4:3 and 3:4 3.6 % and 3.7 %, 5:4 and 4:5 2.9 %, 16:9 and 9:16 3.1 % and 3.0 %, 3:2 and 2:3 4.8 % and 5.0 %); the sample of curated photos stretches by a median 4.8 %, at most 7.6 %.

## Cross attention and conv period at d768x12, 3000 steps (F3, G10, 2026-11-06)

Lunara eval caption loss on the same rows in the same order, batch 256, seed 41, usui lr 2e-3, `configs/128_d768x12_*`; images per second is the median of the training log. The baseline and the conv arms train on the old 47-shard set, the cross arm on the 16-shard trial set (798,005 rows); no arm passes over its data once.

| arm | 1000 | 2000 | 3000 | img/s |
|---|---|---|---|---|
| `swpin_3k`, pooled embedding only | 1.5532 | 1.5023 | 1.4855 | 682 |
| `swpin_cross_3k`, cross attention in every block, pooled embedding in the adaLN | 1.5670 | 1.5159 | 1.5024 | 488 |
| `conv2_3k`, conv in blocks 0, 2, 4, … (1/2) | 1.5449 | 1.5003 | 1.4852 | 822 |
| `conv3_3k`, conv in 2/3 of the blocks | 1.5451 | 1.5015 | 1.4859 | 874 |
| `conv4_3k`, conv in 3/4 of the blocks | 1.5474 | 1.5037 | 1.4884 | 901 |

Against the baseline: cross attention +0.014 / +0.014 / +0.017 and 0.72× the speed, no recompile warnings in its log. Conv 1/2 is level (−0.0003 at 3000) at 1.20× the speed, conv 2/3 level (+0.0004) at 1.28×, conv 3/4 +0.003 at 1.32×. Conv has the lower loss at 1000 steps in all three (−0.006 to −0.008) and gives it back by 3000.
