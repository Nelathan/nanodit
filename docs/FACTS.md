# Facts

Measured numbers only: statistics, throughput, memory, run results. Each entry gives the number, its source (script or run) and the date; a new read replaces the old entry rather than being appended. Estimates stay with the question they inform in `PLAN.md`.

## Sample and shard

Probe sample = first 256 images/captions of `data_512_2M/data_000000.tar` (not random), 128 px uses all 256, 256 px the first 128. Source: `scripts/probe_primal.py`, `scripts/probe_embeddinggemma.py`, 2026-10-31.

- Shard 0: 7,475,189,760 bytes, 50,000 samples (50,000 `.jpg` + 50,000 `.json`), images 512×512, caption in `{"prompt": "..."}`, keys `flux_512_100k_<8 digits>`. Other shards are 7.5 to 11.1 GB (repo listing).
- Shards 0–3, precomputed 2026-11-01 (`precompute.py` log, parquet keys): 0 and 1 are `flux_512_100k_*`, 50,000 pairs each; 2 and 3 are `dalle3_1m_*`, 49,985 and 49,982 pairs, keys numbered 0–49999 with gaps. Shard 2's tar (11.1 GB) was re-downloaded and listed: 49,985 complete jpg+json pairs, all in the parquet, so the gaps are in the source. Shard 3's tar was not re-checked.

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

`tinydit.data.load_shards` over shards 0–3 at 128 px, 2026-11-01, CPU, `uv run python -c` on `sorted(glob('.../precomputed/128/*.parquet'))` under `/usr/bin/time`: 199,967 rows, 3.6 GB of bf16 latents + embeddings, 19 s, peak RSS 11.4 GB (the machine has 31 GB). Latent std 1.06, embedding norm 1.000 (every 97th row).

## Model and Usui wiring (2026-10-31, `src/tinydit/model.py`, `scripts/probe_usui.py`)

- Parameters at depth 12: 37,194,624. Blocks 34.70M (qkv 9.44M, SwiGLU 18.87M, sigmoid gate 3.15M, out-proj 3.15M, conv 55k, tables 37k); shared adaLN projection 1.58M; time MLP 394k; text projection 394k; patch embed 66k; final linear 66k. 2-D 37.13M, 1-D 8,064, 4-D (conv) 55,296.
- Usui on `lab`, 5 steps at lr 1e-2: a zero-init 128×512 matrix stays at norm 0.0; a (512,1,3,3) conv weight and a (64,) vector are rejected; a shared-projection-like matrix with zero gate rows and std-0.02 shift/scale rows grows its gate rows 0 → 0.074 (matrix norm 2.56); a [6, 512] table is accepted (frame [6, 3]). In the zero-init case the fan-in matrix feeding the zero one does not move either: its gradient is zero while the other is.
- Molegrad and accumulation, wiring check only (synthetic latents, depth 2, CPU, batch 32, usui lr 3e-3, fallback lr 1e-3, 60 steps, identical batches per arm; scratch script, not in the repo, so not reproducible from it): Molegrad 1×32 and 4×8 agree to 4 digits at every step (loss 2.5807 → 2.0579). Step 1 (final linear on the fallback): every tracked gradient exactly zero, nothing tracked moves. By step 10 `tangent_live_fraction` 0.91, `relative_step` ≈ 98% of lr. After 60 steps `blocks.0.qkv` shares 0.38 of its step-1 frame (G path without Molegrad: 0.50, loss 2.0544).

## adaLN at init (2026-10-31, `scripts/probe_init.py`)

Depth 12, seed 0, 16 timesteps spread over (0, 1), text = the null embedding or random unit vectors (the two agree within 5%). `c` per-element std 1.07. Block 0 shift/scale rows (shared projection + table): std 0.28 to 0.30. Final layer shift/scale (`final_table + c`): std 1.07, max |scale| 4.0 to 4.1.

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
