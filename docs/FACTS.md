# Facts

Measured numbers only: statistics, throughput, memory, run results. Each entry gives the number, its source (script or run) and the date; a new read replaces the old entry rather than being appended. Estimates stay with the question they inform in `PLAN.md`.

## Sample and shard

Probe sample = first 256 images/captions of `data_512_2M/data_000000.tar` (not random), 128 px uses all 256, 256 px the first 128. Source: `scripts/probe_primal.py`, `scripts/probe_embeddinggemma.py`, 2026-10-31.

- Shard 0: 7,475,189,760 bytes, 50,000 samples (50,000 `.jpg` + 50,000 `.json`), images 512×512, caption in `{"prompt": "..."}`, keys `flux_512_100k_<8 digits>`. Other shards are 7.5 to 11.1 GB (repo listing).

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

`scripts/precompute.py 128`, shard 0, 2026-10-31, 12 threads, machine shared (load ≈ 15): 9.0 rows/s for the first 2k rows, then a steady 8.2 rows/s (Primal encode and ST in series; 1/20.5 + 1/15.2 s per row would give 8.7), ≈ 100 min per 50k shard; peak RSS 9.9 GB (a 1024-row chunk held as fp32 at 512 px).

## Model and Usui wiring (2026-10-31, `src/tinydit/model.py`, `scripts/probe_usui.py`)

- Parameters at depth 12: 37,194,624. Blocks 34.70M (qkv 9.44M, SwiGLU 18.87M, sigmoid gate 3.15M, out-proj 3.15M, conv 55k, tables 37k); shared adaLN projection 1.58M; time MLP 394k; text projection 394k; patch embed 66k; final linear 66k. 2-D 37.13M, 1-D 8,064, 4-D (conv) 55,296.
- Usui on `lab`, 5 steps at lr 1e-2: a zero-init 128×512 matrix stays at norm 0.0; a (512,1,3,3) conv weight and a (64,) vector are rejected; a shared-projection-like matrix with zero gate rows and std-0.02 shift/scale rows grows its gate rows 0 → 0.074 (matrix norm 2.56); a [6, 512] table is accepted (frame [6, 3]). In the zero-init case the fan-in matrix feeding the zero one does not move either: its gradient is zero while the other is.
- Molegrad and accumulation, wiring check only (synthetic latents, depth 2, CPU, batch 32, usui lr 3e-3, fallback lr 1e-3, 60 steps, identical batches per arm; scratch script, not in the repo, so not reproducible from it): Molegrad 1×32 and 4×8 agree to 4 digits at every step (loss 2.5807 → 2.0579). Step 1: every tracked gradient exactly zero, nothing tracked moves. By step 10 `tangent_live_fraction` 0.91, `relative_step` ≈ 98% of lr. After 60 steps `blocks.0.qkv` shares 0.38 of its step-1 frame (G path without Molegrad: 0.50, loss 2.0544).

## adaLN at init (2026-10-31, `scripts/probe_init.py`)

Depth 12, seed 0, 16 timesteps spread over (0, 1), text = the null embedding or random unit vectors (the two agree within 5%). `c` per-element std 1.07. Block 0 shift/scale rows (shared projection + table): std 0.28 to 0.30. Final layer shift/scale (`final_table + c`): std 1.07, max |scale| 4.0 to 4.1.
