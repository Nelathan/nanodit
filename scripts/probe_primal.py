"""PLAN B1: Primal VAE 2x2 latent statistics, flip equivariance and reconstruction, on CPU.

usage: probe_primal.py SHARD.tar OUT_DIR
"""
import io
import json
import sys
import tarfile
import time

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoModel
from PIL import Image

OUT = sys.argv[2]
SIZES = {128: 256, 256: 128}  # resolution -> number of images

torch.manual_seed(0)
shard = sys.argv[1]
n_max = max(SIZES.values())

images = []
with tarfile.open(shard, "r|") as tar:
    for m in tar:
        if m.name.endswith(".jpg"):
            images.append(np.array(Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")))
            if len(images) == n_max:
                break
src = torch.from_numpy(np.stack(images)).permute(0, 3, 1, 2).float() / 127.5 - 1  # [N,3,512,512] in [-1,1]
print("source images", tuple(src.shape))

vae = AutoModel.from_pretrained("ostris/primal_vae", variant="2x2", trust_remote_code=True, torch_dtype=torch.float32)
print("params", sum(p.numel() for p in vae.parameters()), "dtype", next(vae.parameters()).dtype)


def kurtosis(z):  # excess kurtosis over the last dim
    z = z - z.mean(-1, keepdim=True)
    return (z**4).mean(-1) / (z**2).mean(-1) ** 2 - 3


def dist(x):  # min / median / max over dim 0, per channel
    return {"min": x.min(0).values.tolist(), "median": x.median(0).values.tolist(), "max": x.max(0).values.tolist()}


def psnr(a, b):  # images in [-1,1], clamped to [0,1] range
    mse = ((a.clamp(-1, 1) - b) / 2).pow(2).mean((1, 2, 3))
    return -10 * torch.log10(mse)


def jpeg_q95(t):  # [N,3,H,W] float 0..255 -> same after a JPEG q95 round trip
    out = []
    for a in t:
        buf = io.BytesIO()
        Image.fromarray(a.permute(1, 2, 0).byte().numpy()).save(buf, "JPEG", quality=95)
        out.append(torch.from_numpy(np.array(Image.open(buf))).permute(2, 0, 1).float())
    return torch.stack(out)


def batched(fn, x, bs=16):
    return torch.cat([fn(x[i : i + bs]) for i in range(0, len(x), bs)])


results = {"resize": "torch F.interpolate bicubic, antialias=True, from the 512 crop"}
grid_rows = []
with torch.no_grad():
    for res, n in SIZES.items():
        x = F.interpolate(src[:n], size=(res, res), mode="bicubic", antialias=True).clamp(-1, 1)
        t0 = time.time()
        z = batched(vae.encode, x)
        enc_s = time.time() - t0
        t0 = time.time()
        xh = batched(vae.decode, z)
        dec_s = time.time() - t0

        flat = z.flatten(2)  # [N,2,HW]
        mean, std = flat.mean(-1), flat.std(-1, unbiased=False)
        kurt = kurtosis(flat)
        rel = lambda a, b: ((a - b).flatten(1).norm(dim=1) / b.flatten(1).norm(dim=1))
        # Spatial ops applied to the image and to the latent. For each, compare enc(op(img)) with M @ op(enc(img))
        # for the 8 signed channel maps M of 2 channels, and the least-squares 2x2 M.
        ops = {"hflip": lambda t: t.flip(-1), "vflip": lambda t: t.flip(-2), "rot90": lambda t: torch.rot90(t, 1, (-2, -1))}
        maps = {f"{'+' if s0 > 0 else '-'}{'+' if s1 > 0 else '-'}{'swap' if sw else 'id'}": torch.tensor([[0.0, s0], [s1, 0.0]] if sw else [[s0, 0.0], [0.0, s1]])
                for sw in (0, 1) for s0 in (1, -1) for s1 in (1, -1)}  # name: signs of the two output rows
        cm = lambda M, t: torch.einsum("ij,njhw->nihw", M, t)
        equiv = {}
        for name, op in ops.items():
            ze, zo = batched(vae.encode, op(x)), op(z)  # target, spatially transformed latent
            table = {m: {"rel_err": rel(cm(M, zo), ze).mean().item(), "cosine": F.cosine_similarity(cm(M, zo).flatten(1), ze.flatten(1)).mean().item()} for m, M in maps.items()}
            A, B = zo.transpose(0, 1).flatten(1), ze.transpose(0, 1).flatten(1)  # [2, N*H*W]
            Mfit = torch.linalg.lstsq(A.T, B.T).solution.T
            best = min(table, key=lambda m: table[m]["rel_err"])
            equiv[name] = {
                "signed_maps": table,
                "best_signed_map": best,
                "lstsq_matrix": Mfit.tolist(),
                "lstsq_rel_err": rel(cm(Mfit, zo), ze).mean().item(),
                "lstsq_cosine": F.cosine_similarity(cm(Mfit, zo).flatten(1), ze.flatten(1)).mean().item(),
                "decode_best_map_of_op_latent_vs_op_img_psnr_db": psnr(batched(vae.decode, cm(maps[best], zo)), op(x)).mean().item(),
                "decode_unmapped_op_latent_vs_op_img_psnr_db": psnr(batched(vae.decode, zo), op(x)).mean().item(),
                "baseline_rel_err_z_vs_op_z_no_map": rel(z, zo).mean().item(),  # ||z - op(z)|| / ||op(z)||, same image, same latent
            }
        # Stability: enc(img) against enc(perturbed img). Noise is added to the float image (pixel std k/255 on the
        # 0..255 scale, i.e. 2k/255 on [-1,1]). JPEG: q95 re-encode of the 8-bit rounded image, so "quant8" is its control.
        u8 = ((x + 1) * 127.5).round().clamp(0, 255)
        to_pm1 = lambda t: t / 127.5 - 1
        compare = lambda a, b: {"rel_err": rel(a, b).mean().item(), "cosine": F.cosine_similarity(a.flatten(1), b.flatten(1)).mean().item()}
        stab = {
            "noise_1_255": compare(batched(vae.encode, x + torch.randn_like(x) * 2 / 255), z),
            "noise_4_255": compare(batched(vae.encode, x + torch.randn_like(x) * 8 / 255), z),
            "quant8_control": compare(batched(vae.encode, to_pm1(u8)), z),
            "jpeg_q95": compare(batched(vae.encode, to_pm1(jpeg_q95(u8))), z),
            "shift_2px_vs_latent_roll_1": compare(batched(vae.encode, x.roll(2, -1)), z.roll(1, -1)),
            "shift_2px_vs_latent_roll_1_interior": compare(batched(vae.encode, x.roll(2, -1))[..., 8:-8], z.roll(1, -1)[..., 8:-8]),
        }
        # Frequency split: low-pass = avg-pool by k then nearest upsample; high = remainder. Cosine per image, mean.
        def bands(a, b):
            out = {}
            for k in (2, 4, 8):
                lp = lambda t: F.interpolate(F.avg_pool2d(t, k), scale_factor=k, mode="nearest")
                out[f"pool{k}"] = {"low_cosine": F.cosine_similarity(lp(a).flatten(1), lp(b).flatten(1)).mean().item(),
                                   "high_cosine": F.cosine_similarity((a - lp(a)).flatten(1), (b - lp(b)).flatten(1)).mean().item(),
                                   "low_energy_fraction": (lp(a).pow(2).sum() / a.pow(2).sum()).item()}
            return out
        zn = batched(vae.encode, x + torch.randn_like(x) * 2 / 255)
        stab["bands_hflip_enc_flip_img_vs_flip_enc"] = bands(batched(vae.encode, x.flip(-1)), z.flip(-1))
        stab["bands_noise_1_255"] = bands(zn, z)
        # Determinism: same batch twice, and batch of 16 against single images
        stab["same_batch_twice_max_abs_diff"] = (batched(vae.encode, x[:16]) - batched(vae.encode, x[:16])).abs().max().item()
        stab["batch16_vs_single_max_abs_diff"] = (torch.cat([vae.encode(x[i : i + 1]) for i in range(16)]) - z[:16]).abs().max().item()
        r = {
            "stability": stab,
            "latent_shape": list(z.shape),
            "n_images": n,
            "per_image_mean": dist(mean),
            "per_image_std": dist(std),
            "global_mean": z.mean((0, 2, 3)).tolist(),
            "global_std": z.std((0, 2, 3)).tolist(),
            "per_image_excess_kurtosis_median": kurt.median(0).values.tolist(),
            "pooled_excess_kurtosis": kurtosis(((z - mean[..., None, None]) / std[..., None, None]).transpose(0, 1).flatten(1)).tolist(),
            "pooled_abs_gt3_fraction": ((z - mean[..., None, None]) / std[..., None, None]).abs().gt(3).float().mean((0, 2, 3)).tolist(),
            "equivariance": equiv,
            "recon_psnr_db": {"mean": psnr(xh, x).mean().item(), "min": psnr(xh, x).min().item(), "max": psnr(xh, x).max().item()},
            "encode_img_per_s": n / enc_s,
            "decode_img_per_s": n / dec_s,
        }
        results[str(res)] = r
        print(f"\n== {res}px ==")
        print(json.dumps(r, indent=1))
        # grid row: 4 images, input over reconstruction, each shown at 256 px
        show = lambda t: F.interpolate(t[:4], size=(256, 256), mode="nearest")
        grid_rows.append(torch.cat([show(x), show(xh.clamp(-1, 1))], dim=2))  # [4,3,512,256]

grid = torch.cat([torch.cat(list(row), dim=2) for row in grid_rows], dim=1)  # rows: 128 in/recon, 256 in/recon
Image.fromarray(((grid.permute(1, 2, 0) + 1) * 127.5).round().byte().numpy()).save(f"{OUT}/primal_recon.png")
json.dump(results, open(f"{OUT}/primal.json", "w"), indent=1)
