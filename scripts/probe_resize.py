"""PLAN B5: which downsampler for 128 px (and 256 px): spectrum kept and Primal round trip, on CPU.

Streams the first 256 square 512 px jpgs of the first text-to-image-2M shard (no download of the 7.5 GB tar),
resizes them to 128 px six ways, compares each with a brick-wall FFT crop of the source, and round-trips the
first 64 through Primal 2x2.

usage: probe_resize.py
"""
import io
import tarfile

import numpy as np
import requests
import torch
import torch.nn.functional as F
from diffusers import AutoModel
from PIL import Image

URL = "https://huggingface.co/datasets/jackyhate/text-to-image-2M/resolve/main/data_512_2M/data_000000.tar"
N, R = 256, 128

torch.manual_seed(0)
src = []
with requests.get(URL, stream=True, timeout=60) as r:
    r.raise_for_status()
    with tarfile.open(fileobj=r.raw, mode="r|") as tar:
        for m in tar:
            if m.name.endswith(".jpg"):
                im = Image.open(io.BytesIO(tar.extractfile(m).read())).convert("RGB")
                if im.size == (512, 512):
                    src.append(np.array(im))
                if len(src) == N:
                    break
src = np.stack(src)  # [N,512,512,3] uint8

def pil(m):  return lambda a: np.array(Image.fromarray(a).resize((R,R), m)).astype(np.float32)/127.5-1
def tor(mode,aa=True):
    def f(a):
        x=torch.from_numpy(a).permute(2,0,1)[None].float()/127.5-1
        return F.interpolate(x,size=(R,R),mode=mode,**({"antialias":True} if aa else {})).clamp(-1,1)[0].permute(1,2,0).numpy()
    return f
methods={"torch bicubic aa (current)":tor("bicubic"),"PIL lanczos":pil(Image.LANCZOS),"PIL box (area)":pil(Image.BOX),"PIL bicubic":pil(Image.BICUBIC),"torch bilinear aa":tor("bilinear"),"torch bicubic NO aa":tor("bicubic",False)}
out={k:np.stack([f(a) for a in src]) for k,f in methods.items()}      # [N,128,128,3]
# brickwall reference: FFT crop of the source to the 128 Nyquist
x=torch.from_numpy(src).permute(0,3,1,2).float()/127.5-1
X=torch.fft.fftshift(torch.fft.fft2(x),dim=(-2,-1))
c=(512-R)//2
ref=torch.fft.ifft2(torch.fft.ifftshift(X[...,c:c+R,c:c+R],dim=(-2,-1))).real*(R*R)/(512*512)
ref=ref.permute(0,2,3,1).numpy()
def psnr(a,b): return (10*np.log10(4/((a-b)**2).mean((1,2,3)))).mean()
def radial(a):   # mean power vs radial frequency (cycles/px of the 128 grid), 4 bands
    t=torch.from_numpy(a).permute(0,3,1,2); P=(torch.fft.fftshift(torch.fft.fft2(t),dim=(-2,-1)).abs()**2).mean((0,1))
    yy,xx=torch.meshgrid(torch.arange(R)-R//2,torch.arange(R)-R//2,indexing="ij"); r=(yy**2+xx**2).float().sqrt()/R
    return [P[(r>=lo)&(r<hi)].sum().item() for lo,hi in [(0,.125),(.125,.25),(.25,.375),(.375,.5)]]
pr=radial(ref)
print(f"{'method':28s} PSNR vs brickwall | band energy vs brickwall (0-.125 .125-.25 .25-.375 .375-.5 cyc/px)")
for k,a in out.items():
    pw=radial(a); print(f"{k:28s} {psnr(a,ref):6.2f} dB | "+"  ".join(f"{p/q:5.2f}" for p,q in zip(pw,pr)))
# VAE round trip
vae = AutoModel.from_pretrained("ostris/primal_vae",variant="2x2",trust_remote_code=True,torch_dtype=torch.float32).eval()
n=64
print("\nPrimal round trip on first",n,"images (PSNR of decode(encode(x)) vs x; lower-is-harder content):")
with torch.no_grad():
    for k,a in out.items():
        t=torch.from_numpy(a[:n]).permute(0,3,1,2).contiguous()
        z=torch.cat([vae.encode(t[i:i+16]) for i in range(0,n,16)]); xh=torch.cat([vae.decode(z[i:i+16]).clamp(-1,1) for i in range(0,n,16)])
        e=(xh-t)**2
        hp=lambda u: u-F.avg_pool2d(F.pad(u,(2,2,2,2),mode="reflect"),5,1)   # high-pass: detail the VAE must carry
        hf=(hp(xh)-hp(t))**2; hfe=(hp(t)**2).mean().item()
        print(f"{k:28s} PSNR {(10*torch.log10(4/e.mean((1,2,3)))).mean():6.2f} dB | detail energy {hfe:.5f} | detail recon err/energy {hf.mean().item()/hfe:.3f} | latent std {z.std().item():.3f}")
