"""PLAN I1 (REPA, topics/priming.md): whether `VisionTarget`'s per-token features of 128 px decodes carry the token's
content, on CPU. Each latent and its horizontal flip (Primal decodes flips equivariantly, FACTS) go through the target;
the flip's token grid flipped back should match the original token for token. Reported: the cosine of each token to
its own flipped-back counterpart, against that to the other tokens of the same image and to the same token of other
images. Usage: probe_repa_target.py [N]
"""

import json
import sys

import pyarrow.parquet as pq
import torch
import torch.nn.functional as F

from nanodit.align import VisionTarget
from nanodit.data import _bf16
from nanodit.model import LATENT_CHANNELS
from train import load_decoder

EVAL = "/run/media/djg/vel-ssd/tiny-dit/lunara_eval_128_seq512.parquet"
N = int(sys.argv[1]) if len(sys.argv) > 1 else 32
torch.set_grad_enabled(False)

x = _bf16(pq.read_table(EVAL, columns=["latent_128"]).slice(0, N)["latent_128"], LATENT_CHANNELS, 64, 64).float()
vision = VisionTarget(load_decoder(torch.device("cpu")), torch.device("cpu"))
features = torch.cat([vision(x[i : i + 8]) for i in range(0, N, 8)]).float()  # [N, 64, 768]
flipped = torch.cat([vision(x[i : i + 8].flip(-1)) for i in range(0, N, 8)]).float().view(N, 8, 8, -1).flip(2).view(N, 64, -1)

cos = F.cosine_similarity(features[:, :, None], flipped[:, None], dim=-1)  # [N, 64 own tokens, 64 flipped-back tokens]
own = cos.diagonal(dim1=1, dim2=2)
others_in_image = (cos.sum((1, 2)) - own.sum(1)) / (64 * 63)
across = F.cosine_similarity(features[:, None], flipped[None], dim=-1)  # [N, N, 64]: same token, image against image
other_images = (across.sum((0, 1)) - across.diagonal().sum(-1)) / (N * (N - 1))
print(json.dumps({
    "rows": N, "feature_norm": features.norm(dim=-1).mean().item(),
    "own_token": own.mean().item(), "other_tokens_same_image": others_in_image.mean().item(),
    "same_token_other_images": other_images.mean().item(),
    "own_token_is_best_match": (cos.argmax(-1) == torch.arange(64)).float().mean().item(),
}))
