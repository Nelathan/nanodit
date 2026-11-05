"""The DiT of SPEC §2, §3 and §5."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PATCH = 16
LATENT_CHANNELS = 2
TOKEN_DIM = LATENT_CHANNELS * PATCH * PATCH  # 512
HEAD_DIM = 64
TEXT_DIM = 768
TIME_DIM = 256
ROPE_BASE = 10000.0
ADALN_STD = 0.02  # "small random" shift/scale rows
N_MOD = 6  # shift, scale, gate for attention, then for the MLP
GATE_ROWS = (2, 5)
MLP_RATIO = 4  # SwiGLU hidden over the stream width


def patchify(x: Tensor) -> tuple[Tensor, int, int]:
    """[B, 2, H, W] latent -> [B, h*w, 512] row-major tokens and the grid (h, w)."""
    tokens = F.pixel_unshuffle(x, PATCH)
    return tokens.flatten(2).transpose(1, 2), tokens.shape[2], tokens.shape[3]


def unpatchify(tokens: Tensor, h: int, w: int) -> Tensor:
    return F.pixel_shuffle(tokens.transpose(1, 2).reshape(-1, TOKEN_DIM, h, w), PATCH)


def timestep_sinusoid(t: Tensor) -> Tensor:
    freqs = torch.exp(-math.log(10000.0) * torch.arange(TIME_DIM // 2, device=t.device) / (TIME_DIM // 2))
    args = 1000.0 * t[:, None] * freqs  # t in [0, 1]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def rope_angles(h: int, w: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """cos, sin of shape [h*w, HEAD_DIM/2]: the first half of the angles encodes the row, the second the column."""
    pairs = HEAD_DIM // 4
    freqs = ROPE_BASE ** (-torch.arange(pairs, device=device) / pairs)
    rows = torch.arange(h, device=device)[:, None].expand(h, w).reshape(-1, 1)
    cols = torch.arange(w, device=device)[None, :].expand(h, w).reshape(-1, 1)
    angles = torch.cat([rows * freqs, cols * freqs], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate every dim of each head (split-half pairs)."""
    x1, x2 = x.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class Block(nn.Module):
    def __init__(self, width: int):
        super().__init__()
        self.width, self.heads, hidden = width, width // HEAD_DIM, MLP_RATIO * width
        self.table = nn.Parameter(torch.zeros(N_MOD, width))
        self.qkv = nn.Linear(width, 3 * width, bias=False)
        self.gate = nn.Linear(width, width, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM)
        self.k_norm = nn.RMSNorm(HEAD_DIM)
        self.out = nn.Linear(width, width, bias=False)
        self.attn_norm = nn.RMSNorm(width)
        self.mlp_norm = nn.RMSNorm(width)
        self.w_gate = nn.Linear(width, hidden, bias=False)
        self.w_up = nn.Linear(width, hidden, bias=False)
        self.w_down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor]) -> Tensor:
        B, N, _ = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (mod + self.table).unbind(1)

        y = F.rms_norm(x, (self.width,)) * (1 + scale_a[:, None]) + shift_a[:, None]
        q, k, v = self.qkv(y).view(B, N, 3, self.heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, self.width)
        x = x + gate_a[:, None] * self.attn_norm(self.out(attn * torch.sigmoid(self.gate(y))))

        y = F.rms_norm(x, (self.width,)) * (1 + scale_m[:, None]) + shift_m[:, None]
        return x + gate_m[:, None] * self.mlp_norm(self.w_down(F.silu(self.w_gate(y)) * self.w_up(y)))


class DiT(nn.Module):
    def __init__(self, null_embedding: Tensor, depth: int, width: int):
        super().__init__()
        self.width = width
        self.null_embedding = nn.Parameter(null_embedding.detach().clone().float())
        self.patch_embed = nn.Linear(TOKEN_DIM, width - TOKEN_DIM)  # writes the meta channels only
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.text_proj = nn.Linear(TEXT_DIM, width)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, N_MOD * width))
        self.blocks = nn.ModuleList(Block(width) for _ in range(depth))
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(0, m.in_features**-0.5)
                if m.bias is not None:
                    m.bias.zero_()
        mod = self.ada[1]
        mod.weight.view(N_MOD, self.width, self.width)[:] = ADALN_STD * torch.randn(N_MOD, self.width, self.width)
        mod.weight.view(N_MOD, self.width, self.width)[list(GATE_ROWS)] = 0
        for table in [b.table for b in self.blocks]:
            table.normal_(0, ADALN_STD)
            table[list(GATE_ROWS)] = 0

    def forward(self, x: Tensor, t: Tensor, text: Tensor) -> Tensor:
        """x [B, 2, H, W] latent at noise level t [B], text [B, 768] -> F [B, 2, H, W].
        The network runs in the parameters' dtype (bf16 in training); x and text are cast on the way in
        and F comes back fp32, so the objective and the sampler never see the network's dtype. t stays
        fp32 up to the sinusoid, whose phases need the precision."""
        dtype = self.patch_embed.weight.dtype
        tokens, h, w = patchify(x.to(dtype))
        c = self.time_mlp(timestep_sinusoid(t).to(dtype)) + self.text_proj(F.rms_norm(text, (TEXT_DIM,)).to(dtype))
        mod = self.ada(c).view(-1, N_MOD, self.width)
        rope = rope_angles(h, w, x.device)

        tokens = torch.cat([tokens, self.patch_embed(tokens)], dim=-1)  # the first 512 channels carry the patch unchanged
        for block in self.blocks:
            tokens = block(tokens, mod, rope)
        return unpatchify(tokens[..., :TOKEN_DIM], h, w).float()
