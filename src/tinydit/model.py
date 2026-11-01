"""The DiT of SPEC §2, §3 and §5."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PATCH = 8
LATENT_CHANNELS = 2
TOKEN_DIM = LATENT_CHANNELS * PATCH * PATCH  # 128
DIM = 512
HEADS = 8
HEAD_DIM = DIM // HEADS
HIDDEN = 1024
TEXT_DIM = 768
TIME_DIM = 256
ROPE_DIM = HEAD_DIM // 2  # rotated dims per head; the rest is position-free
ROPE_BASE = 10000.0
ADALN_STD = 0.02  # "small random" shift/scale rows
N_MOD = 6  # shift, scale, gate for attention, then for the MLP
GATE_ROWS = (2, 5)
N_FINAL = 2  # shift, scale for the final norm, after the blocks' rows in the shared projection


def patchify(x: Tensor) -> tuple[Tensor, int, int]:
    """[B, 2, H, W] latent -> [B, h*w, 128] row-major tokens and the grid (h, w)."""
    tokens = F.pixel_unshuffle(x, PATCH)
    return tokens.flatten(2).transpose(1, 2), tokens.shape[2], tokens.shape[3]


def unpatchify(tokens: Tensor, h: int, w: int) -> Tensor:
    return F.pixel_shuffle(tokens.transpose(1, 2).reshape(-1, TOKEN_DIM, h, w), PATCH)


def timestep_sinusoid(t: Tensor) -> Tensor:
    freqs = torch.exp(-math.log(10000.0) * torch.arange(TIME_DIM // 2, device=t.device) / (TIME_DIM // 2))
    args = 1000.0 * t[:, None] * freqs  # t in [0, 1]
    return torch.cat([args.cos(), args.sin()], dim=-1)


def rope_angles(h: int, w: int, device: torch.device | None = None) -> tuple[Tensor, Tensor]:
    """cos, sin of shape [h*w, ROPE_DIM/2]: the first half of the angles encodes the row, the second the column."""
    pairs = ROPE_DIM // 4
    freqs = ROPE_BASE ** (-torch.arange(pairs, device=device) / pairs)
    rows = torch.arange(h, device=device)[:, None].expand(h, w).reshape(-1, 1)
    cols = torch.arange(w, device=device)[None, :].expand(h, w).reshape(-1, 1)
    angles = torch.cat([rows * freqs, cols * freqs], dim=-1)
    return angles.cos(), angles.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate the first ROPE_DIM dims of each head (split-half pairs); the other dims pass through."""
    rotated, free = x[..., :ROPE_DIM], x[..., ROPE_DIM:]
    x1, x2 = rotated.chunk(2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin, free], dim=-1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.table = nn.Parameter(torch.zeros(N_MOD, DIM))
        self.conv = nn.Conv2d(DIM, DIM, 3, padding=1, groups=DIM, bias=False)
        self.qkv = nn.Linear(DIM, 3 * DIM, bias=False)
        self.gate = nn.Linear(DIM, DIM, bias=False)
        self.q_norm = nn.RMSNorm(HEAD_DIM)
        self.k_norm = nn.RMSNorm(HEAD_DIM)
        self.out = nn.Linear(DIM, DIM, bias=False)
        self.w_gate = nn.Linear(DIM, HIDDEN, bias=False)
        self.w_up = nn.Linear(DIM, HIDDEN, bias=False)
        self.w_down = nn.Linear(HIDDEN, DIM, bias=False)

    def forward(self, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor], h: int, w: int) -> Tensor:
        B, N, _ = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (mod + self.table).unbind(1)

        y = F.rms_norm(x, (DIM,)) * (1 + scale_a[:, None]) + shift_a[:, None]
        grid = y.transpose(1, 2).reshape(B, DIM, h, w)
        y = y + self.conv(grid).flatten(2).transpose(1, 2)
        q, k, v = self.qkv(y).view(B, N, 3, HEADS, HEAD_DIM).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
        attn = F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, DIM)
        x = x + gate_a[:, None] * self.out(attn * torch.sigmoid(self.gate(y)))

        y = F.rms_norm(x, (DIM,)) * (1 + scale_m[:, None]) + shift_m[:, None]
        return x + gate_m[:, None] * self.w_down(F.silu(self.w_gate(y)) * self.w_up(y))


class DiT(nn.Module):
    def __init__(self, null_embedding: Tensor, depth: int = 12):
        super().__init__()
        self.null_embedding = nn.Parameter(null_embedding.detach().clone().float())
        self.patch_embed = nn.Linear(TOKEN_DIM, DIM)
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, DIM), nn.SiLU(), nn.Linear(DIM, DIM))
        self.text_proj = nn.Linear(TEXT_DIM, DIM)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(DIM, (N_MOD + N_FINAL) * DIM))
        self.blocks = nn.ModuleList(Block() for _ in range(depth))
        self.final_table = nn.Parameter(torch.zeros(N_FINAL, DIM))
        self.final_norm = nn.RMSNorm(DIM)
        self.final = nn.Linear(DIM, TOKEN_DIM)
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(0, m.in_features**-0.5)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Conv2d):
                m.weight.zero_()
        mod = self.ada[1]
        mod.weight.view(N_MOD + N_FINAL, DIM, DIM)[:] = ADALN_STD * torch.randn(N_MOD + N_FINAL, DIM, DIM)
        mod.weight.view(N_MOD + N_FINAL, DIM, DIM)[list(GATE_ROWS)] = 0
        for table in [b.table for b in self.blocks]:
            table.normal_(0, ADALN_STD)
            table[list(GATE_ROWS)] = 0
        self.final_table.normal_(0, ADALN_STD)
        self.final.weight.zero_()

    def forward(self, x: Tensor, t: Tensor, text: Tensor) -> Tensor:
        """x [B, 2, H, W] latent at noise level t [B], text [B, 768] -> F [B, 2, H, W].
        The network runs in the parameters' dtype (bf16 in training); x and text are cast on the way in
        and F comes back fp32, so the objective and the sampler never see the network's dtype. t stays
        fp32 up to the sinusoid, whose phases need the precision."""
        dtype = self.patch_embed.weight.dtype
        tokens, h, w = patchify(x.to(dtype))
        c = self.time_mlp(timestep_sinusoid(t).to(dtype)) + self.text_proj(F.rms_norm(text, (TEXT_DIM,)).to(dtype))
        mod = self.ada(c).view(-1, N_MOD + N_FINAL, DIM)
        rope = rope_angles(h, w, x.device)

        tokens = self.patch_embed(tokens)
        for block in self.blocks:
            tokens = block(tokens, mod[:, :N_MOD], rope, h, w)

        shift, scale = (mod[:, N_MOD:] + self.final_table).unbind(1)
        tokens = self.final_norm(tokens) * (1 + scale[:, None]) + shift[:, None]
        return unpatchify(self.final(tokens), h, w).float()
