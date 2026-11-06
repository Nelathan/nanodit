"""The DiT of SPEC §2, §3 and §5."""

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PATCH = 8
LATENT_CHANNELS = 2
TOKEN_DIM = LATENT_CHANNELS * PATCH * PATCH  # 128
HEAD_DIM = 64
TEXT_DIM = 768
TIME_DIM = 256
ROPE_BASE = 10000.0
ADALN_STD = 0.02  # "small random" shift/scale rows
N_MOD = 6  # shift, scale, gate for attention, then for the MLP
GATE_ROWS = (2, 5)
MLP_RATIO = 4  # SwiGLU hidden over the stream width


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


def rope_angles(h: int, w: int, device: torch.device | None = None, fraction: float = 1.0) -> tuple[Tensor, Tensor]:
    """cos, sin of shape [h*w, HEAD_DIM/2]: the first half of the angles encodes the row, the second the column.
    Only the first `fraction` of each axis' frequencies rotate; the rest get angle 0, so those dims carry no position."""
    pairs = HEAD_DIM // 4
    freqs = ROPE_BASE ** (-torch.arange(pairs, device=device) / pairs)
    freqs = freqs * (torch.arange(pairs, device=device) < fraction * pairs)
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
    """A conv block mixes tokens with a depthwise 3x3 conv over the token grid and writes only the meta
    channels (the ones past TOKEN_DIM); its MLP then decides what reaches the image channels. Otherwise
    the token mixer is gated attention."""

    def __init__(self, width: int, conv: bool, kv_heads: int, parallel_mlp: bool):
        super().__init__()
        self.width, self.heads, self.kv_heads, hidden = width, width // HEAD_DIM, kv_heads, MLP_RATIO * width
        self.parallel_mlp = parallel_mlp
        self.table = nn.Parameter(torch.zeros(N_MOD, width))
        self.is_conv = conv
        if conv:
            self.conv = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
            self.out = nn.Linear(width, width - TOKEN_DIM, bias=False)
            self.conv_norm = nn.RMSNorm(width - TOKEN_DIM)
        else:
            self.qkv = nn.Linear(width, width + 2 * kv_heads * HEAD_DIM, bias=False)
            self.gate = nn.Linear(width, width, bias=False)
            self.q_norm = nn.RMSNorm(HEAD_DIM)
            self.k_norm = nn.RMSNorm(HEAD_DIM)
            self.out = nn.Linear(width, width, bias=False)
            self.attn_norm = nn.RMSNorm(width)
        self.mlp_norm = nn.RMSNorm(width)
        self.w_gate = nn.Linear(width, hidden, bias=False)
        self.w_up = nn.Linear(width, hidden, bias=False)
        self.w_down = nn.Linear(hidden, width, bias=False)

    def forward(self, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor], h: int, w: int) -> Tensor:
        B, N, _ = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (mod + self.table).unbind(1)

        y = F.rms_norm(x, (self.width,)) * (1 + scale_a[:, None]) + shift_a[:, None]
        if self.is_conv:
            mixed = self.conv(y.transpose(1, 2).reshape(B, self.width, h, w)).flatten(2).transpose(1, 2)
            update = gate_a[:, None] * F.pad(self.conv_norm(self.out(mixed)), (TOKEN_DIM, 0))  # zeros on the image channels
        else:
            q, k, v = self.qkv(y).split([self.width, self.kv_heads * HEAD_DIM, self.kv_heads * HEAD_DIM], dim=-1)
            q = q.view(B, N, self.heads, HEAD_DIM).transpose(1, 2)
            k, v = (a.view(B, N, self.kv_heads, HEAD_DIM).transpose(1, 2) for a in (k, v))
            q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
            attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.kv_heads != self.heads).transpose(1, 2).reshape(B, N, self.width)
            update = gate_a[:, None] * self.attn_norm(self.out(attn * torch.sigmoid(self.gate(y))))
        if self.parallel_mlp:  # PaLM: the MLP reads the same modulated input as the mixer, both add to the stream together
            return x + update + gate_m[:, None] * self.mlp_norm(self.w_down(F.silu(self.w_gate(y)) * self.w_up(y)))
        x = x + update
        y = F.rms_norm(x, (self.width,)) * (1 + scale_m[:, None]) + shift_m[:, None]
        return x + gate_m[:, None] * self.mlp_norm(self.w_down(F.silu(self.w_gate(y)) * self.w_up(y)))


class DiT(nn.Module):
    def __init__(self, null_embedding: Tensor, depth: int, width: int, conv_period: int = 2, meta_embed: bool = False,
                 parallel_mlp: bool = False, rope_fraction: float = 1.0, kv_heads: int | None = None):
        super().__init__()
        assert conv_period >= 1, conv_period
        self.width = width
        self.null_embedding = nn.Parameter(null_embedding.detach().clone().float())
        self.meta_embed, self.rope_fraction = meta_embed, rope_fraction
        self.patch_embed = nn.Linear(TOKEN_DIM, width - TOKEN_DIM if meta_embed else width)  # meta_embed: writes the meta channels only
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.text_proj = nn.Linear(TEXT_DIM, width)
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, N_MOD * width))
        self.blocks = nn.ModuleList(Block(width, i % conv_period != conv_period - 1, kv_heads or width // HEAD_DIM, parallel_mlp) for i in range(depth))  # conv first, attention next: every conv_period-th block is attention
        self.init_weights()

    @torch.no_grad()
    def init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                m.weight.normal_(0, m.in_features**-0.5)
                if m.bias is not None:
                    m.bias.zero_()
            elif isinstance(m, nn.Conv2d):
                m.weight.normal_(0, (m.in_channels // m.groups * m.kernel_size[0] * m.kernel_size[1]) ** -0.5)
        mod = self.ada[1]
        mod.weight.view(N_MOD, self.width, self.width)[:] = ADALN_STD * torch.randn(N_MOD, self.width, self.width)
        mod.weight.view(N_MOD, self.width, self.width)[list(GATE_ROWS)] = 0
        for table in [b.table for b in self.blocks]:
            table.normal_(0, ADALN_STD)
            table[list(GATE_ROWS)] = 0
        if not self.meta_embed:
            self.patch_embed.weight[:TOKEN_DIM] = torch.eye(TOKEN_DIM)  # the first 128 channels of the stream start as the input patch

    def forward(self, x: Tensor, t: Tensor, text: Tensor) -> Tensor:
        """x [B, 2, H, W] latent at noise level t [B], text [B, 768] -> F [B, 2, H, W].
        The network runs in the parameters' dtype (bf16 in training); x and text are cast on the way in
        and F comes back fp32, so the objective and the sampler never see the network's dtype. t stays
        fp32 up to the sinusoid, whose phases need the precision."""
        dtype = self.patch_embed.weight.dtype
        tokens, h, w = patchify(x.to(dtype))
        c = self.time_mlp(timestep_sinusoid(t).to(dtype)) + self.text_proj(F.rms_norm(text, (TEXT_DIM,)).to(dtype))
        mod = self.ada(c).view(-1, N_MOD, self.width)
        rope = rope_angles(h, w, x.device, self.rope_fraction)

        tokens = torch.cat([tokens, self.patch_embed(tokens)], dim=-1) if self.meta_embed else self.patch_embed(tokens)
        for block in self.blocks:
            tokens = block(tokens, mod, rope, h, w)
        return unpatchify(tokens[..., :TOKEN_DIM], h, w).float()
