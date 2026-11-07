"""The DiT of SPEC §2, §3 and §5."""

import math
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

PATCH = 8
LATENT_CHANNELS = 2
TOKEN_DIM = LATENT_CHANNELS * PATCH * PATCH  # 128
HEAD_DIM = 64
TEXT_DIM = 768
SEQ_DIM = 512  # the encoder's token states, before its projection to 768
NULL_TOKENS = 2  # the empty caption's tokens after the prompt prefix: a space and <eos>
TIME_DIM = 256
ROPE_BASE = 10000.0
ADALN_STD = 0.02  # "small random" shift/scale rows
N_MOD = 6  # shift, scale, gate for the token mixer (attention or conv), then for the MLP
GATE_ROWS = (2, 5)
MLP_RATIO = 4  # SwiGLU hidden over the stream width
KV_RATIO = 4  # query heads per key/value head
ROPE_FRACTION = 0.25  # of each axis' frequencies rotate: the fast ones; the slow ones barely turn on the grid and carry no position
CONV_PERIOD = 4  # every CONV_PERIOD-th block is attention, the rest are conv blocks


class Prompt(NamedTuple):
    """The text side of a batch: the pooled embedding conditions the adaLN, the token states feed the cross attention."""

    pooled: Tensor  # [B, 768]
    tokens: Tensor  # [B, L, 512], zero past a caption's end, L >= NULL_TOKENS
    mask: Tensor  # [B, L] bool, true on a caption's tokens

    def to(self, device: torch.device, dtype: torch.dtype) -> "Prompt":
        return Prompt(self.pooled.to(device, dtype), self.tokens.to(device, dtype), self.mask.to(device))

    def rows(self, start: int, stop: int) -> "Prompt":
        return Prompt(self.pooled[start:stop], self.tokens[start:stop], self.mask[start:stop])


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


def rope_angles(h: int, w: int, device: torch.device | None = None, fraction: float = ROPE_FRACTION) -> tuple[Tensor, Tensor]:
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
    """A conv block mixes tokens over their 3x3 neighbourhood on the token grid. With `gdn` it replaces each token's
    state by a gated update (GDN2's decoupled forget and write gates, diagonal) and writes every channel; without, a
    depthwise 3x3 conv writes only the meta channels (the ones past TOKEN_DIM), and its MLP then decides what reaches
    the image channels. An attention
    block mixes with gated grouped-query attention and, with `cross`, then reads the caption's token states
    through a cross attention that takes no adaLN, only a zero-initialised gate per channel."""

    def __init__(self, width: int, conv: bool, cross: bool, gdn: bool):
        super().__init__()
        self.width, self.heads, hidden = width, width // HEAD_DIM, MLP_RATIO * width
        self.kv_heads = max(1, self.heads // KV_RATIO)
        assert self.heads % self.kv_heads == 0
        self.is_conv, self.cross, self.gdn = conv, cross and not conv, gdn and conv
        self.table = nn.Parameter(torch.zeros(N_MOD, width))
        if conv:
            if self.gdn:
                self.gdn_proj = nn.Linear(width, 4 * width, bias=False)  # value, current state, forget gate, write gate
            self.conv = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)  # on the value with gdn, else on the input
            self.written = width if self.gdn else width - TOKEN_DIM  # the last `written` channels
            self.out = nn.Linear(width, self.written, bias=False)
            self.conv_norm = nn.RMSNorm(self.written)
        else:
            self.qkv = nn.Linear(width, width + 2 * self.kv_heads * HEAD_DIM, bias=False)
            self.gate = nn.Linear(width, width, bias=False)
            self.q_norm = nn.RMSNorm(HEAD_DIM)
            self.k_norm = nn.RMSNorm(HEAD_DIM)
            self.out = nn.Linear(width, width, bias=False)
            self.attn_norm = nn.RMSNorm(width)
        if self.cross:
            self.cross_q = nn.Linear(width, width, bias=False)
            self.cross_kv = nn.Linear(SEQ_DIM, 2 * self.kv_heads * HEAD_DIM, bias=False)
            self.cross_q_norm = nn.RMSNorm(HEAD_DIM)
            self.cross_k_norm = nn.RMSNorm(HEAD_DIM)
            self.cross_out = nn.Linear(width, width, bias=False)
            self.cross_norm = nn.RMSNorm(width)
            self.cross_gate = nn.Parameter(torch.zeros(width))
        self.mlp_norm = nn.RMSNorm(width)
        self.w_gate = nn.Linear(width, hidden, bias=False)
        self.w_up = nn.Linear(width, hidden, bias=False)
        self.w_down = nn.Linear(hidden, width, bias=False)

    def grid_conv(self, y: Tensor, h: int, w: int) -> Tensor:
        B, N, _ = y.shape
        return self.conv(y.transpose(1, 2).reshape(B, self.width, h, w)).flatten(2).transpose(1, 2)

    def gdn_mix(self, y: Tensor, h: int, w: int) -> Tensor:
        """The update d = wr r - f s of the token's own state s, with r the depthwise 3x3 read of the value over the
        neighbourhood and the forget gate f and write gate wr separate per channel."""
        value, state, forget, write = self.gdn_proj(y).chunk(4, dim=-1)
        return torch.sigmoid(write) * F.silu(self.grid_conv(value, h, w)) - torch.sigmoid(forget) * state

    def forward(self, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor], text: Tensor, mask: Tensor, h: int, w: int) -> Tensor:
        B, N, _ = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (mod + self.table).unbind(1)

        y = F.rms_norm(x, (self.width,)) * (1 + scale_a[:, None]) + shift_a[:, None]
        if self.is_conv:
            mixed = self.gdn_mix(y, h, w) if self.gdn else self.grid_conv(y, h, w)
            x = x + gate_a[:, None] * F.pad(self.conv_norm(self.out(mixed)), (self.width - self.written, 0))  # zeros on the channels not written
        else:
            kv = self.kv_heads * HEAD_DIM
            q, k, v = self.qkv(y).split([self.width, kv, kv], dim=-1)
            q = q.view(B, N, self.heads, HEAD_DIM).transpose(1, 2)
            k, v = (a.view(B, N, self.kv_heads, HEAD_DIM).transpose(1, 2) for a in (k, v))
            q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
            attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.kv_heads != self.heads).transpose(1, 2).reshape(B, N, self.width)
            x = x + gate_a[:, None] * self.attn_norm(self.out(attn * torch.sigmoid(self.gate(y))))
        if self.cross:
            y = F.rms_norm(x, (self.width,))
            q = self.cross_q_norm(self.cross_q(y).view(B, N, self.heads, HEAD_DIM).transpose(1, 2))
            k, v = self.cross_kv(text).view(B, -1, 2, self.kv_heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, self.cross_k_norm(k), v, attn_mask=mask[:, None, None], enable_gqa=self.kv_heads != self.heads)
            x = x + self.cross_gate * self.cross_norm(self.cross_out(attn.transpose(1, 2).reshape(B, N, self.width)))

        y = F.rms_norm(x, (self.width,)) * (1 + scale_m[:, None]) + shift_m[:, None]
        return x + gate_m[:, None] * self.mlp_norm(self.w_down(F.silu(self.w_gate(y)) * self.w_up(y)))


class DiT(nn.Module):
    def __init__(self, null_embedding: Tensor, null_tokens: Tensor, depth: int, width: int, conv_period: int = CONV_PERIOD,
                 cross: bool = False, pooled_ada: bool = True, gdn: bool = True):
        super().__init__()
        assert null_tokens.shape == (NULL_TOKENS, SEQ_DIM), null_tokens.shape
        assert cross or pooled_ada, "the caption must reach the network by cross attention or the adaLN"
        assert conv_period >= 1, conv_period
        self.width = width
        self.null_embedding = nn.Parameter(null_embedding.detach().clone().float())
        self.null_tokens = nn.Parameter(null_tokens.detach().clone().float())  # [NULL_TOKENS, 512]
        self.patch_embed = nn.Linear(TOKEN_DIM, width - TOKEN_DIM)  # fills the meta channels; the first TOKEN_DIM carry the patch unchanged
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.text_proj = nn.Linear(TEXT_DIM, width) if pooled_ada else None
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, N_MOD * width))
        self.blocks = nn.ModuleList(Block(width, i % conv_period != conv_period - 1, cross, gdn) for i in range(depth))  # conv first, attention next
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

    def null_prompt(self, batch: int, length: int) -> Prompt:
        """The empty caption for `batch` rows, its tokens padded to `length`."""
        tokens = F.pad(self.null_tokens, (0, 0, 0, length - NULL_TOKENS)).expand(batch, -1, -1)
        mask = (torch.arange(length, device=tokens.device) < NULL_TOKENS).expand(batch, -1)
        return Prompt(self.null_embedding.expand(batch, -1), tokens, mask)

    def forward(self, x: Tensor, t: Tensor, prompt: Prompt) -> Tensor:
        """x [B, 2, H, W] latent at noise level t [B] -> F [B, 2, H, W].
        The network runs in the parameters' dtype (bf16 in training); x and the prompt are cast on the way in
        and F comes back fp32, so the objective and the sampler never see the network's dtype. t stays
        fp32 up to the sinusoid, whose phases need the precision."""
        dtype = self.patch_embed.weight.dtype
        tokens, h, w = patchify(x.to(dtype))
        c = self.time_mlp(timestep_sinusoid(t).to(dtype))
        if self.text_proj is not None:
            c = c + self.text_proj(F.rms_norm(prompt.pooled, (TEXT_DIM,)).to(dtype))
        text_tokens = F.rms_norm(prompt.tokens, (SEQ_DIM,)).to(dtype)
        mod = self.ada(c).view(-1, N_MOD, self.width)
        rope = rope_angles(h, w, x.device)

        tokens = torch.cat([tokens, self.patch_embed(tokens)], dim=-1)
        for block in self.blocks:
            tokens = block(tokens, mod, rope, text_tokens, prompt.mask, h, w)
        return unpatchify(tokens[..., :TOKEN_DIM], h, w).float()
