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
SEQ_DIM = 512  # the encoder's token states, before its projection to 768
NULL_TOKENS = 2  # the empty caption's <bos> and <eos> states, the null prompt's tokens at init
TIME_DIM = 256
ROPE_BASE = 10000.0
ADALN_STD = 0.02  # "small random" shift/scale rows
N_MOD = 6  # shift, scale, gate for the token mixer (attention or conv), then for the MLP
GATE_ROWS = (2, 5)
MLP_RATIO = 4  # SwiGLU hidden over the stream width
KV_RATIO = 4  # query heads per key/value head
ROPE_FRACTION = 0.25  # of each axis' frequencies rotate: the fast ones; the slow ones barely turn on the grid and carry no position


class BlockType(NamedTuple):
    """One block's settings. A layout is a sequence of them, repeated to the depth (SPEC §3)."""

    mixer: str  # "conv" or "attn"


MIXERS = ("conv", "attn")
LAYOUT = (BlockType("conv"), BlockType("conv"), BlockType("conv"), BlockType("attn"))


class Prompt(NamedTuple):
    """The text side of a batch: the caption's token states, which the cross attention reads."""

    tokens: Tensor  # [B, L, 512], zero past a caption's end, L >= NULL_TOKENS
    mask: Tensor  # [B, L] bool, true on a caption's tokens

    def to(self, device: torch.device, dtype: torch.dtype) -> Prompt:
        return Prompt(self.tokens.to(device, dtype), self.mask.to(device))

    def rows(self, start: int, stop: int) -> Prompt:
        return Prompt(self.tokens[start:stop], self.mask[start:stop])


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
    """cos, sin of shape [h*w, HEAD_DIM/2]: the first half of the angles encodes the row, the second the column.
    Only the first ROPE_FRACTION of each axis' frequencies rotate; the rest get angle 0, so those dims carry no position."""
    pairs = HEAD_DIM // 4
    freqs = ROPE_BASE ** (-torch.arange(pairs, device=device) / pairs)
    freqs = freqs * (torch.arange(pairs, device=device) < ROPE_FRACTION * pairs)
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
    """A conv block mixes tokens by LFM2's gated short conv, a depthwise 3x3 conv over the token grid between two
    per-channel gates, and writes all channels. An attention block mixes with gated grouped-query
    attention, then reads the caption's token states through a cross attention, grouped-query and without RoPE, that
    takes no adaLN, only a zero-initialised gate per channel."""

    def __init__(self, width: int, kind: BlockType):
        super().__init__()
        assert kind.mixer in MIXERS, kind
        self.width, self.heads, hidden = width, width // HEAD_DIM, MLP_RATIO * width
        self.kv_heads = max(1, self.heads // KV_RATIO)
        assert self.heads % self.kv_heads == 0
        self.is_conv = kind.mixer == "conv"
        self.table = nn.Parameter(torch.zeros(N_MOD, width))
        if self.is_conv:
            self.conv = nn.Conv2d(width, width, 3, padding=1, groups=width, bias=False)
            self.out = nn.Linear(width, width, bias=False)
            self.conv_in = nn.Linear(width, 3 * width, bias=False)  # B, C and v
        else:
            self.qkv = nn.Linear(width, width + 2 * self.kv_heads * HEAD_DIM, bias=False)
            self.gate = nn.Linear(width, width, bias=False)
            self.q_norm = nn.RMSNorm(HEAD_DIM)
            self.k_norm = nn.RMSNorm(HEAD_DIM)
            self.out = nn.Linear(width, width, bias=False)
            self.attn_norm = nn.RMSNorm(width)
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

    def forward(self, x: Tensor, mod: Tensor, rope: tuple[Tensor, Tensor], text: Tensor, mask: Tensor, h: int, w: int) -> Tensor:
        B, N, _ = x.shape
        shift_a, scale_a, gate_a, shift_m, scale_m, gate_m = (mod + self.table).unbind(1)

        y = F.rms_norm(x, (self.width,)) * (1 + scale_a[:, None]) + shift_a[:, None]
        if self.is_conv:
            b, c, v = self.conv_in(y).chunk(3, dim=-1)
            read = self.conv((b * v).transpose(1, 2).reshape(B, self.width, h, w)).flatten(2).transpose(1, 2)
            # The norm sits on the read, not after `out`, so a small C makes a small write: a token can abstain.
            x = x + gate_a[:, None] * self.out(c * F.rms_norm(read, (self.width,)))
        else:
            kv = self.kv_heads * HEAD_DIM
            q, k, v = self.qkv(y).split([self.width, kv, kv], dim=-1)
            q = q.view(B, N, self.heads, HEAD_DIM).transpose(1, 2)
            k, v = (a.view(B, N, self.kv_heads, HEAD_DIM).transpose(1, 2) for a in (k, v))
            q, k = apply_rope(self.q_norm(q), *rope), apply_rope(self.k_norm(k), *rope)
            attn = F.scaled_dot_product_attention(q, k, v, enable_gqa=self.kv_heads != self.heads).transpose(1, 2).reshape(B, N, self.width)
            x = x + gate_a[:, None] * self.attn_norm(self.out(attn * torch.sigmoid(self.gate(y))))
            y = F.rms_norm(x, (self.width,))
            q = self.cross_q_norm(self.cross_q(y).view(B, N, self.heads, HEAD_DIM).transpose(1, 2))
            k, v = self.cross_kv(text).view(B, -1, 2, self.kv_heads, HEAD_DIM).permute(2, 0, 3, 1, 4)
            attn = F.scaled_dot_product_attention(q, self.cross_k_norm(k), v, attn_mask=mask[:, None, None], enable_gqa=self.kv_heads != self.heads)
            x = x + self.cross_gate * self.cross_norm(self.cross_out(attn.transpose(1, 2).reshape(B, N, self.width)))

        y = F.rms_norm(x, (self.width,)) * (1 + scale_m[:, None]) + shift_m[:, None]
        return x + gate_m[:, None] * self.mlp_norm(self.w_down(F.silu(self.w_gate(y)) * self.w_up(y)))


class DiT(nn.Module):
    def __init__(self, null_tokens: Tensor, depth: int, width: int, layout: tuple[BlockType, ...] = LAYOUT):
        """`layout` repeats to `depth` blocks, the last repeat cut short when it does not divide the depth."""
        super().__init__()
        assert null_tokens.shape == (NULL_TOKENS, SEQ_DIM), null_tokens.shape
        assert layout, "a layout needs at least one block type"
        kinds = [layout[i % len(layout)] for i in range(depth)]
        assert any(kind.mixer == "attn" for kind in kinds), "the caption reaches the network only by cross attention"
        self.width = width
        self.null_tokens = nn.Parameter(null_tokens.detach().clone().float())  # [NULL_TOKENS, 512]
        self.patch_embed = nn.Linear(TOKEN_DIM, width - TOKEN_DIM)  # fills the meta channels; the first TOKEN_DIM carry the patch unchanged
        self.time_mlp = nn.Sequential(nn.Linear(TIME_DIM, width), nn.SiLU(), nn.Linear(width, width))
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(width, N_MOD * width))
        self.blocks = nn.ModuleList(Block(width, kind) for kind in kinds)
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
        return Prompt(tokens, mask)

    def forward(self, x: Tensor, t: Tensor, prompt: Prompt) -> Tensor:
        """x [B, 2, H, W] latent at noise level t [B] -> F [B, 2, H, W].
        The network runs in the parameters' dtype (bf16 in training); x and the prompt are cast on the way in
        and F comes back fp32, so the objective and the sampler never see the network's dtype. t stays
        fp32 up to the sinusoid, whose phases need the precision. A caption with no tokens is the null prompt: cross
        attention over a fully masked row has no softmax."""
        empty = (~prompt.mask.any(1)).view(-1, 1)
        null = self.null_prompt(*prompt.tokens.shape[:2])
        prompt = Prompt(torch.where(empty[..., None], null.tokens, prompt.tokens), torch.where(empty, null.mask, prompt.mask))
        dtype = self.patch_embed.weight.dtype
        tokens, h, w = patchify(x.to(dtype))
        c = self.time_mlp(timestep_sinusoid(t).to(dtype))
        text_tokens = F.rms_norm(prompt.tokens, (SEQ_DIM,)).to(dtype)
        mod = self.ada(c).view(-1, N_MOD, self.width)
        rope = rope_angles(h, w, x.device)

        tokens = torch.cat([tokens, self.patch_embed(tokens)], dim=-1)
        for block in self.blocks:
            tokens = block(tokens, mod, rope, text_tokens, prompt.mask, h, w)
        return unpatchify(tokens[..., :TOKEN_DIM], h, w).float()
