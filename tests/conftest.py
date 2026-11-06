import pytest
import torch

from tinydit.data import pad_sequences
from tinydit.model import DiT, GATE_ROWS, NULL_TOKENS, SEQ_DIM, TEXT_DIM, Prompt


def open_gates(model):
    """Opens the zero-initialised gates, so every block's output reaches F and every matrix gets a gradient."""
    for block in model.blocks:
        block.table.data[list(GATE_ROWS)] = 0.5
        if block.cross:
            block.cross_gate.data.fill_(0.5)


@pytest.fixture
def model():
    torch.manual_seed(0)
    null = torch.nn.functional.normalize(torch.randn(TEXT_DIM), dim=0)
    return DiT(null, torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, conv_period=2, cross=True)


@pytest.fixture
def prompt():
    """Three captions of 5, 3 and 2 tokens."""
    tokens, mask = pad_sequences([torch.randn(n, SEQ_DIM) for n in (5, 3, 2)], torch.float32)
    return Prompt(torch.nn.functional.normalize(torch.randn(3, TEXT_DIM), dim=-1), tokens, mask)
