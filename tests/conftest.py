import pytest
import torch

from tinydit.model import DiT, GATE_ROWS, TEXT_DIM


def open_gates(model):
    """Opens the zero-initialised gates, so every block's output reaches F and every matrix gets a gradient."""
    for block in model.blocks:
        block.table.data[list(GATE_ROWS)] = 0.5


@pytest.fixture
def model():
    torch.manual_seed(0)
    null = torch.nn.functional.normalize(torch.randn(TEXT_DIM), dim=0)
    return DiT(null, depth=2, width=512)


@pytest.fixture
def text():
    return torch.nn.functional.normalize(torch.randn(3, TEXT_DIM), dim=-1)
