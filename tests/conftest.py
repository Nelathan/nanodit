import pytest
import torch

from tinydit.model import DiT, TEXT_DIM


@pytest.fixture
def model():
    torch.manual_seed(0)
    null = torch.nn.functional.normalize(torch.randn(TEXT_DIM), dim=0)
    return DiT(null, depth=2)


@pytest.fixture
def text():
    return torch.nn.functional.normalize(torch.randn(3, TEXT_DIM), dim=-1)
