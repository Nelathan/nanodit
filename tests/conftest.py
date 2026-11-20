import pytest
import torch

from nanodit.data import pad_sequences
from nanodit.model import BlockType, DiT, GATE_ROWS, NULL_TOKENS, SEQ_DIM, Prompt

PAIR = (BlockType("conv"), BlockType("attn"))  # the smallest layout with every kind of matrix
needs_cuda = pytest.mark.cuda


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: takes gradients through the model, so it runs on CUDA")


def pytest_collection_modifyitems(items):
    for item in items:
        if item.get_closest_marker("cuda") and not torch.cuda.is_available():
            item.add_marker(pytest.mark.skip(reason="flex_attention has no backward on the CPU"))


@pytest.fixture(autouse=True)
def autograd_only_on_cuda(request):
    """flex_attention refuses parameters that require grad on the CPU, so a test runs without autograd unless it is
    marked `cuda`."""
    with torch.set_grad_enabled(request.node.get_closest_marker("cuda") is not None):
        yield


def open_gates(model):
    """Opens the zero-initialised gates, so every block's output reaches F and every matrix gets a gradient."""
    for block in model.blocks:
        block.table.data[list(GATE_ROWS)] = 0.5


@pytest.fixture
def model():
    torch.manual_seed(0)
    return DiT(torch.randn(NULL_TOKENS, SEQ_DIM), depth=2, width=512, layout=PAIR)


@pytest.fixture
def prompt():
    """Three captions of 5, 3 and 2 tokens."""
    tokens, mask = pad_sequences([torch.randn(n, SEQ_DIM) for n in (5, 3, 2)], torch.float32)
    return Prompt(tokens, mask)
