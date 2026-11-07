import torch
from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
from transformers.models.gemma4.image_processing_gemma4 import convert_image_to_patches
from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPooler

from tinydit.align import UPSAMPLE, VisionTarget


class NearestDecoder:
    """Takes a 3-channel 'latent' for the image itself, at 2 image pixels per latent pixel."""

    def decode(self, z):
        return torch.nn.functional.interpolate(z, scale_factor=2, mode="nearest")


class RecordingTower:
    """Records its input; returns one 1-d feature per pooled token, numbered in the order it hands them back."""

    def __call__(self, pixel_values, pixel_position_ids):
        self.pixel_values, self.positions = pixel_values, pixel_position_ids
        pooled = pixel_values.shape[0] * pixel_values.shape[1] // 9
        return type("Out", (), {"last_hidden_state": torch.arange(pooled, dtype=torch.float32)[:, None]})


def run(x):
    vision, tower = object.__new__(VisionTarget), RecordingTower()
    vision.tower, vision.decoder = tower, NearestDecoder()
    return vision(x), tower


def test_the_tower_gets_the_processors_patches_and_positions_of_the_image_at_three_times_its_size():
    torch.manual_seed(0)
    x = torch.rand(2, 3, 16, 24) * 2 - 1  # two images of 32 x 48 px: 2 x 3 of our tokens
    features, tower = run(x)
    images = torch.nn.functional.interpolate((NearestDecoder().decode(x) + 1) / 2, scale_factor=UPSAMPLE, mode="bicubic").clamp(0, 1)
    assert images.shape[-2:] == (96, 144)
    for b in range(2):
        assert torch.equal(tower.pixel_values[b], convert_image_to_patches(images[b], 16).to(torch.bfloat16))
    grid = torch.stack(torch.meshgrid(torch.arange(9), torch.arange(6), indexing="xy"), dim=-1).reshape(-1, 2)  # as the processor builds (x, y)
    assert torch.equal(tower.positions[0], grid) and torch.equal(tower.positions[1], grid)
    assert torch.equal(features[..., 0], torch.arange(12.0).view(2, 6))  # the tower's flat output, one row per image


def test_the_towers_pooling_returns_our_tokens_in_patchify_order():
    _, tower = run(torch.zeros(1, 3, 16, 24))
    x, y = tower.positions[0].unbind(-1)
    token = (y // 3) * 3 + x // 3  # our row-major token over 2 rows of 3 that each patch lies in
    pooler = Gemma4VisionPooler(Gemma4VisionConfig(hidden_size=1))
    pooled, _ = pooler(token.float()[None, :, None], tower.positions, torch.zeros(1, 54, dtype=torch.bool), output_length=6)
    assert torch.allclose(pooled[0, :, 0], torch.arange(6.0))  # the pooler averages, and scales by sqrt(hidden), 1 here
