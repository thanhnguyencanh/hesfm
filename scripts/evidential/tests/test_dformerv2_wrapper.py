"""
Smoke tests for EvidentialDFormerv2 — exercise classifier-stripping and the
forward path without requiring the actual DFormer repo to be importable.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import torch.nn as nn
import pytest

from evidential import EvidentialDFormerv2


def _make_fake_dformer(decoder_kind: str, embed_dim: int = 32, num_classes: int = 5):
    """Stand-in for DFormer EncoderDecoder with the right attribute layout."""

    class FakeBackbone(nn.Module):
        def forward(self, rgb, depth):
            B, _, H, W = rgb.shape
            # Return the four stages typical of DFormerv2 encoders.
            return [
                torch.randn(B, embed_dim, H // 4,  W // 4),
                torch.randn(B, embed_dim, H // 8,  W // 8),
                torch.randn(B, embed_dim, H // 16, W // 16),
                torch.randn(B, embed_dim, H // 32, W // 32),
            ]

    class FakeDecoderMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            self.linear_pred = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        def forward(self, x):
            B = x[0].shape[0]
            f = torch.randn(B, self.embed_dim,
                            x[0].shape[-2], x[0].shape[-1])
            return self.linear_pred(f)

    class FakeDecoderHam(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed_dim = embed_dim
            self.conv_seg = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

        def forward(self, x):
            B = x[0].shape[0]
            f = torch.randn(B, self.embed_dim,
                            x[0].shape[-2], x[0].shape[-1])
            return self.conv_seg(f)

    seg = nn.Module()
    seg.backbone = FakeBackbone()
    seg.decode_head = FakeDecoderMLP() if decoder_kind == "mlp" else FakeDecoderHam()
    return seg


@pytest.mark.parametrize("kind", ["mlp", "ham"])
def test_strip_and_forward(kind):
    seg = _make_fake_dformer(kind, embed_dim=16, num_classes=4)
    model = EvidentialDFormerv2(seg_model=seg, num_classes=4, flexible=True)

    rgb   = torch.randn(2, 3, 64, 96)
    depth = torch.randn(2, 3, 64, 96)
    out = model(rgb, depth, target_size=(64, 96))

    for k in ("alpha", "p", "tau", "mu", "var", "uncertainty"):
        assert k in out
    assert out["mu"].shape  == (2, 4, 64, 96)
    assert out["uncertainty"].shape == (2, 64, 96)
    assert torch.allclose(out["mu"].sum(dim=1),
                          torch.ones_like(out["mu"].sum(dim=1)), atol=1e-4)


def test_classifier_replaced_with_identity():
    seg = _make_fake_dformer("mlp", embed_dim=8, num_classes=3)
    EvidentialDFormerv2(seg_model=seg, num_classes=3)
    assert isinstance(seg.decode_head.linear_pred, nn.Identity)


def test_missing_classifier_raises():
    seg = nn.Module()
    seg.backbone = nn.Identity()
    seg.decode_head = nn.Module()
    with pytest.raises(AttributeError):
        EvidentialDFormerv2(seg_model=seg, num_classes=3)
