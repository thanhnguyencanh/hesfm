"""
Evidential DFormerv2 wrapper.

Wraps a DFormerv2 (`models.builder.EncoderDecoder` with a DFormerv2_S/_B/_L
backbone) and replaces its final classifier conv with our `EvidentialHead`.
The backbone + decoder feature pipeline is reused unchanged so that any
pretrained NYUv2-40 / SUN-RGBD checkpoint can be loaded into the backbone.

Two decoder families are supported:
  * MLPDecoder (default) — final conv is `decode_head.linear_pred`,
    in-channels = `decoder_embed_dim`.
  * HAM / UPerNet / DeepLabV3+ / NL — BaseDecodeHead-derived, final conv is
    `decode_head.conv_seg`, in-channels = `decoder.channels`.

Note on DFormer's RGB-D normalisation
-------------------------------------
DFormer's dataloader feeds an 8-bit replicated-to-3-channel depth and
normalises with mean=[0.48]*3, std=[0.28]*3 (the `sign=True` branch when
`x_is_single_channel=True`). The ROS node mirrors this exactly so train- and
inference-time pre-processing match.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .evidential_head import EvidentialHead


_DFORMERV2_CHANNELS = {
    "DFormerv2_S": [64,  128, 256, 512],
    "DFormerv2_B": [80,  160, 320, 512],
    "DFormerv2_L": [112, 224, 448, 640],
}


class EvidentialDFormerv2(nn.Module):
    """
    Args:
        seg_model:   a DFormer EncoderDecoder instance with a v2 backbone.
        num_classes: classes for the evidential head (e.g. 40 NYUv2-40).
        flexible:    use F-EDL `(alpha, p, tau)` head (default True).
        evidence:    activation: 'softplus' | 'relu' | 'exp'.
    """

    def __init__(
        self,
        seg_model: nn.Module,
        num_classes: int,
        flexible: bool = True,
        evidence: str = "softplus",
    ) -> None:
        super().__init__()
        self.seg = seg_model
        self.num_classes = num_classes

        in_ch = self._strip_classifier(seg_model.decode_head)
        self.evidential_head = EvidentialHead(
            in_channels=in_ch,
            num_classes=num_classes,
            evidence=evidence,
            flexible=flexible,
        )

    @staticmethod
    def _strip_classifier(decode_head: nn.Module) -> int:
        if hasattr(decode_head, "linear_pred") and isinstance(
            decode_head.linear_pred, nn.Conv2d
        ):
            in_ch = decode_head.linear_pred.in_channels
            decode_head.linear_pred = nn.Identity()
            return in_ch
        if hasattr(decode_head, "conv_seg") and isinstance(
            decode_head.conv_seg, nn.Conv2d
        ):
            in_ch = decode_head.conv_seg.in_channels
            decode_head.conv_seg = nn.Identity()
            return in_ch
        raise AttributeError(
            "Could not find DFormer decoder classifier "
            "(expected `linear_pred` or `conv_seg`)."
        )

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb:    [B, 3, H, W]   ImageNet-normalised RGB.
            depth:  [B, 3, H, W]   3-channel depth normalised the way DFormer
                                   expects (mean=[0.48]*3, std=[0.28]*3).
            target_size: (Ht, Wt) to upsample outputs to. If None, native
                         decoder resolution (typically H/4 × W/4) is kept.
        """
        feats = self.seg.backbone(rgb, depth)
        if isinstance(feats, (list, tuple)) and len(feats) == 2 \
                and isinstance(feats[0], (list, tuple)):
            feats = feats[0]

        decoder_feats = self.seg.decode_head(feats)

        out = self.evidential_head(decoder_feats)

        if target_size is not None:
            for k in ("evidence", "alpha", "p", "tau", "mu", "var", "prob", "logits"):
                if k in out:
                    out[k] = F.interpolate(
                        out[k], size=target_size,
                        mode="bilinear", align_corners=False,
                    )
            out["uncertainty"] = F.interpolate(
                out["uncertainty"].unsqueeze(1),
                size=target_size, mode="bilinear", align_corners=False,
            ).squeeze(1)

        return out

    @torch.no_grad()
    def export_onnx(
        self,
        path: str,
        height: int = 480,
        width: int = 640,
        opset: int = 11,
        dynamic_batch: bool = False,
    ) -> None:
        self.eval()
        dummy_rgb   = torch.zeros(1, 3, height, width)
        dummy_depth = torch.zeros(1, 3, height, width)

        class _ExportWrap(nn.Module):
            def __init__(self, parent): super().__init__(); self.p = parent
            def forward(self, rgb, depth):
                o = self.p(rgb, depth, target_size=(height, width))
                return o["mu"], o["uncertainty"]

        dyn_axes = None
        if dynamic_batch:
            dyn_axes = {
                "rgb":         {0: "B"},
                "depth":       {0: "B"},
                "prob":        {0: "B"},
                "uncertainty": {0: "B"},
            }

        torch.onnx.export(
            _ExportWrap(self),
            (dummy_rgb, dummy_depth),
            path,
            input_names=["rgb", "depth"],
            output_names=["prob", "uncertainty"],
            opset_version=opset,
            dynamic_axes=dyn_axes,
        )


# ---------------------------------------------------------------------------
# Convenience builder
# ---------------------------------------------------------------------------

def build_dformerv2_evidential(
    backbone: str = "DFormerv2_L",
    decoder: str = "MLPDecoder",
    num_classes: int = 40,
    decoder_embed_dim: int = 512,
    drop_path_rate: float = 0.1,
    pretrained_backbone: Optional[str] = None,
    flexible: bool = True,
    evidence: str = "softplus",
    bn_eps: float = 1e-3,
    bn_momentum: float = 0.1,
    aux_rate: float = 0.0,
    background: int = 255,
) -> EvidentialDFormerv2:
    """
    Build a DFormerv2 EncoderDecoder + F-EDL head.

    `pretrained_backbone` should point to the original DFormerv2 backbone
    weights (loaded by DFormer's own `init_weights`). Pass None when you
    intend to load a full evidential checkpoint from disk afterwards.
    """
    if backbone not in _DFORMERV2_CHANNELS:
        raise ValueError(
            f"Unknown backbone {backbone!r}; expected one of "
            f"{list(_DFORMERV2_CHANNELS)}"
        )

    from types import SimpleNamespace
    cfg = SimpleNamespace(
        backbone=backbone,
        decoder=decoder,
        num_classes=num_classes,
        decoder_embed_dim=decoder_embed_dim,
        drop_path_rate=drop_path_rate,
        pretrained_model=pretrained_backbone,
        bn_eps=bn_eps,
        bn_momentum=bn_momentum,
        aux_rate=aux_rate,
        background=background,
    )

    from models.builder import EncoderDecoder
    seg = EncoderDecoder(cfg=cfg, criterion=None)
    return EvidentialDFormerv2(
        seg_model=seg,
        num_classes=num_classes,
        flexible=flexible,
        evidence=evidence,
    )
