"""
Evidential ESANet wrapper.

Wraps a stock ESANet model (https://github.com/TUI-NICR/ESANet) and replaces
its final classification conv with our EvidentialHead. The rest of the
encoder/decoder is reused unchanged, so pretrained NYUv2/SUN-RGBD weights
load directly into the backbone.

Usage (training):
    from esanet.model import ESANet
    backbone = ESANet(num_classes=40, encoder_block='NonBottleneck1D',
                      encoder='resnet34', modality='rgbd')
    backbone.load_state_dict(torch.load('r34_NBt1D.pth')['state_dict'],
                             strict=False)
    model = EvidentialESANet(backbone, num_classes=40, flexible=True)

Usage (inference inside a ROS node, see scripts/esanet_evidential_node.py).

Author: HESFM @ JAIST
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple, Optional

from .evidential_head import EvidentialHead


class EvidentialESANet(nn.Module):
    """
    Args:
        backbone:    a stock ESANet instance. Must expose `forward(rgb, depth)`
                     returning final-resolution features OR a logits tensor.
        num_classes: number of classes in the final layer (40 for NYUv2-40).
        flexible:    use F-EDL flexible Dirichlet head (default True).
        evidence:    activation: 'softplus' | 'relu' | 'exp'.
        feature_layer: name of the last feature-map module to tap. If None,
                       the wrapper will try to auto-detect for ESANet's
                       `decoder.out_conv` and replace it.
    """

    def __init__(
        self,
        backbone: nn.Module,
        num_classes: int,
        flexible: bool = True,
        evidence: str = "softplus",
        feature_layer: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes

        in_ch = self._infer_input_channels(backbone, feature_layer)
        self._strip_final_classifier(backbone)

        self.evidential_head = EvidentialHead(
            in_channels=in_ch,
            num_classes=num_classes,
            evidence=evidence,
            flexible=flexible,
        )

    @staticmethod
    def _infer_input_channels(backbone: nn.Module,
                              feature_layer: Optional[str]) -> int:
        if feature_layer is not None:
            mod = dict(backbone.named_modules())[feature_layer]
            return mod.in_channels  # type: ignore[attr-defined]

        for path in ("decoder.conv_out", "conv_out", "classifier"):
            try:
                mod = backbone
                for p in path.split("."):
                    mod = getattr(mod, p)
                if isinstance(mod, nn.Conv2d):
                    return mod.in_channels
            except AttributeError:
                continue

        raise AttributeError(
            "Could not locate the final classifier conv on the ESANet backbone. "
            "Pass `feature_layer='your.module.path'` explicitly."
        )

    @staticmethod
    def _strip_final_classifier(backbone: nn.Module) -> None:
        for path in ("decoder.conv_out", "conv_out", "classifier"):
            try:
                parts = path.split(".")
                parent = backbone
                for p in parts[:-1]:
                    parent = getattr(parent, p)
                if isinstance(getattr(parent, parts[-1]), nn.Conv2d):
                    setattr(parent, parts[-1], nn.Identity())
                    return
            except AttributeError:
                continue
        raise AttributeError("Failed to strip ESANet classifier.")

    def forward(
        self,
        rgb: torch.Tensor,
        depth: torch.Tensor,
        target_size: Optional[Tuple[int, int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            rgb:    [B, 3, H, W]
            depth:  [B, 1, H, W]  (or [B, H, W] - we'll add the channel dim)
            target_size: if given, results are upsampled to (Ht, Wt).
        """
        if depth.ndim == 3:
            depth = depth.unsqueeze(1)

        feats = self.backbone(rgb, depth)
        if isinstance(feats, (list, tuple)):
            feats = feats[0]

        out = self.evidential_head(feats)

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
        """
        Export the wrapped network to ONNX for TensorRT deployment on Jetson.
        Two outputs are emitted: `prob` (softmax-equivalent) and `uncertainty`
        (vacuity scalar). Other tensors are dropped to keep the graph small.
        """
        self.eval()
        dummy_rgb   = torch.zeros(1, 3, height, width)
        dummy_depth = torch.zeros(1, 1, height, width)

        class _ExportWrap(nn.Module):
            def __init__(self, parent): super().__init__(); self.p = parent
            def forward(self, rgb, depth):
                o = self.p(rgb, depth)
                # Export the predictive mean (mu) and the F-EDL vacuity.
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
