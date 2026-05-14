"""
Evidential output head — true Flexible-EDL parameterisation.

For each pixel the head emits three quantities:
    alpha  [B, C, H, W]   — Dirichlet concentration, alpha = softplus(g_a) + 1
    p      [B, C, H, W]   — softmax anchor categorical
    tau    [B, 1, H, W]   — softplus strength of the anchor (>= 1e-3)

Predictive moments (Yoon & Kim 2025):
    mu  = (alpha + tau * p) / (alpha0 + tau)
    var = mu(1-mu)/(alpha0+tau+1)
        + tau^2 * p(1-p) / ((alpha0+tau)(alpha0+tau+1))

Vacuity is generalised to include the anchor strength:
    u = C / (alpha0 + tau)

If `flexible=False`, the head degenerates to vanilla EDL (Sensoy 2018):
mu = p = alpha/alpha0 and tau = 0.

References:
- Sensoy et al. NeurIPS 2018 (vanilla EDL)
- Kim et al. IROS 2024 (uncertainty-aware semantic mapping)
- Yoon & Kim NeurIPS 2025 (Flexible Evidential Deep Learning)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict, Optional


def softplus_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.softplus(logits)


def relu_evidence(logits: torch.Tensor) -> torch.Tensor:
    return F.relu(logits)


def exp_evidence(logits: torch.Tensor, clamp: float = 10.0) -> torch.Tensor:
    return torch.exp(torch.clamp(logits, max=clamp))


_EVIDENCE_FNS = {
    "softplus": softplus_evidence,
    "relu":     relu_evidence,
    "exp":      exp_evidence,
}


def dirichlet_from_evidence(evidence: torch.Tensor) -> torch.Tensor:
    return evidence + 1.0


def dirichlet_mean(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    return alpha / alpha.sum(dim=dim, keepdim=True)


def dirichlet_uncertainty(alpha: torch.Tensor, dim: int = 1) -> torch.Tensor:
    C = alpha.shape[dim]
    return C / alpha.sum(dim=dim)


def fedl_moments(
    alpha: torch.Tensor,
    p: torch.Tensor,
    tau: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (mu, var) per Yoon & Kim 2025. tau has shape [B, 1, H, W]."""
    alpha0 = alpha.sum(dim=1, keepdim=True)
    denom1 = alpha0 + tau
    denom2 = denom1 + 1.0
    mu = (alpha + tau * p) / denom1
    var = mu * (1.0 - mu) / denom2 + (tau * tau) * p * (1.0 - p) / (denom1 * denom2)
    return mu, var


def fedl_uncertainty(alpha: torch.Tensor, tau: torch.Tensor, dim: int = 1) -> torch.Tensor:
    """Generalised vacuity: C / (alpha0 + tau). Returns [B, H, W]."""
    C = alpha.shape[dim]
    alpha0 = alpha.sum(dim=dim)
    return C / (alpha0 + tau.squeeze(dim))


def dirichlet_dissonance(alpha: torch.Tensor, dim: int = 1, eps: float = 1e-9) -> torch.Tensor:
    b = alpha - 1.0
    S = alpha.sum(dim=dim, keepdim=True)
    b = b / S
    b_i = b.unsqueeze(dim + 1)
    b_j = b.unsqueeze(dim)
    diff = (b_i - b_j).abs()
    summ = b_i + b_j + eps
    bal = 1.0 - diff / summ
    C = alpha.shape[dim]
    eye = torch.eye(C, device=alpha.device, dtype=alpha.dtype)
    eye = eye.view([1, C, C] + [1] * (alpha.ndim - 2))
    bal = bal * (1.0 - eye)
    num = (b.unsqueeze(dim) * bal).sum(dim=dim + 1)
    den = (b.unsqueeze(dim).expand_as(bal) * (1.0 - eye)).sum(dim=dim + 1) + eps
    return (b * num / den).sum(dim=dim)


class EvidentialHead(nn.Module):
    """
    F-EDL head with three 1x1 convolutions.

    Args:
        in_channels:  channels of the decoder feature map fed in.
        num_classes:  number of semantic classes.
        evidence:     'softplus' (default), 'relu', or 'exp' for alpha.
        flexible:     if True (default), emit p and tau as well (true F-EDL).
                      if False, degenerate to vanilla EDL.
        dropout:      optional 2D dropout before the head.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        evidence: str = "softplus",
        flexible: bool = True,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if evidence not in _EVIDENCE_FNS:
            raise ValueError(f"Unknown evidence activation: {evidence}")
        self.num_classes = num_classes
        self.evidence_fn = _EVIDENCE_FNS[evidence]
        self.flexible = flexible

        self.dropout = nn.Dropout2d(dropout) if dropout > 0 else nn.Identity()
        self.alpha_conv = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        if flexible:
            self.p_conv   = nn.Conv2d(in_channels, num_classes, kernel_size=1)
            self.tau_conv = nn.Conv2d(in_channels, 1, kernel_size=1)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.kaiming_normal_(self.alpha_conv.weight, mode="fan_out",
                                nonlinearity="relu")
        if self.alpha_conv.bias is not None:
            nn.init.zeros_(self.alpha_conv.bias)
        if self.flexible:
            nn.init.kaiming_normal_(self.p_conv.weight, mode="fan_out",
                                    nonlinearity="relu")
            if self.p_conv.bias is not None:
                nn.init.zeros_(self.p_conv.bias)
            nn.init.zeros_(self.tau_conv.weight)
            nn.init.zeros_(self.tau_conv.bias)

    def forward(self, feats: torch.Tensor) -> Dict[str, torch.Tensor]:
        x = self.dropout(feats)
        a_logits = self.alpha_conv(x)
        evidence = self.evidence_fn(a_logits)
        alpha = dirichlet_from_evidence(evidence)

        if self.flexible:
            p_logits = self.p_conv(x)
            p = F.softmax(p_logits, dim=1)
            tau = F.softplus(self.tau_conv(x)) + 1e-3
            mu, var = fedl_moments(alpha, p, tau)
            unc = fedl_uncertainty(alpha, tau)
        else:
            p = dirichlet_mean(alpha, dim=1)
            tau = torch.zeros_like(alpha[:, :1])
            mu = p
            alpha0 = alpha.sum(dim=1, keepdim=True)
            var = mu * (1.0 - mu) / (alpha0 + 1.0)
            unc = dirichlet_uncertainty(alpha, dim=1)

        return {
            "evidence":    evidence,
            "alpha":       alpha,
            "p":           p,
            "tau":         tau,
            "mu":          mu,
            "var":         var,
            "prob":        mu,
            "uncertainty": unc,
            "logits":      a_logits,
        }


@torch.no_grad()
def alpha_to_per_pixel_outputs(
    alpha: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Vanilla-EDL convenience: ignores tau. Prefer `predictive_outputs`."""
    prob = dirichlet_mean(alpha, dim=1)
    unc  = dirichlet_uncertainty(alpha, dim=1)
    label = prob.argmax(dim=1)
    return label, prob, unc


@torch.no_grad()
def predictive_outputs(
    out: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Returns (label, mu, uncertainty) from an EvidentialHead forward dict."""
    mu = out["mu"]
    unc = out["uncertainty"]
    label = mu.argmax(dim=1)
    return label, mu, unc
