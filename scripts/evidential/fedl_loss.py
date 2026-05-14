"""
F-EDL training loss for semantic segmentation (Yoon & Kim NeurIPS 2025).

Primary loss is the Brier-on-moments form:
    L_brier = sum_c [ (y_c - mu_c)^2 + var_c ]

with an auxiliary regulariser on the anchor categorical p:
    L_p     = sum_c (y_c - p_c)^2

Optional CE warmup on the alpha-logits stabilises fine-tuning of pretrained
ESANet/DFormer backbones. The KL-to-uniform regulariser of Sensoy 2018 /
Kim et al. 2024 is kept for the non-flexible (vanilla EDL) path.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict


def _masked_mean(x: torch.Tensor, mask: Optional[torch.Tensor]) -> torch.Tensor:
    if mask is None:
        return x.mean()
    valid = mask.float()
    denom = valid.sum().clamp_min(1.0)
    return (x * valid).sum() / denom


def _onehot(labels: torch.Tensor, C: int, ignore_index: int = 255):
    valid = labels != ignore_index
    safe = labels.clone()
    safe[~valid] = 0
    onehot = F.one_hot(safe.long(), num_classes=C).permute(0, 3, 1, 2).float()
    onehot = onehot * valid.unsqueeze(1).float()
    return onehot, valid


def fedl_brier_loss(
    mu: torch.Tensor,
    var: torch.Tensor,
    target_onehot: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    err = (target_onehot - mu).pow(2).sum(dim=1)
    v   = var.sum(dim=1)
    return _masked_mean(err + v, mask)


def fedl_p_regulariser(
    p: torch.Tensor,
    target_onehot: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    return _masked_mean((target_onehot - p).pow(2).sum(dim=1), mask)


def edl_mse_loss(alpha, target_onehot, mask=None):
    S = alpha.sum(dim=1, keepdim=True)
    p = alpha / S
    err = (target_onehot - p).pow(2).sum(dim=1)
    var = (p * (1 - p) / (S + 1)).sum(dim=1)
    return _masked_mean(err + var, mask)


def edl_log_loss(alpha, target_onehot, mask=None):
    S = alpha.sum(dim=1, keepdim=True)
    per_class = target_onehot * (torch.log(S + 1e-9) - torch.log(alpha + 1e-9))
    return _masked_mean(per_class.sum(dim=1), mask)


def edl_digamma_loss(alpha, target_onehot, mask=None):
    S = alpha.sum(dim=1, keepdim=True)
    per_class = target_onehot * (torch.digamma(S) - torch.digamma(alpha))
    return _masked_mean(per_class.sum(dim=1), mask)


def kl_dirichlet_to_uniform(alpha_tilde: torch.Tensor) -> torch.Tensor:
    S_alpha = alpha_tilde.sum(dim=1)
    beta = torch.ones_like(alpha_tilde)
    S_beta = beta.sum(dim=1)
    log_B_alpha = torch.lgamma(alpha_tilde).sum(dim=1) - torch.lgamma(S_alpha)
    log_B_beta  = torch.lgamma(beta).sum(dim=1) - torch.lgamma(S_beta)
    digamma_term = (
        (alpha_tilde - beta)
        * (torch.digamma(alpha_tilde) - torch.digamma(S_alpha.unsqueeze(1)))
    ).sum(dim=1)
    return log_B_beta - log_B_alpha + digamma_term


def kl_regulariser(alpha, target_onehot, mask=None):
    alpha_tilde = target_onehot + (1 - target_onehot) * alpha
    return _masked_mean(kl_dirichlet_to_uniform(alpha_tilde), mask)


class FEDLSegLoss(nn.Module):
    """
    Composite F-EDL segmentation loss.

    flexible=True (true F-EDL):
        L = L_brier(mu, var, y) + lambda_p * L_p(p, y)
            [+ aux CE during ce_warmup_epochs]

    flexible=False (vanilla EDL):
        L = L_edl(alpha, y) + lambda_kl(t) * L_kl(alpha, y)
            [+ aux CE during ce_warmup_epochs]
    """

    def __init__(
        self,
        num_classes: int,
        lambda_p: float = 1.0,
        lambda_kl_max: float = 0.5,
        anneal_epochs: int = 30,
        ce_warmup_epochs: int = 0,
        edl_form: str = "digamma",
        ignore_index: int = 255,
        flexible: bool = True,
    ) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.lambda_p = lambda_p
        self.lambda_kl_max = lambda_kl_max
        self.anneal_epochs = max(1, anneal_epochs)
        self.ce_warmup_epochs = ce_warmup_epochs
        self.edl_form = edl_form
        self.ignore_index = ignore_index
        self.flexible = flexible

    def _edl_term(self, alpha, target_oh, mask):
        if   self.edl_form == "mse":     return edl_mse_loss(alpha, target_oh, mask)
        elif self.edl_form == "log":     return edl_log_loss(alpha, target_oh, mask)
        elif self.edl_form == "digamma": return edl_digamma_loss(alpha, target_oh, mask)
        else: raise ValueError(self.edl_form)

    def forward(
        self,
        net_output: Dict[str, torch.Tensor],
        labels: torch.Tensor,
        epoch: int = 0,
    ) -> Dict[str, torch.Tensor]:
        target_oh, mask = _onehot(labels, self.num_classes, self.ignore_index)
        out: Dict[str, torch.Tensor] = {}

        logits = net_output.get("logits")
        if epoch < self.ce_warmup_epochs and logits is not None:
            ce = F.cross_entropy(
                logits, labels.long(),
                ignore_index=self.ignore_index, reduction="mean",
            )
            out["loss_ce"] = ce
            out["loss"] = ce
            return out

        if self.flexible:
            mu  = net_output["mu"]
            var = net_output["var"]
            p   = net_output["p"]
            l_brier = fedl_brier_loss(mu, var, target_oh, mask)
            l_p     = fedl_p_regulariser(p, target_oh, mask)
            total = l_brier + self.lambda_p * l_p
            out["loss_brier"] = l_brier.detach()
            out["loss_p"]     = l_p.detach()
        else:
            alpha = net_output["alpha"]
            l_edl = self._edl_term(alpha, target_oh, mask)
            l_kl  = kl_regulariser(alpha, target_oh, mask)
            lam_kl = self.lambda_kl_max * min(1.0, epoch / float(self.anneal_epochs))
            total = l_edl + lam_kl * l_kl
            out["loss_edl"] = l_edl.detach()
            out["loss_kl"]  = l_kl.detach()
            out["lambda_kl"] = torch.tensor(lam_kl)

        out["loss"] = total
        return out
