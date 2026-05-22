"""
HESFM evidential perception module — F-EDL parameterisation.

Public API:
    EvidentialHead          - drop-in F-EDL head (alpha, p, tau)
    EvidentialESANet        - ESANet backbone + evidential head wrapper
    FEDLSegLoss             - composite F-EDL training loss
    fedl_moments            - (mu, var) from (alpha, p, tau)
    fedl_uncertainty        - generalised vacuity C/(alpha0 + tau)
    predictive_outputs      - (label, mu, unc) from forward dict
    alpha_to_per_pixel_outputs - vanilla-EDL helper (ignores tau)
"""

from .evidential_head import (
    EvidentialHead,
    dirichlet_from_evidence,
    dirichlet_mean,
    dirichlet_uncertainty,
    dirichlet_dissonance,
    fedl_moments,
    fedl_uncertainty,
    alpha_to_per_pixel_outputs,
    predictive_outputs,
)
from .esanet_evidential import EvidentialESANet
from .dformerv2_evidential import EvidentialDFormerv2, build_dformerv2_evidential
from .fedl_loss import (
    FEDLSegLoss,
    fedl_brier_loss,
    fedl_p_regulariser,
    edl_mse_loss,
    edl_log_loss,
    edl_digamma_loss,
    kl_regulariser,
)

__all__ = [
    "EvidentialHead",
    "EvidentialESANet",
    "EvidentialDFormerv2",
    "build_dformerv2_evidential",
    "FEDLSegLoss",
    "dirichlet_from_evidence",
    "dirichlet_mean",
    "dirichlet_uncertainty",
    "dirichlet_dissonance",
    "fedl_moments",
    "fedl_uncertainty",
    "alpha_to_per_pixel_outputs",
    "predictive_outputs",
    "fedl_brier_loss",
    "fedl_p_regulariser",
    "edl_mse_loss",
    "edl_log_loss",
    "edl_digamma_loss",
    "kl_regulariser",
]
