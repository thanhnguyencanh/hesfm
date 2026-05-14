"""
Unit tests for the evidential head and F-EDL losses.
Run with:  pytest scripts/evidential/tests/
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import torch
import pytest

from evidential import (
    EvidentialHead,
    FEDLSegLoss,
    dirichlet_from_evidence,
    dirichlet_mean,
    dirichlet_uncertainty,
    dirichlet_dissonance,
    fedl_moments,
    fedl_uncertainty,
    fedl_brier_loss,
    fedl_p_regulariser,
    edl_digamma_loss,
    kl_regulariser,
)
from evidential.fedl_loss import _onehot, kl_dirichlet_to_uniform


def test_dirichlet_basics():
    e = torch.tensor([[1.0, 3.0, 5.0, 1.0]]).view(1, 4, 1, 1)
    a = dirichlet_from_evidence(e)
    p = dirichlet_mean(a, dim=1)
    u = dirichlet_uncertainty(a, dim=1)
    assert torch.allclose(p.sum(dim=1), torch.ones(1, 1, 1), atol=1e-6)
    assert (u > 0).all() and (u <= 1).all()
    assert p.argmax(dim=1).item() == 2


def test_uncertainty_decreases_with_evidence():
    C = 5
    e_low  = torch.zeros(1, C, 1, 1)
    e_high = torch.full((1, C, 1, 1), 9.0)
    u_low  = dirichlet_uncertainty(dirichlet_from_evidence(e_low),  dim=1)
    u_high = dirichlet_uncertainty(dirichlet_from_evidence(e_high), dim=1)
    assert u_low.item() == pytest.approx(1.0, abs=1e-6)
    assert u_high.item() == pytest.approx(0.1,  abs=1e-6)


def test_dissonance_is_zero_for_dominant_class():
    e = torch.tensor([[100.0, 0.0, 0.0, 0.0]]).view(1, 4, 1, 1)
    a = dirichlet_from_evidence(e)
    diss = dirichlet_dissonance(a, dim=1)
    assert diss.item() == pytest.approx(0.0, abs=1e-3)


def test_dissonance_high_for_balanced_competition():
    e = torch.tensor([[20.0, 20.0, 0.0, 0.0]]).view(1, 4, 1, 1)
    a = dirichlet_from_evidence(e)
    diss = dirichlet_dissonance(a, dim=1)
    assert diss.item() > 0.4


def test_fedl_moments_match_paper():
    """At tau=0, mu = alpha/alpha0; var = mu(1-mu)/(alpha0+1)."""
    alpha = torch.tensor([[2.0, 3.0, 5.0]]).view(1, 3, 1, 1)
    p     = torch.tensor([[0.1, 0.3, 0.6]]).view(1, 3, 1, 1)
    tau   = torch.zeros(1, 1, 1, 1)
    mu, var = fedl_moments(alpha, p, tau)
    a0 = alpha.sum(dim=1, keepdim=True)
    assert torch.allclose(mu, alpha / a0, atol=1e-6)
    assert torch.allclose(var, mu * (1 - mu) / (a0 + 1), atol=1e-6)


def test_fedl_uncertainty_includes_tau():
    """C/(alpha0 + tau) — adding tau shrinks vacuity."""
    alpha = torch.full((1, 4, 1, 1), 1.0)
    tau0  = torch.zeros(1, 1, 1, 1)
    tau1  = torch.full((1, 1, 1, 1), 4.0)
    u0 = fedl_uncertainty(alpha, tau0)
    u1 = fedl_uncertainty(alpha, tau1)
    assert u0.item() == pytest.approx(1.0, abs=1e-6)
    assert u1.item() == pytest.approx(0.5, abs=1e-6)


def test_evidential_head_flexible_shapes():
    head = EvidentialHead(in_channels=64, num_classes=10, flexible=True)
    feats = torch.randn(2, 64, 32, 40)
    out = head(feats)
    assert out["alpha"].shape == (2, 10, 32, 40)
    assert out["p"].shape     == (2, 10, 32, 40)
    assert out["tau"].shape   == (2, 1, 32, 40)
    assert out["mu"].shape    == (2, 10, 32, 40)
    assert out["var"].shape   == (2, 10, 32, 40)
    assert out["uncertainty"].shape == (2, 32, 40)
    # mu, p both lie on the simplex
    assert torch.allclose(out["mu"].sum(dim=1),
                          torch.ones_like(out["mu"].sum(dim=1)), atol=1e-5)
    assert torch.allclose(out["p"].sum(dim=1),
                          torch.ones_like(out["p"].sum(dim=1)), atol=1e-5)
    assert (out["tau"] > 0).all()


def test_evidential_head_vanilla_shapes():
    head = EvidentialHead(in_channels=16, num_classes=4, flexible=False)
    feats = torch.randn(1, 16, 8, 8)
    out = head(feats)
    assert out["alpha"].shape == (1, 4, 8, 8)
    assert torch.allclose(out["mu"], out["p"], atol=1e-6)
    assert (out["tau"] == 0).all()


def test_kl_dirichlet_to_uniform_zero_at_uniform():
    alpha = torch.ones(1, 4, 2, 2)
    kl = kl_dirichlet_to_uniform(alpha)
    assert kl.abs().max().item() < 1e-5


def test_kl_zero_when_perfect_target_evidence():
    C = 4
    label = torch.full((1, 2, 2), 2)
    target_oh, _ = _onehot(label, C, ignore_index=255)
    alpha = target_oh * 100.0 + 1.0
    kl = kl_regulariser(alpha, target_oh)
    assert kl.item() < 0.1


def test_fedl_seg_loss_runs():
    """Full F-EDL flexible pipeline: head -> brier+p -> backward."""
    head = EvidentialHead(in_channels=32, num_classes=5, flexible=True)
    crit = FEDLSegLoss(num_classes=5, ce_warmup_epochs=0,
                       lambda_p=1.0, flexible=True)
    feats = torch.randn(2, 32, 16, 20, requires_grad=True)
    out = head(feats)
    labels = torch.randint(0, 5, (2, 16, 20))
    losses = crit(out, labels, epoch=5)
    assert "loss_brier" in losses
    assert "loss_p" in losses
    assert losses["loss"].requires_grad
    losses["loss"].backward()
    assert feats.grad is not None


def test_fedl_seg_loss_vanilla_path():
    """flexible=False uses EDL+KL terms."""
    head = EvidentialHead(in_channels=16, num_classes=4, flexible=False)
    crit = FEDLSegLoss(num_classes=4, ce_warmup_epochs=0,
                       lambda_kl_max=0.5, anneal_epochs=10, flexible=False)
    feats = torch.randn(1, 16, 8, 8, requires_grad=True)
    out = head(feats)
    labels = torch.randint(0, 4, (1, 8, 8))
    losses = crit(out, labels, epoch=5)
    assert "loss_edl" in losses
    assert "loss_kl" in losses
    losses["loss"].backward()
    assert feats.grad is not None


def test_ce_warmup_uses_ce():
    crit = FEDLSegLoss(num_classes=5, ce_warmup_epochs=10)
    head = EvidentialHead(in_channels=8, num_classes=5)
    feats = torch.randn(1, 8, 4, 4, requires_grad=True)
    out = head(feats)
    labels = torch.randint(0, 5, (1, 4, 4))
    losses = crit(out, labels, epoch=0)
    assert "loss_ce" in losses
    assert "loss_brier" not in losses
    assert "loss_edl" not in losses


def test_brier_zero_at_perfect_prediction():
    target = torch.zeros(1, 4, 2, 2)
    target[:, 1] = 1.0
    mu = target.clone()
    var = torch.zeros_like(mu)
    assert fedl_brier_loss(mu, var, target).item() == pytest.approx(0.0, abs=1e-6)
    assert fedl_p_regulariser(mu, target).item() == pytest.approx(0.0, abs=1e-6)


def test_uncertainty_decreases_with_correct_training():
    """Train on a fixed input/label and verify vacuity shrinks."""
    torch.manual_seed(0)
    head = EvidentialHead(in_channels=16, num_classes=4, flexible=False)
    crit = FEDLSegLoss(num_classes=4, ce_warmup_epochs=0,
                       lambda_kl_max=0.0, flexible=False, edl_form="digamma")
    optim = torch.optim.SGD(head.parameters(), lr=0.5)

    feats = torch.randn(1, 16, 4, 4)
    label = torch.full((1, 4, 4), 2)
    out0 = head(feats)
    u_before = out0["uncertainty"].mean().item()

    for _ in range(50):
        out = head(feats)
        loss = crit(out, label, epoch=0)["loss"]
        optim.zero_grad(); loss.backward(); optim.step()

    out1 = head(feats)
    assert out1["uncertainty"].mean().item() < u_before
    assert (out1["prob"].argmax(1) == 2).float().mean().item() > 0.5


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
