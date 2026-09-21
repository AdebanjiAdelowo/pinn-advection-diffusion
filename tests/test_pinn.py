"""Tests for the periodic boundary condition, the exact solution and the diagnostics.

Run with:  python -m pytest tests -q
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import diagnostics
import pinn
from pinn import C, NU, T, PINN, pde_residual, periodic_bc_losses, u_exact


class Exact(nn.Module):
    def forward(self, x, t):
        return u_exact(x, t)


class ValuePeriodicOnly(nn.Module):
    """u = x (1 - x)(1 + t): u(0,t) = u(1,t) = 0 but u_x(0,t) = 1+t, u_x(1,t) = -(1+t)."""

    def forward(self, x, t):
        return x * (1 - x) * (1 + t)


def _t(n=64, dtype=torch.float64):
    torch.manual_seed(0)
    return torch.rand(n, 1, dtype=dtype) * T


# --- 1, 2: the loss contains a value term and a derivative term -------------

def test_value_term_is_the_mean_squared_endpoint_mismatch():
    torch.manual_seed(0)
    model, t = PINN().double(), _t()
    loss_u, loss_ux = periodic_bc_losses(model, t, derivative=False)
    expected = torch.mean((model(torch.zeros_like(t), t) - model(torch.ones_like(t), t)) ** 2)
    assert loss_u.item() == pytest.approx(expected.item(), rel=1e-12)
    assert loss_ux is None, "value-only formulation must not contain a derivative term"


def test_derivative_term_matches_central_finite_differences():
    torch.manual_seed(0)
    model, t = PINN().double(), _t()
    _, loss_ux = periodic_bc_losses(model, t, derivative=True)
    h = 1e-6

    def fd(x0):
        x = torch.full_like(t, x0)
        with torch.no_grad():
            return (model(x + h, t) - model(x - h, t)) / (2 * h)

    expected = torch.mean((fd(0.0) - fd(1.0)) ** 2)
    assert loss_ux.item() == pytest.approx(expected.item(), rel=1e-5)


def test_derivative_term_detects_what_the_value_term_cannot():
    model, t = ValuePeriodicOnly(), _t()
    loss_u, loss_ux = periodic_bc_losses(model, t, derivative=True)
    assert loss_u.item() == pytest.approx(0.0, abs=1e-30)
    expected = torch.mean((2 * (1 + t)) ** 2)  # u_x(0) - u_x(1) = 2 (1 + t)
    assert loss_ux.item() == pytest.approx(expected.item(), rel=1e-12)


# --- 3: gradients through u_x reach the parameters --------------------------

def test_derivative_term_is_connected_to_autograd():
    torch.manual_seed(0)
    model, t = PINN(), _t(dtype=torch.float32)
    _, loss_ux = periodic_bc_losses(model, t, derivative=True)
    assert loss_ux.requires_grad and loss_ux.grad_fn is not None
    loss_ux.backward()
    # u_x does not depend on the additive output bias, so that one parameter is
    # (correctly) outside the graph; every other parameter must be connected.
    out_bias = model.net[-1].bias
    assert out_bias.grad is None
    for p in model.parameters():
        if p is not out_bias:
            assert p.grad is not None and torch.isfinite(p.grad).all()
    # the x-column of the first layer must receive gradient from an x-derivative loss
    assert model.net[0].weight.grad[:, 0].abs().max() > 0


def test_derivative_term_changes_the_parameter_gradient():
    torch.manual_seed(0)
    t = _t(dtype=torch.float32)

    def flat_grad(derivative):
        torch.manual_seed(1)
        model = PINN()
        loss_u, loss_ux = periodic_bc_losses(model, t, derivative)
        (loss_u if loss_ux is None else loss_u + loss_ux).backward()
        return torch.cat([p.grad.flatten() for p in model.parameters()])

    assert not torch.allclose(flat_grad(False), flat_grad(True))


# --- 4-6: exact solution: PDE, value and derivative periodicity -------------

def test_exact_solution_satisfies_the_pde():
    torch.manual_seed(0)
    x = torch.rand(500, 1, dtype=torch.float64).requires_grad_(True)
    t = (torch.rand(500, 1, dtype=torch.float64) * T).requires_grad_(True)
    r = pde_residual(Exact(), x, t)
    assert r.abs().max().item() < 1e-10


def test_exact_solution_and_initial_condition_are_consistent():
    x = torch.linspace(0, 1, 101, dtype=torch.float64).unsqueeze(1)
    assert torch.allclose(u_exact(x, torch.zeros_like(x)), torch.sin(2 * np.pi * x), atol=1e-15)


def test_exact_solution_is_value_and_derivative_periodic():
    per = diagnostics.periodicity_diagnostics(Exact(), dtype=torch.float64)
    assert per["value"]["max_abs"] < 1e-12
    assert per["derivative"]["max_abs"] < 1e-11
    assert per["total_flux"]["max_abs"] < 1e-11


def test_value_only_network_is_flagged_by_the_diagnostics():
    per = diagnostics.periodicity_diagnostics(ValuePeriodicOnly(), dtype=torch.float64)
    assert per["value"]["max_abs"] < 1e-15
    assert per["derivative"]["max_abs"] == pytest.approx(2 * (1 + T), rel=1e-12)
    assert per["diffusive_flux"]["max_abs"] == pytest.approx(NU * 2 * (1 + T), rel=1e-12)


# --- 7: relative error ------------------------------------------------------

def test_relative_l2():
    ref = np.array([[3.0, 4.0], [0.0, 0.0]])
    assert diagnostics.relative_l2(ref, ref) == 0.0
    assert diagnostics.relative_l2(1.1 * ref, ref) == pytest.approx(0.1)
    assert diagnostics.relative_l2(np.zeros_like(ref), ref) == pytest.approx(1.0)
    # ||(3,4)-(0,0)|| / ||(3,4)|| against a hand value: pred = ref + (0, 0.5) in the zero row
    pred = ref + np.array([[0.0, 0.0], [0.3, 0.4]])
    assert diagnostics.relative_l2(pred, ref) == pytest.approx(0.5 / 5.0)


def test_grid_relative_error_matches_the_plot_formula():
    torch.manual_seed(0)
    model = PINN()
    nx, nt = 300, 150
    x, t = torch.meshgrid(torch.linspace(0, 1, nx), torch.linspace(0, T, nt), indexing="ij")
    xf, tf = x.reshape(-1, 1), t.reshape(-1, 1)
    with torch.no_grad():
        u_pred = model(xf, tf).reshape(nx, nt).numpy()
    u_ref = u_exact(xf, tf).reshape(nx, nt).numpy()
    plot_formula = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
    assert diagnostics.accuracy_metrics(model)["relative_l2_u"] == pytest.approx(plot_formula, rel=1e-5)


# --- the value-only path is exactly the original training loop --------------

def _original_train(n_epochs, lr=1e-3):
    """Verbatim copy of the original (commit f234b1d) training loop, minus printing."""
    model = PINN(width=64, depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    for _ in range(n_epochs):
        optimizer.zero_grad()
        x_ic = torch.rand(200, 1)
        t_ic = torch.zeros(200, 1)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_exact(x_ic, t_ic)) ** 2)
        t_bc = torch.rand(100, 1) * T
        x_bc0 = torch.zeros(100, 1)
        x_bc1 = torch.ones(100, 1)
        loss_bc = torch.mean((model(x_bc0, t_bc) - model(x_bc1, t_bc)) ** 2)
        x_pde = torch.rand(2000, 1).requires_grad_(True)
        t_pde = (torch.rand(2000, 1) * T).requires_grad_(True)
        loss_pde = torch.mean(pde_residual(model, x_pde, t_pde) ** 2)
        (loss_ic + loss_bc + loss_pde).backward()
        optimizer.step()
        scheduler.step()
    return model


def test_value_only_flag_reproduces_the_original_loop():
    torch.manual_seed(7)
    original = _original_train(25)
    new = pinn.train(n_epochs=25, derivative_bc=False, seed=7, verbose=False)
    for p, q in zip(original.parameters(), new.parameters()):
        assert torch.equal(p, q)


def test_corrected_loop_uses_the_same_batches_and_initialisation():
    """Toggling derivative_bc must change only the loss, not the RNG stream or the init."""
    a = pinn.train(n_epochs=0, derivative_bc=False, seed=3, verbose=False)
    b = pinn.train(n_epochs=0, derivative_bc=True, seed=3, verbose=False)
    assert all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))
    pinn.train(n_epochs=3, derivative_bc=True, verbose=False, seed=5)
    next_after_b = torch.rand(1)
    pinn.train(n_epochs=3, derivative_bc=False, verbose=False, seed=5)
    next_after_a = torch.rand(1)
    assert torch.equal(next_after_a, next_after_b), "RNG consumption differs between formulations"
