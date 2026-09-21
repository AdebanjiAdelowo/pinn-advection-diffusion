"""
Independent diagnostics for the 1D advection-diffusion PINN.

Everything here is computed from the network (or the exact solution) on fixed,
deterministic grids. None of it reads the training loss, so it can be used to
check what the training loss does or does not enforce.

Notation: E_u(t) = u(0,t) - u(1,t), E_ux(t) = u_x(0,t) - u_x(1,t).
The total flux of u_t + (c u - nu u_x)_x = 0 is F = c u - nu u_x, so
    diffusive-flux mismatch = -nu * E_ux,   total-flux mismatch = c * E_u - nu * E_ux.
"""

import numpy as np
import torch
import torch.nn as nn

from pinn import C, NU, T, pde_residual, u_exact


class ExactSolution(nn.Module):
    """The exact solution wrapped as a module, so it can go through the same diagnostics."""

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return u_exact(x, t)


def relative_l2(u_pred, u_ref) -> float:
    """||u_pred - u_ref||_2 / ||u_ref||_2 over all entries."""
    u_pred, u_ref = np.asarray(u_pred), np.asarray(u_ref)
    return float(np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref))


def _ddx(model, x, t):
    """u and u_x at the points (x, t), u_x by automatic differentiation."""
    x = x.clone().requires_grad_(True)
    u = model(x, t)
    return u, torch.autograd.grad(u.sum(), x, create_graph=True)[0]


def _summary(err) -> dict:
    err = np.asarray(err)
    return dict(rms=float(np.sqrt(np.mean(err ** 2))), max_abs=float(np.max(np.abs(err))))


def periodicity_diagnostics(model, n_t: int = 2001, dtype=torch.float32) -> dict:
    """Value, derivative and flux mismatch between x=0 and x=1 on a uniform time grid.

    Returns the summaries plus the raw arrays E_u(t), E_ux(t) under "t", "E_u", "E_ux".
    """
    t = torch.linspace(0, T, n_t, dtype=dtype).unsqueeze(1)
    x0, x1 = torch.zeros_like(t), torch.ones_like(t)
    u0, ux0 = _ddx(model, x0, t)
    u1, ux1 = _ddx(model, x1, t)
    e_u = (u0 - u1).detach().numpy().ravel()
    e_ux = (ux0 - ux1).detach().numpy().ravel()
    # Scale for the derivative mismatch: RMS of the exact u_x at x = 0 over the same times.
    _, ux_ref = _ddx(u_exact_module, x0.to(torch.float64), t.to(torch.float64))
    ux_scale = float(np.sqrt(np.mean(ux_ref.detach().numpy() ** 2)))
    return dict(
        n_t=n_t,
        value=_summary(e_u),
        derivative=_summary(e_ux),
        derivative_rms_relative_to_exact_ux=_summary(e_ux)["rms"] / ux_scale,
        diffusive_flux=_summary(NU * e_ux),
        total_flux=_summary(C * e_u - NU * e_ux),
        t=t.numpy().ravel(), E_u=e_u, E_ux=e_ux,
    )


def pde_residual_metrics(model, nx: int = 201, nt: int = 101, dtype=torch.float32) -> dict:
    """PDE residual r = u_t + c u_x - nu u_xx on a fixed grid, boundaries included."""
    x, t = torch.meshgrid(torch.linspace(0, 1, nx, dtype=dtype),
                          torch.linspace(0, T, nt, dtype=dtype), indexing="ij")
    x = x.reshape(-1, 1).requires_grad_(True)
    t = t.reshape(-1, 1).requires_grad_(True)
    return _summary(pde_residual(model, x, t).detach().numpy())


def initial_condition_metrics(model, n: int = 1001, dtype=torch.float32) -> dict:
    """Error of u(x,0) against sin(2 pi x) on a fixed grid."""
    x = torch.linspace(0, 1, n, dtype=dtype).unsqueeze(1)
    t = torch.zeros_like(x)
    with torch.no_grad():
        err = (model(x, t) - u_exact(x, t)).numpy()
    return dict(**_summary(err), relative_l2=relative_l2(model(x, t).detach().numpy(),
                                                          u_exact(x, t).numpy()))


def accuracy_metrics(model, nx: int = 300, nt: int = 150, dtype=torch.float32) -> dict:
    """Relative L2 error of u and of u_x on the grid used for the headline figure."""
    x, t = torch.meshgrid(torch.linspace(0, 1, nx, dtype=dtype),
                          torch.linspace(0, T, nt, dtype=dtype), indexing="ij")
    x, t = x.reshape(-1, 1), t.reshape(-1, 1)
    u, ux = _ddx(model, x, t)
    u_ref, ux_ref = _ddx(u_exact_module, x.to(torch.float64), t.to(torch.float64))
    return dict(
        relative_l2_u=relative_l2(u.detach().numpy(), u_ref.detach().numpy()),
        relative_l2_ux=relative_l2(ux.detach().numpy(), ux_ref.detach().numpy()),
        max_abs_error_u=float(np.max(np.abs(u.detach().numpy() - u_ref.detach().numpy()))),
    )


def all_metrics(model, dtype=torch.float32) -> dict:
    """All independent diagnostics for one network (raw time series left out)."""
    per = periodicity_diagnostics(model, dtype=dtype)
    per = {k: v for k, v in per.items() if k not in ("t", "E_u", "E_ux")}
    return dict(
        accuracy=accuracy_metrics(model, dtype=dtype),
        pde_residual=pde_residual_metrics(model, dtype=dtype),
        initial_condition=initial_condition_metrics(model, dtype=dtype),
        periodicity=per,
    )


u_exact_module = ExactSolution()
