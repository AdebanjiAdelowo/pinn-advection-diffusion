"""
Physics-Informed Neural Network for the 1D Advection-Diffusion Equation

PDE:   u_t + c u_x = nu u_xx,   x in [0, 1],  t in [0, T]
IC:    u(x, 0) = sin(2 pi x)
BC:    periodic  (u(0, t) = u(1, t)  and  u_x(0, t) = u_x(1, t))
Exact: u(x, t) = exp(-nu (2 pi)^2 t) sin(2 pi (x - c t))

The PDE residual, initial condition, and boundary condition are embedded
directly into the training loss, so the neural network is constrained to
satisfy the physics throughout the domain.

The PDE is second order in x, so a well-posed periodic problem needs both
value and derivative periodicity. `derivative_bc=False` in `train` reproduces
the original value-only formulation (kept for the controlled comparison).
"""

import argparse

import torch
import torch.nn as nn
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

torch.manual_seed(42)
np.random.seed(42)

C  = 1.0    # advection speed
NU = 0.05   # diffusion coefficient
T  = 0.5    # final time


# ---------------------------------------------------------------------------
# Exact solution
# ---------------------------------------------------------------------------

def u_exact(x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    return torch.exp(-NU * (2 * np.pi) ** 2 * t) * torch.sin(2 * np.pi * (x - C * t))


# ---------------------------------------------------------------------------
# Network architecture
# ---------------------------------------------------------------------------

class PINN(nn.Module):
    def __init__(self, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(2, width), nn.Tanh()]
        for _ in range(depth - 1):
            layers += [nn.Linear(width, width), nn.Tanh()]
        layers.append(nn.Linear(width, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([x, t], dim=1))


# ---------------------------------------------------------------------------
# PDE residual (automatic differentiation)
# ---------------------------------------------------------------------------

def pde_residual(model: PINN, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    u = model(x, t)
    u_t  = torch.autograd.grad(u.sum(), t, create_graph=True)[0]
    u_x  = torch.autograd.grad(u.sum(), x, create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x.sum(), x, create_graph=True)[0]
    return u_t + C * u_x - NU * u_xx


# ---------------------------------------------------------------------------
# Periodic boundary condition
# ---------------------------------------------------------------------------

def periodic_bc_losses(model: nn.Module, t: torch.Tensor, derivative: bool = True):
    """Periodic-BC loss terms between x = 0 and x = 1 at the times `t`.

    Returns (loss_u, loss_ux). loss_u is the mean squared mismatch of
    u(0,t) - u(1,t). loss_ux is the mean squared mismatch of u_x(0,t) - u_x(1,t)
    (u_x by automatic differentiation), or None when `derivative` is False.
    """
    x0 = torch.zeros_like(t)
    x1 = torch.ones_like(t)
    if derivative:
        x0.requires_grad_(True)
        x1.requires_grad_(True)
    u0, u1 = model(x0, t), model(x1, t)
    loss_u = torch.mean((u0 - u1) ** 2)
    if not derivative:
        return loss_u, None
    ux0 = torch.autograd.grad(u0.sum(), x0, create_graph=True)[0]
    ux1 = torch.autograd.grad(u1.sum(), x1, create_graph=True)[0]
    return loss_u, torch.mean((ux0 - ux1) ** 2)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(n_epochs: int = 15000, lr: float = 1e-3, derivative_bc: bool = True,
          seed: int | None = None, verbose: bool = True, return_history: bool = False):
    """Train the PINN. `derivative_bc=False` is the original value-only periodic BC.

    `seed` re-seeds torch/numpy before the network is built (the module-level
    seed of 42 is used when None). With `return_history=True` returns
    (model, history), history being the component losses of the current batch
    every 500 epochs and at the last epoch.
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    model = PINN(width=64, depth=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5000, gamma=0.5)
    history = []

    for epoch in range(n_epochs):
        optimizer.zero_grad()

        # --- initial condition points ---
        x_ic = torch.rand(200, 1)
        t_ic = torch.zeros(200, 1)
        loss_ic = torch.mean((model(x_ic, t_ic) - u_exact(x_ic, t_ic)) ** 2)

        # --- periodic boundary condition: u(0,t) = u(1,t), u_x(0,t) = u_x(1,t) ---
        t_bc = torch.rand(100, 1) * T
        loss_bc_u, loss_bc_ux = periodic_bc_losses(model, t_bc, derivative_bc)
        loss_bc = loss_bc_u if loss_bc_ux is None else loss_bc_u + loss_bc_ux

        # --- PDE collocation points ---
        x_pde = torch.rand(2000, 1).requires_grad_(True)
        t_pde = (torch.rand(2000, 1) * T).requires_grad_(True)
        r = pde_residual(model, x_pde, t_pde)
        loss_pde = torch.mean(r ** 2)

        loss = loss_ic + loss_bc + loss_pde
        loss.backward()
        optimizer.step()
        scheduler.step()

        if epoch % 500 == 0 or epoch == n_epochs - 1:
            history.append(dict(
                epoch=epoch, loss=loss.item(), pde=loss_pde.item(), ic=loss_ic.item(),
                bc_u=loss_bc_u.item(),
                bc_ux=None if loss_bc_ux is None else loss_bc_ux.item()))

        if verbose and epoch % 1000 == 0:
            with torch.no_grad():
                x_t = torch.linspace(0, 1, 200).unsqueeze(1)
                t_t = torch.full((200, 1), T)
                l2 = torch.mean((model(x_t, t_t) - u_exact(x_t, t_t)) ** 2).sqrt()
            print(f"epoch {epoch:5d}  loss={loss.item():.3e}  L2_err(t=T)={l2.item():.3e}")

    return (model, history) if return_history else model


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot(model: PINN, filename: str = "pinn_result.png") -> None:
    nx, nt = 300, 150
    x = torch.linspace(0, 1, nx)
    t = torch.linspace(0, T, nt)
    X, Tt = torch.meshgrid(x, t, indexing="ij")
    X_flat  = X.reshape(-1, 1)
    T_flat  = Tt.reshape(-1, 1)

    with torch.no_grad():
        u_pred = model(X_flat, T_flat).reshape(nx, nt).numpy()
    u_ref = u_exact(X_flat, T_flat).reshape(nx, nt).numpy()
    err   = np.abs(u_pred - u_ref)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    kw = dict(aspect="auto", origin="lower", extent=[0, T, 0, 1], vmin=-1, vmax=1)

    im0 = axes[0].imshow(u_ref,  cmap="RdBu_r", **kw)
    axes[0].set(title="Exact solution", xlabel="t", ylabel="x")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(u_pred, cmap="RdBu_r", **kw)
    axes[1].set(title="PINN prediction", xlabel="t", ylabel="x")
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(err, cmap="hot_r",
                         aspect="auto", origin="lower", extent=[0, T, 0, 1])
    axes[2].set(title="|Pointwise error|", xlabel="t", ylabel="x")
    plt.colorbar(im2, ax=axes[2])

    plt.suptitle(
        f"PINN: $u_t + {C}\\,u_x = {NU}\\,u_{{xx}}$,  "
        f"IC: $\\sin(2\\pi x)$,  periodic BC",
        fontsize=11
    )
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")

    l2_rel = np.linalg.norm(u_pred - u_ref) / np.linalg.norm(u_ref)
    print(f"Relative L2 error over full domain: {l2_rel:.3e}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--bc", choices=["value", "value+derivative"], default="value+derivative",
                        help="periodic BC: value only (original) or value + derivative (default)")
    parser.add_argument("--out", default=None, help="output figure (default depends on --bc)")
    args = parser.parse_args()
    derivative_bc = args.bc == "value+derivative"
    out = args.out or ("pinn_result_value_derivative.png" if derivative_bc else "pinn_result.png")
    model = train(derivative_bc=derivative_bc)
    plot(model, out)
