# PINN for the 1D Advection-Diffusion Equation

A Physics-Informed Neural Network (PINN) implemented in PyTorch for the 1D advection-diffusion PDE:

$$u_t + c\,u_x = \nu\,u_{xx}, \quad x \in [0,1],\; t \in [0, T]$$

with initial condition $u(x,0) = \sin(2\pi x)$ and periodic boundary conditions $u(0,t) = u(1,t)$.

The exact solution is:

$$u(x,t) = e^{-\nu(2\pi)^2 t}\,\sin\!\bigl(2\pi(x - ct)\bigr)$$

## Approach

The PDE residual, initial condition loss, and boundary condition loss are enforced simultaneously during training via automatic differentiation (no labelled interior data required):

$$\mathcal{L} = \underbrace{\frac{1}{N_r}\sum_i r(x_i,t_i)^2}_{\text{PDE residual}} + \underbrace{\frac{1}{N_0}\sum_j (u_\theta(x_j,0) - \sin(2\pi x_j))^2}_{\text{IC loss}} + \underbrace{\frac{1}{N_b}\sum_k (u_\theta(0,t_k) - u_\theta(1,t_k))^2}_{\text{BC loss}}$$

where $r = u_t + c\,u_x - \nu\,u_{xx}$ is computed by automatic differentiation through the network.

## Architecture

- Fully connected network: input $(x,t)$ → 4 hidden layers (width 64, tanh) → scalar $u$
- Trained with Adam (15 000 iterations, learning rate $10^{-3}$, step decay)
- 2000 PDE collocation points, 200 IC points, 100 BC points per batch

## Results

Relative $L^2$ error over the full space-time domain: **~5 × 10⁻³**

![PINN result](pinn_result.png)

## Usage

```bash
pip install -r requirements.txt
python pinn.py
```

Produces `pinn_result.png` with exact solution, PINN prediction, and pointwise error.
