# ML-PINNs: Physics-Informed Neural Networks for Nonlinear Conservation Laws

A PyTorch implementation of Physics-Informed Neural Networks (PINNs) for solving nonlinear hyperbolic conservation laws, with a focus on two-phase flow problems relevant to Carbon Capture, Utilization and Storage (CCUS) and reservoir simulation.

Corresponding paper: [Begell House](https://www.dl.begellhouse.com/journals/558048804a15188a,24f8785e4156a0df,157258503daf8310.html)

---

## Overview

Standard numerical solvers for hyperbolic PDEs with sharp shock fronts are computationally expensive or require fine tuning of numerical diffusion parameters. This repository explores neural network approaches to tackle these challenges:

1. **Standard PINNs** — enforce PDE residuals as a soft constraint in the loss function
2. **Sequential PINNs** — decompose the time domain into windows with warm-started weights, making shock-capturing tractable
3. **Neural Surrogate Models** — learn compact representations of expensive physics operators (e.g., reservoir fluid mobilities)

---

## Physics Problems

### Buckley-Leverett Equation

Two-phase (water-oil) displacement in porous media:

$$\frac{\partial S}{\partial t} + v \cdot \frac{\partial f(S)}{\partial x} = 0$$

where the fractional flow function is:

$$f(S) = \frac{S^2}{S^2 + (1-S)^2}$$

This equation produces sharp shock fronts, making it a challenging benchmark for PINNs.

### Nonlinear Scalar Conservation Law (CCUS)

$$\frac{\partial u}{\partial t} + 0.556 \cdot \frac{\partial}{\partial x}\left[\frac{u^2}{u^2 + (1-u)^2}\right] = 0$$

- Domain: $x \in [0,1]$, $t \in [0,1]$
- Initial condition: $u(x, 0) = 0$
- Boundary condition: $u(0, t) = 1$

### Dead Oil Reservoir Simulation (Surrogate)

A neural network surrogate replaces expensive equation-of-state calls in reservoir simulation, learning fluid mobilities and densities over a parameter space of pressure (300–800 bar) and composition (0–1).

---

## Repository Structure

```
ML-PINNs/
├── PINNs.py                    # Buckley-Leverett solver: MLP, PhysicsInformedNN, SequentialPINN
├── PINNs for CCUS/
│   ├── std-PINN.py             # Standard PINN for nonlinear conservation law
│   ├── seqPINNs_solver.py      # Sequential PINN with adaptive time-stepping
│   └── PIML_data.npy           # Pre-computed reference solution data
└── Convective operators/
    ├── model.py                # Neural surrogate for reservoir fluid operators
    └── main.py                 # Training and 2D parameter-space visualization
```

---

## Methods

### Standard PINN (`std-PINN.py`)

Trains a single network over the full space-time domain. The total loss is:

$$\mathcal{L} = \mathcal{L}_\text{data} + \mathcal{L}_\text{physics} + \mathcal{L}_\text{BC/IC}$$

- **Architecture:** 8 hidden layers × 20 neurons, tanh activations
- **Collocation points:** 10,000 (Latin Hypercube Sampling via `pyDOE`)
- **Data points:** 300 boundary/initial condition samples
- **Optimizer:** Adam → L-BFGS (two-stage)

### Sequential PINN (`seqPINNs_solver.py`)

Splits the time domain into overlapping windows and trains sequentially, transferring weights between windows as a warm start. Key features:

- **Adaptive collocation:** increases point density in high-residual regions
- **Adaptive time-stepping:** shortens windows when loss fails to converge
- **Predicted IC:** subsequent windows use the model's own prediction as initial condition, rather than exact data

This approach significantly improves shock-capturing compared to single-domain training.

### Neural Network Architecture (`PINNs.py`)

```python
class MLP(nn.Module):
    # Input normalized to [-1, 1]
    # Xavier initialization
    # Fully-connected with tanh activations
```

```
Input (x, t) -> [N_h hidden layers x N_n neurons] -> Output (u)
```

### Surrogate Model (`Convective operators/model.py`)

- **Architecture:** 8-layer fully-connected network (2 → 12 → ... → 1)
- **Input:** (pressure, composition)
- **Output:** fluid mobility / density
- **Optimizer:** Nadam, 500 epochs with early stopping

---

## Installation

```bash
git clone https://github.com/koreantiger/ML-PINNs.git
cd ML-PINNs
pip install torch numpy matplotlib scipy pyDOE
```

> Python 3.8+ recommended. GPU acceleration will be used automatically if available.

---

## Usage

### Solve the Buckley-Leverett Equation

```bash
python PINNs.py
```

Runs both the standard `PhysicsInformedNN` and the `SequentialPINN` on the Buckley-Leverett problem and plots the predicted vs. reference saturation profiles.

### Run CCUS Conservation Law Solvers

```bash
# Standard PINN
python "PINNs for CCUS/std-PINN.py"

# Sequential PINN with adaptive time-stepping
python "PINNs for CCUS/seqPINNs_solver.py"
```

### Train Surrogate for Convective Operators

```bash
python "Convective operators/main.py"
```

Trains the surrogate model and generates a 2D parameter-space visualization of predicted fluid properties.

---

## Key Design Choices

| Choice | Reason |
|---|---|
| Two-stage Adam → L-BFGS | Adam provides fast initial convergence; L-BFGS refines to a sharp minimum |
| Sequential time windows | Full-domain PINNs struggle with shock propagation; windowing improves accuracy |
| Latin Hypercube Sampling | Better space-filling than uniform random for collocation points |
| Input normalization to [-1, 1] | Improves gradient flow and training stability |
| Tanh activations | Smooth, differentiable everywhere — required for automatic differentiation of PDE residuals |

---

## Dependencies

| Package | Purpose |
|---|---|
| `torch` | Neural network training and automatic differentiation |
| `numpy` | Numerical arrays and data handling |
| `matplotlib` | Visualization of solutions |
| `scipy` | Grid interpolation for reference solutions |
| `pyDOE` | Latin Hypercube Sampling for collocation points |

---

## Citation

If you use this code, please cite the associated paper:

```bibtex
@article{pinns_ccus,
  title   = {Physics-Informed Neural Networks for Nonlinear Conservation Laws},
  journal = {Begell House},
  url     = {https://www.dl.begellhouse.com/journals/558048804a15188a,24f8785e4156a0df,157258503daf8310.html}
}
```

---

## License

MIT
