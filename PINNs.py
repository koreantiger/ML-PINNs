"""
Buckley-Leverett PINN — PyTorch conversion
Based on Fuck & Tchelepi 2020 (sequential and standard training)

Key changes from TF1:
  - tf.Session / placeholders  →  pure PyTorch autograd
  - ScipyOptimizerInterface    →  torch-based L-BFGS (built-in) + optional Adam warm-up
  - xavier_init                →  nn.init.xavier_normal_
  - Sequential training added  →  time-domain decomposition with warm re-initialisation

PDE (Buckley-Leverett conservation law):
    ∂S/∂t + v * ∂f(S)/∂x = 0
    f(S) = S² / (S² + (1−S)²)     fractional flow (water)
    v = 0.5̄  (= 5/9, dimensionless velocity factor)

Author: converted to PyTorch by Kiarash — 2024
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyDOE import lhs

# ── reproducibility ──────────────────────────────────────────────────────────
torch.manual_seed(1234)
np.random.seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# 1.  Neural-network architecture
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    """
    Fully-connected network with tanh activations.
    Accepts a flat list of layer widths, e.g. [2, 20, 20, 20, 1].
    Input is normalised to [-1, 1] internally using lb/ub.
    """

    def __init__(self, layers: list[int], lb: torch.Tensor, ub: torch.Tensor):
        super().__init__()
        self.lb = lb  # (2,) lower bounds  [x_min, t_min]
        self.ub = ub  # (2,) upper bounds  [x_max, t_max]

        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))   # output layer, no activation
        self.net = nn.Sequential(*net)

        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """x, t: column tensors of shape (N, 1).  Returns S_hat of shape (N, 1)."""
        xt = torch.cat([x, t], dim=1)                         # (N, 2)
        # normalise to [-1, 1]
        xt = 2.0 * (xt - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(xt)


# ═══════════════════════════════════════════════════════════════════════════════
# 2.  Fractional-flow function and physics residual
# ═══════════════════════════════════════════════════════════════════════════════

def fractional_flow(S: torch.Tensor) -> torch.Tensor:
    """f(S) = S² / (S² + (1−S)²)  — water fractional flow (BL)."""
    return S ** 2 / (S ** 2 + (1.0 - S) ** 2)


V = 5.0 / 9.0   # = 0.555̄  dimensionless velocity factor


def pde_residual(model: MLP,
                 x_f: torch.Tensor,
                 t_f: torch.Tensor) -> torch.Tensor:
    """
    Computes the PDE residual:   r = ∂S/∂t + V * ∂f(S)/∂x
    x_f, t_f must already require grad.
    """
    x_f = x_f.requires_grad_(True)
    t_f = t_f.requires_grad_(True)

    S = model(x_f, t_f)                                        # (N, 1)

    # automatic differentiation
    S_t = torch.autograd.grad(
        S, t_f,
        grad_outputs=torch.ones_like(S),
        create_graph=True, retain_graph=True
    )[0]

    f_S = fractional_flow(S)
    f_x = torch.autograd.grad(
        f_S, x_f,
        grad_outputs=torch.ones_like(f_S),
        create_graph=True, retain_graph=True
    )[0]

    residual = S_t + V * f_x
    return residual


# ═══════════════════════════════════════════════════════════════════════════════
# 3.  Loss function
# ═══════════════════════════════════════════════════════════════════════════════

def compute_loss(model: MLP,
                 x_u: torch.Tensor, t_u: torch.Tensor, u_true: torch.Tensor,
                 x_f: torch.Tensor, t_f: torch.Tensor) -> torch.Tensor:
    """MSE_data + MSE_physics."""
    # data / boundary / initial condition loss
    u_pred = model(x_u, t_u)
    loss_u = torch.mean((u_true - u_pred) ** 2)

    # physics residual loss
    f_pred = pde_residual(model, x_f, t_f)
    loss_f = torch.mean(f_pred ** 2)

    return loss_u + loss_f


# ═══════════════════════════════════════════════════════════════════════════════
# 4.  Standard PINN  (single-domain training)
# ═══════════════════════════════════════════════════════════════════════════════

class PhysicsInformedNN:
    """
    Standard (full-domain) PINN for the Buckley-Leverett equation.

    Training strategy (mirrors original):
      1. Optional Adam warm-up (niter_adam iterations).
      2. L-BFGS full optimisation.
    """

    def __init__(self,
                 X_u: np.ndarray, u: np.ndarray,
                 X_f: np.ndarray,
                 layers: list[int],
                 lb: np.ndarray, ub: np.ndarray):

        self.lb = torch.tensor(lb, dtype=torch.float32, device=DEVICE)
        self.ub = torch.tensor(ub, dtype=torch.float32, device=DEVICE)

        # data tensors
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=DEVICE)
        self.u   = torch.tensor(u,            dtype=torch.float32, device=DEVICE)

        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=DEVICE)

        self.model = MLP(layers, self.lb, self.ub).to(DEVICE)
        self.loss_history: list[float] = []

    # ── Adam warm-up ─────────────────────────────────────────────────────────
    def train_adam(self, n_iter: int = 5000, lr: float = 1e-3):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.model.train()
        for i in range(n_iter):
            optimizer.zero_grad()
            loss = compute_loss(self.model,
                                self.x_u, self.t_u, self.u,
                                self.x_f, self.t_f)
            loss.backward()
            optimizer.step()
            self.loss_history.append(loss.item())
            if i % 500 == 0:
                print(f"  [Adam] iter {i:5d}  loss = {loss.item():.4e}")

    # ── L-BFGS fine-tuning ───────────────────────────────────────────────────
    def train_lbfgs(self, max_iter: int = 5000):
        """
        PyTorch's built-in L-BFGS (full-batch).
        Equivalent to TF's ScipyOptimizerInterface with method='L-BFGS-B'.
        """
        optimizer = torch.optim.LBFGS(
            self.model.parameters(),
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        self.model.train()
        call_count = [0]

        def closure():
            optimizer.zero_grad()
            loss = compute_loss(self.model,
                                self.x_u, self.t_u, self.u,
                                self.x_f, self.t_f)
            loss.backward()
            call_count[0] += 1
            self.loss_history.append(loss.item())
            if call_count[0] % 100 == 0:
                print(f"  [L-BFGS] call {call_count[0]:5d}  loss = {loss.item():.4e}")
            return loss

        optimizer.step(closure)
        print(f"  [L-BFGS] done after {call_count[0]} function evaluations.")

    # ── Combined training (Adam → L-BFGS) ────────────────────────────────────
    def train(self, niter_adam: int = 5000, niter_lbfgs: int = 5000):
        print("=== Adam warm-up ===")
        if niter_adam > 0:
            self.train_adam(niter_adam)
        print("=== L-BFGS ===")
        self.train_lbfgs(niter_lbfgs)

    # ── Inference ────────────────────────────────────────────────────────────

    def predict(self, X_star: np.ndarray):
        self.model.eval()
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)

        #u_pred = self.model(x, t).cpu().numpy()
        u_pred = self.model(x, t).detach().cpu().numpy()
        # ✅ no grad needed here
        with torch.no_grad():
            u_pred = self.model(x, t).cpu().numpy()

        # residual (no grad needed for display)
        f_pred = pde_residual(self.model,
                              x.requires_grad_(True),
                              t.requires_grad_(True)).detach().cpu().numpy()
        return u_pred, f_pred


# ═══════════════════════════════════════════════════════════════════════════════
# 5.  Sequential (time-marching) PINN
# ═══════════════════════════════════════════════════════════════════════════════

class SequentialPINN:
    """
    Trains the PINN one time-window at a time.

    Strategy (inspired by Fuck & Tchelepi 2020 / Wight & Zhao 2020):
      - Divide [0, T] into n_windows equal sub-intervals.
      - In window k: IC = solution predicted at the right edge of window k-1.
      - Network weights are warm-started from the previous window.

    This greatly helps with the Buckley-Leverett shock because the sharp front
    is only present locally in each sub-domain.
    """

    def __init__(self,
                 X_f_all: np.ndarray,
                 layers: list[int],
                 lb: np.ndarray, ub: np.ndarray,
                 x_arr: np.ndarray,
                 n_windows: int = 5,
                 n_ic: int = 100):
        """
        Parameters
        ----------
        X_f_all   : collocation points for the full domain (N_f, 2)
        layers    : NN architecture, e.g. [2, 20, 20, 20, 20, 1]
        lb / ub   : domain bounds [x_min, t_min] / [x_max, t_max]
        x_arr     : 1-D array of spatial positions (for IC sampling)
        n_windows : number of sequential time windows
        n_ic      : number of spatial IC points per window
        """
        self.lb = lb
        self.ub = ub
        self.layers = layers
        self.x_arr = x_arr
        self.n_windows = n_windows
        self.n_ic = n_ic
        self.X_f_all = X_f_all

        # time breakpoints
        t_min, t_max = lb[1], ub[1]
        self.t_windows = np.linspace(t_min, t_max, n_windows + 1)

        self.models: list[PhysicsInformedNN] = []   # one per window
        self.loss_histories: list[list[float]] = []

    # ── IC for first window: S(x, 0) = 0 everywhere except x=0 → S=1 ──────
    @staticmethod
    def _make_initial_bc(x_arr: np.ndarray,
                         t_start: float,
                         bc_inlet_value: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Build IC/BC arrays for the start of a window."""
        # IC: t = t_start, S(x, t_start) = 0  (will be overridden for x=0)
        x_ic = x_arr[:, None]
        t_ic = np.full_like(x_ic, t_start)
        u_ic = np.zeros_like(x_ic)

        # left BC: x = 0, S = 1  (injection)
        t_bc = x_arr[:, None]  # reuse x_arr length for t values across window
        x_bc = np.zeros_like(t_bc)
        u_bc = np.ones_like(t_bc)

        X_u = np.hstack([np.vstack([x_ic, x_bc]),
                         np.vstack([t_ic, t_bc])])
        u   = np.vstack([u_ic, u_bc])
        return X_u, u

    def _predict_at_t(self, model: PhysicsInformedNN,
                      t_val: float) -> np.ndarray:
        """Return S(x, t_val) over all x positions (shape: (Nx, 1))."""
        X_query = np.hstack([self.x_arr[:, None],
                             np.full((len(self.x_arr), 1), t_val)])
        u_pred, _ = model.predict(X_query)
        return u_pred   # (Nx, 1)

    def train_sequential(self,
                         Exact_full: np.ndarray,
                         niter_adam: int = 3000,
                         niter_lbfgs: int = 3000):
        """
        Train window by window.

        Exact_full : shape (Nt, Nx) — only used for the left BC/IC in window 0.
        """
        t_arr = self.t_windows
        prev_S = None   # will hold predicted IC from previous window

        for k in range(self.n_windows):
            t0, t1 = t_arr[k], t_arr[k + 1]
            print(f"\n{'='*60}")
            print(f"  Window {k+1}/{self.n_windows}   t ∈ [{t0:.3f}, {t1:.3f}]")
            print(f"{'='*60}")

            lb_w = np.array([self.lb[0], t0])
            ub_w = np.array([self.ub[0], t1])

            # ── build IC/BC for this window ──────────────────────────────────
            if k == 0:
                # Window 0: true IC  S(x,0)=0, left BC S(0,t)=1
                # IC points
                x_ic  = self.x_arr[:, None]
                t_ic  = np.full_like(x_ic, t0)
                u_ic  = np.zeros_like(x_ic)
                # left BC points (spread over window time range)
                t_bc_vals = np.linspace(t0, t1, self.n_ic)[:, None]
                x_bc  = np.zeros_like(t_bc_vals)
                u_bc  = np.ones_like(t_bc_vals)
            else:
                # Window k>0: IC comes from predicted S at end of previous window
                x_ic = self.x_arr[:, None]
                t_ic = np.full_like(x_ic, t0)
                u_ic = prev_S   # predicted by previous model
                t_bc_vals = np.linspace(t0, t1, self.n_ic)[:, None]
                x_bc  = np.zeros_like(t_bc_vals)
                u_bc  = np.ones_like(t_bc_vals)

            X_u_win = np.hstack([np.vstack([x_ic, x_bc]),
                                 np.vstack([t_ic, t_bc_vals])])
            u_win   = np.vstack([u_ic, u_bc])

            # ── collocation points clipped to this window ────────────────────
            mask   = (self.X_f_all[:, 1] >= t0) & (self.X_f_all[:, 1] <= t1)
            X_f_w  = self.X_f_all[mask]
            if X_f_w.shape[0] < 100:
                # fall-back: re-sample within window
                X_f_w = lb_w + (ub_w - lb_w) * lhs(2, 2000)

            # ── build and train ──────────────────────────────────────────────
            pinn = PhysicsInformedNN(X_u_win, u_win, X_f_w,
                                     self.layers, lb_w, ub_w)

            # warm-start weights from previous window (transfer learning)
            if k > 0:
                prev_model = self.models[-1].model
                pinn.model.load_state_dict(prev_model.state_dict())

            pinn.train(niter_adam=niter_adam, niter_lbfgs=niter_lbfgs)
            self.models.append(pinn)
            self.loss_histories.append(pinn.loss_history)

            # ── predict IC for next window ───────────────────────────────────
            prev_S = self._predict_at_t(pinn, t1)

        print("\nSequential training complete.")

    def predict_full(self, X_star: np.ndarray) -> np.ndarray:
        """
        Predict S over the full domain by routing each point to the correct window.
        X_star : (N, 2) array of [x, t] query points.
        Returns: (N, 1) array of S predictions.
        """
        u_full = np.zeros((X_star.shape[0], 1))
        t_vals = X_star[:, 1]

        for k, pinn in enumerate(self.models):
            t0 = self.t_windows[k]
            t1 = self.t_windows[k + 1]
            # include right edge in last window
            if k < len(self.models) - 1:
                mask = (t_vals >= t0) & (t_vals < t1)
            else:
                mask = (t_vals >= t0) & (t_vals <= t1)
            if mask.any():
                u_part, _ = pinn.predict(X_star[mask])
                u_full[mask] = u_part

        return u_full


# ═══════════════════════════════════════════════════════════════════════════════
# 6.  Plotting helpers
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_data(X_u_train, Exact, x, t):
    plt.figure(figsize=(10, 5))
    plt.title("u(x,t) — training points")
    plt.plot(X_u_train[:, 1:2], X_u_train[:, 0:1], "xr", label="BC/IC points")
    h = plt.imshow(Exact.T, cmap="plasma",
                   extent=[t.min(), t.max(), x.min(), x.max()],
                   origin="lower", aspect="auto")
    plt.colorbar(h)
    plt.xlim([-0.1, 1.1]); plt.ylim([-0.1, 1.1])
    plt.xlabel("t"); plt.ylabel("x")
    plt.legend(); plt.tight_layout(); plt.show()


def plot_snapshots(x, t, Exact, U_pred,
                   sample_indices=None, title_prefix=""):
    if sample_indices is None:
        sample_indices = np.arange(0, Exact.shape[0], Exact.shape[0] // 8)
    for i in sample_indices:
        plt.figure(figsize=(6, 4))
        plt.grid()
        plt.title(f"{title_prefix}  t = {t[i]:.3f}")
        plt.plot(x, Exact[i, :], "-k", label="Exact")
        plt.plot(x, U_pred[i, :], "--r", label="PINN")
        plt.xlim([0, 1]); plt.ylim([-0.05, 1.05])
        plt.xlabel("x"); plt.ylabel("S (water saturation)")
        plt.legend(); plt.tight_layout(); plt.show()


def plot_loss(loss_history, title="Training loss"):
    plt.figure(figsize=(7, 4))
    plt.semilogy(loss_history)
    plt.title(title); plt.xlabel("Iteration"); plt.ylabel("Loss")
    plt.grid(True, which="both"); plt.tight_layout(); plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# 7.  Main script
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    # ── hyperparameters ──────────────────────────────────────────────────────
    N_u      = 300      # boundary/IC sample size
    N_f      = 10_000   # collocation points
    layers   = [2, 20, 20, 20, 20, 20, 20, 20, 20, 1]

    # toggle which training mode to run
    RUN_STANDARD   = True
    RUN_SEQUENTIAL = True

    # ── load data ────────────────────────────────────────────────────────────
    data  = np.load("PIML_data.npy")
    t     = np.linspace(0, 1, data.shape[1])
    x     = np.linspace(0, 1, data.shape[0])
    Exact = data.T   # shape (Nt, Nx)

    X, T  = np.meshgrid(x, t)
    X_temp = np.hstack((X.flatten()[:, None], T.flatten()[:, None]))
    u_temp = Exact.flatten()[:, None]

    lb = X_temp.min(0)
    ub = X_temp.max(0)

    # ── build IC/BC training set ─────────────────────────────────────────────
    # t = 0: S(x, 0) = 0
    xx1 = np.hstack((X[0:1, :].T, T[0:1, :].T))
    uu1 = Exact[0:1, :].T

    # x = 0, t > 0: S(0, t) = 1  (injection boundary)
    xx2 = np.hstack((X[1:, 0:1], T[1:, 0:1]))
    uu2 = Exact[1:, 0:1]

    X_u_train = np.vstack([xx1, xx2])
    u_train   = np.vstack([uu1, uu2])

    # random sub-sample
    idx       = np.random.choice(X_u_train.shape[0], N_u, replace=False)
    X_u_train = X_u_train[idx]
    u_train   = u_train[idx]

    # collocation points via Latin-hypercube sampling
    X_f_train = lb + (ub - lb) * lhs(2, N_f)

    plot_training_data(X_u_train, Exact, x, t)

    # ─────────────────────────────────────────────────────────────────────────
    # 7a.  Standard (full-domain) PINN
    # ─────────────────────────────────────────────────────────────────────────
    if RUN_STANDARD:
        print("\n" + "=" * 60)
        print("  STANDARD PINN (full-domain)")
        print("=" * 60)
        model = PhysicsInformedNN(X_u_train, u_train, X_f_train,
                                  layers, lb, ub)
        model.train(niter_adam=1000, niter_lbfgs=1000)

        u_pred_std, f_pred_std = model.predict(X_temp)

        error_std = (np.linalg.norm(u_temp - u_pred_std, 2) /
                     np.linalg.norm(u_temp, 2))
        print(f"\nStandard PINN  |  relative L2 error = {error_std:.4e}")

        U_pred_std = griddata(X_temp, u_pred_std.flatten(),
                              (X, T), method="linear")
        plot_snapshots(x, t, Exact, U_pred_std,
                       sample_indices=np.arange(0, len(t), max(1, len(t) // 8)),
                       title_prefix="Standard PINN")
        plot_loss(model.loss_history, title="Standard PINN — training loss")

    # ─────────────────────────────────────────────────────────────────────────
    # 7b.  Sequential (time-marching) PINN
    # ─────────────────────────────────────────────────────────────────────────
    if RUN_SEQUENTIAL:
        print("\n" + "=" * 60)
        print("  SEQUENTIAL PINN (time-marching)")
        print("=" * 60)
        seq_model = SequentialPINN(
            X_f_all   = X_f_train,
            layers    = layers,
            lb        = lb,
            ub        = ub,
            x_arr     = x,
            n_windows = 5,       # split [0,1] into 5 time windows
            n_ic      = 100
        )
        seq_model.train_sequential(Exact_full=Exact,
                                   niter_adam=3000,
                                   niter_lbfgs=3000)

        u_pred_seq = seq_model.predict_full(X_temp)
        error_seq = (np.linalg.norm(u_temp - u_pred_seq, 2) /
                     np.linalg.norm(u_temp, 2))
        print(f"\nSequential PINN  |  relative L2 error = {error_seq:.4e}")

        U_pred_seq = griddata(X_temp, u_pred_seq.flatten(),
                              (X, T), method="linear")
        plot_snapshots(x, t, Exact, U_pred_seq,
                       sample_indices=np.arange(0, len(t), max(1, len(t) // 8)),
                       title_prefix="Sequential PINN")

        # plot combined loss across all windows
        all_losses = []
        for h in seq_model.loss_histories:
            all_losses.extend(h)
        plot_loss(all_losses, title="Sequential PINN — combined training loss")

    # ─────────────────────────────────────────────────────────────────────────
    # 7c.  Side-by-side comparison (if both ran)
    # ─────────────────────────────────────────────────────────────────────────
    if RUN_STANDARD and RUN_SEQUENTIAL:
        sample_idx = np.arange(0, len(t), max(1, len(t) // 6))
        for i in sample_idx:
            fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
            for ax, U_pred, title in zip(
                    axes,
                    [U_pred_std, U_pred_seq],
                    ["Standard PINN", "Sequential PINN"]):
                ax.grid()
                ax.set_title(f"{title}  —  t = {t[i]:.3f}")
                ax.plot(x, Exact[i, :], "-k", linewidth=2, label="Exact")
                ax.plot(x, U_pred[i, :], "--r", linewidth=2, label="PINN")
                ax.set_xlim([0, 1]); ax.set_ylim([-0.05, 1.05])
                ax.set_xlabel("x"); ax.legend()
            axes[0].set_ylabel("S (water saturation)")
            plt.tight_layout(); plt.show()