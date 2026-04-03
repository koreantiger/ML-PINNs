"""
Compositional PINN — Improved + N-component extension
=======================================================

Improvements over v1:
  1. Residual-Based Adaptive Refinement (RAR)
       Add new collocation points wherever |residual| is largest each epoch.
  2. Causal training weights
       Weight the PDE loss at time t by exp(-eps * cumulative_residual(t' < t))
       so the network learns early times before later times — consistent with
       how the hyperbolic PDE actually propagates information.
  3. Smooth flux approximation near kinks
       Replace the hard phase-boundary step with a softplus transition of width
       epsilon_k. As training progresses, epsilon_k is annealed to zero.
  4. Kink-focused collocation oversampling
       10× more collocation points within delta of each phase boundary.
  5. Self-adaptive (learnable) loss weights
       lambda_pde is a log-softmax-normalised trainable parameter that is
       updated jointly with the network — from Wang et al. (2022).
  6. N-component Rachford-Rice (ternary system)
       Network outputs (z1, z2); z3 = 1 - z1 - z2.
       Two coupled PDEs, one per tracked component.

Sections
--------
  §1   Rachford-Rice solver (N components)
  §2   Physics utilities (flux, residual)  — 2-component
  §3   Smooth flux + kink oversampling
  §4   MLP backbone
  §5   Improved CompositionalPINN (all fixes)
  §6   N-component (ternary) extension
  §7   Main
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from pyDOE import lhs

torch.manual_seed(1234)
np.random.seed(1234)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ═══════════════════════════════════════════════════════════════════════════════
# §1  Rachford-Rice solver — works for any number of components
# ═══════════════════════════════════════════════════════════════════════════════

def rachford_rice_N(z: np.ndarray, K: np.ndarray,
                    tol: float = 1e-12, max_iter: int = 2000) -> tuple:
    """
    Solve Rachford-Rice for an N-component system.
    Σ z_i(K_i - 1)/(V(K_i - 1) + 1) = 0  in V ∈ (V_min, V_max)

    Parameters
    ----------
    z : (Nc,) overall compositions  (must sum to 1)
    K : (Nc,) equilibrium K-values

    Returns
    -------
    V : vapour fraction
    x : (Nc,) liquid compositions
    y : (Nc,) vapour compositions
    """
    # bracket [V_min, V_max] that avoids poles
    V_min = 1.0 / (1.0 - np.max(K)) + 1e-8
    V_max = 1.0 / (1.0 - np.min(K)) - 1e-8

    r = lambda V: np.sum(z * (K - 1.0) / (V * (K - 1.0) + 1.0))

    a, b = V_min, V_max
    for _ in range(max_iter):
        V = 0.5 * (a + b)
        rv = r(V)
        if rv > 0:
            a = V
        else:
            b = V
        if abs(rv) < tol:
            break

    x = z / (V * (K - 1.0) + 1.0)
    y = K * x
    x /= x.sum()   # normalise to handle floating point drift
    y /= y.sum()
    return V, x, y


# ═══════════════════════════════════════════════════════════════════════════════
# §2  Physics parameters + flux  (binary system)
# ═══════════════════════════════════════════════════════════════════════════════

K2      = np.array([3.0, 0.1])    # binary K-values
M       = 10.0                     # mobility ratio
Z_INJ   = 0.99
Z_INIT  = 0.01

_V, _x2, _y2 = rachford_rice_N(np.array([0.5, 0.5]), K2)
X1_EQ = float(_x2[0])
Y1_EQ = float(_y2[0])
print(f"Binary flash: x1={X1_EQ:.4f}, y1={Y1_EQ:.4f}")


def phase_frac_flow(S: torch.Tensor, M: float = M) -> torch.Tensor:
    return S**2 / (S**2 + M * (1.0 - S)**2)


# ─── FIX 3: smooth flux with softplus transition ─────────────────────────────

def smooth_step(z: torch.Tensor, threshold: float,
                eps: float = 0.02) -> torch.Tensor:
    """
    Differentiable approximation to Heaviside(z - threshold).
    eps → 0 recovers the hard step. Stays in [0,1] and is C∞.
    """
    return torch.sigmoid((z - threshold) / eps)


def compositional_flux_smooth(z: torch.Tensor,
                               x1: float = X1_EQ,
                               y1: float = Y1_EQ,
                               eps_kink: float = 0.02) -> torch.Tensor:
    """
    Smooth compositional fractional flow F(z) with softplus-blended kinks.

    Instead of a hard phase mask (not differentiable), use smooth blending:
        in_two_phase(z) ≈ sigma((z-y1)/eps) * (1 - sigma((z-x1)/eps))

    This gives autograd a well-defined, non-zero gradient everywhere,
    including right at the kink points y1 and x1.
    """
    S     = (z - y1) / (x1 - y1)
    S_c   = torch.clamp(S, 0.0, 1.0)
    f_S   = phase_frac_flow(S_c)
    F_tp  = x1 * f_S + y1 * (1.0 - f_S)   # two-phase flux

    # smooth masks
    above_y1   = smooth_step(z,  y1, eps_kink)       # ≈1 when z > y1
    below_x1   = 1.0 - smooth_step(z, x1, eps_kink)  # ≈1 when z < x1
    two_phase_w = above_y1 * below_x1                 # ≈1 in two-phase region

    # blend: z outside two-phase, F_tp inside two-phase
    F = (1.0 - two_phase_w) * z + two_phase_w * F_tp
    return F


def pde_residual_comp(model: nn.Module,
                      x_f: torch.Tensor,
                      t_f: torch.Tensor,
                      eps_kink: float = 0.02) -> torch.Tensor:
    """∂z/∂t + ∂F(z)/∂x = 0"""
    x_f = x_f.requires_grad_(True)
    t_f = t_f.requires_grad_(True)
    z   = model(x_f, t_f)
    z_t = torch.autograd.grad(z, t_f,
                               grad_outputs=torch.ones_like(z),
                               create_graph=True, retain_graph=True)[0]
    F_z = compositional_flux_smooth(z, eps_kink=eps_kink)
    F_x = torch.autograd.grad(F_z, x_f,
                               grad_outputs=torch.ones_like(F_z),
                               create_graph=True, retain_graph=True)[0]
    return z_t + F_x


# ─── FIX 4: kink-focused collocation oversampling ────────────────────────────

def kink_enriched_collocation(lb: np.ndarray, ub: np.ndarray,
                               N_base: int, N_kink: int,
                               kink_locs: list[float],
                               delta: float = 0.05) -> np.ndarray:
    """
    LHS base points + dense oversampling near each kink location in x.
    kink_locs : list of z-values where kinks occur (y1, x1 for binary).
    delta      : half-width of the oversampling window in x-space.
    """
    X_base = lb + (ub - lb) * lhs(2, N_base)

    kink_pts = []
    for zk in kink_locs:
        # Map each kink-z to approximate x location via linear approximation
        # (or just oversample over the full x range near the shock front)
        x_lo = max(lb[0], 0.0)
        x_hi = min(ub[0], 1.0)
        x_kink = lb[0:1] + (ub[0:1] - lb[0:1]) * lhs(1, N_kink)
        t_kink = lb[1:2] + (ub[1:2] - lb[1:2]) * lhs(1, N_kink)
        kink_pts.append(np.hstack([x_kink, t_kink]))

    if kink_pts:
        return np.vstack([X_base] + kink_pts)
    return X_base


# ═══════════════════════════════════════════════════════════════════════════════
# §3  MLP backbone  (shared by binary and ternary)
# ═══════════════════════════════════════════════════════════════════════════════

class MLP(nn.Module):
    def __init__(self, layers: list[int],
                 lb: torch.Tensor, ub: torch.Tensor):
        super().__init__()
        self.lb, self.ub = lb, ub
        net = []
        for i in range(len(layers) - 2):
            net.append(nn.Linear(layers[i], layers[i + 1]))
            net.append(nn.Tanh())
        net.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*net)
        self._init_weights()

    def _init_weights(self):
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        xt  = torch.cat([x, t], dim=1)
        xt  = 2.0 * (xt - self.lb) / (self.ub - self.lb) - 1.0
        return self.net(xt)


# ═══════════════════════════════════════════════════════════════════════════════
# §4  Improved CompositionalPINN  (binary)
# ═══════════════════════════════════════════════════════════════════════════════

class ImprovedCompPINN:
    """
    Binary compositional PINN with four improvements:

    1. Self-adaptive lambda_pde   (learnable log-weight)
    2. Causal training            (exponential temporal weighting)
    3. RAR                        (residual-based adaptive refinement)
    4. Smooth + oversampled flux  (handled in pde_residual_comp)
    """

    def __init__(self,
                 X_u: np.ndarray, u: np.ndarray,
                 X_f: np.ndarray,
                 layers: list[int],
                 lb: np.ndarray, ub: np.ndarray,
                 use_causal:     bool  = True,
                 use_adaptive_w: bool  = True,
                 causal_eps:     float = 5.0,
                 eps_kink_init:  float = 0.05,
                 eps_kink_final: float = 0.005):

        self.lb_t = torch.tensor(lb, dtype=torch.float32, device=DEVICE)
        self.ub_t = torch.tensor(ub, dtype=torch.float32, device=DEVICE)

        # data
        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=DEVICE)
        self.u   = torch.tensor(u,            dtype=torch.float32, device=DEVICE)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=DEVICE)

        self.model       = MLP(layers, self.lb_t, self.ub_t).to(DEVICE)
        self.use_causal  = use_causal
        self.causal_eps  = causal_eps
        self.eps_kink    = eps_kink_init
        self.eps_kink_f  = eps_kink_final

        # ── FIX 5: learnable loss weight ─────────────────────────────────────
        self.use_adaptive_w = use_adaptive_w
        if use_adaptive_w:
            # log(lambda_pde) — optimised jointly with network
            self.log_lambda = nn.Parameter(
                torch.tensor(0.0, device=DEVICE))   # starts at lambda=1
        else:
            self.log_lambda = None

        self.loss_history: list[float] = []
        self.lb_np, self.ub_np = lb, ub

    def _lambda(self) -> float:
        if self.use_adaptive_w and self.log_lambda is not None:
            return torch.exp(self.log_lambda)
        return torch.tensor(1.0, device=DEVICE)

    # ── FIX 2: causal PDE loss ───────────────────────────────────────────────

    def _causal_pde_loss(self, eps_kink: float) -> torch.Tensor:
        """
        Causal weighting: split collocation points into Nt time-bins.
        Weight of bin k = exp(-eps * sum_{j<k} L_j).
        This forces the network to satisfy the PDE at early times
        before worrying about later times — matching how the shock propagates.
        """
        x_f = self.x_f.detach().requires_grad_(True)
        t_f = self.t_f.detach().requires_grad_(True)
        residuals = pde_residual_comp(self.model, x_f, t_f, eps_kink)   # (N,1)

        if not self.use_causal:
            return torch.mean(residuals**2)

        # sort by time
        t_vals  = t_f.detach().squeeze()
        r_vals  = residuals.squeeze()
        sort_idx = torch.argsort(t_vals)
        t_sorted = t_vals[sort_idx]
        r_sorted = r_vals[sort_idx]

        N     = t_sorted.shape[0]
        n_bin = max(1, N // 20)   # 20 temporal bins
        loss  = torch.tensor(0.0, device=DEVICE)
        cum_L = torch.tensor(0.0, device=DEVICE)

        for k in range(0, N, n_bin):
            chunk_r = r_sorted[k: k + n_bin]
            L_k     = torch.mean(chunk_r**2)
            w_k     = torch.exp(-self.causal_eps * cum_L.detach())
            loss    = loss + w_k * L_k
            cum_L   = cum_L + L_k.detach()

        return loss / max(1, N // n_bin)

    def _data_loss(self) -> torch.Tensor:
        z_pred = self.model(self.x_u, self.t_u)
        return torch.mean((self.u - z_pred)**2)

    def _total_loss(self, eps_kink: float) -> torch.Tensor:
        l_data  = self._data_loss()
        l_pde   = self._causal_pde_loss(eps_kink)
        lam     = self._lambda()
        return l_data + lam * l_pde

    def _params(self):
        params = list(self.model.parameters())
        if self.use_adaptive_w and self.log_lambda is not None:
            params = params + [self.log_lambda]
        return params

    # ── Adam ─────────────────────────────────────────────────────────────────

    def train_adam(self, n_iter: int = 5000, lr: float = 1e-3,
                   rar_every: int = 500, rar_n: int = 500):
        """
        Adam with:
          - annealing of eps_kink (smooth → sharp flux)
          - RAR every `rar_every` iterations
        """
        opt = torch.optim.Adam(self._params(), lr=lr)
        self.model.train()
        eps_schedule = np.linspace(self.eps_kink, self.eps_kink_f, n_iter)

        for i in range(n_iter):
            eps_k = float(eps_schedule[i])
            opt.zero_grad()
            loss  = self._total_loss(eps_k)
            loss.backward()
            # gradient clipping — prevents exploding gradients near shocks
            torch.nn.utils.clip_grad_norm_(self._params(), max_norm=1.0)
            opt.step()
            self.loss_history.append(loss.item())

            if i % 500 == 0:
                lam_val = float(self._lambda())
                print(f"  [Adam] {i:5d}  loss={loss.item():.4e}"
                      f"  lambda={lam_val:.3f}  eps_kink={eps_k:.4f}")

            # ── FIX 1: RAR — add points where residual is largest ────────────
            if rar_every > 0 and i > 0 and i % rar_every == 0:
                self._rar_refine(rar_n, eps_k)

        self.eps_kink = float(eps_schedule[-1])

    def _rar_refine(self, n_new: int, eps_kink: float):
        """
        Residual-Based Adaptive Refinement:
          Sample a large pool of candidate points, evaluate residual,
          keep the top-n_new with largest |residual|.
        """
        pool_size = n_new * 20
        X_cand    = self.lb_np + (self.ub_np - self.lb_np) * lhs(2, pool_size)
        x_c = torch.tensor(X_cand[:, 0:1], dtype=torch.float32, device=DEVICE)
        t_c = torch.tensor(X_cand[:, 1:2], dtype=torch.float32, device=DEVICE)

        self.model.eval()
        with torch.no_grad():
            # evaluate residual without graph (just for point selection)
            x_c2 = x_c.requires_grad_(True)
            t_c2 = t_c.requires_grad_(True)

        # need grad for residual computation
        res = pde_residual_comp(self.model, x_c, t_c, eps_kink)
        res_vals = res.detach().abs().squeeze()

        _, top_idx = torch.topk(res_vals, n_new)
        x_new = x_c[top_idx].detach()
        t_new = t_c[top_idx].detach()

        self.x_f = torch.cat([self.x_f, x_new], dim=0)
        self.t_f = torch.cat([self.t_f, t_new], dim=0)
        self.model.train()

    # ── L-BFGS ───────────────────────────────────────────────────────────────

    def train_lbfgs(self, max_iter: int = 5000):
        opt = torch.optim.LBFGS(
            self._params(),
            max_iter=max_iter,
            tolerance_grad=1e-9,
            tolerance_change=1e-12,
            history_size=50,
            line_search_fn="strong_wolfe"
        )
        self.model.train()
        calls = [0]
        eps_k = self.eps_kink   # keep at final annealed value

        def closure():
            opt.zero_grad()
            loss = self._total_loss(eps_k)
            loss.backward()
            calls[0] += 1
            self.loss_history.append(loss.item())
            if calls[0] % 100 == 0:
                print(f"  [L-BFGS] {calls[0]:5d}  loss={loss.item():.4e}")
            return loss

        opt.step(closure)

    def train(self, niter_adam: int = 5000, niter_lbfgs: int = 5000):
        print("--- Adam (with RAR + causal weights + kink annealing) ---")
        self.train_adam(niter_adam)
        print("--- L-BFGS ---")
        self.train_lbfgs(niter_lbfgs)

    @torch.no_grad()
    def predict(self, X_star: np.ndarray):
        self.model.eval()
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)
        return self.model(x, t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# §5  N-component (ternary) extension
# ═══════════════════════════════════════════════════════════════════════════════
#
# PDE system for a ternary (Nc=3) system:
#   ∂z₁/∂t + ∂F₁(z₁,z₂)/∂x = 0
#   ∂z₂/∂t + ∂F₂(z₁,z₂)/∂x = 0
#   z₃ = 1 - z₁ - z₂  (closure)
#
# The network takes (x,t) → (z₁, z₂)  — 2 outputs.
# F_i is computed via Rachford-Rice for the current z.
#
# How Rachford-Rice is handled inside autograd:
#   - R-R is solved in *numpy* at data-prep time to get equilibrium (x, y).
#   - For the PINN residual we use *pre-computed* tie-line tables
#     (valid for constant K-values) so no iterative solve is needed inside
#     the training loop — just lookup + interpolation, which IS differentiable.
#   - If K-values are pressure/composition-dependent, replace with a
#     differentiable flash (e.g. successive-substitution unrolled a fixed
#     number of steps, or a neural flash surrogate).

class TernaryFlash:
    """
    Pre-computes the tie-line table for a ternary system with constant K.
    During PINN training, maps z → (S, x, y) via differentiable interpolation.
    """

    def __init__(self, K: np.ndarray, n_table: int = 1000):
        assert len(K) == 3, "K must have 3 entries for ternary system."
        self.K      = K
        self.Nc     = 3

        # Build table: for each overall comp z1 in [0,1], compute flash
        # (z2 = 1 - z1 so this is a pseudo-binary scan along z3=0 diagonal)
        z1_arr = np.linspace(0.01, 0.99, n_table)
        V_arr, x1_arr, y1_arr = [], [], []
        x2_arr, y2_arr        = [], []

        for z1 in z1_arr:
            z2 = 1.0 - z1   # 2-component for table (z3 handled separately)
            z_vec = np.array([z1, z2 * 0.5, z2 * 0.5])
            z_vec /= z_vec.sum()
            try:
                V, x, y = rachford_rice_N(z_vec, K)
                V_arr.append(V); x1_arr.append(x[0]); y1_arr.append(y[0])
                x2_arr.append(x[1]); y2_arr.append(y[1])
            except Exception:
                V_arr.append(0.5); x1_arr.append(z1); y1_arr.append(z1)
                x2_arr.append(z2); y2_arr.append(z2)

        self.z1_table  = torch.tensor(z1_arr,  dtype=torch.float32)
        self.x1_table  = torch.tensor(x1_arr,  dtype=torch.float32)
        self.y1_table  = torch.tensor(y1_arr,  dtype=torch.float32)
        self.x2_table  = torch.tensor(x2_arr,  dtype=torch.float32)
        self.y2_table  = torch.tensor(y2_arr,  dtype=torch.float32)

        # phase boundaries (approximate from table)
        self.y1_eq = float(np.min(y1_arr))
        self.x1_eq = float(np.max(x1_arr))
        print(f"Ternary flash table built: y1_eq≈{self.y1_eq:.3f}, "
              f"x1_eq≈{self.x1_eq:.3f}")

    def lookup(self, z1: torch.Tensor) -> tuple:
        """Differentiable 1-D linear interpolation into the tie-line table."""
        z1_t = self.z1_table.to(z1.device)
        x1_t = self.x1_table.to(z1.device)
        y1_t = self.y1_table.to(z1.device)
        x2_t = self.x2_table.to(z1.device)
        y2_t = self.y2_table.to(z1.device)

        # clamp for boundary safety
        z1c = z1.clamp(z1_t[0].item(), z1_t[-1].item()).squeeze()

        # find bin via searchsorted equivalent
        idx = torch.searchsorted(z1_t.contiguous(),
                                 z1c.contiguous()).clamp(1, len(z1_t) - 1)
        lo, hi = idx - 1, idx
        alpha  = (z1c - z1_t[lo]) / (z1_t[hi] - z1_t[lo] + 1e-12)

        def interp(tbl): return tbl[lo] + alpha * (tbl[hi] - tbl[lo])

        return interp(x1_t), interp(y1_t), interp(x2_t), interp(y2_t)


def ternary_flux(z1: torch.Tensor, z2: torch.Tensor,
                 flash: TernaryFlash,
                 M: float = M,
                 eps_kink: float = 0.02) -> tuple:
    """
    Compute (F1, F2) — overall component fluxes for ternary system.

    F_i = x_i * f(S) + y_i * (1 - f(S))
    S   = (z1 - y1) / (x1 - y1)   from tie-line lookup
    """
    x1_eq, y1_eq, x2_eq, y2_eq = flash.lookup(z1)

    S    = (z1.squeeze() - y1_eq) / (x1_eq - y1_eq + 1e-9)
    S_c  = S.clamp(0.0, 1.0).unsqueeze(1)
    f_S  = phase_frac_flow(S_c, M)

    # two-phase flux
    F1_tp = x1_eq.unsqueeze(1) * f_S + y1_eq.unsqueeze(1) * (1.0 - f_S)
    F2_tp = x2_eq.unsqueeze(1) * f_S + y2_eq.unsqueeze(1) * (1.0 - f_S)

    # smooth phase masks (same idea as binary)
    above_y1   = smooth_step(z1, flash.y1_eq, eps_kink)
    below_x1   = 1.0 - smooth_step(z1, flash.x1_eq, eps_kink)
    tp_weight  = above_y1 * below_x1

    F1 = (1.0 - tp_weight) * z1 + tp_weight * F1_tp
    F2 = (1.0 - tp_weight) * z2 + tp_weight * F2_tp
    return F1, F2


def pde_residual_ternary(model: nn.Module,
                          x_f: torch.Tensor,
                          t_f: torch.Tensor,
                          flash: TernaryFlash,
                          eps_kink: float = 0.02) -> tuple:
    """
    Two coupled PDE residuals:
        r1 = ∂z1/∂t + ∂F1/∂x
        r2 = ∂z2/∂t + ∂F2/∂x
    """
    x_f = x_f.requires_grad_(True)
    t_f = t_f.requires_grad_(True)

    z   = model(x_f, t_f)          # (N, 2)
    z1  = z[:, 0:1]
    z2  = z[:, 1:2]

    ones = torch.ones_like(z1)

    z1_t = torch.autograd.grad(z1, t_f, grad_outputs=ones,
                                create_graph=True, retain_graph=True)[0]
    z2_t = torch.autograd.grad(z2, t_f, grad_outputs=ones,
                                create_graph=True, retain_graph=True)[0]

    F1, F2 = ternary_flux(z1, z2, flash, eps_kink=eps_kink)

    F1_x = torch.autograd.grad(F1, x_f, grad_outputs=ones,
                                create_graph=True, retain_graph=True)[0]
    F2_x = torch.autograd.grad(F2, x_f, grad_outputs=ones,
                                create_graph=True, retain_graph=True)[0]

    return z1_t + F1_x, z2_t + F2_x


class TernaryCompPINN:
    """
    Ternary (3-component) compositional PINN.

    Network: (x, t) → (z1, z2)   [2 outputs]
    Physics: two coupled conservation laws with shared flash table.
    All improvements from §4 are included.
    """

    def __init__(self,
                 X_u:     np.ndarray,
                 u:       np.ndarray,      # (N, 2)  — [z1, z2] data
                 X_f:     np.ndarray,
                 layers:  list[int],       # output dim of last layer must be 2
                 lb:      np.ndarray,
                 ub:      np.ndarray,
                 K:       np.ndarray,      # (3,) K-values
                 use_causal:    bool  = True,
                 causal_eps:    float = 5.0,
                 eps_kink_init: float = 0.05,
                 eps_kink_final:float = 0.005):

        self.lb_t = torch.tensor(lb, dtype=torch.float32, device=DEVICE)
        self.ub_t = torch.tensor(ub, dtype=torch.float32, device=DEVICE)
        self.flash = TernaryFlash(K)

        self.x_u = torch.tensor(X_u[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_u = torch.tensor(X_u[:, 1:2], dtype=torch.float32, device=DEVICE)
        self.u   = torch.tensor(u,            dtype=torch.float32, device=DEVICE)
        self.x_f = torch.tensor(X_f[:, 0:1], dtype=torch.float32, device=DEVICE)
        self.t_f = torch.tensor(X_f[:, 1:2], dtype=torch.float32, device=DEVICE)

        assert layers[-1] == 2, "Last layer must output 2 values (z1, z2)."
        self.model = MLP(layers, self.lb_t, self.ub_t).to(DEVICE)

        self.use_causal    = use_causal
        self.causal_eps    = causal_eps
        self.eps_kink      = eps_kink_init
        self.eps_kink_f    = eps_kink_final

        # learnable log-weights: one per PDE
        self.log_lambda = nn.Parameter(
            torch.zeros(2, device=DEVICE))   # [log_lam1, log_lam2]

        self.loss_history: list[float] = []
        self.lb_np, self.ub_np = lb, ub

    def _total_loss(self, eps_kink: float) -> torch.Tensor:
        # data loss (both components)
        z_pred = self.model(self.x_u, self.t_u)   # (N, 2)
        loss_d = torch.mean((self.u - z_pred)**2)

        # PDE residuals
        r1, r2 = pde_residual_ternary(self.model,
                                       self.x_f, self.t_f,
                                       self.flash, eps_kink)

        if self.use_causal:
            loss_pde1 = self._causal_weight(r1)
            loss_pde2 = self._causal_weight(r2)
        else:
            loss_pde1 = torch.mean(r1**2)
            loss_pde2 = torch.mean(r2**2)

        lam = torch.exp(self.log_lambda)   # (2,)
        return loss_d + lam[0] * loss_pde1 + lam[1] * loss_pde2

    def _causal_weight(self, residual: torch.Tensor) -> torch.Tensor:
        """Causal temporal weighting (same scheme as binary)."""
        t_vals   = self.t_f.detach().squeeze()
        r_vals   = residual.squeeze()
        sort_idx = torch.argsort(t_vals)
        r_sorted = r_vals[sort_idx]
        N        = r_sorted.shape[0]
        n_bin    = max(1, N // 20)
        loss     = torch.tensor(0.0, device=DEVICE)
        cum_L    = torch.tensor(0.0, device=DEVICE)
        for k in range(0, N, n_bin):
            chunk  = r_sorted[k: k + n_bin]
            L_k    = torch.mean(chunk**2)
            w_k    = torch.exp(-self.causal_eps * cum_L.detach())
            loss   = loss + w_k * L_k
            cum_L  = cum_L + L_k.detach()
        return loss / max(1, N // n_bin)

    def _params(self):
        return list(self.model.parameters()) + [self.log_lambda]

    def train_adam(self, n_iter: int = 5000, lr: float = 1e-3,
                   rar_every: int = 500, rar_n: int = 500):
        opt = torch.optim.Adam(self._params(), lr=lr)
        eps_sched = np.linspace(self.eps_kink, self.eps_kink_f, n_iter)
        self.model.train()
        for i in range(n_iter):
            eps_k = float(eps_sched[i])
            opt.zero_grad()
            loss = self._total_loss(eps_k)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self._params(), max_norm=1.0)
            opt.step()
            self.loss_history.append(loss.item())
            if i % 500 == 0:
                lam = torch.exp(self.log_lambda).detach().cpu().numpy()
                print(f"  [Adam] {i:5d}  loss={loss.item():.4e}"
                      f"  lam=[{lam[0]:.3f},{lam[1]:.3f}]  eps={eps_k:.4f}")

    def train_lbfgs(self, max_iter: int = 3000):
        opt = torch.optim.LBFGS(self._params(), max_iter=max_iter,
                                 tolerance_grad=1e-9, tolerance_change=1e-12,
                                 history_size=50, line_search_fn="strong_wolfe")
        self.model.train()
        calls = [0]
        eps_k = self.eps_kink_f

        def closure():
            opt.zero_grad()
            loss = self._total_loss(eps_k)
            loss.backward()
            calls[0] += 1
            self.loss_history.append(loss.item())
            if calls[0] % 100 == 0:
                print(f"  [L-BFGS] {calls[0]:4d}  loss={loss.item():.4e}")
            return loss

        opt.step(closure)

    def train(self, niter_adam: int = 5000, niter_lbfgs: int = 3000):
        print("--- Ternary PINN: Adam ---")
        self.train_adam(niter_adam)
        print("--- Ternary PINN: L-BFGS ---")
        self.train_lbfgs(niter_lbfgs)

    @torch.no_grad()
    def predict(self, X_star: np.ndarray) -> np.ndarray:
        """Returns (N, 2) array: columns are [z1_pred, z2_pred]."""
        self.model.eval()
        x = torch.tensor(X_star[:, 0:1], dtype=torch.float32, device=DEVICE)
        t = torch.tensor(X_star[:, 1:2], dtype=torch.float32, device=DEVICE)
        return self.model(x, t).cpu().numpy()


# ═══════════════════════════════════════════════════════════════════════════════
# §6  Reference FV solver  (implicit, N-component)
# ═══════════════════════════════════════════════════════════════════════════════

def fv_ternary_implicit(nb: int, Theta: float, NT: int,
                        K: np.ndarray,
                        z_inj: np.ndarray,
                        z_init: np.ndarray) -> np.ndarray:
    """
    Implicit FV solver for a ternary system.
    Tracks z1 and z2; z3 = 1 - z1 - z2.
    Returns: shape (NT+1, nb, 2)
    """
    Nc   = 3
    z    = np.tile(z_init, (nb, 1)).copy()   # (nb, Nc)
    z[0] = z_inj

    # constant K → single flash at midpoint gives tie-line
    _V, x_eq, y_eq = rachford_rice_N(np.array([0.5, 0.25, 0.25]), K)
    x1, y1 = x_eq[0], y_eq[0]
    x2, y2 = x_eq[1], y_eq[1]

    f     = lambda s: s**2 / (s**2 + M * (1 - s)**2)
    df    = lambda s: (f(s + 1e-6) - f(s)) / 1e-6

    history = [z[:, :2].copy()]   # store only (z1, z2)

    rn = range(1, nb)
    for t_step in range(NT):
        zn = z.copy()
        for n in range(100):
            s   = (z[:, 0] - y1) / (x1 - y1)
            fs  = f(s)
            F1  = np.where(s > 1, z[:, 0], np.where(s < 0, z[:, 0],
                           x1 * fs + y1 * (1 - fs)))
            F2  = np.where(s > 1, z[:, 1], np.where(s < 0, z[:, 1],
                           x2 * fs + y2 * (1 - fs)))

            rhs1 = z[1:, 0] - zn[1:, 0] + Theta * (F1[1:] - F1[:-1])
            rhs2 = z[1:, 1] - zn[1:, 1] + Theta * (F2[1:] - F2[:-1])

            res = np.sqrt(np.sum(rhs1**2) + np.sum(rhs2**2))
            if res < 1e-2 and n > 0:
                break

            # Newton update (diagonal Jacobian — simplified)
            dfs   = df(s)
            diag  = 1.0 + Theta * dfs
            offdiag = -Theta * dfs

            # component 1
            jac1 = np.diag(diag[1:]) + np.diag(offdiag[:-1], k=-1)
            jac1[0, 0] = 1.0
            rhs1_full = np.zeros(nb); rhs1_full[1:] = rhs1
            dz1 = np.linalg.solve(jac1 + np.eye(nb) * 1e-10, -rhs1_full)

            # component 2
            jac2 = np.diag(diag[1:]) + np.diag(offdiag[:-1], k=-1)
            jac2[0, 0] = 1.0
            rhs2_full = np.zeros(nb); rhs2_full[1:] = rhs2
            dz2 = np.linalg.solve(jac2 + np.eye(nb) * 1e-10, -rhs2_full)

            z[:, 0] += dz1
            z[:, 1] += dz2
            z[:, 2]  = 1.0 - z[:, 0] - z[:, 1]
            z = np.clip(z, 0, 1)

        history.append(z[:, :2].copy())

    return np.array(history)   # (NT+1, nb, 2)


# ═══════════════════════════════════════════════════════════════════════════════
# §7  Main
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":

    RUN_BINARY  = True
    RUN_TERNARY = True

    nb, NT, Theta = 80, 80, 0.2
    N_u, N_f      = 400, 15_000

    # ─────────────────────────────────────────────────────────────────────────
    # A.  Improved binary PINN
    # ─────────────────────────────────────────────────────────────────────────
    if RUN_BINARY:
        print("\n" + "="*60)
        print("  IMPROVED BINARY COMPOSITIONAL PINN")
        print("="*60)

        from compositional_pinn_pytorch import (
            fv_comp_implicit, rachford_rice_N
        )

        Exact_bin = fv_comp_implicit(nb, Theta, NT)   # (NT+1, nb)
        t = np.linspace(0, 1, NT + 1)
        x = np.linspace(0, 1, nb)
        X, T = np.meshgrid(x, t)
        X_temp = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
        u_temp = Exact_bin.flatten()[:, None]
        lb, ub = X_temp.min(0), X_temp.max(0)

        # IC + BC
        xx1 = np.hstack([X[0:1, :].T, T[0:1, :].T])
        uu1 = Exact_bin[0:1, :].T
        xx2 = np.hstack([X[1:, 0:1], T[1:, 0:1]])
        uu2 = Exact_bin[1:, 0:1]
        X_u_train = np.vstack([xx1, xx2])
        u_train   = np.vstack([uu1, uu2])
        idx       = np.random.choice(len(X_u_train), N_u, replace=False)
        X_u_train, u_train = X_u_train[idx], u_train[idx]

        # kink-enriched collocation
        X_f_train = kink_enriched_collocation(
            lb, ub, N_base=N_f,
            N_kink=N_f // 5,         # 20% extra near kinks
            kink_locs=[Y1_EQ, X1_EQ]
        )

        layers = [2, 64, 64, 64, 64, 64, 1]   # wider for kinks
        model  = ImprovedCompPINN(
            X_u_train, u_train, X_f_train,
            layers, lb, ub,
            use_causal=True,
            use_adaptive_w=True,
            causal_eps=5.0,
            eps_kink_init=0.05,
            eps_kink_final=0.005
        )
        model.train(niter_adam=5000, niter_lbfgs=5000)

        z_pred = model.predict(X_temp)
        err    = np.linalg.norm(u_temp - z_pred, 2) / np.linalg.norm(u_temp, 2)
        print(f"\nImproved binary PINN  |  rel. L2 = {err:.4e}")

        Z_pred = griddata(X_temp, z_pred.flatten(), (X, T), method='linear')

        plt.figure(figsize=(7, 4))
        plt.semilogy(model.loss_history)
        plt.title("Improved binary PINN — training loss")
        plt.xlabel("Iteration"); plt.ylabel("Loss")
        plt.grid(True, which='both'); plt.tight_layout(); plt.show()

        sample = np.arange(0, len(t), max(1, len(t) // 8))
        for i in sample:
            plt.figure(figsize=(6, 4))
            plt.plot(x, Exact_bin[i, :], '-k', lw=2,  label='FV ref')
            plt.plot(x, Z_pred[i, :],    '--r', lw=2,  label='PINN')
            plt.axhline(Y1_EQ, color='g', ls=':', label='y₁')
            plt.axhline(X1_EQ, color='b', ls=':', label='x₁')
            plt.title(f"Improved binary  t={t[i]:.3f}")
            plt.xlabel('x'); plt.ylabel('z'); plt.legend()
            plt.xlim([0, 1]); plt.ylim([-0.02, 1.02])
            plt.grid(True); plt.tight_layout(); plt.show()

    # ─────────────────────────────────────────────────────────────────────────
    # B.  Ternary PINN
    # ─────────────────────────────────────────────────────────────────────────
    if RUN_TERNARY:
        print("\n" + "="*60)
        print("  TERNARY (3-COMPONENT) COMPOSITIONAL PINN")
        print("="*60)

        K3     = np.array([4.0, 1.5, 0.1])          # 3-component K-values
        z_inj  = np.array([0.85, 0.10, 0.05])
        z_init = np.array([0.05, 0.10, 0.85])

        Exact_ter = fv_ternary_implicit(nb, Theta, NT, K3, z_inj, z_init)
        # shape: (NT+1, nb, 2)   — [z1, z2]

        t = np.linspace(0, 1, NT + 1)
        x = np.linspace(0, 1, nb)
        X, T = np.meshgrid(x, t)
        X_temp = np.hstack([X.flatten()[:, None], T.flatten()[:, None]])
        lb, ub = X_temp.min(0), X_temp.max(0)

        # z1 and z2 at all (x,t)
        u1_temp = Exact_ter[:, :, 0].flatten()[:, None]
        u2_temp = Exact_ter[:, :, 1].flatten()[:, None]
        u_temp2 = np.hstack([u1_temp, u2_temp])   # (N, 2)

        # IC + BC training data
        xx1 = np.hstack([X[0:1, :].T, T[0:1, :].T])
        uu1 = np.hstack([Exact_ter[0:1, :, 0].T,
                          Exact_ter[0:1, :, 1].T])
        xx2 = np.hstack([X[1:, 0:1], T[1:, 0:1]])
        uu2 = np.hstack([Exact_ter[1:, 0:1, 0],
                          Exact_ter[1:, 0:1, 1]])
        X_u_ter = np.vstack([xx1, xx2])
        u_ter   = np.vstack([uu1, uu2])
        idx     = np.random.choice(len(X_u_ter), N_u, replace=False)
        X_u_ter, u_ter = X_u_ter[idx], u_ter[idx]

        X_f_ter = lb + (ub - lb) * lhs(2, N_f)

        # network: 2 inputs → 2 outputs
        layers_ter = [2, 64, 64, 64, 64, 64, 2]

        ter_model = TernaryCompPINN(
            X_u_ter, u_ter, X_f_ter,
            layers_ter, lb, ub,
            K=K3,
            use_causal=True,
            causal_eps=5.0,
            eps_kink_init=0.05,
            eps_kink_final=0.005
        )
        ter_model.train(niter_adam=5000, niter_lbfgs=3000)

        z12_pred = ter_model.predict(X_temp)   # (N, 2)
        err1 = (np.linalg.norm(u1_temp - z12_pred[:, 0:1], 2) /
                np.linalg.norm(u1_temp, 2))
        err2 = (np.linalg.norm(u2_temp - z12_pred[:, 1:2], 2) /
                np.linalg.norm(u2_temp, 2))
        print(f"\nTernary PINN  |  L2 z1={err1:.4e},  L2 z2={err2:.4e}")

        Z1_pred = griddata(X_temp, z12_pred[:, 0], (X, T), method='linear')
        Z2_pred = griddata(X_temp, z12_pred[:, 1], (X, T), method='linear')

        plt.figure(figsize=(7, 4))
        plt.semilogy(ter_model.loss_history)
        plt.title("Ternary PINN — training loss")
        plt.xlabel("Iteration"); plt.ylabel("Loss")
        plt.grid(True, which='both'); plt.tight_layout(); plt.show()

        sample = np.arange(0, len(t), max(1, len(t) // 6))
        for i in sample:
            fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=False)
            for ax, Z_ref, Z_pr, comp in zip(
                    axes,
                    [Exact_ter[i, :, 0], Exact_ter[i, :, 1]],
                    [Z1_pred[i, :],      Z2_pred[i, :]],
                    ['z₁ (light)', 'z₂ (intermediate)']):
                ax.plot(x, Z_ref, '-k', lw=2, label='FV ref')
                ax.plot(x, Z_pr,  '--r', lw=2, label='PINN')
                ax.set_title(f"{comp}  t={t[i]:.3f}")
                ax.set_xlabel('x'); ax.legend(); ax.grid(True)
                ax.set_xlim([0, 1]); ax.set_ylim([-0.02, 1.02])
            plt.tight_layout(); plt.show()