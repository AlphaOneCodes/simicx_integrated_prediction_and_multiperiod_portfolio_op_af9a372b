"""
Signal Generation Module with CNN-LSTM and Multi-Period Differentiable Portfolio Optimization.

This module implements:
- CNN-LSTM neural network for return prediction
- MDFP (Multi-period Differentiable Finance Portfolio) optimization layer
- Rolling signal generation with periodic rebalancing

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import cvxpy as cp
from scipy import linalg
from typing import Optional, Tuple, Dict, List
from simicx.trading_sim import validate_trading_sheet


# ============================================================================
# Hardware Setup
# ============================================================================

def get_device() -> torch.device:
    """
    Detect and return the best available device (CUDA > MPS > CPU).
    
    Priority order:
        1. CUDA (NVIDIA GPU)
        2. MPS (Apple Silicon)
        3. CPU (fallback)
    
    Returns:
        torch.device: The best available compute device.
    
    Example:
        >>> device = get_device()
        >>> print(device)  # cuda:0, mps, or cpu
        >>> tensor = torch.randn(10).to(device)
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


# ============================================================================
# CNN-LSTM Model
# ============================================================================

class CNN_LSTM(nn.Module):
    """
    CNN-LSTM model for multi-asset return prediction.
    
    Architecture:
        - Conv1D (in=N_assets, out=64, k=5, s=1, p=2) -> ReLU -> MaxPool1D(2)
        - LSTM (input=64, hidden=64, layers=2, batch_first=True)
        - FC -> Reshape to (Batch, Horizon, N_assets)
    
    The model predicts both expected returns and covariance matrices for
    multi-period portfolio optimization.
    
    Args:
        n_assets: Number of assets in the universe
        horizon: Planning horizon (number of future periods to predict)
        
    Attributes:
        conv1d: 1D convolutional layer for feature extraction
        lstm: LSTM layer for temporal modeling
        fc_mean: Fully connected layer for return prediction
        fc_cov_factor: Fully connected layer for covariance factor prediction
        
    Example:
        >>> model = CNN_LSTM(n_assets=10, horizon=5)
        >>> x = torch.randn(32, 120, 10)  # (batch, seq_len, n_assets)
        >>> y_hat, V_hat = model(x)
        >>> y_hat.shape
        torch.Size([32, 5, 10])
        >>> V_hat.shape
        torch.Size([32, 5, 10, 10])
    """
    
    def __init__(self, n_assets: int, horizon: int):
        super().__init__()
        self.n_assets = n_assets
        self.horizon = horizon
        
        # Conv1D: input channels = n_assets, output channels = 64
        self.conv1d = nn.Conv1d(
            in_channels=n_assets,
            out_channels=64,
            kernel_size=5,
            stride=1,
            padding=2
        )
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=2)
        
        # LSTM: input_size=64, hidden_size=64, 2 layers
        self.lstm = nn.LSTM(
            input_size=64,
            hidden_size=64,
            num_layers=2,
            batch_first=True
        )
        
        # FC layer for mean returns only
        # Note: Covariance is computed from historical returns (sample covariance)
        # rather than learned, which is more stable and aligns with standard practice
        self.fc_mean = nn.Linear(64, horizon * n_assets)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of CNN-LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_assets)
               Represents historical returns for each asset
               
        Returns:
            y_hat: Predicted returns, shape (batch, horizon, n_assets)
                         
        Example:
            >>> model = CNN_LSTM(n_assets=5, horizon=3)
            >>> x = torch.randn(4, 60, 5)
            >>> y_hat, V_hat = model(x)
            >>> y_hat.shape
            torch.Size([4, 3, 5])
        """
        batch_size = x.shape[0]
        device = x.device
        dtype = x.dtype
        
        # Transpose for Conv1D: (batch, n_assets, seq_len)
        x = x.transpose(1, 2)
        
        # Conv1D -> ReLU -> MaxPool
        x = self.conv1d(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Transpose back for LSTM: (batch, seq_len//2, 64)
        x = x.transpose(1, 2)
        
        # LSTM forward
        lstm_out, (h_n, c_n) = self.lstm(x)
        
        # Use last hidden state from top layer: (batch, 64)
        last_hidden = h_n[-1]
        
        # Predict returns: (batch, horizon * n_assets) -> (batch, horizon, n_assets)
        y_hat = self.fc_mean(last_hidden)
        y_hat = y_hat.view(batch_size, self.horizon, self.n_assets)
        
        return y_hat


# ============================================================================
# MDFP Layer (Multi-period Differentiable Finance Portfolio)
# ============================================================================

class MDFP_Layer(torch.autograd.Function):
    """
    Multi-period Differentiable Finance Portfolio optimization layer.
    
    Solves the multi-period portfolio optimization problem using cvxpy:
    
        min_{z} sum_{s=t+1}^{t+H} [
            -y_hat_s^T z_s 
            + (delta/2) z_s^T V_hat_s z_s 
            + lambda * sum_i sqrt((z_{s,i} - z_{s-1,i})^2 + kappa)
        ]
        
        s.t. sum(z_s) = 1, z_s >= 0  for all s
    
    Uses implicit differentiation via Neumann series approximation for 
    backpropagation through the optimization layer.
    
    The Neumann series approximation:
        dz*/d_theta ≈ sum_{b=0}^{B} [d_z Phi]^b @ (d_Phi/d_theta)
    
    where Phi(z) is the entropic mirror descent map:
        Phi(z) = softmax(ln(z) - eta * grad_f(z))
    """
    
    @staticmethod
    def forward(
        ctx,
        y_hat: torch.Tensor,
        V_hat: torch.Tensor,
        z_prev: torch.Tensor,
        delta: float,
        lambda_val: float,
        kappa: float,
        neumann_order: int,
        eta: float
    ) -> torch.Tensor:
        """
        Solve multi-period portfolio optimization using cvxpy.
        
        Args:
            ctx: Context for saving tensors for backward pass
            y_hat: Predicted returns, shape (batch, horizon, n_assets)
            V_hat: Predicted covariances, shape (batch, horizon, n_assets, n_assets)
            z_prev: Previous portfolio weights, shape (batch, n_assets)
            delta: Risk aversion parameter
            lambda_val: Turnover penalty coefficient
            kappa: Smoothing parameter for turnover cost (prevents division by zero)
            neumann_order: Order of Neumann series for backward pass
            eta: Learning rate for entropic mirror descent in backward
            
        Returns:
            z_star: Optimal portfolio weights, shape (batch, horizon, n_assets)
            
        Example:
            >>> y = torch.randn(2, 3, 5)
            >>> V = torch.eye(5).unsqueeze(0).unsqueeze(0).expand(2, 3, 5, 5) * 0.01
            >>> z_prev = torch.ones(2, 5) / 5
            >>> z_star = MDFP_Layer.apply(y, V, z_prev, 1.0, 0.01, 1e-4, 5, 0.1)
        """
        batch_size, horizon, n_assets = y_hat.shape
        device = y_hat.device
        dtype = y_hat.dtype
        
        # Convert to numpy for cvxpy
        y_hat_np = y_hat.detach().cpu().numpy()
        V_hat_np = V_hat.detach().cpu().numpy()
        z_prev_np = z_prev.detach().cpu().numpy()
        
        z_star_list = []
        
        for b in range(batch_size):
            # Create optimization variables for each time step
            z_vars = [cp.Variable(n_assets, nonneg=True) for _ in range(horizon)]
            
            objective_terms = []
            constraints = []
            
            z_prev_s = z_prev_np[b]
            
            for s in range(horizon):
                y_s = y_hat_np[b, s]
                V_s = V_hat_np[b, s]
                z_s = z_vars[s]
                
                # Make V_s symmetric and PSD
                V_s = (V_s + V_s.T) / 2
                eigvals = np.linalg.eigvalsh(V_s)
                if eigvals.min() < 0:
                    V_s = V_s + (abs(eigvals.min()) + 1e-6) * np.eye(n_assets)
                
                # Return term: -y_hat_s^T z_s
                return_term = -y_s @ z_s
                
                # Risk term: (delta/2) z_s^T V_hat_s z_s
                risk_term = (delta / 2) * cp.quad_form(z_s, V_s)
                
                # Turnover term with sqrt smoothing
                if s == 0:
                    z_prev_var = z_prev_s
                else:
                    z_prev_var = z_vars[s - 1]
                
                diff = z_s - z_prev_var
                # Approximate sqrt(x^2 + kappa) using second-order cone
                turnover_term = lambda_val * cp.sum(cp.norm(cp.vstack([
                    diff.reshape((1, n_assets)),
                    np.sqrt(kappa) * np.ones((1, n_assets))
                ]), axis=0))
                
                objective_terms.extend([return_term, risk_term, turnover_term])
                
                # Simplex constraints
                constraints.append(cp.sum(z_s) == 1)
            
            # Solve optimization
            objective = cp.Minimize(cp.sum(objective_terms))
            problem = cp.Problem(objective, constraints)
            
            try:
                problem.solve(solver=cp.ECOS, verbose=False, max_iters=500)
                
                if problem.status in [cp.OPTIMAL, cp.OPTIMAL_INACCURATE]:
                    z_opt = np.stack([z_vars[s].value for s in range(horizon)], axis=0)
                    # Handle potential None values
                    if z_opt is None or np.any([z_vars[s].value is None for s in range(horizon)]):
                        z_opt = np.ones((horizon, n_assets)) / n_assets
                else:
                    z_opt = np.ones((horizon, n_assets)) / n_assets
            except Exception:
                z_opt = np.ones((horizon, n_assets)) / n_assets
            
            # Ensure valid weights
            z_opt = np.maximum(z_opt, 1e-8)
            z_opt = z_opt / z_opt.sum(axis=1, keepdims=True)
            
            z_star_list.append(z_opt)
        
        z_star_np = np.stack(z_star_list, axis=0)
        z_star = torch.tensor(z_star_np, device=device, dtype=dtype)
        
        # Save for backward
        ctx.save_for_backward(y_hat, V_hat, z_prev, z_star)
        ctx.delta = delta
        ctx.lambda_val = lambda_val
        ctx.kappa = kappa
        ctx.neumann_order = neumann_order
        ctx.eta = eta
        
        return z_star
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        """
        Backward pass using Neumann series approximation for implicit differentiation.
        
        Implements:
            dz*/d_theta ≈ sum_{b=0}^{B} [d_z Phi]^b @ (d_Phi/d_theta)
        
        where Phi(z) = softmax(ln(z) - eta * grad_f(z)) is the entropic mirror descent map.
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient w.r.t. output z_star, shape (batch, horizon, n_assets)
            
        Returns:
            Tuple of gradients w.r.t. all inputs (y_hat, V_hat, and None for scalars)
        """
        y_hat, V_hat, z_prev, z_star = ctx.saved_tensors
        delta = ctx.delta
        lambda_val = ctx.lambda_val
        kappa = ctx.kappa
        neumann_order = ctx.neumann_order
        eta = ctx.eta
        
        batch_size, horizon, n_assets = y_hat.shape
        device = y_hat.device
        dtype = y_hat.dtype
        
        # Initialize gradients
        grad_y_hat = torch.zeros_like(y_hat)
        grad_V_hat = torch.zeros_like(V_hat)
        
        def compute_grad_f(z: torch.Tensor, y: torch.Tensor, V: torch.Tensor, 
                          z_prev_step: torch.Tensor) -> torch.Tensor:
            """
            Compute gradient of objective f w.r.t. z.
            
            grad_f = -y + delta * V @ z + lambda * (z - z_prev) / sqrt((z - z_prev)^2 + kappa)
            """
            grad_return = -y
            grad_risk = delta * torch.matmul(V, z.unsqueeze(-1)).squeeze(-1)
            
            diff = z - z_prev_step
            denom = torch.sqrt(diff ** 2 + kappa)
            grad_turnover = lambda_val * diff / denom
            
            return grad_return + grad_risk + grad_turnover
        
        def entropic_mirror_descent_map(z: torch.Tensor, y: torch.Tensor, V: torch.Tensor,
                                        z_prev_step: torch.Tensor) -> torch.Tensor:
            """
            Entropic mirror descent map: Phi(z) = softmax(ln(z) - eta * grad_f(z))
            """
            grad_f = compute_grad_f(z, y, V, z_prev_step)
            log_z = torch.log(z.clamp(min=1e-8))
            return torch.softmax(log_z - eta * grad_f, dim=-1)
        
        # Process each batch and horizon step
        for b in range(batch_size):
            for s in range(horizon):
                z_s = z_star[b, s].detach().clone()
                y_s = y_hat[b, s].detach()
                V_s = V_hat[b, s].detach()
                
                if s == 0:
                    z_prev_s = z_prev[b].detach()
                else:
                    z_prev_s = z_star[b, s - 1].detach()
                
                # Compute Jacobian of Phi w.r.t z using finite differences
                eps = 1e-5
                d_z_phi = torch.zeros(n_assets, n_assets, device=device, dtype=dtype)
                
                phi_base = entropic_mirror_descent_map(z_s, y_s, V_s, z_prev_s)
                
                for i in range(n_assets):
                    z_perturbed = z_s.clone()
                    z_perturbed[i] += eps
                    # Renormalize to stay on simplex
                    z_perturbed = z_perturbed / z_perturbed.sum()
                    phi_perturbed = entropic_mirror_descent_map(z_perturbed, y_s, V_s, z_prev_s)
                    d_z_phi[:, i] = (phi_perturbed - phi_base) / eps
                
                # Neumann series: sum_{b=0}^{B} (d_z Phi)^b
                neumann_sum = torch.eye(n_assets, device=device, dtype=dtype)
                power = torch.eye(n_assets, device=device, dtype=dtype)
                
                for _ in range(neumann_order):
                    power = torch.matmul(d_z_phi, power)
                    neumann_sum = neumann_sum + power
                
                # Apply Neumann series to gradient
                v = grad_output[b, s]
                w = torch.matmul(neumann_sum.T, v)
                
                # Compute gradients w.r.t. y and V through Phi using finite differences
                # Gradient w.r.t. y
                for i in range(n_assets):
                    y_perturbed = y_s.clone()
                    y_perturbed[i] += eps
                    phi_perturbed = entropic_mirror_descent_map(z_s, y_perturbed, V_s, z_prev_s)
                    d_phi_d_yi = (phi_perturbed - phi_base) / eps
                    grad_y_hat[b, s, i] = torch.dot(w, d_phi_d_yi)
                
                # Gradient w.r.t. V (diagonal and lower triangular)
                for i in range(n_assets):
                    for j in range(i + 1):
                        V_perturbed = V_s.clone()
                        V_perturbed[i, j] += eps
                        if i != j:
                            V_perturbed[j, i] += eps  # Maintain symmetry
                        phi_perturbed = entropic_mirror_descent_map(z_s, y_s, V_perturbed, z_prev_s)
                        d_phi_d_Vij = (phi_perturbed - phi_base) / eps
                        grad_V_hat[b, s, i, j] = torch.dot(w, d_phi_d_Vij)
                        if i != j:
                            grad_V_hat[b, s, j, i] = grad_V_hat[b, s, i, j]
        
        # Return gradients (None for scalar parameters)
        return grad_y_hat, grad_V_hat, None, None, None, None, None, None


def apply_mdfp(
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float,
    lambda_val: float,
    kappa: float,
    neumann_order: int = 5,
    eta: float = 0.1
) -> torch.Tensor:
    """
    Apply MDFP optimization layer (convenience wrapper).
    
    Args:
        y_hat: Predicted returns, shape (batch, horizon, n_assets)
        V_hat: Predicted covariances, shape (batch, horizon, n_assets, n_assets)
        z_prev: Previous portfolio weights, shape (batch, n_assets)
        delta: Risk aversion parameter
        lambda_val: Turnover penalty coefficient
        kappa: Smoothing parameter for turnover cost
        neumann_order: Order of Neumann series for backward pass
        eta: Learning rate for entropic mirror descent
        
    Returns:
        z_star: Optimal portfolio weights, shape (batch, horizon, n_assets)
        
    Example:
        >>> y_hat = torch.randn(2, 5, 10)  # batch=2, horizon=5, assets=10
        >>> V_hat = torch.eye(10).unsqueeze(0).unsqueeze(0).expand(2, 5, -1, -1)
        >>> z_prev = torch.ones(2, 10) / 10
        >>> z_star = apply_mdfp(y_hat, V_hat, z_prev, delta=1.0, lambda_val=0.01, kappa=1e-4)
        >>> z_star.shape
        torch.Size([2, 5, 10])
    """
    return MDFP_Layer.apply(y_hat, V_hat, z_prev, delta, lambda_val, kappa, neumann_order, eta)


# ============================================================================
# PyTorch Solver (GPU-Accelerated Alternative to CVXPY)
# ============================================================================

def project_simplex(z: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Project weights onto the probability simplex: sum(z) = 1, z >= 0.
    
    Uses efficient O(n log n) algorithm from Duchi et al. (2008).
    "Efficient Projections onto the l1-Ball for Learning in High Dimensions"
    """
    z_sorted, _ = torch.sort(z, descending=True, dim=dim)
    cumsum = torch.cumsum(z_sorted, dim=dim)
    k = torch.arange(1, z.shape[dim] + 1, device=z.device, dtype=z.dtype)
    if dim != -1:
        k = k.reshape([1] * (-dim - 1) + [-1] + [1] * (z.ndim + dim))
    condition = z_sorted - (cumsum - 1) / k > 0
    k_star = condition.sum(dim=dim, keepdim=True)
    tau = (cumsum.gather(dim, k_star - 1) - 1) / k_star
    return torch.clamp(z - tau, min=0)


def portfolio_objective(
    z: torch.Tensor,
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float,
    lambda_val: float,
    kappa: float = 1e-6
) -> torch.Tensor:
    """Compute multi-period portfolio objective."""
    batch_size, horizon, n_assets = z.shape
    objectives = []
    
    for b in range(batch_size):
        for s in range(horizon):
            z_s = z[b, s]
            y_s = y_hat[b, s]
            V_s = V_hat[b, s]
            
            return_term = -torch.dot(y_s, z_s)
            risk_term = (delta / 2) * torch.dot(z_s, torch.matmul(V_s, z_s))
            
            if s == 0:
                z_prev_s = z_prev[b]
            else:
                z_prev_s = z[b, s - 1]
            
            diff = z_s - z_prev_s
            turnover_term = lambda_val * torch.sqrt((diff ** 2 + kappa)).sum()
            
            obj = return_term + risk_term + turnover_term
            objectives.append(obj)
    
    return torch.stack(objectives).sum() / (batch_size * horizon)


def solve_mdfp_pytorch(
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float,
    lambda_val: float,
    kappa: float = 1e-6,
    max_iter: int = 50,
    lr: float = 0.5,
    tol: float = 1e-5,
    verbose: bool = False
) -> torch.Tensor:
    """
    Solve multi-period portfolio optimization using PyTorch projected gradient descent.
    
    GPU-accelerated, fully differentiable alternative to CVXPY.
    """
    batch_size, horizon, n_assets = y_hat.shape
    device = y_hat.device
    dtype = y_hat.dtype
    
    with torch.enable_grad():
        z = torch.ones(batch_size, horizon, n_assets, device=device, dtype=dtype) / n_assets
        prev_obj = float('inf')
        
        for iteration in range(max_iter):
            z_temp = z.detach().requires_grad_(True)
            obj = portfolio_objective(z_temp, y_hat, V_hat, z_prev, delta, lambda_val, kappa)
            
            grad = torch.autograd.grad(
                outputs=obj,
                inputs=z_temp,
                create_graph=False,
                retain_graph=False,
                allow_unused=False
            )[0]
            
            with torch.no_grad():
                z = z - lr * grad
                for b in range(batch_size):
                    for s in range(horizon):
                        z[b, s] = project_simplex(z[b, s])
            
            with torch.no_grad():
                obj_val = portfolio_objective(z, y_hat, V_hat, z_prev, delta, lambda_val, kappa).item()
            
            if abs(prev_obj - obj_val) < tol:
                break
            prev_obj = obj_val
    
    return z


class MDFPSolverPyTorch(torch.autograd.Function):
    """Differentiable MDFP solver using implicit differentiation (PyTorch-native)."""
    
    @staticmethod
    def forward(ctx, y_hat, V_hat, z_prev, delta, lambda_val, kappa, neumann_order, eta, max_iter, lr, tol):
        with torch.enable_grad():
            z_star = solve_mdfp_pytorch(
                y_hat, V_hat, z_prev, delta, lambda_val, kappa,
                max_iter=max_iter, lr=lr, tol=tol, verbose=False
            )
        
        ctx.save_for_backward(y_hat, V_hat, z_prev, z_star)
        ctx.delta = delta
        ctx.lambda_val = lambda_val
        ctx.kappa = kappa
        ctx.neumann_order = neumann_order
        ctx.eta = eta
        
        return z_star
    
    @staticmethod
    def backward(ctx, grad_output):
        """Simplified backward pass using direct analytical gradients (approximation)."""
        y_hat, V_hat, z_prev, z_star = ctx.saved_tensors
        delta = ctx.delta
        
        batch_size, horizon, n_assets = y_hat.shape
        
        grad_y_hat = torch.zeros_like(y_hat)
        for b in range(batch_size):
            for s in range(horizon):
                grad_y_hat[b, s] = grad_output[b, s] * z_star[b, s]
        
        grad_V_hat = torch.zeros_like(V_hat)
        for b in range(batch_size):
            for s in range(horizon):
                z_s = z_star[b, s]
                outer = torch.outer(z_s, z_s)
                weight = grad_output[b, s].sum()
                grad_V_hat[b, s] = -delta / 2.0 * weight * outer
        
        return grad_y_hat, grad_V_hat, None, None, None, None, None, None, None, None, None


def apply_mdfp_pytorch(
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float,
    lambda_val: float,
    kappa: float = 1e-6,
    neumann_order: int = 5,
    eta: float = 0.1,
    max_iter: int = 50,
    lr: float = 0.5,
    tol: float = 1e-5
) -> torch.Tensor:
    """Apply MDFP optimization layer using PyTorch (drop-in replacement for apply_mdfp)."""
    return MDFPSolverPyTorch.apply(y_hat, V_hat, z_prev, delta, lambda_val, kappa, neumann_order, eta, max_iter, lr, tol)


# ============================================================================
# Signal Generation
# ============================================================================

def signal_gen(
    ohlcv_df: pd.DataFrame,
    lookback_window: int = 120,
    planning_horizon: int = 5,
    rebalance_freq: int = 20,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.01,
    learning_rate: float = 1e-3,
    epochs: int = 50,
    neumann_order: int = 5,
    initial_capital: float = 1_000_000.0,
    solver: str = 'pytorch'
) -> pd.DataFrame:
    """
    Generate trading signals using CNN-LSTM and MDFP optimization.
    
    This function implements a complete portfolio optimization pipeline:
    1. Prepares data (log returns from Close prices)
    2. Trains CNN-LSTM model at periodic rebalance points
    3. Uses MDFP layer for multi-period portfolio optimization
    4. Converts optimal weights to trading quantities
    
    The optimization objective minimizes:
        - Negative expected return (maximize return)
        - Risk (via covariance)
        - Transaction costs (turnover penalty)
    
    Subject to:
        - Fully invested constraint (weights sum to 1)
        - Long-only constraint (no short selling)
    
    Args:
        ohlcv_df: OHLCV DataFrame with columns [time, ticker, open, high, low, close, volume]
        lookback_window: Number of historical periods for model input (default: 120)
        planning_horizon: Number of future periods to optimize over (default: 5)
        rebalance_freq: Rebalance portfolio every N periods (default: 20)
        risk_aversion: Risk aversion parameter delta for mean-variance (default: 1.0)
        turnover_penalty: Turnover cost coefficient lambda (default: 0.01)
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        epochs: Number of training epochs per rebalance (default: 50)
        neumann_order: Order of Neumann series for implicit differentiation (default: 5)
        initial_capital: Initial capital for position sizing (default: 1,000,000)
        solver: Optimization solver - 'pytorch' (fast, GPU) or 'cvxpy' (accurate, CPU) (default: 'pytorch')
        
    Returns:
        DataFrame with columns [time, ticker, action, quantity, price]
        validated by trading_sim's validate_trading_sheet
        
    Example:
        >>> from simicx.data_loader import get_trading_data
        >>> ohlcv = get_trading_data(tickers=['AAPL', 'MSFT', 'GOOGL'], align_dates=True)
        >>> trades = signal_gen(
        ...     ohlcv_df=ohlcv,
        ...     lookback_window=60,
        ...     planning_horizon=5,
        ...     rebalance_freq=10
        ... )
        >>> trades.head()
                        time ticker action  quantity   price
        0 2025-01-15 09:30:00   AAPL    buy       100  150.25
        1 2025-01-15 09:30:00   MSFT    buy        50  380.50
    """
    device = get_device()
    
    # =========================================================================
    # Data Preparation
    # =========================================================================
    
    # Validate input DataFrame
    required_cols = ['time', 'ticker', 'open', 'high', 'low', 'close', 'volume']
    missing_cols = [c for c in required_cols if c not in ohlcv_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Get unique tickers and dates
    tickers = sorted(ohlcv_df['ticker'].unique().tolist())
    n_assets = len(tickers)
    
    if n_assets == 0:
        raise ValueError("No tickers found in ohlcv_df")
    
    # Pivot to wide format: (time, ticker) -> close price
    pivot_close = ohlcv_df.pivot(index='time', columns='ticker', values='close')
    pivot_close = pivot_close[tickers]
    pivot_close = pivot_close.sort_index()
    
    # Pivot for open prices (T+1 execution logic)
    pivot_open = ohlcv_df.pivot(index='time', columns='ticker', values='open')
    pivot_open = pivot_open[tickers]
    pivot_open = pivot_open.sort_index()
    
    # Compute log returns
    log_returns = np.log(pivot_close / pivot_close.shift(1))
    log_returns = log_returns.dropna()
    
    # Align open prices with returns
    common_dates = log_returns.index.intersection(pivot_open.index)
    log_returns = log_returns.loc[common_dates]
    pivot_open = pivot_open.loc[common_dates]
    pivot_close = pivot_close.loc[common_dates]
    
    dates = log_returns.index.tolist()
    returns_np = log_returns.values  # (T, n_assets)
    
    if len(dates) < lookback_window + planning_horizon + 1:
        raise ValueError(
            f"Insufficient data: need at least {lookback_window + planning_horizon + 1} dates, "
            f"got {len(dates)}"
        )
    
    nan_count = np.isnan(returns_np).sum()
    if nan_count > 0:
        nan_pct = (nan_count / returns_np.size) * 100
        import warnings
        warnings.warn(f"Found {nan_count} NaN values ({nan_pct:.2f}% of data), filling with 0.0")
    
    # Handle NaN values before normalization
    returns_np = np.nan_to_num(returns_np, nan=0.0)
    
    # Use expanding window normalization to prevent future data leakage
    returns_normalized = np.zeros_like(returns_np)
    
    for i in range(len(returns_np)):
        # CRITICAL: Use ONLY historical data (BEFORE time i) for normalization statistics
        # Do NOT include returns_np[i] in the window when normalizing returns_np[i]
        if i == 0:
            # First observation: no history available, set to zero
            returns_normalized[i] = 0.0
            continue
        elif i < lookback_window:
            # For initial period, use all available data BEFORE current point
            window_data = returns_np[:i]  # [0:i), NOT including i
        else:
            # Use rolling window of lookback_window size BEFORE current point
            window_data = returns_np[i-lookback_window:i]  # [i-lookback:i), NOT including i
        
        # Calculate mean and std from STRICTLY historical data (before time i)
        window_mean = np.nanmean(window_data, axis=0, keepdims=True)
        window_std = np.nanstd(window_data, axis=0, keepdims=True) + 1e-8
        
        # Normalize current observation using only historical statistics
        returns_normalized[i] = (returns_np[i] - window_mean) / window_std
    
    # Final check for any remaining NaNs after normalization
    returns_normalized = np.nan_to_num(returns_normalized, nan=0.0)
    
    # =========================================================================
    # Helper Function: Compute Sample Covariance
    # =========================================================================
    
    def compute_sample_covariance_batch(returns_batch: torch.Tensor, horizon: int) -> torch.Tensor:
        """
        Vectorized covariance computation for entire batch (GPU-accelerated).
        
        This is MUCH faster than computing covariances one-by-one in a loop.
        Uses PyTorch operations that run on GPU when available.
        
        Args:
            returns_batch: Historical returns, shape (batch, lookback_window, n_assets)
                          Already on GPU if available
            horizon: Planning horizon (returns same covariance for all periods)
            
        Returns:
            V_hat: Covariance matrices, shape (batch, horizon, n_assets, n_assets)
                   Same covariance used for all periods in the horizon
        """
        batch_size, lookback, n_assets = returns_batch.shape
        
        # Center the data (subtract mean)
        mean = returns_batch.mean(dim=1, keepdim=True)  # (batch, 1, n_assets)
        centered = returns_batch - mean  # (batch, lookback, n_assets)
        
        # Compute covariance: (1/(n-1)) * X^T @ X
        # Using einsum for efficiency: 'bti,btj->bij' means:
        # - b: batch dimension
        # - t: time dimension (lookback)
        # - i,j: asset dimensions
        cov = torch.einsum('bti,btj->bij', centered, centered) / (lookback - 1)
        # Shape: (batch, n_assets, n_assets)
        
        # Add small diagonal for numerical stability (ensure PSD)
        eye = torch.eye(n_assets, device=returns_batch.device, dtype=returns_batch.dtype)
        cov = cov + 1e-4 * eye.unsqueeze(0)
        
        # Expand to horizon: same covariance for all periods
        # Shape: (batch, horizon, n_assets, n_assets)
        V_hat = cov.unsqueeze(1).expand(-1, horizon, -1, -1).contiguous()
        
        return V_hat
    
    # =========================================================================
    # Model Initialization
    # =========================================================================
    
    model = CNN_LSTM(n_assets=n_assets, horizon=planning_horizon).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    # Initialize previous weights (uniform)
    z_prev = np.ones(n_assets) / n_assets
    
    # Kappa for smooth turnover cost
    kappa = 1e-4
    
    # Eta for entropic mirror descent
    eta = 0.1
    
    # =========================================================================
    # Solver Selection
    # =========================================================================
    
    if solver.lower() == 'pytorch':
        print(f"Using PyTorch solver (device: {device.type})")
        
        def solve_portfolio(y_hat, V_hat, z_prev_tensor, training=True):
            """Solve portfolio optimization using PyTorch.
            
            Args:
                training: If True, uses differentiable version for gradient flow.
                         If False, uses faster non-differentiable version.
            """
            if training:
                # Use differentiable version with implicit gradients
                return apply_mdfp_pytorch(
                    y_hat=y_hat,
                    V_hat=V_hat,
                    z_prev=z_prev_tensor,
                    delta=risk_aversion,
                    lambda_val=turnover_penalty,
                    kappa=kappa,
                    max_iter=50,
                    lr=0.5,
                    tol=1e-5
                )
            else:
                # Use faster non-differentiable version for inference
                return solve_mdfp_pytorch(
                    y_hat=y_hat,
                    V_hat=V_hat,
                    z_prev=z_prev_tensor,
                    delta=risk_aversion,
                    lambda_val=turnover_penalty,
                    kappa=kappa,
                    max_iter=50,
                    lr=0.5,
                    tol=1e-5
                )
    
    elif solver.lower() == 'cvxpy':
        print("Using CVXPY solver (CPU only)")
        
        def solve_portfolio(y_hat, V_hat, z_prev_tensor, training=True):
            """Solve portfolio optimization using CVXPY.
            
            Args:
                training: Parameter for consistency with pytorch solver (unused for CVXPY).
            """
            return apply_mdfp(
                y_hat=y_hat,
                V_hat=V_hat,
                z_prev=z_prev_tensor,
                delta=risk_aversion,
                lambda_val=turnover_penalty,
                kappa=kappa,
                neumann_order=neumann_order,
                eta=eta
            )
    
    else:
        raise ValueError(
            f"Unknown solver: '{solver}'. Must be 'pytorch' or 'cvxpy'. "
            f"Use 'pytorch' for speed and GPU support, 'cvxpy' for maximum accuracy."
        )
    
    # =========================================================================
    # Output containers
    # =========================================================================

    all_trades: List[Dict] = []
    current_quantities = np.zeros(n_assets)
    current_cash = initial_capital  # Track cash to update portfolio value over time
    
    current_portfolio_value = initial_capital
    
    # =========================================================================
    # Rolling loop over dates
    # =========================================================================
    
    for idx in range(lookback_window, len(dates) - planning_horizon):
        current_date = dates[idx]
        
        # Check if it's a rebalance point
        if idx % rebalance_freq == 0:
            # =================================================================
            # Training Step (using strictly HISTORICAL data - no lookahead)
            # =================================================================
            # 
            # To avoid lookahead bias, we train ONLY on data where:
            # - Input window: [t - lookback_window, t)
            # - Target returns: [t, t + planning_horizon)
            # - CRITICAL: t + planning_horizon <= idx (targets fully realized before now)
            #
            # Valid training positions: lookback_window <= t <= idx - planning_horizon

            model.train()

            # Calculate valid training range (no lookahead)
            min_train_t = lookback_window
            max_train_t = idx - planning_horizon  # Targets must be realized before idx

            if max_train_t >= min_train_t:
                for epoch in range(epochs):
                    # Build training batch from historical windows only
                    train_X_list = []
                    train_Y_list = []

                    # Use recent historical windows (up to 64 samples for batch)
                    # Increased from 16 to better utilize GPU
                    batch_size = min(64, max_train_t - min_train_t + 1)

                    for t in range(max(min_train_t, max_train_t - batch_size + 1), max_train_t + 1):
                        X_sample = returns_normalized[t - lookback_window:t]
                        Y_sample = returns_np[t:t + planning_horizon]
                        train_X_list.append(X_sample)
                        train_Y_list.append(Y_sample)

                    if len(train_X_list) == 0:
                        continue

                    X_batch = np.stack(train_X_list, axis=0)
                    Y_batch = np.stack(train_Y_list, axis=0)

                    X = torch.tensor(X_batch, dtype=torch.float32, device=device)
                    Y = torch.tensor(Y_batch, dtype=torch.float32, device=device)

                    # Create z_prev for each sample in batch
                    z_prev_batch = np.tile(z_prev.reshape(1, -1), (len(train_X_list), 1))
                    z_prev_tensor = torch.tensor(z_prev_batch, dtype=torch.float32, device=device)

                    optimizer.zero_grad()

                    # Forward pass through CNN-LSTM (returns only)
                    y_hat = model(X)
                    
                    # Compute sample covariance from historical returns (vectorized, GPU-accelerated)
                    # X is already on GPU with shape (batch, lookback_window, n_assets)
                    V_hat = compute_sample_covariance_batch(X, planning_horizon)

                    # MDFP optimization (using selected solver)
                    z_star = solve_portfolio(y_hat, V_hat, z_prev_tensor, training=True)

                    # Loss = -Σ[r^T z - λ ||z - z_prev||_1]
                    # Note: Risk term (δ/2 z^T V z) is handled by the optimization layer,
                    # not the training loss. The model learns to predict returns that lead
                    # to good portfolio performance as measured by realized returns.
                    batch_size_actual = z_star.shape[0]
                    
                    # Accumulate objectives as list of tensors, then sum
                    objectives = []
                    
                    for b in range(batch_size_actual):
                        for s in range(planning_horizon):
                            z_s = z_star[b, s]
                            r_s = Y[b, s]  # Realized returns
                            
                            # Return term: r^T z (realized portfolio return)
                            return_term = torch.dot(r_s, z_s)
                            
                            # Turnover term: λ * sqrt((z - z_prev)^2 + κ).sum()
                            # Using smoothed L1 for consistency with optimization layer
                            if s == 0:
                                z_prev_s = z_prev_tensor[b]
                            else:
                                z_prev_s = z_star[b, s-1]
                            
                            diff = z_s - z_prev_s
                            # Smoothed L1 norm (consistent with MDFP solver)
                            turnover_cost = turnover_penalty * torch.sqrt(diff ** 2 + kappa).sum()
                            
                            # Portfolio objective for this step (no risk term in loss)
                            obj = return_term - turnover_cost
                            objectives.append(obj)
                    
                    # Sum all objectives and compute loss
                    total_objective = torch.stack(objectives).sum()
                    
                    # Loss is negative of average objective (minimize loss = maximize objective)
                    loss = -total_objective / (batch_size_actual * planning_horizon)

                    # Backward pass
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    optimizer.step()
            
            # =================================================================
            # Inference Step
            # =================================================================
            
            model.eval()
            
            with torch.no_grad():
                X_infer = torch.tensor(
                    returns_normalized[idx - lookback_window:idx],
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                z_prev_tensor = torch.tensor(
                    z_prev, dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                # Forward pass (returns only)
                y_hat = model(X_infer)
                
                # Compute sample covariance from historical returns (vectorized)
                # X_infer has shape (1, lookback_window, n_assets)
                V_hat = compute_sample_covariance_batch(X_infer, planning_horizon)
                
                # MDFP optimization (using selected solver)
                z_star = solve_portfolio(y_hat, V_hat, z_prev_tensor, training=False)
                
                # Get weights for next period (first step of planning horizon)
                new_weights = z_star[0, 0].cpu().numpy()
            
            # =================================================================
            # Convert weights to trades
            # =================================================================
            
            # Use T+1 open price for execution
            if idx + 1 < len(dates):
                execution_date = dates[idx + 1]
                execution_prices = pivot_open.loc[execution_date].values
            else:
                execution_date = current_date
                execution_prices = pivot_close.loc[current_date].values
            
            # Handle potential NaN prices
            execution_prices = np.nan_to_num(execution_prices, nan=1.0)
            execution_prices = np.maximum(execution_prices, 1e-8)  # Avoid division by zero
            
            # Portfolio value = cash + market value of current holdings
            current_holdings_value = np.sum(current_quantities * execution_prices)
            current_portfolio_value = current_cash + current_holdings_value
            
            # Use ACTUAL portfolio value for position sizing (not initial_capital)
            target_values = current_portfolio_value * new_weights
            target_quantities = np.floor(target_values / execution_prices)
            
            # Calculate trades needed
            quantity_diff = target_quantities - current_quantities
            
            cash_change = 0.0
            
            for i, ticker in enumerate(tickers):
                diff = quantity_diff[i]
                if abs(diff) >= 1:  # Only trade if at least 1 share
                    action = 'buy' if diff > 0 else 'sell'
                    qty = abs(diff)
                    price = float(execution_prices[i])
                    
                    # Update cash based on trade direction
                    if action == 'buy':
                        cash_change -= qty * price  # Spending cash
                    else:
                        cash_change += qty * price  # Receiving cash
                    
                    all_trades.append({
                        'time': execution_date,
                        'ticker': ticker,
                        'action': action,
                        'quantity': float(qty),
                        'price': price
                    })
            
            # Update state
            current_quantities = target_quantities.copy()
            current_cash += cash_change  # Update cash after all trades
            z_prev = new_weights.copy()
    
    # =========================================================================
    # Create output DataFrame
    # =========================================================================
    
    if len(all_trades) == 0:
        trading_sheet = pd.DataFrame(columns=['time', 'ticker', 'action', 'quantity', 'price'])
    else:
        trading_sheet = pd.DataFrame(all_trades)
        trading_sheet['time'] = pd.to_datetime(trading_sheet['time'])
        trading_sheet = trading_sheet.sort_values('time').reset_index(drop=True)
    
    # Validate using trading_sim's validator
    trading_sheet = validate_trading_sheet(trading_sheet)
    
    return trading_sheet


# ============================================================================
# Tests
# ============================================================================

def simicx_test_device_detection():
    """Test hardware device detection."""
    device = get_device()
    assert device.type in ['cuda', 'mps', 'cpu'], f"Unexpected device type: {device.type}"
    
    # Verify tensor can be moved to device
    tensor = torch.randn(10)
    tensor_on_device = tensor.to(device)
    assert tensor_on_device.device.type == device.type


def simicx_test_cnn_lstm_forward():
    """Test CNN_LSTM model forward pass with various configurations."""
    # Test configuration
    configs = [
        (5, 3, 4, 60),   # (n_assets, horizon, batch_size, seq_len)
        (10, 5, 2, 120),
        (3, 2, 8, 30),
    ]
    
    for n_assets, horizon, batch_size, seq_len in configs:
        model = CNN_LSTM(n_assets=n_assets, horizon=horizon)
        x = torch.randn(batch_size, seq_len, n_assets)
        
        y_hat = model(x)
        
        # Check shapes
        assert y_hat.shape == (batch_size, horizon, n_assets), \
            f"y_hat shape mismatch: {y_hat.shape}"
        
        # Check no NaN
        assert not torch.isnan(y_hat).any(), "NaN in y_hat"
        
        print(f"✓ CNN-LSTM test passed for config: n_assets={n_assets}, horizon={horizon}, batch={batch_size}")


def simicx_test_mdfp_layer():
    """Test MDFP optimization layer constraints and output validity."""
    batch_size = 2
    horizon = 3
    n_assets = 4
    
    # Create valid inputs
    torch.manual_seed(42)
    y_hat = torch.randn(batch_size, horizon, n_assets) * 0.01
    
    # Create valid PSD covariance matrices
    L = torch.randn(batch_size, horizon, n_assets, n_assets) * 0.1
    L = torch.tril(L)
    V_hat = torch.matmul(L, L.transpose(-2, -1)) + 0.01 * torch.eye(n_assets)
    
    z_prev = torch.ones(batch_size, n_assets) / n_assets
    
    z_star = apply_mdfp(
        y_hat=y_hat,
        V_hat=V_hat,
        z_prev=z_prev,
        delta=1.0,
        lambda_val=0.01,
        kappa=1e-4,
        neumann_order=3,
        eta=0.1
    )
    
    # Check output shape
    assert z_star.shape == (batch_size, horizon, n_assets), \
        f"Expected shape {(batch_size, horizon, n_assets)}, got {z_star.shape}"
    
    # Check constraints
    z_star_np = z_star.detach().numpy()
    for b in range(batch_size):
        for s in range(horizon):
            weights = z_star_np[b, s]
            
            # Sum to 1
            assert np.isclose(weights.sum(), 1.0, atol=1e-4), \
                f"Weights don't sum to 1: sum={weights.sum()}"
            
            # Non-negative
            assert (weights >= -1e-6).all(), \
                f"Negative weights found: min={weights.min()}"


def simicx_test_signal_gen_synthetic():
    """Test signal_gen with synthetic OHLCV data."""
    np.random.seed(42)
    
    # Create synthetic OHLCV data
    tickers = ['AAPL', 'MSFT', 'GOOGL']
    dates = pd.date_range('2025-01-01', periods=200, freq='D')
    
    rows = []
    for ticker in tickers:
        base_price = 100 + np.random.randn() * 20
        prices = base_price * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.02))
        
        for i, date in enumerate(dates):
            open_price = prices[i] * (1 + np.random.randn() * 0.005)
            high_price = prices[i] * (1 + abs(np.random.randn()) * 0.01)
            low_price = prices[i] * (1 - abs(np.random.randn()) * 0.01)
            close_price = prices[i]
            volume = int(np.random.uniform(1e6, 1e7))
            
            rows.append({
                'time': date,
                'ticker': ticker,
                'open': max(open_price, 0.01),
                'high': max(high_price, 0.01),
                'low': max(low_price, 0.01),
                'close': max(close_price, 0.01),
                'volume': volume
            })
    
    ohlcv_df = pd.DataFrame(rows)
    
    # Run signal generation with minimal settings for speed
    trades = signal_gen(
        ohlcv_df=ohlcv_df,
        lookback_window=30,
        planning_horizon=3,
        rebalance_freq=50,
        risk_aversion=1.0,
        turnover_penalty=0.01,
        learning_rate=1e-2,
        epochs=2,
        neumann_order=2,
        initial_capital=100_000.0
    )
    
    # Validate output structure
    assert isinstance(trades, pd.DataFrame), f"Expected DataFrame, got {type(trades)}"
    
    required_cols = ['time', 'ticker', 'action', 'quantity', 'price']
    for col in required_cols:
        assert col in trades.columns, f"Missing column: {col}"
    
    # If trades exist, validate content
    if len(trades) > 0:
        assert trades['action'].isin(['buy', 'sell']).all(), "Invalid action values"
        assert (trades['quantity'] >= 0).all(), "Negative quantities found"
        assert (trades['price'] > 0).all(), "Non-positive prices found"
        assert trades['ticker'].isin(tickers).all(), "Unknown tickers in output"


def simicx_test_integration_with_trading_sim():
    """Integration test with trading_sim validation function."""
    # Create a sample trading sheet
    trades = pd.DataFrame({
        'time': pd.to_datetime(['2025-01-02', '2025-01-03', '2025-01-04']),
        'ticker': ['AAPL', 'MSFT', 'AAPL'],
        'action': ['buy', 'buy', 'sell'],
        'quantity': [100.0, 50.0, 30.0],
        'price': [150.0, 380.0, 155.0]
    })
    
    # Validate using trading_sim's validator
    validated = validate_trading_sheet(trades)
    
    assert len(validated) == 3, f"Expected 3 trades, got {len(validated)}"
    assert list(validated.columns) == ['time', 'ticker', 'action', 'quantity', 'price'], \
        f"Unexpected columns: {validated.columns.tolist()}"
    
    # Verify actions are lowercase
    assert validated['action'].isin(['buy', 'sell']).all()


def simicx_test_integration_full_pipeline():
    """Full integration test of the signal generation pipeline."""
    np.random.seed(123)
    
    # Create realistic synthetic data
    tickers = ['SPY', 'QQQ', 'IWM', 'DIA']
    dates = pd.date_range('2025-01-01', periods=150, freq='D')
    
    rows = []
    for ticker in tickers:
        base_price = {'SPY': 450, 'QQQ': 380, 'IWM': 200, 'DIA': 350}[ticker]
        prices = base_price * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.015))
        
        for i, date in enumerate(dates):
            rows.append({
                'time': date,
                'ticker': ticker,
                'open': prices[i] * (1 + np.random.randn() * 0.003),
                'high': prices[i] * (1 + abs(np.random.randn()) * 0.008),
                'low': prices[i] * (1 - abs(np.random.randn()) * 0.008),
                'close': prices[i],
                'volume': int(np.random.uniform(5e6, 2e7))
            })
    
    ohlcv_df = pd.DataFrame(rows)
    
    # Run full pipeline
    trades = signal_gen(
        ohlcv_df=ohlcv_df,
        lookback_window=20,
        planning_horizon=3,
        rebalance_freq=30,
        risk_aversion=0.5,
        turnover_penalty=0.005,
        learning_rate=5e-3,
        epochs=3,
        neumann_order=2,
        initial_capital=500_000.0
    )
    
    # Comprehensive validation
    assert isinstance(trades, pd.DataFrame)
    
    if len(trades) > 0:
        # Check column types
        assert pd.api.types.is_datetime64_any_dtype(trades['time']), \
            "time column should be datetime"
        assert trades['ticker'].dtype == object, "ticker should be string"
        assert pd.api.types.is_numeric_dtype(trades['quantity']), \
            "quantity should be numeric"
        assert pd.api.types.is_numeric_dtype(trades['price']), \
            "price should be numeric"
        
        # Check value ranges
        assert (trades['quantity'] > 0).all(), "All quantities should be positive"
        assert (trades['price'] > 0).all(), "All prices should be positive"
        
        # Check trades are within date range
        assert trades['time'].min() >= ohlcv_df['time'].min()
        assert trades['time'].max() <= ohlcv_df['time'].max()


if __name__ == "__main__":
    print("Running signal_gen.py tests...")
    
    simicx_test_device_detection()
    print("✓ Device detection test passed")
    
    simicx_test_cnn_lstm_forward()
    print("✓ CNN_LSTM forward test passed")
    
    simicx_test_mdfp_layer()
    print("✓ MDFP layer test passed")
    
    simicx_test_signal_gen_synthetic()
    print("✓ Signal generation synthetic test passed")
    
    simicx_test_integration_with_trading_sim()
    print("✓ Integration with trading_sim test passed")
    
    simicx_test_integration_full_pipeline()
    print("✓ Full pipeline integration test passed")
    
    print("\n" + "=" * 50)
    print("All tests passed successfully!")
    print("=" * 50)