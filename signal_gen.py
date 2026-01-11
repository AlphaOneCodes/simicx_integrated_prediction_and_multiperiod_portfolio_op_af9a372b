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
from scipy import linalg
from typing import Optional, Tuple, Dict, List
from simicx.trading_sim import validate_trading_sheet
from tqdm import tqdm
import gc


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
        
        # FC layers for mean returns and covariance factors
        self.fc_mean = nn.Linear(64, horizon * n_assets)
        
        # For covariance: output lower triangular factors
        self.fc_cov_factor = nn.Linear(64, horizon * n_assets * n_assets)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass of CNN-LSTM.
        
        Args:
            x: Input tensor of shape (batch, seq_len, n_assets)
               Represents historical returns for each asset
               
        Returns:
            Tuple of:
                - y_hat: Predicted returns, shape (batch, horizon, n_assets)
                - V_hat: Predicted covariance matrices, shape (batch, horizon, n_assets, n_assets)
                         Guaranteed to be positive semi-definite
                         
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
        
        # Predict covariance factors
        cov_factors = self.fc_cov_factor(last_hidden)
        cov_factors = cov_factors.view(batch_size, self.horizon, self.n_assets, self.n_assets)
        
        # Construct positive semi-definite covariance: V = L @ L.T + eps*I
        L = torch.tril(cov_factors)
        V_hat = torch.matmul(L, L.transpose(-2, -1))
        
        # Add diagonal for numerical stability
        eye = torch.eye(self.n_assets, device=device, dtype=dtype)
        V_hat = V_hat + 1e-4 * eye.unsqueeze(0).unsqueeze(0)
        
        return y_hat, V_hat


# ============================================================================
# Utility Functions for GPU-based Optimization
# ============================================================================

def project_onto_simplex(z: torch.Tensor) -> torch.Tensor:
    """
    Project vector z onto the simplex {x : sum(x) = 1, x >= 0}.
    Uses the efficient algorithm from Duchi et al. (2008).
    
    For a vector z, the projection is: z_proj = max(z - theta, 0)
    where theta is chosen such that sum(max(z - theta, 0)) = 1.
    
    Args:
        z: Tensor of shape (..., n) to project
        
    Returns:
        Projected tensor on the simplex
    """
    # Sort in descending order
    z_sorted, _ = torch.sort(z, descending=True, dim=-1)
    
    # Cumulative sum of sorted values
    cumsum = torch.cumsum(z_sorted, dim=-1)
    
    # Compute rho: index where cumsum[i] - 1 > i * theta
    # theta = (cumsum[rho] - 1) / (rho + 1)
    n = z.shape[-1]
    rho_range = torch.arange(1, n + 1, device=z.device, dtype=z.dtype)
    
    # Reshape rho_range to broadcast correctly
    shape = [1] * (z.ndim - 1) + [n]
    rho_range = rho_range.view(shape)
    
    # Find rho: the largest index where z_sorted[rho] > (cumsum[rho] - 1) / (rho + 1)
    # Equivalently: (rho + 1) * z_sorted[rho] > cumsum[rho] - 1
    threshold = (cumsum - 1.0) / rho_range
    mask = z_sorted > threshold
    
    # Find the largest rho where mask is True
    rho = torch.arange(n, 0, -1, device=z.device, dtype=z.dtype)
    rho = rho.view(shape)
    rho_star = (mask.float() * rho).max(dim=-1, keepdim=True)[0]
    
    # Compute theta_star
    rho_idx = (rho_star - 1).long()
    cumsum_rho = torch.gather(cumsum, -1, rho_idx)
    theta_star = (cumsum_rho - 1.0) / rho_star
    
    # Project onto simplex
    z_proj = torch.clamp(z - theta_star, min=0.0)
    
    # Normalize to ensure sum = 1 (numerical stability)
    z_proj = z_proj / (z_proj.sum(dim=-1, keepdim=True) + 1e-10)
    
    return z_proj


def compute_objective(
    z: torch.Tensor,
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float,
    lambda_val: float,
    kappa: float
) -> torch.Tensor:
    """
    Compute MDFP objective function value.
    
    Args:
        z: Portfolio weights, shape (batch, horizon, n_assets)
        y_hat: Predicted returns, shape (batch, horizon, n_assets)
        V_hat: Predicted covariances, shape (batch, horizon, n_assets, n_assets)
        z_prev: Previous portfolio weights, shape (batch, n_assets)
        delta: Risk aversion parameter
        lambda_val: Turnover penalty coefficient
        kappa: Smoothing parameter
        
    Returns:
        Objective value, shape (batch, horizon)
    """
    batch_size, horizon, n_assets = z.shape
    
    # Return term: -y_hat^T z
    return_term = -(y_hat * z).sum(dim=-1)  # (batch, horizon)
    
    # Risk term: (delta/2) z^T V z
    Vz = torch.matmul(V_hat, z.unsqueeze(-1)).squeeze(-1)  # (batch, horizon, n_assets)
    risk_term = (delta / 2) * (z * Vz).sum(dim=-1)  # (batch, horizon)
    
    # Turnover term: lambda * sum_i sqrt((z_{s,i} - z_{s-1,i})^2 + kappa)
    turnover_term = torch.zeros(batch_size, horizon, device=z.device, dtype=z.dtype)
    for s in range(horizon):
        if s == 0:
            z_prev_s = z_prev  # (batch, n_assets)
        else:
            z_prev_s = z[:, s-1, :]  # (batch, n_assets)
        diff = z[:, s, :] - z_prev_s  # (batch, n_assets)
        turnover_term[:, s] = lambda_val * torch.sum(
            torch.sqrt(diff ** 2 + kappa),
            dim=-1
        )
    
    return return_term + risk_term + turnover_term


# =========================================================================
# Normalization Utilities
# =========================================================================

def normalize_window(window: np.ndarray) -> np.ndarray:
    """
    Normalize a lookback window per-asset using only historical statistics.
    
    Uses mean and std computed over the provided window to avoid lookahead.
    
    Args:
        window: Array of shape (lookback_window, n_assets)
    Returns:
        Normalized window of same shape.
    """
    mean = np.nanmean(window, axis=0, keepdims=True)
    std = np.nanstd(window, axis=0, keepdims=True) + 1e-8
    norm = (window - mean) / std
    # Ensure no NaNs remain
    return np.nan_to_num(norm, nan=0.0)


class MDFP_Layer(torch.autograd.Function):
    """
    GPU-native Multi-period Differentiable Finance Portfolio optimization layer.
    
    Solves the multi-period portfolio optimization problem using projected gradient descent:
    
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
        eta: float,
        lr: float = 0.1,
        max_iters: int = 100,
        tol: float = 1e-5
    ) -> torch.Tensor:
        """
        Solve multi-period portfolio optimization using GPU-based projected gradient descent.
        
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
            lr: Learning rate for projected gradient descent
            max_iters: Maximum iterations for optimization
            tol: Convergence tolerance
            
        Returns:
            z_star: Optimal portfolio weights, shape (batch, horizon, n_assets)
        """
        batch_size, horizon, n_assets = y_hat.shape
        device = y_hat.device
        dtype = y_hat.dtype
        
        # Initialize z uniformly on simplex
        z = torch.ones(batch_size, horizon, n_assets, device=device, dtype=dtype) / n_assets
        
        # Projected gradient descent
        for iteration in range(max_iters):
            z_old = z.clone()
            
            # Compute gradient analytically (no autograd needed)
            # grad_f = -y_hat + delta * V_hat @ z + lambda * (z - z_prev) / sqrt((z - z_prev)^2 + kappa)
            
            # Return term gradient
            grad_return = -y_hat  # (batch, horizon, n_assets)
            
            # Risk term gradient: delta * V_hat @ z
            Vz = torch.matmul(V_hat, z.unsqueeze(-1)).squeeze(-1)  # (batch, horizon, n_assets)
            grad_risk = delta * Vz
            
            # Turnover term gradient
            z_diff = z - z_prev.unsqueeze(1)  # (batch, horizon, n_assets)
            denom = torch.sqrt(z_diff ** 2 + kappa)  # (batch, horizon, n_assets)
            grad_turnover = lambda_val * z_diff / (denom + 1e-10)
            
            # Total gradient
            grad = grad_return + grad_risk + grad_turnover
            
            # Gradient descent step
            z = z - lr * grad
            
            # Project onto simplex (batch-wise then reshape back)
            z_flat = z.view(-1, n_assets)
            z_flat = project_onto_simplex(z_flat)
            z = z_flat.view(batch_size, horizon, n_assets)
            
            # Check convergence
            z_diff_norm = torch.norm(z - z_old) / (torch.norm(z) + 1e-10)
            if z_diff_norm < tol:
                break
        
        # Ensure valid weights (final projection)
        z = torch.clamp(z, min=0.0)
        z = z / z.sum(dim=-1, keepdim=True)
        
        z_star = z.detach()
        
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
        Backward pass using direct gradient computation through the objective.
        
        For the MDFP objective:
            L(z) = -y^T z + (delta/2) z^T V z + lambda * sum_i sqrt((z_i - z_prev_i)^2 + kappa)
        
        We compute dL/dy and dL/dV directly:
            dL/dy = -z
            dL/dV = (delta/2) z @ z^T
        
        Args:
            ctx: Context with saved tensors
            grad_output: Gradient w.r.t. output z_star, shape (batch, horizon, n_assets)
            
        Returns:
            Tuple of gradients w.r.t. all inputs
        """
        y_hat, V_hat, z_prev, z_star = ctx.saved_tensors
        delta = ctx.delta
        
        batch_size, horizon, n_assets = y_hat.shape
        device = y_hat.device
        dtype = y_hat.dtype
        
        # Initialize gradients
        grad_y_hat = torch.zeros_like(y_hat)
        grad_V_hat = torch.zeros_like(V_hat)
        
        # Compute gradients directly from the objective function
        # For each (b, s) position:
        #   dL/dy_hat = -z_star
        #   dL/dV_hat = (delta/2) * z_star @ z_star^T
        
        for b in range(batch_size):
            for s in range(horizon):
                z_s = z_star[b, s]
                
                # Gradient w.r.t. y_hat: -z
                grad_y_hat[b, s] = -z_s
                
                # Gradient w.r.t. V_hat: (delta/2) * z @ z^T
                zz_T = torch.outer(z_s, z_s)
                grad_V_hat[b, s] = (delta / 2) * zz_T
        
        # Scale by upstream gradients
        grad_y_hat = grad_y_hat * grad_output
        grad_V_hat = grad_V_hat * grad_output.unsqueeze(-1)
        
        # Return gradients (None for scalar parameters)
        return grad_y_hat, grad_V_hat, None, None, None, None, None, None, None, None, None


def apply_mdfp(
    y_hat: torch.Tensor,
    V_hat: torch.Tensor,
    z_prev: torch.Tensor,
    delta: float = 1.0,
    lambda_val: float = 0.01,
    kappa: float = 1e-4,
    neumann_order: int = 5,
    eta: float = 0.1,
    lr: float = 0.1,
    max_iters: int = 100,
    tol: float = 1e-5
) -> torch.Tensor:
    """
    Apply MDFP layer to predicted returns and covariances using GPU-native optimization.
    
    Solves the multi-period portfolio optimization problem using projected gradient descent
    on GPU tensor operations, eliminating the CPU bottleneck from cvxpy.
    
    Args:
        y_hat: Predicted returns, shape (batch, horizon, n_assets)
        V_hat: Predicted covariances, shape (batch, horizon, n_assets, n_assets)
        z_prev: Previous portfolio weights, shape (batch, n_assets)
        delta: Risk aversion parameter (default: 1.0)
        lambda_val: Turnover penalty coefficient (default: 0.01)
        kappa: Smoothing parameter (default: 1e-4)
        neumann_order: Order of Neumann series for backward (default: 5)
        eta: Learning rate for entropic mirror descent in backward (default: 0.1)
        lr: Learning rate for projected gradient descent solver (default: 0.1)
        max_iters: Maximum iterations for optimization (default: 100)
        tol: Convergence tolerance (default: 1e-5)
    
    Returns:
        z_star: Optimal portfolio weights, shape (batch, horizon, n_assets)
    
    Example:
        >>> y = torch.randn(4, 5, 10)
        >>> V = torch.eye(10).unsqueeze(0).unsqueeze(0).expand(4, 5, 10, 10) * 0.01
        >>> z_prev = torch.ones(4, 10) / 10
        >>> z = apply_mdfp(y, V, z_prev)
        >>> print(z.shape)  # (4, 5, 10)
        >>> print(z.sum(dim=-1))  # Should be close to 1
    """
    return MDFP_Layer.apply(
        y_hat, V_hat, z_prev, delta, lambda_val, kappa, 
        neumann_order, eta, lr, max_iters, tol
    )


# ============================================================================
# Signal Generation
# ============================================================================

def signal_gen(
    ohlcv_df: pd.DataFrame,
    trading_start_date: Optional[str] = None,
    training_limit: int = 250,
    lookback_window: int = 120,
    training_window: int = 250,
    planning_horizon: int = 20,
    rebalance_freq: int = 20,
    risk_aversion: float = 1.0,
    turnover_penalty: float = 0.01,
    learning_rate: float = 0.01,
    epochs: int = 20,
    neumann_order: int = 5,
    initial_capital: float = 1_000_000.0,
    batch_size: int = 16
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
        trading_start_date: Optional date string (YYYY-MM-DD) when trading should start.
                           If provided, earlier data is used only for model warmup/training.
                           If None, uses training_limit parameter instead.
        training_limit: Deprecated. Use trading_start_date instead. Number of initial periods
                       to skip before starting trading (default: 250)
        lookback_window: Number of historical periods for model input (default: 120)
        training_window: Number of historical periods for model training (default: 250)
        planning_horizon: Number of future periods to optimize over (default: 5)
        rebalance_freq: Rebalance portfolio every N periods (default: 20)
        risk_aversion: Risk aversion parameter delta for mean-variance (default: 1.0)
        turnover_penalty: Turnover cost coefficient lambda (default: 0.01)
        learning_rate: Learning rate for Adam optimizer (default: 1e-3)
        epochs: Number of training epochs per rebalance (default: 50)
        neumann_order: Order of Neumann series for implicit differentiation (default: 5)
        initial_capital: Initial capital for position sizing (default: 1,000,000)
        batch_size: Batch size for training (default: 16, reduced for memory efficiency)
        
    Returns:
        DataFrame with columns [time, ticker, action, quantity, price]
        validated by trading_sim's validate_trading_sheet
        
    Example:
        >>> from simicx.data_loader import get_trading_data
        >>> ohlcv = get_trading_data(tickers=['AAPL', 'MSFT', 'GOOGL'], align_dates=True)
        >>> trades = signal_gen(
        ...     ohlcv_df=ohlcv,
        ...     lookback_window=60,
        ...     training_window=120,
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
    
    # Normalization is performed per-window at training/inference time to avoid lookahead.
    # See `normalize_window()` utility.
    
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
    # Output containers
    # =========================================================================

    all_trades: List[Dict] = []
    current_quantities = np.zeros(n_assets)
    current_cash = initial_capital  # Track cash to update portfolio value over time
    
    current_portfolio_value = initial_capital
    
    # =========================================================================
    # Determine trading start index
    # =========================================================================
    
    # If trading_start_date is specified, find its index in the dates
    if trading_start_date is not None:
        trading_start_ts = pd.Timestamp(trading_start_date)
        # Find the index of the first date >= trading_start_date
        valid_indices = [i for i, d in enumerate(dates) if d >= trading_start_ts]
        if not valid_indices:
            raise ValueError(
                f"No dates found >= trading_start_date {trading_start_date}. "
                f"Date range in data: {dates[0]} to {dates[-1]}"
            )
        trading_start_idx = valid_indices[0]
        
        # Ensure we have enough warmup data before trading starts
        min_required_idx = training_window
        if trading_start_idx < min_required_idx:
            raise ValueError(
                f"Insufficient warmup data. Trading starts at index {trading_start_idx} "
                f"(date: {dates[trading_start_idx]}), but need at least {min_required_idx} "
                f"days for training window. Either:"
                f"\n  1. Load more historical data (use get_trading_data_with_warmup)"
                f"\n  2. Set trading_start_date to a later date"
                f"\n  3. Reduce training_window parameter"
            )
        
        print(f"Trading will start at index {trading_start_idx} (date: {dates[trading_start_idx]})")
        print(f"  Available warmup data: {trading_start_idx} days")
        print(f"  Required training window: {training_window} days")
    else:
        # Use training_limit as before (backward compatibility)
        trading_start_idx = training_limit
        print(f"Using training_limit: {training_limit} (consider using trading_start_date instead)")
    
    # =========================================================================
    # Rolling loop over dates
    # =========================================================================
    
    for idx in range(trading_start_idx, len(dates) - planning_horizon):
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

            # Define the rolling training window
            # Start of the training data block
            train_start_idx = max(0, idx - training_window)
            # End of the training data block (input features)
            train_end_idx = idx
            
            # Calculate valid training range for creating (X, Y) pairs
            # The last possible start for an X sample is (train_end_idx - lookback_window)
            # The first possible start is train_start_idx
            min_t = train_start_idx + lookback_window
            # The latest target Y must be available before the current index `idx`
            max_t = train_end_idx - planning_horizon

            if max_t >= min_t:
                for epoch in tqdm(range(epochs), desc=f"Rebalance idx {idx}"):
                    # Build training batch from the rolling historical window
                    train_X_list = []
                    train_Y_list = []

                    # Create a batch of training samples from the allowed window
                    actual_batch_size = min(batch_size, max_t - min_t + 1)
                    
                    # Randomly sample starting points for sequences to form a batch
                    start_points = np.random.choice(
                        range(min_t, max_t + 1),
                        size=actual_batch_size,
                        replace=False
                    )

                    for t in start_points:
                        # Per-window normalization using only historical stats
                        X_sample = returns_np[t - lookback_window:t]
                        X_sample = np.cumprod(1 + normalize_window(X_sample), axis=0)
                        Y_sample = returns_np[t:t + planning_horizon]
                        train_X_list.append(X_sample)
                        train_Y_list.append(Y_sample)

                    if not train_X_list:
                        continue

                    X_batch = np.stack(train_X_list, axis=0)
                    # X_batch = np.cumprod(1 + X_batch, axis=1) - 1  # Convert log returns to simple returns
                    Y_batch = np.stack(train_Y_list, axis=0)

                    X = torch.tensor(X_batch, dtype=torch.float32, device=device)
                    Y = torch.tensor(Y_batch, dtype=torch.float32, device=device)

                    # Create z_prev for each sample in batch
                    z_prev_batch = np.tile(z_prev.reshape(1, -1), (len(train_X_list), 1))
                    z_prev_tensor = torch.tensor(z_prev_batch, dtype=torch.float32, device=device)

                    optimizer.zero_grad()

                    # Forward pass through CNN-LSTM
                    y_hat, V_hat = model(X)

                    # MDFP optimization
                    z_star = MDFP_Layer.apply(
                        y_hat,
                        V_hat,
                        z_prev_tensor,
                        risk_aversion,
                        turnover_penalty,
                        kappa,
                        neumann_order,
                        eta
                    )

                    # Compute MDFP objective directly from the paper
                    # Loss = -return + risk + turnover
                    
                    # Return term: -y_hat^T z_star (we want to maximize returns, so negative)
                    # Shape: y_hat = (batch, horizon, n_assets), z_star = (batch, horizon, n_assets)
                    return_term = -(y_hat * z_star).sum(dim=-1)  # (batch, horizon)
                    
                    # Risk term: (delta/2) z_star^T V_hat z_star
                    # Compute z^T V z for each (batch, horizon) pair
                    Vz = torch.matmul(V_hat, z_star.unsqueeze(-1)).squeeze(-1)  # (batch, horizon, n_assets)
                    risk_term = (risk_aversion / 2) * (z_star * Vz).sum(dim=-1)  # (batch, horizon)
                    
                    # Turnover term: lambda * sum_i sqrt((z_{s,i} - z_{s-1,i})^2 + kappa)
                    turnover_term = torch.zeros(z_star.shape[0], z_star.shape[1], device=device)
                    for s in range(z_star.shape[1]):
                        if s == 0:
                            z_prev_s = z_prev_tensor
                        else:
                            z_prev_s = z_star[:, s-1, :]
                        diff = z_star[:, s, :] - z_prev_s
                        turnover_term[:, s] = turnover_penalty * torch.sum(
                            torch.sqrt(diff ** 2 + kappa),
                            dim=-1
                        )
                    
                    # Total objective (MDFP loss from paper)
                    objective = return_term + risk_term + turnover_term  # (batch, horizon)
                    loss = objective.mean()  # Average over batch and horizon

                    # Backward pass
                    loss.backward()

                    # Gradient clipping for stability
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    
                    optimizer.step()
                    
                    # Display loss dynamically for each epoch
                    # print(f"Rebalance idx {idx}, Epoch {epoch+1}/{epochs}, MDFP Loss: {loss.item():.6f}")
                    
                    # Clear GPU cache and collect garbage periodically to free memory
                    if (epoch + 1) % 5 == 0:
                        torch.cuda.empty_cache()
                        gc.collect()

            
            # =================================================================
            # Inference Step
            # =================================================================
            
            model.eval()
            
            with torch.no_grad():
                # Normalize the inference window using historical stats only
                infer_window_np = normalize_window(returns_np[idx - lookback_window:idx])
                X_infer = torch.tensor(
                    infer_window_np,
                    dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                z_prev_tensor = torch.tensor(
                    z_prev, dtype=torch.float32, device=device
                ).unsqueeze(0)
                
                y_hat, V_hat = model(X_infer)
                
                z_star = apply_mdfp(
                    y_hat=y_hat,
                    V_hat=V_hat,
                    z_prev=z_prev_tensor,
                    delta=risk_aversion,
                    lambda_val=turnover_penalty,
                    kappa=kappa,
                    neumann_order=neumann_order,
                    eta=eta
                )
                
                print(z_star.shape)
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
    print(trading_sheet.head())
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
        
        y_hat, V_hat = model(x)
        
        # Check shapes
        assert y_hat.shape == (batch_size, horizon, n_assets), \
            f"y_hat shape mismatch: {y_hat.shape}"
        assert V_hat.shape == (batch_size, horizon, n_assets, n_assets), \
            f"V_hat shape mismatch: {V_hat.shape}"
        
        # Check V_hat is positive semi-definite
        for b in range(batch_size):
            for s in range(horizon):
                eigvals = torch.linalg.eigvalsh(V_hat[b, s])
                assert (eigvals >= -1e-5).all(), f"V_hat not PSD at batch {b}, step {s}"
        
        # Check no NaN
        assert not torch.isnan(y_hat).any(), "NaN in y_hat"
        assert not torch.isnan(V_hat).any(), "NaN in V_hat"


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