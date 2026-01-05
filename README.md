# Project Documentation

**Author**: SimicX AI Quant  
**Copyright**: (C) 2025-2026 SimicX. All rights reserved.  
**Generated**: 2026-01-05 11:28

## Overview

This project consists of a high-fidelity quantitative trading and backtesting engine designed to develop, optimize, and validate predictive financial strategies. The system is engineered to generate actionable trading signals across a multi-day horizon, focusing on out-of-sample performance for a specified universe of assets. By strictly separating training and execution phases, the framework ensures a rigorous evaluation of alpha-generating models while maintaining high standards of data integrity and financial realism.

The methodology centers on an iterative, vectorized architecture that processes comprehensive historical datasets without lookahead bias. To ensure execution realism, the strategy utilizes mid-price estimates and delayed execution logic—executing trades at next-day opening prices rather than same-day closes. The engine incorporates strict risk management constraints, including a no-leverage policy and capital-weighted position sizing. Furthermore, the pipeline is optimized for high-performance computing, leveraging multi-GPU acceleration (dual NVIDIA RTX 3060 configuration) for model training and hyperparameter optimization, while implementing robust data-cleaning protocols to handle financial noise and numerical instabilities.

The primary outputs of this project include a standardized hyperparameter registry and a comprehensive trading ledger covering the full backtest period. The system produces optimized parameter sets derived from historical training data, which are then applied to out-of-sample trading simulations to measure predictive accuracy and portfolio performance. Detailed execution logs provide transparency into trade rejections, cash flow management, and total returns, offering a validated assessment of the strategy’s viability in a simulated production environment.

## Implementation Plan

### Progress
- Total: 3 | Done: 3 | In Progress: 0 | Failed: 0

### Verification Order
The following files will be executed (in order) to verify the generated code works:
`tune.py -> main.py`

### Files
| Status | Verified | File | Description | Dependencies |
|--------|----------|------|-------------|--------------|
| [x] | - | `signal_gen.py` | Core strategy logic containing the IPMO class, Neural Network, MDFP optimization layer, and the rolling signal generation loop. | simicx/trading_sim.py, simicx/data_loader.py |
| [x] | ✓ | `tune.py` | Hyperparameter optimization script running on Phase 1 training data. | signal_gen.py, simicx/trading_sim.py, simicx/data_loader.py |
| [x] | ✓ | `main.py` | Production execution script running on Phase 2 trading data. | signal_gen.py, simicx/trading_sim.py, simicx/data_loader.py |


### Progress Log

#### `signal_gen.py`
Completed successfully
Fixed review issue: Critical Lookahead Bias in Online Training Loop. The strategy trains the model on the current window's future returns (Y_np derived from 'future_start = idx') immediately before generating a prediction for that same window. The model effectively 'peeks' at the answer for the current timeframe, updates its weights, and then predicts, leading to 100% unrealistic accuracy.
Fixed review issue: Static Capital Assumption prevents compounding and risks invalid orders. `signal_gen` calculates target positions based on a constant `initial_capital` (1,000,000) at every step. It does not track the actual evolving portfolio equity. If the portfolio value drops, it will generate buy orders exceeding available cash (rejected by sim). If value rises, it effectively de-leverages.

#### `tune.py`
Completed successfully

#### `main.py`
Audit failed: CRITICAL ERRORS FOUND:

1. **DUPLICATE FUNCTION DEFINITION**: There are TWO `main()` function definitions in the code (lines ~200-213 and ~216-229). The second definition overwrites the first, which is a code quality issue.

2. **ATTRIBUTE ACCESS ERROR**: The second `main()` function uses `args.phase` (attribute access) but `parse_args()` returns a dictionary, so it should be `args["phase"]` (dictionary access). The first `main()` correctly uses `args["phase"]`, but since the second definition overwrites it, running the code will raise `AttributeError: 'dict' object has no attribute 'phase'`.

All other checks PASSED:
- All 5 instruction features implemented correctly
- API parameter names match exactly (signal_gen, trading_sim, get_trading_data)
- String literals case-matched ('limited', 'full', 'buy')
- No invented parameters - all are in API reference
- Imports are correct and consistent
- Data path 'simicx/alpha_config.json' matches DATA ASSETS
- get_trading_data called with correct params (tickers, align_dates)

FIX REQUIRED: Remove the duplicate `main()` function definition (keep only the first one that uses `args["phase"]`).
Completed successfully
Fixed review issue: Data Pipeline Cold Start / Missing Warmup. `get_trading_data` loads data strictly from 2025-01-01. `signal_gen` requires `lookback_window` (120 days) to initialize the first window. Consequently, the strategy will produce NO signals for the first ~4 months of 2025. Additionally, since weights are initialized randomly and the online learning needs history, the model starts completely untrained in Phase 2.



## Verification Log

| File | Result | Duration | Notes |
|------|--------|----------|-------|
| `tune.py` | ✓ Passed | 88.4s |  |
| `main.py` | ✓ Passed | 1552.4s |  |



## API Reference

### `main.py`

> **Import**: `from main import ...`

> Production Execution Script for CNN-LSTM Portfolio Optimization.

This module implements the production pipeline for the portfolio optimization
system, loading tuned hyperparameters and executing the trading strategy
on out-of-sample data (2025 onwards).

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT


**`load_alpha_config`**
```python
def load_alpha_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]
```
> Load alpha configuration containing ticker lists.

Args:
    config_path: Path to the alpha configuration JSON file.
    
Returns:
    Dictionary containing configuration with LIMITED_TICKERS and FULL_TICKERS.
    
Raises:
    FileNotFoundError: If configuration file does not exist.
    json.JSONDecodeError: If configuration file contains invalid JSON.
    
Example:
    >>> config = load_alpha_config()
    >>> 'LIMITED_TICKERS' in config
    True
    >>> 'FULL_TICKERS' in config
    True

**`load_best_params`**
```python
def load_best_params(params_path: Path = None) -> Dict[str, Any]
```
> Load best hyperparameters from tuning phase.

Args:
    params_path: Path to the best_params.json file. If None, uses path
                 relative to this script's directory.

Returns:
    Dictionary containing tuned hyperparameters:
        - planning_horizon: int - prediction horizon H
        - risk_aversion: float - delta parameter
        - turnover_penalty: float - lambda parameter
        - learning_rate: float
        - rebalance_freq: int
        - lookback_window: int
        - epochs: int
        - neumann_order: int

Raises:
    FileNotFoundError: If best_params.json does not exist.
        This indicates tuning phase has not been run.
    json.JSONDecodeError: If file contains invalid JSON.
    KeyError: If required parameters are missing.

Example:
    >>> params = load_best_params()
    >>> params['planning_horizon']
    5
    >>> params['risk_aversion']
    1.0

**`get_tickers_for_phase`**
```python
def get_tickers_for_phase(phase: str, config: Dict[str, Any]) -> List[str]
```
> Get ticker list based on phase.

Args:
    phase: Either 'limited' or 'full'.
    config: Alpha configuration dictionary containing ticker lists.
    
Returns:
    List of ticker symbols for the specified phase.
    
Raises:
    ValueError: If phase is not 'limited' or 'full'.
    KeyError: If config missing required ticker list.
    
Example:
    >>> config = {'LIMITED_TICKERS': ['SPY', 'QQQ'], 'FULL_TICKERS': ['SPY', 'QQQ', 'AAPL']}
    >>> get_tickers_for_phase('limited', config)
    ['SPY', 'QQQ']
    >>> get_tickers_for_phase('full', config)
    ['SPY', 'QQQ', 'AAPL']

**`run_production`**
```python
def run_production(phase: str) -> Dict[str, Any]
```
> Run production trading pipeline.

Executes the full production pipeline:
1. Load configuration and best parameters
2. Fetch trading data for specified phase
3. Generate trading signals using CNN-LSTM + MDFP
4. Run trading simulation
5. Report and save results

Args:
    phase: Data phase to use - 'limited' or 'full'.
    
Returns:
    Dictionary containing:
        - sharpe_ratio: float - Annualized Sharpe ratio
        - total_return: float - Total return over period
        - pnl: float - Total profit/loss
        - trading_sheet_path: str - Path to saved trading sheet
        - pnl_path: str - Path to saved PnL details
        
Raises:
    FileNotFoundError: If configuration or best_params.json not found.
    ValueError: If phase is invalid or data issues occur.
    
Example:
    >>> results = run_production(phase='limited')
    >>> print(f"Sharpe: {results['sharpe_ratio']:.3f}")
    >>> print(f"Return: {results['total_return']*100:.2f}%")

**`parse_args`**
```python
def parse_args() -> Dict[str, str]
```
> Parse command line arguments.

Returns:
    Dictionary with parsed arguments:
        - phase: str - Either 'limited' or 'full'

Example:
    >>> # From command line: python main.py --phase limited
    >>> args = parse_args()
    >>> args['phase']
    'limited'

**`main`**
```python
def main() -> None
```
> Main entry point for production execution.

Parses CLI arguments and runs the production trading pipeline.

Example:
    $ python main.py --phase limited
    $ python main.py --phase full

**`simicx_test_load_alpha_config`**
```python
def simicx_test_load_alpha_config()
```
> Test loading alpha configuration.

**`simicx_test_load_best_params`**
```python
def simicx_test_load_best_params()
```
> Test loading and validating best parameters.

**`simicx_test_get_tickers_for_phase`**
```python
def simicx_test_get_tickers_for_phase()
```
> Test ticker selection by phase.

**`simicx_test_integration_with_deps`**
```python
def simicx_test_integration_with_deps()
```
> Integration test exercising dependency interfaces with minimal data.

---

### `signal_gen.py`

> **Import**: `from signal_gen import ...`

> 
Signal Generation Module with CNN-LSTM and Multi-Period Differentiable Portfolio Optimization.

This module implements:
- CNN-LSTM neural network for return prediction
- MDFP (Multi-period Differentiable Finance Portfolio) optimization layer
- Rolling signal generation with periodic rebalancing

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT


**`get_device`**
```python
def get_device() -> torch.device
```
> Detect and return the best available device (CUDA > MPS > CPU).

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

**class `CNN_LSTM`**
```python
class CNN_LSTM(nn.Module):
```
> CNN-LSTM model for multi-asset return prediction.

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

**`CNN_LSTM.__init__`**
```python
def __init__(self, n_assets: int, horizon: int)
```
**`CNN_LSTM.forward`**
```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]
```
> Forward pass of CNN-LSTM.

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

**class `MDFP_Layer`**
```python
class MDFP_Layer(torch.autograd.Function):
```
> Multi-period Differentiable Finance Portfolio optimization layer.

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

**`MDFP_Layer.forward`**
```python
def forward(ctx, y_hat: torch.Tensor, V_hat: torch.Tensor, z_prev: torch.Tensor, delta: float, lambda_val: float, kappa: float, neumann_order: int, eta: float) -> torch.Tensor
```
> Solve multi-period portfolio optimization using cvxpy.

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

**`MDFP_Layer.backward`**
```python
def backward(ctx, grad_output: torch.Tensor)
```
> Backward pass using Neumann series approximation for implicit differentiation.

Implements:
    dz*/d_theta ≈ sum_{b=0}^{B} [d_z Phi]^b @ (d_Phi/d_theta)

where Phi(z) = softmax(ln(z) - eta * grad_f(z)) is the entropic mirror descent map.

Args:
    ctx: Context with saved tensors
    grad_output: Gradient w.r.t. output z_star, shape (batch, horizon, n_assets)
    
Returns:
    Tuple of gradients w.r.t. all inputs (y_hat, V_hat, and None for scalars)

**`compute_grad_f`**
```python
def compute_grad_f(z: torch.Tensor, y: torch.Tensor, V: torch.Tensor, z_prev_step: torch.Tensor) -> torch.Tensor
```
> Compute gradient of objective f w.r.t. z.

grad_f = -y + delta * V @ z + lambda * (z - z_prev) / sqrt((z - z_prev)^2 + kappa)

**`entropic_mirror_descent_map`**
```python
def entropic_mirror_descent_map(z: torch.Tensor, y: torch.Tensor, V: torch.Tensor, z_prev_step: torch.Tensor) -> torch.Tensor
```
> Entropic mirror descent map: Phi(z) = softmax(ln(z) - eta * grad_f(z))

**`apply_mdfp`**
```python
def apply_mdfp(y_hat: torch.Tensor, V_hat: torch.Tensor, z_prev: torch.Tensor, delta: float, lambda_val: float, kappa: float, neumann_order: int = 5, eta: float = 0.1) -> torch.Tensor
```
> Apply MDFP optimization layer (convenience wrapper).

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

**`signal_gen`**
```python
def signal_gen(ohlcv_df: pd.DataFrame, lookback_window: int = 120, planning_horizon: int = 5, rebalance_freq: int = 20, risk_aversion: float = 1.0, turnover_penalty: float = 0.01, learning_rate: float = 0.001, epochs: int = 50, neumann_order: int = 5, initial_capital: float = 1000000.0) -> pd.DataFrame
```
> Generate trading signals using CNN-LSTM and MDFP optimization.

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

**`simicx_test_device_detection`**
```python
def simicx_test_device_detection()
```
> Test hardware device detection.

**`simicx_test_cnn_lstm_forward`**
```python
def simicx_test_cnn_lstm_forward()
```
> Test CNN_LSTM model forward pass with various configurations.

**`simicx_test_mdfp_layer`**
```python
def simicx_test_mdfp_layer()
```
> Test MDFP optimization layer constraints and output validity.

**`simicx_test_signal_gen_synthetic`**
```python
def simicx_test_signal_gen_synthetic()
```
> Test signal_gen with synthetic OHLCV data.

**`simicx_test_integration_with_trading_sim`**
```python
def simicx_test_integration_with_trading_sim()
```
> Integration test with trading_sim validation function.

**`simicx_test_integration_full_pipeline`**
```python
def simicx_test_integration_full_pipeline()
```
> Full integration test of the signal generation pipeline.

---

### `simicx/data_loader.py`

> **Import**: `from simicx.data_loader import ...`

> 
SimicX Data Loader Module (Database Version)

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Centralized OHLCV data loading from MongoDB with strict temporal controls
for alpha discovery and backtesting.

Key Features:
- Strict train/test split: Training ≤ 2024-12-31, Trading ≥ 2025-01-01
- Date alignment across tickers (consistent coverage)
- 2-phase testing support: LIMITED → FULL tickers
- Multi-asset extensibility (not equity-specific)

Usage:
    from data_loader import get_training_data, get_trading_data
    
    # For tune.py (hyperparameter optimization)
    train_df = get_training_data(LIMITED_TICKERS, years_back=3)
    
    # For main.py (backtesting)
    trade_df = get_trading_data(FULL_TICKERS)


**`get_mongo_client`**
```python
def get_mongo_client() -> MongoClient
```
> Get or create MongoDB client connection (thread-safe singleton).

Returns:
    MongoClient instance with connection pooling.

Raises:
    RuntimeError: If connection to MongoDB fails (Fail Fast).

Example:
    >>> client = get_mongo_client()
    >>> db = client[MONGODB_DATABASE]

**`get_collection`**
```python
def get_collection()
```
> Get OHLCV collection instance.

Returns:
    MongoDB collection for OHLCV data.

**`get_tickers`**
```python
def get_tickers() -> List[str]
```
> Get list of unique ticker symbols available in the database.

Returns:
    List[str]: Sorted list of available ticker symbols.

Example:
    >>> tickers = get_tickers()
    >>> print(f"Found {len(tickers)} tickers")
    >>> 'SPY' in tickers
    True

**`get_date_range`**
```python
def get_date_range(ticker: str) -> Tuple[datetime, datetime]
```
> Get the date range (start and end dates) for a specific ticker.

Args:
    ticker: Stock ticker symbol (e.g., 'SPY', 'AAPL').

Returns:
    Tuple[datetime, datetime]: (start_date, end_date) for the ticker.

Raises:
    ValueError: If the ticker is not found in the database.

Example:
    >>> start, end = get_date_range('SPY')
    >>> print(f"SPY data: {start.date()} to {end.date()}")

**`get_data`**
```python
def get_data(ticker: Optional[str] = None, tickers: Optional[List[str]] = None, phase: Optional[str] = None, start_date: Optional[str] = None, end_date: Optional[str] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get OHLCV data with optional filtering by ticker(s), phase, and date range.

Args:
    ticker: Single ticker symbol (e.g., 'SPY'). Mutually exclusive with tickers/phase.
    tickers: List of ticker symbols. Mutually exclusive with ticker/phase.
    phase: 'limited' (LIMITED_TICKERS) or 'full' (FULL_TICKERS). Overrides ticker/tickers.
    start_date: Start date (inclusive) in 'YYYY-MM-DD' format.
    end_date: End date (inclusive) in 'YYYY-MM-DD' format.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: OHLCV data with columns: time, ticker, open, high, low, close, volume

Raises:
    ValueError: If neither phase, ticker nor tickers is provided.

**`get_training_data`**
```python
def get_training_data(tickers: Optional[List[str]] = None, phase: Optional[str] = None, years_back: Optional[int] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get training/tuning data (all data up to and including 2024-12-31).

CRITICAL: This function ensures NO data after 2024-12-31 is included.

Args:
    tickers: List of ticker symbols. Defaults to FULL_TICKERS if phase not set.
    phase: 'limited' or 'full'. Sets tickers and default years_back from config.
    years_back: Override default years_back.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: Training OHLCV data.

**`get_trading_data`**
```python
def get_trading_data(tickers: Optional[List[str]] = None, align_dates: bool = True) -> pd.DataFrame
```
> Get trading simulation data (all data from start of 2025 onwards).

CRITICAL: This function ensures ONLY data from 2025-Jan-01 onwards is returned,
which should be used for backtesting and performance reporting.

Args:
    tickers: List of ticker symbols. Defaults to FULL_TICKERS.
    align_dates: If True, only return dates where ALL tickers have data.

Returns:
    pd.DataFrame: Trading OHLCV data (2025 onwards).

Example:
    >>> # Trading data for backtesting
    >>> trade_df = get_trading_data(FULL_TICKERS)
    >>> trade_df['time'].min()  # Should be >= 2025-Jan-01

**`get_ohlcv`**
```python
def get_ohlcv(ticker: str, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame
```
> Convenience alias for get_data() - get OHLCV data for a single ticker.

Args:
    ticker: Stock ticker symbol.
    start_date: Optional start date.
    end_date: Optional end date.

Returns:
    pd.DataFrame: OHLCV data for the specified ticker.

**`simicx_test_data_loader`**
```python
def simicx_test_data_loader()
```
> Test function for data_loader module.

Verifies:
1. Database connectivity
2. Ticker availability
3. Date range queries
4. Temporal split integrity (training ≤ 2024, trading ≥ 2025)
5. Date alignment across tickers

---

### `simicx/trading_sim.py`

> **Import**: `from simicx.trading_sim import ...`

> 
SimicX Trading Simulation Module

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT

Comprehensive backtesting engine for trading strategies with realistic fee modeling,
position tracking, constraint validation, and performance metrics.

Usage:
    from alpha_stream.tools.trading_sim import trading_sim
    
    pnl, pnl_details = trading_sim(
        trading_sheet=trading_df,
        initial_capital=1_000_000.0
    )


**class `Position`**
```python
class Position:
```
> Represents a position in a single asset with FIFO (First-In-First-Out) cost basis tracking.

This class manages both long and short positions for a single security ticker,
tracking individual lots for accurate cost basis and P&L calculations.

Position Types:
    - **Long positions**: positive quantity (bought stock, expecting price increase)
    - **Short positions**: negative quantity (borrowed and sold stock, expecting price decrease)

FIFO Accounting:
    When selling/covering positions, the oldest lots are consumed first. This provides
    accurate cost basis tracking for tax purposes and precise realized P&L calculations.

Attributes:
    ticker: The asset ticker (e.g., 'AAPL', 'DIA')
    lots: List of (quantity, price_per_share) tuples. Each tuple represents a lot:
          - quantity > 0 for long lots (shares owned)
          - quantity < 0 for short lots (shares owed)
          - price is the cost basis per share (including commission for buys)

Example:
    >>> # Create a new position and add lots
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Buy 100 shares at $150
    >>> pos.add(50, 155.0)   # Buy 50 more at $155
    >>> 
    >>> # Check position state
    >>> pos.quantity  # Total shares: 150
    150.0
    >>> pos.avg_cost  # Weighted average: (100*150 + 50*155) / 150 = 151.67
    151.66666666666666
    >>> pos.market_value  # Total cost basis: 100*150 + 50*155 = 22750
    22750.0
    >>> 
    >>> # Sell using FIFO - removes from oldest lot first
    >>> removed_qty, cost_basis = pos.remove(75)  # Sell 75 shares
    >>> removed_qty  # Actually removed
    75.0
    >>> cost_basis  # Cost basis of sold shares (75 * $150 from first lot)
    11250.0
    >>> pos.lots  # Remaining: 25 shares at $150, 50 shares at $155
    [(25, 150.0), (50, 155.0)]
    
    >>> # Short position example
    >>> short_pos = Position(ticker='TSLA')
    >>> short_pos.add(-100, 200.0)  # Short 100 shares at $200 (net proceeds per share)
    >>> short_pos.quantity  # Negative = owe shares
    -100

**`Position.quantity`**
```python
def quantity(self) -> float
```
> Total quantity held across all lots.

Returns:
    float: Sum of all lot quantities.
           Positive for net long position (own shares).
           Negative for net short position (owe shares).
           Zero if no position or balanced.

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Long 100
    >>> pos.add(-30, 155.0)  # Short 30 (partially close)
    >>> pos.quantity
    70.0

**`Position.avg_cost`**
```python
def avg_cost(self) -> float
```
> Weighted average cost basis per share across all lots.

Calculates the volume-weighted average price of all open lots.
For long positions, this represents the average purchase price.
For short positions, this represents the average sale price (proceeds received).

Returns:
    float: Average cost per share. Returns 0.0 if no position held.

Formula:
    avg_cost = Σ(quantity_i × price_i) / Σ(quantity_i)

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # 100 × $150 = $15,000
    >>> pos.add(50, 160.0)   # 50 × $160 = $8,000
    >>> pos.avg_cost  # ($15,000 + $8,000) / 150 = $153.33
    153.33333333333334
    
    >>> # With no position
    >>> empty_pos = Position(ticker='MSFT')
    >>> empty_pos.avg_cost
    0.0

**`Position.market_value`**
```python
def market_value(self) -> float
```
> Total cost basis of the position (quantity × cost per share for each lot).

This represents the total capital invested in the position. For unrealized
P&L calculation, compare this with `quantity × current_market_price`.

Returns:
    float: Sum of (quantity × price) for all lots.
           Positive for long positions (capital deployed).
           Negative for short positions (proceeds received).

Note:
    This is NOT the current market value. Use with current prices:
    `unrealized_pnl = (quantity × current_price) - market_value`

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Cost: $15,000
    >>> pos.add(50, 160.0)   # Cost: $8,000
    >>> pos.market_value     # Total cost basis: $23,000
    23000.0
    
    >>> # Calculate unrealized P&L if current price is $170
    >>> current_price = 170.0
    >>> unrealized_pnl = (pos.quantity * current_price) - pos.market_value
    >>> unrealized_pnl  # 150 × $170 - $23,000 = $2,500
    2500.0

**`Position.add`**
```python
def add(self, quantity: float, price_per_share: float) -> None
```
> Add shares to this position as a new lot.

Creates a new lot entry in the position's lot list. Does NOT merge with
existing lots to maintain accurate FIFO tracking.

Position Direction:
    - quantity > 0: Adding long position (buying shares)
    - quantity < 0: Adding short position (shorting shares)

Args:
    quantity: Number of shares to add. Can be positive (long) or negative (short).
             Zero quantity is ignored (no lot created).
    price_per_share: Cost basis per share, should INCLUDE commission for accuracy.
                    For buys: (execution_price × quantity + commission) / quantity
                    For shorts: net_proceeds / quantity (after commission)

Returns:
    None. Modifies the position in-place.

Example:
    >>> pos = Position(ticker='AAPL')
    >>> 
    >>> # Add long position (100 shares at $150.15 including commission)
    >>> pos.add(100, 150.15)
    >>> pos.lots
    [(100, 150.15)]
    >>> 
    >>> # Add another lot at different price
    >>> pos.add(50, 155.0)
    >>> pos.lots  # Two separate lots maintained
    [(100, 150.15), (50, 155.0)]
    >>> 
    >>> # Add short position
    >>> short_pos = Position(ticker='TSLA')
    >>> short_pos.add(-100, 200.0)  # Short 100 at $200/share net proceeds
    >>> short_pos.quantity
    -100
    
Note:
    - Each call creates a new lot, even if price matches existing lots
    - Zero quantity is silently ignored (useful for conditional adds)
    - For accurate P&L, include commission in price_per_share

**`Position.remove`**
```python
def remove(self, quantity: float, is_short_covering: bool = False) -> Tuple[float, float]
```
> Remove shares from position using FIFO (First-In-First-Out) accounting.

Consumes lots in chronological order (oldest first) until the requested
quantity is removed. Handles partial lot consumption correctly.

FIFO Logic:
    1. Start with the oldest lot (first in the list)
    2. If lot quantity ≤ remaining to remove: consume entire lot
    3. If lot quantity > remaining: consume partial lot, update lot size
    4. Repeat until requested quantity is removed or no lots remain

Args:
    quantity: Number of shares to remove (always pass a POSITIVE value).
             The method handles both long and short position removal internally.
    is_short_covering: Optional flag indicating if this remove is for covering
                      a short position. Currently used for documentation purposes
                      but may affect future logic extensions.

Returns:
    Tuple[float, float]: A tuple containing:
        - actual_quantity_removed: How many shares were actually removed.
          May be less than requested if position is smaller.
        - cost_basis: Total cost basis of removed shares (quantity × price
          summed across consumed lots). For short positions, this is negative
          (representing proceeds received).

Example:
    >>> pos = Position(ticker='AAPL')
    >>> pos.add(100, 150.0)  # Lot 1: 100 @ $150
    >>> pos.add(50, 160.0)   # Lot 2: 50 @ $160
    >>> 
    >>> # Remove 75 shares (FIFO takes from oldest lot first)
    >>> removed_qty, cost_basis = pos.remove(75)
    >>> removed_qty
    75.0
    >>> cost_basis  # 75 × $150 from first lot
    11250.0
    >>> pos.lots  # First lot reduced, second untouched
    [(25, 150.0), (50, 160.0)]
    >>> 
    >>> # Remove remaining 25 from first lot + 30 from second
    >>> removed_qty, cost_basis = pos.remove(55)
    >>> cost_basis  # (25 × $150) + (30 × $160) = 3750 + 4800 = 8550
    8550.0
    >>> pos.lots  # 20 shares remaining from second lot
    [(20, 160.0)]
    >>> 
    >>> # Try to remove more than available
    >>> removed_qty, cost_basis = pos.remove(50)
    >>> removed_qty  # Only 20 available
    20.0
    >>> pos.lots  # Position fully closed
    []
    
    >>> # Edge case: remove from empty position
    >>> empty_pos = Position(ticker='MSFT')
    >>> empty_pos.remove(100)
    (0.0, 0.0)

Note:
    - For short positions (negative lot quantities), cost_basis represents
      the proceeds received when the short was opened (negative value).
    - The function gracefully handles partial fills when position is smaller
      than requested quantity.
    - Zero or negative quantity returns (0.0, 0.0) immediately.

**class `Portfolio`**
```python
class Portfolio:
```
> Portfolio state tracking with cash and positions.

Manages the overall portfolio state including cash balance and all open positions
(both long and short). Provides methods to calculate total portfolio value and
track holdings across multiple assets.

Attributes:
    cash: Current cash balance in the portfolio
    positions: Dictionary mapping ticker -> Position for all active positions

Example:
    >>> portfolio = Portfolio(cash=100_000.0)
    >>> # Buy 100 shares of AAPL at $150
    >>> pos = portfolio.get_position('AAPL')
    >>> pos.add(100, 150.0)
    >>> portfolio.cash -= 100 * 150.0
    >>> 
    >>> # Check portfolio value
    >>> prices = {'AAPL': 155.0}
    >>> total_value = portfolio.get_total_value(prices)
    >>> print(f"Portfolio value: ${total_value:,.2f}")  # Cash + holdings

**`Portfolio.get_position`**
```python
def get_position(self, ticker: str) -> Position
```
> Get or create position for a given ticker.

If the ticker doesn't exist in the portfolio, a new empty Position
is created and added to the portfolio.

Args:
    ticker: Asset ticker (e.g., 'AAPL', 'DIA')
    
Returns:
    Position object for the specified ticker

**`Portfolio.get_holdings_value`**
```python
def get_holdings_value(self, prices: Dict[str, float]) -> float
```
> Calculate total market value of all holdings at current prices.

For long positions (quantity > 0): positive value (asset)
For short positions (quantity < 0): negative value (liability)

Args:
    prices: Dictionary mapping ticker -> current market price

Returns:
    Total market value of all holdings. Positive for net long positions,
    negative for net short positions.
    
Example:
    >>> portfolio = Portfolio(cash=50_000)
    >>> # Long 100 AAPL, Short 50 TSLA
    >>> portfolio.get_position('AAPL').add(100, 150.0)
    >>> portfolio.get_position('TSLA').add(-50, 200.0)  # Short
    >>> prices = {'AAPL': 155.0, 'TSLA': 195.0}
    >>> holdings = portfolio.get_holdings_value(prices)
    >>> # = (100 * 155) + (-50 * 195) = 15,500 - 9,750 = 5,750

**`Portfolio.get_total_value`**
```python
def get_total_value(self, prices: Dict[str, float]) -> float
```
> Calculate total portfolio value (cash + holdings).

This represents the liquidation value of the portfolio if all positions
were closed at the given prices.

Args:
    prices: Dictionary mapping ticker -> current market price
    
Returns:
    Total portfolio value = cash + holdings_value
    
Note:
    For portfolios with short positions, holdings_value may be negative,
    representing the liability from short positions.

**`validate_trading_sheet`**
```python
def validate_trading_sheet(trading_sheet: pd.DataFrame) -> pd.DataFrame
```
> Validate and normalize trading sheet input.

Performs comprehensive validation and normalization of trading signals:
- Normalizes column names to lowercase
- Validates required columns are present
- Ensures action values are 'buy' or 'sell'
- Converts time to datetime format
- Validates numeric columns (quantity, price)
- Removes invalid rows with warnings
- Sorts by time chronologically

Args:
    trading_sheet: DataFrame with trade instructions containing columns:
        - time: Trading timestamp (str or datetime)
        - ticker: Asset ticker symbol (str)
        - action: 'buy' or 'sell' (case-insensitive)
        - quantity: Trade quantity (numeric)
        - price: Target execution price (numeric)
    
Returns:
    Validated DataFrame with normalized column names, sorted by time,
    with only valid rows retained
    
Raises:
    ValueError: If required columns are missing or action values are invalid
    
Warnings:
    Issues warnings when removing rows with invalid quantity/price values
    
Example:
    >>> import pandas as pd
    >>> # Valid input
    >>> trades = pd.DataFrame({
    ...     'TIME': ['2024-01-02 09:30:00', '2024-01-02 10:00:00'],
    ...     'TICKER': ['AAPL', 'MSFT'],
    ...     'Action': ['BUY', 'SELL'],  # Case-insensitive
    ...     'Quantity': [100, 50],
    ...     'Price': [150.50, 380.25]
    ... })
    >>> validated = validate_trading_sheet(trades)
    >>> validated.columns.tolist()
    ['time', 'ticker', 'action', 'quantity', 'price']
    
    >>> # Invalid input - missing column
    >>> bad_trades = pd.DataFrame({'time': ['2024-01-02'], 'ticker': ['AAPL']})
    >>> validate_trading_sheet(bad_trades)  # Raises ValueError
    
    >>> # Invalid input - bad action value
    >>> bad_trades = pd.DataFrame({
    ...     'time': ['2024-01-02'], 'ticker': ['AAPL'],
    ...     'action': ['HOLD'], 'quantity': [100], 'price': [150]
    ... })
    >>> validate_trading_sheet(bad_trades)  # Raises ValueError
    
Note:
    - Column names are case-insensitive and trimmed of whitespace
    - Empty DataFrames return an empty DataFrame with correct columns
    - Rows with NaN quantity or price are removed with a warning
    - All trades are sorted chronologically for proper execution order

**`calculate_execution_price`**
```python
def calculate_execution_price(target_price: float, action: str, slippage_rate: float = 0.0005, spread_rate: float = 0.0001) -> float
```
> Calculate realistic execution price with slippage and bid-ask spread.

Models the market impact of executing a trade by adjusting the target price
for realistic market conditions. Both slippage and spread work against the trader.

Cost Components:
    **Slippage**: The difference between expected and actual execution price due to
    market movement, order size impact, or latency. Always works against you.
    
    **Spread**: The difference between bid and ask prices. Buyers pay the ask (higher),
    sellers receive the bid (lower). The `spread_rate` represents half the spread.

Formulas:
    - Buy:  execution_price = target_price × (1 + slippage_rate + spread_rate)
    - Sell: execution_price = target_price × (1 - slippage_rate - spread_rate)

Args:
    target_price: The target/signal price from your trading strategy.
                 This is typically the close price or a calculated entry price.
    action: Trade direction - 'buy' or 'sell' (case-sensitive).
    slippage_rate: Slippage as a fraction of price. Default 0.0005 (0.05% or 5 bps).
                  Larger orders or less liquid assets typically have higher slippage.
    spread_rate: Half-spread as a fraction of price. Default 0.0001 (0.01% or 1 bp).
                Represents half of the bid-ask spread. Full spread = 2 × spread_rate.

Returns:
    float: Adjusted execution price after accounting for slippage and spread.

Example:
    >>> # Buying at $100 with default rates
    >>> calculate_execution_price(100.0, 'buy')
    100.06  # = 100 × (1 + 0.0005 + 0.0001)
    
    >>> # Selling at $100 with default rates  
    >>> calculate_execution_price(100.0, 'sell')
    99.94  # = 100 × (1 - 0.0005 - 0.0001)
    
    >>> # High-impact trade with larger slippage
    >>> calculate_execution_price(50.0, 'buy', slippage_rate=0.002, spread_rate=0.001)
    50.15  # = 50 × (1 + 0.002 + 0.001) = 50 × 1.003
    
    >>> # Calculate round-trip cost (buy then sell same price)
    >>> buy_price = calculate_execution_price(100.0, 'buy')
    >>> sell_price = calculate_execution_price(100.0, 'sell')
    >>> round_trip_cost = buy_price - sell_price  # $0.12 per share
    >>> round_trip_cost_pct = (round_trip_cost / 100.0) * 100  # 0.12%

Note:
    - Default values represent typical costs for liquid US equities
    - Crypto, forex, and less liquid assets often have higher rates
    - These costs are IN ADDITION to commissions
    - For limit orders with guaranteed fills, you may set both rates to 0

**`calculate_performance_metrics`**
```python
def calculate_performance_metrics(returns: pd.Series, risk_free_rate: float = 0.02) -> Dict[str, float]
```
> Calculate comprehensive performance metrics for trading strategy evaluation.

Computes a wide range of industry-standard metrics to assess strategy performance,
risk characteristics, and return quality. All calculations assume 252 trading days
per year for annualization.

Metric Categories:
    **Return Metrics**:
    - `total_return`: Cumulative return over the entire period
    - `annualized_return`: Geometric mean annual return
    - `avg_daily_return`: Arithmetic mean of daily returns
    - `volatility`: Annualized standard deviation of returns
    
    **Risk-Adjusted Metrics**:
    - `sharpe_ratio`: Excess return per unit of total risk
    - `sortino_ratio`: Excess return per unit of downside risk
    - `calmar_ratio`: Annualized return / max drawdown
    
    **Drawdown Analysis**:
    - `max_drawdown`: Largest peak-to-trough decline (negative value)
    - `avg_drawdown`: Average drawdown when in drawdown
    - `max_drawdown_duration`: Longest drawdown period in days
    
    **Win/Loss Statistics**:
    - `win_rate`: Percentage of positive return days
    - `profit_factor`: Gross profit / gross loss
    - `payoff_ratio`: Average win / average loss (absolute)
    
    **Distribution Characteristics**:
    - `skewness`: Return distribution asymmetry (positive = right tail)
    - `kurtosis`: Return distribution tail thickness (>3 = fat tails)
    - `var_95`: 5th percentile daily return (Value at Risk)
    - `cvar_95`: Mean of returns below VaR (Conditional VaR / Expected Shortfall)

Args:
    returns: pandas Series of daily returns in decimal format.
            Example: 0.01 represents a +1% daily return, -0.02 represents a -2% return.
            Must be simple returns, not log returns.
    risk_free_rate: Annual risk-free rate in decimal format.
                   Default 0.02 represents 2% annual risk-free rate.
                   Used for Sharpe and Sortino ratio calculations.

Returns:
    Dict[str, float]: Dictionary containing all computed metrics.
    Returns empty dict if returns series is empty or has fewer than 2 values.

Example:
    >>> import pandas as pd
    >>> import numpy as np
    >>> 
    >>> # Generate sample returns (252 trading days)
    >>> np.random.seed(42)
    >>> daily_returns = pd.Series(np.random.normal(0.0005, 0.02, 252))
    >>> 
    >>> metrics = calculate_performance_metrics(daily_returns)
    >>> 
    >>> # Access individual metrics
    >>> print(f"Total Return: {metrics['total_return']*100:.2f}%")
    >>> print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.2f}")
    >>> print(f"Max Drawdown: {metrics['max_drawdown']*100:.2f}%")
    >>> print(f"Win Rate: {metrics['win_rate']*100:.1f}%")
    
    >>> # Using with trading_sim results
    >>> pnl, details = trading_sim(trading_sheet=trades, ohlcv_path='data.csv')
    >>> if 'metrics' in details.attrs:
    ...     metrics = details.attrs['metrics']
    ...     print(f"Strategy Sharpe: {metrics['sharpe_ratio']:.2f}")

Formulas:
    - Sharpe Ratio: (E[R] - Rf) × √252 / (σ × √252)
    - Sortino Ratio: (E[R] - Rf) × 252 / (σ_downside × √252)
    - Max Drawdown: min((cumulative - running_max) / running_max)
    - VaR 95%: 5th percentile of return distribution
    - CVaR 95%: E[R | R ≤ VaR_95]

Note:
    - Returns 999.0 for infinite ratios (e.g., Sortino with no downside days)
    - Metrics are computed on non-NaN values only
    - Assumes continuous daily data (gaps may affect drawdown duration)

**`convert_signals_to_trades`**
```python
def convert_signals_to_trades(signals: pd.DataFrame, signal_type: str, ohlcv_df: pd.DataFrame, initial_capital: float = 1000000.0, max_position_pct: float = 0.25) -> pd.DataFrame
```
> Convert raw alpha signals into a trading sheet.

Args:
    signals: DataFrame with [time, ticker, signal]
    signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', or 'BINARY'
    ohlcv_df: Market data for price lookup
    initial_capital: For calculating quantities
    max_position_pct: Max allocation per asset
    
Returns:
    DataFrame trading_sheet [time, ticker, action, quantity, price]

**`trading_sim`**
```python
def trading_sim(trading_sheet: Optional[pd.DataFrame] = None, signals: Optional[pd.DataFrame] = None, signal_type: Optional[str] = None, initial_capital: float = 1000000.0, commission_rate: float = 0.001, slippage_rate: float = 0.0005, spread_rate: float = 0.0001, min_trade_value: float = 100.0, allow_short: bool = False, allow_leverage: bool = False, max_position_pct: float = 0.25, risk_free_rate: float = 0.02) -> Tuple[float, pd.DataFrame]
```
> Comprehensive trading simulation/backtesting engine.

Can accept EITHER:
1. trading_sheet: Explicit buy/sell orders
2. signals + signal_type: Raw signals to be converted to trades

Args:
    trading_sheet: DataFrame with columns [time, ticker, action, quantity, price(optional)]
    signals: DataFrame with columns [time, ticker, signal]
    signal_type: 'TARGET_WEIGHT', 'ALPHA_SCORE', 'BINARY'
    initial_capital: Starting cash (default $1,000,000)
    commission_rate: Commission as fraction of trade value (default 0.1%)
    slippage_rate: Slippage as fraction of price (default 0.05%)
    spread_rate: Half-spread as fraction of price (default 0.01%)
    min_trade_value: Minimum trade value threshold (default $100)
    allow_short: Allow short selling (default False)
    allow_leverage: Allow leverage/margin (default False)
    max_position_pct: Max single position as pct of portfolio (default 25%)
    risk_free_rate: Annual risk-free rate for metrics (default 2%)
    
Returns:
    Tuple of:
    - pnl (float): Final portfolio P&L (final_value - initial_capital)
    - pnl_details (pd.DataFrame): Detailed trade-by-trade breakdown
        
Raises:
    ValueError: If inputs are invalid
    FileNotFoundError: If OHLCV file not found

**`generate_performance_report`**
```python
def generate_performance_report(pnl_details: pd.DataFrame) -> str
```
> Generate a comprehensive, formatted performance report from trading simulation results.

Creates a professional text-based report suitable for logging, display, or export.
The report includes capital summary, return metrics, risk-adjusted ratios,
drawdown analysis, win/loss statistics, and trade execution summary.

Report Sections:
    **CAPITAL SUMMARY**: Initial capital, final value, total P&L, and return %
    
    **RETURN METRICS**: Annualized return, volatility, average daily return
    
    **RISK-ADJUSTED METRICS**: Sharpe, Sortino, and Calmar ratios
    
    **DRAWDOWN ANALYSIS**: Maximum drawdown, average drawdown, max duration
    
    **WIN/LOSS STATISTICS**: Win rate, profit factor, payoff ratio
    
    **RISK METRICS**: VaR 95%, CVaR 95%, skewness, kurtosis
    
    **TRADE SUMMARY**: Total trades, executed/rejected counts, commissions, realized P&L

Args:
    pnl_details: DataFrame returned by `trading_sim()` function. Must have `.attrs`
                dictionary containing 'metrics', 'initial_capital', 'final_value',
                and 'total_pnl' keys. These are automatically attached by trading_sim.

Returns:
    str: A multi-line formatted string containing the complete performance report.
         Returns "No performance metrics available." if pnl_details lacks metrics.

Example:
    >>> import pandas as pd
    >>> 
    >>> # Create trading signals
    >>> trades = pd.DataFrame({
    ...     'time': ['2024-01-02', '2024-01-15', '2024-02-01'],
    ...     'ticker': ['AAPL', 'AAPL', 'AAPL'],
    ...     'action': ['buy', 'sell', 'buy'],
    ...     'quantity': [100, 100, 50],
    ...     'price': [150.0, 155.0, 152.0]
    ... })
    >>> 
    >>> # Run simulation and generate report
    >>> pnl, details = trading_sim(
    ...     trading_sheet=trades,
    ...     ohlcv_path='market_data/ohlcv.csv',
    ...     initial_capital=100_000
    ... )
    >>> 
    >>> # Generate and print report
    >>> report = generate_performance_report(details)
    >>> print(report)
    ================================================================================
                         TRADING SIMULATION REPORT
    ================================================================================
    
    CAPITAL SUMMARY
    ---------------
      Initial Capital:    $     100,000.00
      Final Value:        $     100,450.00
      ...
    
    >>> # Save report to file
    >>> with open('backtest_report.txt', 'w') as f:
    ...     f.write(report)

Note:
    - Requires pnl_details to have metrics attached via .attrs dictionary
    - All percentages are displayed with proper formatting
    - Currency values use comma separators and 2 decimal places
    - Infinite ratios (e.g., from no losing trades) display as 999.000

**`simicx_test_trading_sim`**
```python
def simicx_test_trading_sim()
```
---

### `tune.py`

> **Import**: `from tune import ...`

> Hyperparameter Tuning Module for Multi-Period Portfolio Optimization.

This module performs grid search over hyperparameters for the CNN-LSTM model
and MDFP portfolio optimization layer. It evaluates parameter combinations
using Sharpe ratio as the optimization metric.

Grid Search Parameters:
    - planning_horizon: [5, 10]
    - risk_aversion: [1.0, 5.0]
    - turnover_penalty: [0.01, 0.1]
    - learning_rate: [1e-3]
    - rebalance_freq: [20]

Fixed Parameters:
    - lookback_window: 120
    - epochs: 50
    - neumann_order: 5


**`generate_param_combinations`**
```python
def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]
```
> Generate all combinations of parameters from a grid.

Uses itertools.product to create Cartesian product of all parameter values.

Args:
    grid: Dictionary mapping parameter names to lists of values.
          Example: {"lr": [0.01, 0.001], "batch": [32, 64]}
    
Returns:
    List of dictionaries, each representing one parameter combination.
    Returns [{}] for empty grid.
    
Example:
    >>> grid = {"a": [1, 2], "b": [10]}
    >>> combos = generate_param_combinations(grid)
    >>> len(combos)
    2
    >>> combos[0]
    {'a': 1, 'b': 10}

**`evaluate_params`**
```python
def evaluate_params(ohlcv_df: pd.DataFrame, params: Dict[str, Any], fixed_params: Dict[str, Any]) -> Tuple[float, float]
```
> Evaluate a parameter combination using signal_gen and trading_sim.

This function:
1. Merges grid params with fixed params
2. Calls signal_gen to generate trading signals
3. Calls trading_sim to simulate trading
4. Computes and returns Sharpe ratio and total PnL

Args:
    ohlcv_df: OHLCV DataFrame for training/evaluation
    params: Dictionary of grid search parameters
    fixed_params: Dictionary of fixed (non-tuned) parameters
    
Returns:
    Tuple of (sharpe_ratio, pnl). Returns (-inf, 0.0) on failure.

**`run_grid_search`**
```python
def run_grid_search(phase: str) -> Dict[str, Any]
```
> Run grid search over parameter combinations.

Loads training data, evaluates all parameter combinations using
Sharpe ratio as the optimization metric, and returns the best configuration.

Args:
    phase: Either 'limited' or 'full' - determines dataset size.
           'limited' uses a subset of tickers for faster tuning.
           'full' uses all available tickers.
    
Returns:
    Dictionary containing:
        - best_params: Complete parameter dictionary for signal_gen
        - best_sharpe: Best Sharpe ratio achieved
        - best_pnl: PnL corresponding to best params
        - all_results: List of all evaluation results

**`save_best_params`**
```python
def save_best_params(params: Dict[str, Any], filepath: str) -> None
```
> Save best parameters to JSON file.

The saved JSON contains all parameters needed by signal_gen
(except initial_capital which is handled separately).

Args:
    params: Dictionary of parameters to save. Should include:
        - planning_horizon
        - risk_aversion  
        - turnover_penalty
        - learning_rate
        - rebalance_freq
        - lookback_window
        - epochs
        - neumann_order
    filepath: Path to output JSON file.

**`tune`**
```python
def tune(phase: str) -> Dict[str, Any]
```
> Main tuning entry point.

Runs grid search over the parameter space and saves the best
configuration to best_params.json.

Args:
    phase: Either 'limited' or 'full' data phase.
           'limited' - faster tuning with subset of tickers
           'full' - complete tuning with all tickers
    
Returns:
    Dictionary with:
        - best_params: Best parameter configuration
        - best_sharpe: Best Sharpe ratio achieved
        - best_pnl: PnL from best configuration
        - all_results: All evaluation results

**`simicx_test_generate_param_combinations`**
```python
def simicx_test_generate_param_combinations() -> None
```
> Test that parameter combination generation works correctly.

**`simicx_test_config_completeness`**
```python
def simicx_test_config_completeness() -> None
```
> Test that output config contains all params needed by signal_gen.

**`simicx_test_integration_minimal_pipeline`**
```python
def simicx_test_integration_minimal_pipeline() -> None
```
> Minimal integration test for the tuning pipeline.

**`main`**
```python
def main() -> None
```
> Command-line entry point for hyperparameter tuning.

---

## Project Structure

```
coding/
├── simicx/
│   ├── alpha_config.json (469b)
│   ├── data_loader.py (17162b)
│   └── trading_sim.py (69193b)
├── _verify_tune.py (4920b)
├── best_params.json (193b)
├── full_doc.md (56451b)
├── main.py (17871b)
├── pnl.csv (7190b)
├── signal_gen.py (41374b)
├── simicx.research.db (5099520b)
├── trading_sheet.csv (1314b)
└── tune.py (19857b)
```


## Final Backtest Results (Phase 2: FULL_TICKERS)

**Paper ID**: `integrated_prediction_and_multiperiod_portfolio_op_af9a372b`  
**Execution Date**: 2026-01-05 11:58:12  
**Overall Status**: ✗ FAILED

### Performance Summary
| Step | Status | Details |
|------|--------|---------|
| tune.py | ✗ Failed | full phase |
| main.py | ✗ Failed | Backtest execution |


### Output Excerpt
```

```

---
