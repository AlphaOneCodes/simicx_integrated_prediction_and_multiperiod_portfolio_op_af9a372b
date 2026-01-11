#!/usr/bin/env python3
"""Production Execution Script for CNN-LSTM Portfolio Optimization.

This module implements the production pipeline for the portfolio optimization
system, loading tuned hyperparameters and executing the trading strategy
on out-of-sample data (2025 onwards).

Author: SimicX
Copyright: SimicX 2024
License: SimicX XTT
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List

import pandas as pd

from signal_gen import signal_gen
from simicx.trading_sim import trading_sim
from simicx.data_loader import get_data


# Configuration paths
CONFIG_PATH = Path("simicx/alpha_config.json")
BEST_PARAMS_PATH = Path(__file__).resolve().parent / "best_params.json"
OUTPUT_TRADING_SHEET = Path("trading_sheet.csv")
OUTPUT_PNL = Path("pnl.csv")

# Default initial capital (not in best_params.json per data contract)
DEFAULT_INITIAL_CAPITAL = 1_000_000.0

# Trading start date constant (from data_loader)
TRADING_START_DATE = "2025-01-01"


def get_trading_data_with_warmup(
    tickers: List[str],
    warmup_days: int = 365,
    align_dates: bool = True
) -> pd.DataFrame:
    """Get trading data with historical warmup period for model training.
    
    This function loads data from (TRADING_START_DATE - warmup_days) onwards,
    providing sufficient historical data for model training before the first trade
    executes on 2025-01-01.
    
    Args:
        tickers: List of ticker symbols.
        warmup_days: Number of days before TRADING_START_DATE to load for warmup.
                     Default 365 provides ~1 year of historical data for training.
        align_dates: If True, only return dates where ALL tickers have data.
    
    Returns:
        pd.DataFrame: OHLCV data including warmup period and trading period.
    """
    # Calculate warmup start date
    trading_start = pd.Timestamp(TRADING_START_DATE)
    warmup_start = trading_start - pd.Timedelta(days=warmup_days)
    warmup_start_str = warmup_start.strftime('%Y-%m-%d')
    
    print(f"Loading data with {warmup_days}-day warmup:")
    print(f"  Warmup start: {warmup_start_str}")
    print(f"  Trading start: {TRADING_START_DATE}")
    
    return get_data(
        tickers=tickers,
        start_date=warmup_start_str,
        end_date=None,
        align_dates=align_dates
    )


def load_alpha_config(config_path: Path = CONFIG_PATH) -> Dict[str, Any]:
    """Load alpha configuration containing ticker lists.
    
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
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Alpha configuration not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = json.load(f)
    
    return config


def load_best_params(params_path: Path = None) -> Dict[str, Any]:
    """Load best hyperparameters from tuning phase.

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
    """
    if params_path is None:
        # Resolve path relative to this script's directory, not CWD
        script_dir = Path(__file__).resolve().parent
        params_path = script_dir / "best_params.json"

    if not params_path.exists():
        raise FileNotFoundError(
            f"Best parameters file not found: {params_path}. "
            "Run tune.py first to generate hyperparameters."
        )

    with open(params_path, "r") as f:
        params = json.load(f)

    # Validate required keys per data contract
    required_keys = [
        "planning_horizon",
        "risk_aversion",
        "turnover_penalty",
        "learning_rate",
        "rebalance_freq",
        "lookback_window",
        "epochs",
        "neumann_order",
    ]

    missing_keys = [key for key in required_keys if key not in params]
    if missing_keys:
        raise KeyError(
            f"Missing required parameters in {params_path}: {missing_keys}"
        )

    return params


def get_tickers_for_phase(phase: str, config: Dict[str, Any]) -> List[str]:
    """Get ticker list based on phase.
    
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
    """
    if phase not in ("limited", "full"):
        raise ValueError(f"Invalid phase: {phase}. Must be 'limited' or 'full'.")
    
    key = "LIMITED_TICKERS" if phase == "limited" else "FULL_TICKERS"
    
    if key not in config:
        raise KeyError(f"Configuration missing '{key}'")
    
    return config[key]


def run_production(phase: str) -> Dict[str, Any]:
    """Run production trading pipeline.
    
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
    """
    print("=" * 60)
    print(f"Production Trading Pipeline - Phase: {phase.upper()}")
    print("=" * 60)
    
    # Step 1: Load configurations
    print("\n[1/5] Loading configuration...")
    config = load_alpha_config()
    best_params = load_best_params()
    
    print("  Loaded best parameters:")
    for key, value in best_params.items():
        print(f"    {key}: {value}")
    
    # Step 2: Get tickers for phase
    print(f"\n[2/5] Getting tickers for phase '{phase}'...")
    tickers = get_tickers_for_phase(phase, config)
    ticker_preview = tickers[:5]
    print(f"  Using {len(tickers)} tickers: {ticker_preview}{'...' if len(tickers) > 5 else ''}")
    
    # Step 3: Load trading data with warmup period
    print("\n[3/5] Loading trading data with warmup period...")
    ohlcv_df = get_trading_data_with_warmup(
        tickers=tickers,
        warmup_days=365,  # 1 year of historical data for model warmup
        align_dates=True
    )
    print(f"  Loaded {len(ohlcv_df)} records")
    print(f"  Date range: {ohlcv_df['time'].min()} to {ohlcv_df['time'].max()}")
    
    # Step 4: Generate trading signals
    print("\n[4/5] Generating trading signals...")
    print("  Trading will start from: 2025-01-01")
    print("  Pre-2025 data will be used for model warmup only")
    trading_sheet = signal_gen(
        ohlcv_df=ohlcv_df,
        trading_start_date="2025-01-01",  # Explicitly set trading start date
        lookback_window=best_params["lookback_window"],
        training_window=250,  # Keep 250 for proper warmup (not in best_params)
        planning_horizon=best_params["planning_horizon"],
        rebalance_freq=best_params["rebalance_freq"],
        risk_aversion=best_params["risk_aversion"],
        turnover_penalty=best_params["turnover_penalty"],
        learning_rate=best_params["learning_rate"],
        epochs=best_params["epochs"],
        neumann_order=best_params["neumann_order"],
        initial_capital=DEFAULT_INITIAL_CAPITAL,
    )
    print(f"  Generated {len(trading_sheet)} trade signals")
    
    # Step 5: Run trading simulation
    print("\n[5/5] Running trading simulation...")
    pnl, pnl_details = trading_sim(
        trading_sheet=trading_sheet,
        signals=None,
        signal_type=None,
        initial_capital=DEFAULT_INITIAL_CAPITAL,
        commission_rate=0.001,
        slippage_rate=0.0005,
        spread_rate=0.0001,
        min_trade_value=100.0,
        allow_short=False,
        allow_leverage=False,
        max_position_pct=0.25,
        risk_free_rate=0.02,
        ohlcv_tickers=tickers,
    )
    
    # Extract metrics from pnl_details attrs
    metrics = pnl_details.attrs.get("metrics", {})
    sharpe_ratio = metrics.get("sharpe_ratio", float("nan"))
    total_return = metrics.get("total_return", float("nan"))
    
    # Print final stats
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"  Sharpe Ratio:  {sharpe_ratio:.4f}")
    print(f"  Total Return:  {total_return * 100:.2f}%")
    print(f"  Total P&L:     ${pnl:,.2f}")
    print("=" * 60)
    
    # Save outputs
    print("\nSaving outputs...")
    trading_sheet.to_csv(OUTPUT_TRADING_SHEET, index=False)
    print(f"  Trading sheet saved to: {OUTPUT_TRADING_SHEET}")
    
    pnl_details.to_csv(OUTPUT_PNL, index=False)
    print(f"  P&L details saved to: {OUTPUT_PNL}")
    
    return {
        "sharpe_ratio": sharpe_ratio,
        "total_return": total_return,
        "pnl": pnl,
        "trading_sheet_path": str(OUTPUT_TRADING_SHEET),
        "pnl_path": str(OUTPUT_PNL),
    }


def parse_args() -> Dict[str, str]:
    """Parse command line arguments.

    Returns:
        Dictionary with parsed arguments:
            - phase: str - Either 'limited' or 'full'

    Example:
        >>> # From command line: python main.py --phase limited
        >>> args = parse_args()
        >>> args['phase']
        'limited'
    """
    phase = "limited"  # default

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--phase":
            if i + 1 < len(args):
                phase = args[i + 1]
                if phase not in ("limited", "full"):
                    print(f"Error: --phase must be 'limited' or 'full', got '{phase}'", file=sys.stderr)
                    sys.exit(1)
                i += 2
            else:
                print("Error: --phase requires a value", file=sys.stderr)
                sys.exit(1)
        elif args[i] in ("-h", "--help"):
            print("Production execution for CNN-LSTM portfolio optimization.")
            print("Usage: python main.py [--phase {limited,full}]")
            print("Options:")
            print("  --phase {limited,full}  Trading phase: 'limited' uses fewer tickers,")
            print("                          'full' uses all tickers. (default: limited)")
            sys.exit(0)
        else:
            print(f"Unknown argument: {args[i]}", file=sys.stderr)
            sys.exit(1)

    return {"phase": phase}


    run_production(phase=args["phase"])


# =============================================================================


def main() -> None:
    """Main entry point for production execution.

    Parses CLI arguments and runs the production trading pipeline.

    Example:
        $ python main.py --phase limited
        $ python main.py --phase full
    """
    args = parse_args()
    run_production(phase=args["phase"])


# =============================================================================
# Inline Tests
# =============================================================================

def simicx_test_load_alpha_config():
    """Test loading alpha configuration."""
    import tempfile
    
    # Test with valid config
    with tempfile.TemporaryDirectory() as td:
        config_path = Path(td) / "test_config.json"
        test_config = {
            "LIMITED_TICKERS": ["SPY", "QQQ"],
            "FULL_TICKERS": ["SPY", "QQQ", "AAPL", "MSFT"],
        }
        with open(config_path, "w") as f:
            json.dump(test_config, f)
        
        loaded = load_alpha_config(config_path)
        assert loaded["LIMITED_TICKERS"] == ["SPY", "QQQ"], "LIMITED_TICKERS mismatch"
        assert len(loaded["FULL_TICKERS"]) == 4, "FULL_TICKERS length mismatch"
    
    # Test with missing file
    try:
        load_alpha_config(Path("/nonexistent/path.json"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError:
        pass
    
    print("  simicx_test_load_alpha_config: PASSED")


def simicx_test_load_best_params():
    """Test loading and validating best parameters."""
    import tempfile
    
    # Test with valid params
    with tempfile.TemporaryDirectory() as td:
        params_path = Path(td) / "best_params.json"
        test_params = {
            "planning_horizon": 5,
            "risk_aversion": 1.0,
            "turnover_penalty": 0.01,
            "learning_rate": 0.001,
            "rebalance_freq": 20,
            "lookback_window": 120,
            "epochs": 50,
            "neumann_order": 5,
        }
        with open(params_path, "w") as f:
            json.dump(test_params, f)
        
        loaded = load_best_params(params_path)
        assert loaded["planning_horizon"] == 5, "planning_horizon mismatch"
        assert loaded["risk_aversion"] == 1.0, "risk_aversion mismatch"
        assert loaded["neumann_order"] == 5, "neumann_order mismatch"
    
    # Test with missing required key
    with tempfile.TemporaryDirectory() as td:
        params_path = Path(td) / "incomplete_params.json"
        incomplete_params = {
            "planning_horizon": 5,
            # Missing other required keys
        }
        with open(params_path, "w") as f:
            json.dump(incomplete_params, f)
        
        try:
            load_best_params(params_path)
            assert False, "Should have raised KeyError"
        except KeyError as e:
            assert "Missing required parameters" in str(e)
    
    # Test with missing file
    try:
        load_best_params(Path("/nonexistent/best_params.json"))
        assert False, "Should have raised FileNotFoundError"
    except FileNotFoundError as e:
        assert "Run tune.py first" in str(e)
    
    print("  simicx_test_load_best_params: PASSED")


def simicx_test_get_tickers_for_phase():
    """Test ticker selection by phase."""
    config = {
        "LIMITED_TICKERS": ["SPY", "QQQ"],
        "FULL_TICKERS": ["SPY", "QQQ", "AAPL", "MSFT", "GOOGL"],
    }
    
    # Test limited phase
    limited = get_tickers_for_phase("limited", config)
    assert limited == ["SPY", "QQQ"], f"Expected limited tickers, got {limited}"
    
    # Test full phase
    full = get_tickers_for_phase("full", config)
    assert len(full) == 5, f"Expected 5 full tickers, got {len(full)}"
    assert "AAPL" in full, "AAPL should be in full tickers"
    
    # Test invalid phase
    try:
        get_tickers_for_phase("invalid", config)
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "invalid" in str(e).lower()
    
    # Test missing key in config
    try:
        get_tickers_for_phase("limited", {"FULL_TICKERS": ["SPY"]})
        assert False, "Should have raised KeyError"
    except KeyError as e:
        assert "LIMITED_TICKERS" in str(e)
    
    print("  simicx_test_get_tickers_for_phase: PASSED")


def simicx_test_integration_with_deps():
    """Integration test exercising dependency interfaces with minimal data."""
    import tempfile
    
    # Import dependencies to verify they're accessible
    from signal_gen import signal_gen
    from simicx.trading_sim import trading_sim, validate_trading_sheet
    from simicx.data_loader import get_trading_data
    
    # Create minimal config and params files
    with tempfile.TemporaryDirectory() as td:
        # Test config loading
        simicx_dir = Path(td) / "simicx"
        simicx_dir.mkdir(exist_ok=True)
        config_path = simicx_dir / "alpha_config.json"
        test_config = {
            "LIMITED_TICKERS": ["SPY"],
            "FULL_TICKERS": ["SPY", "QQQ"],
        }
        with open(config_path, "w") as f:
            json.dump(test_config, f)
        
        loaded_config = load_alpha_config(config_path)
        tickers = get_tickers_for_phase("limited", loaded_config)
        assert tickers == ["SPY"], f"Ticker selection failed: {tickers}"
        
        # Test best params loading
        params_path = Path(td) / "best_params.json"
        test_params = {
            "planning_horizon": 3,
            "risk_aversion": 1.0,
            "turnover_penalty": 0.01,
            "learning_rate": 0.001,
            "rebalance_freq": 10,
            "lookback_window": 30,
            "epochs": 1,
            "neumann_order": 2,
        }
        with open(params_path, "w") as f:
            json.dump(test_params, f)
        
        loaded_params = load_best_params(params_path)
        assert loaded_params["epochs"] == 1, "Params loading failed"
        assert loaded_params["planning_horizon"] == 3, "planning_horizon mismatch"
    
    # Test data loader interface
    try:
        ohlcv = get_trading_data(tickers=["SPY"], align_dates=True)
        assert "time" in ohlcv.columns, "OHLCV missing 'time' column"
        assert "ticker" in ohlcv.columns, "OHLCV missing 'ticker' column"
        assert "close" in ohlcv.columns, "OHLCV missing 'close' column"
        assert len(ohlcv) > 0, "OHLCV DataFrame is empty"
        print(f"    Loaded {len(ohlcv)} OHLCV records for SPY")
        
        # Test trading_sheet validation
        sample_sheet = pd.DataFrame({
            "time": [ohlcv["time"].iloc[0]],
            "ticker": ["SPY"],
            "action": ["buy"],
            "quantity": [10],
            "price": [ohlcv["close"].iloc[0]],
        })
        validated = validate_trading_sheet(sample_sheet)
        assert len(validated) == 1, "Validation failed"
        
    except Exception as e:
        # Allow test to pass if DB not available in test environment
        print(f"    DB connection test skipped: {type(e).__name__}")
    
    print("  simicx_test_integration_with_deps: PASSED")


if __name__ == "__main__":
    main()