"""Hyperparameter Tuning Module for Multi-Period Portfolio Optimization.

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
"""

import json
import itertools
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import argparse
import warnings

import pandas as pd
import numpy as np

# Import dependencies
from simicx.data_loader import get_training_data
from signal_gen import signal_gen
from simicx.trading_sim import trading_sim, calculate_performance_metrics


# ============================================================================
# CONFIGURATION
# ============================================================================

# Grid search parameter space (as per specification)
PARAM_GRID: Dict[str, List[Any]] = {
    "planning_horizon": [5, 10],
    "risk_aversion": [1.0, 5.0],
    "turnover_penalty": [0.01, 0.1],
    "learning_rate": [1e-3],
    "rebalance_freq": [20],
}

# Fixed parameters (not tuned but required by signal_gen)
FIXED_PARAMS: Dict[str, Any] = {
    "lookback_window": 120,
    "epochs": 50,
    "neumann_order": 5,
}

# Default initial capital for simulation (not saved in best_params.json)
DEFAULT_INITIAL_CAPITAL: float = 1_000_000.0

# Output path for best parameters (same directory as this file)
BEST_PARAMS_PATH: Path = Path(__file__).resolve().parent / "best_params.json"


# ============================================================================
# CORE FUNCTIONS
# ============================================================================

def generate_param_combinations(grid: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
    """Generate all combinations of parameters from a grid.
    
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
    """
    if not grid:
        return [{}]
    
    keys = list(grid.keys())
    values = list(grid.values())
    
    combinations = []
    for combo_values in itertools.product(*values):
        combo_dict = dict(zip(keys, combo_values))
        combinations.append(combo_dict)
    
    return combinations


def evaluate_params(
    ohlcv_df: pd.DataFrame,
    params: Dict[str, Any],
    fixed_params: Dict[str, Any]
) -> Tuple[float, float]:
    """Evaluate a parameter combination using signal_gen and trading_sim.
    
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
    """
    # Merge parameters
    all_params = {**fixed_params, **params}
    
    try:
        # Suppress warnings during evaluation
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            
            # Generate trading signals using signal_gen
            trading_sheet = signal_gen(
                ohlcv_df=ohlcv_df,
                lookback_window=int(all_params["lookback_window"]),
                planning_horizon=int(all_params["planning_horizon"]),
                rebalance_freq=int(all_params["rebalance_freq"]),
                risk_aversion=float(all_params["risk_aversion"]),
                turnover_penalty=float(all_params["turnover_penalty"]),
                learning_rate=float(all_params["learning_rate"]),
                epochs=int(all_params["epochs"]),
                neumann_order=int(all_params["neumann_order"]),
                initial_capital=float(DEFAULT_INITIAL_CAPITAL),
            )
            
            # Check if trading sheet is valid and non-empty
            if trading_sheet is None or trading_sheet.empty:
                print("  Warning: Empty trading sheet generated")
                return float("-inf"), 0.0
            
            # Run trading simulation
            pnl, pnl_details = trading_sim(
                trading_sheet=trading_sheet,
                signals=None,
                signal_type=None,
                initial_capital=DEFAULT_INITIAL_CAPITAL,
                commission_rate=0.001,   # 0.1%
                slippage_rate=0.0005,    # 0.05%
                spread_rate=0.0001,      # 0.01%
                min_trade_value=100.0,
                allow_short=False,
                allow_leverage=False,
                max_position_pct=0.25,   # 25%
                risk_free_rate=0.02,     # 2%
            )
        
        # Extract Sharpe ratio from simulation results
        sharpe_ratio = float("-inf")
        
        # Try to get Sharpe from metrics attached to pnl_details
        if hasattr(pnl_details, 'attrs') and isinstance(pnl_details.attrs, dict):
            metrics = pnl_details.attrs.get('metrics', {})
            if isinstance(metrics, dict) and 'sharpe_ratio' in metrics:
                sharpe_val = metrics['sharpe_ratio']
                if sharpe_val is not None and not np.isnan(sharpe_val):
                    sharpe_ratio = float(sharpe_val)
        
        # Fallback: calculate from portfolio values if available
        if sharpe_ratio == float("-inf") and 'portfolio_value' in pnl_details.columns:
            portfolio_values = pnl_details['portfolio_value'].dropna()
            if len(portfolio_values) > 1:
                returns = portfolio_values.pct_change().dropna()
                if len(returns) > 1:
                    daily_rf = 0.02 / 252  # Daily risk-free rate
                    excess_returns = returns - daily_rf
                    if returns.std() > 1e-10:
                        sharpe_ratio = float((excess_returns.mean() / returns.std()) * np.sqrt(252))
        
        # Handle invalid values
        if np.isnan(sharpe_ratio) or np.isinf(sharpe_ratio) and sharpe_ratio > 0:
            sharpe_ratio = float("-inf")
            
        return sharpe_ratio, float(pnl)
        
    except Exception as e:
        print(f"  Error evaluating params: {type(e).__name__}: {e}")
        return float("-inf"), 0.0


def run_grid_search(phase: str) -> Dict[str, Any]:
    """Run grid search over parameter combinations.
    
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
    """
    print(f"="*60)
    print(f"HYPERPARAMETER GRID SEARCH")
    print(f"="*60)
    print(f"Phase: {phase}")
    print()
    
    # Load training data using get_training_data
    print("Loading training data...")
    ohlcv_df = get_training_data(phase=phase)
    
    if ohlcv_df is None or ohlcv_df.empty:
        raise ValueError(f"No training data available for phase='{phase}'")
    
    n_rows = len(ohlcv_df)
    tickers = sorted(ohlcv_df['ticker'].unique().tolist())
    date_range = (ohlcv_df['time'].min(), ohlcv_df['time'].max())
    
    print(f"  Loaded {n_rows:,} rows")
    print(f"  Tickers ({len(tickers)}): {tickers}")
    print(f"  Date range: {date_range[0]} to {date_range[1]}")
    print()
    
    # Generate all parameter combinations
    param_combinations = generate_param_combinations(PARAM_GRID)
    n_combinations = len(param_combinations)
    
    print(f"Grid Search Configuration:")
    for param_name, param_values in PARAM_GRID.items():
        print(f"  {param_name}: {param_values}")
    print(f"\nFixed Parameters:")
    for param_name, param_value in FIXED_PARAMS.items():
        print(f"  {param_name}: {param_value}")
    print(f"\nTotal combinations to evaluate: {n_combinations}")
    print("-"*60)
    
    # Track best results
    best_sharpe: float = float("-inf")
    best_params: Optional[Dict[str, Any]] = None
    best_pnl: float = 0.0
    
    all_results: List[Dict[str, Any]] = []
    
    # Evaluate each parameter combination
    for i, params in enumerate(param_combinations, 1):
        print(f"\n[{i}/{n_combinations}] Evaluating:")
        for k, v in params.items():
            print(f"    {k}: {v}")
        
        sharpe, pnl = evaluate_params(ohlcv_df, params, FIXED_PARAMS)
        
        result = {
            "params": params.copy(),
            "sharpe_ratio": sharpe,
            "pnl": pnl,
        }
        all_results.append(result)
        
        if sharpe != float("-inf"):
            print(f"  -> Sharpe Ratio: {sharpe:.4f}, PnL: ${pnl:,.2f}")
        else:
            print(f"  -> Evaluation failed or invalid result")
        
        # Update best if improved
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_params = params.copy()
            best_pnl = pnl
            print(f"  *** NEW BEST ***")
    
    print("\n" + "="*60)
    print("GRID SEARCH COMPLETE")
    print("="*60)
    
    # Fallback to first combination if all evaluations failed
    if best_params is None:
        print("\nWarning: All evaluations failed, using default parameters")
        best_params = param_combinations[0].copy()
        best_sharpe = 0.0
        best_pnl = 0.0
    
    # Create complete parameter set (grid params + fixed params)
    # Note: initial_capital is NOT included per data contract
    complete_params: Dict[str, Any] = {**FIXED_PARAMS, **best_params}
    
    print(f"\nBest Parameters:")
    for key, value in sorted(complete_params.items()):
        print(f"  {key}: {value}")
    print(f"\nBest Sharpe Ratio: {best_sharpe:.4f}")
    print(f"Best PnL: ${best_pnl:,.2f}")
    
    return {
        "best_params": complete_params,
        "best_sharpe": best_sharpe,
        "best_pnl": best_pnl,
        "all_results": all_results,
    }


def save_best_params(params: Dict[str, Any], filepath: str) -> None:
    """Save best parameters to JSON file.
    
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
    """
    filepath_obj = Path(filepath)
    
    # Ensure parent directory exists
    filepath_obj.parent.mkdir(parents=True, exist_ok=True)
    
    # Save with proper formatting
    with open(filepath_obj, 'w') as f:
        json.dump(params, f, indent=2)
    
    print(f"\nSaved best parameters to: {filepath_obj}")


def tune(phase: str) -> Dict[str, Any]:
    """Main tuning entry point.
    
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
    """
    # Run grid search
    results = run_grid_search(phase=phase)
    
    # Extract best parameters
    best_params = results["best_params"]
    
    # Save to JSON file (mandatory - required by main.py)
    save_best_params(best_params, str(BEST_PARAMS_PATH))
    
    return results


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def simicx_test_generate_param_combinations() -> None:
    """Test that parameter combination generation works correctly."""
    # Test basic grid with multiple values
    grid = {"a": [1, 2], "b": [10, 20]}
    combos = generate_param_combinations(grid)
    
    assert len(combos) == 4, f"Expected 4 combinations, got {len(combos)}"
    assert {"a": 1, "b": 10} in combos
    assert {"a": 1, "b": 20} in combos
    assert {"a": 2, "b": 10} in combos
    assert {"a": 2, "b": 20} in combos
    
    # Test single-value grid
    grid = {"x": [5]}
    combos = generate_param_combinations(grid)
    assert len(combos) == 1, f"Expected 1 combination, got {len(combos)}"
    assert combos[0] == {"x": 5}
    
    # Test empty grid
    combos = generate_param_combinations({})
    assert len(combos) == 1, "Empty grid should produce 1 empty combination"
    assert combos[0] == {}
    
    # Test actual PARAM_GRID from specification
    combos = generate_param_combinations(PARAM_GRID)
    expected_count = (
        len(PARAM_GRID["planning_horizon"]) *
        len(PARAM_GRID["risk_aversion"]) *
        len(PARAM_GRID["turnover_penalty"]) *
        len(PARAM_GRID["learning_rate"]) *
        len(PARAM_GRID["rebalance_freq"])
    )
    assert len(combos) == expected_count, f"Expected {expected_count}, got {len(combos)}"
    
    # Verify each combination has all grid keys
    for combo in combos:
        for key in PARAM_GRID.keys():
            assert key in combo, f"Missing key '{key}' in combination"
    
    print("simicx_test_generate_param_combinations: PASSED")


def simicx_test_config_completeness() -> None:
    """Test that output config contains all params needed by signal_gen."""
    # Required parameters from signal_gen signature (except initial_capital)
    required_params = {
        "planning_horizon",
        "risk_aversion",
        "turnover_penalty",
        "learning_rate",
        "rebalance_freq",
        "lookback_window",
        "epochs",
        "neumann_order",
    }
    
    # Check that PARAM_GRID + FIXED_PARAMS cover all required
    available_params = set(PARAM_GRID.keys()) | set(FIXED_PARAMS.keys())
    
    missing = required_params - available_params
    assert not missing, f"Missing required parameters: {missing}"
    
    # Simulate what would be saved to best_params.json
    sample_grid_params = generate_param_combinations(PARAM_GRID)[0]
    complete_params = {**FIXED_PARAMS, **sample_grid_params}
    
    missing_in_output = required_params - set(complete_params.keys())
    assert not missing_in_output, f"Output config missing params: {missing_in_output}"
    
    # Verify all values have correct types
    assert isinstance(complete_params["planning_horizon"], (int, float))
    assert isinstance(complete_params["risk_aversion"], (int, float))
    assert isinstance(complete_params["turnover_penalty"], (int, float))
    assert isinstance(complete_params["learning_rate"], (int, float))
    assert isinstance(complete_params["rebalance_freq"], (int, float))
    assert isinstance(complete_params["lookback_window"], (int, float))
    assert isinstance(complete_params["epochs"], (int, float))
    assert isinstance(complete_params["neumann_order"], (int, float))
    
    print("simicx_test_config_completeness: PASSED")


def simicx_test_integration_minimal_pipeline() -> None:
    """Minimal integration test for the tuning pipeline."""
    import tempfile
    import os
    
    # Test 1: save_best_params creates valid JSON
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
    
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "test_params.json")
        save_best_params(test_params, filepath)
        
        # Verify file was created
        assert os.path.exists(filepath), "Parameters file was not created"
        
        # Verify content is valid JSON with correct values
        with open(filepath, 'r') as f:
            loaded = json.load(f)
        
        assert loaded == test_params, f"Saved params don't match: {loaded}"
    
    # Test 2: generate_param_combinations produces valid output
    combos = generate_param_combinations(PARAM_GRID)
    assert len(combos) > 0, "No parameter combinations generated"
    
    # Test 3: All combinations are valid dictionaries
    for i, combo in enumerate(combos):
        assert isinstance(combo, dict), f"Combination {i} is not a dict"
        for key in PARAM_GRID.keys():
            assert key in combo, f"Missing key '{key}' in combination {i}"
    
    # Test 4: Verify PARAM_GRID matches specification
    assert PARAM_GRID["planning_horizon"] == [5, 10], "planning_horizon mismatch"
    assert PARAM_GRID["risk_aversion"] == [1.0, 5.0], "risk_aversion mismatch"
    assert PARAM_GRID["turnover_penalty"] == [0.01, 0.1], "turnover_penalty mismatch"
    assert PARAM_GRID["learning_rate"] == [1e-3], "learning_rate mismatch"
    assert PARAM_GRID["rebalance_freq"] == [20], "rebalance_freq mismatch"
    
    print("simicx_test_integration_minimal_pipeline: PASSED")


# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

def main() -> None:
    """Command-line entry point for hyperparameter tuning."""
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning for multi-period portfolio optimization"
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="limited",
        choices=["limited", "full"],
        help="Data phase: 'limited' (subset of tickers) or 'full' (all tickers). Default: limited"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run unit tests instead of tuning"
    )
    
    args = parser.parse_args()
    
    if args.test:
        print("Running tests...")
        print("-" * 40)
        simicx_test_generate_param_combinations()
        simicx_test_config_completeness()
        simicx_test_integration_minimal_pipeline()
        print("-" * 40)
        print("\nAll tests passed!")
    else:
        # Run hyperparameter tuning
        results = tune(phase=args.phase)
        
        print("\n" + "=" * 60)
        print("TUNING COMPLETE")
        print("=" * 60)
        print(f"Best parameters saved to: {BEST_PARAMS_PATH}")
        print(f"Best Sharpe Ratio: {results['best_sharpe']:.4f}")
        print(f"Best PnL: ${results['best_pnl']:,.2f}")


if __name__ == "__main__":
    main()