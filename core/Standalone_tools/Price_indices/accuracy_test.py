#!/usr/bin/env python3
"""
Load calibration results from Material_param.py and test GBM fit quality.

Simple utility to:
1. Load calibration JSON files
2. Test log returns for normality
3. Calculate goodness-of-fit metrics
4. Display results
"""

import json
from pathlib import Path
from typing import Dict
import pandas as pd
import numpy as np
from scipy import stats
import sys


def load_calibration(json_path: str) -> Dict:
    """Load a single calibration JSON file."""
    path = Path(json_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {json_path}")
    
    with open(path, 'r') as f:
        return json.load(f)


def test_normality(log_returns: np.ndarray) -> Dict:
    """
    Test normality of log returns using multiple tests.
    
    Returns dict with Shapiro-Wilk, Anderson-Darling, Jarque-Bera results.
    """
    returns_clean = log_returns[~np.isnan(log_returns)]
    
    if len(returns_clean) < 3:
        raise ValueError("Need at least 3 observations")
    
    # Shapiro-Wilk test
    shapiro_stat, shapiro_p = stats.shapiro(returns_clean)
    
    # Anderson-Darling test
    anderson_result = stats.anderson(returns_clean, dist='norm')
    
    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(returns_clean)
    
    # Summary: pass if at least 2 of 3 tests pass (α=0.05)
    alpha = 0.05
    tests_passed = sum([
        shapiro_p > alpha,
        anderson_result.statistic < anderson_result.critical_values[-1],
        jb_p > alpha
    ])
    is_normal = tests_passed >= 2
    
    return {
        'n_obs': len(returns_clean),
        'shapiro_p': float(shapiro_p),
        'anderson_stat': float(anderson_result.statistic),
        'anderson_crit_5pct': float(anderson_result.critical_values[-1]),
        'jb_p': float(jb_p),
        'skewness': float(stats.skew(returns_clean)),
        'kurtosis': float(stats.kurtosis(returns_clean)),
        'is_normal': is_normal,
        'tests_passed': tests_passed
    }


def test_gbm_fit(log_returns: np.ndarray, mu: float, sigma: float, dt: float) -> Dict:
    """
    Test goodness of fit for GBM parameters.
    
    Parameters
    ----------
    log_returns : np.ndarray
        Observed log returns
    mu : float
        GBM drift parameter (annualized)
    sigma : float
        GBM volatility parameter (annualized)
    dt : float
        Time step in years
    """
    returns_clean = log_returns[~np.isnan(log_returns)]
    
    # Expected parameters from GBM
    mu_eff = (mu - 0.5 * sigma**2) * dt
    sigma_eff = sigma * np.sqrt(dt)
    
    # Observed statistics
    obs_mean = np.mean(returns_clean)
    obs_std = np.std(returns_clean, ddof=1)
    
    # Standardized residuals
    standardized = (returns_clean - mu_eff) / sigma_eff
    
    # Log-likelihood
    log_likelihood = np.sum(
        -0.5 * np.log(2 * np.pi) 
        - np.log(sigma_eff) 
        - 0.5 * standardized ** 2
    )
    
    n_obs = len(returns_clean)
    aic = 4 - 2 * log_likelihood
    bic = 2 * np.log(n_obs) - 2 * log_likelihood
    
    return {
        'observed_mean': float(obs_mean),
        'theoretical_mean': float(mu_eff),
        'mean_error': float(abs(obs_mean - mu_eff)),
        'observed_std': float(obs_std),
        'theoretical_std': float(sigma_eff),
        'std_error': float(abs(obs_std - sigma_eff)),
        'residuals_mean': float(np.mean(standardized)),
        'residuals_std': float(np.std(standardized, ddof=1)),
        'log_likelihood': float(log_likelihood),
        'aic': float(aic),
        'bic': float(bic)
    }


def print_results(commodity: str, cal_data: Dict, prices: pd.Series, dt: float) -> None:
    """Print test results for a commodity."""
    print("\n" + "=" * 80)
    print(f"GBM FIT ASSESSMENT: {commodity}")
    print("=" * 80)
    
    # Get GBM parameters
    gbm = cal_data.get('gbm', {})
    if not gbm:
        print("No GBM parameters found")
        return
    
    mu = gbm['mu']
    sigma = gbm['sigma']
    
    # Calculate log returns
    prices_clean = prices.astype(float).dropna()
    log_returns = np.log(prices_clean / prices_clean.shift(1)).dropna().values
    
    # Test normality
    print("\n1. NORMALITY TESTS")
    print("-" * 80)
    norm_results = test_normality(log_returns)
    
    print(f"Sample size: {norm_results['n_obs']} observations")
    print(f"Skewness: {norm_results['skewness']:.6f}")
    print(f"Kurtosis: {norm_results['kurtosis']:.6f}")
    print(f"\nShapiro-Wilk test p-value: {norm_results['shapiro_p']:.6f}")
    print(f"  Result: {'PASS' if norm_results['shapiro_p'] > 0.05 else 'FAIL'}")
    print(f"\nAnderson-Darling test statistic: {norm_results['anderson_stat']:.6f}")
    print(f"  Critical value (5%): {norm_results['anderson_crit_5pct']:.6f}")
    print(f"  Result: {'PASS' if norm_results['anderson_stat'] < norm_results['anderson_crit_5pct'] else 'FAIL'}")
    print(f"\nJarque-Bera test p-value: {norm_results['jb_p']:.6f}")
    print(f"  Result: {'PASS' if norm_results['jb_p'] > 0.05 else 'FAIL'}")
    print(f"\nOverall: {norm_results['tests_passed']}/3 tests passed")
    
    if norm_results['is_normal']:
        print("✓ Log returns are normally distributed")
    else:
        print("⚠ Log returns deviate from normality")
    
    # Test fit
    print("\n2. GOODNESS OF FIT")
    print("-" * 80)
    print(f"Calibrated parameters:")
    print(f"  μ = {mu:.6f}")
    print(f"  σ = {sigma:.6f}")
    print(f"  Δt = {dt:.6f} years")
    
    fit_results = test_gbm_fit(log_returns, mu, sigma, dt)
    
    print(f"\nMean of log returns:")
    print(f"  Observed:    {fit_results['observed_mean']:.6f}")
    print(f"  Theoretical: {fit_results['theoretical_mean']:.6f}")
    print(f"  Error:       {fit_results['mean_error']:.6f}")
    
    print(f"\nStd Dev of log returns:")
    print(f"  Observed:    {fit_results['observed_std']:.6f}")
    print(f"  Theoretical: {fit_results['theoretical_std']:.6f}")
    print(f"  Error:       {fit_results['std_error']:.6f}")
    
    print(f"\nStandardized residuals (should be N(0,1)):")
    print(f"  Mean: {fit_results['residuals_mean']:.6f}")
    print(f"  Std:  {fit_results['residuals_std']:.6f}")
    
    print(f"\nModel selection criteria:")
    print(f"  Log-likelihood: {fit_results['log_likelihood']:.4f}")
    print(f"  AIC: {fit_results['aic']:.4f}")
    print(f"  BIC: {fit_results['bic']:.4f}")
    
    # Verdict
    print("\n3. VERDICT")
    print("-" * 80)
    
    score = 0
    if norm_results['is_normal']:
        score += 1
        print("✓ Normality: PASS")
    else:
        print("✗ Normality: FAIL")
    
    if fit_results['mean_error'] < 0.01 and fit_results['std_error'] < 0.05:
        score += 1
        print("✓ Parameter fit: EXCELLENT")
    elif fit_results['mean_error'] < 0.05 and fit_results['std_error'] < 0.15:
        print("◆ Parameter fit: GOOD")
    else:
        print("✗ Parameter fit: POOR")
    
    if abs(fit_results['residuals_mean']) < 0.05 and abs(fit_results['residuals_std'] - 1.0) < 0.2:
        score += 1
        print("✓ Residuals: WELL-STANDARDIZED")
    else:
        print("✗ Residuals: DEVIATE FROM N(0,1)")
    
    print(f"\nOverall score: {score}/3")
    if score == 3:
        print("CONCLUSION: Excellent GBM fit ✓")
    elif score == 2:
        print("CONCLUSION: Good GBM fit ◆")
    else:
        print("CONCLUSION: Poor GBM fit - consider alternative models")
    
    print("=" * 80)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python load_calibrations.py <calibration_json> <price_csv> --date-col DATE --price-col PRICE --dt DT")
        print("\nExample:")
        print("  python load_calibrations.py calibrations/calibration_steel.json data.csv --date-col Date --price-col Price --dt 0.083")
        sys.exit(1)
    
    cal_file = sys.argv[1]
    price_file = sys.argv[2]
    
    # Parse optional arguments
    date_col = "Date"
    price_col = "Price"
    dt = 1.0
    
    for i in range(3, len(sys.argv), 2):
        if sys.argv[i] == "--date-col":
            date_col = sys.argv[i+1]
        elif sys.argv[i] == "--price-col":
            price_col = sys.argv[i+1]
        elif sys.argv[i] == "--dt":
            dt = float(sys.argv[i+1])
    
    try:
        # Load calibration
        cal_data = load_calibration(cal_file)
        commodity = cal_data.get('commodity', 'Unknown')
        
        # Load prices
        df = pd.read_csv(price_file, sep=';')
        prices = df[price_col]
        
        # Print results
        print_results(commodity, cal_data, prices, dt)
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)