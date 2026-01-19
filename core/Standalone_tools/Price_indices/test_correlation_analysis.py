#!/usr/bin/env python3
"""
Test script for correlation analysis functionality in Material_param.py

This script demonstrates how the new correlation analysis methods work
with multiple commodities.
"""

import numpy as np
import pandas as pd
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from Material_param import (
    compute_log_returns,
    calculate_pearson_correlation,
    calculate_spearman_correlation,
    compare_correlation_matrices,
    analyze_commodity_correlations
)


def create_sample_data():
    """
    Create sample commodity price data with correlated returns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with dates and three correlated commodity prices
    """
    np.random.seed(42)
    n_points = 252  # One year of trading days
    
    # Generate base random returns
    base_returns = np.random.normal(0.0003, 0.01, n_points)
    
    # Create three correlated price series
    # Commodity 1 (baseline)
    returns_1 = base_returns + np.random.normal(0, 0.005, n_points)
    
    # Commodity 2 (high correlation with 1)
    returns_2 = base_returns * 0.8 + np.random.normal(0, 0.005, n_points)
    
    # Commodity 3 (moderate correlation with 1, low with 2)
    returns_3 = base_returns * 0.4 + np.random.normal(0, 0.008, n_points)
    
    # Convert returns to prices
    price_1 = 100 * np.exp(np.cumsum(returns_1))
    price_2 = 50 * np.exp(np.cumsum(returns_2))
    price_3 = 75 * np.exp(np.cumsum(returns_3))
    
    # Create DataFrame
    dates = pd.date_range(start='2023-01-01', periods=n_points, freq='B')
    df = pd.DataFrame({
        'Date': dates,
        'Steel': price_1,
        'Copper': price_2,
        'Aluminum': price_3
    })
    
    return df


def test_individual_functions():
    """Test individual correlation functions."""
    print("=" * 80)
    print("TEST 1: Individual Correlation Functions")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data()
    
    print("\nSample data created:")
    print(f"  Shape: {df.shape}")
    print(f"  Columns: {df.columns.tolist()}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    
    # Compute log returns
    log_returns = compute_log_returns(df, ['Steel', 'Copper', 'Aluminum'])
    print(f"\n\nLog returns computed:")
    print(f"  Shape: {log_returns.shape}")
    print(f"  First 5 rows:")
    print(log_returns.head())
    
    # Calculate Pearson correlation
    pearson_results = calculate_pearson_correlation(log_returns)
    print(f"\n\nPearson Correlation Matrix:")
    print(pearson_results['correlation_matrix'])
    print(f"\nPearson p-values:")
    print(pearson_results['p_values'])
    
    # Calculate Spearman correlation
    spearman_results = calculate_spearman_correlation(log_returns)
    print(f"\n\nSpearman Rank Correlation Matrix:")
    print(spearman_results['correlation_matrix'])
    print(f"\nSpearman p-values:")
    print(spearman_results['p_values'])
    
    # Compare matrices
    comparison = compare_correlation_matrices(pearson_results, spearman_results)
    print(f"\n\nComparison Results:")
    print(f"  Max difference: {comparison['max_difference']:.6f}")
    print(f"  Mean difference: {comparison['mean_difference']:.6f}")
    print(f"  Agreement rate: {comparison['agreement_pct']:.1f}%")
    
    print(f"\n  Most discordant pairs:")
    for pair_info in comparison['discordant_pairs']:
        print(f"    {pair_info['pair']}: Δ = {pair_info['difference']:.6f}")


def test_integrated_analysis():
    """Test the integrated analyze_commodity_correlations function."""
    print("\n\n" + "=" * 80)
    print("TEST 2: Integrated Correlation Analysis")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data()
    
    # Run the integrated analysis
    analyze_commodity_correlations(df, 'Date', ['Steel', 'Copper', 'Aluminum'])


def test_single_commodity_handling():
    """Test that single commodity is handled gracefully."""
    print("\n\n" + "=" * 80)
    print("TEST 3: Single Commodity Handling")
    print("=" * 80)
    
    # Create sample data
    df = create_sample_data()
    
    # Try to analyze with single commodity
    print("\nAttempting correlation analysis with single commodity:")
    analyze_commodity_correlations(df, 'Date', ['Steel'])
    print("(Should skip analysis with appropriate message)")


if __name__ == "__main__":
    print("\nTesting Correlation Analysis Functionality")
    print("=" * 80)
    
    try:
        test_individual_functions()
        test_integrated_analysis()
        test_single_commodity_handling()
        
        print("\n\n" + "=" * 80)
        print("All tests completed successfully!")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n\nError during testing: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
