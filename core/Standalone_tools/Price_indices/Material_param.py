#!/usr/bin/env python3
"""
Calibrate GBM (mu, sigma) and mean-reverting OU (kappa, theta, sigma)
from historical price/index time series stored in a CSV.

Usage examples
--------------
Single series (column 'Price'):
    python calibrate_commodities.py data.csv --date-col Date --price-cols Price

Multiple materials in same file (columns 'Steel','Copper',...):
    python calibrate_commodities.py materials.csv --date-col Date --price-cols Steel Copper

If --price-cols is omitted, all numeric columns except the date column are used.
"""

import argparse
import sys
import numpy as np
import pandas as pd


def infer_dt_years(dates: pd.Series) -> float:
    """Infer time step Δt in years from a pandas datetime series."""
    dates = dates.sort_values()
    diffs = dates.diff().dropna().dt.days.values

    if len(diffs) == 0:
        raise ValueError("Not enough rows to infer time step (need at least two timestamps).")

    median_days = np.median(diffs)
    std_days = np.std(diffs)

    if std_days > 0.1 * median_days:
        print(
            f"WARNING: time step is irregular (median={median_days:.2f} days, std={std_days:.2f} days). "
            "Using median step for Δt.",
            file=sys.stderr,
        )

    dt_years = median_days / 365.25
    if dt_years <= 0:
        raise ValueError("Inferred non-positive time step.")
    return dt_years


def calibrate_gbm(prices: pd.Series, dt_years: float):
    """
    Calibrate GBM parameters (mu, sigma) from a price series.

    GBM: dS = μ S dt + σ S dW
    Discrete log-returns r_t = ln(S_t / S_{t-1}):
        r_t ~ N( (μ - 0.5 σ^2) Δt, σ^2 Δt )
    """
    prices = prices.astype(float)
    log_returns = np.log(prices / prices.shift(1)).dropna()

    if len(log_returns) < 2:
        raise ValueError("Not enough data points for GBM calibration (need at least 3 prices).")

    r_bar = log_returns.mean()
    var_r = log_returns.var(ddof=1)

    sigma = np.sqrt(var_r / dt_years)
    mu = r_bar / dt_years + 0.5 * sigma ** 2

    return {
        "mu": float(mu),
        "sigma": float(sigma),
        "r_bar": float(r_bar),
        "var_r": float(var_r),
    }


def calibrate_ou_logprice(prices: pd.Series, dt_years: float):
    """
    Calibrate OU parameters (kappa, theta, sigma) on log-prices.

    OU on X_t = ln S_t:
        dX = κ(θ - X) dt + σ dW

    Discrete form:
        X_{t+Δt} = a + b X_t + ε_t
    where:
        b = exp(-κ Δt)
        a = θ (1 - b)
    """
    X = np.log(prices.astype(float))
    X_t = X.shift(1).dropna()
    X_tp = X.loc[X_t.index]  # aligned X_{t+Δt}

    if len(X_t) < 2:
        raise ValueError("Not enough data points for OU calibration (need at least 3 prices).")

    # OLS regression: X_tp = a + b * X_t + eps
    X_mean = X_t.mean()
    Y_mean = X_tp.mean()

    cov = ((X_t - X_mean) * (X_tp - Y_mean)).sum()
    var_X = ((X_t - X_mean) ** 2).sum()

    b = cov / var_X
    a = Y_mean - b * X_mean

    # residuals
    residuals = X_tp - (a + b * X_t)
    var_eps = residuals.var(ddof=2)  # N-2 dof for regression

    if b <= 0 or b >= 1:
        print(
            f"WARNING: estimated b={b:.4f} is outside (0,1); OU mean reversion may be a poor fit.",
            file=sys.stderr,
        )

    # Convert to continuous-time OU parameters
    kappa = -np.log(b) / dt_years
    theta = a / (1.0 - b)

    sigma = np.sqrt(2 * kappa * var_eps / (1.0 - b ** 2))

    return {
        "kappa": float(kappa),
        "theta": float(theta),
        "sigma": float(sigma),
        "a": float(a),
        "b": float(b),
        "var_eps": float(var_eps),
    }


def main():
    parser = argparse.ArgumentParser(description="Calibrate GBM and OU parameters from CSV price series.")
    parser.add_argument("csv_path", help="Path to CSV file with a date column and one or more price/index columns.")
    parser.add_argument(
        "--date-col",
        required=True,
        help="Name of the date/time column in the CSV."
    )
    parser.add_argument(
        "--price-cols",
        nargs="+",
        help="Names of price/index columns. If omitted, all numeric columns except the date column are used."
    )
    parser.add_argument(
        "--date-format",
        default=None,
        help="Optional explicit date format (passed to pandas.to_datetime)."
    )

    args = parser.parse_args()

    # Load data
    try:
        df = pd.read_csv(args.csv_path)
    except Exception as e:
        print(f"Error reading CSV: {e}", file=sys.stderr)
        sys.exit(1)

    if args.date_col not in df.columns:
        print(f"Date column '{args.date_col}' not found in CSV.", file=sys.stderr)
        sys.exit(1)

    # Parse dates
    try:
        df[args.date_col] = pd.to_datetime(df[args.date_col], format=args.date_format)
    except Exception as e:
        print(f"Error parsing dates: {e}", file=sys.stderr)
        sys.exit(1)

    df = df.sort_values(args.date_col).reset_index(drop=True)

    # Infer Δt in years
    try:
        dt_years = infer_dt_years(df[args.date_col])
    except Exception as e:
        print(f"Error inferring time step: {e}", file=sys.stderr)
        sys.exit(1)

    print(f"Inferred time step Δt ≈ {dt_years:.6f} years\n")

    # Determine which price columns to use
    if args.price_cols is not None:
        price_cols = args.price_cols
    else:
        # Use all numeric columns except the date column
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        price_cols = [c for c in numeric_cols if c != args.date_col]

    if not price_cols:
        print("No price columns specified or found as numeric columns.", file=sys.stderr)
        sys.exit(1)

    for col in price_cols:
        if col not in df.columns:
            print(f"Price column '{col}' not found in CSV, skipping.", file=sys.stderr)
            continue

        series = df[col].dropna()
        if len(series) < 3:
            print(f"Not enough data for column '{col}' (need at least 3 non-NaN rows), skipping.", file=sys.stderr)
            continue

        print("=" * 80)
        print(f"Series: {col}")

        # GBM calibration
        try:
            gbm_params = calibrate_gbm(series, dt_years)
            print("\nGBM parameters (annualised):")
            print(f"  mu    = {gbm_params['mu']:.6f}")
            print(f"  sigma = {gbm_params['sigma']:.6f}")
        except Exception as e:
            print(f"  GBM calibration failed: {e}", file=sys.stderr)

        # OU calibration
        try:
            ou_params = calibrate_ou_logprice(series, dt_years)
            print("\nOU (mean-reverting on log-price) parameters (annualised):")
            print(f"  kappa = {ou_params['kappa']:.6f}")
            print(f"  theta = {ou_params['theta']:.6f}")
            print(f"  sigma = {ou_params['sigma']:.6f}")
        except Exception as e:
            print(f"  OU calibration failed: {e}", file=sys.stderr)

        print()

    print("=" * 80)
    print("Done.")


if __name__ == "__main__":
    main()
