
# Author: [Vasilis Pettas]

import pandas as pd
import numpy as np
import re
from surrogates_interface.surrogates import TensorFlowModel
import os




def load_wind_timeseries(path, resolution, require_price=False):
    """
    Loads wind timeseries data and optionally checks for price column.

    Parameters:
    - path: path to CSV with columns ['timestamp', 'wsp', 'TI', 'wdir'] (and optionally 'price')
    - resolution: expected resolution, '10min' or '1h'
    - require_price: if True, checks that a 'price' column exists

    Returns:
    - DataFrame with wind (and optionally price) data
    """
    df = pd.read_csv(path)
    expected_cols = {"timestamp", "wsp", "TI", "wdir"}
    if require_price:
        expected_cols.add("price")
    missing = expected_cols - set(df.columns)
    if missing:
        raise ValueError(f"Wind timeseries is missing required columns: {missing}")

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Resolution check
    actual_freq = pd.infer_freq(df["timestamp"])
    if actual_freq is None:
        raise ValueError("Could not infer frequency of timestamps in wind data.")
    freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
    normalized = freq_map.get(actual_freq.lower(), actual_freq.lower())
    if normalized != resolution:
        raise ValueError(f"Wind timeseries resolution mismatch: expected {resolution}, got {normalized}")

    return df


def load_iec_distribution(path):
    """
    Loads 2D IEC distribution.

    Parameters:
    - path: CSV with columns ['wsp', 'TI', 'probability']

    Returns:
    - DataFrame for IEC analysis
    """
    df = pd.read_csv(path)
    required_cols = {"wsp", "TI", "probability"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"IEC distribution file is missing columns: {missing}")
    return df

def load_joint_distribution(path, require_price=False):
    """
    Loads joint distribution for wsp, TI, wdir and probability.

    Parameters:
    - path: CSV with required columns
    - require_price: if True, 'price' column must be present

    Returns:
    - DataFrame with columns ['wsp', 'TI', 'wdir', 'probability', ...]
    """
    df = pd.read_csv(path)
    required_cols = {"wsp", "TI", "wdir", "probability"}
    if require_price:
        required_cols.add("price")
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Joint distribution is missing required columns: {missing}")
    return df
