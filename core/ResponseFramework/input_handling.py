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

def load_additional_inputs(path, resolution):
    """
    Loads additional time series inputs (e.g. temperature, bats, curtailment).

    Parameters:
    - path: CSV with timestamp column and one or more auxiliary signals
    - resolution: expected resolution, '10min' or '1h'

    Returns:
    - DataFrame with timestamp-aligned signals
    """
    df = pd.read_csv(path)
    if "timestamp" not in df.columns:
        raise ValueError("Auxiliary input file must contain a 'timestamp' column.")
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Resolution check
    actual_freq = pd.infer_freq(df["timestamp"])
    if actual_freq is None:
        raise ValueError("Could not infer frequency of timestamps in auxiliary inputs.")
    freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
    normalized = freq_map.get(actual_freq.lower(), actual_freq.lower())
    if normalized != resolution:
        raise ValueError(f"Auxiliary input resolution mismatch: expected {resolution}, got {normalized}")

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

def check_alignment(df1, df2):
    """
    Verifies that timestamps in two DataFrames are perfectly aligned.

    Parameters:
    - df1, df2: DataFrames with 'timestamp' columns

    Raises:
    - ValueError if timestamps differ
    """
    if not df1["timestamp"].equals(df2["timestamp"]):
        raise ValueError("Timestamps in input files are not aligned.")


def load_surrogate(path=None, use_sector_average=False):
    """
    Dummy surrogate model for testing, compatible with RA and SA inflow.

    Parameters
    ----------
    path : str or None
        Ignored. Included for interface compatibility.
    use_sector_average : bool
        If True, uses SA inflow from 4 sectors. If False, uses RA inflow.

    Returns
    -------
    surrogate_fn : callable
        Function accepting structured input from extract_surrogate_inputs,
        returning per-turbine output arrays.
    """
    def dummy_surrogate(input_dict):
        """
        Computes fake surrogate outputs using RA or SA inflow.

        Parameters
        ----------
        input_dict : dict
            Must contain:
                - RA: dict with WS, TI, yaw, power
                - SA: optional, 4-sector inflow dict
                - controller_mode: (n_turbines, n_times)

        Returns
        -------
        dict
            Each output is a 1D array of shape (n_turbines,)
        """
        op_mode = input_dict["controller_mode"][:, 0]  # shape (n_turbines,)
        n_turbines = len(op_mode)

        if use_sector_average:
            # Use sector 1 (first of 4 sectors)
            SA = input_dict["SA"]
            WS = SA["WS"][:, 0, 0]   # (n_turbines,)
            TI = SA["TI"][:, 0, 0]
            yaw = SA["yaw"][:, 0]
            power = SA["power"][:, 0]
        else:
            RA = input_dict["RA"]
            WS = RA["WS"][:, 0]
            TI = RA["TI"][:, 0]
            yaw = RA["yaw"][:, 0]
            power = RA["power"][:, 0]

        # Initialize output arrays
        D_TBFA = np.zeros(n_turbines)
        D_TBSS = np.zeros(n_turbines)
        ADC_pitch = np.zeros(n_turbines)
        ADC_yaw = np.zeros(n_turbines)

        for i in range(n_turbines):
            if op_mode[i] == "normal":
                D_TBFA[i] = WS[i] * TI[i] * 1e5
                D_TBSS[i] = WS[i] * TI[i] * 5e4
                ADC_pitch[i] = abs(yaw[i]) * 50
                ADC_yaw[i] = abs(yaw[i]) * 10
            else:
                D_TBFA[i] = 0
                D_TBSS[i] = 0
                ADC_pitch[i] = 0
                ADC_yaw[i] = 0

        return {
            "D_TBFA": D_TBFA,
            "D_TBSS": D_TBSS,
            "ADC_pitch": ADC_pitch,
            "ADC_yaw": ADC_yaw
        }

    return dummy_surrogate


def load_surrogate_models(
    model_path_base,
    scaler_path_base,
    channels,
    use_sector_average=False,
    supports_operating_modes=False,
    modes=("normal",),
):
    """
    Loads TensorFlow surrogate models with optional mode-specific separation.

    Parameters
    ----------
    model_path_base : str
        Path to directory (or base directory) containing model .keras files.
    scaler_path_base : str
        Path to directory (or base directory) containing scaler .h5 files.
    channels : list of str
        List of surrogate output channels (e.g. ['tbfa', 'tbss', ...]).
    use_sector_average : bool
        If True, loads SA models; otherwise assumes RA (naming is user responsibility).
    supports_operating_modes : bool
        If True, loads a separate surrogate per operational mode.
    modes : tuple of str
        Which operational modes to load if supports_operating_modes is True.

    Returns
    -------
    surrogates : dict
        Either a flat dict (channel → model) or nested dict (mode → channel → model).
    metadata : dict
        Dictionary describing surrogate configuration for settings.json
    """
    def load_single_model(model_dir, scaler_dir, channel):
        model_file = os.path.join(model_dir, f"{channel}.keras")
        scaler_file = os.path.join(scaler_dir, f"scaler_{channel}.h5")
        return TensorFlowModel.load_h5(model_path=model_file, extra_data_path=scaler_file)

    if not supports_operating_modes:
        models = {}
        for ch in channels:
            models[ch] = load_single_model(model_path_base, scaler_path_base, ch)

        metadata = {
            "type": "SA" if use_sector_average else "RA",
            "channels": channels,
            "supports_operating_modes": False,
            "model_base_path": model_path_base,
            "scaler_base_path": scaler_path_base
        }
        return models, metadata

    # Mode-specific structure
    models_by_mode = {}
    for mode in modes:
        mode_model_dir = os.path.join(model_path_base, mode)
        mode_scaler_dir = os.path.join(scaler_path_base, mode)

        mode_models = {}
        for ch in channels:
            mode_models[ch] = load_single_model(mode_model_dir, mode_scaler_dir, ch)

        models_by_mode[mode] = mode_models

    metadata = {
        "type": "SA" if use_sector_average else "RA",
        "channels": channels,
        "supports_operating_modes": True,
        "modes": list(modes),
        "model_base_path": model_path_base,
        "scaler_base_path": scaler_path_base
    }
    return models_by_mode, metadata


def load_baseline_inputs(baseline_dir,expected_timestamps):
    """
    Load farm-level baseline series from '<baseline_dir>/farm_timeseries.parquet'
    and return them as arrays under names:
      - 'baseline.energy'  (required, from 'FarmEnergy')
      - 'baseline.revenue' (optional, from 'FarmRevenue' if present)

    Assumptions (v1, hardcoded by design):
    - Baseline outputs are in a directory with fixed filenames.
    - We only read 'farm_timeseries.parquet'.
    - Time column is 'timestamp' (or the index is a DatetimeIndex named 'timestamp').
    - No resampling/merging here; timestamps must match the main input exactly.

    Parameters
    ----------
    baseline_dir : str
        Path to the baseline results directory containing 'farm_timeseries.parquet'.
    expected_timestamps : array-like / pandas.DatetimeIndex / pandas.Series
        The timeline from the main wind–price input. Must match exactly (length and values).

    Returns
    -------
    dict
        {'baseline.energy': np.ndarray, 'baseline.revenue': np.ndarray (if available)}

    Raises
    ------
    ValueError
        - If the directory/file doesn't exist.
        - If 'timestamp' is missing or timestamps don't exactly match expected_timestamps.
        - If 'FarmEnergy' is missing.
    """
    # Resolve file path and check existence
    farm_ts_path = os.path.join(baseline_dir, "farm_timeseries.parquet")
    if not os.path.isdir(baseline_dir):
        raise ValueError(f"Baseline directory not found: {baseline_dir}")
    if not os.path.exists(farm_ts_path):
        raise ValueError(f"'farm_timeseries.parquet' not found in: {baseline_dir}")

    # Read parquet
    try:
        df = pd.read_parquet(farm_ts_path)
    except Exception as e:
        raise ValueError(f"Failed to read baseline file: {farm_ts_path}\n{e}")

    # Extract timestamps from file
    if "timestamp" in df.columns:
        ts_file = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
    elif isinstance(df.index, pd.DatetimeIndex) and df.index.name == "timestamp":
        ts_file = df.index
    else:
        raise ValueError(
            f"[farm_timeseries.parquet] Missing 'timestamp' column (or DatetimeIndex named 'timestamp'). "
            f"Available columns: {list(df.columns)}"
        )

    # Normalize expected timestamps
    ts_expected = pd.DatetimeIndex(pd.to_datetime(expected_timestamps))

    # Exact equality checks (length + value-by-value)
    if len(ts_file) != len(ts_expected):
        raise ValueError(
            f"[farm_timeseries.parquet] Timestamp length mismatch: "
            f"file={len(ts_file)}, expected={len(ts_expected)}."
        )
    if not ts_file.equals(ts_expected):
        # show a few mismatching positions to help debugging
        mism = np.where(ts_file.values != ts_expected.values)[0]
        preview = ", ".join(map(str, mism[:5]))
        raise ValueError(
            "[farm_timeseries.parquet] Timestamps do not exactly match the main input. "
            f"First mismatching positions: {preview}"
        )

    # Required/optional columns
    if "FarmEnergy" not in df.columns:
        raise ValueError(
            "[farm_timeseries.parquet] Required column 'FarmEnergy' not found. "
            f"Available: {list(df.columns)}"
        )
    out = pd.DataFrame({"timestamp":ts_expected })  # keep timestamps for consistency
    out ["baseline.energy"]= df["FarmEnergy"].to_numpy()

    if "FarmRevenue" in df.columns:
        out["baseline.revenue"] = df["FarmRevenue"].to_numpy()

    return out


def load_aux_inputs(expected_timestamps,aux_paths):
    """
    Load one or more CSV auxiliary files and return a DataFrame with:
      - 'timestamp' (first column; must exactly match expected_timestamps)
      - all other columns from the files (as-is)

    Rules (v1, per your spec):
    - Only CSV is supported here.
    - Each file MUST contain a 'timestamp' column.
    - Timestamps must match expected_timestamps exactly (length and values).
    - We do not resample or fill; fail fast on mismatch.
    - If the same column name appears in more than one file, we error and
      report the duplicate column and the two file paths.
    - We keep whatever dtypes the CSV yields (numeric or strings).
    - If aux_paths is None or empty, we return a DataFrame with just 'timestamp'.

    Parameters
    ----------
    aux_paths : list[str] | None
        Paths to CSV files to load. If None/empty, returns just the timestamp column.
    expected_timestamps : array-like / pandas.DatetimeIndex / pandas.Series
        Master timeline from the main wind–price input; must match exactly.

    Returns
    -------
    pandas.DataFrame
        Columns: 'timestamp' + all loaded aux columns. Order preserves file order.
    """

    
    
    # Normalize expected timestamps
    ts_expected = pd.DatetimeIndex(pd.to_datetime(expected_timestamps))

    # Start output with timestamp only
    out = pd.DataFrame({"timestamp": ts_expected})

    if not aux_paths:
       
        return out

    # Track where each column came from to detect duplicates across files
    seen_cols: dict[str, str] = {}

    for path in aux_paths:
        path = str(path)
        if not os.path.exists(path):
            raise ValueError(f"[aux] File not found: {path}")

        # Read CSV
        try:
            df = pd.read_csv(path)
        except Exception as e:
            raise ValueError(f"[aux] Failed to read CSV: {path}\n{e}")

        # Require 'timestamp' column (no index fallback here, per your instruction)
        if "timestamp" not in df.columns:
            raise ValueError(
                f"[aux] Missing 'timestamp' column in {path}. "
                f"Available columns: {list(df.columns)}"
            )

        # Exact alignment checks
        ts_file = pd.DatetimeIndex(pd.to_datetime(df["timestamp"]))
        if len(ts_file) != len(ts_expected):
            raise ValueError(
                f"[aux::{os.path.basename(path)}] Timestamp length mismatch: "
                f"file={len(ts_file)}, expected={len(ts_expected)}."
            )
        if not ts_file.equals(ts_expected):
            mism = np.where(ts_file.values != ts_expected.values)[0]
            preview = ", ".join(map(str, mism[:5]))
            raise ValueError(
                f"[aux::{os.path.basename(path)}] Timestamps do not exactly match the main input. "
                f"First mismatching positions: {preview}"
            )

        # Add every non-timestamp column, checking for duplicates across files
        cols = [c for c in df.columns if c != "timestamp"]
        for c in cols:
            if c in seen_cols:
                prev_path = seen_cols[c]
                raise ValueError(
                    "Duplicate auxiliary column detected across files:\n"
                    f"  column: '{c}'\n"
                    f"  files : {prev_path}\n"
                    f"          {path}\n"
                    "Rename one of these columns to proceed."
                )
            seen_cols[c] = path

        # Append columns to output (order preserved by file order)
        for c in cols:
            out[c] = df[c].to_numpy()

    return out


def centers_to_edges(centers):
    """
    Given an array of bin centers, return the inferred bin edges.

    Assumes centers are sorted in ascending order.
    """
    centers = np.array(sorted(centers))
    midpoints = (centers[1:] + centers[:-1]) / 2

    edges = np.zeros(len(centers) + 1)
    edges[1:-1] = midpoints
    edges[0] = centers[0] - (centers[1] - centers[0]) / 2
    edges[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2

    return edges

def calculate_bin_edges(lookup_df):
    """
    Builds bin specifications for wsp, wdir, and TI.

    For each dimension, returns a dict with:
      - 'edges':  ndarray of edges if multi-bin, else None
      - 'constant': bool, True if only one unique center in lookup_df
      - 'center':   the single center if constant, else None

    Returns:
    --------
    specs : dict
        {'wsp': {...}, 'wdir': {...}, 'TI': {...}}
    """
    specs = {}
    for dim in ('wsp', 'wdir', 'TI'):
        col = f"{dim}_bin"
        centers = np.sort(lookup_df[col].unique())
        if len(centers) == 1:
            # Single-value → “big bin” on that center
            specs[dim] = {
                'edges':    None,
                'constant': True,
                'center':   float(centers[0])
            }
        else:
            # Build true edges from multiple centers
            specs[dim] = {
                'edges':    centers_to_edges(centers),
                'constant': False,
                'center':   None
            }
    return specs


def validate_lookup_turbine_ids(lookup_df, layout_df):
    """
    Validates that the turbine IDs defined in the lookup table match the turbine IDs
    in the wind farm layout.

    Parameters:
    ----------
    lookup_df : pd.DataFrame
        DataFrame for the lookup controller, must have 'yaw_*' and 'power_*' columns.
    layout_df : pd.DataFrame
        Wind farm layout DataFrame with a column 'id' for turbine IDs.

    Raises:
    -------
    ValueError
        If there is a mismatch between the turbine IDs in the lookup table
        and the turbine IDs in the layout.

    Behavior:
    ---------
    - Looks for all unique suffixes in yaw_* and power_* columns.
    - Verifies yaw_* and power_* IDs match each other.
    - Verifies the inferred IDs match layout['id'] exactly.
    - Raises an error with clear info if not.
    """

    # Extract turbine IDs from lookup columns
    yaw_cols = [col for col in lookup_df.columns if col.startswith('yaw_')]
    power_cols = [col for col in lookup_df.columns if col.startswith('power_')]

    yaw_ids = {int(re.findall(r'\d+', col)[0]) for col in yaw_cols}
    power_ids = {int(re.findall(r'\d+', col)[0]) for col in power_cols}

    if yaw_ids != power_ids:
        raise ValueError(
            f"Mismatch in yaw and power columns in lookup table:\n"
            f"Yaw IDs: {sorted(yaw_ids)}\n"
            f"Power IDs: {sorted(power_ids)}"
        )

    lookup_turbine_ids = sorted(yaw_ids)

    # Extract turbine IDs from layout
    if 'id' not in layout_df.columns:
        raise ValueError("Layout DataFrame must contain an 'id' column for turbine IDs.")

    layout_turbine_ids = sorted(layout_df['id'].tolist())

    if lookup_turbine_ids != layout_turbine_ids:
        raise ValueError(
            f"Turbine ID mismatch detected!\n"
            f"Lookup file turbine IDs: {lookup_turbine_ids}\n"
            f"Layout turbine IDs: {layout_turbine_ids}\n"
            f"Ensure your lookup file and farm layout are aligned."
        )

    # If all good, no return value — validation passed


