import pandas as pd
import numpy as np
import time
from typing import Optional
import xarray as xr
from farm_setup import  compute_effective_inflow
from wind_farm_loads.py_wake import predict_loads_rotor_average, predict_loads_sector_average


def simulate_block(
    wind_data,
    farm,
    surrogate,
    resolution,
    control_fn,
    use_sector_average=False,
    supports_operating_modes=False,
    channel_processing_specs=None,
    wsp_min=3.0,
    wsp_max=25.0,
    price_series=None,
    baseline_df=None,       
    aux_df=None,           
    return_turbine_timeseries=False,
    return_turbine_timeseries_cum=False,
    return_farm_timeseries=False,
):
    """
    Vectorized simulation of a wind farm over a time series.

    1. Generate control setpoints and modes for all timesteps.
    2. Call flow model once over the whole series.
    3. Extract surrogate inputs and predict outputs in batch.
    4. Build instantaneous DataFrame with repeat-factor handling by channel type.
    5. Aggregate per-turbine, cumulative time-series, and farm-level metrics.

    Parameters
    ----------
    wind_data : pd.DataFrame
        Must contain columns ['timestamp', 'wsp', 'TI', 'wdir']
    farm : dict
        Output of initialize_HKN_pywake_farm(), with 'layout_df', 'flow_model', 'x', 'y'.
    surrogate : dict
        Surrogate model dict or mode-based dict.
    resolution : {'10min', '1h'}
    control_fn : callable
        control_fn(wsp, TI, wdir, ts, surrogate_output=None, extra_inputs) -> dict
    use_sector_average : bool
    supports_operating_modes : bool
    channel_processing_specs : dict
        Mapping channel -> {'type': 'sum'|'damage'|'harmonic', 'wohler':..., 'nref':...}
    wsp_min, wsp_max : float
    price_series : np.ndarray or None
    baseline_df : pd.DataFrame or None with baseline results
    aux_df : pd.DataFrame or None with auxiliary inputs combined from aux files
    return_turbine_timeseries : bool
    return_turbine_timeseries_cum : bool
    return_farm_timeseries : bool

    Returns
    -------
    result_df : pd.DataFrame of final cumulative values per turbine
    turbine_timeseries : pd.DataFrame or None instantaneous metrics per turbine
    turbine_timeseries_cum : pd.DataFrame or None Cumulative ts metrics per turbine
    farm_timeseries : pd.DataFrame or None reveneue and energy of the farm over time
    """
    # --- Setup ---
    layout = farm['layout_df']
    turbine_ids = layout['id'].tolist()
    n_turbs = len(turbine_ids)
    # Repeat factor for resolution
    repeat_factor = 6 if resolution == '1h' else 1
    
    # Create data inputs cleaning nans
    required_cols = ['wsp', 'TI', 'wdir']
    full_timestamps = wind_data['timestamp'].values
    if price_series is not None:
        required_cols.append('price')

    
    valid_mask = ~wind_data[required_cols].isna().any(axis=1)

    timestamps = full_timestamps[valid_mask]

    wsp = wind_data['wsp'].values[valid_mask]
    TI = wind_data['TI'].values[valid_mask]
    wdir = wind_data['wdir'].values[valid_mask]

    
    extras = {}  # Keep adding here if more values come in the main file
    if price_series is not None:
            extras["price"] = price_series
    if "temperature" in wind_data.columns:
        extras["temperature"] = wind_data["temperature"].to_numpy()
    if baseline_df is not None:
        for c in baseline_df.columns:
            if c == "timestamp":
                continue
            if c in extras:
                raise ValueError(f"Duplicate extras key '{c}' from baseline and another source.")
            extras[c] = baseline_df[c].to_numpy()    
    if aux_df is not None:
        for c in aux_df.columns:
            if c == "timestamp":
                continue
            if c in extras:
                raise ValueError(f"Duplicate extras key '{c}' from baseline and another source.")
            extras[c] = aux_df[c].to_numpy()
    
    extra_inputs = {k: np.asarray(v)[valid_mask] for k, v in extras.items()}

      # --- 1. Control & state arrays ---
    prev_states = {tid: 'normal' for tid in turbine_ids}
    yaw_mat = np.zeros((n_turbs, len(timestamps)))
    power_mat = np.zeros_like(yaw_mat)
    tilt_mat    = np.zeros_like(yaw_mat)
    op_mode_mat = np.empty_like(yaw_mat, dtype='<U8')
    operating_mat = np.zeros_like(yaw_mat, dtype=int)


    for t in range(len(timestamps)):
        wsp_t, TI_t, wdir_t, ts_t = wsp[t], TI[t], wdir[t], timestamps[t]
        extra_inputs_t = {k: v[t] for k, v in extra_inputs.items()}

        # call existing helper
        yaw_vec, p_vec, tilt_vec, op_vec, op_flag, new_states = get_setpoint_opmode(
            turbine_ids, 
            control_fn, 
            prev_states,
            wsp_t, 
            TI_t, 
            wdir_t, 
            ts_t, 
            extra_inputs_t,
            wsp_min, 
            wsp_max, 
            track_state=True
        )
        prev_states = new_states
        yaw_mat[:, t]       = yaw_vec.flatten()
        power_mat[:, t]     = p_vec.flatten()
        op_mode_mat[:, t]   = op_vec.flatten()
        operating_mat[:, t] = op_flag.flatten()
        tilt_mat[:, t] = tilt_vec.flatten()

    tik = time.time()
    # --- 2. Flow model batch call ---
    sim_res, sa, inflow_df, power_df = compute_effective_inflow(
        wsp=wsp,
        TI=TI, 
        wdir=wdir,
        x=farm['x'], 
        y=farm['y'],
        yaw=yaw_mat, 
        tilt=tilt_mat,
        operating=operating_mat,
        flow_model=farm['flow_model'],
        flow_type=farm.get('flow_type'),
        use_sector_average=use_sector_average,
        dtype=np.float64
    )
    tok = time.time()
    print(f"Execution time inflow: {tok - tik:.6f} seconds")

    # --- 2c. Clamp TI range ---
    if use_sector_average:
        TI_vals = sa["TI"]
        sa["TI"] = np.clip(TI_vals, 0.045, 0.349)  # returns a new array and replaces the dict value
    else:
        TI_vals = np.asarray(sim_res.TI_eff.values)
        sim_res["TI_eff"].values = np.clip(TI_vals, 0.045, 0.349)  # update in-place

    
    # Raw 10-min energy, then scale for resolution
    raw_energy = power_df.values.T / 6.0  # MWh
    energy_inst = raw_energy * repeat_factor

    # Revenue
    if price_series is not None:
        rev_inst = energy_inst * extra_inputs["price"][np.newaxis, :]
    else:
        rev_inst = None

    # --- 3. Surrogate batch ---
    input_dict = extract_surrogate_inputs(
        sim_res=sim_res, sa=sa,
        yaw=yaw_mat, tilt=tilt_mat,
        power_demand=power_mat,
        operating=operating_mat,
        controller_mode=op_mode_mat,
        timestamps=timestamps
    )

    tik = time.time()
    out_sur = predict_surrogate_outputs(
        surrogates=surrogate,
        surrogate_inputs=input_dict,
        channels=list(channel_processing_specs.keys()),
        use_sector_average=use_sector_average,
        supports_operating_modes=supports_operating_modes,
        dtype=np.float64,
        wsp_min=wsp_min,
        wsp_max=wsp_max,
    )
    tok = time.time()
    print(f"Execution time getting RA/SA and probing surrogate: {tok - tik:.6f} seconds")

    # --- 4. Yaw travel ---
    total_yaw = yaw_mat + (wdir[np.newaxis, :]% 360)
    yaw_travel_inst = np.zeros_like(total_yaw)
    yaw_travel_inst[:, 1:] = np.abs(
        (total_yaw[:, 1:] - total_yaw[:, :-1] + 180) % 360 - 180
    )
    yaw_travel_inst[:, 0] = 0.0

    # --- 5. Build inst_data in long form ---
    rows = []
    for j, tid in enumerate(turbine_ids):
        for t in range(len(timestamps)):
            row = {
                'timestamp': timestamps[t],
                'id': tid,
                'YawSetpoint': yaw_mat[j, t],
                'PowerSetpoint': power_mat[j, t],
                'OpMode': op_mode_mat[j, t],
                'Energy': energy_inst[j, t],
                'YawTravel': yaw_travel_inst[j, t],
            }

            if rev_inst is not None:
                row['Revenue'] = rev_inst[j,t]
            # surrogate channels
            for ch, vals in out_sur.items():
                spec = channel_processing_specs[ch]
                v = vals[j,t]
                if spec['type'] == 'sum':
                    row[ch] = v * repeat_factor
                elif spec['type'] == 'damage':
                    damage = spec['nref'] * (v**spec['wohler'])
                    # to get DEL for inst ts:
                    D_rep = damage * repeat_factor
                    row[ch] = (D_rep/spec['nref'])**(1/spec['wohler'])
                else:  # harmonic
                    row[ch] = v
            rows.append(row)
    inst_df = pd.DataFrame(rows)


    # --- 6. Aggregations ---
    # result_df
    agg = []
    for tid in turbine_ids:
        df_t = inst_df[inst_df['id']==tid]
        out = {'id': tid}
        for ch, spec in channel_processing_specs.items():
            if spec['type']=='sum':
                out[ch] = df_t[ch].sum()
            elif spec['type']=='damage':
                # sum linear damage
                d = spec['nref']*(df_t[ch]**spec['wohler'])
                out[ch] = d.sum()
            elif  spec['type']=='harmonic':
                # arr = df_t[ch].values
                out[ch] = harmonic_mean(df_t[ch].values)
        # also include Energy, Revenue etc
        out['Energy'] = df_t['Energy'].sum()
        if rev_inst is not None:
            out['Revenue'] = df_t['Revenue'].sum()
        out['YawTravel'] = df_t['YawTravel'].sum()
        agg.append(out)
    result_df = pd.DataFrame(agg)

    # turbine_timeseries
    if return_turbine_timeseries:
        turbine_timeseries = inst_df.pivot(index='timestamp', columns='id')  
        turbine_timeseries = turbine_timeseries.reindex(full_timestamps)
        turbine_timeseries.index.name = 'timestamp'
        turbine_timeseries.reset_index(inplace=True)  # Ensure timestamp is a column
        # turbine_timeseries = insert_nan_timestamps(turbine_timeseries, missing_timestamps)
    else:
        turbine_timeseries = None

    # turbine_timeseries_cum
    turbine_timeseries_cum = None
    if return_turbine_timeseries_cum:
        cum = inst_df.copy()
        # for each channel
        for col, spec in channel_processing_specs.items():
            if spec['type'] == 'sum':
                cum[col] = cum.groupby('id')[col].cumsum()

            elif spec['type'] == 'damage':
                # Step 1: convert DEL → damage increment
                damage_inst = spec['nref'] * (cum[col] ** spec['wohler'])
                # Step 2: accumulate linear damage
                cum[col] = damage_inst.groupby(cum['id']).cumsum()
            elif spec['type'] == 'harmonic':
                # running harmonic mean
                cum[col] = cum.groupby('id')[col].transform(running_harmonic)
            else:
                raise ValueError(f"Unknown channel type: {spec['type']}")
        # also cumulative Energy and Revenue
        cum['Energy'] = cum.groupby('id')['Energy'].cumsum()
        if rev_inst is not None:
            cum['Revenue'] = cum.groupby('id')['Revenue'].cumsum()
        # cumulative YawTravel
        cum['YawTravel'] = cum.groupby('id')['YawTravel'].cumsum()

        turbine_timeseries_cum = cum.pivot(index='timestamp', columns='id')
        turbine_timeseries_cum = turbine_timeseries_cum.reindex(full_timestamps)
        turbine_timeseries_cum.index.name = 'timestamp'
        turbine_timeseries_cum.reset_index(inplace=True)  # Ensure timestamp is a column
        # turbine_timeseries_cum = insert_nan_timestamps(turbine_timeseries_cum, missing_timestamps)


    # farm_timeseries
    farm_timeseries = None
    if return_farm_timeseries:
        grp = inst_df.groupby('timestamp')
        farm_timeseries = pd.DataFrame({
            'FarmEnergy': grp['Energy'].sum(),
            'FarmRevenue': grp['Revenue'].sum() if rev_inst is not None else None
        })

        farm_timeseries.index.name = 'timestamp'
        farm_timeseries = farm_timeseries.reindex(full_timestamps)
        farm_timeseries = farm_timeseries.reset_index()
            

    return result_df, turbine_timeseries, turbine_timeseries_cum, farm_timeseries

def harmonic_mean(x):
    """
    Compute the harmonic mean of 1D array-like x, ignoring zeros.
    Returns a single scalar (np.nan if all zeros).
    """
    a = np.asarray(x, dtype=float)
    mask = a > 0
    if not mask.any():
        return np.nan
    return mask.sum() / np.sum(1.0 / a[mask])


def running_harmonic(x: pd.Series) -> np.ndarray:
    """
    Compute the *running* harmonic mean for a 1-D series x,
    skipping any zero entries (i.e. no divide-by-zero).
    If all seen values are zero, returns 0.0 for that timestamp.
    """
    vals = x.values.astype(float)
    n = len(vals)

    # build inv = 1/val for val>0, else inv=0
    inv = np.zeros_like(vals)
    mask = vals > 0
    inv[mask] = 1.0 / vals[mask]

    # cumulative sum of inverses
    cum_inv = np.cumsum(inv)

    # running count of total samples (including zeros, if you prefer)
    counts = np.arange(1, n + 1, dtype=float)

    # harmonic mean so far = counts / cum_inv, but only where cum_inv>0
    harm = np.zeros_like(vals)
    nonzero = cum_inv > 0
    harm[nonzero] = counts[nonzero] / cum_inv[nonzero]

    return harm


def get_setpoint_opmode(
    turbine_ids,
    control_fn,
    prev_states,
    wsp,
    TI,
    wdir,
    ts,
    extra_inputs,
    wsp_min,
    wsp_max,
    track_state=True
):
    """
    Evaluate per-turbine control setpoints and operational mode for a given ambient condition.

    Supports both time series mode (with turbine state tracking) and distribution mode
    (stateless logic).

    Parameters
    ----------
    turbine_ids : list of int
        List of turbine IDs.
    control_fn : callable
        Control function returning yaw, power, and shutdown request per turbine.
    prev_states : dict of int → str
        Previous operational state for each turbine (e.g., 'normal', 'shutdown', 'idle').
        Only used if track_state=True.
    wsp, TI, wdir : float
        Ambient wind speed, turbulence intensity, and wind direction.
    ts : datetime or None
        Timestamp of the current timestep. Can be None in distribution mode.
    extra_inputs : dict
        Additional inputs like 'price', 'temperature', etc., used in rule-based control.
    wsp_min, wsp_max : float
        Cut-in and cut-out wind speed thresholds.
    track_state : bool
        If True, uses state machine logic (time series mode). If False, uses stateless logic (distribution mode).

    Returns
    -------
    yaw_array : ndarray of shape (n_turbines, 1)
        Yaw setpoints for each turbine.
    power_array : ndarray of shape (n_turbines, 1)
        Power setpoints for each turbine.
    tilt_array : ndarray of shape (n_turbines, 1)
        Currently zero for all turbines; placeholder for future tilt control.
    op_mode_array : ndarray of shape (n_turbines, 1)
        Operational mode per turbine ('normal', 'shutdown', etc.).
    operating_array : ndarray of shape (n_turbines, 1)
        Binary flag: 1 if turbine is in normal operation, 0 otherwise.
    new_states : dict of int → str
        Updated operational states to carry into the next timestep (used only in time series mode).
    """
    yaw_list = []
    power_list = []
    op_mode_list = []
    new_states = {}

    # Evaluate base control (yaw, power, shutdown flag) for all turbines
    control_dict = control_fn(
        wsp, TI, wdir, ts,
        extra_inputs=extra_inputs
    )

    for tid in turbine_ids:
        control = control_dict.get(tid, {})
        yaw = control.get("yaw", 0.0)
        power = control.get("power", 100.0)
        shutdown = control.get("control_requested_shutdown", False)

        # Ambient and control-based feasibility
        wind_ok = (wsp >= wsp_min) and (wsp <= wsp_max)
        control_ok = (power > 0.0) and not shutdown

        if track_state:
            prev_state = prev_states.get(tid, "normal")

            if not wind_ok or not control_ok:
                if prev_state == "normal" or prev_state == "startup":
                    op_mode = "shutdown" 
                else:
                    op_mode = "idle"
            elif prev_state in ["shutdown", "idle"] and wind_ok and control_ok:
                op_mode = "startup"
            else:
                op_mode = "normal"
        else:
            # Stateless evaluation (distribution mode)
            op_mode = "normal" if wind_ok and control_ok else "shutdown"

        new_states[tid] = op_mode
        yaw_list.append(yaw)
        power_list.append(power)
        op_mode_list.append(op_mode)

    # Format outputs as column vectors
    yaw_array = np.array(yaw_list, dtype=np.float64)[:, np.newaxis]
    power_array = np.array(power_list, dtype=np.float64)[:, np.newaxis]
    tilt_array = np.zeros_like(yaw_array)  # currently unused
    op_mode_array = np.array(op_mode_list, dtype=str)[:, np.newaxis]
    operating_array = (op_mode_array == "normal").astype(int)

    return yaw_array, power_array, tilt_array, op_mode_array, operating_array, new_states



def simulate_distribution(distribution,
                          farm,
                          surrogate,
                          control_fn,
                          channel_processing_specs,
                          lifetime_years=20,
                          wsp_min=3.0,
                          wsp_max=25.0,
                          supports_operating_modes=False,
                          use_sector_average=False,
                          use_prices=False,
                          price_type=None,
                          fixed_price_value=None,
                          return_bin_outputs=False):
    """
    Simulate wind farm performance over a 3D joint distribution of wind conditions.

    Parameters
    ----------
    distribution : pd.DataFrame
        Joint distribution with columns ['wsp', 'TI', 'wdir', 'probability'].
        Optionally includes 'price' if use_prices=True and price_type='DA_spot'.
    farm : dict
        Output from initialize_pywake_farm(). Must contain layout, flow_model, etc.
    surrogate : dict
        Loaded surrogate models per output channel.
    control_fn : callable
        Function: (wsp, TI, wdir, tid, ts, surrogate_output, extra_inputs) -> dict
        Provides 'yaw', 'power', and 'control_requested_shutdown'.
    channel_processing_specs : dict
        Dict mapping each output channel to its processing type: sum, damage, harmonic.
    total_years : int
        Duration of simulation in years (used to scale outputs).
    wsp_min, wsp_max : float
        Cut-in / cut-out thresholds.
    supports_operating_modes : bool
        If True, surrogate requires 'op_mode'.
    use_sector_average : bool
        If True, extract sector-averaged inflow quantities.
    use_prices : bool
        If True, pricing is enabled per bin.
    price_type : str or None
        "DA_spot" or "fixed_PPA".
    fixed_price_value : float
        Used only if price_type is "fixed_PPA".
    return_bin_outputs : bool
        If True, return a DataFrame with per-bin raw surrogate outputs per turbine.

    Returns
    -------
    result_df : pd.DataFrame
        Per-turbine aggregated (lifetime-scaled) results.
    bin_outputs : pd.DataFrame or None
        Per-bin raw surrogate outputs per turbine (pivoted format), or None if not requested.
    """

    layout = farm["layout_df"]
    flow_model = farm["flow_model"]
    flow_type = farm["flow_type"]
    x, y = farm["x"], farm["y"]

    turbine_ids = layout["id"].tolist()
    n_turbs = len(turbine_ids)   

    wsp = distribution["wsp"].values
    TI =distribution["TI"].values
    wdir = distribution["wdir"].values
    prob = distribution["probability"].values
    n_bins = len(distribution)

    # Build auxiliary input dictionary. Add more in extra_inputs as needed
    extra_inputs = {}
    if use_prices:
        if price_type == "DA_spot":
            price_series = distribution["price"].values
        elif price_type == "fixed_PPA":
            price_series = np.full(n_bins, fixed_price_value)
        else:
            raise ValueError(f"Unknown price_type: {price_type}")
        extra_inputs["price"] = price_series
    else:
        price_series = None

    # --- 1. Compute control inputs and operating flags ---
    yaw_mat = np.zeros((n_turbs, n_bins))
    power_mat = np.zeros_like(yaw_mat)
    tilt_mat = np.zeros_like(yaw_mat)
    op_mode_mat = np.empty_like(yaw_mat, dtype='<U8')
    operating_mat = np.zeros_like(yaw_mat, dtype=int)

    prev_states = {tid: "normal" for tid in turbine_ids} # dummy not used here
    for i in range(n_bins):
        yaw, power, tilt, op_mode, op_flag, new_states = get_setpoint_opmode(
            turbine_ids=turbine_ids,
            control_fn=control_fn,
            prev_states=prev_states,
            wsp=wsp[i],
            TI=TI[i],
            wdir=wdir[i],
            ts=None,
            extra_inputs=extra_inputs,
            wsp_min=wsp_min,
            wsp_max=wsp_max,
            track_state=False
        )
        yaw_mat[:, i] = yaw.flatten()
        power_mat[:, i] = power.flatten()
        tilt_mat[:, i] = tilt.flatten()
        op_mode_mat[:, i] = op_mode.flatten()
        operating_mat[:, i] = op_flag.flatten()

    # --- 2. Flow model call ---
    sim_res, sa, inflow_df, power_df = compute_effective_inflow(
        wsp=wsp,
        TI=TI,
        wdir=wdir,
        x=x,
        y=y,
        yaw=yaw_mat, 
        tilt=tilt_mat,
        operating=operating_mat,
        flow_model=flow_model,
        flow_type=flow_type,
        use_sector_average=use_sector_average,
        dtype=np.float64
    )
    

      # --- 2b. Enforce wind speed limits ---
    if use_sector_average:
        ws_vals = sa["WS"].mean(axis=2)
    else:
        # ws_vals = sim_res.WS_eff.values[:, 0]
        ws_vals = np.asarray(sim_res.WS_eff.values)


    # Apply WS range mask
    # ws_invalid_turbs = (ws_vals < wsp_min) | (ws_vals > wsp_max)
    # operating_mat[ws_invalid_turbs] = 0

    # --- 2c. Clamp TI range ---
    if use_sector_average:
        TI_vals = sa["TI"]
        sa["TI"] = np.clip(TI_vals, 0.045, 0.349)  # returns a new array and replaces the dict value
    else:
        TI_vals = np.asarray(sim_res.TI_eff.values)
        sim_res["TI_eff"].values = np.clip(TI_vals, 0.045, 0.349)  # update in-place

    # --- 3. Surrogate call ---
    input_dict = extract_surrogate_inputs(
        sim_res=sim_res, sa=sa,
        yaw=yaw_mat,
        tilt=tilt_mat,
        power_demand=power_mat,
        operating=operating_mat,
        controller_mode=op_mode_mat,
        timestamps=None
    )

    out_sur = predict_surrogate_outputs(
        surrogates=surrogate,
        surrogate_inputs=input_dict,
        channels=list(channel_processing_specs.keys()),
        use_sector_average=use_sector_average,
        supports_operating_modes=supports_operating_modes,
        dtype=np.float64,
        wsp_min=wsp_min,
        wsp_max=wsp_max,
    )  
    
    # --- 4. Energy and Revenue ---
    raw_energy = power_df.values.T / 6.0  # shape (n_turb, n_bins)
    revenue = raw_energy * price_series[np.newaxis, :] if use_prices else None

    # --- 5. Scale to total lifetime samples ---
    n_samples = prob * (lifetime_years * 52560)  #  10-min samples over lifetime

    # --- 6. Aggregate results ---
    agg = [] # for aggreagation
    bin_rows = [] # for per-bin results

    for j, tid in enumerate(turbine_ids):
        out = {"id": tid}
        for ch, vals in out_sur.items():
            spec = channel_processing_specs[ch]
            x = vals[j, :] if vals.ndim == 2 else vals[j]
            if spec["type"] == "damage":
                damage = spec["nref"] * (x ** spec["wohler"])
                out[ch] = np.sum(damage * n_samples)
            elif spec["type"] == "harmonic":
                out[ch] = float(len(x) / np.sum(1.0 / np.maximum(x, 1e-6)))
            else:  # sum
                out[ch] = np.sum(x * n_samples)

        out["Energy"] = np.sum(raw_energy[j] * n_samples)
        if use_prices:
            out["Revenue"] = np.sum(revenue[j] * n_samples)
        agg.append(out)

        # Build per-bin outputs ONCE (all turbines & all channels), after agg loop
    if return_bin_outputs:
        chan_list = list(out_sur.keys())
        # raw arrays: (n_turbs, n_bins)
        energy_arr  = raw_energy
        revenue_arr = revenue if use_prices else None

        for i in range(n_bins):
            row = {
                "wsp": float(wsp[i]),
                "TI": float(TI[i]),
                "wdir": float(wdir[i]),
                "probability": float(prob[i]),
            }

            # Energy / (optional) Revenue for ALL turbines at this bin
            row.update({("Energy", tid): float(energy_arr[k, i])
                        for k, tid in enumerate(turbine_ids)})
            if use_prices:
                row.update({("Revenue", tid): float(revenue_arr[k, i])
                            for k, tid in enumerate(turbine_ids)})

            # Surrogate channels for ALL turbines at this bin
            for ch in chan_list:
                vals_i = out_sur[ch][:, i] if out_sur[ch].ndim == 2 else out_sur[ch]
                row.update({(ch, tid): float(vals_i[k])
                            for k, tid in enumerate(turbine_ids)})

            bin_rows.append(row)

        bin_df = pd.DataFrame(bin_rows)
    else:
        bin_df = None


    result_df = pd.DataFrame(agg)
 

    return result_df, bin_df

def simulate_distribution_single_turbine(
    distribution,
    surrogate,
    farm,
    channel_processing_specs,
    lifetime_years=25,    
    use_sector_average=False,
    wsp_min=3,
    wsp_max=25
):
    """
    Simulate a single turbine using an IEC-style joint distribution.
    
    Parameters
    ----------
    distribution : DataFrame
        Columns: ['wsp', 'TI', 'probability'].
    surrogate : dict
        Surrogate models per output channel.
    farm : dict
        Dictionary from initialize_single_turbine_farm_IEA22().
    lifetime_years : float
        Number of operational years.
    channel_processing_specs : dict
        Processing rules per channel: {"type": "sum"/"damage"/"harmonic", ...}
    use_sector_average : bool
        Whether to use sector-averaged inflow.
    wsp_min, wsp_max : float
        Wind speed cut-in and cut-out thresholds.

    Returns
    -------
    result_df : pd.DataFrame
        One-row DataFrame with lifetime-aggregated metrics.
    """

    # --- Setup ---
    layout = farm["layout_df"]
    flow_model = farm["flow_model"]
    flow_type = farm["flow_type"]
    x, y = farm["x"], farm["y"]
    turbine_id = layout["id"].iloc[0]
    
    wsp = distribution["wsp"].values
    TI = distribution["TI"].values
    prob = distribution["probability"].values
    wdir = np.full_like(wsp, 90.0)  # constant inflow direction

    n_bins = len(distribution)
    n_samples = prob * lifetime_years * 52560  # 10-min samples per bin

    # --- 1. Control setpoints (greedy) ---
    yaw_mat = np.zeros((1, n_bins))            # yaw = 0
    tilt_mat = np.zeros_like(yaw_mat)          # tilt = 0
    power_mat = np.full_like(yaw_mat, 100.0)   # power = 100
    op_mode_mat = np.full_like(yaw_mat, "normal", dtype="<U8")
    operating_mat = np.ones_like(yaw_mat, dtype=int)

    # --- 2. Flow model call  ---
    sim_res, sa, inflow_df, power_df = compute_effective_inflow(
        wsp=wsp,
        TI=TI,
        wdir=wdir,
        x=x,
        y=y,
        yaw=yaw_mat,
        tilt=tilt_mat,
        operating=operating_mat,
        flow_model=flow_model,
        flow_type=flow_type,
        use_sector_average=use_sector_average,
        dtype=np.float64
    )
    
        # --- 2b. Apply cut-in/cut-out wind speed limits (stateless IEC) ---
    if use_sector_average:
        # sa["WS"] has shape (wt, time, dir); take the sector-mean per time
        ws_vals = sa["WS"].mean(axis=2)[0, :]              # (n_bins,)
    else:
        # sim_res.WS_eff is (wt, time)
        ws_vals = np.asarray(sim_res.WS_eff.values)[0, :]  # (n_bins,)

    invalid = (ws_vals < wsp_min) | (ws_vals > wsp_max)

    # if np.any(invalid):
    #     # mark bins as shutdown for the surrogate routing
    #     op_mode_mat[0, invalid] = "shutdown"
    #     # and mark them as non-operating for any downstream logic that reads this flag
    #     operating_mat[0, invalid] = 0


    # --- 2c. Clamp turbulence intensity ---
    if use_sector_average:
        sa["TI"] = np.clip(sa["TI"], 0.045, 0.349)
    else:
        sim_res["TI_eff"].values = np.clip(sim_res["TI_eff"].values, 0.045, 0.349)


# --- 3. Surrogate input preparation ---
    input_dict = extract_surrogate_inputs(
        sim_res=sim_res,
        sa=sa,
        yaw=yaw_mat,
        tilt=tilt_mat,
        power_demand=power_mat,
        operating=operating_mat,
        controller_mode=op_mode_mat,
        timestamps=None
    )

    out_sur = predict_surrogate_outputs(
        surrogates=surrogate,
        surrogate_inputs=input_dict,
        channels=list(channel_processing_specs.keys()),
        use_sector_average=use_sector_average,
        supports_operating_modes=False,
        dtype=np.float64,
        wsp_min=wsp_min,
        wsp_max=wsp_max,
    )

        # --- 4. Energy output ---
    raw_energy = power_df.values.T[0] / 6.0  # (n_bins,)
    if 'invalid' in locals():
        raw_energy[invalid] = 0.0


    # --- 5. Aggregate results ---
    final_outputs = {}

    for ch, vals in out_sur.items():
        vals = vals[0, :] if vals.ndim == 2 else vals
        spec = channel_processing_specs[ch]

        if spec["type"] == "damage":
            damage = spec["nref"] * (vals ** spec["wohler"])
            final_outputs[ch] = float(np.sum(damage * n_samples))

        elif spec["type"] == "harmonic":
            weights = n_samples
            inv_vals = 1.0 / np.maximum(vals, 1e-6)
            final_outputs[ch] = float(np.sum(weights) / np.sum(weights * inv_vals))

        else:  # sum-type
            final_outputs[ch] = float(np.sum(vals * n_samples))

    final_outputs["Energy"] = float(np.sum(raw_energy * n_samples))
    final_outputs["id"] = turbine_id

    result_df = pd.DataFrame([final_outputs])
    return result_df


def extract_surrogate_inputs(
    sim_res: xr.Dataset,
    sa: Optional[xr.DataArray],
    yaw: np.ndarray,
    tilt: np.ndarray,
    power_demand: np.ndarray,
    operating: np.ndarray,
    controller_mode: np.ndarray,
    timestamps: Optional[np.ndarray] = None,
) -> dict:
    """
    Extract surrogate model inputs from PyWake simulation results and sector-averaged inflow data.

    Supports both time series mode (with timestamps) and distribution mode (no timestamps).

    Parameters
    ----------
    sim_res : xr.Dataset
        Simulation result from PyWake (rotor-averaged outputs).
    sa : xr.DataArray or None
        Sector-averaged flow output from `compute_sector_average`, or None if not used.
    yaw : ndarray, shape (n_turbines, n_times)
        Yaw control input per turbine and timestep/bin.
    tilt : ndarray, shape (n_turbines, n_times)
        Tilt control input per turbine and timestep/bin.
    power_demand : ndarray, shape (n_turbines, n_times)
        Power demand input per turbine and timestep/bin.
    operating : ndarray, shape (n_turbines, n_times)
        Turbine operational state (0 = off, 1 = normal) for flow model.
    controller_mode : ndarray, shape (n_turbines, n_times)
        Mode name per turbine and timestep/bin (e.g., "normal", "idling", etc.).
    timestamps : ndarray or None
        Timestamps to use (e.g., pd.Timestamp array), or None to use np.arange(n_times).

    Returns
    -------
    dict
        Dictionary with keys:
        - "time": 1D array of timestamps or range
        - "RA": dict of rotor-averaged inputs (or None if missing)
        - "SA": dict of sector-averaged inputs (or None if missing)
        - "operating": operational state array
        - "controller_mode": control mode string array
    """
    n_turbines = yaw.shape[0]
    n_times = yaw.shape[1]

    # === Time index ===
    time_arr = timestamps if timestamps is not None else np.arange(n_times)

    # === Rotor-averaged inflow ===
    RA = None
    if sim_res is not None:
        RA = {
            "WS": sim_res["WS_eff"].values.astype(np.float64),        # (n_turbines, n_times)
            "TI": sim_res["TI_eff"].values.astype(np.float64),
            "yaw": yaw,
            "tilt": tilt,
            "power": power_demand,
        }

    # === Sector-averaged inflow ===
    SA = None
    if sa is not None:
        ws = sa.sel(quantity="WS_eff").transpose("wt", "time", "direction").values.astype(np.float64)
        ti = sa.sel(quantity="TI_eff").transpose("wt", "time", "direction").values.astype(np.float64)

        SA = {
            "WS": ws,  # (n_turbines, n_times, 4)
            "TI": ti,
            "yaw": yaw,
            "tilt": tilt,
            "power": power_demand,
        }

    return {
        "time": time_arr,                   # (n_times,)
        "RA": RA,
        "SA": SA,
        "operating": operating,            # (n_turbines, n_times)
        "controller_mode": controller_mode # (n_turbines, n_times)
    }

def predict_surrogate_outputs(
    surrogates,
    surrogate_inputs,
    channels,
    use_sector_average = False,
    supports_operating_modes = False,
    dtype=np.float64,
    wsp_min=3,
    wsp_max=25,
) :
    """
    Predicts turbine-level outputs from surrogate models given inflow and control inputs.

    Supports rotor-averaged (RA) and sector-averaged (SA) modes, and can use multiple
    surrogates per operational mode if enabled.

    Parameters
    ----------
    surrogates : dict
        - If supports_operating_modes=False: channel → TensorFlowModel
        - If supports_operating_modes=True: mode → (channel → TensorFlowModel)
    surrogate_inputs : dict
        Output from extract_surrogate_inputs(). Contains "RA", "SA", "controller_mode", etc.
    channels : list of str
        Names of output channels (e.g. ["tbfa", "tbss", ...]).
    use_sector_average : bool
        If True, uses sector-averaged surrogate models.
    supports_operating_modes : bool
        If True, uses different surrogates for different operational modes.
    dtype : np.dtype
        Output data type (default: np.float64)

    Returns
    -------
    output : dict of str → np.ndarray
        Dictionary of output arrays, each shaped (n_turbines, n_steps)
    """
    modes = surrogate_inputs["controller_mode"]              # (n_turbs, n_times)
    operating = surrogate_inputs["operating"]                # (n_turbs, n_times)
    n_turbs, n_times = operating.shape

    # === Construct surrogate-compatible inflow structures ===
    wsp_dummy = np.full(n_times, 8.0)  # placeholder, not used by surrogate directly
    wdir_dummy = np.full(n_times, 208)

    flow_input_RA, flow_input_SA = build_surrogate_inflow_structures(
        RA=surrogate_inputs["RA"],
        SA=surrogate_inputs["SA"],
        wsp=wsp_dummy,
        wdir=wdir_dummy,
    )


    # === Preallocate outputs for each channel ===
    output = {ch: np.zeros((n_turbs, n_times), dtype=dtype) for ch in channels}

    # Compute WS array
    if use_sector_average:
        ws_vals = surrogate_inputs["SA"]["WS"].mean(axis=2)  # (n_turbs, n_times)
    else:
        ws_vals = surrogate_inputs["RA"]["WS"]

    
    # # --- DEBUG: check what we are about to feed the surrogate (RA only) ---
    # DEBUG_DOMAIN_CHECK = True  # set False to silence

    # if DEBUG_DOMAIN_CHECK:
    #     # pick any one channel's model (domains should match across channels)
    #     if supports_operating_modes:
    #         _any_mode = next(iter(surrogates))
    #         _any_model = next(iter(surrogates[_any_mode].values()))
    #     else:
    #         _any_model = next(iter(surrogates.values()))

    #     # Flatten RA inputs into (N,4) = [WS, TI%, yaw, power]
    #     WS_flat   = surrogate_inputs["RA"]["WS"].reshape(-1)
    #     TI_flat   = (surrogate_inputs["RA"]["TI"].reshape(-1) * 100.0)  # TI must be in %
    #     yaw_flat  = surrogate_inputs["RA"]["yaw"].reshape(-1)
    #     pow_flat  = surrogate_inputs["RA"]["power"].reshape(-1)
    #     X_flat = np.stack([WS_flat, TI_flat, yaw_flat, pow_flat], axis=1)

    #     # Model's domain check
    #     in_dom = _any_model.domain.in_domain(X_flat)  # boolean shape (N,)
    #     n_total = in_dom.size
    #     n_ood = n_total - int(in_dom.sum())
    #     frac_ood = n_ood / n_total

    #     print(f"[DEBUG] RA surrogate domain: {frac_ood:.1%} outside ({n_ood}/{n_total})")

    #     # Map back to (turbine, time) indices & collect offending samples
    #     ood_mask_2d = (~in_dom).reshape(n_turbs, n_times)
    #     ti_pct_2d   = TI_flat.reshape(n_turbs, n_times)

    #     # Show a few examples (with op_mode and operating flag)
    #     _ii, _jj = np.where(ood_mask_2d)
    #     for k in range(min(10, _ii.size)):
    #         i, j = int(_ii[k]), int(_jj[k])
    #         print(
    #             f"  OOD @ turb {i}, t {j}: "
    #             f"WS={ws_vals[i,j]:.3f}, TI%={ti_pct_2d[i,j]:.2f}, "
    #             f"yaw={surrogate_inputs['RA']['yaw'][i,j]:.2f}, "
    #             f"power={surrogate_inputs['RA']['power'][i,j]:.2f}, "
    #             f"op_mode={surrogate_inputs['controller_mode'][i,j]}, "
    #             f"operating={surrogate_inputs['operating'][i,j]}"
    #         )

    #     # Export all offending points for your own inspection
    #     domain_debug = {
    #         "turbine_idx": _ii,
    #         "time_idx": _jj,
    #         "WS": ws_vals[ood_mask_2d],
    #         "TI_pct": ti_pct_2d[ood_mask_2d],
    #         "yaw": surrogate_inputs["RA"]["yaw"][ood_mask_2d],
    #         "power": surrogate_inputs["RA"]["power"][ood_mask_2d],
    #         "op_mode": surrogate_inputs["controller_mode"][ood_mask_2d],
    #         "operating": surrogate_inputs["operating"][ood_mask_2d],
    #     }
    # # --- END DEBUG ---





    # Mask: out-of-range wind speeds
    invalid_ws = (ws_vals < wsp_min) | (ws_vals > wsp_max)

    # Create a mode copy for routing
    masked_modes = np.copy(surrogate_inputs["controller_mode"])

    # For both single and multi-mode logic, replace "normal" with "idling" where WS invalid
    masked_modes[(masked_modes == "normal") & invalid_ws] = "idling"

    if supports_operating_modes:
        # Loop over all modes (e.g., "normal", "startup", ...)
        unique_modes = np.unique(modes)
        for mode in unique_modes:
            if mode not in surrogates:
                continue  # No model defined for this mode

            mask = (modes == mode)
            if not np.any(mask):
                continue

            # Call surrogate for this mode
            if use_sector_average:
                out = predict_loads_sector_average(
                    surrogates[mode],
                    flow_input_SA,
                    surrogate_inputs["SA"]["yaw"],
                    surrogate_inputs["SA"]["power"],
                    ti_in_percent=True
                )
            else:
                out = predict_loads_rotor_average(
                    surrogates[mode],
                    flow_input_RA,
                    surrogate_inputs["RA"]["yaw"],
                    surrogate_inputs["RA"]["power"],
                    dtype=dtype,
                    ti_in_percent=True
                )

            # Assign only masked entries
            for ch in channels:
                output[ch][mask] = out[ch][mask]

    else:
        # Single surrogate for all modes; only 'normal' is meaningful
        normal_mask = (masked_modes == "normal")
        other_mask = ~normal_mask

        if np.any(normal_mask):
            if use_sector_average:
                out = predict_loads_sector_average(
                    surrogates,
                    flow_input_SA,
                    surrogate_inputs["SA"]["yaw"],
                    surrogate_inputs["SA"]["power"],
                    ti_in_percent=True
                )
            else:
                out = predict_loads_rotor_average(
                    surrogates,
                    flow_input_RA,
                    surrogate_inputs["RA"]["yaw"],
                    surrogate_inputs["RA"]["power"],
                    ti_in_percent=True,
                    # dtype=dtype
                )        

            for ch in channels:
               
                if ch not in out.coords["name"].values:
                    raise KeyError(f"Channel '{ch}' not found in surrogate output.")
                arr = out.sel(name=ch)    
                arr = arr.values  # Convert to NumPy for fancy masking

                output[ch][normal_mask] = arr[normal_mask]
                output[ch][other_mask] = 0.0 # Explicitly zero-out other modes
  
    

    return output

def build_surrogate_inflow_structures(RA, SA, wsp, wdir):
    """
    Constructs xarray structures for rotor-averaged and sector-averaged inflow,
    matching the format expected by predict_loads_* functions.

    Parameters
    ----------
    RA : dict
        Extracted rotor-averaged inputs (keys: WS, TI, yaw, power).
    SA : dict or None
        Extracted sector-averaged inputs (keys: WS, TI, yaw, power).
    wsp : ndarray, shape (n_times,)
        Free-stream wind speed.
    wdir : ndarray, shape (n_times,)
        Free-stream wind direction.

    Returns
    -------
    sim_res_like : xarray.Dataset
    sector_avg_like : xarray.DataArray or None
    """
    n_turbs, n_times = RA["WS"].shape

    sim_res_like = xr.Dataset(
        {
            "WS_eff": (("wt", "time"), RA["WS"]),
            "TI_eff": (("wt", "time"), RA["TI"]),
        },
        coords={
            "wt": np.arange(n_turbs),
            "time": np.arange(n_times),
            "ws": ("time", wsp),
            "wd": ("time", wdir),
        }
    )

    sector_avg_like = None
    if SA is not None:
        ws = SA["WS"]  # (n_turbs, n_times, 4)
        ti = SA["TI"]
        sector_avg_like = xr.DataArray(
            np.stack([ws, ti], axis=2),  # (wt, time, quantity=2, direction=4)
            dims=("wt", "time", "quantity", "direction"),
            coords={
                "quantity": ["WS_eff", "TI_eff"],
                "direction": ["up", "right", "down", "left"],
                "wt": np.arange(n_turbs),
                "time": np.arange(n_times),
                "ws": ("time", wsp),
                "wd": ("time", wdir),
            }
        )

    return sim_res_like, sector_avg_like

