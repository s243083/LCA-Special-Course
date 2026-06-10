import pandas as pd
import numpy as np
from pathlib import Path
from core.File_Handling import load_yaml, process_duration_fields, loadcsv
from core.utils import repeat_timeseries, gap_fill_timeseries, get_input_parameter, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration

# Stochastic market model imports: 

from core.market_model_util import (
    MONTHS,
    _load_json_strict,
    _require,
    _require_float,
    _require_bool,
    _clip01,
    _scale_prob_logit,
    _load_hour_month_profile,
    _seasonality_from_profile,
    _simulate_seasonal_ar1_cf,
    _load_capacity_series,
    _parse_mean_model,
    _simulate_mu_ar1_period,
    _simulate_bounded_ou_drift,
    _load_shape_profile_required,
    _load_jump_prob_profile_required,
    _lookup_hour_month_prob,
    _parse_residual_fast_v5,
    _adjust_transition_matrix_stress_p11,
    _simulate_fast_components_conditional_jumps,
    _map_cf_sys_to_cf_asset,
    diagnose_and_plot_price_series,

)

class MarketEnv:

    def __init__(self,env):
        self.env = env
        self.config = env.config
        self.market_inputs = load_marketInput(self.env.config)
        self.market_inputs = get_input_parameter(self.market_inputs, 'MA')

        self.market_type = get_input_parameter(self.market_inputs, 'Market', 'mode')  

        self.el_price_records = pd.DataFrame()
        self.asset_cf_proxy_records = pd.DataFrame

        

    def create_electricityprice(self):
        market_type = self.market_type

        if market_type == 'external':
            self.load_external_price()
        elif market_type == 'fixed':
            self.create_electricityprice_fixed()
        elif market_type == 'stochastic_market_model':
            self.create_electricityprice_stochastic_market_model()
        else:
            raise ValueError(f"Market type '{market_type}' not recognized.")

    def _resolve_model_path(self, p: str | Path) -> Path:
        """
        Resolve input artifact paths from YAML.
        Strategy:
          - if absolute: use as-is
          - if relative: resolve against cfg.valuewind_inputFolder if present, else CWD
        """
        cfg = self.env.config
        p = Path(str(p))
        if p.is_absolute():
            return p
        base = None
        if hasattr(cfg, "valuewind_inputFolder") and cfg.valuewind_inputFolder:
            base = Path(str(cfg.valuewind_inputFolder))
        else:
            base = Path.cwd()
        return (base / p).resolve()

    def _build_operation_timestamps(self, freq: str, tz: str) -> pd.DatetimeIndex:
        """
        Build timestamps covering WF operation window, and localize to tz.
        Output is tz-aware.
        """
        cfg = self.env.config

        start_h = float(cfg.WF_OperationsStart_h)
        end_h   = float(cfg.WF_OperationsEnd_h)

        start_ts = pd.to_datetime(cfg.Project_StartDate, format="%d.%m.%Y")
        op_start_ts = start_ts + pd.to_timedelta(start_h, unit="h")
        op_end_ts   = start_ts + pd.to_timedelta(end_h, unit="h")

        # naive range first
        ts = pd.date_range(start=op_start_ts, end=op_end_ts, freq=freq, inclusive="left")

        tz = (tz or "UTC").strip()
        # localize naive -> tz-aware; do not convert unless you explicitly want UTC internally
        return ts.tz_localize(tz)

    
    def create_electricityprice_fixed(self): 
        """
        Create a fixed electricity price time series from operations start to operations end.
        The resulting DataFrame is stored in self.el_price_records and has the same format
        as the one created in create_electricityprice_fromEnv, i.e. columns:
            - 'timestamp'
            - 'price'
        """

        cfg = self.env.config

        # resolution directly from input, e.g. "10min", "1h", "1d"
        freq = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'fixed', 'resolution')
        self.resolution = freq

        # Operation start / end in hours (relative)
        start_h = float(cfg.WF_OperationsStart_h)
        end_h   = float(cfg.WF_OperationsEnd_h)

        # Project start TS
        start_ts = pd.to_datetime(cfg.Project_StartDate, format="%d.%m.%Y")

        op_start_ts = start_ts + pd.to_timedelta(start_h, unit="h")
        op_end_ts   = start_ts + pd.to_timedelta(end_h, unit="h")

        # Build timestamps
        timestamps = pd.date_range(
            start=op_start_ts,
            end=op_end_ts,
            freq=freq,
            inclusive="left"
        )

        # Fixed price
        price = float(
            get_input_parameter(
                self.market_inputs,
                'Market', 'timeseries', 'fixed', 'price'
            )
        )

        self.el_price_records = pd.DataFrame({
            "timestamp": timestamps,
            "price": price,
        })

    def load_external_price(self):
        """
        Load external electricity price time series.

        Expects a CSV with at least:
        - 'timestamp'
        - 'price' (by default, or whatever is configured in expected_columns)

        YAML config (example):

        Market:
        timeseries:
            external:
            file: "../examples/Inputs/Market/price_timeseries.csv"
            resolution: "1h"         # or "10min", "1d"
            expected_columns:
                - "price"              # can be extended, e.g. ["price", "price_EUR"]
            target_duration: "8760h" # or whatever format your helper expects
        """

        # --- Load configuration ---
        path = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'file')
        self.resolution = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'resolution')
        resolution = self.resolution
        expected_cols = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'expected_columns')
        duration = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'external', 'target_duration')

        # Default required data columns (besides timestamp)
        if expected_cols is None:
            expected_cols = {"price"}
        else:
            expected_cols = set(expected_cols)

        # We always require timestamp as well
        required_cols = {"timestamp"} | expected_cols

        # --- Load CSV ---
        base_df = pd.read_csv(path)

        # --- Check required columns ---
        missing = required_cols - set(base_df.columns)
        if missing:
            raise ValueError(f"Price timeseries is missing required columns: {missing}")

        # --- Keep only timestamp + expected columns (ordered) ---
        ordered_cols = ["timestamp"] + [c for c in expected_cols if c != "timestamp"]
        base_df = base_df[ordered_cols]

        # --- Convert timestamp to datetime ---
        base_df["timestamp"] = pd.to_datetime(base_df["timestamp"])

        # --- Check time resolution ---
        actual_freq = pd.infer_freq(base_df["timestamp"])
        if actual_freq is None:
            raise ValueError("Could not infer frequency of timestamps in price data.")

        # Map pandas frequency strings to your resolution convention
        freq_map = {"h": "1h", "10T": "10min", "60T": "1h"}
        normalized = freq_map.get(actual_freq, actual_freq)
        normalized = normalized.lower()

        if resolution is not None and normalized != resolution.lower():
            raise ValueError(
                f"Price timeseries resolution mismatch: expected {resolution}, got {normalized}"
            )

        # --- 1) Remove gaps and rebuild timestamps to be continuous ---
        base_df = remove_gaps_rebuild_timestamps(
            base_df,
            timestamp_col="timestamp",
            freq=None,  # infer from data
            sort=True,
        )

        # --- 2) Repeat until the desired total duration is reached ---
        if duration is not None:
            base_df = repeat_timeseries_to_duration(
                base_df,
                duration=duration,
                timestamp_col="timestamp",
                trim_to_duration=True,
            )

        # Store result
        self.el_price_records = base_df

    def create_electricityprice_stochastic_market_model(self):
        """
        Single-path artifact-driven stochastic market model.

        YAML block: Market.timeseries.stochastic_market_model
          - inputs: paths to merit/residual/mean json + wind/solar profiles + capacity csv + feedin json
          - knobs: alpha_*, multipliers, stress_p11_mult, drift.*
          - outputs: store_* flags, write_debug_files
        """
        model_cfg = get_input_parameter(self.market_inputs, 'Market', 'timeseries', 'stochastic_market_model')
        if model_cfg is None:
            raise ValueError("Missing Market.timeseries.stochastic_market_model config block.")

        # --- core settings ---
        freq = get_input_parameter(model_cfg, 'resolution') or "1h"
        tz = get_input_parameter(model_cfg, 'timezone') or "UTC"
        seed = get_input_parameter(model_cfg, 'seed')
        rng = np.random.default_rng(int(seed)) if seed is not None else np.random.default_rng()

        self.resolution = freq

        # --- timestamps (tz-aware) ---
        timestamps = self._build_operation_timestamps(freq=freq, tz=tz)
        n = len(timestamps)

        cfg = self.env.config
        base_path = Path(cfg.valuewind_inputFolder)

        inputs = get_input_parameter(model_cfg, 'inputs') or {}

        def join_with_base(p):
            return (base_path / str(p).lstrip("/")).resolve()

        merit_json = join_with_base(_require(inputs, "merit_json"))
        resid_json = join_with_base(_require(inputs, "residual_json"))
        mean_json  = join_with_base(_require(inputs, "mean_json"))

        wind_prof_csv  = join_with_base(_require(inputs, "wind_profile_csv"))
        solar_prof_csv = join_with_base(_require(inputs, "solar_profile_csv"))

        capacity_csv = join_with_base(_require(inputs, "capacity_csv"))
        feedin_json  = join_with_base(_require(inputs, "feedin_model_json"))

        # --- load calibration artifacts ---
        merit = _load_json_strict(merit_json)
        resid_model = _load_json_strict(resid_json)
        mean_model = _load_json_strict(mean_json)
        feedin_model = _load_json_strict(feedin_json)

        b0 = _require_float(merit, "coefficients.b0")
        b1 = _require_float(merit, "coefficients.b1_wind")
        b2 = _require_float(merit, "coefficients.b2_solar")

        period_mode, a_mu, rho_mu, sig_mu = _parse_mean_model(mean_model)

        wind_prof  = _load_hour_month_profile(wind_prof_csv)
        solar_prof = _load_hour_month_profile(solar_prof_csv)

        wind_phi   = _require_float(feedin_model, "wind.ar1.phi")
        wind_sigma = _require_float(feedin_model, "wind.ar1.sigma")
        solar_phi  = _require_float(feedin_model, "solar.ar1.phi")
        solar_sigma= _require_float(feedin_model, "solar.ar1.sigma")

        # REQUIRED residual shape profile and conditional jump probs
        shape_prof_path = _load_shape_profile_required(resid_model, resid_json)
        shape_prof = _load_hour_month_profile(shape_prof_path)

        jump_prob_profile = _load_jump_prob_profile_required(resid_model)
        jpp_pos_rates = _require(jump_prob_profile, "pos.rates")
        jpp_neg_rates = _require(jump_prob_profile, "neg.rates")

        fast_cfg = _parse_residual_fast_v5(resid_model)

        # --- knobs ---
        knobs = get_input_parameter(model_cfg, 'knobs') or {}

        alpha_shape      = float(get_input_parameter(knobs, 'alpha_shape') or 0.0)
        alpha_mu         = float(get_input_parameter(knobs, 'alpha_mu') or 0.0)
        mu_shift         = float(get_input_parameter(knobs, 'mu_shift') or 0.0)

        alpha_u          = float(get_input_parameter(knobs, 'alpha_u') or 1.0)
        alpha_j_pos      = float(get_input_parameter(knobs, 'alpha_j_pos') or 1.0)
        alpha_j_neg      = float(get_input_parameter(knobs, 'alpha_j_neg') or 1.0)

        alpha_wind_feed  = float(get_input_parameter(knobs, 'alpha_wind_feed') or 1.0)
        alpha_solar_feed = float(get_input_parameter(knobs, 'alpha_solar_feed') or 1.0)

        jump_prob_mult_pos = float(get_input_parameter(knobs, 'jump_prob_mult_pos') or 1.0)
        jump_prob_mult_neg = float(get_input_parameter(knobs, 'jump_prob_mult_neg') or 1.0)
        stress_p11_mult    = float(get_input_parameter(knobs, 'stress_p11_mult') or 1.0)

        P_adj = _adjust_transition_matrix_stress_p11(
            fast_cfg["P"], fast_cfg["stress_idx"], stress_p11_mult
        )

        # drift block
        drift_cfg = get_input_parameter(knobs, 'drift') or {}
        drift_enabled   = bool(get_input_parameter(drift_cfg, 'enabled') or False)
        drift_period    = str(get_input_parameter(drift_cfg, 'period_mode') or "year").lower()
        drift_rho       = float(get_input_parameter(drift_cfg, 'rho') or 0.0)
        drift_sigma     = float(get_input_parameter(drift_cfg, 'sigma') or 0.0)
        drift_center    = float(get_input_parameter(drift_cfg, 'center') or 0.0)
        drift_half_rng  = float(get_input_parameter(drift_cfg, 'half_range') or 0.0)
        alpha_drift     = float(get_input_parameter(drift_cfg, 'alpha') or 1.0)

        # outputs
        out_cfg = get_input_parameter(model_cfg, 'outputs') or {}
        store_components = bool(get_input_parameter(out_cfg, 'store_components') or False)
        store_regime     = bool(get_input_parameter(out_cfg, 'store_regime') or False)
        store_feed_in    = bool(get_input_parameter(out_cfg, 'store_feed_in') or False)
        write_debug      = bool(get_input_parameter(out_cfg, 'write_debug_files') or False)
        debug_dir        = get_input_parameter(out_cfg, 'debug_dir')

        # --- capacities ---
        wind_cap  = _load_capacity_series(capacity_csv, timestamps, "wind_capacity_mw")
        solar_cap = _load_capacity_series(capacity_csv, timestamps, "solar_capacity_mw")

        # --- deterministic shape component (hour×month) ---
        shape_component = alpha_shape * _seasonality_from_profile(timestamps, shape_prof)

        # --- simulate system CF and feed-in ---
        wind_cf  = _simulate_seasonal_ar1_cf(rng, timestamps, wind_prof,  wind_phi,  wind_sigma,  0.0, 1.0)
        solar_cf = _simulate_seasonal_ar1_cf(rng, timestamps, solar_prof, solar_phi, solar_sigma, 0.0, 1.0)

        wind_feed  = (wind_cf  * wind_cap)  * alpha_wind_feed
        solar_feed = (solar_cf * solar_cap) * alpha_solar_feed

        # --- merit component ---
        merit_component = (b0 + b1 * wind_feed + b2 * solar_feed)

        # --- mean layer ---
        mu_raw = _simulate_mu_ar1_period(rng, period_mode, timestamps, a_mu, rho_mu, sig_mu)
        mu_component = alpha_mu * mu_raw + mu_shift

        # --- residual with required conditional jumps ---
        base_ar, state, jpos, jneg, _u_fast = _simulate_fast_components_conditional_jumps(
            rng=rng,
            timestamps=timestamps,
            P=P_adj,
            normal_idx=fast_cfg["normal_idx"],
            stress_idx=fast_cfg["stress_idx"],
            normal_params=fast_cfg["normal_params"],
            stress_params=fast_cfg["stress_params"],
            jump_pos=fast_cfg["jump_pos"],
            jump_neg=fast_cfg["jump_neg"],
            jpp_pos_rates=jpp_pos_rates,
            jpp_neg_rates=jpp_neg_rates,
            jump_prob_mult_pos=jump_prob_mult_pos,
            jump_prob_mult_neg=jump_prob_mult_neg,
            start_state=fast_cfg["normal_idx"],
        )

        base_ar_scaled = alpha_u * base_ar
        jpos_scaled    = alpha_j_pos * jpos
        jneg_scaled    = alpha_j_neg * jneg
        u_fast_total   = base_ar_scaled + jpos_scaled - jneg_scaled

        # --- drift component (optional) ---
        if drift_enabled:
            drift_component = alpha_drift * _simulate_bounded_ou_drift(
                rng=rng,
                timestamps_hourly=timestamps,
                period_mode=drift_period,
                rho=drift_rho,
                sigma=drift_sigma,
                center=drift_center,
                half_range=drift_half_rng,
            )
        else:
            drift_component = np.zeros(n, dtype=float)

        # --- final price ---
        price = merit_component + shape_component + mu_component + drift_component + u_fast_total

        # Store timestamps as naive UTC to match typical WINPACT convention
        ts_out = timestamps.tz_convert("UTC").tz_localize(None)

        self.el_price_records = pd.DataFrame({
            "timestamp": ts_out,
            "price": price,
        })


        # Optional asset CF mapping
        # --- optional: system CF -> asset CF proxy mapping (kept from original model) ---
        mapping_cfg = get_input_parameter(model_cfg, 'asset_cf_mapping')

        if mapping_cfg and bool(get_input_parameter(mapping_cfg, 'enabled') or False):
            cf_asset = _map_cf_sys_to_cf_asset(
                cf_sys=wind_cf,          # mapping uses system wind CF as driver (same as original)
                mapping_cfg=mapping_cfg
            )

            self.asset_cf_proxy_records = pd.DataFrame({
                "timestamp": ts_out,
                "cf_asset_proxy": cf_asset
            })

        # Optional diagnostics
        if store_feed_in:
            self.system_feed_in_records = pd.DataFrame({
                "timestamp": ts_out,
                "cf_wind_sys": wind_cf,
                "wind_feed_in": wind_feed,
                "cf_solar_sys": solar_cf,
                "solar_feed_in": solar_feed,
            })

        if store_components:
            self.price_components_records = pd.DataFrame({
                "timestamp": ts_out,
                "merit_component": merit_component,
                "shape_component": shape_component,
                "mu_component": mu_component,
                "drift_component": drift_component,
                "u_base_ar": base_ar_scaled,
                "jump_pos_excess": jpos_scaled,
                "jump_neg_excess": jneg_scaled,
                "u_fast_total": u_fast_total,
            })

        if store_regime:
            self.regime_records = pd.DataFrame({
                "timestamp": ts_out,
                "regime_state": state.astype(int),
                "is_stress": (state == fast_cfg["stress_idx"]).astype(int),
            })

        stats = diagnose_and_plot_price_series(
            self.el_price_records,
            timestamp_col="timestamp",
            price_col="price",
            title="Stochastic market model price", 
            verbose = True,
        )

        rows = []

        # level / acf / spike (simple key-value)
        for section in ("level", "acf", "spike"):
            for metric, value in stats[section].items():
                rows.append({
                    "section": section,
                    "key": None,               # no secondary key
                    "metric": str(metric),
                    "value": float(value) if pd.notna(value) else np.nan,
                })

        # ramps (key = horizon_h)
        ramps_df = stats["ramps"].reset_index()  # horizon_h becomes column
        for _, r in ramps_df.iterrows():
            h = int(r["horizon_h"])
            for metric in ["std", "mad", "q95_abs", "q99_abs"]:
                rows.append({
                    "section": "ramps",
                    "key": h,
                    "metric": metric,
                    "value": float(r[metric]) if pd.notna(r[metric]) else np.nan,
                })

        # negative hours per year (key = year)
        neg_df = stats["negative_hours_per_year"].reset_index()  # year becomes column
        for _, r in neg_df.iterrows():
            y = int(r["year"])
            for metric in ["neg_hours", "hours", "neg_share_pct"]:
                rows.append({
                    "section": "negative_hours_per_year",
                    "key": y,
                    "metric": metric,
                    "value": float(r[metric]) if pd.notna(r[metric]) else np.nan,
                })

        self.market_statistics_records = pd.DataFrame(rows)

        neg_year = stats["negative_hours_per_year"]

        # summary stats for export
        neg_hours_total = float(neg_year["neg_hours"].sum()) if not neg_year.empty else np.nan
        hours_total     = float(neg_year["hours"].sum()) if not neg_year.empty else np.nan
        neg_share = (neg_hours_total / hours_total) if (np.isfinite(neg_hours_total) and np.isfinite(hours_total) and hours_total > 0) else np.nan

        summary = {
            "neg_hours_total": neg_hours_total,
            "hours_total": hours_total,
            "neg_share": float(neg_share) if np.isfinite(neg_share) else np.nan,
            "neg_share_pct": float(100.0 * neg_share) if np.isfinite(neg_share) else np.nan,

            # optional: a few top-level stats you already compute
            "price_mean": float(stats["level"]["mean"]),
            "price_std": float(stats["level"]["std"]) if pd.notna(stats["level"]["std"]) else np.nan,
            "acf_lag1": float(stats["acf"]["acf_lag1"]) if pd.notna(stats["acf"]["acf_lag1"]) else np.nan,
        }

        self.market_statistics_summary_records = pd.DataFrame([summary])


        # Optional debug writing (single path)
        if write_debug:
            if not debug_dir:
                raise ValueError("outputs.write_debug_files=true but outputs.debug_dir is not set.")
            out_base = self._resolve_model_path(debug_dir)
            out_base.mkdir(parents=True, exist_ok=True)

            run_tag = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            out_dir = out_base / f"run_{run_tag}"
            out_dir.mkdir(parents=True, exist_ok=True)

            out_parquet = out_dir / "simulated_path.parquet"
            out_meta    = out_dir / "sim_metadata.json"

            # build a tidy debug table (single path)
            dbg = pd.DataFrame({
                "timestamp_utc": ts_out,
                "price_eur_mwh": price,
                "merit_component": merit_component,
                "shape_component": shape_component,
                "mu_component": mu_component,
                "drift_component": drift_component,
                "u_base_ar": base_ar_scaled,
                "jump_pos_excess": jpos_scaled,
                "jump_neg_excess": jneg_scaled,
                "u_fast_total": u_fast_total,
                "regime_state": state.astype(int),
                "is_stress": (state == fast_cfg["stress_idx"]).astype(int),
                "wind_feed_mw": wind_feed,
                "solar_feed_mw": solar_feed,
            })
            dbg.to_parquet(out_parquet, index=False)

            meta = {
                "run_tag": run_tag,
                "timezone_input": tz,
                "timestamps_saved_as": "naive_UTC",
                "freq": freq,
                "seed": int(seed) if seed is not None else None,
                "out_parquet": str(out_parquet),
                "inputs": {
                    "merit_json": str(merit_json),
                    "residual_json": str(resid_json),
                    "mean_json": str(mean_json),
                    "wind_profile_csv": str(wind_prof_csv),
                    "solar_profile_csv": str(solar_prof_csv),
                    "capacity_csv": str(capacity_csv),
                    "feedin_model_json": str(feedin_json),
                    "shape_profile": str(shape_prof_path),
                    "jump_prob_profile": "embedded_in_residual_model.json",
                },
                "knobs": {
                    "alpha_shape": alpha_shape,
                    "alpha_mu": alpha_mu,
                    "mu_shift": mu_shift,
                    "alpha_u": alpha_u,
                    "alpha_j_pos": alpha_j_pos,
                    "alpha_j_neg": alpha_j_neg,
                    "alpha_wind_feed": alpha_wind_feed,
                    "alpha_solar_feed": alpha_solar_feed,
                    "jump_prob_mult_pos": jump_prob_mult_pos,
                    "jump_prob_mult_neg": jump_prob_mult_neg,
                    "stress_p11_mult": stress_p11_mult,
                    "drift": {
                        "enabled": drift_enabled,
                        "period_mode": drift_period,
                        "rho": drift_rho,
                        "sigma": drift_sigma,
                        "center": drift_center,
                        "half_range": drift_half_rng,
                        "alpha": alpha_drift,
                    },
                },
                "model_params_used": {
                    "b0": b0, "b1": b1, "b2": b2,
                    "period_mode": period_mode, "a_mu": a_mu, "rho_mu": rho_mu, "sig_mu": sig_mu,
                    "wind_phi": wind_phi, "wind_sigma": wind_sigma,
                    "solar_phi": solar_phi, "solar_sigma": solar_sigma,
                    "P_resid_base": fast_cfg["P"].tolist(),
                    "P_resid_adj": P_adj.tolist(),
                    "normal_idx": fast_cfg["normal_idx"],
                    "stress_idx": fast_cfg["stress_idx"],
                    "normal_params": list(fast_cfg["normal_params"]),
                    "stress_params": list(fast_cfg["stress_params"]),
                    "jump_pos_enabled": bool(fast_cfg["jump_pos"].get("enabled", False)),
                    "jump_neg_enabled": bool(fast_cfg["jump_neg"].get("enabled", False)),
                    "shape_layer_enabled": True,
                    "shape_layer_type": "hour_month",
                    "jump_prob_profile_enabled": True,
                    "jump_prob_profile_type": "hour_month",
                }
            }

            with open(out_meta, "w") as f:
                json.dump(meta, f, indent=2)

def load_marketInput(config):
    """
    Loads wind farm input parameters from the configuration file.

    Returns
    -------
    dict
        Dictionary with wind farm parameters.
    """
    market_inputs = {}

    if hasattr(config, 'Market_inputFiles'):
        for identifier, file_name in config.Market_inputFiles.items():
            market_inputs[identifier] = load_yaml(config.valuewind_inputFolder, file_name)
            market_inputs[identifier] = process_duration_fields(market_inputs[identifier])

    return market_inputs

# Helper functions for Keles-style model

