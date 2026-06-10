
# ============================================================
# Stochastic market model runtime (single-path generator)
# - Artifact-driven (merit/residual/mean json + profiles)
# - Strict: required blocks/files must exist, no silent fallbacks
# - Produces ONE path; Monte Carlo handled outside
# ============================================================

import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
from core.utils import repeat_timeseries, gap_fill_timeseries, get_input_parameter, remove_gaps_rebuild_timestamps, repeat_timeseries_to_duration


MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

# -----------------------------
# Strict helpers
# -----------------------------
def _load_json_strict(p: Path) -> dict:
    if not p.exists():
        raise FileNotFoundError(str(p))
    with open(p, "r") as f:
        return json.load(f)

def _require(d: dict, path: str):
    cur = d
    for k in path.split("."):
        if not isinstance(cur, dict) or k not in cur:
            raise ValueError(f"Missing required field: '{path}'")
        cur = cur[k]
    if cur is None:
        raise ValueError(f"Field is null but required: '{path}'")
    return cur

def _require_float(d: dict, path: str) -> float:
    v = _require(d, path)
    try:
        return float(v)
    except Exception:
        raise ValueError(f"Field '{path}' must be numeric. Got: {v!r}")

def _require_bool(d: dict, path: str) -> bool:
    v = _require(d, path)
    if isinstance(v, bool):
        return v
    raise ValueError(f"Field '{path}' must be boolean. Got: {v!r}")

def _clip01(x: float) -> float:
    x = float(x)
    return float(min(max(x, 0.0), 1.0))

def _scale_prob_logit(p: float, mult: float) -> float:
    """Scale probability p in (0,1) by multiplying its logit by mult."""
    p = float(p)
    if not (0.0 < p < 1.0):
        raise ValueError(f"_scale_prob_logit expects p in (0,1). Got p={p}")
    mult = float(mult)
    logit = np.log(p / (1.0 - p))
    logit2 = logit * mult
    return float(1.0 / (1.0 + np.exp(-logit2)))


# -----------------------------
# Profiles + feed-in simulation
# -----------------------------
def _load_hour_month_profile(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(str(csv_path))
    prof = pd.read_csv(csv_path)
    if "hour" not in prof.columns:
        raise ValueError(f"{csv_path.name} must have column 'hour'")
    for m in MONTHS:
        if m not in prof.columns:
            raise ValueError(f"{csv_path.name} missing month column '{m}'")
    prof = prof[["hour"] + MONTHS].copy()
    prof["hour"] = pd.to_numeric(prof["hour"], errors="raise").astype(int)
    prof = prof.sort_values("hour").reset_index(drop=True)
    return prof

def _seasonality_from_profile(timestamps: pd.DatetimeIndex, prof: pd.DataFrame) -> np.ndarray:
    prof_idx = prof.set_index("hour")
    months = timestamps.month.to_numpy()
    hours = timestamps.hour.to_numpy()
    out = np.zeros(len(timestamps), dtype=float)
    for i, (m, h) in enumerate(zip(months, hours)):
        out[i] = float(prof_idx.loc[int(h), MONTHS[int(m) - 1]])
    return out

def _simulate_seasonal_ar1_cf(
    rng: np.random.Generator,
    timestamps: pd.DatetimeIndex,
    prof: pd.DataFrame,
    phi: float,
    sigma: float,
    lo: float = 0.0,
    hi: float = 1.0,
) -> np.ndarray:
    S = _seasonality_from_profile(timestamps, prof)
    n = len(timestamps)
    x = np.zeros(n, dtype=float)
    eps = rng.normal(0.0, 1.0, size=n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + sigma * eps[t]
    return np.clip(S + x, lo, hi)

def _load_capacity_series(capacity_csv: Path, timestamps: pd.DatetimeIndex, col: str) -> np.ndarray:
    if not capacity_csv.exists():
        raise FileNotFoundError(str(capacity_csv))
    cap = pd.read_csv(capacity_csv).copy()
    if "date" not in cap.columns or col not in cap.columns:
        raise ValueError(f"{capacity_csv.name} must have columns: date, {col}")

    cap["date"] = cap["date"].astype(str)
    cap["date_dt"] = pd.to_datetime(
        cap["date"].str.replace(r"^(\d{4})$", r"\1-01-01", regex=True),
        errors="coerce",
    )
    cap[col] = pd.to_numeric(cap[col], errors="coerce")
    cap = cap.dropna(subset=["date_dt", col]).sort_values("date_dt")
    if cap.empty:
        raise ValueError(f"{capacity_csv.name}: no valid rows after parsing for column '{col}'")

    cap = cap.set_index("date_dt")[[col]]
    cap = cap[~cap.index.duplicated(keep="last")]

    # align using UTC-naive timestamps (capacity is typically date-based)
    idx_naive = timestamps.tz_convert("UTC").tz_localize(None) if timestamps.tz is not None else timestamps
    aligned = cap.reindex(idx_naive, method="ffill").bfill()
    if aligned[col].isna().any():
        raise ValueError(f"{capacity_csv.name}: capacity alignment produced NaNs for '{col}'")
    return aligned[col].to_numpy(dtype=float)


# -----------------------------
# Mean layer (Tool E)
# -----------------------------
def _parse_mean_model(mean_cfg: dict):
    src = _require(mean_cfg, "source")
    period_mode = str(_require(src, "period_mode")).lower()
    if period_mode not in {"year", "month"}:
        raise ValueError(f"mean_model.source.period_mode must be 'year' or 'month'. Got {period_mode!r}")

    a = _require_float(mean_cfg, "ar1.a")
    rho = _require_float(mean_cfg, "ar1.rho")
    sigma = _require_float(mean_cfg, "ar1.sigma")
    include_intercept = _require_bool(mean_cfg, "ar1.include_intercept")
    if not include_intercept:
        raise ValueError("This simulator expects ar1.include_intercept = true.")
    return period_mode, a, rho, sigma

def _simulate_mu_ar1_period(
    rng: np.random.Generator,
    period_mode: str,
    timestamps_hourly: pd.DatetimeIndex,
    a: float,
    rho: float,
    sigma: float,
) -> np.ndarray:
    if period_mode == "year":
        start_y = int(timestamps_hourly[0].year)
        end_y   = int(timestamps_hourly[-1].year)
        keys = list(range(start_y, end_y + 1))
        hour_keys = timestamps_hourly.year.astype(int)

    elif period_mode == "month":
        start_m = timestamps_hourly[0].to_period("M")
        end_m   = timestamps_hourly[-1].to_period("M")
        periods = list(pd.period_range(start=start_m, end=end_m, freq="M"))
        keys = [str(p) for p in periods]  # "YYYY-MM"
        hour_keys = timestamps_hourly.to_period("M").astype(str)

    else:
        raise ValueError(f"Unknown period_mode: {period_mode!r}")

    T = len(keys)
    mu = np.zeros(T, dtype=float)
    eps = rng.normal(0.0, sigma, size=T)

    mu0 = a / (1.0 - rho) if abs(1.0 - rho) > 1e-6 else 0.0
    mu[0] = mu0 + eps[0]
    for t in range(1, T):
        mu[t] = a + rho * mu[t - 1] + eps[t]

    mapping = dict(zip(keys, mu))
    return np.array([mapping[k] for k in hour_keys], dtype=float)


def _simulate_bounded_ou_drift(
    rng: np.random.Generator,
    timestamps_hourly: pd.DatetimeIndex,
    period_mode: str,     # "year" or "month"
    rho: float,
    sigma: float,         # latent shocks
    center: float,        # €/MWh
    half_range: float,    # max deviation
) -> np.ndarray:
    if period_mode == "year":
        keys = pd.Index(sorted(timestamps_hourly.year.unique().tolist()))
        hour_keys = timestamps_hourly.year
    elif period_mode == "month":
        hour_keys = timestamps_hourly.to_period("M").astype(str)
        keys = pd.Index(pd.unique(hour_keys))
    else:
        raise ValueError("period_mode must be 'year' or 'month'")

    T = len(keys)
    z = np.zeros(T, dtype=float)
    eps = rng.normal(0.0, 1.0, size=T)
    for t in range(1, T):
        z[t] = rho * z[t - 1] + sigma * eps[t]

    d = center + half_range * np.tanh(z)
    mapping = dict(zip(keys.tolist(), d.tolist()))
    return np.array([mapping[k] for k in hour_keys], dtype=float)


# -----------------------------
# Shape layer (Tool C v5)
# -----------------------------
def _load_shape_profile_required(resid_cfg: dict, resid_json_path: Path) -> Path:
    shape_layer = _require(resid_cfg, "shape_layer")
    if not isinstance(shape_layer, dict):
        raise ValueError("shape_layer must be a dict")
    if not bool(shape_layer.get("enabled", False)):
        raise ValueError("shape_layer.enabled must be true (no fallbacks in this simulator)")
    stype = str(_require(shape_layer, "type")).lower()
    if stype != "hour_month":
        raise ValueError(f"shape_layer.type must be 'hour_month'. Got: {stype!r}")
    profile_file = _require(shape_layer, "profile_file")
    prof_path = resid_json_path.parent / str(profile_file)
    if not prof_path.exists():
        raise FileNotFoundError(str(prof_path))
    return prof_path


# -----------------------------
# Conditional jump probabilities (REQUIRED)
# -----------------------------
def _load_jump_prob_profile_required(resid_cfg: dict) -> dict:
    jpp = _require(resid_cfg, "jump_prob_profile")
    if not isinstance(jpp, dict):
        raise ValueError("jump_prob_profile must be a dict")
    if not bool(jpp.get("enabled", False)):
        raise ValueError("jump_prob_profile.enabled must be true (no fallbacks)")
    if str(_require(jpp, "type")).lower() != "hour_month":
        raise ValueError(f"jump_prob_profile.type must be 'hour_month'. Got {jpp.get('type')!r}")

    for side in ("pos", "neg"):
        side_block = _require(jpp, side)
        rates = _require(side_block, "rates")
        if not isinstance(rates, dict):
            raise ValueError(f"jump_prob_profile.{side}.rates must be a dict")
    return jpp

def _lookup_hour_month_prob(rates: dict, hour: int, month_name: str) -> float:
    if hour in rates:
        hblock = rates[hour]
    else:
        hblock = rates.get(str(hour), None)
    if hblock is None or not isinstance(hblock, dict):
        return 0.0
    v = hblock.get(month_name, None)
    if v is None:
        return 0.0
    return float(v)


# -----------------------------
# Residual (Tool C v5): regimes AR(1) + explicit jumps
# -----------------------------
def _parse_residual_fast_v5(resid_cfg: dict):
    P = np.asarray(_require(resid_cfg, "transitions.P"), dtype=float)
    if P.shape != (2, 2):
        raise ValueError(f"transitions.P must be 2x2. Got {P.shape}.")

    phi_n = _require_float(resid_cfg, "regimes.normal.phi")
    sig_n = _require_float(resid_cfg, "regimes.normal.sigma")
    phi_s = _require_float(resid_cfg, "regimes.stress.phi")
    sig_s = _require_float(resid_cfg, "regimes.stress.sigma")

    stress_idx = int(_require(resid_cfg, "regimes.stress_regime_index"))
    normal_idx = int(_require(resid_cfg, "regimes.normal_regime_index"))
    if set([stress_idx, normal_idx]) != {0, 1}:
        raise ValueError(
            f"Expected stress/normal indices to be 0/1. Got stress={stress_idx}, normal={normal_idx}"
        )

    jumps = _require(resid_cfg, "jumps")

    def parse_jump(side: str):
        j = _require(jumps, side)
        enabled = bool(j.get("enabled", False))
        if not enabled:
            return {"enabled": False}

        dist = _require(j, "excess_dist")
        dtype = str(_require(dist, "type")).lower()
        if dtype != "lognormal":
            raise ValueError(f"Only lognormal excess_dist supported. Got {dtype!r}")

        mu = _require_float(dist, "mu")
        sigma = _require_float(dist, "sigma")
        return {"enabled": True, "mu": float(mu), "sigma": float(sigma)}

    return {
        "P": P,
        "normal_idx": normal_idx,
        "stress_idx": stress_idx,
        "normal_params": (float(phi_n), float(sig_n)),
        "stress_params": (float(phi_s), float(sig_s)),
        "jump_pos": parse_jump("pos"),
        "jump_neg": parse_jump("neg"),
    }

def _adjust_transition_matrix_stress_p11(P: np.ndarray, stress_idx: int, mult: float) -> np.ndarray:
    P = np.asarray(P, float).copy()
    if float(mult) == 1.0:
        return P
    p_ss = float(P[stress_idx, stress_idx])
    if not (0.0 < p_ss < 1.0):
        raise ValueError(f"P[stress,stress] must be in (0,1). Got {p_ss}")
    p_ss_new = _scale_prob_logit(p_ss, float(mult))
    P[stress_idx, stress_idx] = p_ss_new
    P[stress_idx, 1 - stress_idx] = 1.0 - p_ss_new
    return P

def _simulate_fast_components_conditional_jumps(
    rng: np.random.Generator,
    timestamps: pd.DatetimeIndex,
    P: np.ndarray,
    normal_idx: int,
    stress_idx: int,
    normal_params,
    stress_params,
    jump_pos: dict,
    jump_neg: dict,
    jpp_pos_rates: dict,
    jpp_neg_rates: dict,
    jump_prob_mult_pos: float = 1.0,
    jump_prob_mult_neg: float = 1.0,
    start_state: int | None = None,
):
    n = len(timestamps)
    phi_n, sig_n = normal_params
    phi_s, sig_s = stress_params

    state = np.zeros(n, dtype=int)
    state[0] = int(normal_idx if start_state is None else start_state)

    base = np.zeros(n, dtype=float)
    jpos = np.zeros(n, dtype=float)
    jneg = np.zeros(n, dtype=float)

    base[0] = (sig_s if state[0] == stress_idx else sig_n) * rng.normal()

    hours = timestamps.hour.to_numpy()
    month_names = np.array([MONTHS[m - 1] for m in timestamps.month.to_numpy()], dtype=object)

    for t in range(1, n):
        prev = state[t - 1]
        state[t] = prev if (rng.random() < float(P[prev, prev])) else (1 - prev)
        is_stress = (state[t] == stress_idx)

        eps = rng.normal()
        if is_stress:
            base[t] = phi_s * base[t - 1] + sig_s * eps
        else:
            base[t] = phi_n * base[t - 1] + sig_n * eps

        h = int(hours[t])
        mn = str(month_names[t])

        if jump_pos.get("enabled", False):
            lam_pos = _lookup_hour_month_prob(jpp_pos_rates, h, mn)
            lam_pos = _clip01(float(lam_pos) * float(jump_prob_mult_pos))
            if rng.random() < lam_pos:
                jpos[t] = rng.lognormal(mean=float(jump_pos["mu"]), sigma=float(jump_pos["sigma"]))

        if jump_neg.get("enabled", False):
            lam_neg = _lookup_hour_month_prob(jpp_neg_rates, h, mn)
            lam_neg = _clip01(float(lam_neg) * float(jump_prob_mult_neg))
            if rng.random() < lam_neg:
                jneg[t] = rng.lognormal(mean=float(jump_neg["mu"]), sigma=float(jump_neg["sigma"]))

    u = base + jpos - jneg
    return base, state, jpos, jneg, u


def _map_cf_sys_to_cf_asset(cf_sys, mapping_cfg):
    """
    Deterministic mapping from system wind CF to asset CF.

    CF_asset(t) = a0 + a1*CF_sys(t) + a2*CF_sys(t)^2 + ... + ak*CF_sys(t)^k
    """
    coeffs = np.asarray(_require(mapping_cfg, 'coefficients'), dtype=float)
    if coeffs.size == 0:
        raise ValueError("asset_cf_mapping.coefficients must be provided.")

    x = np.asarray(cf_sys, dtype=float)
    cf_asset = np.polyval(coeffs[::-1], x)

    lo, hi = get_input_parameter(mapping_cfg, 'clip_bounds') or (0.0, 1.0)
    cf_asset = np.clip(cf_asset, float(lo), float(hi))
    return cf_asset


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def diagnose_and_plot_price_series(
    df: pd.DataFrame,
    *,
    timestamp_col: str = "timestamp",
    price_col: str = "price",
    title: str = "Simulated electricity price",
    ramps_h: tuple[int, ...] = (1, 3, 6, 24),
    acf_max_lag: int = 48,
    spike_q: float = 0.99,
    assume_utc: bool = True,
    verbose: bool = False,   # <-- NEW FLAG
) -> dict:
    """
    Diagnose + plot a single simulated price time series.

    Parameters
    ----------
    df : DataFrame
        Must contain timestamp_col and price_col.
    timestamp_col : str
        Column with timestamps.
    price_col : str
        Column with price values (€/MWh).
    title : str
        Plot title.
    ramps_h : tuple[int,...]
        Ramp horizons in hours for Δ_h P.
    acf_max_lag : int
        ACF lags to compute.
    spike_q : float
        Quantile threshold for spike intensity (e.g. 0.99).
    assume_utc : bool
        If True, parse timestamps as UTC for yearly grouping.
    verbose : bool
        If True, print diagnostics and show plot.

    Returns
    -------
    dict with computed diagnostics.
    """

    d = df[[timestamp_col, price_col]].copy()
    d[timestamp_col] = pd.to_datetime(d[timestamp_col], utc=assume_utc, errors="coerce")
    d = d.dropna(subset=[timestamp_col, price_col]).sort_values(timestamp_col)

    x = d[price_col].to_numpy(dtype=float)
    x = x[np.isfinite(x)]

    if len(x) < 10:
        raise ValueError("Not enough finite samples for diagnostics.")

    # -----------------------------
    # Level stats
    # -----------------------------
    lvl = {
        "mean": float(np.mean(x)),
        "std": float(np.std(x, ddof=1)) if len(x) > 1 else np.nan,
        "p05": float(np.quantile(x, 0.05)),
        "p50": float(np.quantile(x, 0.50)),
        "p95": float(np.quantile(x, 0.95)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "neg_share": float(np.mean(x < 0.0)),
    }

    # -----------------------------
    # Ramps Δ_h P
    # -----------------------------
    ramps = []
    for h in ramps_h:
        if len(x) <= h:
            ramps.append((h, np.nan, np.nan, np.nan, np.nan))
            continue
        dx = x[h:] - x[:-h]
        absdx = np.abs(dx)
        ramps.append((
            int(h),
            float(np.std(dx, ddof=1)) if len(dx) > 1 else np.nan,
            float(np.median(absdx)) if len(absdx) else np.nan,
            float(np.quantile(absdx, 0.95)) if len(absdx) else np.nan,
            float(np.quantile(absdx, 0.99)) if len(absdx) else np.nan,
        ))
    ramps_df = pd.DataFrame(
        ramps,
        columns=["horizon_h", "std", "mad", "q95_abs", "q99_abs"]
    ).set_index("horizon_h")

    # -----------------------------
    # ACF summary
    # -----------------------------
    def acf_summary(xx: np.ndarray, max_lag: int) -> dict:
        if len(xx) < max_lag + 5:
            return {"acf_lag1": np.nan, "acf_mean_abs": np.nan, "acf_max_abs": np.nan}
        x0 = xx - np.mean(xx)
        denom = float(np.dot(x0, x0))
        if denom <= 1e-12:
            return {"acf_lag1": np.nan, "acf_mean_abs": np.nan, "acf_max_abs": np.nan}
        acfs = []
        for lag in range(1, max_lag + 1):
            num = float(np.dot(x0[lag:], x0[:-lag]))
            acfs.append(num / denom)
        acfs = np.asarray(acfs, float)
        return {
            "acf_lag1": float(acfs[0]),
            "acf_mean_abs": float(np.mean(np.abs(acfs))),
            "acf_max_abs": float(np.max(np.abs(acfs))),
        }

    acf = acf_summary(x, acf_max_lag)

    # -----------------------------
    # Spike intensity
    # -----------------------------
    thr = float(np.quantile(x, spike_q))
    exc = x[x > thr] - thr
    spike = {
        "q": float(spike_q),
        "thr": thr,
        "exceed_rate": float(np.mean(x > thr)),
        "mean_exceed": float(np.mean(exc)) if len(exc) else 0.0,
        "topq_mean": float(np.mean(x[x >= thr])) if np.any(x >= thr) else np.nan,
    }

    # -----------------------------
    # Negative hours per year
    # -----------------------------
    d["year"] = d[timestamp_col].dt.year.astype(int)
    d["is_neg"] = (d[price_col] < 0.0).astype(int)
    neg_year = d.groupby("year").agg(
        neg_hours=("is_neg", "sum"),
        hours=("is_neg", "size")
    )
    neg_year["neg_share_pct"] = 100.0 * neg_year["neg_hours"] / neg_year["hours"]

    # -----------------------------
    # Print summary (optional)
    # -----------------------------
    if verbose:
        print("\n========== Diagnostics: current simulated series ==========")
        print(f"Rows: {len(d)}")
        print(f"Start: {d[timestamp_col].min()}")
        print(f"End:   {d[timestamp_col].max()}")

        print("\n---- Level stats ----")
        for k in ["mean", "std", "p05", "p50", "p95", "min", "max", "neg_share"]:
            print(f"{k:<10} {lvl[k]:.4f}")

        print("\n---- Negative hours per year ----")
        print(neg_year)

        print("\n---- Ramp stats Δ_h P ----")
        print(ramps_df)

        print("\n---- ACF summary ----")
        print(f"ACF lag-1: {acf['acf_lag1']:.5f}")
        print(f"ACF mean |ACF| lags 1..{acf_max_lag}: {acf['acf_mean_abs']:.5f}")
        print(f"ACF max  |ACF| lags 1..{acf_max_lag}: {acf['acf_max_abs']:.5f}")

        print("\n---- Spike intensity ----")
        print(f"P{int(100*spike_q)} threshold: {spike['thr']:.4f} €/MWh")
        print(f"Exceed rate: {100.0*spike['exceed_rate']:.2f}%")
        print(f"Mean exceedance: {spike['mean_exceed']:.4f} €/MWh")
        print(f"Top {(1.0-spike_q)*100:.1f}% mean: {spike['topq_mean']:.4f} €/MWh")
        print("==========================================================\n")

        # Plot
        plt.figure()
        plt.plot(d[timestamp_col].to_numpy(), d[price_col].to_numpy())
        plt.title(title)
        plt.xlabel("Time")
        plt.ylabel(price_col)
        plt.tight_layout()
        plt.show()

    return {
        "level": lvl,
        "ramps": ramps_df,
        "acf": acf,
        "spike": spike,
        "negative_hours_per_year": neg_year,
    }