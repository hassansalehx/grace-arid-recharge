"""
Groundwater observation data preprocessing functions.

This module processes groundwater monitoring data from GGMN network,
reading ODS files and creating formatted dataframes for analysis.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings

# -----------------------------------------------------------------------------
# Shared constants
# -----------------------------------------------------------------------------
QC_BASELINE_START = "2004-01-01"
QC_BASELINE_END = "2009-12-31"
QC_OUTLIER_THRESHOLD_M = 30.0
QC_INTERP_LIMIT_MONTHS = 3
QC_STUCK_WINDOW_MONTHS = 60
QC_STUCK_STD_THRESHOLD = 0.05


# -----------------------------------------------------------------------------
# Helper utilities (reuse across correlation/plotting)
# -----------------------------------------------------------------------------
def _import_tqdm():
    """Import tqdm with Jupyter-friendly fallback. Returns (tqdm_or_None, use_tqdm)."""
    try:
        from tqdm.auto import tqdm
        return tqdm, True
    except ImportError:
        try:
            from tqdm import tqdm
            return tqdm, True
        except ImportError:
            warnings.warn("tqdm not available. Install with 'pip install tqdm' for progress bars.")
            return None, False


def _require_cartopy(error_message: str):
    """Import cartopy CRS/feature or raise ModuleNotFoundError with *error_message*."""
    try:
        import cartopy.crs as ccrs
        import cartopy.feature as cfeature
        return ccrs, cfeature
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(error_message) from e


def _format_pvalue(p) -> str:
    """Format a correlation p-value as stored on well GeoDataFrames."""
    if p < 0.01:
        return '<0.01'
    if p < 0.05:
        return '<0.05'
    return f'{p:.2f}'


def _apply_sign_flip(series: pd.Series, parameter_type) -> pd.Series:
    """Flip sign if parameter_type indicates depth measurement."""
    if parameter_type is None:
        return series
    param_lower = str(parameter_type).lower()
    if 'depth' in param_lower and ('ground' in param_lower or 'well' in param_lower):
        return -series
    return series


def _align_timezones(primary: pd.Series, reference: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Align timezones between two series if one is tz-aware and the other is not."""
    primary = primary.copy()
    reference = reference.copy()
    if primary.index.tz is None and reference.index.tz is not None:
        primary.index = primary.index.tz_localize('UTC')
    elif primary.index.tz is not None and reference.index.tz is None:
        reference.index = reference.index.tz_localize('UTC')
    return primary, reference


def _apply_qc_cleaning(series: pd.Series) -> tuple[pd.Series, bool, float, int, int, float]:
    """
    Apply QC to a well series:
      1) Remove outliers >30m from baseline mean (2004-2009)
      2) Interpolate gaps up to 3 months (inside only)
      3) Flag stuck sensor if rolling std over 60 months < 0.05
    Returns: cleaned_series, stuck_invalid, baseline_mean, n_outliers, n_filled, min_roll_std
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    
    # Handle timezone for baseline comparison
    idx_tz = s.index.tz
    if idx_tz is not None:
        b0 = pd.Timestamp(QC_BASELINE_START, tz='UTC')
        b1 = pd.Timestamp(QC_BASELINE_END, tz='UTC')
    else:
        b0 = pd.Timestamp(QC_BASELINE_START)
        b1 = pd.Timestamp(QC_BASELINE_END)

    baseline_vals = s.loc[(s.index >= b0) & (s.index <= b1)].dropna()
    baseline_mean = baseline_vals.mean() if len(baseline_vals) > 0 else s.dropna().mean()

    if pd.notna(baseline_mean):
        outlier_mask = s.notna() & ((s - baseline_mean).abs() > QC_OUTLIER_THRESHOLD_M)
    else:
        outlier_mask = pd.Series(False, index=s.index)
    n_outliers = int(outlier_mask.sum())
    if n_outliers > 0:
        s.loc[outlier_mask] = np.nan

    na_before = s.isna()
    try:
        s_interp = s.interpolate(method='time', limit=QC_INTERP_LIMIT_MONTHS, limit_area='inside')
    except Exception:
        s_interp = s
    n_filled = int((na_before & s_interp.notna()).sum())

    roll_std = s_interp.rolling(window=QC_STUCK_WINDOW_MONTHS, min_periods=QC_STUCK_WINDOW_MONTHS).std()
    min_roll_std = float(roll_std.min()) if len(roll_std.dropna()) > 0 else np.nan
    stuck_invalid = bool((roll_std < QC_STUCK_STD_THRESHOLD).any()) if len(roll_std) > 0 else False

    return s_interp, stuck_invalid, float(baseline_mean) if pd.notna(baseline_mean) else np.nan, n_outliers, n_filled, min_roll_std


def _clean_flagged_well_series(series: pd.Series, jump_threshold_m: float = 50.0, 
                               suspicious_threshold_m: float = 50.0,
                               interp_limit_months: int = 3) -> tuple[pd.Series, int, int, int]:
    """
    Remove jump, suspicious, and zero values from a well time series, then interpolate gaps.
    
    - Zero: remove points that are exactly 0
    - Jump: remove points where |diff| > jump_threshold_m between consecutive values
    - Suspicious: remove points that deviate > suspicious_threshold_m from rolling median (12-month)
    - Interpolate gaps up to interp_limit_months (inside only)
    
    Returns: (cleaned_series, n_jumps_removed, n_suspicious_removed, n_zeros_removed)
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    
    n_jumps = 0
    n_suspicious = 0
    n_zeros = 0
    
    # Remove zeros: points that are exactly 0
    zero_mask = (s == 0) & s.notna()
    if zero_mask.any():
        s.loc[zero_mask] = np.nan
        n_zeros = int(zero_mask.sum())
    
    # Remove jumps: points where |diff| > threshold
    diff = s.diff()
    jump_mask = diff.abs() > jump_threshold_m
    if jump_mask.any():
        # Remove the point that causes the jump (the one after the diff)
        s.loc[jump_mask] = np.nan
        n_jumps = int(jump_mask.sum())
    
    # Remove suspicious: points > threshold from 12-month rolling median
    roll_med = s.rolling(window=12, min_periods=1, center=True).median()
    dev = (s - roll_med).abs()
    suspicious_mask = s.notna() & (dev > suspicious_threshold_m)
    if suspicious_mask.any():
        s.loc[suspicious_mask] = np.nan
        n_suspicious = int(suspicious_mask.sum())
    
    # Interpolate gaps up to interp_limit_months (inside only)
    try:
        s_clean = s.interpolate(method='time', limit=interp_limit_months, limit_area='inside')
    except Exception:
        s_clean = s
    
    return s_clean, n_jumps, n_suspicious, n_zeros


def _get_well_ts_key(row) -> str:
    """Return the key to use for time_series column lookup (unique_well_id when present, else ID)."""
    if 'unique_well_id' in row.index and pd.notna(row.get('unique_well_id')):
        return str(row['unique_well_id']).strip()
    return str(row['ID']).strip()


def _remove_baseline_mean(series: pd.Series) -> tuple[pd.Series, float]:
    """Remove mean over 2004-01-01 to 2009-12-31 window (falls back to full mean)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    if s.index.tz is not None:
        start = pd.Timestamp(QC_BASELINE_START, tz='UTC')
        end = pd.Timestamp(QC_BASELINE_END, tz='UTC')
    else:
        start = pd.Timestamp(QC_BASELINE_START)
        end = pd.Timestamp(QC_BASELINE_END)
    mask = (s.index >= start) & (s.index <= end)
    mean_val = s.loc[mask].mean() if mask.sum() > 0 else s.mean()
    return s - mean_val, float(mean_val) if pd.notna(mean_val) else np.nan


def _decompose_series_full(
    series: pd.Series,
    decomposition_method: str = 'harmonic',
) -> dict:
    """
    Decompose series into trend, annual, semi-annual, and residual components.

    decomposition_method='harmonic' (default):
        Same fit as _decompose_series: y = a*t + b + annual + semi_annual + residual,
        using global linear trend and fixed annual/semi-annual sinusoids.

    decomposition_method='stl_13':
        Use STL decomposition with monthly period=12 and 13-month windows for both
        trend and seasonality, similar to a 13-month long-term filter. The STL
        seasonal component is returned as 'annual', and 'semi_annual' is set to 0.

    Returns dict with keys: 'original', 'trend', 'annual', 'semi_annual', 'residual'.
    Uses 2004-2009 baseline for consistency; call _remove_baseline_mean first for anomaly.
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    valid = s.notna()
    if valid.sum() < 12:
        return {
            'original': s,
            'trend': pd.Series(np.nan, index=s.index),
            'annual': pd.Series(np.nan, index=s.index),
            'semi_annual': pd.Series(np.nan, index=s.index),
            'residual': s,
        }

    method = (decomposition_method or 'harmonic').lower()

    if method == 'stl_13':
        # STL expects a regular frequency; resample to monthly means first.
        # Many GWL/GRACE series have dates mid-month, so asfreq('MS') would be all-NaN.
        s_monthly = s.resample('MS').mean()
        if s_monthly.notna().sum() < 12:
            # Fallback: treat as insufficient data
            return {
                'original': s,
                'trend': pd.Series(np.nan, index=s.index),
                'annual': pd.Series(np.nan, index=s.index),
                'semi_annual': pd.Series(np.nan, index=s.index),
                'residual': s,
            }
        s_interp = s_monthly.interpolate(method='time')
        try:
            from statsmodels.tsa.seasonal import STL  # local import to avoid hard dependency at module import
        except ImportError:
            # If STL is unavailable, fall back to harmonic decomposition
            method = 'harmonic'
        else:
            stl = STL(s_interp, period=12, seasonal=13, trend=13, robust=True)
            res = stl.fit()
            # Map monthly STL components back to the original timestamps (nearest month).
            trend = res.trend.reindex(s.index, method='nearest')
            seasonal = res.seasonal.reindex(s.index, method='nearest')
            resid = res.resid.reindex(s.index, method='nearest')
            # Restore original NaNs in residual so gaps stay aligned with source data.
            resid = resid.where(s.notna())
            semi_zeros = pd.Series(0.0, index=s.index)
            return {
                'original': s,
                'trend': trend,
                'annual': seasonal,
                'semi_annual': semi_zeros,
                'residual': resid,
            }

    # Default: harmonic linear trend + fixed annual and semi-annual sinusoids
    t0 = s.index.min()
    months = (s.index - t0).total_seconds() / (365.25 / 12 * 24 * 3600)
    t = np.asarray(months, dtype=float)
    y = np.asarray(s.values, dtype=float)
    X = np.column_stack([
        t,
        np.ones_like(t),
        np.cos(2 * np.pi * t / 12),
        np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 6),
        np.sin(2 * np.pi * t / 6),
    ])
    mask = ~np.isnan(y)
    if mask.sum() < 12:
        return {
            'original': s,
            'trend': pd.Series(np.nan, index=s.index),
            'annual': pd.Series(np.nan, index=s.index),
            'semi_annual': pd.Series(np.nan, index=s.index),
            'residual': s,
        }
    X_fit = X[mask]
    y_fit = y[mask]
    try:
        coeffs, _, _, _ = np.linalg.lstsq(X_fit, y_fit, rcond=None)
    except Exception:
        return {
            'original': s,
            'trend': pd.Series(np.nan, index=s.index),
            'annual': pd.Series(np.nan, index=s.index),
            'semi_annual': pd.Series(np.nan, index=s.index),
            'residual': s,
        }
    trend_vals = coeffs[0] * t + coeffs[1]
    annual_vals = coeffs[2] * X[:, 2] + coeffs[3] * X[:, 3]
    semi_annual_vals = coeffs[4] * X[:, 4] + coeffs[5] * X[:, 5]
    fitted = trend_vals + annual_vals + semi_annual_vals
    residual_vals = y - fitted
    return {
        'original': s,
        'trend': pd.Series(trend_vals, index=s.index),
        'annual': pd.Series(annual_vals, index=s.index),
        'semi_annual': pd.Series(semi_annual_vals, index=s.index),
        'residual': pd.Series(residual_vals, index=s.index),
    }


def _extract_grace_series(grace_data, lat, lon) -> pd.Series:
    """Extract GRACE series at nearest lat/lon and return pandas Series with datetime index."""
    grace_ts = grace_data.sel(lat=lat, lon=lon, method='nearest')
    return pd.Series(grace_ts.values, index=pd.to_datetime(grace_ts.time.values))


def _calculate_best_lag_correlation(well_series: pd.Series, grace_series: pd.Series, max_lag_months: int, min_common_dates: int, corr_func) -> dict:
    """
    Compute correlation across lags [0, max_lag_months]; return lag-0 and best (highest positive) correlation.
    Returns: dict with r_lag0, p_lag0, r_max, p_max, lag_max
    """
    # Make copies to avoid modifying originals
    well_series = well_series.copy().sort_index().dropna()
    grace_series = grace_series.copy().sort_index().dropna()
    
    # Ensure both indexes are DatetimeIndex
    well_series.index = pd.to_datetime(well_series.index)
    grace_series.index = pd.to_datetime(grace_series.index)
    
    # Align timezones - make both timezone-naive for comparison
    if hasattr(well_series.index, 'tz') and well_series.index.tz is not None:
        well_series.index = well_series.index.tz_localize(None)
    if hasattr(grace_series.index, 'tz') and grace_series.index.tz is not None:
        grace_series.index = grace_series.index.tz_localize(None)
    
    # Normalize both indexes to start-of-month (GRACE is typically monthly)
    # This ensures dates like 2010-01-15 and 2010-01-01 will match as 2010-01-01
    well_series.index = well_series.index.to_period('M').to_timestamp()
    grace_series.index = grace_series.index.to_period('M').to_timestamp()
    
    # If there are duplicate months after normalization, take the mean
    if well_series.index.duplicated().any():
        well_series = well_series.groupby(well_series.index).mean()
    if grace_series.index.duplicated().any():
        grace_series = grace_series.groupby(grace_series.index).mean()
    
    # Find common dates (intersection of indexes)
    common_dates = well_series.index.intersection(grace_series.index)
    if len(common_dates) < min_common_dates:
        return {
            'r_lag0': np.nan, 'p_lag0': np.nan,
            'r_max': np.nan, 'p_max': np.nan, 'lag_max': np.nan
        }
    
    # Align both series to common dates
    well_aligned = well_series.loc[common_dates].sort_index()
    grace_aligned = grace_series.loc[common_dates].sort_index()
    
    svals = well_aligned.values
    gvals = grace_aligned.values
    
    # Initialize results
    r_lag0, p_lag0 = np.nan, np.nan
    best_r, best_p, best_lag = np.nan, np.nan, np.nan
    
    for lag in range(0, max_lag_months + 1):
        if lag == 0:
            ws, gs = svals, gvals
        else:
            if len(svals) <= lag:
                continue
            ws, gs = svals[lag:], gvals[:-lag]
        min_len = min(len(ws), len(gs))
        if min_len < 2:
            continue
        ws = ws[:min_len]
        gs = gs[:min_len]
        mask = ~(np.isnan(ws) | np.isnan(gs))
        n_common = int(mask.sum())
        # Enforce min_common_dates ONLY at lag 0; for other lags just require at least 2 points
        if lag == 0:
            if n_common < min_common_dates:
                # If lag-0 itself doesn't meet the requirement, we stop early for this well
                return {
                    'r_lag0': np.nan, 'p_lag0': np.nan,
                    'r_max': np.nan, 'p_max': np.nan, 'lag_max': np.nan
                }
        else:
            if n_common < 2:
                continue
        r, p = corr_func(ws[mask], gs[mask])
        
        # Store lag-0 correlation
        if lag == 0:
            r_lag0, p_lag0 = r, p
        
        # Track best (max) correlation across all lags (no additional per-lag threshold)
        if np.isnan(best_r) or r > best_r:
            best_r, best_p, best_lag = r, p, lag
    
    return {
        'r_lag0': r_lag0, 'p_lag0': p_lag0,
        'r_max': best_r, 'p_max': best_p, 'lag_max': best_lag
    }


def preprocess_groundwater_data(
    country_folder_path,
    value_column='Water Level [masl]',
    id_column='ID',
    lat_column='Latitude',
    lon_column='Longitude',
    date_column='Date and Time',
    save_log_path=None,
    min_measurements=50,
    start_date=None,
    end_date=None
):
    """
    Preprocess groundwater observation data from GGMN network.
    
    Reads well metadata and monitoring files from a country folder,
    processes time series data, and creates two dataframes:
    1. Well locations (lat/lon for each well ID)
    2. Time series (dates as rows, well IDs as columns)
    
    Parameters
    ----------
    country_folder_path : str
        Path to the country folder (e.g., 'SOM - GGMN')
    value_column : str, default='Water Level [masl]'
        Name of the column containing water level measurements
    id_column : str, default='ID'
        Name of the ID column in wells.ods and monitoring files
    lat_column : str, default='Latitude'
        Name of the latitude column in wells.ods
    lon_column : str, default='Longitude'
        Name of the longitude column in wells.ods
    date_column : str, default='Date and Time'
        Name of the date column in monitoring files
    save_log_path : str, optional
        Path to save the processing log CSV
    min_measurements : int, default=50
        Minimum number of measurements required to keep a well. Wells with fewer
        measurements will be excluded from the final dataset. This is applied AFTER
        date filtering.
    start_date : str or datetime, optional
        Start date for filtering time series (inclusive). Only data from this date
        onwards will be kept. Applied before min_measurements filtering.
        If None, no start date filtering is applied.
    end_date : str or datetime, optional
        End date for filtering time series (inclusive). If None, uses the most recent
        date in the data. Applied before min_measurements filtering.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'well_locations': DataFrame with columns [ID, Latitude, Longitude, ...]
        - 'time_series': DataFrame with dates as index and well IDs as columns
        - 'processing_log': DataFrame with processing statistics
        - 'processing_errors': List of processing errors (if any)
    
    Raises
    ------
    FileNotFoundError
        If wells.ods or monitoring folder is not found
    ValueError
        If required columns are missing
    """
    # Convert to Path object for easier manipulation
    country_path = Path(country_folder_path)
    
    if not country_path.exists():
        raise FileNotFoundError(f"Country folder not found: {country_folder_path}")
    
    # Paths to key files/folders
    wells_file = country_path / 'wells.ods'
    monitoring_folder = country_path / 'monitoring'
    
    if not wells_file.exists():
        raise FileNotFoundError(f"wells.ods not found in {country_folder_path}")
    
    if not monitoring_folder.exists():
        raise FileNotFoundError(f"monitoring folder not found in {country_folder_path}")
    
    print(f"Processing data from: {country_folder_path}")
    print(f"Reading well metadata from: {wells_file}")
    
    # ============================================================================
    # STEP 1: Read well metadata (locations)
    # ============================================================================
    # Initialize logging dictionary
    country_name = country_path.name  # Store country name for all log entries
    log_data = {
        'Country': [],
        'Step': [],
        'Description': [],
        'Count': []
    }
    
    # Read the first sheet (well data)
    # ODS files can be read with pandas using engine='odf'
    # Structure: Row 1 = header, Row 2 = description, Row 3+ = data
    try:
        # Read with header in row 0 (first row)
        wells_df_raw = pd.read_excel(wells_file, engine='odf', sheet_name=0, header=0)
        
        # Extract description row (row 1, index 1) - second row after header for quality flag column names
        description_row = None
        if len(wells_df_raw) > 0:
            description_row = wells_df_raw.iloc[0].copy()
            # Skip the description row from data
            # Check if first row after header looks like description (all non-numeric or same value)
            first_data_row = wells_df_raw.iloc[0]
            # Skip if all columns are strings or if all values are the same (likely description)
            if first_data_row.dtype == 'object' or first_data_row.nunique() <= 1:
                wells_df = wells_df_raw.iloc[1:].reset_index(drop=True)
            else:
                wells_df = wells_df_raw
        else:
            wells_df = wells_df_raw
            
    except Exception as e:
        raise ValueError(f"Error reading wells.ods: {e}. Make sure 'odfpy' is installed: pip install odfpy")
    
    # Clean column names (remove extra whitespace)
    wells_df.columns = wells_df.columns.str.strip()
    
    # Find quality flag columns from description row; categorize as exclude vs clean
    # We do NOT exclude wells for "gap of more than 3 years"—correlation already requires
    # min_common_dates with GRACE, and late-start wells can still have good overlap.
    # clean: jump±50m, suspicious (remove values + interpolate)
    quality_flag_cols = []
    quality_flag_exclude_patterns = []  # was: ['gap of more than 3 years']; removed so overlap is decided at correlation
    quality_flag_clean_patterns = ['jump of +50 or -50 m', 'suspicious groundwater level value']
    flag_col_to_type = {}  # col_name -> 'exclude' or 'clean'
    
    if description_row is not None:
        description_row = description_row.astype(str).str.strip()
        for col_name in wells_df.columns:
            if col_name in description_row.index:
                desc_value = str(description_row[col_name]).lower()
                for pattern in quality_flag_exclude_patterns:
                    if pattern.lower() in desc_value:
                        quality_flag_cols.append(col_name)
                        flag_col_to_type[col_name] = 'exclude'
                        break
                else:
                    for pattern in quality_flag_clean_patterns:
                        if pattern.lower() in desc_value:
                            quality_flag_cols.append(col_name)
                            flag_col_to_type[col_name] = 'clean'
                            break
    
    total_wells_in_file = len(wells_df)
    log_data['Country'].append(country_name)
    log_data['Step'].append('1')
    log_data['Description'].append('Total rows in wells.ods (after skipping description row)')
    log_data['Count'].append(total_wells_in_file)
    
    # Check for required columns
    required_cols = [id_column, lat_column, lon_column]
    missing_cols = [col for col in required_cols if col not in wells_df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns in wells.ods: {missing_cols}. "
                        f"Available columns: {list(wells_df.columns)}")
    
    # Look for ground surface elevation column (might be merged column with description)
    # Try common names: "Ground surface elevation", "Ground Surface Elevation", "Elevation", etc.
    ground_elev_col = None
    ground_elev_unit_col = None
    
    for col in wells_df.columns:
        col_lower = col.lower().strip()
        if 'ground' in col_lower and ('surface' in col_lower or 'elevation' in col_lower):
            ground_elev_col = col
            # Check if there's a corresponding unit column
            for unit_col in wells_df.columns:
                if 'unit' in unit_col.lower() and (ground_elev_col in unit_col or unit_col in ground_elev_col):
                    ground_elev_unit_col = unit_col
                    break
            break
        elif 'elevation' in col_lower and ground_elev_col is None:
            ground_elev_col = col
    
    # If not found, check all columns
    if ground_elev_col is None:
        print(f"Warning: Ground surface elevation column not found. Available columns: {list(wells_df.columns)}")
        print("Will set ground_surface_elevation_m to NaN for all wells.")
    
    # Select and clean well location data
    base_cols = [id_column, lat_column, lon_column]
    if ground_elev_col:
        base_cols.append(ground_elev_col)
        if ground_elev_unit_col:
            base_cols.append(ground_elev_unit_col)
    
    well_locations = wells_df[base_cols].copy()
    
    # Rename columns
    col_mapping = {
        id_column: 'ID',
        lat_column: 'Latitude',
        lon_column: 'Longitude'
    }
    if ground_elev_col:
        col_mapping[ground_elev_col] = 'Ground_Surface_Elevation'
    if ground_elev_unit_col:
        col_mapping[ground_elev_unit_col] = 'Ground_Elevation_Unit'
    
    well_locations.rename(columns=col_mapping, inplace=True)
    
    # Convert ID to string for consistency
    well_locations['ID'] = well_locations['ID'].astype(str).str.strip()
    
    # Remove duplicates (keep first occurrence)
    n_duplicates = len(well_locations) - len(well_locations.drop_duplicates(subset=['ID'], keep='first'))
    well_locations = well_locations.drop_duplicates(subset=['ID'], keep='first')
    
    if n_duplicates > 0:
        log_data['Country'].append(country_name)
        log_data['Step'].append('1a')
        log_data['Description'].append(f'Duplicate well IDs removed')
        log_data['Count'].append(n_duplicates)
    
    # Process ground surface elevation: convert to numeric and handle units
    if 'Ground_Surface_Elevation' in well_locations.columns:
        # Convert to numeric
        well_locations['Ground_Surface_Elevation'] = pd.to_numeric(
            well_locations['Ground_Surface_Elevation'], errors='coerce'
        )
        
        # Handle unit conversion (ft to m)
        if 'Ground_Elevation_Unit' in well_locations.columns:
            # Convert ft to m (1 ft = 0.3048 m)
            ft_mask = well_locations['Ground_Elevation_Unit'].str.strip().str.lower() == 'ft'
            well_locations.loc[ft_mask, 'Ground_Surface_Elevation'] = (
                well_locations.loc[ft_mask, 'Ground_Surface_Elevation'] * 0.3048
            )
        
        # Rename to indicate it's in meters
        well_locations.rename(columns={'Ground_Surface_Elevation': 'ground_surface_elevation_m'}, inplace=True)
        well_locations.drop(columns=['Ground_Elevation_Unit'], errors='ignore', inplace=True)
    else:
        # Add empty column if not found
        well_locations['ground_surface_elevation_m'] = np.nan
    
    # Remove rows with missing coordinates (NaN)
    n_missing_coords = len(well_locations[well_locations[['Latitude', 'Longitude']].isna().any(axis=1)])
    well_locations = well_locations.dropna(subset=['Latitude', 'Longitude'])
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('1b')
    log_data['Description'].append('Wells with missing coordinates (NaN) removed')
    log_data['Count'].append(n_missing_coords)
    
    # Check that coordinates are numeric and within valid ranges
    # Convert to numeric, coercing errors to NaN
    well_locations['Latitude'] = pd.to_numeric(well_locations['Latitude'], errors='coerce')
    well_locations['Longitude'] = pd.to_numeric(well_locations['Longitude'], errors='coerce')
    
    # Remove rows where conversion failed or values are out of range
    n_invalid_coords = len(well_locations[
        (well_locations['Latitude'].isna()) | 
        (well_locations['Longitude'].isna()) |
        (well_locations['Latitude'] < -90) | (well_locations['Latitude'] > 90) |
        (well_locations['Longitude'] < -180) | (well_locations['Longitude'] > 180)
    ])
    
    well_locations = well_locations[
        (well_locations['Latitude'].notna()) & 
        (well_locations['Longitude'].notna()) &
        (well_locations['Latitude'] >= -90) & (well_locations['Latitude'] <= 90) &
        (well_locations['Longitude'] >= -180) & (well_locations['Longitude'] <= 180)
    ]
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('1c')
    log_data['Description'].append('Wells with invalid coordinate values (non-numeric or out of range) removed')
    log_data['Count'].append(n_invalid_coords)
    
    wells_with_valid_coords = len(well_locations)
    log_data['Country'].append(country_name)
    log_data['Step'].append('1d')
    log_data['Description'].append('Wells with valid coordinates (final count from metadata)')
    log_data['Count'].append(wells_with_valid_coords)
    
    # ============================================================================
    # STEP 1e: Filter wells based on data quality flags
    # Exclude: gap>3yr only. Clean: jump±50m, suspicious (keep well, fix values later)
    # ============================================================================
    wells_to_clean = set()
    if quality_flag_cols:
        wells_df['ID'] = wells_df[id_column].astype(str).str.strip()
        
        wells_to_exclude = set()
        n_excluded_by_flag = {}
        n_to_clean_by_flag = {}
        
        for flag_col in quality_flag_cols:
            if flag_col not in wells_df.columns:
                continue
            flag_type = flag_col_to_type.get(flag_col, 'exclude')
            flag_values_normalized = wells_df[flag_col].astype(str).str.strip().str.lower()
            flagged_wells = set(wells_df[flag_values_normalized == 'yes']['ID'].tolist())
            
            if flag_type == 'exclude':
                wells_to_exclude.update(flagged_wells)
                n_excluded_by_flag[flag_col] = len(flagged_wells)
            else:
                wells_to_clean.update(flagged_wells)
                n_to_clean_by_flag[flag_col] = len(flagged_wells)
        
        # Remove from wells_to_clean any that are excluded (gap takes precedence)
        wells_to_clean -= wells_to_exclude
        
        # Filter well_locations to exclude only gap-flagged wells
        wells_before_quality_filter = len(well_locations)
        well_locations = well_locations[~well_locations['ID'].isin(wells_to_exclude)]
        wells_after_quality_filter = len(well_locations)
        n_excluded_total = wells_before_quality_filter - wells_after_quality_filter
        
        if n_excluded_total > 0:
            log_data['Country'].append(country_name)
            log_data['Step'].append('1e')
            log_data['Description'].append('Wells excluded due to gap>3yr flag (cannot fix)')
            log_data['Count'].append(n_excluded_total)
            for flag_col, count in n_excluded_by_flag.items():
                if count > 0:
                    log_data['Country'].append(country_name)
                    log_data['Step'].append(f'1e_{flag_col[:10]}')
                    log_data['Description'].append(f'  - Excluded by "{flag_col}" flag')
                    log_data['Count'].append(count)
            print(f"  Excluded {n_excluded_total} wells due to gap>3yr flag")
        
        if wells_to_clean:
            print(f"  {len(wells_to_clean)} wells with jump/suspicious flags will be cleaned (remove bad values + interpolate)")
    else:
        print(f"  Warning: Quality flag columns not found in wells.ods. Skipping quality flag filtering.")
        print(f"  Available columns: {list(wells_df.columns)}")
        if description_row is not None:
            print(f"  Description row sample: {description_row.head().to_dict()}")
    
    print(f"Found {wells_with_valid_coords} wells with valid coordinates (from {total_wells_in_file} total rows)")
    
    # Get list of well IDs from metadata
    well_ids_from_metadata = set(well_locations['ID'].values)
    
    # ============================================================================
    # STEP 2: Read monitoring files
    # ============================================================================
    tqdm, use_tqdm = _import_tqdm()
    
    # Find all ODS files in monitoring folder (recursively, including subfolders)
    monitoring_files = sorted(monitoring_folder.rglob("*.ods"))
    monitoring_files = [str(f) for f in monitoring_files]  # Convert Path objects to strings
    
    total_monitoring_files = len(monitoring_files)
    log_data['Country'].append(country_name)
    log_data['Step'].append('2')
    log_data['Description'].append('Total monitoring files found')
    log_data['Count'].append(total_monitoring_files)
    
    if not monitoring_files:
        warnings.warn(f"No ODS files found in {monitoring_folder} (searched recursively in all subfolders)")
    
    print(f"Found {total_monitoring_files} monitoring files")
    
    # Extract well IDs from filenames (filename without .ods extension = well ID)
    monitoring_file_ids = {}
    for file_path in monitoring_files:
        file_name = Path(file_path).stem  # Get filename without extension (e.g., "10000078" from "10000078.ods")
        monitoring_file_ids[file_name] = file_path
    
    # Match monitoring files to wells in metadata
    matched_file_ids = set(monitoring_file_ids.keys()) & well_ids_from_metadata
    unmatched_file_ids = set(monitoring_file_ids.keys()) - well_ids_from_metadata
    
    n_matched = len(matched_file_ids)
    n_unmatched = len(unmatched_file_ids)
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('2a')
    log_data['Description'].append('Monitoring files matched to well IDs in metadata')
    log_data['Count'].append(n_matched)
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('2b')
    log_data['Description'].append('Monitoring files NOT matched to well IDs in metadata (will be skipped)')
    log_data['Count'].append(n_unmatched)
    
    # Add unmatched well IDs to log file
    if n_unmatched > 0:
        # Print warning without listing all IDs
        print(f"Warning: {n_unmatched} monitoring files do not match any well ID in metadata (see log file for details)")
        
        # Add unmatched IDs to log as comma-separated list in Description
        unmatched_ids_str = ', '.join(sorted(unmatched_file_ids))
        log_data['Country'].append(country_name)
        log_data['Step'].append('2b_ids')
        log_data['Description'].append(f'Unmatched monitoring file IDs: {unmatched_ids_str}')
        log_data['Count'].append(n_unmatched)
    
    # Dictionary to store time series data for each well
    well_time_series = {}
    # Dictionary to store Parameter and Unit for each well
    well_parameter_info = {}
    processing_errors = []
    
    # Only process files that match well IDs in metadata
    # Use progress bar if available (tqdm already imported above)
    matched_file_ids_list = list(matched_file_ids)
    if use_tqdm:
        iterator = tqdm(matched_file_ids_list, 
                       desc=f"Reading {n_matched} monitoring files",
                       unit="file",
                       leave=True,  # Keep progress bar visible after completion
                       dynamic_ncols=True)  # Adjust to terminal width
    else:
        print(f"Processing {n_matched} monitoring files that match well IDs in metadata")
        iterator = matched_file_ids_list
    
    for well_id in iterator:
        file_path = monitoring_file_ids[well_id]
        file_name = Path(file_path).stem
        
        try:
            # Read ODS file
            # Structure: Row 1 = header, Row 2 = description, Row 3+ = data
            # Read with header in row 0 (first row)
            df_raw = pd.read_excel(file_path, engine='odf', header=0)
            
            # Clean column names
            df_raw.columns = df_raw.columns.str.strip()
            
            # Remove the description row (row 1, index 1) - second row after header
            if len(df_raw) > 0:
                # Check if first data row looks like a description (all non-numeric or same value)
                first_data_row = df_raw.iloc[0]
                if first_data_row.dtype == 'object' or first_data_row.nunique() <= 1:
                    df = df_raw.iloc[1:].reset_index(drop=True)
                else:
                    df = df_raw
            else:
                df = df_raw
            
            # Check for required columns
            if date_column not in df.columns:
                error_msg = f"Date column '{date_column}' not found in {file_name}. Available columns: {list(df.columns)}"
                warnings.warn(error_msg)
                processing_errors.append({'well_id': well_id, 'file': file_name, 'error': error_msg})
                continue
            
            # Extract Parameter and Unit information (should be consistent per well)
            parameter_type = None
            unit_type = None
            
            if 'Parameter' in df.columns:
                # Get the most common parameter value (should be consistent)
                parameter_values = df['Parameter'].dropna().unique()
                if len(parameter_values) > 0:
                    parameter_type = parameter_values[0]  # Use first non-null value
                    # If multiple values, take the most common
                    if len(parameter_values) > 1:
                        parameter_counts = df['Parameter'].value_counts()
                        parameter_type = parameter_counts.index[0]
            
            if 'Unit' in df.columns:
                # Get the most common unit value (should be consistent)
                unit_values = df['Unit'].dropna().str.strip().str.lower().unique()
                if len(unit_values) > 0:
                    unit_type = unit_values[0]  # Use first non-null value (lowercase)
                    # If multiple values, take the most common
                    if len(unit_values) > 1:
                        unit_counts = df['Unit'].str.strip().str.lower().value_counts()
                        unit_type = unit_counts.index[0]
            
            # Store parameter info for this well
            well_parameter_info[well_id] = {
                'parameter_type': parameter_type,
                'unit_type': unit_type
            }
            
            if value_column not in df.columns:
                # Try to find a column with 'water' or 'level' in the name (case insensitive)
                value_col_candidates = [col for col in df.columns 
                                       if 'water' in col.lower() or 'level' in col.lower()]
                if value_col_candidates:
                    value_column_actual = value_col_candidates[0]
                    warnings.warn(f"Using '{value_column_actual}' instead of '{value_column}' for {file_name}")
                else:
                    error_msg = f"Value column '{value_column}' not found in {file_name}. Available columns: {list(df.columns)}"
                    warnings.warn(error_msg)
                    processing_errors.append({'well_id': well_id, 'file': file_name, 'error': error_msg})
                    continue
            else:
                value_column_actual = value_column
            
            # Extract date and value columns
            df_subset = df[[date_column, value_column_actual]].copy()
            
            # Convert date column to datetime
            # Handle format "2023-05-04 00:00:00 UTC"
            df_subset[date_column] = pd.to_datetime(df_subset[date_column], 
                                                     errors='coerce',
                                                     utc=True)
            
            # Remove rows with invalid dates
            df_subset = df_subset.dropna(subset=[date_column])
            
            if len(df_subset) == 0:
                error_msg = f"No valid dates found in {file_name}"
                warnings.warn(error_msg)
                processing_errors.append({'well_id': well_id, 'file': file_name, 'error': error_msg})
                continue
            
            # Convert value column to numeric
            df_subset[value_column_actual] = pd.to_numeric(df_subset[value_column_actual], errors='coerce')
            
            # Convert units: ft to m if needed
            if unit_type and unit_type.lower() == 'ft':
                # Convert feet to meters (1 ft = 0.3048 m)
                df_subset[value_column_actual] = df_subset[value_column_actual] * 0.3048
            elif unit_type and unit_type.lower() not in ['m', 'ft']:
                warnings.warn(f"Unknown unit '{unit_type}' for well {well_id}. Keeping values as-is.")
            
            # Remove rows with invalid values
            df_subset = df_subset.dropna(subset=[value_column_actual])
            
            if len(df_subset) == 0:
                error_msg = f"No valid values found in {file_name}"
                warnings.warn(error_msg)
                processing_errors.append({'well_id': well_id, 'file': file_name, 'error': error_msg})
                continue
            
            # Rename columns for consistency
            df_subset.columns = ['Date', 'Value']
            
            # Store in dictionary with well ID as key
            well_time_series[well_id] = df_subset
            
        except Exception as e:
            error_msg = f"Error processing {file_name}: {e}"
            warnings.warn(error_msg)
            processing_errors.append({'well_id': well_id, 'file': file_name, 'error': str(e)})
            continue
    
    n_successfully_processed = len(well_time_series)
    n_failed_processing = n_matched - n_successfully_processed
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('2c')
    log_data['Description'].append('Monitoring files successfully processed')
    log_data['Count'].append(n_successfully_processed)
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('2d')
    log_data['Description'].append('Monitoring files that failed processing')
    log_data['Count'].append(n_failed_processing)
    
    print(f"Successfully processed {n_successfully_processed} monitoring files")
    
    if len(well_time_series) == 0:
        if processing_errors:
            error_summary = "\n".join([f"  - {err['file']}: {err['error']}" for err in processing_errors[:5]])
            raise ValueError(f"No valid monitoring data found. Processing errors:\n{error_summary}")
        else:
            raise ValueError("No valid monitoring data found. Please check file formats and column names.")
    
    # ============================================================================
    # STEP 3: Aggregate to monthly data (last day of month, average multiple measurements)
    # ============================================================================
    monthly_data = {}
    
    # Use progress bar for monthly aggregation if available
    iterator = tqdm(well_time_series.items(), desc="  Aggregating to monthly") if use_tqdm else well_time_series.items()
    
    for well_id, df in iterator:
        # Set date as index
        df_ts = df.set_index('Date')
        
        # Resample to monthly, using last day of month, averaging values
        # 'M' = month end frequency
        monthly_df = df_ts.resample('M').mean()
        
        # Use last day of month as index (pd.offsets.MonthEnd(0))
        monthly_df.index = monthly_df.index + pd.offsets.MonthEnd(0)
        
        monthly_data[well_id] = monthly_df['Value']
    
    # ============================================================================
    # STEP 3b: Clean wells with jump/suspicious flags (remove bad values + interpolate)
    # ============================================================================
    if wells_to_clean:
        total_jumps_removed = 0
        total_suspicious_removed = 0
        total_zeros_removed = 0
        wells_cleaned_count = 0
        for well_id in wells_to_clean:
            if well_id not in monthly_data:
                continue
            s = monthly_data[well_id]
            s_clean, n_jumps, n_suspicious, n_zeros = _clean_flagged_well_series(
                s, jump_threshold_m=50.0, suspicious_threshold_m=50.0, interp_limit_months=3
            )
            monthly_data[well_id] = s_clean
            if n_jumps > 0 or n_suspicious > 0 or n_zeros > 0:
                wells_cleaned_count += 1
                total_jumps_removed += n_jumps
                total_suspicious_removed += n_suspicious
                total_zeros_removed += n_zeros
        if wells_cleaned_count > 0:
            log_data['Country'].append(country_name)
            log_data['Step'].append('3b')
            log_data['Description'].append('Wells cleaned (jump/suspicious/zeros removed + interpolated, limit=3 months)')
            log_data['Count'].append(wells_cleaned_count)
            parts = [p for p in [
                f"{total_jumps_removed} jumps" if total_jumps_removed else None,
                f"{total_suspicious_removed} suspicious" if total_suspicious_removed else None,
                f"{total_zeros_removed} zeros" if total_zeros_removed else None,
            ] if p]
            print(f"  Cleaned {wells_cleaned_count} wells: {' + '.join(parts)} removed, gaps interpolated (max 3 months)")
    
    # ============================================================================
    # STEP 4: Create unified time series DataFrame
    # ============================================================================
    # Combine all monthly series into one dataframe
    time_series_df = pd.DataFrame(monthly_data)
    
    # Ensure column names (well IDs) are strings to match well_locations['ID']
    # This prevents type mismatch issues when matching IDs later
    time_series_df.columns = time_series_df.columns.astype(str)
    
    # Sort by date
    time_series_df = time_series_df.sort_index()
    
    # ============================================================================
    # STEP 5: Apply date filtering (before min_measurements filtering)
    # ============================================================================
    if start_date is not None or end_date is not None:
        # Convert dates to datetime if string
        is_tz_aware = time_series_df.index.tz is not None
        
        if start_date is None:
            start_date_dt = time_series_df.index.min()
        elif isinstance(start_date, str):
            start_date_dt = pd.to_datetime(start_date)
        else:
            start_date_dt = start_date
        if is_tz_aware and start_date_dt.tz is None:
            start_date_dt = start_date_dt.tz_localize('UTC')
        
        if end_date is None:
            end_date_dt = time_series_df.index.max()
        elif isinstance(end_date, str):
            end_date_dt = pd.to_datetime(end_date)
            if is_tz_aware and end_date_dt.tz is None:
                end_date_dt = end_date_dt.tz_localize('UTC')
        else:
            end_date_dt = end_date
            if is_tz_aware and end_date_dt.tz is None:
                end_date_dt = end_date_dt.tz_localize('UTC')
        
        # Filter time series to date range
        time_series_df = time_series_df.loc[
            (time_series_df.index >= start_date_dt) & (time_series_df.index <= end_date_dt)
        ].copy()
        
        log_data['Country'].append(country_name)
        log_data['Step'].append('4')
        log_data['Description'].append(f'Time series records after date filter ({start_date_dt.date()} to {end_date_dt.date()})')
        log_data['Count'].append(len(time_series_df))
    
    # ============================================================================
    # STEP 6: Filter by minimum number of measurements (after date filtering)
    # ============================================================================
    if min_measurements > 0 and len(time_series_df) > 0:
        wells_to_keep = []
        wells_excluded = []
        
        for well_id in time_series_df.columns:
            # Count non-NaN values in date-filtered series
            n_measurements = time_series_df[well_id].notna().sum()
            if n_measurements >= min_measurements:
                wells_to_keep.append(well_id)
            else:
                wells_excluded.append(well_id)
        
        # Filter time_series_df
        time_series_df = time_series_df[wells_to_keep]
        
        log_data['Country'].append(country_name)
        log_data['Step'].append('5')
        log_data['Description'].append(f'Wells with >= {min_measurements} measurements (after date filtering)')
        log_data['Count'].append(len(wells_to_keep))
        
        log_data['Country'].append(country_name)
        log_data['Step'].append('5a')
        log_data['Description'].append(f'Wells excluded (< {min_measurements} measurements)')
        log_data['Count'].append(len(wells_excluded))
        
        if len(wells_excluded) > 0:
            print(f"  Filtered out {len(wells_excluded)} wells with < {min_measurements} measurements")
    else:
        wells_to_keep = list(time_series_df.columns) if len(time_series_df) > 0 else []
        log_data['Country'].append(country_name)
        log_data['Step'].append('5')
        log_data['Description'].append('Wells in time series (no minimum filter applied)')
        log_data['Count'].append(len(wells_to_keep))
    
    # ============================================================================
    # STEP 7: Final alignment of well locations and time series
    # ============================================================================
    
    # Ensure we only include wells that have both location and time series data
    wells_with_data = set(well_locations['ID'].values) & set(time_series_df.columns)
    wells_missing_ts = set(well_locations['ID'].values) - set(time_series_df.columns)
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('6')
    log_data['Description'].append('Wells with both location and time series data (final dataset)')
    log_data['Count'].append(len(wells_with_data))
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('6a')
    log_data['Description'].append('Wells with location data but missing time series')
    log_data['Count'].append(len(wells_missing_ts))
    
    if len(wells_with_data) == 0:
        warnings.warn("No wells found with both location and time series data")
    
    # Filter to only include wells with both location and time series
    well_locations = well_locations[well_locations['ID'].isin(wells_with_data)]
    time_series_df = time_series_df[[col for col in time_series_df.columns if col in wells_with_data]]
    
    # Add Parameter and Unit information to well_locations
    well_locations['parameter_type'] = well_locations['ID'].map(
        lambda x: well_parameter_info.get(x, {}).get('parameter_type', None)
    )
    well_locations['unit'] = well_locations['ID'].map(
        lambda x: well_parameter_info.get(x, {}).get('unit_type', None)
    )
    # Note: values in time_series are already converted to meters, so we mark original unit
    # For display/logging purposes, we'll keep 'unit' as the original unit before conversion
    
    # Sort well_locations by ID for consistency
    well_locations = well_locations.sort_values('ID').reset_index(drop=True)
    
    final_well_count = len(well_locations)
    
    # Calculate water level range for summary
    if len(time_series_df) > 0 and len(time_series_df.columns) > 0:
        all_values = time_series_df.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        min_water_level = np.nanmin(all_values) if len(all_values) > 0 else np.nan
        max_water_level = np.nanmax(all_values) if len(all_values) > 0 else np.nan
        
        # Get date range safely
        min_date = time_series_df.index.min()
        max_date = time_series_df.index.max()
        
        # Check if dates are valid (not NaN)
        if pd.notna(min_date) and pd.notna(max_date):
            date_range_str = f"{min_date.strftime('%Y-%m')} to {max_date.strftime('%Y-%m')}"
        else:
            date_range_str = "N/A"
        
        # Print simplified summary (detailed info is in log file)
        if not np.isnan(min_water_level) and not np.isnan(max_water_level):
            print(f"  Wells: {final_well_count} | "
                  f"Date range: {date_range_str} | "
                  f"Water levels: {min_water_level:.2f} to {max_water_level:.2f} m")
        else:
            print(f"  Wells: {final_well_count} | "
                  f"Date range: {date_range_str} | "
                  f"Water levels: N/A")
    else:
        # No time series data available
        print(f"  Wells: {final_well_count} | "
              f"Date range: N/A | "
              f"Water levels: N/A (all wells filtered out)")
    
    # ============================================================================
    # Create log summary DataFrame and save to CSV
    # ============================================================================
    log_df = pd.DataFrame(log_data)
    
    if save_log_path:
        log_df.to_csv(save_log_path, index=False)
    else:
        # Save to default location in country folder
        default_log_path = country_path / f"{country_path.name}_processing_log.csv"
        log_df.to_csv(default_log_path, index=False)
    
    # Detailed summary is saved to log file, not printed to console
    # (Unit and parameter statistics are included in the log file)
    
    # Count wells with ground surface elevation (for log only)
    n_with_elevation = well_locations['ground_surface_elevation_m'].notna().sum()
    n_without_elevation = well_locations['ground_surface_elevation_m'].isna().sum()
    
    # Add to log
    log_data['Country'].append(country_name)
    log_data['Step'].append('6')
    log_data['Description'].append('Wells with ground surface elevation data')
    log_data['Count'].append(n_with_elevation)
    
    log_data['Country'].append(country_name)
    log_data['Step'].append('6a')
    log_data['Description'].append('Wells without ground surface elevation data')
    log_data['Count'].append(n_without_elevation)
    
    # Recreate log_df with updated data
    log_df = pd.DataFrame(log_data)
    
    # ============================================================================
    # Return results
    # ============================================================================
    results = {
        'well_locations': well_locations,
        'time_series': time_series_df,
        'processing_log': log_df,
        'processing_errors': processing_errors if processing_errors else None
    }
    
    return results


def preprocess_all_countries(
    base_path,
    value_column='Value',
    id_column='ID',
    lat_column='Latitude',
    lon_column='Longitude',
    date_column='Date and Time',
    save_summary_log_path=None,
    skip_failed=True,
    min_measurements=50,
    start_date='2002-04-01',
    end_date=None
):
    """
    Process groundwater data for all country folders and merge results.
    
    Processes all country folders in the base path, adds country names,
    and merges all well locations and time series into unified dataframes.
    
    Parameters
    ----------
    base_path : str
        Path to the directory containing country folders (e.g., 'SOM - GGMN')
    value_column : str, default='Value'
        Name of the column containing water level measurements
    id_column : str, default='ID'
        Name of the ID column in wells.ods and monitoring files
    lat_column : str, default='Latitude'
        Name of the latitude column in wells.ods
    lon_column : str, default='Longitude'
        Name of the longitude column in wells.ods
    date_column : str, default='Date and Time'
        Name of the date column in monitoring files
    save_summary_log_path : str, optional
        Path to save the summary log of all countries
    skip_failed : bool, default=True
        If True, skip countries that fail processing and continue with others
    min_measurements : int, default=50
        Minimum number of measurements required to keep a well. Wells with fewer
        measurements will be excluded from the final dataset. This is applied AFTER
        date filtering.
    start_date : str or datetime, default='2002-04-01'
        Start date for filtering time series (inclusive). Only data from this date
        onwards will be kept. Applied before min_measurements filtering.
    end_date : str or datetime, optional
        End date for filtering time series (inclusive). If None, uses the most recent
        date in the data. Applied before min_measurements filtering.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'well_locations': Merged DataFrame with columns [ID, Country, Latitude, Longitude]
        - 'time_series': Merged DataFrame with dates as index and well IDs as columns
        - 'country_results': Dictionary with individual country results
        - 'summary_log': DataFrame summarizing processing for all countries
    """
    base_path_obj = Path(base_path)
    
    if not base_path_obj.exists():
        raise FileNotFoundError(f"Base path not found: {base_path}")
    
    # Find all country folders (format: "XXX - GGMN")
    country_folders = sorted([d for d in base_path_obj.iterdir() 
                             if d.is_dir() and ' - GGMN' in d.name])
    
    if not country_folders:
        raise ValueError(f"No country folders found in {base_path}")
    
    print(f"Found {len(country_folders)} country folders to process")
    print("="*70)
    
    tqdm, use_tqdm = _import_tqdm()
    
    # Storage for results
    all_well_locations = []
    all_time_series = []
    country_results = {}
    summary_log_data = []
    
    # Process each country with progress bar
    iterator = tqdm(country_folders, desc="Processing countries") if use_tqdm else country_folders
    
    for country_folder in iterator:
        country_code = country_folder.name.split(' - ')[0]  # Extract country code (e.g., "SOM" from "SOM - GGMN")
        
        print(f"\n{'='*70}")
        print(f"Processing: {country_folder.name} ({country_code})")
        print(f"{'='*70}")
        
        try:
            # Process this country (without min_measurements filter - will apply after date filtering)
            result = preprocess_groundwater_data(
                country_folder_path=str(country_folder),
                value_column=value_column,
                id_column=id_column,
                lat_column=lat_column,
                lon_column=lon_column,
                date_column=date_column,
                min_measurements=0  # Don't filter here - will filter after date filtering
            )
            
            # Add country column to well_locations
            well_locs = result['well_locations'].copy()
            well_locs.insert(1, 'Country', country_code)  # Insert after ID column
            
            # Store results
            all_well_locations.append(well_locs)
            all_time_series.append(result['time_series'])
            country_results[country_code] = result
            
            # Extract summary from processing log
            log_df = result['processing_log']
            
            # Get unit and parameter statistics
            unit_counts = well_locs['unit'].value_counts().to_dict() if 'unit' in well_locs.columns else {}
            param_counts = well_locs['parameter_type'].value_counts().to_dict() if 'parameter_type' in well_locs.columns else {}
            
            # Format unit statistics
            units_summary = []
            for unit, count in sorted(unit_counts.items(), key=lambda x: x[1], reverse=True):
                if unit:
                    units_summary.append(f"{unit}:{count}")
                else:
                    units_summary.append(f"unknown:{count}")
            units_str = ", ".join(units_summary) if units_summary else "unknown"
            
            # Format parameter statistics
            params_summary = []
            for param, count in sorted(param_counts.items(), key=lambda x: x[1], reverse=True)[:3]:  # Top 3
                param_display = str(param)[:40] if param else "unknown"  # Truncate long names
                params_summary.append(f"{param_display}:{count}")
            params_str = ", ".join(params_summary) if params_summary else "unknown"
            
            # Count wells with elevation data
            n_with_elevation = well_locs['ground_surface_elevation_m'].notna().sum() if 'ground_surface_elevation_m' in well_locs.columns else 0
            
            summary_log_data.append({
                'Country': country_code,
                'Folder_Name': country_folder.name,
                'Wells_with_Valid_Coordinates': int(log_df[log_df['Description'].str.contains('Wells with valid coordinates', case=False)]['Count'].values[0]) if len(log_df[log_df['Description'].str.contains('Wells with valid coordinates', case=False)]) > 0 else 0,
                'Monitoring_Files_Found': int(log_df[log_df['Description'].str.contains('Total monitoring files found', case=False)]['Count'].values[0]) if len(log_df[log_df['Description'].str.contains('Total monitoring files found', case=False)]) > 0 else 0,
                'Monitoring_Files_Matched': int(log_df[log_df['Description'].str.contains('Monitoring files matched', case=False)]['Count'].values[0]) if len(log_df[log_df['Description'].str.contains('Monitoring files matched', case=False)]) > 0 else 0,
                'Files_Successfully_Processed': int(log_df[log_df['Description'].str.contains('successfully processed', case=False)]['Count'].values[0]) if len(log_df[log_df['Description'].str.contains('successfully processed', case=False)]) > 0 else 0,
                'Final_Wells_Count': len(well_locs),
                'Wells_with_Elevation_Data': n_with_elevation,
                'Units_Used': units_str,
                'Parameter_Types': params_str,
                'Status': 'Success'
            })
            
            print(f"✅ Successfully processed {country_code}: {len(well_locs)} wells")
            
        except Exception as e:
            error_msg = str(e)
            print(f"❌ Failed to process {country_code}: {error_msg}")
            
            summary_log_data.append({
                'Country': country_code,
                'Folder_Name': country_folder.name,
                'Wells_with_Valid_Coordinates': 0,
                'Monitoring_Files_Found': 0,
                'Monitoring_Files_Matched': 0,
                'Files_Successfully_Processed': 0,
                'Final_Wells_Count': 0,
                'Status': f'Failed: {error_msg[:100]}'  # Truncate long error messages
            })
            
            if not skip_failed:
                raise
            continue
    
    print(f"\n{'='*70}")
    print("MERGING RESULTS")
    print(f"{'='*70}")
    
    # Merge well locations
    if all_well_locations:
        merged_well_locations = pd.concat(all_well_locations, ignore_index=True)
        merged_well_locations = merged_well_locations.sort_values(['Country', 'ID']).reset_index(drop=True)
        merged_well_locations['ID'] = merged_well_locations['ID'].astype(str).str.strip()
        
        # Always use unique_well_id (Country_ID) for all wells so time_series and wells match on one key
        merged_well_locations['unique_well_id'] = (
            merged_well_locations['Country'].astype(str) + '_' + merged_well_locations['ID'].astype(str)
        )
        
        print(f"Merged well locations: {len(merged_well_locations)} total wells")
        print(f"  Countries: {merged_well_locations['Country'].unique()}")
        print(f"  Using unique_well_id (Country_ID) for all wells")
    else:
        merged_well_locations = pd.DataFrame(columns=['ID', 'Country', 'Latitude', 'Longitude'])
        warnings.warn("No well locations to merge")
    
    # Merge time series
    if all_time_series:
        # Combine all time series dataframes
        # Find the full date range
        all_dates = set()
        for ts_df in all_time_series:
            all_dates.update(ts_df.index)
        
        # Create date range
        if all_dates:
            date_range = pd.date_range(
                start=min(all_dates),
                end=max(all_dates),
                freq='M'
            )
            
            # Reindex each dataframe to the full date range; use unique_well_id (Country_ID) as column names
            merged_time_series = pd.DataFrame(index=date_range)
            
            for idx, ts_df in enumerate(all_time_series):
                country_code = country_folders[idx].name.split(' - ')[0]
                ts_reindexed = ts_df.reindex(date_range)
                ts_reindexed.columns = [
                    f"{country_code}_{str(c).strip()}" for c in ts_reindexed.columns
                ]
                merged_time_series = pd.concat([merged_time_series, ts_reindexed], axis=1)
            
            # Sort by date
            merged_time_series = merged_time_series.sort_index()
            
            print(f"Merged time series: {len(merged_time_series)} monthly records")
            print(f"  Date range: {merged_time_series.index.min()} to {merged_time_series.index.max()}")
            print(f"  Total wells: {len(merged_time_series.columns)}")
            
            # ============================================================================
            # Apply date filtering (before min_measurements filtering)
            # ============================================================================
            print(f"\n{'='*70}")
            print("APPLYING DATE FILTER")
            print(f"{'='*70}")
            
            # Convert dates to datetime if string
            is_tz_aware = merged_time_series.index.tz is not None
            
            if isinstance(start_date, str):
                start_date_dt = pd.to_datetime(start_date)
            else:
                start_date_dt = start_date
            if is_tz_aware and start_date_dt.tz is None:
                start_date_dt = start_date_dt.tz_localize('UTC')
            
            if end_date is None:
                # Use most recent date in the data
                end_date_dt = merged_time_series.index.max()
            elif isinstance(end_date, str):
                end_date_dt = pd.to_datetime(end_date)
                if is_tz_aware and end_date_dt.tz is None:
                    end_date_dt = end_date_dt.tz_localize('UTC')
            else:
                end_date_dt = end_date
                if is_tz_aware and end_date_dt.tz is None:
                    end_date_dt = end_date_dt.tz_localize('UTC')
            
            print(f"Filtering to date range: {start_date_dt.date()} to {end_date_dt.date()}")
            
            # Filter time series to date range
            merged_time_series = merged_time_series.loc[
                (merged_time_series.index >= start_date_dt) & (merged_time_series.index <= end_date_dt)
            ].copy()
            
            n_records_before_date_filter = len(merged_time_series)
            print(f"  Records after date filter: {len(merged_time_series)}")
            
            if len(merged_time_series) == 0:
                warnings.warn(f"No data remaining after date filter ({start_date_dt} to {end_date_dt})")
            else:
                print(f"  Date range after filtering: {merged_time_series.index.min()} to {merged_time_series.index.max()}")
            
            # ============================================================================
            # Apply min_measurements filter AFTER date filtering
            # ============================================================================
            if min_measurements > 0 and len(merged_time_series) > 0:
                print(f"\n{'='*70}")
                print(f"APPLYING MIN_MEASUREMENTS FILTER (>= {min_measurements} measurements)")
                print(f"{'='*70}")
                
                wells_before_filter = len(merged_time_series.columns)
                wells_to_keep = []
                wells_excluded = []
                
                for well_id in merged_time_series.columns:
                    well_series = merged_time_series[well_id]
                    # If duplicate column names exist, well_series can be a DataFrame; .sum() then returns a Series
                    _count = well_series.notna().sum()
                    n_measurements = int(_count.sum()) if isinstance(_count, pd.Series) else int(_count)
                    
                    if n_measurements >= min_measurements:
                        wells_to_keep.append(well_id)
                    else:
                        wells_excluded.append({
                            'well_id': well_id,
                            'measurements': n_measurements
                        })
                
                # Filter time series
                merged_time_series = merged_time_series[wells_to_keep].copy()
                
                # Filter well locations by unique_well_id (time_series columns are unique_well_id)
                merged_well_locations = merged_well_locations[
                    merged_well_locations['unique_well_id'].isin(wells_to_keep)
                ].copy()
                
                print(f"  Wells before filter: {wells_before_filter}")
                print(f"  Wells after filter: {len(wells_to_keep)}")
                print(f"  Wells excluded: {len(wells_excluded)}")
        else:
            merged_time_series = pd.DataFrame()
            warnings.warn("No time series data to merge")
    else:
        merged_time_series = pd.DataFrame()
        warnings.warn("No time series to merge")
    
    # Create summary log
    summary_log_df = pd.DataFrame(summary_log_data)
    
    if save_summary_log_path:
        summary_log_df.to_csv(save_summary_log_path, index=False)
        print(f"\nSummary log saved to: {save_summary_log_path}")
    
    # Print simplified summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    # Calculate key statistics
    total_wells = len(merged_well_locations)
    
    if len(merged_time_series) > 0:
        min_date = merged_time_series.index.min()
        max_date = merged_time_series.index.max()
        # Calculate min/max water levels (ignoring NaN)
        all_values = merged_time_series.values.flatten()
        all_values = all_values[~np.isnan(all_values)]
        if len(all_values) > 0:
            min_water_level = np.nanmin(all_values)
            max_water_level = np.nanmax(all_values)
        else:
            min_water_level = np.nan
            max_water_level = np.nan
    else:
        min_date = None
        max_date = None
        min_water_level = np.nan
        max_water_level = np.nan
    
    print(f"Total number of wells: {total_wells}")
    if min_date and max_date:
        print(f"Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    if not np.isnan(min_water_level) and not np.isnan(max_water_level):
        print(f"Water level range: {min_water_level:.2f} to {max_water_level:.2f}")
    
    # ============================================================================
    # Add depth classification based on average measurement (Shallow ≤ threshold, Deep > threshold)
    # ============================================================================
    # Depth classification moved to correlate_wells_with_grace (with configurable depth_threshold)
    
    # ============================================================================
    # Return results
    # ============================================================================
    results = {
        'well_locations': merged_well_locations,
        'time_series': merged_time_series,
        'country_results': country_results,
        'summary_log': summary_log_df
    }
    
    return results


def classify_well_depths(well_locations, time_series, depth_threshold=50, verbose=True):
    """
    Classify wells by depth based on average measurement (before baseline removal).
    
    Creates new columns in well_locations:
    - 'depth_class': Shallow (≤threshold), Deep (>threshold)
    - 'avg_depth_m': Average depth from all measurements
    
    The threshold is 50m by default (Shallow ≤50m, Deep >50m).
    
    For wells with parameter_type indicating depth from ground surface,
    uses the average of all measurement values directly.
    
    For wells with parameter_type indicating elevation AMSL,
    calculates depth = ground_surface_elevation_m - average_measurement_value.
    
    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns [ID, parameter_type, ground_surface_elevation_m, ...]
    time_series : pd.DataFrame
        DataFrame with dates as index and well IDs as columns
    
    Returns
    -------
    pd.DataFrame
        well_locations DataFrame with added 'depth_class' and 'avg_depth_m' columns
    """
    well_locs = well_locations.copy()
    
    # Initialize columns
    well_locs['depth_class'] = None
    well_locs['avg_depth_m'] = np.nan
    
    # Track classification statistics
    classified_count = 0
    unclassified_count = 0
    
    for idx, row in well_locs.iterrows():
        # Use same key as time_series columns (unique_well_id when present, else ID)
        well_ts_key = _get_well_ts_key(row)
        
        # Check if well has time series data
        if well_ts_key not in time_series.columns:
            unclassified_count += 1
            continue
        
        # Get all non-NaN measurements
        well_ts = time_series[well_ts_key].dropna()
        if len(well_ts) == 0:
            unclassified_count += 1
            continue
        
        # Calculate average measurement (before any baseline removal)
        avg_measurement = well_ts.mean()
        
        # Get parameter type
        parameter_type = row.get('parameter_type', None)
        
        # Calculate depth based on parameter type
        depth_m = np.nan
        
        if parameter_type:
            param_lower = str(parameter_type).lower()
            
            # Check if it's depth from ground surface
            if 'depth' in param_lower and ('ground' in param_lower or 'well' in param_lower):
                # Depth is measured from ground surface, so average measurement IS the depth
                depth_m = avg_measurement
            
            # Check if it's elevation AMSL
            elif 'elevation' in param_lower and ('a.m.s.l' in param_lower or 'amsl' in param_lower):
                # Need to calculate depth from elevation
                ground_surface_elevation = row.get('ground_surface_elevation_m', None)
                
                if pd.notna(ground_surface_elevation) and pd.notna(avg_measurement):
                    # Depth = ground surface elevation - water level elevation
                    depth_m = ground_surface_elevation - avg_measurement
                else:
                    # Missing elevation data, cannot calculate depth
                    unclassified_count += 1
                    continue
        
        # Classify depth (binary: Shallow ≤ threshold, Deep > threshold)
        if pd.notna(depth_m):
            well_locs.at[idx, 'avg_depth_m'] = depth_m
            if depth_m <= depth_threshold:
                well_locs.at[idx, 'depth_class'] = 'Shallow'
            else:
                well_locs.at[idx, 'depth_class'] = 'Deep'
            classified_count += 1
        else:
            unclassified_count += 1
    
    if verbose:
        shallow_count = (well_locs['depth_class'] == 'Shallow').sum()
        deep_count = (well_locs['depth_class'] == 'Deep').sum()
        print(f"\nDepth Classification (threshold = {depth_threshold}m):")
        print(f"  Shallow (≤{depth_threshold}m): {shallow_count} wells")
        print(f"  Deep (>{depth_threshold}m): {deep_count} wells")
        print(f"  Unclassified: {unclassified_count} wells")
    
    return well_locs


def _nice_lonlat_grid_step_deg(lon0, lon1, lat0, lat1, target_lines=6):
    """
    Degree spacing so small map extents still show several lon/lat grid lines.
    Chooses the smallest 'nice' step >= span/target_lines from a fixed menu.
    """
    span_lon = max(float(lon1 - lon0), 1e-9)
    span_lat = max(float(lat1 - lat0), 1e-9)
    span = max(span_lon, span_lat)
    raw = span / max(int(target_lines), 1)
    for step in (0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 15.0, 20.0):
        if step >= raw * 0.85:
            return float(step)
    return 20.0


def plot_all_well_locations(
    well_locations,
    figsize=(15, 10),
    save_path=None,
    grace_mean=None,
    aoi_geometry=None,
    clip_wells_to_aoi=True,
    country=None,
    draw_haversine=True,
    show_geo_grid=True,
):
    """
    Plot all well locations on a map with data-based extent.
    
    **GRACE “pixel centers” on this map** (`grace_lat` / `grace_lon` on each well):
    These come from ``correlate_wells_with_grace``, which assigns each well to the GRACE
    **grid node** (a ``(lat, lon)`` pair taken from ``grace_mean.lat`` and
    ``grace_mean.lon``) with **minimum great-circle (haversine) distance** to the well.
    It is **not** “nearest latitude then nearest longitude” as two separate 1-D searches;
    it scans the full 2-D grid of candidate pixels and picks the closest node.
    The gray dashed lines (when ``grace_mean`` is passed) are drawn **at those same 1-D
    coordinate values**, so they pass through cell **centers** on a typical rectilinear
    GRACE grid. If another plot uses ``pcolormesh`` with **edges** built as midpoints
    between centers, filled tiles can look half a cell offset from these lines even when
    the underlying assignment uses the same centers—compare using the same edge rule.
    
    Parameters
    ----------
    well_locations : pd.DataFrame or geopandas.GeoDataFrame
        DataFrame with columns [ID, Country, Latitude, Longitude]
        Optionally can have 'grace_lat' and 'grace_lon' columns
    figsize : tuple, default=(15, 10)
        Figure size
    save_path : str, optional
        Path to save the plot
    grace_mean : xarray.DataArray, optional
        GRACE data to plot grid lines. If provided, will overlay GRACE grid.
    aoi_geometry : geopandas.GeoDataFrame, GeoSeries, or geometry, optional
        Area-of-interest boundary to overlay as an outline (e.g. arid-region mask).
        When ``clip_wells_to_aoi`` is True, wells are restricted to points inside this geometry.
    clip_wells_to_aoi : bool, default=True
        If True and ``aoi_geometry`` is set, plot only wells whose coordinates fall inside the AOI.
        Set False to draw the AOI outline but keep all wells (e.g. for comparison).
    country : str, optional
        If set, keep only wells whose ``Country`` matches (case-insensitive, stripped).
        Map extent follows the bounding box of those wells (plus padding). Lon/lat grid labels
        are drawn every 5° instead of the default sparse tick lists.
    draw_haversine : bool, default=True
        If True and GRACE-assigned coordinates are present (``grace_lat``, ``grace_lon``),
        draw both GRACE pixel centers (``x`` markers) and dotted connection lines
        from wells to assigned GRACE pixels. Set False to hide both overlays.
    show_geo_grid : bool, default=True
        If True, draw labeled lon/lat grid lines on the map (Cartopy ``gridlines`` at
        5° when ``country`` is set, else the built-in fixed tick lists). When
        ``grace_mean`` is provided, fine dashed lines at native GRACE coordinates are
        still drawn; enable this to also show the coarser geographic grid.

    Returns
    -------
    dict
        Dictionary with 'figure' and 'axis' keys
    """
    # Cartopy is an optional dependency: only needed for map plotting
    ccrs, cfeature = _require_cartopy(
        "Cartopy is required for `plot_all_well_locations`, but it is not installed in this environment. "
        "Install it (e.g., `conda install -c conda-forge cartopy`) or skip plotting."
    )

    if len(well_locations) == 0:
        warnings.warn("No well locations to plot")
        return None

    import geopandas as gpd
    from shapely.ops import unary_union

    aoi_gdf = None
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs)
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    plot_wells = well_locations
    if aoi_gdf is not None and clip_wells_to_aoi:
        wells_gdf = gpd.GeoDataFrame(
            well_locations,
            geometry=gpd.points_from_xy(
                well_locations["Longitude"], well_locations["Latitude"]
            ),
            crs="EPSG:4326",
        )
        aoi_union = unary_union(aoi_gdf.geometry.tolist())
        mask = wells_gdf.within(aoi_union)
        plot_wells = well_locations.loc[mask].copy()
        if len(plot_wells) == 0:
            warnings.warn("No well locations fall inside the AOI; nothing to plot")
            return None

    if country is not None:
        cstr = str(country).strip()
        col = plot_wells["Country"].astype(str).str.strip()
        mask_c = col.str.lower() == cstr.lower()
        plot_wells = plot_wells.loc[mask_c].copy()
        if len(plot_wells) == 0:
            warnings.warn(f"No well locations for country={country!r}; nothing to plot")
            return None
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Extent: country mode uses well bounding box; else AOI or all wells
    if country is not None:
        lat_min = float(plot_wells["Latitude"].min())
        lat_max = float(plot_wells["Latitude"].max())
        lon_min = float(plot_wells["Longitude"].min())
        lon_max = float(plot_wells["Longitude"].max())
    elif aoi_gdf is not None:
        minx, miny, maxx, maxy = aoi_gdf.total_bounds
        if not clip_wells_to_aoi:
            minx = min(minx, plot_wells["Longitude"].min())
            maxx = max(maxx, plot_wells["Longitude"].max())
            miny = min(miny, plot_wells["Latitude"].min())
            maxy = max(maxy, plot_wells["Latitude"].max())
        lon_min, lon_max = minx, maxx
        lat_min, lat_max = miny, maxy
    else:
        lat_min = plot_wells['Latitude'].min()
        lat_max = plot_wells['Latitude'].max()
        lon_min = plot_wells['Longitude'].min()
        lon_max = plot_wells['Longitude'].max()
    
    # Add padding (10% of range, minimum 1 degree)
    lat_padding = max((lat_max - lat_min) * 0.05, 1.0)
    lon_padding = max((lon_max - lon_min) * 0.02, 1.0)
    
    extent = [
        lon_min - lon_padding,
        lon_max + lon_padding,
        lat_min - lat_padding,
        lat_max + lat_padding
    ]
    
    # Set extent
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    
    # Basemap styling aligned with plot_arid_watersheds (grace_analysis_utils)
    ax.add_feature(cfeature.LAND, facecolor="lightgrey")
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    # Country boundaries: solid lines (match other project maps; not dashed)
    ax.add_feature(
        cfeature.BORDERS, linestyle="-", edgecolor="black", linewidth=0.5
    )
    ax.coastlines()
    
    # Arid / AOI boundary (dark blue; above basemap, below well graphics)
    _aoi_edgecolor = "#0d2857"
    if aoi_gdf is not None:
        ax.add_geometries(
            aoi_gdf.geometry,
            crs=ccrs.PlateCarree(),
            facecolor="none",
            edgecolor=_aoi_edgecolor,
            linewidth=1,
            linestyle="-",
            zorder=3,
        )
    
    # Grid label ticks: adaptive ° spacing in country mode (small extents still get lines); else fixed global lists
    use_five_degree_grid = country is not None
    if not use_five_degree_grid:
        well_map_lon_ticks = np.array([-120.0, -60.0, 0.0, 60.0, 120.0])
        well_map_lat_ticks = np.array([-40.0, -20.0, 0.0, 20.0, 40.0])

    # Plot GRACE grid lines if grace_mean is provided
    if grace_mean is not None:
        grace_lats = grace_mean.lat.values
        grace_lons = grace_mean.lon.values
        
        # Filter grid lines to extent
        lat_mask = (grace_lats >= lat_min - lat_padding) & (grace_lats <= lat_max + lat_padding)
        lon_mask = (grace_lons >= lon_min - lon_padding) & (grace_lons <= lon_max + lon_padding)
        
        grid_lats = grace_lats[lat_mask]
        grid_lons = grace_lons[lon_mask]
        
        # Plot latitude grid lines
        for glat in grid_lats:
            ax.plot(
                [lon_min - lon_padding, lon_max + lon_padding],
                [glat, glat],
                color="gray",
                linewidth=0.5,
                linestyle="--",
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # Plot longitude grid lines
        for glon in grid_lons:
            ax.plot(
                [glon, glon],
                [lat_min - lat_padding, lat_max + lat_padding],
                color="gray",
                linewidth=0.5,
                linestyle="--",
                alpha=0.7,
                transform=ccrs.PlateCarree(),
                zorder=1,
            )

        # GRACE grid labels (tick spacing: 5° in country mode, else fixed lists)
        from matplotlib.ticker import FixedLocator, MultipleLocator

        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.65,
            color="gray",
            alpha=0.55,
            linestyle="-",
            zorder=2,
        )
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = True
        gl.bottom_labels = True
        gl.xlabel_style = {"rotation": 0, "size": 10, "color": "black"}
        gl.ylabel_style = {"rotation": 90, "size": 10, "color": "black"}
        if use_five_degree_grid:
            geo_step = _nice_lonlat_grid_step_deg(
                lon_min - lon_padding,
                lon_max + lon_padding,
                lat_min - lat_padding,
                lat_max + lat_padding,
                target_lines=6,
            )
            gl.xlocator = MultipleLocator(geo_step)
            gl.ylocator = MultipleLocator(geo_step)
        else:
            gl.xlocator = FixedLocator(well_map_lon_ticks)
            gl.ylocator = FixedLocator(well_map_lat_ticks)
        # Labeled lon/lat grid (adaptive ° in country mode; else fixed ticks). Fine dashed lines at native GRACE
        # coordinates are drawn in the loop above; both are optional via show_geo_grid.
        gl.xlines = show_geo_grid
        gl.ylines = show_geo_grid
    else:
        import matplotlib.ticker as mticker

        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.65,
            color="gray",
            alpha=0.55,
            linestyle="-",
        )
        gl.top_labels = False
        gl.right_labels = False
        if use_five_degree_grid:
            geo_step = _nice_lonlat_grid_step_deg(
                lon_min - lon_padding,
                lon_max + lon_padding,
                lat_min - lat_padding,
                lat_max + lat_padding,
                target_lines=6,
            )
            gl.xlocator = mticker.MultipleLocator(geo_step)
            gl.ylocator = mticker.MultipleLocator(geo_step)
        else:
            gl.xlocator = mticker.FixedLocator(well_map_lon_ticks)
            gl.ylocator = mticker.FixedLocator(well_map_lat_ticks)
        gl.xlabel_style = {"rotation": 0, "size": 10, "color": "black"}
        gl.ylabel_style = {"rotation": 90, "size": 10, "color": "black"}
        gl.xlines = show_geo_grid
        gl.ylines = show_geo_grid
    
    # Plot by country with different colors
    countries = plot_wells["Country"].unique()
    colors = plt.cm.tab20(np.linspace(0, 1, len(countries)))

    for country, color in zip(countries, colors):
        country_wells = plot_wells[plot_wells["Country"] == country]
        ax.scatter(
            country_wells["Longitude"],
            country_wells["Latitude"],
            c=[color],
            s=12,
            marker="o",
            edgecolors="black",
            linewidths=0.35,
            transform=ccrs.PlateCarree(),
            zorder=5,
            label=f"{country} (n={len(country_wells)})",
        )
    
    # Plot GRACE assigned pixels/links only when explicitly requested
    if draw_haversine and 'grace_lat' in plot_wells.columns and 'grace_lon' in plot_wells.columns:
        # Plot GRACE pixel centers
        wells_with_grace = plot_wells[
            plot_wells['grace_lat'].notna() & plot_wells['grace_lon'].notna()
        ]
        if len(wells_with_grace) > 0:
            ax.scatter(
                wells_with_grace['grace_lon'],
                wells_with_grace['grace_lat'],
                c='red',
                s=22,
                marker='x',
                linewidths=0.45,
                transform=ccrs.PlateCarree(),
                zorder=6,
                label='GRACE pixel centers',
                alpha=0.7
            )
            
            # Draw connection lines (well -> assigned GRACE pixel)
            for idx, row in wells_with_grace.iterrows():
                if pd.notna(row['grace_lat']) and pd.notna(row['grace_lon']):
                    ax.plot(
                        [row['Longitude'], row['grace_lon']],
                        [row['Latitude'], row['grace_lat']],
                        color='red',
                        linewidth=0.8,
                        linestyle=':',
                        alpha=0.4,
                        transform=ccrs.PlateCarree(),
                        zorder=4
                    )
    
    from matplotlib.lines import Line2D

    legend_handles, legend_labels = ax.get_legend_handles_labels()
    if aoi_gdf is not None:
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=_aoi_edgecolor,
                linewidth=0.5,
                linestyle="-",
                label="(Arid regions)",
            )
        )
        legend_labels.append("Arid regions")
    ax.legend(
        legend_handles,
        legend_labels,
        loc="best",
        fontsize=10,
        framealpha=0.9,
        borderaxespad=0.0,
        labelspacing=0.45,
        handlelength=1.6,
    )

    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plot_info = {'figure': fig, 'axis': ax}
    return plot_info


def print_well_info(gw_data):
    """Print one-line summary: well count, date range, water level range."""
    locations = gw_data.get('well_locations') if isinstance(gw_data, dict) else None
    ts = gw_data.get('time_series') if isinstance(gw_data, dict) else None
    if locations is None or len(locations) == 0:
        print("Wells: 0 | Date range: N/A | Water level: N/A")
        warnings.warn("print_well_info: no well locations available.")
        return
    n = len(locations)
    if ts is None or getattr(ts, 'empty', True) or len(ts) == 0:
        print(f"Wells: {n} | Date range: N/A | Water level: N/A")
        warnings.warn("print_well_info: time series is empty; cannot summarize date/water-level range.")
        return
    vals = ts.values.flatten()[~np.isnan(ts.values.flatten())]
    wl = f"{np.nanmin(vals):.2f} to {np.nanmax(vals):.2f}" if len(vals) else "N/A"
    print(f"Wells: {n} | Date range: {ts.index.min().strftime('%Y-%m-%d')} to {ts.index.max().strftime('%Y-%m-%d')} | Water level: {wl}")


def save_groundwater_data(well_locations, time_series, base_path, prefix='all_countries'):
    """
    Save well locations and time series dataframes to CSV files.
    
    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with well location data
    time_series : pd.DataFrame
        DataFrame with time series data
    base_path : str
        Base directory path to save files
    prefix : str, default='all_countries'
        Prefix for output filenames
    
    Returns
    -------
    dict
        Dictionary with paths to saved files
    """
    if well_locations is None or len(well_locations) == 0:
        warnings.warn("save_groundwater_data: well_locations is empty; nothing to save.")
        return {}
    if time_series is None or getattr(time_series, 'empty', True) or len(time_series) == 0:
        warnings.warn("save_groundwater_data: time_series is empty; nothing to save.")
        return {}

    base_path_obj = Path(base_path)
    try:
        base_path_obj.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise OSError(f"Failed to create output directory {base_path_obj}: {e}") from e
    
    # Save well locations
    locations_path = base_path_obj / f"{prefix}_well_locations.csv"
    try:
        well_locations.to_csv(locations_path, index=False)
    except OSError as e:
        raise OSError(f"Failed to save well locations to {locations_path}: {e}") from e
    
    # Save time series
    timeseries_path = base_path_obj / f"{prefix}_time_series.csv"
    try:
        time_series.to_csv(timeseries_path)
    except OSError as e:
        raise OSError(f"Failed to save time series to {timeseries_path}: {e}") from e
    
    saved_paths = {
        'well_locations': str(locations_path),
        'time_series': str(timeseries_path)
    }
    
    print(f"Saved well locations to: {locations_path}")
    print(f"Saved time series to: {timeseries_path}")
    
    return saved_paths


# TWSA / GWSA colors shared across correlation and lag distribution plots
TWSA_PLOT_COLOR = '#3498db'
GWSA_PLOT_COLOR = '#e74c3c'


def plot_correlation_distributions_by_country(
    wells_gdf_raw,
    wells_gdf_residual,
    variable='TWS',
    ncols=None,
    figsize=(12, 8),
    save_path=None,
    require_common_tws_gws=False,
):
    """
    Plot distribution of correlation coefficients (GWLA vs GRACE) by country.
    
    For ``variable='tws'`` or ``'gws'``, draw two stacked subplots: anomaly and
    residual. For ``variable='both'``, draw TWSA/GWSA anomaly and residual panels
    in either a 2x2 layout (``ncols=2``) or a vertical 4x1 stack (``ncols=1``).
    
    Uses violin + box plots per country, similar to plot_well_grace_distributions_by_country.
    
    Parameters
    ----------
    wells_gdf_raw : geopandas.GeoDataFrame
        From correlate_wells_with_grace['raw']. Must have Country and corr_tws (or corr_gws).
    wells_gdf_residual : geopandas.GeoDataFrame
        From correlate_wells_with_grace['residual']. Same columns.
    variable : {'TWS', 'GWS', 'both'}, default='TWS'
        Which correlation to plot: corr_tws, corr_gws, or both.
    ncols : {1, 2}, optional
        Number of subplot columns. Only ``variable='both'`` supports ``ncols=2``;
        ``None`` defaults to 2 for ``both`` and 1 otherwise.
    figsize : tuple, default=(12, 8)
    save_path : str, optional
        Path to save the figure.
    require_common_tws_gws : bool, default False
        When True, keep only wells with valid TWSA and GWSA optimal-lag correlations
        (``corr_tws`` and ``corr_gws``) separately for anomaly and residual GeoDataFrames.
        TWSA and GWSA panels then use the same wells within each country and state.

    Returns
    -------
    pandas.DataFrame
        Summary statistics per country (n, median, mean) for both anomaly and residual.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    
    base = variable.strip().lower()
    if base not in ('tws', 'gws', 'both'):
        raise ValueError("variable must be 'TWS', 'GWS', or 'both'.")
    bases = ['tws', 'gws'] if base == 'both' else [base]
    if ncols is None:
        ncols = 2 if base == 'both' else 1
    ncols = int(ncols)
    if ncols not in (1, 2):
        raise ValueError("ncols must be 1 or 2.")
    if base != 'both' and ncols != 1:
        raise ValueError("ncols=2 is only supported when variable='both'.")
    
    for b in bases:
        corr_col = f'corr_{b}'
        if corr_col not in wells_gdf_raw.columns or corr_col not in wells_gdf_residual.columns:
            raise ValueError(f"Required column '{corr_col}' not found in both GeoDataFrames.")

    filter_counts = {}
    if require_common_tws_gws:
        print("plot_correlation_distributions_by_country: require_common_tws_gws=True")
        fc_raw = _count_common_tws_gws_wells(wells_gdf_raw)
        fc_res = _count_common_tws_gws_wells(wells_gdf_residual)
        filter_counts = {'anomaly': fc_raw.get('Max Lag', {}), 'residual': fc_res.get('Max Lag', {})}
        for state_label, fc in filter_counts.items():
            print(
                f"  {state_label}: TWS={fc.get('n_tws', 0)}, GWS={fc.get('n_gws', 0)}, "
                f"common={fc.get('n_common', 0)} (TWS-only={fc.get('n_tws_only', 0)})"
            )
        wells_gdf_raw = _filter_common_tws_gws_gdf(wells_gdf_raw, 'Max Lag')
        wells_gdf_residual = _filter_common_tws_gws_gdf(wells_gdf_residual, 'Max Lag')
    
    # Collect correlation values per country
    def _collect_by_country(gdf, corr_col):
        out = {}
        for _, row in gdf.iterrows():
            c = str(row.get('Country', 'unknown')).strip()
            v = row.get(corr_col)
            if pd.notna(v) and np.isfinite(v):
                if c not in out:
                    out[c] = []
                out[c].append(float(v))
        return out
    
    def _var_label_for_base(b):
        return 'TWSA' if b == 'tws' else 'GWSA'
    
    def _color_for_base(b):
        return TWSA_PLOT_COLOR if b == 'tws' else GWSA_PLOT_COLOR
    
    panel_data_by_base = {}
    all_country_names = set()
    for b in bases:
        corr_col = f'corr_{b}'
        country_raw = _collect_by_country(wells_gdf_raw, corr_col)
        country_res = _collect_by_country(wells_gdf_residual, corr_col)
        panel_data_by_base[b] = {
            'country_raw': country_raw,
            'country_res': country_res,
            'var_label': _var_label_for_base(b),
        }
        all_country_names.update(country_raw.keys())
        all_country_names.update(country_res.keys())
    
    all_countries = sorted(all_country_names)
    if not all_countries:
        print("No correlation data to plot.")
        return pd.DataFrame()
    
    def _to_df(country_vals):
        data = []
        for c in all_countries:
            for x in country_vals.get(c, []):
                data.append({'Country': c, 'Value': x})
        return pd.DataFrame(data) if data else pd.DataFrame(columns=['Country', 'Value'])
    
    panels = []
    if base == 'both':
        for b in ('tws', 'gws'):
            var_label = panel_data_by_base[b]['var_label']
            panels.append({
                'base': b,
                'kind': 'raw',
                'df': _to_df(panel_data_by_base[b]['country_raw']),
                'country_vals': panel_data_by_base[b]['country_raw'],
                'ylabel': f'Spearman ρ ({var_label})',
                'color': _color_for_base(b),
            })
        for b in ('tws', 'gws'):
            var_label = panel_data_by_base[b]['var_label']
            panels.append({
                'base': b,
                'kind': 'res',
                'df': _to_df(panel_data_by_base[b]['country_res']),
                'country_vals': panel_data_by_base[b]['country_res'],
                'ylabel': f'Spearman ρ ({var_label} residual)',
                'color': _color_for_base(b),
            })
        panel_letters = ['a', 'b', 'c', 'd']
    else:
        var_label = panel_data_by_base[base]['var_label']
        panels = [
            {
                'base': base,
                'kind': 'raw',
                'df': _to_df(panel_data_by_base[base]['country_raw']),
                'country_vals': panel_data_by_base[base]['country_raw'],
                'ylabel': f'Spearman ρ ({var_label})',
                'color': _color_for_base(base),
            },
            {
                'base': base,
                'kind': 'res',
                'df': _to_df(panel_data_by_base[base]['country_res']),
                'country_vals': panel_data_by_base[base]['country_res'],
                'ylabel': f'Spearman ρ ({var_label} residual)',
                'color': _color_for_base(base),
            },
        ]
        panel_letters = ['a', 'b']
    
    n_panels = len(panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=False, sharey=True)
    axes = np.asarray(axes, dtype=object).reshape(-1)
    fig.patch.set_facecolor('white')
    
    _axis_fs = 11  # match x tick labels and y tick labels

    def _y_corr_axis_buffer(ax):
        """Extra vertical space above y=1 for stats boxes; ticks stay at -1..1 only (no 1.2/1.4)."""
        ax.set_ylim(-1, 1.4)
        ax.set_yticks(np.array([-1.0, -0.5, 0.0, 0.5, 1.0]))
    
    def _panel_letter(ax, letter, fontsize=11):
        ax.text(
            0.02, 0.98, letter,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, fontweight='bold', zorder=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9),
        )
    
    def _show_xlabels_for_panel(idx):
        return idx >= (n_panels - ncols)

    def _plot_violin(ax, df, color, ylabel, country_vals_dict, show_xlabels=False):
        ax.set_facecolor('#fafafa')
        if len(df) == 0:
            ax.set_ylabel(ylabel, fontsize=_axis_fs, fontweight='medium')
            _y_corr_axis_buffer(ax)
            ax.tick_params(axis='y', labelsize=_axis_fs)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
            ax.grid(axis='y', alpha=0.4, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if not show_xlabels:
                ax.set_xticklabels([])
            return
        # Matplotlib's violinplot can't handle empty arrays; plot only countries with data.
        countries_with_data = [c for c in all_countries if len(country_vals_dict.get(c, [])) > 0]
        if not countries_with_data:
            ax.set_ylabel(ylabel, fontsize=_axis_fs, fontweight='medium')
            _y_corr_axis_buffer(ax)
            ax.tick_params(axis='y', labelsize=_axis_fs)
            ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
            ax.grid(axis='y', alpha=0.4, linestyle='--')
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            if not show_xlabels:
                ax.set_xticklabels([])
            return
        data_by_country = [df[df['Country'] == c]['Value'].values for c in countries_with_data]
        positions = np.arange(len(countries_with_data), dtype=float) * 0.82
        parts = ax.violinplot(data_by_country, positions=positions,
                              showmeans=False, showmedians=False, showextrema=False)
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.5)
        bp = ax.boxplot(data_by_country, positions=positions, widths=0.13,
                        patch_artist=True, showfliers=True,
                        flierprops=dict(marker='o', markersize=4, alpha=0.5, markeredgecolor='none'))
        for patch in bp['boxes']:
            patch.set_facecolor('white')
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        for line in bp['medians']:
            line.set_color('#2c3e50')
            line.set_linewidth(2)
        _y_corr_axis_buffer(ax)
        ymin, ymax = ax.get_ylim()
        for i, c in enumerate(countries_with_data):
            vals = country_vals_dict.get(c, [])
            if len(vals) > 0:
                arr = np.array(vals)
                n, med = len(arr), np.median(arr)
                ax.text(positions[i], ymax - 0.02 * (ymax - ymin), f'n={n}\nmed={med:.2f}',
                        ha='center', va='top', fontsize=10, fontweight='medium',
                        bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.set_xticks(positions)
        if show_xlabels:
            ax.set_xticklabels(
                countries_with_data,
                fontsize=_axis_fs,
                fontweight='medium',
                rotation=0,
                ha='right',
            )
            ax.tick_params(axis='x', labelsize=_axis_fs)
        else:
            ax.set_xticklabels([])
        ax.set_ylabel(ylabel, fontsize=_axis_fs, fontweight='medium')
        ax.tick_params(axis='y', labelsize=_axis_fs)
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    for idx, panel in enumerate(panels):
        ax = axes[idx]
        _plot_violin(
            ax,
            panel['df'],
            panel['color'],
            panel['ylabel'],
            panel['country_vals'],
            show_xlabels=_show_xlabels_for_panel(idx),
        )
        _panel_letter(ax, panel_letters[idx])
    
    for ax in axes[n_panels:]:
        ax.set_visible(False)

    # Summary DataFrame
    summary_rows = []
    for c in all_countries:
        row = {'Country': c}
        for b in bases:
            var_label = panel_data_by_base[b]['var_label']
            raw_vals = panel_data_by_base[b]['country_raw'].get(c, [])
            res_vals = panel_data_by_base[b]['country_res'].get(c, [])
            ar, rr = np.array(raw_vals), np.array(res_vals)
            row.update({
                f'{var_label}_raw_n': len(ar),
                f'{var_label}_raw_median': np.median(ar) if len(ar) > 0 else np.nan,
                f'{var_label}_raw_mean': np.mean(ar) if len(ar) > 0 else np.nan,
                f'{var_label}_res_n': len(rr),
                f'{var_label}_res_median': np.median(rr) if len(rr) > 0 else np.nan,
                f'{var_label}_res_mean': np.mean(rr) if len(rr) > 0 else np.nan,
            })
        summary_rows.append(row)
    summary_df = pd.DataFrame(summary_rows)
    if require_common_tws_gws and filter_counts:
        summary_df.attrs['filter_counts'] = filter_counts
    
    fig.tight_layout(pad=0.6, h_pad=0.35, w_pad=0.35)
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
    
    return summary_df



def _correct_pumping_artifacts(
    series: pd.Series,
    interpolate: bool = True,
    return_debug: bool = False,
    recharge_lookback_months: int = 3,
    min_run_len: int = 0,
    recharge_level_quantile: float | None = 0.5,
    mask_by_rate: bool = False,
) -> tuple:
    """
    Mask outliers using a lower fence (Q1 - 1.5*IQR). The distribution can be based on
    GW level or on rate of change (dh/dt).

    mask_by_rate=False (default): use level distribution; mask months where level < level_fence.
    mask_by_rate=True: use rate distribution; mask months where dh/dt < rate_fence (negative outliers).
    min_run_len: runs of consecutive masked months shorter than this are discarded (unmasked); 0 = keep all.

    If interpolate is True, fill masked values with linear interpolation; otherwise leave as NaN.
    Returns (corrected_series, n_masked) by default.
    If return_debug=True, returns (corrected_series, n_masked, debug_dict) with
    level_fence and/or rate_fence, masked_dates; c_pump/c_stab_q1/c_stab_q3 for plot compatibility.
    """
    series = series.copy()
    series.index = pd.to_datetime(series.index)
    series = series.sort_index()
    level_fence = np.nan
    rate_fence = np.nan
    c_pump = np.nan
    c_stab_q1 = np.nan
    c_stab_q3 = np.nan

    if mask_by_rate:
        dh_dt = series.diff()
        rates = dh_dt.dropna()
        if len(rates) < 4:
            if return_debug:
                return series, 0, {
                    'c_pump': c_pump, 'c_stab_q1': c_stab_q1, 'c_stab_q3': c_stab_q3,
                    'level_fence': level_fence, 'rate_fence': rate_fence, 'masked_dates': [],
                }
            return series, 0
        q1 = rates.quantile(0.25)
        q3 = rates.quantile(0.75)
        iqr = q3 - q1
        rate_fence = float(q1 - 1.5 * iqr)
        c_pump = rate_fence
        masked_dates = [idx for idx in dh_dt.index if pd.notna(dh_dt.loc[idx]) and dh_dt.loc[idx] < rate_fence]
    else:
        levels = series.dropna()
        if len(levels) < 4:
            if return_debug:
                return series, 0, {
                    'c_pump': c_pump, 'c_stab_q1': c_stab_q1, 'c_stab_q3': c_stab_q3,
                    'level_fence': level_fence, 'rate_fence': rate_fence, 'masked_dates': [],
                }
            return series, 0
        q1 = levels.quantile(0.25)
        q3 = levels.quantile(0.75)
        iqr = q3 - q1
        level_fence = float(q1 - 1.5 * iqr)
        masked_dates = [idx for idx in series.index if pd.notna(series.loc[idx]) and series.loc[idx] < level_fence]

    # Anti-flicker: drop runs shorter than min_run_len (treat as noise)
    if min_run_len > 0 and len(masked_dates) > 0:
        in_index = [d for d in masked_dates if d in series.index]
        if in_index:
            positions = sorted(series.index.get_loc(d) for d in in_index)
            runs = []
            current = [positions[0]]
            for p in positions[1:]:
                if p == current[-1] + 1:
                    current.append(p)
                else:
                    runs.append(current)
                    current = [p]
            runs.append(current)
            kept_positions = {p for run in runs for p in run if len(run) >= min_run_len}
            masked_dates = [series.index[p] for p in sorted(kept_positions)]

    if len(masked_dates) == 0:
        if return_debug:
            return series, 0, {
                'c_pump': c_pump, 'c_stab_q1': c_stab_q1, 'c_stab_q3': c_stab_q3,
                'level_fence': level_fence, 'rate_fence': rate_fence, 'masked_dates': [],
            }
        return series, 0

    orig_nan_mask = series.isna()
    for d in masked_dates:
        if d in series.index:
            series.loc[d] = np.nan
    if interpolate:
        series = series.interpolate(method='linear', limit=3, limit_direction='both')
    series = series.where(~orig_nan_mask)
    if return_debug:
        return series, len(masked_dates), {
            'c_pump': c_pump, 'c_stab_q1': c_stab_q1, 'c_stab_q3': c_stab_q3,
            'level_fence': level_fence, 'rate_fence': rate_fence, 'masked_dates': list(masked_dates),
        }
    return series, len(masked_dates)


def _mask_flat_gwl_segments(series: pd.Series, min_flat_months: int = 12) -> tuple[pd.Series, int]:
    """
    Set to NaN any GWL values that lie in a segment where dh/dt is 0 for at least min_flat_months.
    Such plateaus often indicate stuck sensors or missing data.
    Returns (masked_series, n_masked).
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    s = s.sort_index()
    diff = s.diff()
    # Run of 12 months with 0 change = 11 consecutive zero diffs (between 12 points)
    min_run = min_flat_months - 1  # number of consecutive zero diffs required
    if len(diff) < min_run + 1:
        return s, 0
    # Treat near-zero diff as flat (float safety)
    flat = (diff.abs() < 1e-9) | (diff == 0)
    flat = flat.fillna(False)
    n_masked = 0
    i = 0
    while i < len(flat):
        if not flat.iloc[i]:
            i += 1
            continue
        run_start = i
        while i < len(flat) and flat.iloc[i]:
            i += 1
        run_len = i - run_start
        if run_len >= min_run:
            # Segment in s: indices run_start to run_start+run_len (inclusive) = run_len+1 points
            start_pos = run_start
            end_pos = min(run_start + run_len + 1, len(s))
            n_masked += s.iloc[start_pos:end_pos].notna().sum()
            s.iloc[start_pos:end_pos] = np.nan
    return s, int(n_masked)


def correlate_wells_with_grace(
    well_locations,
    well_time_series,
    grace_mean,
    grace_gws,
    aoi_geometry=None,
    method='spearman',
    min_common_dates=60,
    max_lag_months=34,
    tws_only=False,
    depth_threshold=50,
    correct_pumping=False,
    interp_pumping=True,
    min_run_len=0,
    recharge_lookback_months=3,
    recharge_level_quantile=0.5,
    mask_by_rate=False,
    decomposition_method: str = 'harmonic',
    verbose=False,
):
    """
    Correlate groundwater well time series with GRACE TWS and GWS.
    
    Computes correlations for BOTH raw (original) and residual (trend + annual/semi-annual
    removed) series in one pass. Returns both so downstream can use either without
    recomputing.
    
    Converts well locations to GeoDataFrame, clips to AOI, finds nearest
    GRACE pixel for each well, extracts time series, and calculates correlations.
    
    Parameters
    ----------
    well_locations : pd.DataFrame
        DataFrame with columns [ID, Country, Latitude, Longitude, ...]
    well_time_series : pd.DataFrame
        DataFrame with dates as index and well IDs as columns
    grace_mean : xarray.DataArray
        GRACE TWS data with dimensions (time, lat, lon)
    grace_gws : xarray.DataArray
        GRACE GWS data with dimensions (time, lat, lon)
    aoi_geometry : geopandas.GeoDataFrame, GeoSeries, or geometry, optional
        Area of interest boundary for clipping wells
    method : str, default='spearman'
        Correlation method ('pearson', 'spearman', 'kendall')
    min_common_dates : int, default=60
        Minimum number of common non-NaN dates required for correlation.
    max_lag_months : int, default=34
        Maximum lag to test (in months). Positive lag means well responds after GRACE.
    tws_only : bool, default=False
        If True, compute and print only TWS correlation (skip GWS). Use when
        grace_gws equals grace_mean to avoid duplicate statistics.
    depth_threshold : int, default=50
        Threshold in meters for classifying wells as Shallow (<=threshold) or
        Deep (>threshold). Classification is based on the temporal mean of the
        cleaned time series (post-QC, post-pumping correction).
    correct_pumping : bool, default=False
        If True, mask pumping artifacts using an IQR lower-fence rule
        (Q1 - 1.5*IQR) on the QC-cleaned level series before computing anomaly
        and correlation.  Column pump_corr_n stores the number of masked points.
    interp_pumping : bool, default=True
        Only used when correct_pumping is True. If True, linearly interpolate
        the masked pumping points; if False, leave them as NaN.
    min_run_len : int, default=0
        Only used when correct_pumping is True. Anti-flicker: masked runs
        shorter than this many consecutive months are unmasked; 0 = keep all.
    recharge_lookback_months : int, default=3
        Reserved for future use (not used in current IQR-fence implementation).
    recharge_level_quantile : float or None, default=0.5
        Reserved for future use (not used in current IQR-fence implementation).
    mask_by_rate : bool, default=False
        Only used when correct_pumping is True. If False, apply the IQR fence
        to GW levels (mask values below Q1 - 1.5*IQR); if True, apply the
        fence to month-to-month rates of change dh/dt (mask negative rate
        outliers).
    verbose : bool, default=False
        If True, print step-by-step progress, exclusion diagnostics, and the full
        Raw/Residual x Shallow/Deep correlation summary. Default prints only a
        concise wells-kept line and shallow-well summary (the violin plots show
        the full distributions).
    decomposition_method : {'harmonic', 'stl_13'}, default='harmonic'
        Decomposition used for GW and GRACE anomaly time series. 'harmonic' uses the existing
        global linear + annual + semi-annual fit. 'stl_13' uses STL with 13-month seasonal/trend
        windows (long-term at 13 months, similar to the referenced paper).

    Returns
    -------
    dict
        Dictionary with keys:
        - 'raw': GeoDataFrame with TWS/GWS anomaly correlations (GWL 2004-2009 mean removed)
        - 'residual': GeoDataFrame with residual correlations (trend+annual+semi-annual removed)
        - 'well_series': dict, well_id -> {'anomaly', 'residual', 'trend', 'annual', 'semi_annual'}
        - 'grace_series': dict, (lat, lon) -> {'tws', 'tws_residual', 'tws_trend', ...}
    """
    import geopandas as gpd
    import numpy as np
    import pandas as pd
    from shapely.geometry import Point
    from scipy.stats import pearsonr, spearmanr, kendalltau

    def _vprint(*args, **kwargs):
        if verbose:
            print(*args, **kwargs)
    
    _vprint("="*70)
    _vprint("CORRELATING WELLS WITH GRACE DATA (raw + residual in one pass)")
    _vprint("="*70)
    
    # Classify wells by depth (Shallow/Deep) before correlation
    _vprint(f"\nStep 0: Classifying wells by depth (threshold = {depth_threshold}m)...")
    well_locations_classified = classify_well_depths(
        well_locations.copy(), well_time_series, depth_threshold=depth_threshold, verbose=verbose
    )
    
    # Convert well_locations to GeoDataFrame
    _vprint(f"\nStep 1: Converting well locations to GeoDataFrame...")
    well_gdf = gpd.GeoDataFrame(
        well_locations_classified,
        geometry=[Point(lon, lat) for lon, lat in zip(well_locations['Longitude'], well_locations['Latitude'])],
        crs="EPSG:4326"
    )
    _vprint(f"  Total wells: {len(well_gdf)}")
    
    # Clip to AOI if provided
    if aoi_geometry is not None:
        _vprint(f"\nStep 2: Clipping wells to AOI geometry...")
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        
        wells_before_clip = len(well_gdf)
        well_gdf = gpd.sjoin(well_gdf, aoi_gdf, how='inner', predicate='within')
        # Remove the index_right column added by sjoin
        well_gdf = well_gdf.drop(columns=[col for col in well_gdf.columns if col == 'index_right'], errors='ignore')
        wells_after_clip = len(well_gdf)
        _vprint(f"  Wells before clipping: {wells_before_clip}")
        _vprint(f"  Wells removed: {wells_before_clip - wells_after_clip}")
        _vprint(f"  Wells after clipping: {wells_after_clip}")   
    else:
        _vprint(f"\nStep 2: No AOI geometry provided, using all wells...")
    
    if len(well_gdf) == 0:
        warnings.warn(
            "No wells remaining after clipping. Returning structured empty result "
            "({'raw': empty GDF, 'residual': empty GDF, 'well_series': {}, 'grace_series': {}})."
        )
        well_gdf['corr_tws'] = np.nan
        well_gdf['corr_gws'] = np.nan
        well_gdf['pvalue_tws'] = None
        well_gdf['pvalue_gws'] = None
        well_gdf['grace_lat'] = np.nan
        well_gdf['grace_lon'] = np.nan
        gdf_raw = well_gdf.copy()
        gdf_res = well_gdf.copy()
        gdf_raw.attrs['residual'] = False
        gdf_res.attrs['residual'] = True
        return {'raw': gdf_raw, 'residual': gdf_res, 'well_series': {}, 'grace_series': {}}
    
    # Get GRACE grid coordinates
    _vprint(f"\nStep 3: Finding nearest GRACE pixels for each well...")
    
    # Find nearest GRACE pixel for each well using xarray's nearest neighbor selection
    # This correctly finds the grid point closest to the well location
    well_gdf['grace_lat'] = np.nan
    well_gdf['grace_lon'] = np.nan
    
    for idx, row in well_gdf.iterrows():
        well_lat = row['Latitude']
        well_lon = row['Longitude']
        
        # Calculate geographic distance to all grid points using haversine formula
        # This is more accurate than xarray's .sel() which may find lat/lon independently
        grace_lats = grace_mean.lat.values
        grace_lons = grace_mean.lon.values
        
        # Calculate haversine distance to all grid points (true geographic distance)
        # This considers both lat and lon together, accounting for Earth's curvature
        lat_diff_rad = np.radians(grace_lats[:, np.newaxis] - well_lat)
        lon_diff_rad = np.radians(grace_lons[np.newaxis, :] - well_lon)
        
        # Haversine formula: a = sin²(Δlat/2) + cos(lat1) * cos(lat2) * sin²(Δlon/2)
        # c = 2 * arcsin(√a)
        a = (np.sin(lat_diff_rad/2)**2 + 
             np.cos(np.radians(grace_lats[:, np.newaxis])) * 
             np.cos(np.radians(well_lat)) * 
             np.sin(lon_diff_rad/2)**2)
        
        # Calculate great circle distance in radians
        distances_rad = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))  # Clip to avoid numerical errors
        distances_deg = np.degrees(distances_rad)  # Convert to degrees for easier interpretation
        
        # Find minimum distance (closest grid point considering both lat and lon)
        min_idx = np.unravel_index(np.argmin(distances_rad), distances_rad.shape)
        
        assigned_lat = float(grace_lats[min_idx[0]])
        assigned_lon = float(grace_lons[min_idx[1]])
        min_distance = float(distances_deg[min_idx[0], min_idx[1]])
        
        well_gdf.at[idx, 'grace_lat'] = assigned_lat
        well_gdf.at[idx, 'grace_lon'] = assigned_lon
        
        # Debug output for first few wells (or when debugging)
        if idx < 3:  # Print for first 3 wells
            # Find nearby grid points to show
            lat_mask = (grace_lats >= well_lat - 1.0) & (grace_lats <= well_lat + 1.0)
            lon_mask = (grace_lons >= well_lon - 1.0) & (grace_lons <= well_lon + 1.0)
            nearby_lats = grace_lats[lat_mask]
            nearby_lons = grace_lons[lon_mask]
            
            _vprint(f"\n  Well {idx}: ({well_lat:.5f}°, {well_lon:.5f}°)")
            _vprint(f"    Nearby GRACE grid points:")
            for glat in nearby_lats:
                for glon in nearby_lons:
                    glat_idx = np.where(grace_lats == glat)[0][0]
                    glon_idx = np.where(grace_lons == glon)[0][0]
                    dist = distances_deg[glat_idx, glon_idx]
                    marker = " <-- SELECTED" if (glat == assigned_lat and glon == assigned_lon) else ""
                    _vprint(f"      ({glat:.1f}°, {glon:.1f}°): distance = {dist:.6f}°{marker}")
            _vprint(f"    Assigned to: ({assigned_lat:.1f}°, {assigned_lon:.1f}°) with distance {min_distance:.6f}°")
    
    _vprint(f"  Mapped {len(well_gdf)} wells to GRACE grid")
    
    # Convert well_time_series index to datetime if needed and handle timezone
    _vprint(f"\nStep 4: Extracting GRACE time series and calculating correlations...")
    
    well_ts_index = pd.to_datetime(well_time_series.index)
    grace_time_index = pd.to_datetime(grace_mean.time.values)
    
    # Handle timezone if needed
    if well_ts_index.tz is not None and grace_time_index.tz is None:
        grace_time_index = grace_time_index.tz_localize('UTC')
    elif well_ts_index.tz is None and grace_time_index.tz is not None:
        well_ts_index = well_ts_index.tz_localize('UTC')
    
    # Initialize correlation columns
    # TWS: lag 0 and best lag
    well_gdf['corr_tws_lag0'] = np.nan
    well_gdf['pvalue_tws_lag0'] = None
    well_gdf['corr_tws'] = np.nan  # Best correlation
    well_gdf['lag_tws'] = np.nan   # Lag of best correlation
    well_gdf['pvalue_tws'] = None  # P-value at best lag
    
    # GWS: lag 0 and best lag
    well_gdf['corr_gws_lag0'] = np.nan
    well_gdf['pvalue_gws_lag0'] = None
    well_gdf['corr_gws'] = np.nan  # Best correlation
    well_gdf['lag_gws'] = np.nan   # Lag of best correlation
    well_gdf['pvalue_gws'] = None  # P-value at best lag

    # QC columns (well-level filtering)
    well_gdf['qc_baseline_mean_2004_2009_m'] = np.nan
    well_gdf['qc_baseline_n_2004_2009'] = 0
    well_gdf['qc_outliers_removed_30m_rule'] = 0
    well_gdf['qc_interpolated_points_limit3'] = 0
    well_gdf['qc_stuck_invalid'] = False
    #well_gdf['qc_stuck_min_rolling_std_60m'] = np.nan
    well_gdf['qc_excluded_reason'] = None
    well_gdf['pump_corr_n'] = 0
    
    # Select correlation function
    if method == 'pearson':
        corr_func = pearsonr
    elif method == 'spearman':
        corr_func = spearmanr
    elif method == 'kendall':
        corr_func = kendalltau
    else:
        raise ValueError(f"Unknown correlation method: {method}")
    
    # Calculate correlations for each well
    correlations_found = 0
    correlations_failed = 0
    correlations_gws_failed = 0
    
    # Diagnostic counters
    excluded_no_timeseries = 0
    excluded_stuck_sensor = 0
    excluded_insufficient_data_after_qc = 0
    excluded_insufficient_common_dates = 0
    excluded_grace_extraction_error = 0
    
    # Ensure ID types match between well_gdf and well_time_series
    well_gdf['ID'] = well_gdf['ID'].astype(str).str.strip()
    well_time_series.columns = well_time_series.columns.astype(str).str.strip()
    # When time_series uses unique_well_id (Country_ID), match on that
    if 'unique_well_id' in well_gdf.columns:
        well_gdf['unique_well_id'] = well_gdf['unique_well_id'].astype(str).str.strip()
    
    well_ids_in_timeseries = set(well_time_series.columns)
    well_ids_in_gdf = set(
        well_gdf['unique_well_id'].dropna().astype(str).str.strip().values
    ) if 'unique_well_id' in well_gdf.columns else set(well_gdf['ID'].values)
    matching_ids = well_ids_in_gdf & well_ids_in_timeseries
    missing_ids = well_ids_in_gdf - well_ids_in_timeseries
    
    _vprint(f"\n  Well ID matching check:")
    _vprint(f"    Wells in GeoDataFrame: {len(well_ids_in_gdf)}")
    _vprint(f"    Wells in time_series: {len(well_ids_in_timeseries)}")
    _vprint(f"    Matching IDs: {len(matching_ids)}")
    _vprint(f"    Missing IDs: {len(missing_ids)}")
    if len(missing_ids) > 0 and len(missing_ids) <= 10:
        _vprint(f"    Sample missing IDs: {list(missing_ids)[:10]}")
    elif len(missing_ids) > 10:
        _vprint(f"    Sample missing IDs (first 10): {list(missing_ids)[:10]}")
    if len(matching_ids) > 0 and len(matching_ids) <= 10:
        _vprint(f"    Sample matching IDs: {list(matching_ids)[:10]}")
    
    # Two GeoDataFrames: raw and residual correlations (computed in one pass)
    well_gdf_raw = well_gdf.copy()
    well_gdf_res = well_gdf.copy()
    well_series = {}
    grace_series = {}

    for idx, row in well_gdf.iterrows():
        well_ts_key = _get_well_ts_key(row)  # unique_well_id when present, else ID
        grace_lat = row['grace_lat']
        grace_lon = row['grace_lon']
        
        if well_ts_key not in well_time_series.columns:
            excluded_no_timeseries += 1
            continue
        
        # Extract well time series (keep NaNs for outlier masking + interpolation)
        well_ts = well_time_series[well_ts_key].copy()
        well_ts.index = pd.to_datetime(well_ts.index)
        well_ts = well_ts.sort_index()

        # Sign flip if needed
        parameter_type = row.get('parameter_type', None)
        well_ts_proc = _apply_sign_flip(well_ts, parameter_type)

        # QC cleaning (outliers/interp/stuck)
        well_ts_qc, stuck_invalid, baseline_mean, n_outliers, n_filled, min_roll_std = _apply_qc_cleaning(well_ts_proc)
        for _gdf in (well_gdf_raw, well_gdf_res):
            _gdf.at[idx, 'qc_baseline_mean_2004_2009_m'] = baseline_mean
        _idx_tz = well_ts_proc.index.tz
        _b0 = pd.Timestamp(QC_BASELINE_START, tz='UTC') if _idx_tz is not None else pd.Timestamp(QC_BASELINE_START)
        _b1 = pd.Timestamp(QC_BASELINE_END, tz='UTC') if _idx_tz is not None else pd.Timestamp(QC_BASELINE_END)
        qc_n = int(well_ts_proc.loc[(well_ts_proc.index >= _b0) & (well_ts_proc.index <= _b1)].dropna().shape[0])
        for _gdf in (well_gdf_raw, well_gdf_res):
            _gdf.at[idx, 'qc_baseline_n_2004_2009'] = qc_n
            _gdf.at[idx, 'qc_outliers_removed_30m_rule'] = n_outliers
            _gdf.at[idx, 'qc_interpolated_points_limit3'] = n_filled
            _gdf.at[idx, 'qc_stuck_min_rolling_std_60m'] = min_roll_std
            _gdf.at[idx, 'qc_stuck_invalid'] = stuck_invalid
        if stuck_invalid:
            for _gdf in (well_gdf_raw, well_gdf_res):
                _gdf.at[idx, 'qc_excluded_reason'] = 'stuck_sensor_rolling_std_60m_lt_0.05'
            excluded_stuck_sensor += 1
            continue

        # Mask GWL segments where dh/dt is 0 for 12+ months (stuck/no-change plateaus)
        well_ts_qc, n_flat_masked = _mask_flat_gwl_segments(well_ts_qc, min_flat_months=12)
        for _gdf in (well_gdf_raw, well_gdf_res):
            _gdf.at[idx, 'qc_flat_12m_masked'] = n_flat_masked

        # Require enough usable data after QC before attempting correlation
        n_valid_after_qc = well_ts_qc.dropna().shape[0]
        if n_valid_after_qc < min_common_dates:
            for _gdf in (well_gdf_raw, well_gdf_res):
                _gdf.at[idx, 'qc_excluded_reason'] = f'insufficient_data_after_qc_{n_valid_after_qc}_lt_{min_common_dates}'
            excluded_insufficient_data_after_qc += 1
            continue

        # Optional: correct pumping artifacts before anomaly and correlation
        if correct_pumping:
            well_ts_qc, pump_n = _correct_pumping_artifacts(
                well_ts_qc,
                interpolate=interp_pumping,
                min_run_len=min_run_len,
                recharge_lookback_months=recharge_lookback_months,
                recharge_level_quantile=recharge_level_quantile,
                mask_by_rate=mask_by_rate,
            )
            for _gdf in (well_gdf_raw, well_gdf_res):
                _gdf.at[idx, 'pump_corr_n'] = pump_n

        # Reclassify depth on the cleaned series (post-QC, post-pumping correction)
        _cleaned_vals = well_ts_qc.dropna()
        if len(_cleaned_vals) > 0 and parameter_type:
            _param_lower = str(parameter_type).lower()
            if 'depth' in _param_lower and ('ground' in _param_lower or 'well' in _param_lower):
                # well_ts_qc is sign-flipped (negative); undo to get positive depth
                _depth_m = float(-_cleaned_vals.mean())
            elif 'elevation' in _param_lower and ('a.m.s.l' in _param_lower or 'amsl' in _param_lower):
                _gse = row.get('ground_surface_elevation_m', None)
                if pd.notna(_gse):
                    _depth_m = float(_gse - _cleaned_vals.mean())
                else:
                    _depth_m = np.nan
            else:
                _depth_m = np.nan
            if pd.notna(_depth_m):
                _dc = 'Shallow' if _depth_m <= depth_threshold else 'Deep'
                for _gdf in (well_gdf_raw, well_gdf_res):
                    _gdf.at[idx, 'avg_depth_m'] = _depth_m
                    _gdf.at[idx, 'depth_class'] = _dc

        # GWL anomaly: remove 2004-2009 baseline mean (consistent with GRACE anomaly)
        well_anomaly, _ = _remove_baseline_mean(well_ts_qc)
        well_decomp = _decompose_series_full(well_anomaly, decomposition_method=decomposition_method)
        well_ts_for_tws = well_ts_qc.copy()
        well_ts_for_gws = well_ts_qc.copy()

        # Store well series for downstream (anomaly = 2004-2009 mean removed)
        well_series[well_ts_key] = {
            'anomaly': well_anomaly,
            'residual': well_decomp['residual'],
            'trend': well_decomp['trend'],
            'annual': well_decomp['annual'],
            'semi_annual': well_decomp['semi_annual'],
        }

        try:
            grace_key = (float(grace_lat), float(grace_lon))
            if grace_key not in grace_series:
                grace_tws_ext = _extract_grace_series(grace_mean, grace_lat, grace_lon).dropna()
                grace_tws_dec = _decompose_series_full(grace_tws_ext, decomposition_method=decomposition_method)
                grace_series[grace_key] = {
                    'tws': grace_tws_ext,
                    'tws_residual': grace_tws_dec['residual'],
                    'tws_trend': grace_tws_dec['trend'],
                    'tws_annual': grace_tws_dec['annual'],
                    'tws_semi_annual': grace_tws_dec['semi_annual'],
                }
                if not tws_only:
                    grace_gws_ext = _extract_grace_series(grace_gws, grace_lat, grace_lon).dropna()
                    grace_gws_dec = _decompose_series_full(grace_gws_ext, decomposition_method=decomposition_method)
                    grace_series[grace_key]['gws'] = grace_gws_ext
                    grace_series[grace_key]['gws_residual'] = grace_gws_dec['residual']
                    grace_series[grace_key]['gws_trend'] = grace_gws_dec['trend']
                    grace_series[grace_key]['gws_annual'] = grace_gws_dec['annual']
                    grace_series[grace_key]['gws_semi_annual'] = grace_gws_dec['semi_annual']

            grace_tws_series = grace_series[grace_key]['tws'].copy()
            well_ts_for_tws, grace_tws_series = _align_timezones(well_ts_for_tws, grace_tws_series)

            # Raw TWS: GWL anomaly vs GRACE TWS anomaly
            well_tws_raw = well_anomaly.copy()
            grace_tws_raw = grace_tws_series.copy()
            well_tws_raw, grace_tws_raw = _align_timezones(well_tws_raw, grace_tws_raw)
            common_raw = well_tws_raw.index.intersection(grace_tws_raw.index)
            if len(common_raw) >= min_common_dates:
                well_vals = well_tws_raw.loc[common_raw].sort_index().values
                grace_vals = grace_tws_raw.loc[common_raw].sort_index().values
                mask0 = ~(np.isnan(well_vals) | np.isnan(grace_vals))
                if mask0.sum() >= min_common_dates:
                    corr_tws_lag0, pvalue_tws_lag0 = corr_func(well_vals[mask0], grace_vals[mask0])
                    well_gdf_raw.at[idx, 'corr_tws_lag0'] = corr_tws_lag0
                    well_gdf_raw.at[idx, 'pvalue_tws_lag0'] = _format_pvalue(pvalue_tws_lag0)
                tws_res = _calculate_best_lag_correlation(well_tws_raw, grace_tws_raw, max_lag_months, min_common_dates, corr_func)
                if not np.isnan(tws_res['r_max']):
                    well_gdf_raw.at[idx, 'corr_tws'] = tws_res['r_max']
                    well_gdf_raw.at[idx, 'lag_tws'] = tws_res['lag_max']
                    well_gdf_raw.at[idx, 'pvalue_tws'] = _format_pvalue(tws_res['p_max'])
                    correlations_found += 1

            # Residual TWS: use precomputed decomposition
            well_tws_res = well_decomp['residual'].dropna()
            grace_tws_res = grace_series[grace_key]['tws_residual'].dropna()
            well_tws_res, grace_tws_res = _align_timezones(well_tws_res, grace_tws_res)
            common_res = well_tws_res.index.intersection(grace_tws_res.index)
            if len(common_res) >= min_common_dates:
                well_vals = well_tws_res.loc[common_res].sort_index().values
                grace_vals = grace_tws_res.loc[common_res].sort_index().values
                mask0 = ~(np.isnan(well_vals) | np.isnan(grace_vals))
                if mask0.sum() >= min_common_dates:
                    corr_tws_lag0, pvalue_tws_lag0 = corr_func(well_vals[mask0], grace_vals[mask0])
                    well_gdf_res.at[idx, 'corr_tws_lag0'] = corr_tws_lag0
                    well_gdf_res.at[idx, 'pvalue_tws_lag0'] = _format_pvalue(pvalue_tws_lag0)
                tws_res = _calculate_best_lag_correlation(well_tws_res, grace_tws_res, max_lag_months, min_common_dates, corr_func)
                if not np.isnan(tws_res['r_max']):
                    well_gdf_res.at[idx, 'corr_tws'] = tws_res['r_max']
                    well_gdf_res.at[idx, 'lag_tws'] = tws_res['lag_max']
                    well_gdf_res.at[idx, 'pvalue_tws'] = _format_pvalue(tws_res['p_max'])
        except Exception as e:
            excluded_grace_extraction_error += 1
            correlations_failed += 1
            continue
        
        if not tws_only:
            try:
                grace_gws_series = grace_series[grace_key]['gws'].copy()
                well_ts_for_gws, grace_gws_series = _align_timezones(well_ts_for_gws, grace_gws_series)

                # Raw GWS: GWL anomaly vs GRACE GWS anomaly
                well_gws_raw = well_anomaly.copy()
                grace_gws_raw = grace_gws_series.copy()
                well_gws_raw, grace_gws_raw = _align_timezones(well_gws_raw, grace_gws_raw)
                common_dates = well_gws_raw.index.intersection(grace_gws_raw.index)
                if len(common_dates) >= min_common_dates:
                    well_vals = well_gws_raw.loc[common_dates].sort_index().values
                    grace_gws_vals = grace_gws_raw.loc[common_dates].sort_index().values
                    mask0 = ~(np.isnan(well_vals) | np.isnan(grace_gws_vals))
                    if mask0.sum() >= min_common_dates:
                        corr_gws_lag0, pvalue_gws_lag0 = corr_func(well_vals[mask0], grace_gws_vals[mask0])
                        well_gdf_raw.at[idx, 'corr_gws_lag0'] = corr_gws_lag0
                        well_gdf_raw.at[idx, 'pvalue_gws_lag0'] = _format_pvalue(pvalue_gws_lag0)

                    gws_corr_result = _calculate_best_lag_correlation(
                        well_gws_raw, grace_gws_raw, max_lag_months, min_common_dates, corr_func
                    )
                    best_corr_gws = gws_corr_result['r_max']
                    best_pvalue_gws = gws_corr_result['p_max']
                    best_lag_gws = gws_corr_result['lag_max']
                    if not np.isnan(best_corr_gws):
                        well_gdf_raw.at[idx, 'corr_gws'] = best_corr_gws
                        well_gdf_raw.at[idx, 'lag_gws'] = best_lag_gws
                        well_gdf_raw.at[idx, 'pvalue_gws'] = _format_pvalue(best_pvalue_gws)

                # Residual GWS: use precomputed decomposition
                well_gws_res = well_decomp['residual'].dropna()
                grace_gws_res = grace_series[grace_key]['gws_residual'].dropna()
                well_gws_res, grace_gws_res = _align_timezones(well_gws_res, grace_gws_res)
                common_res_gws = well_gws_res.index.intersection(grace_gws_res.index)
                if len(common_res_gws) >= min_common_dates:
                    well_vals = well_gws_res.loc[common_res_gws].sort_index().values
                    grace_gws_vals = grace_gws_res.loc[common_res_gws].sort_index().values
                    mask0 = ~(np.isnan(well_vals) | np.isnan(grace_gws_vals))
                    if mask0.sum() >= min_common_dates:
                        corr_gws_lag0, pvalue_gws_lag0 = corr_func(well_vals[mask0], grace_gws_vals[mask0])
                        well_gdf_res.at[idx, 'corr_gws_lag0'] = corr_gws_lag0
                        well_gdf_res.at[idx, 'pvalue_gws_lag0'] = _format_pvalue(pvalue_gws_lag0)

                    gws_res_result = _calculate_best_lag_correlation(
                        well_gws_res, grace_gws_res, max_lag_months, min_common_dates, corr_func
                    )
                    if not np.isnan(gws_res_result['r_max']):
                        well_gdf_res.at[idx, 'corr_gws'] = gws_res_result['r_max']
                        well_gdf_res.at[idx, 'lag_gws'] = gws_res_result['lag_max']
                        well_gdf_res.at[idx, 'pvalue_gws'] = _format_pvalue(gws_res_result['p_max'])
            except Exception as e:
                correlations_gws_failed += 1
                continue
    
    _vprint(f"  Correlations calculated: {correlations_found}")
    if correlations_failed > 0:
        _vprint(f"  Correlations failed: {correlations_failed}")
    if not tws_only and correlations_gws_failed > 0:
        _vprint(f"  GWS correlations failed: {correlations_gws_failed}")
    
    # Print diagnostic information
    _vprint(f"\n  Diagnostic: Why wells were excluded:")
    _vprint(f"    - Missing time series data: {excluded_no_timeseries}")
    _vprint(f"    - Stuck sensor (rolling std < 0.05m): {excluded_stuck_sensor}")
    _vprint(f"    - Insufficient data after QC (< {min_common_dates} points): {excluded_insufficient_data_after_qc}")
    _vprint(f"    - Insufficient common dates with GRACE (< {min_common_dates}): {excluded_insufficient_common_dates}")
    _vprint(f"    - GRACE extraction error: {excluded_grace_extraction_error}")
    
    # Print summary statistics
    _vprint("\n" + "="*70)
    _vprint(f"CORRELATION SUMMARY  —  method: {method}")
    _vprint("="*70)

    def _print_corr_stats(subset, indent="  "):
        n_tws = subset['corr_tws'].notna().sum()
        n_gws = subset['corr_gws'].notna().sum()
        if n_tws > 0:
            _vprint(f"{indent}TWS: n={n_tws}, mean={subset['corr_tws'].mean():.2f}, "
                  f"median={subset['corr_tws'].median():.2f}, "
                  f"min={subset['corr_tws'].min():.2f}, max={subset['corr_tws'].max():.2f}")
            if 'lag_tws' in subset.columns:
                _vprint(f"{indent}TWS mean lag (months): {subset['lag_tws'].mean():.1f}")
        if not tws_only and n_gws > 0:
            _vprint(f"{indent}GWS: n={n_gws}, mean={subset['corr_gws'].mean():.2f}, "
                  f"median={subset['corr_gws'].median():.2f}, "
                  f"min={subset['corr_gws'].min():.2f}, max={subset['corr_gws'].max():.2f}")
            if 'lag_gws' in subset.columns:
                _vprint(f"{indent}GWS mean lag (months): {subset['lag_gws'].mean():.1f}")

    def _print_summary(gdf, label):
        n_tws = gdf['corr_tws'].notna().sum()
        n_gws = gdf['corr_gws'].notna().sum()
        _vprint(f"\n{label} (all wells):")
        _vprint(f"  Wells with TWS correlation: {n_tws} ({n_tws/len(gdf)*100:.1f}%)")
        if not tws_only:
            _vprint(f"  Wells with GWS correlation: {n_gws} ({n_gws/len(gdf)*100:.1f}%)")
        _print_corr_stats(gdf)
        if 'depth_class' in gdf.columns:
            for dc in ['Shallow', 'Deep']:
                sub = gdf[gdf['depth_class'] == dc]
                n_dc = sub['corr_tws'].notna().sum()
                if n_dc == 0:
                    continue
                _vprint(f"  {label} — {dc} ({n_dc} wells):")
                _print_corr_stats(sub, indent="    ")

    _print_summary(well_gdf_raw, "Raw")
    _print_summary(well_gdf_res, "Residual")
    
    # ------------------------------------------------------------------
    # Keep only wells with a valid TWS correlation (corr_tws not NaN)
    # and filter associated series dictionaries accordingly.
    # ------------------------------------------------------------------
    valid_mask = well_gdf_raw['corr_tws'].notna()
    n_before = len(well_gdf_raw)
    well_gdf_raw = well_gdf_raw[valid_mask].copy()
    well_gdf_res = well_gdf_res[valid_mask].copy()
    n_after = len(well_gdf_raw)
    if n_after < n_before:
        _vprint(f"\nFiltered wells by TWS correlation: kept {n_after}/{n_before} wells with non-NaN corr_tws.")

    # Concise summary (always printed; use verbose=True for the full breakdown)
    print(
        f"Wells: {n_before} evaluated, {n_after} kept with valid TWS correlation "
        f"({method}, max lag {max_lag_months} mo)"
    )
    if 'depth_class' in well_gdf_res.columns:
        _sh = well_gdf_res[well_gdf_res['depth_class'] == 'Shallow']
        if len(_sh) > 0:
            print(
                f"Shallow: n={len(_sh)}  TWS residual median rho={_sh['corr_tws'].median():.2f}"
                + (f"  GWS n={int(_sh['corr_gws'].notna().sum())}" if not tws_only else "")
            )

    # Filter well_series to only wells that remain in well_gdf_raw
    allowed_well_keys = set()
    for _, row in well_gdf_raw.iterrows():
        allowed_well_keys.add(_get_well_ts_key(row))
    well_series_filtered = {k: v for k, v in well_series.items() if k in allowed_well_keys}

    # Filter grace_series to only GRACE pixels still referenced by remaining wells
    allowed_grace_keys = set(
        zip(well_gdf_raw['grace_lat'].astype(float), well_gdf_raw['grace_lon'].astype(float))
    )
    grace_series_filtered = {k: v for k, v in grace_series.items() if k in allowed_grace_keys}
    
    well_gdf_raw.attrs['residual'] = False
    well_gdf_res.attrs['residual'] = True
    return {
        'raw': well_gdf_raw,
        'residual': well_gdf_res,
        'well_series': well_series_filtered,
        'grace_series': grace_series_filtered,
    }


def plot_grace_well_timeseries_comparison(
    wells_gdf,
    well_series_precomputed,
    grace_series_precomputed,
    well_time_series=None,
    grace_mean=None,
    grace_gws=None,
    rainfall_coarse=None,
    locations=None,
    save_dir=None,
    figsize=(12, 6),
    remove_mean=True,
    max_wells_per_plot=None,
    residual=None,
    two_panels=False,
    min_common_dates=120,
    max_lag_months=36,
    plot_gws=True,
    show_plot = False,
    file_format='jpeg'
):
    """
    Plot time series comparisons of GRACE TWS, GWS (optional), well measurements, and rainfall for specific locations.
    Requires precomputed series from correlate_wells_with_grace (no fallback computation).
    
    Investigates correlations by plotting all time series together for each GRACE pixel location.
    
    Parameters
    ----------
    wells_gdf : geopandas.GeoDataFrame
        GeoDataFrame with well locations and GRACE assignments.
        Must have columns: 'ID', 'Latitude', 'Longitude', 'grace_lat', 'grace_lon', 'Country'
    well_series_precomputed : dict
        From correlate_wells_with_grace: well_id -> {'anomaly','residual','trend','annual','semi_annual'}.
    grace_series_precomputed : dict
        From correlate_wells_with_grace: (lat,lon) -> {'tws','tws_residual','gws','gws_residual',...}.
    well_time_series : pd.DataFrame, optional
        Unused (precomputed only); kept for API compatibility.
    grace_mean : xarray.DataArray, optional
        Unused (precomputed only); kept for API compatibility.
    grace_gws : xarray.DataArray, optional
        Unused (precomputed only); kept for API compatibility.
    rainfall_coarse : xarray.DataArray, optional
        Rainfall data with dimensions (time, lat, lon) in mm. If provided, will be plotted as blue bars.
    locations : list of tuples, optional
        List of (lat, lon) tuples to plot. If None, generates from all unique (grace_lat, grace_lon) in wells_gdf
    save_dir : str, optional
        Directory to save plots. If None, plots are displayed but not saved
    figsize : tuple, default=(12, 6)
        Figure size for each plot
    remove_mean : bool, default=True
        If True, subtract the mean from each well's time series (plot anomalies) to reduce range differences.
        If False, plot absolute values.
    max_wells_per_plot : int, optional
        Maximum number of wells to plot per GRACE pixel. If None, plots all wells.
        If set, selects the wells with the highest correlations (max of TWS and GWS).
    residual : bool, optional
        If True, titles and saved filenames include "residual" (e.g. for residual correlation runs).
        If None, uses wells_gdf.attrs.get('residual', False).
    two_panels : bool, default=False
        If True, plot two stacked panels: top = raw GRACE+GWL with max corr/lag; bottom = residuals with corr/lag.
        Panels share x-axis, no gap between them. If False, single panel (current behavior).
    min_common_dates : int, default=120
        Minimum overlapping months between well and GRACE required to compute correlation.
        Wells with insufficient overlap are not plotted.
    max_lag_months : int, default=36
        Used when two_panels=True to compute correlations.
    plot_gws : bool, default=True
        If True, plot GRACE GWS. If False, plot only GRACE TWS.

    Returns
    -------
    dict
        Dictionary with keys:
        - 'saved_files': List of file paths where plots were saved
        - 'correlations': List of dicts per well with correlation info:
            - well_id, grace_lat, grace_lon, country, depth_m, depth_class
            - corr_tws_lag0, pvalue_tws_lag0: TWS correlation at lag 0
            - corr_tws_max, lag_tws_max, pvalue_tws_max: Best TWS correlation
            - corr_gws_lag0, pvalue_gws_lag0: GWS correlation at lag 0
            - corr_gws_max, lag_gws_max, pvalue_gws_max: Best GWS correlation
    """
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from pathlib import Path
    import pandas as pd
    import numpy as np
    from scipy.stats import pearsonr

    if well_series_precomputed is None or grace_series_precomputed is None:
        raise ValueError("well_series_precomputed and grace_series_precomputed are required (from correlate_wells_with_grace)")
    if residual is None:
        residual = getattr(wells_gdf, 'attrs', {}).get('residual', False)
    residual_suffix = "_residual" if residual else ""
    residual_title = " (Residual)" if residual else ""

    # ------------------------------------------------------------------------
    # Drop invalid wells if QC flags exist (from correlate_wells_with_grace)
    # ------------------------------------------------------------------------
    wells_gdf_plot = wells_gdf.copy()
    n_before_qc = len(wells_gdf_plot)

    if 'qc_stuck_invalid' in wells_gdf_plot.columns:
        wells_gdf_plot = wells_gdf_plot[~wells_gdf_plot['qc_stuck_invalid'].fillna(False)].copy()

    if 'qc_excluded_reason' in wells_gdf_plot.columns:
        wells_gdf_plot = wells_gdf_plot[wells_gdf_plot['qc_excluded_reason'].isna()].copy()

    n_after_qc = len(wells_gdf_plot)
    if n_after_qc < n_before_qc:
        print(f"Filtered out {n_before_qc - n_after_qc} invalid wells (QC flags) before plotting.")
    
    # Generate locations from geodataframe if not provided
    if locations is None:
        # Get all unique GRACE pixel locations
        wells_with_grace = wells_gdf_plot[
            wells_gdf_plot['grace_lat'].notna() & wells_gdf_plot['grace_lon'].notna()
        ]
        
        if len(wells_with_grace) == 0:
            warnings.warn("No wells with GRACE assignments found. Cannot generate locations.")
            return {'saved_files': [], 'correlations': []}
        
        # Create tuples of unique (grace_lat, grace_lon) combinations
        unique_locations = wells_with_grace[['grace_lat', 'grace_lon']].drop_duplicates()
        locations = [(row['grace_lat'], row['grace_lon']) 
                     for _, row in unique_locations.iterrows()]
        
    # Drop pixels with no usable GRACE TWS *or* GWS series before rendering
    # (same validity rule as the in-loop check; aligns with common TWS/GWS well set).
    def _grace_series_usable(grace_lat, grace_lon):
        key = (float(grace_lat), float(grace_lon))
        if key not in grace_series_precomputed:
            return False
        gs = grace_series_precomputed[key]
        tws = gs.get('tws')
        gws = gs.get('gws', tws)
        if tws is None or gws is None:
            return False
        return int(tws.notna().sum()) > 0 and int(gws.notna().sum()) > 0

    n_loc_before = len(locations)
    locations = [loc for loc in locations if _grace_series_usable(*loc)]
    n_dropped = n_loc_before - len(locations)
    if n_dropped > 0:
        print(
            f"Excluded {n_dropped}/{n_loc_before} GRACE pixels with all-NaN TWS or GWS "
            f"(kept {len(locations)} for plotting)"
        )
    if len(locations) == 0:
        warnings.warn("No GRACE pixels with usable TWS and GWS series. Nothing to plot.")
        return {'saved_files': [], 'correlations': []}

    if save_dir:
        save_path_obj = Path(save_dir)
        save_path_obj.mkdir(parents=True, exist_ok=True)
    
    saved_files = []
    correlation_results = []  # Collect individual well correlation data
    
    # Process each location with a single progress bar (per-location prints removed)
    try:
        from tqdm.auto import tqdm as _tqdm
    except ImportError:
        from tqdm import tqdm as _tqdm
    _verb = "Saving" if save_dir else "Rendering"
    print(f"{_verb} well/GRACE time series plots ({len(locations)} locations)...")
    for loc_idx, (grace_lat, grace_lon) in enumerate(_tqdm(locations, desc="locations", unit="loc")):
        
        # Find all wells assigned to this GRACE pixel
        wells_at_pixel = wells_gdf_plot[
            (wells_gdf_plot['grace_lat'] == grace_lat) & 
            (wells_gdf_plot['grace_lon'] == grace_lon)
        ].copy()
        
        if len(wells_at_pixel) == 0:
            print(f"  No wells found at this GRACE pixel. Skipping...")
            continue
        
        total_wells_at_pixel = len(wells_at_pixel)  # Track total before filtering
        wells_filtered = False
        
        # Limit to top N wells by highest correlation if max_wells_per_plot is set
        if max_wells_per_plot is not None and len(wells_at_pixel) > max_wells_per_plot:
            # Calculate best correlation score (max of TWS and GWS, handling NaN)
            def _get_best_corr(row):
                tws = row.get('corr_tws', np.nan)
                gws = row.get('corr_gws', np.nan)
                if pd.isna(tws) and pd.isna(gws):
                    return -999  # Put wells without correlations at the bottom
                return np.nanmax([tws if pd.notna(tws) else -999, gws if pd.notna(gws) else -999])
            
            wells_at_pixel['_best_corr'] = wells_at_pixel.apply(_get_best_corr, axis=1)
            wells_at_pixel = wells_at_pixel.nlargest(max_wells_per_plot, '_best_corr')
            wells_at_pixel = wells_at_pixel.drop(columns=['_best_corr'])
            wells_filtered = True
        
        # Get countries for this pixel (for filename)
        countries = wells_at_pixel['Country'].unique()
        country_str = '_'.join(sorted(countries)) if len(countries) > 0 else 'unknown'
        
        # Get GRACE series from precomputed (required; locations already pre-filtered)
        grace_key = (float(grace_lat), float(grace_lon))
        gs = grace_series_precomputed[grace_key]
        grace_tws_series = gs['tws'].copy()
        grace_gws_series = gs.get('gws', gs['tws']).copy()
        
        # Extract rainfall time series at this pixel (if provided)
        rainfall_series = None
        if rainfall_coarse is not None:
            try:
                rainfall_ts = rainfall_coarse.sel(lat=grace_lat, lon=grace_lon, method='nearest')
                rainfall_series = pd.Series(
                    rainfall_ts.values,
                    index=pd.to_datetime(rainfall_ts.time.values)
                )
                # Handle timezone alignment
                if grace_tws_series.index.tz is not None:
                    if rainfall_series.index.tz is None:
                        rainfall_series.index = rainfall_series.index.tz_localize('UTC')
                    elif rainfall_series.index.tz != grace_tws_series.index.tz:
                        rainfall_series.index = rainfall_series.index.tz_convert(grace_tws_series.index.tz)
            except Exception as e:
                print(f"  Warning: Could not extract rainfall data: {e}")
                rainfall_series = None
        
        # Prepare raw and residual series from precomputed
        grace_tws_raw = gs['tws'].dropna()
        grace_gws_raw = gs.get('gws', gs['tws']).dropna()
        grace_tws_res = gs['tws_residual'].dropna()
        grace_gws_res = gs.get('gws_residual', gs['tws_residual']).dropna()
        
        if two_panels:
            fig, (ax1_top, ax1_bot) = plt.subplots(2, 1, figsize=(figsize[0], figsize[1] * 1.6), sharex=True, gridspec_kw={'hspace': 0.05})
            ax1_top.tick_params(axis='x', labelbottom=False)
            axes_list = [(ax1_top, 'raw'), (ax1_bot, 'residual')]
        else:
            if residual:
                grace_tws_plot = grace_tws_res
                grace_gws_plot = grace_gws_res
            else:
                grace_tws_plot = grace_tws_raw
                grace_gws_plot = grace_gws_raw
            fig, ax1 = plt.subplots(figsize=figsize)
            axes_list = [(ax1, 'single')]
        
        # Define panels to plot
        raw_grace_ylabel = 'TWSA/GWSA (cm)' if plot_gws else 'TWSA (cm)'
        res_grace_ylabel = 'TWSA/GWSA Residual (cm)' if plot_gws else 'TWSA Residual (cm)'
        if two_panels:
            panels = [
                (ax1_top, 'raw', grace_tws_raw, grace_gws_raw, raw_grace_ylabel, 'GWLA (m)' if remove_mean else 'Well Water Level (m)'),
                (ax1_bot, 'residual', grace_tws_res, grace_gws_res, res_grace_ylabel, 'GWLA Residual (m)'),
            ]
        else:
            panels = [(ax1, 'single', grace_tws_plot, grace_gws_plot,
                       res_grace_ylabel if residual else raw_grace_ylabel,
                       'GWLA Residual (m)' if residual else ('GWLA (m)' if remove_mean else 'GWLA (m)'))]
        
        # Pre-collect well data from precomputed (required). wells_gdf only defines which wells; correlation is computed from series (or from residual flag in one-panel).
        well_data_list = []
        for _, well_row in wells_at_pixel.iterrows():
            well_ts_key = _get_well_ts_key(well_row)
            if well_ts_key not in well_series_precomputed:
                continue
            ws = well_series_precomputed[well_ts_key]
            well_anomaly = ws['anomaly'].dropna()
            well_residual = ws['residual'].dropna()
            if len(well_anomaly) == 0 and len(well_residual) == 0:
                continue
            if well_anomaly.index.tz is None and grace_tws_series.index.tz is not None:
                well_anomaly = well_anomaly.copy()
                well_anomaly.index = well_anomaly.index.tz_localize('UTC')
                well_residual = well_residual.copy()
                well_residual.index = well_residual.index.tz_localize('UTC')
            well_data_list.append((well_row, well_anomaly, well_residual))
        
        if len(well_data_list) == 0:
            plt.close(fig)
            print("  No wells with precomputed series at this pixel. Skipping plot...")
            continue
        
        well_colors = plt.cm.tab10(np.linspace(0, 1, min(len(well_data_list), 10)))
        # In one-panel mode, store computed correlation per well so legend and return dict match (both from residual flag)
        computed_single = {}
        plotted_well_ids = set()  # Wells actually plotted (have valid correlation in at least one panel)

        for panel_idx, (ax1_curr, mode, g_tws, g_gws, grace_ylabel, well_ylabel) in enumerate(panels):
            tws_lbl = 'GRACE TWSA residual' if (mode == 'residual' or (mode == 'single' and residual)) else 'GRACE TWSA'
            gws_lbl = 'GRACE GWSA residual' if (mode == 'residual' or (mode == 'single' and residual)) else 'GRACE GWSA'
            ax1_curr.plot(
                g_tws.index, g_tws.values, label=tws_lbl, color='blue', linewidth=2, alpha=0.8, zorder=2
            )
            if plot_gws:
                ax1_curr.plot(
                    g_gws.index, g_gws.values, label=gws_lbl, color='green', linewidth=2, alpha=0.8, zorder=2
                )
            ax1_curr.set_ylabel(grace_ylabel, fontsize=14, color='black')
            ax1_curr.yaxis.tick_left()
            ax1_curr.yaxis.set_label_position("left")
            ax1_curr.tick_params(axis='y', labelsize=12, labelcolor='black')
            ax1_curr.tick_params(axis='x', labelsize=12)
            # Grid from GRACE TWS (left) axis only; behind lines; twins have no grid (avoids double y-scale grids)
            ax1_curr.set_axisbelow(True)
            ax1_curr.grid(True, axis="both", alpha=0.4, linestyle="--")
            if two_panels and panel_idx == 0:
                ax1_curr.tick_params(axis='x', labelbottom=False)
            
            ax2_curr = ax1_curr.twinx()
            ax2_curr.grid(False)
            ax2_curr.set_axisbelow(False)
            ax2_curr.yaxis.tick_right()
            ax2_curr.yaxis.set_label_position("right")
            ax2_curr.spines['left'].set_visible(False)
            
            for well_idx, (well_row, well_anomaly, well_residual) in enumerate(well_data_list):
                well_id = well_row['ID']
                if mode == 'single':
                    well_ts_plot = well_residual if residual else well_anomaly
                    grace_for_corr = grace_tws_res if residual else grace_tws_raw
                    tws_res = _calculate_best_lag_correlation(well_ts_plot, grace_for_corr, max_lag_months, min_common_dates, pearsonr)
                    r_tws = tws_res['r_max']
                    lag_tws = tws_res['lag_max'] if pd.notna(tws_res['lag_max']) else 0
                    computed_single[well_id] = {
                        'corr_tws': r_tws, 'lag_tws': lag_tws,
                        'corr_tws_lag0': tws_res.get('r_lag0', np.nan), 'pvalue_tws_lag0': tws_res.get('p_lag0', np.nan),
                        'pvalue_tws': tws_res.get('p_max', np.nan),
                    }
                else:
                    well_ts_plot = well_residual if mode == 'residual' else well_anomaly
                    grace_for_corr = grace_tws_res if mode == 'residual' else grace_tws_raw
                    tws_res = _calculate_best_lag_correlation(well_ts_plot, grace_for_corr, max_lag_months, min_common_dates, pearsonr)
                    r_tws = tws_res['r_max']
                    lag_tws = tws_res['lag_max'] if pd.notna(tws_res['lag_max']) else 0
                
                color = well_colors[well_idx % len(well_colors)]
                well_depth = well_row.get('avg_depth_m', np.nan)
                if pd.isna(well_depth):
                    well_depth = well_row.get('first_measurement_depth_m', np.nan)
                
                if pd.isna(r_tws):
                    continue  # Skip wells with no correlation (insufficient overlap)
                plotted_well_ids.add(well_id)
                well_label = f"Well {well_id}"
                if pd.notna(well_depth):
                    well_label += f" (depth = {well_depth:.0f} m)"
                well_label += f" | ρ(TWSA) = {r_tws:.2f} Lag = {int(lag_tws)}"
                #if has_gws_corr:
                    #lag_gws = well_row.get('lag_gws', 0)
                    #well_label += f" | r_GWS = {well_row['corr_gws']:.2f} Lag = {int(lag_gws) if pd.notna(lag_gws) else 0}"
                
                ax2_curr.plot(
                    well_ts_plot.index, well_ts_plot.values,
                    label=well_label, color=color, linewidth=1.5,
                    alpha=0.7, marker='o', markersize=3, zorder=3,
                )
            
            ax2_curr.set_ylabel(well_ylabel, fontsize=14, color='red')
            ax2_curr.yaxis.tick_right()
            ax2_curr.tick_params(axis='y', labelsize=12, labelcolor='red')
            
            # Rainfall on this panel (if provided)
            if rainfall_series is not None and len(rainfall_series) > 0:
                ax3_curr = ax2_curr.twinx()
                ax3_curr.spines['right'].set_position(('outward', 60))
                ax3_curr.yaxis.set_label_position("right")
                ax3_curr.yaxis.set_ticks_position("right")
                ax3_curr.spines['left'].set_visible(False)
                ax3_curr.spines['right'].set_visible(True)
                ax3_curr.grid(False)
                if len(rainfall_series) > 1:
                    time_delta = (rainfall_series.index[1] - rainfall_series.index[0]).days
                    bar_width = time_delta * 0.7
                else:
                    bar_width = 30
                ax3_curr.bar(rainfall_series.index, rainfall_series.values, width=bar_width, alpha=0.5,
                             color='blue', label='Rainfall', align='center', zorder=3)
                ax3_curr.set_ylabel('Rainfall (mm)', fontsize=14, color='blue')
                ax3_curr.tick_params(axis='y', which='both', labelsize=12, labelcolor='blue', right=True, labelright=True, left=False, labelleft=False)
                lines1, labels1 = ax1_curr.get_legend_handles_labels()
                lines2, labels2 = ax2_curr.get_legend_handles_labels()
                lines3, labels3 = ax3_curr.get_legend_handles_labels()
                if two_panels and panel_idx == 1:
                    # Lower panel: wells only (GRACE TWS, GWS, Rainfall are in upper panel)
                    ax1_curr.legend(lines2, labels2, loc='upper left', fontsize=11, framealpha=0.9)
                else:
                    ax1_curr.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best', fontsize=11, framealpha=0.9)
            else:
                lines1, labels1 = ax1_curr.get_legend_handles_labels()
                lines2, labels2 = ax2_curr.get_legend_handles_labels()
                if two_panels and panel_idx == 1:
                    # Lower panel: wells only
                    ax1_curr.legend(lines2, labels2, loc='upper left', fontsize=11, framealpha=0.9)
                else:
                    ax1_curr.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=11, framealpha=0.9)
        
        # Collect correlation data: in one-panel mode use computed values (from residual flag); else from wells_gdf
        for well_row, _, _ in well_data_list:
            well_id = well_row['ID']
            well_depth = well_row.get('avg_depth_m', np.nan)
            if pd.isna(well_depth):
                well_depth = well_row.get('first_measurement_depth_m', np.nan)
            if not two_panels and well_id in computed_single:
                c = computed_single[well_id]
                corr_tws_lag0, pval_tws_lag0 = c.get('corr_tws_lag0', np.nan), c.get('pvalue_tws_lag0', np.nan)
                corr_tws_max, lag_tws_max = c.get('corr_tws', np.nan), c.get('lag_tws', np.nan)
                pval_tws_max = c.get('pvalue_tws', np.nan)
            else:
                corr_tws_lag0 = well_row.get('corr_tws_lag0', np.nan)
                pval_tws_lag0 = well_row.get('pvalue_tws_lag0', np.nan)
                corr_tws_max = well_row.get('corr_tws', np.nan)
                lag_tws_max = well_row.get('lag_tws', np.nan)
                pval_tws_max = well_row.get('pvalue_tws', np.nan)
            correlation_results.append({
                'well_id': well_id,
                'grace_lat': grace_lat,
                'grace_lon': grace_lon,
                'country': well_row.get('Country', 'unknown'),
                'depth_m': well_depth,
                'depth_class': well_row.get('depth_class', None),
                # TWS correlation (from computed_single in one-panel mode, else wells_gdf)
                'corr_tws_lag0': corr_tws_lag0,
                'pvalue_tws_lag0': pval_tws_lag0,
                'corr_tws_max': corr_tws_max,
                'lag_tws_max': lag_tws_max,
                'pvalue_tws_max': pval_tws_max,
                # GWS correlation
                'corr_gws_lag0': well_row.get('corr_gws_lag0', np.nan),
                'pvalue_gws_lag0': well_row.get('pvalue_gws_lag0', np.nan),
                'corr_gws_max': well_row.get('corr_gws', np.nan),
                'lag_gws_max': well_row.get('lag_gws', np.nan),
                'pvalue_gws_max': well_row.get('pvalue_gws', np.nan),
            })

        # Skip plot if no wells had valid correlation
        if len(plotted_well_ids) == 0:
            plt.close(fig)
            print(f"  No wells with valid correlation at this pixel. Skipping plot...")
            continue

        # Formatting - single line title (on top panel). Use plotted_well_ids so title matches wells actually plotted.
        n_plotted = len(plotted_well_ids)
        title_ax = panels[0][0]
        if wells_filtered:
            title_wells = f"{n_plotted} of {total_wells_at_pixel} wells, top by corr"
        else:
            title_wells = f"{n_plotted} well{'s' if n_plotted != 1 else ''}"
        title_str = 'Time Series Comparison (Raw + Residual)' if two_panels else f'Time Series Comparison{residual_title}'
        title_ax.set_title(f'{title_str} ({country_str}) for GRACE Pixel ({grace_lat:.2f}°, {grace_lon:.2f}°) and GWL ({title_wells})', 
                    fontsize=14, pad=10)
        
        # Format x-axis dates (bottom panel when two_panels, else single panel)
        x_ax = panels[-1][0]
        #x_ax.set_xlabel('Date', fontsize=14)
        x_ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        x_ax.xaxis.set_major_locator(mdates.YearLocator())
        x_mins, x_maxs = [], []
        for _, _, g_tws, _, _, _ in panels:
            if len(g_tws.index) > 0:
                x_mins.append(pd.to_datetime(g_tws.index.min()))
                x_maxs.append(pd.to_datetime(g_tws.index.max()))
        if x_mins and x_maxs:
            x_ax.set_xlim(min(x_mins), max(x_maxs))
        x_ax.margins(x=0)
        plt.setp(x_ax.xaxis.get_majorticklabels(), rotation=45, fontsize=12)
        
        # Add 25% buffer to y-axis limits
        def _add_ylim_buffer(ax, buffer_frac=0.25, min_at_zero=False):
            ymin, ymax = ax.get_ylim()
            yrange = ymax - ymin
            new_min = 0 if min_at_zero else ymin - yrange * buffer_frac
            ax.set_ylim(new_min, ymax + yrange * buffer_frac)
        
        for ax in fig.axes:
            lbl = ax.get_ylabel() or ''
            if 'Rainfall' in str(lbl):
                _add_ylim_buffer(ax, buffer_frac=0.50, min_at_zero=True)
            else:
                _add_ylim_buffer(ax)
        
        # FINAL tick enforcement
        for ax1_curr, _, _, _, _, _ in panels:
            ax1_curr.yaxis.set_ticks_position('left')
            ax1_curr.yaxis.set_label_position('left')
            for label in ax1_curr.yaxis.get_ticklabels():
                label.set_color('black')
        for ax in fig.axes:
            lbl = ax.get_ylabel() or ''
            if ('GWL' in str(lbl) or 'Well' in str(lbl) or 'Anomaly' in str(lbl)):
                ax.yaxis.set_ticks_position('right')
                ax.yaxis.set_label_position('right')
                for label in ax.yaxis.get_ticklabels():
                    label.set_color('red')
        
        with warnings.catch_warnings():
            # Some panels use twin axes that tight_layout cannot handle exactly
            warnings.simplefilter("ignore", UserWarning)
            plt.tight_layout()
        if two_panels:
            plt.subplots_adjust(hspace=0.05)

        # Save if directory provided
        if save_dir:
            # Create filename based on lat, lon, and country (include residual in name when applicable)
            filename = f"timeseries{residual_suffix}_lat{grace_lat:.2f}_lon{grace_lon:.2f}_{country_str}.{file_format}"
            save_path = save_path_obj / filename
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            saved_files.append(str(save_path))
        elif show_plot == True:
            plt.show()
        
        plt.close()
    
    if save_dir:
        print(f"Saved {len(saved_files)} plots to: {save_dir}")
    
    # Return both saved files and correlation results (for distribution analysis)
    return {'saved_files': saved_files, 'correlations': correlation_results}


def _corr_dist_state_and_variable(variable: str) -> tuple:
    """Map internal variable label to (state, grace_variable) for summary tables."""
    mapping = {
        'TWS': ('anomaly', 'TWSA'),
        'TWS residual': ('residual', 'TWSA'),
        'GWS': ('anomaly', 'GWSA'),
        'GWS residual': ('residual', 'GWSA'),
    }
    return mapping.get(variable, ('unknown', str(variable)))


def _is_finite_corr(val) -> bool:
    """True when *val* is a usable correlation coefficient."""
    import numpy as np
    import pandas as pd

    try:
        x = float(val)
    except (TypeError, ValueError):
        return False
    return pd.notna(x) and np.isfinite(x)


def _well_has_common_tws_gws(row, lag_type: str) -> bool:
    """True when both TWSA and GWSA correlations exist for *lag_type* ('Lag 0' or 'Max Lag')."""
    if lag_type == 'Lag 0':
        tws_col, gws_col = 'corr_tws_lag0', 'corr_gws_lag0'
    elif lag_type == 'Max Lag':
        tws_col, gws_col = 'corr_tws', 'corr_gws'
    else:
        raise ValueError("lag_type must be 'Lag 0' or 'Max Lag'")
    return _is_finite_corr(row.get(tws_col)) and _is_finite_corr(row.get(gws_col))


def _count_common_tws_gws_wells(gdf) -> dict:
    """Count wells with TWS only, GWS only, or both, per lag type."""
    import pandas as pd

    out = {}
    if gdf is None or len(gdf) == 0:
        for lag_type in ('Lag 0', 'Max Lag'):
            out[lag_type] = {'n_total': 0, 'n_tws': 0, 'n_gws': 0, 'n_common': 0, 'n_tws_only': 0, 'n_gws_only': 0}
        return out
    for lag_type in ('Lag 0', 'Max Lag'):
        tws_col = 'corr_tws_lag0' if lag_type == 'Lag 0' else 'corr_tws'
        gws_col = 'corr_gws_lag0' if lag_type == 'Lag 0' else 'corr_gws'
        tws_ok = gdf[tws_col].map(_is_finite_corr) if tws_col in gdf.columns else pd.Series(False, index=gdf.index)
        gws_ok = gdf[gws_col].map(_is_finite_corr) if gws_col in gdf.columns else pd.Series(False, index=gdf.index)
        common = tws_ok & gws_ok
        out[lag_type] = {
            'n_total': int(len(gdf)),
            'n_tws': int(tws_ok.sum()),
            'n_gws': int(gws_ok.sum()),
            'n_common': int(common.sum()),
            'n_tws_only': int((tws_ok & ~gws_ok).sum()),
            'n_gws_only': int((gws_ok & ~tws_ok).sum()),
        }
    return out


def _parse_corr_pvalue(val):
    """Parse a stored p-value to a float.

    Individual-well p-values are stored as strings ('<0.01', '<0.05', '0.37');
    depth-class p-values are floats. Strings like '<0.01'/'<0.05' are mapped just
    below the stated threshold so they count as significant at that level but not
    a stricter one. Missing/None -> NaN.
    """
    import numpy as np

    if val is None:
        return np.nan
    if isinstance(val, str):
        s = val.strip()
        if not s:
            return np.nan
        if s.startswith('<'):
            try:
                return float(s[1:]) - 1e-9
            except ValueError:
                return np.nan
        if s.startswith('>'):
            try:
                return float(s[1:]) + 1e-9
            except ValueError:
                return np.nan
        try:
            return float(s)
        except ValueError:
            return np.nan
    try:
        return float(val)
    except (TypeError, ValueError):
        return np.nan


def _build_correlation_distribution_stats_df(
    stats_summary: dict,
    active_depth_order: list,
) -> "pd.DataFrame":
    """Tidy summary: one row per depth class × state × GRACE variable × lag type."""
    import pandas as pd

    lag_labels = {'Lag 0': 'zero_lag', 'Max Lag': 'optimal_lag'}
    records = []
    for key, s in stats_summary.items():
        matched = False
        for dc in active_depth_order:
            suffix = f"_{dc}"
            if not key.endswith(suffix):
                continue
            stem = key[: -len(suffix)]
            for lag_type, lag_label in lag_labels.items():
                lag_suffix = f"_{lag_type}"
                if not stem.endswith(lag_suffix):
                    continue
                variable = stem[: -len(lag_suffix)]
                state, grace_variable = _corr_dist_state_and_variable(variable)
                records.append({
                    'depth_class': dc,
                    'state': state,
                    'grace_variable': grace_variable,
                    'lag_type': lag_label,
                    'n': s.get('n'),
                    'std': s.get('std'),
                    'median': s.get('median'),
                    'mean': s.get('mean'),
                    'min': s.get('min'),
                    'max': s.get('max'),
                    'n_sig_05': s.get('n_sig_05'),
                    'pct_sig_05': s.get('pct_sig_05'),
                    'n_sig_01': s.get('n_sig_01'),
                    'pct_sig_01': s.get('pct_sig_01'),
                    'avg_lag': s.get('avg_lag'),
                    'lag_std': s.get('lag_std'),
                    'lag_median': s.get('lag_median'),
                })
                matched = True
                break
            if matched:
                break

    if not records:
        return pd.DataFrame(
            columns=[
                'depth_class', 'state', 'grace_variable', 'lag_type',
                'n', 'std', 'median', 'mean', 'min', 'max',
                'n_sig_05', 'pct_sig_05', 'n_sig_01', 'pct_sig_01',
                'avg_lag', 'lag_std', 'lag_median',
            ]
        )

    df = pd.DataFrame(records)
    state_order = pd.Categorical(df['state'], categories=['anomaly', 'residual'], ordered=True)
    var_order = pd.Categorical(df['grace_variable'], categories=['TWSA', 'GWSA'], ordered=True)
    lag_order = pd.Categorical(df['lag_type'], categories=['zero_lag', 'optimal_lag'], ordered=True)
    depth_order = pd.Categorical(
        df['depth_class'],
        categories=[d for d in active_depth_order if d in df['depth_class'].unique()],
        ordered=True,
    )
    return (
        df.assign(state=state_order, grace_variable=var_order, lag_type=lag_order, depth_class=depth_order)
        .sort_values(['depth_class', 'state', 'grace_variable', 'lag_type'])
        .reset_index(drop=True)
    )


def plot_correlation_distributions(
    wells_gdf=None,
    wells_gdf_raw=None,
    wells_gdf_residual=None,
    depthclass_correlations=None,
    save_path=None,
    figsize=(12, 10),
    show_stats=True,
    residual=None,
    require_common_tws_gws=False,
    show_cdf=True,
    show_tables=True,
):
    """
    (1) Violin + box distributions in 2×2 (individual raw+residual) or depth-class layout, unchanged
        from the original design: Lag 0 | Max Lag × anomaly | residual as applicable.

    (2) **Additional** figure: empirical **CDF** (%) of **max-lag** correlations, 1×2 panels TWS | GWS.
        Light grid (alpha=0.5) behind curves, no vertical threshold lines, legend lower right.
        No figure or axes titles on the CDF plot (axis labels only).

    (3) Summary tables in a third small figure.

    Accepts:
    - wells_gdf_raw + wells_gdf_residual: individual wells
    - wells_gdf: single GeoDataFrame (backward compat)
    - depthclass_correlations: depth-class averaged correlations per pixel

    Returns
    -------
    dict
        ``stats_summary`` — nested stats keyed by ``'{variable}_{lag_type}_{depth_class}'``.
        ``stats_df`` — tidy :class:`pandas.DataFrame` with one row per depth class × state
        (anomaly/residual) × GRACE variable (TWSA/GWSA) × lag type (zero_lag/optimal_lag),
        including correlation ``std`` / ``median`` / ``mean``, significance counts
        ``n_sig_05`` / ``pct_sig_05`` and ``n_sig_01`` / ``pct_sig_01`` (percent of wells
        with valid p-value below 0.05 / 0.01), and for optimal-lag rows ``avg_lag``,
        ``lag_std``, ``lag_median`` (spread of chosen lags, in months).

        Note: ``pct_sig_*`` is descriptive only. Significance is not used to filter the
        distribution (which would bias |ρ| upward), and optimal-lag p-values are not
        corrected for the multi-lag search or serial correlation.

    show_cdf : bool, default True
        If False, skip the extra empirical-CDF figure.
    show_tables : bool, default True
        If False, skip the summary-table figure.
    require_common_tws_gws : bool, default False
        When True (individual-well mode), keep only wells with valid TWSA **and** GWSA
        correlations for each lag type separately (zero lag and optimal lag). This makes
        TWSA and GWSA violin counts match within each panel and state (anomaly/residual).
        ``filter_counts`` in the return dict reports before/after well counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MultipleLocator
    import pandas as pd
    import numpy as np
    
    use_individual = wells_gdf_raw is not None and wells_gdf_residual is not None
    has_residual_layer = False  # depth-class dicts include residual block
    if residual is None:
        residual = getattr(wells_gdf, 'attrs', {}).get('residual', False) if wells_gdf is not None else False
    residual_title = " (Residual)" if residual else ""
    residual_stem = "_residual" if residual else ""
    
    data_records = []
    filter_counts = {}
    
    if use_individual:
        for gdf, mode in [(wells_gdf_raw, ''), (wells_gdf_residual, ' residual')]:
            state_label = 'anomaly' if mode == '' else 'residual'
            filter_counts[state_label] = _count_common_tws_gws_wells(gdf)
            for _, row in gdf.iterrows():
                depth_class = row.get('depth_class', None)
                if pd.isna(depth_class):
                    continue
                lag_pairs = [
                    ('Lag 0', 'corr_tws_lag0', 'pvalue_tws_lag0', 0,
                     'corr_gws_lag0', 'pvalue_gws_lag0', 0),
                    ('Max Lag', 'corr_tws', 'pvalue_tws', 'lag_tws',
                     'corr_gws', 'pvalue_gws', 'lag_gws'),
                ]
                for lag_type, tws_corr_col, tws_p_col, tws_lag, gws_corr_col, gws_p_col, gws_lag_col in lag_pairs:
                    if require_common_tws_gws and not _well_has_common_tws_gws(row, lag_type):
                        continue
                    gws_lag = row.get(gws_lag_col, np.nan) if lag_type == 'Max Lag' else 0
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': f'TWS{mode}', 'Lag Type': lag_type,
                        'Lag': row.get(tws_lag, 0) if lag_type == 'Max Lag' else 0,
                        'Correlation': row.get(tws_corr_col, np.nan),
                        'Pvalue': row.get(tws_p_col, np.nan),
                    })
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': f'GWS{mode}', 'Lag Type': lag_type,
                        'Lag': gws_lag,
                        'Correlation': row.get(gws_corr_col, np.nan),
                        'Pvalue': row.get(gws_p_col, np.nan),
                    })
        data_source = "Individual Wells"
    elif wells_gdf is not None:
        # Individual well correlations
        if require_common_tws_gws:
            filter_counts['single'] = _count_common_tws_gws_wells(wells_gdf)
        for _, row in wells_gdf.iterrows():
            depth_class = row.get('depth_class', None)
            if pd.isna(depth_class):
                continue
            lag_pairs = [
                ('Lag 0', 'TWS', 'corr_tws_lag0', 'pvalue_tws_lag0', 0,
                 'GWS', 'corr_gws_lag0', 'pvalue_gws_lag0', 0),
                ('Max Lag', 'TWS', 'corr_tws', 'pvalue_tws', 'lag_tws',
                 'GWS', 'corr_gws', 'pvalue_gws', 'lag_gws'),
            ]
            for lag_type, tws_var, tws_corr_col, tws_p_col, tws_lag, gws_var, gws_corr_col, gws_p_col, gws_lag_col in lag_pairs:
                if require_common_tws_gws and not _well_has_common_tws_gws(row, lag_type):
                    continue
                corr_type_tws = f'TWS @ {lag_type}' if lag_type == 'Lag 0' else 'TWS @ Max Lag'
                corr_type_gws = f'GWS @ {lag_type}' if lag_type == 'Lag 0' else 'GWS @ Max Lag'
                data_records.append({
                    'Depth Class': depth_class,
                    'Correlation Type': corr_type_tws,
                    'Correlation': row.get(tws_corr_col, np.nan),
                    'Variable': tws_var,
                    'Lag Type': lag_type,
                    'Lag': row.get(tws_lag, 0) if lag_type == 'Max Lag' else 0,
                    'Pvalue': row.get(tws_p_col, np.nan),
                })
                data_records.append({
                    'Depth Class': depth_class,
                    'Correlation Type': corr_type_gws,
                    'Correlation': row.get(gws_corr_col, np.nan),
                    'Variable': gws_var,
                    'Lag Type': lag_type,
                    'Lag': row.get(gws_lag_col, np.nan) if lag_type == 'Max Lag' else 0,
                    'Pvalue': row.get(gws_p_col, np.nan),
                })
        data_source = "Individual Wells"
        
    elif depthclass_correlations is not None:
        # Depth-class averaged correlations: anomaly + residual when depth_classes_residual present
        has_residual_layer = any(
            'depth_classes_residual' in p and p.get('depth_classes_residual')
            for p in depthclass_correlations
        )
        for pixel_data in depthclass_correlations:
            # Anomaly (raw) correlations
            for depth_class, corr_data in pixel_data.get('depth_classes', {}).items():
                data_records.append({
                    'Depth Class': depth_class, 'Variable': 'TWS', 'Lag Type': 'Lag 0', 'Lag': 0,
                    'Correlation': corr_data.get('r_tws_lag0', np.nan),
                    'Pvalue': corr_data.get('p_tws_lag0', np.nan),
                })
                data_records.append({
                    'Depth Class': depth_class, 'Variable': 'TWS', 'Lag Type': 'Max Lag', 'Lag': corr_data.get('lag_tws_max', np.nan),
                    'Correlation': corr_data.get('r_tws_max', np.nan),
                    'Pvalue': corr_data.get('p_tws_max', np.nan),
                })
                data_records.append({
                    'Depth Class': depth_class, 'Variable': 'GWS', 'Lag Type': 'Lag 0', 'Lag': 0,
                    'Correlation': corr_data.get('r_gws_lag0', np.nan),
                    'Pvalue': corr_data.get('p_gws_lag0', np.nan),
                })
                data_records.append({
                    'Depth Class': depth_class, 'Variable': 'GWS', 'Lag Type': 'Max Lag', 'Lag': corr_data.get('lag_gws_max', np.nan),
                    'Correlation': corr_data.get('r_gws_max', np.nan),
                    'Pvalue': corr_data.get('p_gws_max', np.nan),
                })
            # Residual correlations (when available)
            if has_residual_layer:
                for depth_class, corr_data in pixel_data.get('depth_classes_residual', {}).items():
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': 'TWS residual', 'Lag Type': 'Lag 0', 'Lag': 0,
                        'Correlation': corr_data.get('r_tws_lag0', np.nan),
                        'Pvalue': corr_data.get('p_tws_lag0', np.nan),
                    })
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': 'TWS residual', 'Lag Type': 'Max Lag', 'Lag': corr_data.get('lag_tws_max', np.nan),
                        'Correlation': corr_data.get('r_tws_max', np.nan),
                        'Pvalue': corr_data.get('p_tws_max', np.nan),
                    })
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': 'GWS residual', 'Lag Type': 'Lag 0', 'Lag': 0,
                        'Correlation': corr_data.get('r_gws_lag0', np.nan),
                        'Pvalue': corr_data.get('p_gws_lag0', np.nan),
                    })
                    data_records.append({
                        'Depth Class': depth_class, 'Variable': 'GWS residual', 'Lag Type': 'Max Lag', 'Lag': corr_data.get('lag_gws_max', np.nan),
                        'Correlation': corr_data.get('r_gws_max', np.nan),
                        'Pvalue': corr_data.get('p_gws_max', np.nan),
                    })
        data_source = "Depth-Class Averages"
    else:
        raise ValueError("Must provide either wells_gdf or depthclass_correlations")
    
    df = pd.DataFrame(data_records)
    df = df.dropna(subset=['Correlation'])
    if 'Pvalue' not in df.columns:
        df['Pvalue'] = np.nan
    df['Pvalue_num'] = df['Pvalue'].map(_parse_corr_pvalue)
    
    if len(df) == 0:
        print("No valid correlation data found!")
        return {'stats_summary': {}, 'stats_df': pd.DataFrame(), 'filter_counts': filter_counts}

    use_four_var_table = use_individual or has_residual_layer
    
    # Colors: TWSA blue, GWSA orange (consistent across anomaly/residual and depth classes)
    _grace_var_colors = {'TWS': TWSA_PLOT_COLOR, 'GWS': GWSA_PLOT_COLOR}
    depth_colors = {
        'Shallow': '#3498db',
        'Deep': '#e74c3c',
    }
    
    # Order depth classes (only those present in this run)
    depth_order = ['Shallow', 'Deep']
    active_depth_order = [dc for dc in depth_order if (df['Depth Class'] == dc).any()]
    if not active_depth_order:
        active_depth_order = sorted(df['Depth Class'].dropna().astype(str).unique().tolist())
    df['Depth Class'] = pd.Categorical(df['Depth Class'], categories=active_depth_order, ordered=True)
    
    # Build stats_summary
    var_list = ['TWS', 'TWS residual', 'GWS', 'GWS residual'] if use_four_var_table else ['TWS', 'GWS']
    stats_summary = {}
    for variable in var_list:
        for lag_type in ['Lag 0', 'Max Lag']:
            subset = df[(df['Variable'] == variable) & (df['Lag Type'] == lag_type)]
            for dc in active_depth_order:
                dc_subset = subset[subset['Depth Class'] == dc]
                if len(dc_subset) == 0:
                    continue
                dc_data = dc_subset['Correlation']
                if lag_type == 'Max Lag' and 'Lag' in dc_subset.columns:
                    lag_vals = pd.to_numeric(dc_subset['Lag'], errors='coerce').dropna()
                    avg_lag = lag_vals.mean() if len(lag_vals) else np.nan
                    lag_std = lag_vals.std() if len(lag_vals) > 1 else (0.0 if len(lag_vals) == 1 else np.nan)
                    lag_median = lag_vals.median() if len(lag_vals) else np.nan
                else:
                    avg_lag = 0.0
                    lag_std = 0.0
                    lag_median = 0.0
                # Significance: % of wells with a valid p-value below threshold
                pvals = pd.to_numeric(dc_subset['Pvalue_num'], errors='coerce') if 'Pvalue_num' in dc_subset.columns else pd.Series(dtype=float)
                pvals = pvals.dropna()
                n_p = int(len(pvals))
                if n_p > 0:
                    n_sig_05 = int((pvals < 0.05).sum())
                    n_sig_01 = int((pvals < 0.01).sum())
                    pct_sig_05 = 100.0 * n_sig_05 / n_p
                    pct_sig_01 = 100.0 * n_sig_01 / n_p
                else:
                    n_sig_05 = n_sig_01 = 0
                    pct_sig_05 = pct_sig_01 = np.nan
                key = f'{variable}_{lag_type}_{dc}'
                stats_summary[key] = {
                    'n': len(dc_data), 'median': dc_data.median(), 'mean': dc_data.mean(),
                    'std': dc_data.std(), 'min': dc_data.min(), 'max': dc_data.max(),
                    'n_sig_05': n_sig_05, 'pct_sig_05': pct_sig_05,
                    'n_sig_01': n_sig_01, 'pct_sig_01': pct_sig_01,
                    'avg_lag': avg_lag, 'lag_std': lag_std, 'lag_median': lag_median,
                }

    stats_df = _build_correlation_distribution_stats_df(stats_summary, active_depth_order)
    
    FONT_LABEL = 13
    FONT_TICK = 12
    FONT_TITLE = 14
    FONT_ANNOT = 12
    # CDF panels: slightly larger again for readability
    CDF_FS_AXIS = 15
    CDF_FS_TICK = 13
    CDF_FS_LEGEND = 13

    def _draw_panel(ax, variables, lag_type, title, show_ylabel=True):
        """Draw one panel: 4 groups (TWS Shallow, TWS Deep, GWS Shallow, GWS Deep)."""
        positions, data_groups, labels, colors_list, hatches = [], [], [], [], []
        for var in variables:
            for dc in active_depth_order:
                subset = df[(df['Variable'] == var) & (df['Lag Type'] == lag_type) & (df['Depth Class'] == dc)]
                vals = subset['Correlation'].dropna().values
                if len(vals) == 0:
                    continue
                positions.append(len(positions))
                data_groups.append(vals)
                var_short = 'TWSA' if 'TWS' in var else 'GWSA'
                labels.append(var_short)
                grace_key = 'TWS' if 'TWS' in var else 'GWS'
                colors_list.append(_grace_var_colors[grace_key])
                hatches.append(None)
        if len(data_groups) == 0:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=FONT_ANNOT)
            ax.set_ylim(-1.05, 1.25)
            ax.set_yticks(np.arange(-1.0, 1.01, 0.2))
            ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            return
        parts = ax.violinplot(data_groups, positions=positions, showmeans=False, showmedians=False, showextrema=False)
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.6)
            pc.set_edgecolor('gray')
            pc.set_linewidth(0.8)
        bp = ax.boxplot(data_groups, positions=positions, widths=0.2, patch_artist=True,
                        showfliers=True, flierprops=dict(marker='o', markersize=4, alpha=0.6, markerfacecolor='none', markeredgecolor='gray'))
        for j, (patch, color, hatch) in enumerate(zip(bp['boxes'], colors_list, hatches)):
            patch.set_facecolor('white')
            patch.set_edgecolor(color)
            patch.set_linewidth(1.2)
            if hatch:
                patch.set_hatch(hatch)
        for line in bp['medians']:
            line.set_color('black')
            line.set_linewidth(1.2)
        np.random.seed(42)
        for i, (pos, dg) in enumerate(zip(positions, data_groups)):
            jitter = np.random.uniform(-0.08, 0.08, len(dg))
            ax.scatter(pos + jitter, dg, c=colors_list[i], alpha=0.5, s=16, edgecolors='none', zorder=2)
        if show_stats:
            npos = max(len(positions), 1)
            for i, (pos, label, dg) in enumerate(zip(positions, labels, data_groups)):
                dc = active_depth_order[i % max(len(active_depth_order), 1)]
                var = 'TWS' if 'TWS' in label else 'GWS'
                var_key = var if variables[0] == 'TWS' else f'{var} residual'
                key = f'{var_key}_{lag_type}_{dc}'
                if key in stats_summary:
                    s = stats_summary[key]
                    ann = f"n={s['n']} med={s['median']:.2f}"
                    if lag_type == 'Max Lag':
                        med_lag = s.get('lag_median', np.nan)
                        if pd.notna(med_lag):
                            ann += f"\nmed lag={float(med_lag):.0f}"
                    x_frac = (pos + 0.5) / npos
                    ax.annotate(ann, xy=(x_frac, 0.98), ha='center', va='top', xycoords='axes fraction',
                                fontsize=FONT_ANNOT,
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='gray', alpha=0.9))
        ax.set_xticks(positions)
        ax.set_xticklabels(labels, fontsize=FONT_TICK)
        ax.set_ylim(-1.05, 1.25)
        ax.set_yticks(np.arange(-1.0, 1.01, 0.2))
        ax.axhline(0, color='gray', linestyle='--', alpha=0.4)
        ax.grid(axis='y', alpha=0.3, linestyle=':')
        ax.tick_params(axis='y', labelsize=FONT_TICK)
        if show_ylabel:
            ax.set_ylabel('Spearman ρ', fontsize=FONT_TICK)

    def _panel_letter(ax, letter, fontsize=11):
        ax.text(
            0.02, 0.98, letter,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, fontweight='bold', zorder=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9),
        )

    _title_pad = 12
    # Match horizontal and vertical gap between violin panels (matplotlib uses same relative units).
    _violin_panel_gap = 0.022

    if use_individual:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
        fig.subplots_adjust(
            wspace=_violin_panel_gap, hspace=_violin_panel_gap,
            left=0.08, right=0.98, top=0.9, bottom=0.1,
        )
        _draw_panel(axes[0, 0], ['TWS', 'GWS'], 'Lag 0', '', show_ylabel=True)
        _draw_panel(axes[0, 1], ['TWS', 'GWS'], 'Max Lag', '', show_ylabel=False)
        _draw_panel(axes[1, 0], ['TWS residual', 'GWS residual'], 'Lag 0', '', show_ylabel=True)
        _draw_panel(axes[1, 1], ['TWS residual', 'GWS residual'], 'Max Lag', '', show_ylabel=False)
        axes[0, 0].set_title('Zero lag', fontsize=FONT_TITLE, fontweight='bold', pad=_title_pad)
        axes[0, 1].set_title('Optimal lag', fontsize=FONT_TITLE, fontweight='bold', pad=_title_pad)
        axes[0, 0].set_ylabel('Spearman ρ (anomaly)', fontsize=FONT_TICK)
        axes[0, 1].set_ylabel('')
        axes[1, 0].set_ylabel('Spearman ρ (residual)', fontsize=FONT_TICK)
        axes[1, 1].set_ylabel('')
        for ax in axes[0, :]:
            ax.set_xticklabels([])
        _panel_letter(axes[0, 0], 'a')
        _panel_letter(axes[0, 1], 'b')
        _panel_letter(axes[1, 0], 'c')
        _panel_letter(axes[1, 1], 'd')
    elif depthclass_correlations is not None:
        fig, axes = plt.subplots(2, 2, figsize=figsize, sharey=True)
        fig.subplots_adjust(
            wspace=_violin_panel_gap, hspace=_violin_panel_gap,
            left=0.08, right=0.98, top=0.9, bottom=0.1,
        )
        _draw_panel(axes[0, 0], ['TWS', 'GWS'], 'Lag 0', '', show_ylabel=True)
        _draw_panel(axes[0, 1], ['TWS', 'GWS'], 'Max Lag', '', show_ylabel=False)
        axes[0, 0].set_title('Zero lag', fontsize=FONT_TITLE, fontweight='bold', pad=_title_pad)
        axes[0, 1].set_title('Optimal lag', fontsize=FONT_TITLE, fontweight='bold', pad=_title_pad)
        axes[0, 0].set_ylabel('Spearman ρ (anomaly)', fontsize=FONT_TICK)
        axes[0, 1].set_ylabel('')
        if has_residual_layer:
            _draw_panel(axes[1, 0], ['TWS residual', 'GWS residual'], 'Lag 0', '', show_ylabel=True)
            _draw_panel(axes[1, 1], ['TWS residual', 'GWS residual'], 'Max Lag', '', show_ylabel=False)
            axes[1, 0].set_ylabel('Spearman ρ (residual)', fontsize=FONT_TICK)
            axes[1, 1].set_ylabel('')
            for ax in axes[0, :]:
                ax.set_xticklabels([])
            _panel_letter(axes[0, 0], 'a')
            _panel_letter(axes[0, 1], 'b')
            _panel_letter(axes[1, 0], 'c')
            _panel_letter(axes[1, 1], 'd')
        else:
            axes[1, 0].set_visible(False)
            axes[1, 1].set_visible(False)
            _panel_letter(axes[0, 0], 'a')
            _panel_letter(axes[0, 1], 'b')
    else:
        fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)
        fig.subplots_adjust(wspace=0.14, left=0.1, right=0.96, top=0.88, bottom=0.12)

    if not use_individual and depthclass_correlations is None:
        for idx, lag_type in enumerate(['Lag 0', 'Max Lag']):
            ax = axes[idx]
            subset = df[df['Lag Type'] == lag_type]

            if len(subset) == 0:
                ax.text(0.1, 0.5, 'No Data', ha='center', va='center', fontsize=FONT_LABEL)
                ax.set_title(lag_type, fontsize=FONT_TITLE, fontweight='bold')
                continue

            positions = []
            labels = []
            data_groups = []
            colors_list = []
            hatches = []

            for i, dc in enumerate(active_depth_order):
                dc_subset = subset[subset['Depth Class'] == dc]
                if len(dc_subset) == 0:
                    continue
                tws_data = dc_subset[dc_subset['Variable'] == 'TWS']['Correlation'].values
                gws_data = dc_subset[dc_subset['Variable'] == 'GWS']['Correlation'].values

                if len(tws_data) > 0:
                    positions.append(len(positions))
                    data_groups.append(tws_data)
                    labels.append('TWSA')
                    colors_list.append(_grace_var_colors['TWS'])
                    hatches.append(None)
                if len(gws_data) > 0:
                    positions.append(len(positions))
                    data_groups.append(gws_data)
                    labels.append('GWSA')
                    colors_list.append(_grace_var_colors['GWS'])
                    hatches.append(None)

            if len(data_groups) == 0:
                continue

            parts = ax.violinplot(data_groups, positions=positions, showmeans=False, showmedians=False, showextrema=False)
            for pc, color in zip(parts['bodies'], colors_list):
                pc.set_facecolor(color)
                pc.set_alpha(0.6)
                pc.set_edgecolor('black')
                pc.set_linewidth(1)

            bp = ax.boxplot(data_groups, positions=positions, widths=0.15, patch_artist=True,
                            showfliers=True, flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.6})

            for j, (patch, color, hatch) in enumerate(zip(bp['boxes'], colors_list, hatches)):
                patch.set_facecolor('white')
                patch.set_edgecolor(color)
                patch.set_linewidth(1.5)
                if hatch:
                    patch.set_hatch(hatch)

            for element in ['whiskers', 'caps']:
                for line in bp[element]:
                    line.set_color('gray')
                    line.set_linewidth(1)
            for line in bp['medians']:
                line.set_color('black')
                line.set_linewidth(1.5)

            np.random.seed(42)
            for i, (pos, dg) in enumerate(zip(positions, data_groups)):
                jitter = np.random.uniform(-0.12, 0.12, len(dg))
                ax.scatter(pos + jitter, dg, c=colors_list[i], alpha=0.5, s=18, edgecolors='none', zorder=1)

            for i, (pos, label, dg) in enumerate(zip(positions, labels, data_groups)):
                dc = active_depth_order[i % max(len(active_depth_order), 1)]
                var = 'TWS' if 'TWS' in label else 'GWS'
                key = f'{var}_{lag_type}_{dc}'
                if key in stats_summary:
                    s = stats_summary[key]
                    ann = f"n={s['n']}\nmed={s['median']:.2f}\nmean={s['mean']:.2f}"
                    if lag_type == 'Max Lag':
                        med_lag = s.get('lag_median', np.nan)
                        if pd.notna(med_lag):
                            ann += f"\nmed lag={float(med_lag):.0f}"
                    ax.annotate(ann, xy=(pos, 0.9), ha='center', va='bottom', fontsize=FONT_ANNOT, fontweight='medium')

            ax.set_xticks(positions)
            ax.set_xticklabels(labels, fontsize=FONT_TICK)
            if idx == 0:
                ax.set_ylabel('Spearman ρ', fontsize=FONT_TICK)
            else:
                ax.set_ylabel('')
            ax.set_title(
                'Zero lag' if lag_type == 'Lag 0' else 'Optimal lag',
                fontsize=FONT_TITLE, fontweight='bold', pad=_title_pad,
            )
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
            ax.yaxis.set_major_locator(MultipleLocator(0.2))
            ax.grid(axis='y', alpha=0.3, linestyle=':')
            ax.set_ylim(-1.15, 1.25)
            ax.tick_params(axis='both', labelsize=FONT_TICK)

            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor='white', edgecolor='black', label='TWSA'),
                Patch(facecolor='white', edgecolor='black', hatch='///', label='GWSA'),
            ]
            ax.legend(handles=legend_elements, loc='lower right', fontsize=FONT_LABEL)

        _panel_letter(axes[0], 'a')
        _panel_letter(axes[1], 'b')

    # --- Additional figure: empirical CDF of max-lag correlations (TWS | GWS) ---
    def _max_lag_series(var_name):
        return df[(df['Variable'] == var_name) & (df['Lag Type'] == 'Max Lag')]['Correlation'].dropna().values

    def _plot_ecdf_max_lag_ax(ax, curve_specs):
        """Empirical CDF (%); y-grid every 10% (alpha=0.7); no vertical reference lines."""
        any_line = False
        for spec in curve_specs:
            vals = np.asarray(spec['values'], dtype=float)
            vals = vals[np.isfinite(vals)]
            label = spec['label']
            color = spec['color']
            ls = spec.get('linestyle', '-')
            if len(vals) == 0:
                continue
            xs = np.sort(vals)
            n = len(xs)
            y_pct = np.arange(1, n + 1) / n * 100.0
            x_step = np.concatenate([[xs[0]], xs])
            y_step = np.concatenate([[0.0], y_pct])
            ax.step(x_step, y_step, where='post', color=color, ls=ls, lw=2.4, label=f"{label} (n={n})", zorder=3)
            any_line = True
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(0, 100)
        ax.set_xlabel('Correlation r (max lag)', fontsize=CDF_FS_AXIS)
        ax.set_ylabel('CDF (%)', fontsize=CDF_FS_AXIS)
        ax.set_yticks(np.arange(0, 101, 10))
        ax.yaxis.set_major_locator(MultipleLocator(10))
        ax.xaxis.set_major_locator(MultipleLocator(0.2))
        ax.grid(True, which='major', axis='y', alpha=0.7, linestyle='--', zorder=0)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both', which='major', labelsize=CDF_FS_TICK)
        ax.legend(loc='upper left', fontsize=CDF_FS_LEGEND, framealpha=0.95, edgecolor='black', fancybox=False)
        if not any_line:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', fontsize=CDF_FS_AXIS, color='#666')

    dual_cdf = use_four_var_table
    if dual_cdf:
        tws_cdf_specs = [
            {'label': 'TWSA', 'values': _max_lag_series('TWS'), 'color': '#0d3b66', 'linestyle': '-'},
            {'label': 'TWSA residual', 'values': _max_lag_series('TWS residual'), 'color': '#1d7874', 'linestyle': '--'},
        ]
        gws_cdf_specs = [
            {'label': 'GWSA', 'values': _max_lag_series('GWS'), 'color': GWSA_PLOT_COLOR, 'linestyle': '-'},
            {'label': 'GWSA residual', 'values': _max_lag_series('GWS residual'), 'color': GWSA_PLOT_COLOR, 'linestyle': '--'},
        ]
    else:
        tws_cdf_specs = [{'label': 'TWSA', 'values': _max_lag_series('TWS'), 'color': '#0d3b66', 'linestyle': '-'}]
        gws_cdf_specs = [{'label': 'GWSA', 'values': _max_lag_series('GWS'), 'color': GWSA_PLOT_COLOR, 'linestyle': '-'}]

    fig_cdf = None
    if show_cdf:
        cdf_h = max(4.8, figsize[1] * 0.48)
        fig_cdf, axes_cdf = plt.subplots(1, 2, figsize=(figsize[0], cdf_h))
        fig_cdf.patch.set_facecolor('white')
        fig_cdf.subplots_adjust(wspace=0.26, left=0.08, right=0.98, top=0.96, bottom=0.14)
        _plot_ecdf_max_lag_ax(axes_cdf[0], tws_cdf_specs)
        _plot_ecdf_max_lag_ax(axes_cdf[1], gws_cdf_specs)

    def _summary_cell(var, lag_type, dc):
        key = f'{var}_{lag_type}_{dc}'
        if key not in stats_summary:
            return '—'
        s = stats_summary[key]
        lag_str = ''
        if lag_type == 'Max Lag':
            med_lag = s.get('lag_median', np.nan)
            if pd.notna(med_lag):
                lag_str = f"  med lag={float(med_lag):.0f}"
        return f"{s['median']:.2f} ({s['mean']:.2f})  n={s['n']}{lag_str}"

    def _make_table(ax, title, variables):
        """Legacy depth-row table (non–raw+residual runs)."""
        ax.axis('off')
        table_data = []
        rows_with_data = []
        for dc in active_depth_order:
            row = [dc]
            row_has_data = False
            for var in variables:
                key_lag0 = f'{var}_Lag 0_{dc}'
                if key_lag0 in stats_summary:
                    s = stats_summary[key_lag0]
                    row.append(f"{s['median']:.2f} ({s['mean']:.2f})  n={s['n']}")
                    row_has_data = True
                else:
                    row.append('—')
                key_max = f'{var}_Max Lag_{dc}'
                if key_max in stats_summary:
                    s = stats_summary[key_max]
                    lag_str = ''
                    if pd.notna(s.get('lag_median', np.nan)):
                        lag_str = f"  med lag={float(s['lag_median']):.0f}"
                    row.append(f"{s['median']:.2f} ({s['mean']:.2f})  n={s['n']}{lag_str}")
                    row_has_data = True
                else:
                    row.append('—')
            if row_has_data:
                rows_with_data.append(dc)
                table_data.append(row)
        col_labels = ['Depth']
        for v in variables:
            col_labels.append(f'{v} Lag 0')
            col_labels.append(f'{v} Max')
        tbl = ax.table(cellText=table_data, colLabels=col_labels, loc='center', cellLoc='center')
        tbl.auto_set_font_size(False)
        tbl.set_fontsize(10)
        tbl.scale(1.1, 1.6)
        for i in range(len(col_labels)):
            tbl[(0, i)].set_facecolor('#333')
            tbl[(0, i)].set_text_props(color='white', fontweight='bold', fontsize=11)
        for i, dc in enumerate(rows_with_data):
            tbl[(i + 1, 0)].set_facecolor(depth_colors.get(dc, 'white'))
            tbl[(i + 1, 0)].set_text_props(fontweight='bold', fontsize=10, color='white')
        if title is not None:
            ax.set_title(title, fontsize=12, fontweight='bold', pad=4)
        else:
            ax.set_title('', fontsize=12, fontweight='bold', pad=4)

    def _make_merged_anomaly_residual_table(ax, stats_summary, active_depth_order):
        """Single table: rows Anomaly / Residual (× depth if both Shallow and Deep)."""
        ax.axis('off')
        col_labels = [
            '',
            'TWSA Zero Lag',
            'TWSA Optimal Lag',
            'GWSA Zero Lag',
            'GWSA Optimal Lag',
        ]
        table_data = []
        if len(active_depth_order) == 1:
            dc = active_depth_order[0]
            row_a = [
                'Anomaly',
                _summary_cell('TWS', 'Lag 0', dc),
                _summary_cell('TWS', 'Max Lag', dc),
                _summary_cell('GWS', 'Lag 0', dc),
                _summary_cell('GWS', 'Max Lag', dc),
            ]
            row_r = [
                'Residual',
                _summary_cell('TWS residual', 'Lag 0', dc),
                _summary_cell('TWS residual', 'Max Lag', dc),
                _summary_cell('GWS residual', 'Lag 0', dc),
                _summary_cell('GWS residual', 'Max Lag', dc),
            ]
            if any(c != '—' for c in row_a[1:]):
                table_data.append(row_a)
            if any(c != '—' for c in row_r[1:]):
                table_data.append(row_r)
        else:
            for dc in active_depth_order:
                row_a = [
                    f'Anomaly ({dc})',
                    _summary_cell('TWS', 'Lag 0', dc),
                    _summary_cell('TWS', 'Max Lag', dc),
                    _summary_cell('GWS', 'Lag 0', dc),
                    _summary_cell('GWS', 'Max Lag', dc),
                ]
                if any(c != '—' for c in row_a[1:]):
                    table_data.append(row_a)
            for dc in active_depth_order:
                row_r = [
                    f'Residual ({dc})',
                    _summary_cell('TWS residual', 'Lag 0', dc),
                    _summary_cell('TWS residual', 'Max Lag', dc),
                    _summary_cell('GWS residual', 'Lag 0', dc),
                    _summary_cell('GWS residual', 'Max Lag', dc),
                ]
                if any(c != '—' for c in row_r[1:]):
                    table_data.append(row_r)

        if not table_data:
            ax.text(0.5, 0.5, 'No summary data', ha='center', va='center', fontsize=10)
            return

        nrows = len(table_data)
        tbl = ax.table(
            cellText=table_data,
            colLabels=col_labels,
            loc='center',
            cellLoc='center',
        )
        tbl.auto_set_font_size(False)
        # Match legacy `_make_table` readability (was reduced only to shrink figure height).
        tbl.set_fontsize(10)
        tbl.scale(1.1, 1.55)
        for j in range(len(col_labels)):
            tbl[(0, j)].set_facecolor('#333')
            tbl[(0, j)].set_text_props(color='white', fontweight='bold', fontsize=11)
        for i in range(nrows):
            tbl[(i + 1, 0)].set_facecolor('#e8f4fc')
            tbl[(i + 1, 0)].set_text_props(fontweight='bold', fontsize=10, color='black')
        ax.set_title('Summary — median (mean)', fontsize=12, fontweight='bold', pad=6)

    fig3 = None
    if show_tables:
        n_rows_tbl = 1
        if use_four_var_table:
            n_depth = len(active_depth_order)
            n_rows_tbl = 2 if n_depth == 1 else 2 * n_depth
        fig3_h = min(3.5, max(1.15, 0.42 * n_rows_tbl + 0.9))
        fig3, axes_t = plt.subplots(1, 1, figsize=(10.5, fig3_h))
        fig3.patch.set_facecolor('white')
        ax_tbl = fig3.axes[0]
        if use_four_var_table:
            _make_merged_anomaly_residual_table(ax_tbl, stats_summary, active_depth_order)
            cap = (
                f"Depth class: {', '.join(active_depth_order)}."
                if len(active_depth_order) == 1
                else f"Depth classes: {', '.join(active_depth_order)} (row labels indicate depth)."
            )
            fig3.text(0.5, 0.02, cap, ha='center', va='bottom', fontsize=8.5, color='#333')
            fig3.tight_layout(rect=[0.02, 0.08, 0.98, 0.96])
        else:
            _make_table(ax_tbl, 'Summary - median (mean)', var_list)
            fig3.tight_layout()
    
    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        stem_base = save_path.stem + residual_stem
        saved_parts = ['violin']
        fig.savefig(save_path.with_stem(stem_base), dpi=300, bbox_inches='tight')
        if fig_cdf is not None:
            fig_cdf.savefig(save_path.with_stem(stem_base + '_cdf'), dpi=300, bbox_inches='tight')
            saved_parts.append('CDF')
        if fig3 is not None:
            fig3.savefig(save_path.with_stem(stem_base + '_summary'), dpi=300, bbox_inches='tight')
            saved_parts.append('summary')
        print(f"Saved figures to {save_path.parent} ({', '.join(saved_parts)})")

    # Always show in the notebook / interactive session (even when also saving)
    plt.show()
    plt.close('all')
    
    
    return {'stats_summary': stats_summary, 'stats_df': stats_df, 'filter_counts': filter_counts}


def _filter_wells_by_depth_class(gdf, depth_class=None):
    """Optional depth-class filter for well GeoDataFrames."""
    import pandas as pd

    if gdf is None or len(gdf) == 0:
        return gdf
    if depth_class is None:
        return gdf
    if 'depth_class' not in gdf.columns:
        raise ValueError("depth_class filter requested but column 'depth_class' is missing")
    return gdf[gdf['depth_class'].astype(str) == str(depth_class)].copy()


def _extract_optimal_lags(gdf, lag_col: str):
    """Return finite optimal-lag values (months) from a well GeoDataFrame."""
    import numpy as np
    import pandas as pd

    if gdf is None or len(gdf) == 0 or lag_col not in gdf.columns:
        return np.array([], dtype=float)
    vals = pd.to_numeric(gdf[lag_col], errors='coerce').dropna().astype(float)
    vals = vals[np.isfinite(vals)]
    return vals.values


def _optimal_lag_case_specs():
    """Four optimal-lag cases: state × GRACE variable."""
    return [
        ('anomaly', 'TWSA', 'lag_tws'),
        ('anomaly', 'GWSA', 'lag_gws'),
        ('residual', 'TWSA', 'lag_tws'),
        ('residual', 'GWSA', 'lag_gws'),
    ]


def _filter_common_tws_gws_gdf(gdf, lag_type: str = 'Max Lag'):
    """Keep rows with valid TWSA and GWSA correlations for *lag_type*."""
    import pandas as pd

    if gdf is None or len(gdf) == 0:
        return gdf
    tws_col = 'corr_tws_lag0' if lag_type == 'Lag 0' else 'corr_tws'
    gws_col = 'corr_gws_lag0' if lag_type == 'Lag 0' else 'corr_gws'
    tws_ok = gdf[tws_col].map(_is_finite_corr) if tws_col in gdf.columns else pd.Series(False, index=gdf.index)
    gws_ok = gdf[gws_col].map(_is_finite_corr) if gws_col in gdf.columns else pd.Series(False, index=gdf.index)
    return gdf[tws_ok & gws_ok].copy()


def _collect_optimal_lag_cases(
    wells_gdf_raw,
    wells_gdf_residual,
    depth_class=None,
    require_common_tws_gws=False,
):
    """Build lag arrays for the four optimal-lag cases."""
    import pandas as pd

    raw = _filter_wells_by_depth_class(wells_gdf_raw, depth_class)
    res = _filter_wells_by_depth_class(wells_gdf_residual, depth_class)
    filter_counts = {}
    for state_label, gdf in (('anomaly', raw), ('residual', res)):
        fc = _count_common_tws_gws_wells(gdf if gdf is not None else pd.DataFrame())
        filter_counts[state_label] = fc.get('Max Lag', {})
    if require_common_tws_gws:
        raw = _filter_common_tws_gws_gdf(raw, 'Max Lag')
        res = _filter_common_tws_gws_gdf(res, 'Max Lag')
        for state_label, gdf in (('anomaly', raw), ('residual', res)):
            filter_counts[state_label]['n_after_filter'] = int(len(gdf)) if gdf is not None else 0
    cases = {}
    for state, grace_var, lag_col in _optimal_lag_case_specs():
        gdf = raw if state == 'anomaly' else res
        cases[(state, grace_var)] = _extract_optimal_lags(gdf, lag_col)
    return cases, filter_counts


def _lag_spread_metrics(values, spread: str = 'std'):
    """Compute spread metric(s) for a 1-D lag sample."""
    import numpy as np
    from scipy import stats as scipy_stats

    spread = str(spread).lower()
    allowed = {'std', 'iqr', 'mad', 'all'}
    if spread not in allowed:
        raise ValueError(f"spread must be one of {sorted(allowed)}, got {spread!r}")

    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    n = int(len(vals))
    if n == 0:
        base = {
            'n': 0, 'mean': np.nan, 'median': np.nan, 'min': np.nan, 'max': np.nan,
            'skew': np.nan, 'std': np.nan, 'iqr': np.nan, 'mad': np.nan,
            'spread': np.nan,
        }
        if spread != 'all':
            base['spread'] = np.nan
        return base

    mean = float(np.mean(vals))
    median = float(np.median(vals))
    std = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    mad = float(scipy_stats.median_abs_deviation(vals, scale=1.0, nan_policy='omit'))
    skew = float(scipy_stats.skew(vals, bias=False)) if n > 2 else np.nan

    out = {
        'n': n, 'mean': mean, 'median': median,
        'min': float(np.min(vals)), 'max': float(np.max(vals)),
        'skew': skew, 'std': std, 'iqr': iqr, 'mad': mad,
    }
    if spread == 'all':
        return out
    out['spread'] = {'std': std, 'iqr': iqr, 'mad': mad}[spread]
    return out


def summarize_optimal_lag_spread(
    wells_gdf_raw,
    wells_gdf_residual,
    spread: str = 'std',
    depth_class=None,
    require_common_tws_gws=False,
) -> "pd.DataFrame":
    """
    Spread of selected optimal lags (months) for the four cases:
    TWSA/GWSA × anomaly/residual.

    Parameters
    ----------
    spread : {'std', 'iqr', 'mad', 'all'}, default 'std'
        ``std`` — sample standard deviation (ddof=1).
        ``iqr`` — inter-quartile range (Q3 − Q1); robust for skewed lags.
        ``mad`` — median absolute deviation from the median.
        ``all`` — return ``std``, ``iqr``, and ``mad`` columns together.
    require_common_tws_gws : bool, default False
        When True, use only wells with valid TWSA and GWSA optimal-lag correlations
        (same well set for TWSA and GWSA within each state).

    Returns
    -------
    pandas.DataFrame
        One row per case with ``n``, ``mean``, ``median``, ``skew``, and the
        requested spread metric(s).
    """
    import pandas as pd

    cases, _ = _collect_optimal_lag_cases(
        wells_gdf_raw, wells_gdf_residual,
        depth_class=depth_class,
        require_common_tws_gws=require_common_tws_gws,
    )
    records = []
    for (state, grace_var), vals in cases.items():
        metrics = _lag_spread_metrics(vals, spread=spread)
        row = {
            'depth_class': depth_class if depth_class is not None else 'all',
            'state': state,
            'grace_variable': grace_var,
            'n': metrics['n'],
            'mean': metrics['mean'],
            'median': metrics['median'],
            'min': metrics['min'],
            'max': metrics['max'],
            'skew': metrics['skew'],
        }
        if spread == 'all':
            row.update({'std': metrics['std'], 'iqr': metrics['iqr'], 'mad': metrics['mad']})
        else:
            row['spread_metric'] = spread
            row['spread'] = metrics['spread']
        records.append(row)

    df = pd.DataFrame(records)
    if df.empty:
        cols = ['depth_class', 'state', 'grace_variable', 'n', 'mean', 'median', 'min', 'max', 'skew']
        if spread == 'all':
            cols += ['std', 'iqr', 'mad']
        else:
            cols += ['spread_metric', 'spread']
        return pd.DataFrame(columns=cols)

    state_order = pd.Categorical(df['state'], categories=['anomaly', 'residual'], ordered=True)
    var_order = pd.Categorical(df['grace_variable'], categories=['TWSA', 'GWSA'], ordered=True)
    return (
        df.assign(state=state_order, grace_variable=var_order)
        .sort_values(['state', 'grace_variable'])
        .reset_index(drop=True)
    )


def plot_optimal_lag_histograms(
    wells_gdf_raw,
    wells_gdf_residual,
    depth_class=None,
    bins=None,
    max_lag_months=36,
    figsize=(12, 9),
    save_path=None,
    show=True,
    show_kde=True,
    show_normal_fit=True,
    show_stats=True,
    dpi=300,
    require_common_tws_gws=False,
):
    """
    Violin + box plots of optimal lags (months) in a 2×1 layout (anomaly / residual),
    with TWSA and GWSA side-by-side in each row — styled like
    :func:`plot_correlation_distributions`.

    Parameters
    ----------
    require_common_tws_gws : bool, default False
        When True, use only wells with valid TWSA and GWSA optimal-lag correlations
        (same well set for TWSA and GWSA panels within each state).
    show_stats : bool, default True
        Annotate each violin with ``n`` and median lag (months).
    show_kde, show_normal_fit, bins
        Ignored (kept for backward compatibility with earlier histogram API).

    Returns
    -------
    dict
        ``fig``, ``spread_df``, ``cases``, ``filter_counts``.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    _grace_var_colors = {'TWSA': TWSA_PLOT_COLOR, 'GWSA': GWSA_PLOT_COLOR}
    FONT_TICK = 12

    if require_common_tws_gws:
        print("plot_optimal_lag_histograms: require_common_tws_gws=True (same wells for TWSA & GWSA).")
    cases, filter_counts = _collect_optimal_lag_cases(
        wells_gdf_raw, wells_gdf_residual,
        depth_class=depth_class,
        require_common_tws_gws=require_common_tws_gws,
    )
    if require_common_tws_gws:
        for state_label, fc in filter_counts.items():
            print(
                f"  {state_label} optimal lag: TWS={fc.get('n_tws', 0)}, GWS={fc.get('n_gws', 0)}, "
                f"common={fc.get('n_common', 0)} (after filter n={fc.get('n_after_filter', fc.get('n_common', 0))})"
            )
    spread_df = summarize_optimal_lag_spread(
        wells_gdf_raw, wells_gdf_residual, spread='all', depth_class=depth_class,
        require_common_tws_gws=require_common_tws_gws,
    )

    row_specs = [
        ('anomaly', 'Optimal lag (anomaly)', 'a'),
        ('residual', 'Optimal lag (residual)', 'b'),
    ]

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharey=True)
    fig.subplots_adjust(wspace=0.12, hspace=0.18, left=0.1, right=0.98, top=0.97, bottom=0.08)
    axes = np.atleast_1d(axes)

    def _panel_letter(ax, letter, fontsize=11):
        ax.text(
            0.02, 0.98, letter,
            transform=ax.transAxes, ha='left', va='top',
            fontsize=fontsize, fontweight='bold', zorder=10,
            bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9),
        )

    def _y_lag_axis_buffer(ax, max_lag):
        """Headroom above max lag for stat boxes; y ticks stay on 0…max_lag."""
        top = float(max_lag)
        pad = max(3.0, top * 0.15)
        ax.set_ylim(-0.5, top + pad)
        ax.set_yticks(np.arange(0, int(max_lag) + 1, 6))

    for idx, (ax, (state, ylabel, letter)) in enumerate(zip(axes, row_specs)):
        ax.set_facecolor('#fafafa')
        positions, data_groups, labels, colors_list = [], [], [], []
        for grace_var in ('TWSA', 'GWSA'):
            vals = np.asarray(cases.get((state, grace_var), []), dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) == 0:
                continue
            positions.append(len(positions))
            data_groups.append(vals)
            labels.append(grace_var)
            colors_list.append(_grace_var_colors[grace_var])

        if len(data_groups) == 0:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_ylabel(ylabel, fontsize=FONT_TICK, fontweight='medium')
            _y_lag_axis_buffer(ax, max_lag_months)
            _panel_letter(ax, letter)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            continue

        parts = ax.violinplot(
            data_groups, positions=positions, showmeans=False, showmedians=False, showextrema=False,
        )
        for pc, color in zip(parts['bodies'], colors_list):
            pc.set_facecolor(color)
            pc.set_alpha(0.65)
            pc.set_edgecolor(color)
            pc.set_linewidth(1.5)
        bp = ax.boxplot(
            data_groups, positions=positions, widths=0.13, patch_artist=True,
            showfliers=True,
            flierprops=dict(marker='o', markersize=4, alpha=0.5, markeredgecolor='none'),
        )
        for patch, color in zip(bp['boxes'], colors_list):
            patch.set_facecolor('white')
            patch.set_edgecolor(color)
            patch.set_linewidth(2)
        for line in bp['medians']:
            line.set_color('#2c3e50')
            line.set_linewidth(2)

        np.random.seed(42)
        for pos, dg, color in zip(positions, data_groups, colors_list):
            jitter = np.random.uniform(-0.08, 0.08, len(dg))
            ax.scatter(pos + jitter, dg, c=color, alpha=0.5, s=16, edgecolors='none', zorder=2)

        _y_lag_axis_buffer(ax, max_lag_months)
        ymin, ymax = ax.get_ylim()

        if show_stats:
            for pos, label, dg in zip(positions, labels, data_groups):
                grace_var = label
                spread_row = spread_df[
                    (spread_df['state'] == state) & (spread_df['grace_variable'] == grace_var)
                ]
                med_lag = float(np.median(dg))
                n = int(len(dg))
                if not spread_row.empty:
                    med_lag = float(spread_row.iloc[0]['median'])
                    n = int(spread_row.iloc[0]['n'])
                ann = f"n={n}\nmed lag={med_lag:.0f}"
                ax.text(
                    pos, ymax - 0.02 * (ymax - ymin), ann,
                    ha='center', va='top', fontsize=10, fontweight='medium',
                    bbox=dict(boxstyle='round,pad=0.35', facecolor='white', edgecolor='gray', alpha=0.9),
                )

        ax.set_xticks(positions)
        if idx == len(row_specs) - 1:
            ax.set_xticklabels(labels, fontsize=FONT_TICK, fontweight='medium')
            ax.tick_params(axis='x', labelsize=FONT_TICK)
        else:
            ax.set_xticklabels([])
            ax.tick_params(axis='x', labelbottom=False)
        ax.set_ylabel(ylabel, fontsize=FONT_TICK, fontweight='medium')
        ax.tick_params(axis='y', labelsize=FONT_TICK)
        ax.grid(axis='y', alpha=0.4, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        _panel_letter(ax, letter)

    if save_path:
        from pathlib import Path
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved optimal-lag distributions to {save_path}")
    elif show:
        plt.show()
    else:
        plt.close(fig)

    return {'fig': fig, 'spread_df': spread_df, 'cases': cases, 'filter_counts': filter_counts}


def plot_grace_gwl_correlation_lag_maps(
    wells_gdf,
    variable='TWS',
    depth_class=None,
    p_sig=0.05,
    lag_max_months=36,
    vmin_corr=-1.0,
    vmax_corr=1.0,
    cmap_corr='RdBu',
    cmap_lag='viridis',
    figsize=(6, 9),
    aoi_geometry=None,
    save_path=None,
    title_prefix=None,
    pixel_aggregate='max_abs',
    n_cols=1,
    grid_spacing_deg=1.0,
    grid_target_lines=6,
    title_fontsize=9,
    label_fontsize=8,
    tick_fontsize=7,
    plot_wells=False,
    well_legend_label='Wells',
    well_marker_size=26,
):
    """
    Plot maps of Spearman correlation (ρ) and lag between GRACE and GWL.
    
    Expects per-well columns from correlate_wells_with_grace (default method spearman).

    Layout is set by ``n_cols``:
      - ``n_cols=1`` (default): two stacked rows — (a) ρ, (b) lag.
      - ``n_cols=2``: one row, two columns — ρ | lag (same as other map helpers).

    Lon/lat grid lines use Cartopy ``gridlines`` with spacing ``grid_spacing_deg``
    (default 1°). Pass ``grid_spacing_deg=None`` for automatic spacing from map
    extent (several lines on small regions / single-country views).

    Uses GeoDataFrames returned by correlate_wells_with_grace (raw or residual).
    For pixels with **multiple wells**, ``pixel_aggregate`` controls how values
    are combined: keep the well with largest |ρ| (``max_abs``), or take the
    **mean** or **median** of ρ, p-value, and lag across wells in that pixel
    (avoids biasing toward the single largest correlation).

    Stippling marks locations where the (aggregated) p-value is **> p_sig**
    (non-significant).

    Colored tiles use the **same GRACE lon/lat nodes** as ``grace_lat`` / ``grace_lon``
    from ``correlate_wells_with_grace`` (native grid), so they align with
    ``plot_all_well_locations`` red crosses when both use the same correlation table.
    
    Parameters
    ----------
    wells_gdf : geopandas.GeoDataFrame
        Output from correlate_wells_with_grace (either 'raw' or 'residual'), with columns:
          - 'grace_lat', 'grace_lon'
          - 'Latitude', 'Longitude' or point ``geometry`` (used when ``plot_wells=True``)
          - 'depth_class' (optional, for filtering)
          - 'corr_tws', 'pvalue_tws', 'lag_tws'   (for variable='TWS')
          - 'corr_gws', 'pvalue_gws', 'lag_gws'   (for variable='GWS')
    variable : {'TWS', 'GWS'}, default 'TWS'
        Which GRACE-vs-GWL variable to map. For residual analyses, pass the
        residual GeoDataFrame; column names are the same.
    depth_class : {'Shallow', 'Deep', None}, default None
        If provided, restrict to a single depth class before aggregating per pixel.
    p_sig : float, default 0.05
        Significance threshold. Stippling is drawn where p-value > p_sig.
    lag_max_months : int, default 36
        Maximum lag in months used when computing correlations (upper bound for color scale).
    vmin_corr, vmax_corr : float, default (-1.0, 1.0)
        Limits for correlation colorbar (symmetric diverging scale).
    cmap_corr : str, default 'RdBu_r'
        Colormap for correlation (diverging, red/blue).
    cmap_lag : str, default 'viridis'
        Colormap for lag (sequential).
    figsize : tuple, default (6, 9)
        Figure size in inches (width, height). Wider ``figsize`` works well with
        ``n_cols=2`` (e.g. ``(12, 5)``).
    n_cols : int, default 1
        ``1`` = vertical stack (2×1 subplots). ``2`` = side-by-side (1×2).
    grid_spacing_deg : float or None, default 1.0
        Major grid line spacing in degrees (lon and lat). ``None`` selects a step
        from map extent (see ``grid_target_lines``), then caps it by the native
        GRACE pixel spacing so dashed grid lines match ``pcolormesh`` cell edges.
        If you set this **larger** than the data resolution (e.g. ``2`` for 1° pixels),
        grid lines and colored cells will look misaligned — use ``None`` or match
        the product spacing (often ``1.0``).
    grid_target_lines : int, default 6
        When ``grid_spacing_deg`` is ``None``, approximate target number of
        intervals along the longer map side for picking a nice degree step.
    aoi_geometry : geopandas.GeoDataFrame, GeoSeries, or geometry, optional
        Area-of-interest boundary to overlay as an outline (e.g. arid-region mask).
    save_path : str or Path, optional
        If provided, save the figure to this path. Otherwise, display with plt.show().
    title_prefix : str, optional
        Ignored (kept for call-site compatibility). Subplot titles are not drawn;
        use colorbar labels for semantics.
    pixel_aggregate : {'max_abs', 'mean', 'median'}, default 'max_abs'
        How to combine multiple wells that share the same GRACE pixel.
        ``max_abs`` uses the well with largest absolute ρ and its lag/p-value;
        ``mean`` / ``median`` take the mean or median of ρ, p-values, and lags
        across wells in the pixel.
    title_fontsize, label_fontsize, tick_fontsize : float
        ``title_fontsize``: panel letters (a/b). ``label_fontsize``: colorbar labels;
        ``tick_fontsize``: axis and colorbar ticks.
    plot_wells : bool, default False
        If True, overlay individual well locations (``Longitude``/``Latitude`` or point
        ``geometry``) on both maps and add a legend on panel (b), lower right.
    well_legend_label : str, default 'Wells'
        Legend entry when ``plot_wells`` is True.
    well_marker_size : float, default 26
        Marker size (points²) for well scatters.
    
    Returns
    -------
    dict
        Dictionary with keys:
          - 'fig': the matplotlib Figure object
          - 'data': pandas.DataFrame used for plotting (per-pixel aggregates)
    """
    import matplotlib.pyplot as plt
    ccrs, cfeature = _require_cartopy(
        "Cartopy is required for `plot_grace_gwl_correlation_lag_maps`, but it is not installed. "
        "Install it (e.g., `conda install -c conda-forge cartopy`) or skip this plot."
    )
    import numpy as np
    import pandas as pd
    import matplotlib.ticker as mticker
    from matplotlib.collections import PolyCollection
    from pathlib import Path
    
    if wells_gdf is None or len(wells_gdf) == 0:
        print("wells_gdf is empty; nothing to plot.")
        return {'fig': None, 'data': pd.DataFrame()}
    
    df = wells_gdf.copy()
    
    # Require GRACE pixel coordinates
    required_xy = {'grace_lat', 'grace_lon'}
    missing_xy = [c for c in required_xy if c not in df.columns]
    if missing_xy:
        raise ValueError(f"Missing required columns in wells_gdf: {missing_xy}")
    
    base = variable.strip().lower()
    if base not in ('tws', 'gws'):
        raise ValueError("variable must be 'TWS' or 'GWS'.")
    
    r_col = f'corr_{base}'
    p_col = f'pvalue_{base}'
    lag_col = f'lag_{base}'
    for col in [r_col, p_col, lag_col]:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in wells_gdf.")
    
    if depth_class is not None and 'depth_class' in df.columns:
        df = df[df['depth_class'] == depth_class]
    
    # Ensure p-values are numeric; handle string forms like "<0.01"
    if not np.issubdtype(df[p_col].dtype, np.number):
        df[p_col] = (
            df[p_col]
            .astype(str)
            .str.replace('<', '', regex=False)
            .str.strip()
        )
        df[p_col] = pd.to_numeric(df[p_col], errors='coerce')
    
    # Keep only rows with valid correlation and lag
    df = df[df[r_col].notna() & df[lag_col].notna()].copy()
    if len(df) == 0:
        print("No valid correlation/lag data to plot for the given filters.")
        return {'fig': None, 'data': df}
    
    pa = str(pixel_aggregate).strip().lower()
    allowed_pa = {'max_abs', 'mean', 'median'}
    if pa not in allowed_pa:
        raise ValueError(f"pixel_aggregate must be one of {sorted(allowed_pa)}, got {pixel_aggregate!r}")
    
    # Aggregate per GRACE pixel (explicit loop: avoids pandas apply + include_groups issues,
    # and future deprecations around grouping columns inside apply.)
    rows = []
    for (glat, glon), group in df.groupby(['grace_lat', 'grace_lon']):
        glat = float(glat)
        glon = float(glon)
        if pa == 'max_abs':
            idx = group[r_col].abs().idxmax()
            row = group.loc[idx]
            rows.append(
                {
                    'grace_lat': glat,
                    'grace_lon': glon,
                    r_col: row[r_col],
                    p_col: row[p_col],
                    lag_col: row[lag_col],
                }
            )
        elif pa == 'mean':
            rows.append(
                {
                    'grace_lat': glat,
                    'grace_lon': glon,
                    r_col: group[r_col].mean(),
                    p_col: group[p_col].mean(),
                    lag_col: group[lag_col].mean(),
                }
            )
        else:
            rows.append(
                {
                    'grace_lat': glat,
                    'grace_lon': glon,
                    r_col: group[r_col].median(),
                    p_col: group[p_col].median(),
                    lag_col: group[lag_col].median(),
                }
            )
    pixel_df = pd.DataFrame(rows)
    
    if len(pixel_df) == 0:
        print("No pixels with valid aggregated correlation data.")
        return {'fig': None, 'data': pixel_df}
    
    lat_vals = pixel_df['grace_lat'].values
    lon_vals = pixel_df['grace_lon'].values
    lat_min, lat_max = float(lat_vals.min()), float(lat_vals.max())
    lon_min, lon_max = float(lon_vals.min()), float(lon_vals.max())
    
    # Native GRACE grid nodes from the correlation table (same as correlate_wells_with_grace /
    # plot_all_well_locations). Do **not** rebuild coordinates with np.arange(floor..ceil),
    # or pcolormesh cells sit on a synthetic grid and shift ~½–1 cell vs true pixel centers.
    u_lats = np.sort(np.unique(lat_vals))
    u_lons = np.sort(np.unique(lon_vals))
    lat_centers = u_lats.astype(float)
    lon_centers = u_lons.astype(float)
    lat_res = float(np.median(np.diff(u_lats))) if len(u_lats) > 1 else 1.0
    lon_res = float(np.median(np.diff(u_lons))) if len(u_lons) > 1 else 1.0
    lat_res = float(np.clip(lat_res, 0.25, 2.0))
    lon_res = float(np.clip(lon_res, 0.25, 2.0))
    if len(lat_centers) == 0 or len(lon_centers) == 0:
        print("Insufficient range to build grid.")
        return {'fig': None, 'data': pixel_df}
    
    # Use the minimum positive spacing as native pixel size. Median spacing can be inflated
    # for sparse subsets and creates overly large corner cells.
    lat_diffs = np.diff(lat_centers)
    lon_diffs = np.diff(lon_centers)
    lat_pos = lat_diffs[lat_diffs > 0]
    lon_pos = lon_diffs[lon_diffs > 0]
    if lat_pos.size:
        lat_res = float(np.min(lat_pos))
    if lon_pos.size:
        lon_res = float(np.min(lon_pos))
    lat_res = float(np.clip(lat_res, 0.25, 2.0))
    lon_res = float(np.clip(lon_res, 0.25, 2.0))

    pixel_lats = pd.to_numeric(pixel_df['grace_lat'], errors='coerce').values.astype(float)
    pixel_lons = pd.to_numeric(pixel_df['grace_lon'], errors='coerce').values.astype(float)
    corr_vals = pd.to_numeric(pixel_df[r_col], errors='coerce').values.astype(float)
    lag_vals = pd.to_numeric(pixel_df[lag_col], errors='coerce').values.astype(float)
    p_vals = pd.to_numeric(pixel_df[p_col], errors='coerce').values.astype(float)

    valid_cells = np.isfinite(pixel_lats) & np.isfinite(pixel_lons)
    if not valid_cells.any():
        print("No valid GRACE pixel centers to plot.")
        return {'fig': None, 'data': pixel_df}

    pixel_lats = pixel_lats[valid_cells]
    pixel_lons = pixel_lons[valid_cells]
    corr_vals = corr_vals[valid_cells]
    lag_vals = lag_vals[valid_cells]
    p_vals = p_vals[valid_cells]

    # Draw only observed GRACE cells (no implicit cartesian grid expansion).
    verts = [
        [
            (lon - lon_res / 2.0, lat - lat_res / 2.0),
            (lon + lon_res / 2.0, lat - lat_res / 2.0),
            (lon + lon_res / 2.0, lat + lat_res / 2.0),
            (lon - lon_res / 2.0, lat + lat_res / 2.0),
        ]
        for lat, lon in zip(pixel_lats, pixel_lons)
    ]
    
    wlons = np.array([])
    wlats = np.array([])
    if plot_wells:
        _geom = getattr(df, 'geometry', None)
        if _geom is not None and hasattr(_geom, 'isna') and not _geom.isna().all():
            try:
                wlons = np.asarray(_geom.x, dtype=float)
                wlats = np.asarray(_geom.y, dtype=float)
            except (ValueError, AttributeError):
                wlons = np.array([])
                wlats = np.array([])
        if wlons.size == 0 and 'Longitude' in df.columns and 'Latitude' in df.columns:
            wlons = pd.to_numeric(df['Longitude'], errors='coerce').values.astype(float)
            wlats = pd.to_numeric(df['Latitude'], errors='coerce').values.astype(float)
        if wlons.size:
            _ok = np.isfinite(wlons) & np.isfinite(wlats)
            wlons, wlats = wlons[_ok], wlats[_ok]
    
    ncol_layout = int(n_cols)
    if ncol_layout not in (1, 2):
        raise ValueError("n_cols must be 1 (stacked maps) or 2 (ρ | lag side by side)")
    
    if ncol_layout == 1:
        fig, axes = plt.subplots(
            2,
            1,
            figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()},
            sharex=True,
        )
        ax_corr, ax_lag = axes[0], axes[1]
    else:
        fig, axes = plt.subplots(
            1,
            2,
            figsize=figsize,
            subplot_kw={'projection': ccrs.PlateCarree()},
            sharey=True,
        )
        ax_corr, ax_lag = axes[0], axes[1]
    fig.patch.set_facecolor('white')
    
    # Build AOI object early so extent and outline use the same geometry.
    aoi_gdf = None
    if aoi_geometry is not None:
        import geopandas as gpd
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs)
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")

    # Common extent for both panels.
    # Use filtered wells bounds first (the map subset the user selected), then pixels.
    # AOI is drawn as an outline overlay only; it should not force full-AOI map extent.
    if wlons.size:
        lon_min_raw, lon_max_raw = float(np.nanmin(wlons)), float(np.nanmax(wlons))
        lat_min_raw, lat_max_raw = float(np.nanmin(wlats)), float(np.nanmax(wlats))
    else:
        lon_min_raw = float(np.nanmin(pixel_lons) - lon_res / 2.0)
        lon_max_raw = float(np.nanmax(pixel_lons) + lon_res / 2.0)
        lat_min_raw = float(np.nanmin(pixel_lats) - lat_res / 2.0)
        lat_max_raw = float(np.nanmax(pixel_lats) + lat_res / 2.0)

    lat_padding = max((lat_max_raw - lat_min_raw) * 0.05, 1.0)
    lon_padding = max((lon_max_raw - lon_min_raw) * 0.02, 1.0)
    extent = [
        lon_min_raw - lon_padding,
        lon_max_raw + lon_padding,
        lat_min_raw - lat_padding,
        lat_max_raw + lat_padding,
    ]
    ax_corr.set_extent(extent, crs=ccrs.PlateCarree())
    ax_lag.set_extent(extent, crs=ccrs.PlateCarree())

    glon0, glon1 = float(extent[0]), float(extent[1])
    glat0, glat1 = float(extent[2]), float(extent[3])
    if grid_spacing_deg is None:
        grid_step = _nice_lonlat_grid_step_deg(
            glon0, glon1, glat0, glat1, target_lines=grid_target_lines
        )
        # Auto mode can pick e.g. 2° while GRACE cells are 1° — then dashed lines fall on
        # whole degrees but pcolormesh edges sit on half-degrees; snap grid to data res.
        _dr = min(float(lat_res), float(lon_res))
        if np.isfinite(_dr) and _dr > 0:
            grid_step = min(grid_step, _dr)
    else:
        grid_step = float(grid_spacing_deg)

    def _decorate_map_grid(ax, *, left_labels, bottom_labels):
        gl = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=True,
            linewidth=0.5,
            color='gray',
            alpha=0.75,
            linestyle='--',
            zorder=4,
        )
        gl.xlocator = mticker.MultipleLocator(grid_step)
        gl.ylocator = mticker.MultipleLocator(grid_step)
        gl.top_labels = False
        gl.right_labels = False
        gl.left_labels = left_labels
        gl.bottom_labels = bottom_labels
        gl.xlabel_style = {'size': tick_fontsize, 'color': 'black'}
        gl.ylabel_style = {'size': tick_fontsize, 'color': 'black', 'rotation': 90}

    if ncol_layout == 1:
        _decorate_map_grid(ax_corr, left_labels=True, bottom_labels=False)
        _decorate_map_grid(ax_lag, left_labels=True, bottom_labels=True)
    else:
        _decorate_map_grid(ax_corr, left_labels=True, bottom_labels=True)
        _decorate_map_grid(ax_lag, left_labels=False, bottom_labels=True)
    
    # Add base map features
    for ax in (ax_corr, ax_lag):
        ax.add_feature(cfeature.LAND, facecolor='#f0f0f0', alpha=0.9)
        ax.add_feature(cfeature.OCEAN, facecolor='#d8e6f3', alpha=1.0)
        ax.add_feature(cfeature.COASTLINE, linewidth=0.6)
        ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    
    # Add AOI boundary if provided
    if aoi_gdf is not None:
        for ax in (ax_corr, ax_lag):
            ax.add_geometries(
                aoi_gdf.geometry,
                crs=ccrs.PlateCarree(),
                facecolor='none',
                edgecolor='black',
                linewidth=0.5,
                linestyle='-',
                zorder=3,
            )
    
    # Panel (a): correlation at best lag
    corr_mask = np.isfinite(corr_vals)
    corr_verts = [verts[i] for i in np.where(corr_mask)[0]]
    pcm_corr = PolyCollection(
        corr_verts,
        array=np.asarray(corr_vals[corr_mask], dtype=float),
        cmap=cmap_corr,
        clim=(vmin_corr, vmax_corr),
        edgecolors='none',
        linewidths=0.0,
        zorder=2,
    )
    pcm_corr.set_transform(ccrs.PlateCarree())
    ax_corr.add_collection(pcm_corr)
    cbar_corr = fig.colorbar(
        pcm_corr, ax=ax_corr, orientation='horizontal', fraction=0.046, pad=0.05
    )
    cbar_corr.set_ticks(np.linspace(-1.0, 1.0, 5))
    cbar_corr.ax.xaxis.set_major_formatter(plt.FormatStrFormatter('%.1f'))
    cbar_corr.set_label(r'Spearman $\rho$', fontsize=label_fontsize)
    cbar_corr.ax.tick_params(labelsize=tick_fontsize)
    
    # Stippling where p-value > p_sig (non-significant) — draw on top so it's visible
    mask_nsig = np.isfinite(p_vals) & (p_vals > p_sig)
    if mask_nsig.any():
        ax_corr.scatter(
            pixel_lons[mask_nsig],
            pixel_lats[mask_nsig],
            s=18,
            c='black',
            marker='.',
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
    
    ax_corr.set_ylabel('Lat. (°)', fontsize=label_fontsize)
    ax_corr.tick_params(axis='both', labelsize=tick_fontsize)
    # Panel label (a) — upper-left inside map axes (transAxes)
    ax_corr.text(
        0.02, 0.98, 'a',
        transform=ax_corr.transAxes, ha='left', va='top',
        fontsize=title_fontsize + 1, fontweight='bold', zorder=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    )
    
    # Panel (b): lag at best correlation
    lag_mask = np.isfinite(lag_vals)
    lag_verts = [verts[i] for i in np.where(lag_mask)[0]]
    pcm_lag = PolyCollection(
        lag_verts,
        array=np.asarray(lag_vals[lag_mask], dtype=float),
        cmap=cmap_lag,
        clim=(0, lag_max_months),
        edgecolors='none',
        linewidths=0.0,
        zorder=2,
    )
    pcm_lag.set_transform(ccrs.PlateCarree())
    ax_lag.add_collection(pcm_lag)
    cbar_lag = fig.colorbar(
        pcm_lag, ax=ax_lag, orientation='horizontal', fraction=0.046, pad=0.05
    )
    cbar_lag.set_label('Lag (months)', fontsize=label_fontsize)
    cbar_lag.ax.tick_params(labelsize=tick_fontsize)
    
    if mask_nsig.any():
        ax_lag.scatter(
            pixel_lons[mask_nsig],
            pixel_lats[mask_nsig],
            s=18,
            c='black',
            marker='.',
            alpha=0.85,
            transform=ccrs.PlateCarree(),
            zorder=5,
        )
    
    if wlons.size:
        _well_kw = dict(
            transform=ccrs.PlateCarree(),
            s=well_marker_size,
            facecolors='none',
            edgecolors='black',
            linewidths=1.1,
            zorder=6,
        )
        ax_corr.scatter(wlons, wlats, label='_nolegend_', **_well_kw)
        ax_lag.scatter(wlons, wlats, label=well_legend_label, **_well_kw)
        ax_lag.legend(
            loc='lower right',
            fontsize=max(float(tick_fontsize), 7.0),
            framealpha=0.92,
            edgecolor='0.35',
        )
    
    if ncol_layout == 1:
        ax_lag.set_xlabel('Lon. (°)', fontsize=label_fontsize)
        ax_lag.set_ylabel('Lat. (°)', fontsize=label_fontsize)
    else:
        ax_corr.set_xlabel('Lon. (°)', fontsize=label_fontsize)
        ax_lag.set_xlabel('Lon. (°)', fontsize=label_fontsize)
    ax_lag.tick_params(axis='both', labelsize=tick_fontsize)
    # Panel label (b) — upper-left inside map axes (transAxes)
    ax_lag.text(
        0.02, 0.98, 'b',
        transform=ax_lag.transAxes, ha='left', va='top',
        fontsize=title_fontsize + 1, fontweight='bold', zorder=10,
        bbox=dict(boxstyle='round', facecolor='white', edgecolor='black', alpha=0.9)
    )
    
    plt.tight_layout()
    if ncol_layout == 2:
        fig.subplots_adjust(wspace=0.04)
    else:
        fig.subplots_adjust(hspace=0.22)
    
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(p, dpi=300, bbox_inches='tight')
        print(f"Saved correlation/lag map figure to {p}")
    else:
        plt.show()
    
    #return {'fig': fig, 'data': pixel_df}


# ---------------------------------------------------------------------------
# Variance decomposition (MAD-based, Kim et al. 2009)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Scatter plots: GWL vs GRACE per well
# ---------------------------------------------------------------------------

