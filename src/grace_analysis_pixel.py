"""
Pixel-based GRACE response to extreme precipitation (EPE) analysis.

Notebook-facing surface (see ``__all__``):
  ``analyze_grace_response_by_pixel``, ``plot_pixel_analysis_maps``,
  ``plot_pixel_epe_grace_relationship``, ``plot_pixel_results_distribution_diagnostics``,
  ``plot_event_cluster_distribution_and_relationship``,
  ``join_lake_points_to_recharge_catalog``, ``export_table_s3_lake_recharge``,
  ``summarize_arid_response_coverage``, ``plot_epe_grace_agg_uncertainty_collage``.

Residual note:
  When ``decompose_grace=True``, this module currently uses index-based
  ``decompose_grace_sin_cosin`` (gap years already dropped from the series).
  The TWSA-CPA correlation path in ``grace_analysis_utils`` uses calendar-locked
  ``_decompose_grace_calendar``. Do not mix residual products without checking
  that difference. Aligning both to calendar-locked residuals is a follow-up.

Resources:
  Pixel loops use joblib (``n_jobs=-1`` = all CPUs). Prefer clamping with
  ``download_data.get_resource_config()`` from the notebook so Dask and joblib
  do not oversubscribe. GPU is unused here.
"""

from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd
import xarray as xr
import geopandas as gpd
import rioxarray
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import matplotlib.ticker as mticker
from scipy.stats import linregress, pearsonr
from scipy.optimize import curve_fit
from scipy.stats import chi2, f as f_dist
from tqdm import tqdm
from shapely.geometry import Point, mapping
from rasterio.features import rasterize
from joblib import Parallel, delayed

from grace_analysis_utils import format_pvalue

__all__ = [
    "analyze_grace_response_by_pixel",
    "plot_event_cluster_distribution_and_relationship",
    "plot_pixel_analysis_maps",
    "plot_pixel_results_distribution_diagnostics",
    "plot_pixel_epe_grace_relationship",
    "plot_epe_grace_agg_uncertainty_collage",
    "join_lake_points_to_recharge_catalog",
    "export_table_s3_lake_recharge",
    "build_pixel_analysis_catalog",
    "summarize_arid_response_coverage",
    "format_pvalue",
]

# Match grace_analysis_utils map layout (plot_aridity_raster / plot_multiple_maps_with_balanced_colorbar)
_MAP_SUBPLOT_WSPACE = 0.0
_MAP_SUBPLOT_HSPACE = 0.02
_MAP_CBAR_PAD = 0.01
_MAP_CBAR_FRACTION = 0.05
_MAP_CBAR_SHRINK_V = 0.95
_MAP_CBAR_SHRINK_H = 0.95
_MAP_TIGHT_LAYOUT_PAD = 0.2
# Cartopy gridline labels: lon/lat must use the same size or only one axis gets an explicit fontsize
_MAP_GRID_LABEL_FONTSIZE = 9

from status_io import (  # noqa: E402
    announce as _announce,
    detect_repo_root as _repo_root,
    item as _item,
    note as _note,
    raise_ctx as _raise_ctx,
    rel as _rel,
    summarize_skipped as _summarize_skipped,
)


def fit_model(x, y, model="linear"):
    """
    Fit a parametric relationship between x and y.

    Parameters
    ----------
    x, y : array-like
        1D samples.
    model : {'linear', 'power'}
        - linear: y = a*x + b (OLS via linregress)
        - power : y = a*x^b (fit in log-space with curve_fit)

    Returns
    -------
    dict
        Standardized result:
        - model: str
        - params: dict
        - r: float
        - p: float
        - stderr: dict
        - predict: callable(x_new) -> y_pred
        - n: int (number of points used in fit)
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 2:
        raise ValueError("Need at least 2 finite points to fit a model.")

    mdl = str(model).strip().lower()
    if mdl not in {"linear", "power"}:
        raise ValueError("model must be 'linear' or 'power'")

    if mdl == "linear":
        slope, intercept, r, p, stderr = linregress(x, y)

        def _predict(x_new):
            x_new = np.asarray(x_new, dtype=float)
            return slope * x_new + intercept

        return {
            "model": "linear",
            "params": {"a": float(slope), "b": float(intercept)},
            "r": float(r),
            "p": float(p),
            "stderr": {"a": float(stderr), "b": np.nan},
            "predict": _predict,
            "n": int(x.size),
        }

    # power-law: y = a * x^b  -> log(y) = c + b*log(x), where c = log(a)
    mp = (x > 0) & (y > 0)
    x_p = x[mp]
    y_p = y[mp]
    if x_p.size < 3:
        raise ValueError("Power-law fit requires at least 3 points with x>0 and y>0.")

    lx = np.log(x_p)
    ly = np.log(y_p)

    def _lin_form(lx_in, c, b):
        return c + b * lx_in

    popt, pcov = curve_fit(_lin_form, lx, ly)
    c_hat, b_hat = popt
    if pcov is not None and np.all(np.isfinite(pcov)) and pcov.shape == (2, 2):
        se_c, se_b = np.sqrt(np.diag(pcov))
    else:
        se_c, se_b = (np.nan, np.nan)
    a_hat = float(np.exp(c_hat))
    se_a = float(a_hat * se_c) if np.isfinite(se_c) else np.nan

    r_log, _ = pearsonr(lx, ly)

    # p-value via F-test on slope in log-space
    yhat_log = _lin_form(lx, c_hat, b_hat)
    resid = ly - yhat_log
    sse = float(np.sum(resid**2))
    tss = float(np.sum((ly - np.mean(ly)) ** 2))
    ssr = max(tss - sse, 0.0)
    n = int(lx.size)
    df2 = max(n - 2, 1)
    if sse <= 0 or tss <= 0:
        p_f = np.nan
    else:
        f_stat = (ssr / 1.0) / (sse / df2)
        p_f = float(f_dist.sf(f_stat, 1, df2))

    def _predict(x_new):
        x_new = np.asarray(x_new, dtype=float)
        return a_hat * np.power(x_new, b_hat)

    return {
        "model": "power",
        "params": {"a": float(a_hat), "b": float(b_hat)},
        "r": float(r_log),
        "p": float(p_f),
        "stderr": {"a": float(se_a), "b": float(se_b)},
        "predict": _predict,
        "n": int(x_p.size),
    }


def compare_models(x, y):
    """
    Fit both linear and power-law models and compare them on original y-space.

    Metrics:
    - R² = 1 - RSS/TSS
    - RMSE = sqrt(RSS/n)
    - AIC = n*log(RSS/n) + 2*k, with k=2 parameters

    Notes
    -----
    The power-law model requires x>0 and y>0. For a fair comparison, both models
    are evaluated on the common subset where x>0 and y>0.
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    if x.size < 3:
        raise ValueError("Need at least 3 finite points to compare models.")

    mp = (x > 0) & (y > 0)
    x_cmp = x[mp]
    y_cmp = y[mp]
    if x_cmp.size < 3:
        raise ValueError("Need at least 3 points with x>0,y>0 to compare models.")

    res_lin = fit_model(x_cmp, y_cmp, model="linear")
    res_pow = fit_model(x_cmp, y_cmp, model="power")

    def _metrics(res):
        yhat = res["predict"](x_cmp)
        resid = y_cmp - yhat
        rss = float(np.sum(resid**2))
        tss = float(np.sum((y_cmp - np.mean(y_cmp)) ** 2))
        r2 = 1.0 - (rss / tss) if tss > 0 else np.nan
        rmse = float(np.sqrt(rss / max(len(y_cmp), 1)))
        k = 2
        aic = float(len(y_cmp) * np.log(rss / len(y_cmp)) + 2 * k) if rss > 0 else np.nan
        return {"r2": r2, "rmse": rmse, "rss": rss, "aic": aic}

    out = {
        "linear": {**res_lin, **_metrics(res_lin)},
        "power": {**res_pow, **_metrics(res_pow)},
    }
    aic_lin = out["linear"]["aic"]
    aic_pow = out["power"]["aic"]
    if np.isfinite(aic_lin) and np.isfinite(aic_pow):
        out["preferred"] = "linear" if aic_lin < aic_pow else "power"
    else:
        out["preferred"] = "linear"
    return out


def _coarsen_precipitation_to_grace(precip_data, grace_reference, method='interp'):
    """
    Coarsen precipitation data to match GRACE spatial resolution.
    
    Parameters:
    -----------
    precip_data : xarray.DataArray
        Precipitation data (may be higher resolution than GRACE)
    grace_reference : xarray.DataArray
        Reference GRACE data to match resolution
    method : str, default='interp'
        Method to use: 'interp' (interpolation) or 'coarsen' (coarsening)
        
    Returns:
    --------
    xarray.DataArray : Coarsened precipitation data matching GRACE resolution
    """
    precip_data.rio.write_crs("EPSG:4326", inplace=True)
    precip_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    
    grace_lat = grace_reference.lat
    grace_lon = grace_reference.lon
    
    if method == 'interp':
        precip_coarsened = precip_data.interp(lat=grace_lat, lon=grace_lon, method='linear')
    else:
        if len(grace_lat) < 2 or len(grace_lon) < 2:
            _raise_ctx(
                ValueError,
                f"GRACE reference lat/lon must have length >= 2 to estimate "
                f"resolution for coarsening (got lat={len(grace_lat)}, lon={len(grace_lon)})",
            )
        if len(precip_data.lat) < 2 or len(precip_data.lon) < 2:
            _raise_ctx(
                ValueError,
                f"Precipitation lat/lon must have length >= 2 to estimate "
                f"resolution for coarsening (got lat={len(precip_data.lat)}, "
                f"lon={len(precip_data.lon)})",
            )
        delta_lat_grace = np.abs(grace_lat.values[1] - grace_lat.values[0])
        delta_lat_precip = np.abs(precip_data.lat.values[1] - precip_data.lat.values[0])
        
        if delta_lat_precip < delta_lat_grace:
            coarsen_factor = int(np.round(delta_lat_grace / delta_lat_precip))
            precip_coarsened = precip_data.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary="trim").mean()
            precip_coarsened = precip_coarsened.interp(lat=grace_lat, lon=grace_lon, method='nearest')
        else:
            precip_coarsened = precip_data.interp(lat=grace_lat, lon=grace_lon, method='nearest')
    
    return precip_coarsened


def _cluster_events_within_window(extreme_dates, window_days=365):
    """
    Cluster extreme event dates within a specified window.
    
    Parameters:
    -----------
    extreme_dates : pandas.DatetimeIndex or array-like
        Dates of extreme events
    window_days : int, default=365
        Window size in days for clustering
        
    Returns:
    --------
    list : List of tuples (cluster_start, cluster_end)
    """
    if len(extreme_dates) == 0:
        return []
    
    if len(extreme_dates) == 1:
        return [(extreme_dates[0], extreme_dates[0])]
    
    sorted_dates = pd.DatetimeIndex(extreme_dates).sort_values()
    date_diffs = np.diff(sorted_dates.values)
    
    if isinstance(sorted_dates, pd.DatetimeIndex):
        gaps = date_diffs > pd.Timedelta(days=window_days)
    else:
        gaps = date_diffs > np.timedelta64(window_days, 'D')
    
    clusters = []
    if not gaps.any():
        clusters.append((sorted_dates[0], sorted_dates[-1]))
    else:
        cluster_starts = np.concatenate([[0], np.where(gaps)[0] + 1])
        cluster_ends = np.concatenate([np.where(gaps)[0], [len(sorted_dates) - 1]])
        
        for start_idx, end_idx in zip(cluster_starts, cluster_ends):
            clusters.append((sorted_dates[start_idx], sorted_dates[end_idx]))
    
    return clusters


def _nanstd_safe(values, ddof=1, default=0.0):
    """Sample std that returns ``default`` when fewer than ``ddof + 1`` finite values."""
    arr = np.asarray(values, dtype=float)
    n_finite = int(np.isfinite(arr).sum())
    if n_finite <= ddof:
        return default
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice",
            category=RuntimeWarning,
        )
        return float(np.nanstd(arr, ddof=ddof))


def _epe_baseline_window_bounds(start_evt, end_evt):
    """
    Calendar-month windows that exclude months overlapping the EPE cluster.

    Pre-baseline: the 12 months ending in the month *before* the calendar month of ``start_evt``
    (e.g. cluster starting in June uses May and the 11 preceding months).

    Post-baseline: the 12 months starting in the month *after* the calendar month of ``end_evt``
    (e.g. cluster ending in September uses October and the following 11 months).

    ``start_evt`` / ``end_evt`` may be any timestamp within the cluster month range; bounds are
    derived from their month (``Period('M')``).
    """
    p_start = pd.Timestamp(start_evt).to_period('M')
    p_end = pd.Timestamp(end_evt).to_period('M')
    pre_lo = (p_start - 12).to_timestamp(how='start')
    pre_hi = (p_start - 1).to_timestamp(how='end')
    post_lo = (p_end + 1).to_timestamp(how='start')
    post_hi = (p_end + 12).to_timestamp(how='end')
    return pre_lo, pre_hi, post_lo, post_hi


def _calculate_pixel_response(grace_solution_ts_list, precip_ts, clusters, min_vals_before_after=6, grace_threshold=0.0):
    """
    Calculate valid GRACE responses for a single pixel.
    
    For each event cluster:
    1. For each GRACE solution, compute pre- and post-baseline means in 12-month windows that
       **exclude** cluster months (see ``_epe_baseline_window_bounds``).
    2. Per solution, difference = post_mean - pre_mean.
    3. Ensemble cluster response = mean of those per-solution differences; σ_k = sample std
       across solutions (ddof=1), or 0 if only one solution contributes.
    4. Validity: ensemble mean > ``grace_threshold`` and ensemble mean > σ_k.
    
    Parameters:
    -----------
    grace_solution_ts_list : list of pandas.Series
        List of GRACE solution time series for the pixel (one per solution, typically 3)
    precip_ts : pandas.Series
        Precipitation time series for the pixel
    clusters : list
        List of (start_date, end_date) tuples for event clusters
    min_vals_before_after : int, default=6
        Minimum number of values required before and after event
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_precip': Total precipitation from extreme events
        - 'valid_response_sum': Sum of valid GRACE responses
        - 'valid_std_sum': Combined uncertainty (std) for valid responses, summed in quadrature
                          (sqrt(sum of squares)) for independent uncertainties
        - 'n_valid_events': Number of valid events
        - 'clusters_data': List of cluster data dictionaries
    """
    total_precip = 0.0
    valid_response_sum = 0.0
    valid_std_sum_squared = 0.0  # Accumulate squares for quadrature summation
    n_valid_events = 0
    clusters_data = []
    
    for start_evt, end_evt in clusters:
        pre_lo, pre_hi, post_lo, post_hi = _epe_baseline_window_bounds(start_evt, end_evt)
        
        solution_diffs = []
        solution_before_means = []
        solution_after_means = []
        
        for grace_sol_ts in grace_solution_ts_list:
            before_mask = (grace_sol_ts.index >= pre_lo) & (grace_sol_ts.index <= pre_hi)
            after_mask = (grace_sol_ts.index >= post_lo) & (grace_sol_ts.index <= post_hi)
            before_vals = grace_sol_ts[before_mask]
            after_vals = grace_sol_ts[after_mask]
            
            before_count = (~np.isnan(before_vals.values)).sum() if len(before_vals) > 0 else 0
            after_count = (~np.isnan(after_vals.values)).sum() if len(after_vals) > 0 else 0
            
            if (before_count < min_vals_before_after) or (after_count < min_vals_before_after):
                continue
            
            avg_before = float(np.nanmean(before_vals.values)) if len(before_vals) > 0 else np.nan
            avg_after = float(np.nanmean(after_vals.values)) if len(after_vals) > 0 else np.nan
            
            if np.isnan(avg_before) or np.isnan(avg_after):
                continue
            
            diff = avg_after - avg_before
            solution_diffs.append(diff)
            solution_before_means.append(avg_before)
            solution_after_means.append(avg_after)
        
        try:
            cluster_precip = precip_ts.loc[start_evt:end_evt]
            precip_sum = float(cluster_precip.sum()) if len(cluster_precip) > 0 else 0.0
        except (KeyError, IndexError):
            precip_sum = 0.0
        
        total_precip += precip_sum
        
        if len(solution_diffs) == 0:
            clusters_data.append({
                'cluster_start': start_evt,
                'cluster_end': end_evt,
                'diff_mean': np.nan,
                'diff_std': np.nan,
                'is_valid': False,
                'precip_sum': precip_sum,
                'avg_before_mean': np.nan,
                'avg_after_mean': np.nan,
                'avg_before_std': np.nan,
                'avg_after_std': np.nan
            })
            continue
        
        solution_diffs_array = np.array(solution_diffs, dtype=float)
        diff_mean = np.nanmean(solution_diffs_array)
        diff_std = _nanstd_safe(solution_diffs_array, ddof=1, default=0.0)

        avg_before_mean = np.nanmean(solution_before_means) if len(solution_before_means) > 0 else np.nan
        avg_after_mean = np.nanmean(solution_after_means) if len(solution_after_means) > 0 else np.nan
        avg_before_std = _nanstd_safe(solution_before_means, ddof=1, default=0.0)
        avg_after_std = _nanstd_safe(solution_after_means, ddof=1, default=0.0)
        
        if np.isnan(diff_mean):
            clusters_data.append({
                'cluster_start': start_evt,
                'cluster_end': end_evt,
                'diff_mean': np.nan,
                'diff_std': np.nan,
                'is_valid': False,
                'precip_sum': precip_sum,
                'avg_before_mean': avg_before_mean,
                'avg_after_mean': avg_after_mean,
                'avg_before_std': avg_before_std,
                'avg_after_std': avg_after_std
            })
            continue
        
        is_valid = (diff_mean > grace_threshold) and (diff_mean > diff_std)
        
        if is_valid:
            valid_response_sum += diff_mean
            valid_std_sum_squared += diff_std ** 2  # Sum in quadrature (accumulate squares)
            n_valid_events += 1
        
        clusters_data.append({
            'cluster_start': start_evt,
            'cluster_end': end_evt,
            'diff_mean': diff_mean,
            'diff_std': diff_std,
            'is_valid': is_valid,
            'precip_sum': precip_sum,
            'avg_before_mean': avg_before_mean,
            'avg_after_mean': avg_after_mean,
            'avg_before_std': avg_before_std,
            'avg_after_std': avg_after_std
        })
    
    # Calculate final uncertainty by taking square root of sum of squares (quadrature)
    valid_std_sum = np.sqrt(valid_std_sum_squared) if valid_std_sum_squared > 0 else 0.0
    
    return {
        'total_precip': total_precip,
        'valid_response_sum': valid_response_sum,
        'valid_std_sum': valid_std_sum,
        'n_valid_events': n_valid_events,
        'clusters_data': clusters_data
    }


def _process_single_pixel(
    i, j, lat, lon,
    grace_aligned_np, precip_np, extreme_mask_pixel,
    common_time_idx, decompose_grace, grace_threshold,
    decompose_fn
):
    """
    Process a single pixel: decompose GRACE (optional), cluster extreme events,
    and calculate the GRACE response. Designed to be called in parallel via joblib.

    Parameters
    ----------
    i, j : int
        Row/column indices into the (lat, lon) grid.
    lat, lon : float
        Geographic coordinates for this pixel.
    grace_aligned_np : list of np.ndarray
        Pre-extracted numpy arrays, each shape (n_time,), one per GRACE solution.
    precip_np : np.ndarray
        Precipitation values for this pixel, shape (n_time,).
    extreme_mask_pixel : np.ndarray
        Boolean array, shape (n_time,), True where precipitation is extreme.
    common_time_idx : pd.DatetimeIndex
        Shared time axis.
    decompose_grace : bool
        Whether to decompose GRACE into residuals.
    grace_threshold : float
        Minimum response amplitude for validity.
    decompose_fn : callable or None
        The decompose_grace_sin_cosin function (passed to avoid import issues in workers).

    Returns
    -------
    dict with keys 'i', 'j', 'total_precip', 'valid_response_sum', 'valid_std_sum', 'events'.
    """
    precip_ts = pd.Series(precip_np, index=common_time_idx, name="precip")

    grace_solution_ts_list = []
    if decompose_grace and decompose_fn is not None:
        for sol_vals in grace_aligned_np:
            if np.all(np.isnan(sol_vals)):
                grace_solution_ts_list.append(
                    pd.Series(np.full(len(common_time_idx), np.nan), index=common_time_idx)
                )
                continue

            pixel_da = xr.DataArray(sol_vals, dims=['time'],
                                    coords={'time': common_time_idx})
            _, _, residual = decompose_fn(pixel_da, pixel_da.time)

            residual_arr = residual.values if hasattr(residual, 'values') else np.array(residual)

            if len(residual_arr) == len(common_time_idx):
                grace_solution_ts_list.append(
                    pd.Series(residual_arr, index=common_time_idx)
                )
            else:
                grace_solution_ts_list.append(
                    pd.Series(np.full(len(common_time_idx), np.nan), index=common_time_idx)
                )
    else:
        for sol_vals in grace_aligned_np:
            grace_solution_ts_list.append(pd.Series(sol_vals, index=common_time_idx))

    extreme_dates = common_time_idx[extreme_mask_pixel]
    if len(extreme_dates) == 0:
        return {'i': i, 'j': j, 'total_precip': 0.0,
                'valid_response_sum': 0.0, 'valid_std_sum': 0.0, 'events': []}

    clusters = _cluster_events_within_window(extreme_dates, window_days=365)

    pixel_result = _calculate_pixel_response(
        grace_solution_ts_list, precip_ts, clusters,
        min_vals_before_after=6, grace_threshold=grace_threshold
    )

    events = []
    for cd in pixel_result['clusters_data']:
        events.append({
            'lat': lat, 'lon': lon,
            'cluster_start': cd['cluster_start'],
            'cluster_end': cd['cluster_end'],
            'diff_mean': cd['diff_mean'],
            'diff_std': cd['diff_std'],
            'is_valid': cd['is_valid'],
            'precip_sum': cd['precip_sum'],
            'avg_before_mean': cd.get('avg_before_mean', np.nan),
            'avg_after_mean': cd.get('avg_after_mean', np.nan),
            'avg_before_std': cd.get('avg_before_std', np.nan),
            'avg_after_std': cd.get('avg_after_std', np.nan),
        })

    return {
        'i': i, 'j': j,
        'total_precip': pixel_result['total_precip'],
        'valid_response_sum': pixel_result['valid_response_sum'],
        'valid_std_sum': pixel_result['valid_std_sum'],
        'events': events,
    }


def analyze_grace_response_by_pixel(
    grace_solutions,
    precip_data,
    threshold_percentile=0.95,
    precip_floor=10.0,
    exclude_years=[2017, 2018],
    aoi_geometry=None,
    decompose_grace=False,
    grace_threshold=0.0,
    aquifer_gdf=None,
    aquifer_ids=None,
    id_col="subbasin_id",
    n_jobs=-1
):
    """
    Analyze GRACE response to extreme precipitation events at pixel level.
    
    For each pixel:
    1. Calculate mean and std across GRACE solutions
    2. Flag extreme precipitation events (percentile threshold and > precip_floor)
    3. Cluster events within 12 months
    4. Per cluster and solution, baseline means in 12-month windows before/after the cluster
       (excluding cluster months; see ``_epe_baseline_window_bounds``)
    5. Count valid responses (ensemble mean > grace_threshold AND ensemble mean > cross-solution std)
    
    Parameters:
    -----------
    grace_solutions : list of xarray.DataArray
        List of GRACE solution DataArrays (e.g., [grace_csr, grace_jpl, grace_gsfc])
    precip_data : xarray.DataArray
        Monthly precipitation data
    threshold_percentile : float, default=0.95
        Percentile threshold for extreme precipitation (0-1)
    precip_floor : float, default=10.0
        Minimum precipitation value (mm) to be considered extreme
    exclude_years : list, default=[2017, 2018]
        Years to exclude from analysis
    aoi_geometry : GeoSeries or GeoDataFrame, optional
        Area of interest geometry to clip data. If aquifer_gdf and aquifer_ids are provided,
        this will be overridden by the filtered aquifer geometry.
    decompose_grace : bool, default=False
        If True, decompose GRACE to residual before analysis
    aquifer_gdf : GeoDataFrame, optional
        GeoDataFrame with aquifer boundaries (e.g., arid_aquifers_EPE95_sorted).
        Used with aquifer_ids to filter specific aquifers.
    aquifer_ids : list, optional
        List of aquifer IDs to filter from aquifer_gdf. Filters based on id_col.
        Only used if aquifer_gdf is provided.
    id_col : str, default="subbasin_id"
        Column name for aquifer ID in aquifer_gdf. Only used if aquifer_gdf is provided.
    n_jobs : int, default=-1
        Number of parallel workers for pixel processing (joblib).
        -1 uses all available CPU cores; 1 disables parallelism (sequential).
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'total_precip': xarray.DataArray with total precipitation from EPEs per pixel
        - 'valid_response_sum': xarray.DataArray with sum of valid GRACE responses per pixel
        - 'valid_std_sum': xarray.DataArray with sum of std for valid responses per pixel
        - 'grace_mean': xarray.DataArray with mean GRACE time series
        - 'grace_std': xarray.DataArray with std GRACE time series
        - 'precip_coarsened': xarray.DataArray with coarsened precipitation
        - 'events_dataframe': pandas.DataFrame with all event/cluster data per pixel
    """
    try:
        from grace_analysis_utils import decompose_grace_sin_cosin
    except ImportError:
        raise ImportError("Cannot import decompose_grace_sin_cosin from grace_analysis_utils.")

    if grace_solutions is None or len(grace_solutions) == 0:
        _raise_ctx(
            ValueError,
            "grace_solutions is empty; provide at least one GRACE DataArray",
        )
    _required_dims = ("time", "lat", "lon")
    for idx, grace in enumerate(grace_solutions):
        missing = [d for d in _required_dims if d not in getattr(grace, "dims", ())]
        if missing:
            _raise_ctx(
                ValueError,
                f"grace_solutions[{idx}] missing required dimensions {missing}; "
                f"need {_required_dims}",
            )
    precip_missing = [d for d in _required_dims if d not in getattr(precip_data, "dims", ())]
    if precip_missing:
        _raise_ctx(
            ValueError,
            f"precip_data missing required dimensions {precip_missing}; "
            f"need {_required_dims}",
        )
    
    # Process GRACE solutions
    grace_processed = []
    for grace in grace_solutions:
        grace_copy = grace.copy()
        grace_copy.rio.write_crs("EPSG:4326", inplace=True)
        grace_copy.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        grace_copy = grace_copy.sel(time=~grace_copy.time.dt.year.isin(exclude_years))
        grace_processed.append(grace_copy)
    
    grace_solutions = grace_processed
    
    # Filter aquifer_gdf by IDs if provided (overrides aoi_geometry if both provided)
    if aquifer_gdf is not None and aquifer_ids is not None:
        aquifer_gdf_filtered = aquifer_gdf[aquifer_gdf[id_col].isin(aquifer_ids)].copy()
        if len(aquifer_gdf_filtered) == 0:
            raise ValueError(f"No aquifers found with IDs: {aquifer_ids}")
        # Use filtered aquifer geometry as aoi_geometry
        aoi_geometry = aquifer_gdf_filtered.geometry
        _note(f"filtered to {len(aquifer_gdf_filtered)} aquifer(s) with IDs: {aquifer_ids}")
    
    # Clip to AOI if provided (simplified for GeoSeries/GeoDataFrame)
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.to_crs("EPSG:4326")
        else:
            # Fallback for single geometry
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        
        geom_clip = [mapping(geom.buffer(0)) for geom in aoi_gdf.geometry]
        
        try:
            grace_solutions = [g.rio.clip(geom_clip, crs="EPSG:4326", drop=True) for g in grace_solutions]
            precip_data = precip_data.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
        except Exception as e:
            raise ValueError(f"Cannot clip data to AOI geometry: {e}") from e
    
    # Remove excluded years from precipitation
    precip_data = precip_data.sel(time=~precip_data.time.dt.year.isin(exclude_years))
    
    # Align GRACE solutions
    grace_ref = grace_solutions[0]
    grace_aligned = []
    for grace in grace_solutions:
        grace_aligned_i, _ = xr.align(grace, grace_ref, join='inner')
        grace_aligned.append(grace_aligned_i)
    
    # Calculate reference mean for precipitation coarsening
    grace_stack_ref = xr.concat(grace_aligned, dim='solution', coords='minimal')
    grace_mean_ref = grace_stack_ref.mean(dim='solution')
    
    # Coarsen precipitation to match GRACE resolution
    precip_coarsened = _coarsen_precipitation_to_grace(precip_data, grace_mean_ref, method='coarsen')
    
    # Align time dimensions
    grace_mean_ref, precip_coarsened = xr.align(grace_mean_ref, precip_coarsened, join='inner')
    if len(grace_mean_ref.time) == 0:
        _raise_ctx(
            ValueError,
            "No overlapping time between GRACE solutions and precipitation "
            "after alignment and year exclusion",
        )
    
    # Get aligned times (calculate once, reuse for all pixels)
    common_time_idx = pd.to_datetime(precip_coarsened.time.values)
    n_cpus = os.cpu_count() or 1
    if n_jobs is None or int(n_jobs) < 0:
        # Prefer get_resource_config clamp so joblib does not oversubscribe with Dask.
        try:
            from download_data import get_resource_config

            effective_jobs = int(get_resource_config().get("dask_workers") or max(1, n_cpus - 1))
        except Exception:
            effective_jobs = max(1, n_cpus - 1)
        n_jobs = effective_jobs
    else:
        effective_jobs = int(n_jobs)
    _announce(
        f"pixel EPE analysis: decompose_grace={bool(decompose_grace)}  "
        f"n_jobs={effective_jobs}  (index-based residual if decompose_grace=True)"
    )
    
    # Initialize output arrays
    total_precip_array = np.zeros(grace_mean_ref.shape[1:])
    valid_response_array = np.zeros(grace_mean_ref.shape[1:])
    valid_std_array = np.zeros(grace_mean_ref.shape[1:])
    
    all_events_data = []
    lats = grace_mean_ref.lat.values
    lons = grace_mean_ref.lon.values
    
    # OPTIMIZATION: Create mask of valid pixels upfront (pixels with at least some non-NaN data)
    # This allows us to skip all-NaN pixels immediately without processing them
    # Check GRACE data: at least one solution must have non-NaN data
    grace_valid_mask = np.zeros(grace_mean_ref.shape[1:], dtype=bool)
    for grace_sol in grace_aligned:
        grace_valid_mask |= ~np.isnan(grace_sol.values).all(axis=0)
    
    # Check precipitation: must have non-NaN data
    precip_valid_mask = ~np.isnan(precip_coarsened.values).all(axis=0)
    
    # Combined mask: pixel is valid if both GRACE and precipitation have data
    valid_pixel_mask = grace_valid_mask & precip_valid_mask
    
    # OPTIMIZATION: Vectorized threshold and extreme-mask computation for the full grid
    threshold_da = precip_coarsened.quantile(threshold_percentile, dim='time', skipna=True)
    threshold_da = xr.where(threshold_da < precip_floor, precip_floor, threshold_da)
    extreme_mask = (precip_coarsened > threshold_da) & (precip_coarsened > precip_floor)
    extreme_mask_np = extreme_mask.values  # (time, lat, lon) boolean array

    # Precompute indices of valid pixels so the loop and progress bar reflect only actual work
    valid_pixel_indices = np.argwhere(valid_pixel_mask)
    n_valid_pixels = len(valid_pixel_indices)
    n_total_pixels = len(lats) * len(lons)
    
    print(f"Grid: {n_total_pixels} total pixels, {n_valid_pixels} valid (skipping {n_total_pixels - n_valid_pixels} all-NaN)")

    # Pre-extract numpy arrays to avoid passing large xarray objects to workers
    grace_aligned_np_all = [sol.values for sol in grace_aligned]  # each (time, lat, lon)
    precip_np_all = precip_coarsened.values                       # (time, lat, lon)

    decompose_fn = decompose_grace_sin_cosin if decompose_grace else None

    # Parallel pixel processing
    results = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_process_single_pixel)(
            i, j, lats[i], lons[j],
            [sol[:, i, j] for sol in grace_aligned_np_all],
            precip_np_all[:, i, j],
            extreme_mask_np[:, i, j],
            common_time_idx, decompose_grace, grace_threshold,
            decompose_fn
        )
        for i, j in tqdm(valid_pixel_indices, desc="Dispatching valid pixels")
    )

    # Assemble results from parallel workers
    for res in results:
        ri, rj = res['i'], res['j']
        total_precip_array[ri, rj] = res['total_precip']
        valid_response_array[ri, rj] = res['valid_response_sum']
        valid_std_array[ri, rj] = res['valid_std_sum']
        all_events_data.extend(res['events'])
    
    # Create output DataArrays
    total_precip_da = xr.DataArray(
        total_precip_array,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='total_precip_from_epe'
    )
    
    valid_response_da = xr.DataArray(
        valid_response_array,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='valid_grace_response_sum'
    )
    
    valid_std_da = xr.DataArray(
        valid_std_array,
        dims=['lat', 'lon'],
        coords={'lat': lats, 'lon': lons},
        name='valid_std_sum'
    )
    
    # Calculate reference mean/std arrays
    grace_mean_ref_array = grace_mean_ref.values
    # ddof=1 needs >=2 finite solutions; sparse NaNs at some (time,lat,lon) are harmless NaN std
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="Degrees of freedom <= 0 for slice",
            category=RuntimeWarning,
        )
        grace_std_ref_array = grace_stack_ref.std(dim='solution', ddof=1).values

    grace_mean_da = xr.DataArray(
        grace_mean_ref_array,
        dims=grace_mean_ref.dims,
        coords=grace_mean_ref.coords,
        name='grace_mean_reference'
    )

    grace_std_da = xr.DataArray(
        grace_std_ref_array,
        dims=grace_mean_ref.dims,
        coords=grace_mean_ref.coords,
        name='grace_std_reference'
    )

    events_df = pd.DataFrame(all_events_data)
    if events_df.empty or 'diff_mean' not in events_df.columns:
        n_events_analyzed = 0
        n_above_threshold = 0
        n_valid_events = 0
    else:
        analyzed_mask = np.isfinite(events_df['diff_mean'].to_numpy(dtype=float))
        n_events_analyzed = int(analyzed_mask.sum())
        n_above_threshold = int(
            (analyzed_mask & (events_df['diff_mean'].to_numpy(dtype=float) > float(grace_threshold))).sum()
        )
        if 'is_valid' in events_df.columns:
            n_valid_events = int(events_df['is_valid'].fillna(False).astype(bool).sum())
        else:
            n_valid_events = 0
    n_pixels_with_valid = int(np.count_nonzero(valid_response_array > 0))
    _announce(
        f"pixel EPE analysis done: {n_events_analyzed} events analyzed; "
        f"{n_above_threshold} above {float(grace_threshold):g} cm; "
        f"{n_valid_events} valid (> threshold and > inter-solution σ)  "
        f"[{n_pixels_with_valid}/{n_valid_pixels} pixels with valid response]"
    )

    return {
        'total_precip': total_precip_da,
        'valid_response_sum': valid_response_da,
        'valid_std_sum': valid_std_da,
        'grace_mean': grace_mean_da,
        'grace_std': grace_std_da,
        'precip_coarsened': precip_coarsened,
        'events_dataframe': events_df
    }


def analyze_grace_response_by_aquifer_pixel(
    grace_solutions,
    precip_data,
    aquifer_gdf,
    threshold_percentile=0.95,
    precip_floor=10.0,
    exclude_years=[2017, 2018],
    id_col="subbasin_id",
    aq_name_col="aq_name",
    decompose_grace=False,
    grace_threshold=0.0,
    pixel_sum=True,
    one_plot=True,
    aquifer_ids=None,
    annotations=False,
    annotation_threshold=1.5,
    save_dir=None
):
    """
    Analyze GRACE response by aquifer using pixel-based approach.
    
    Parameters:
    -----------
    grace_solutions : list of xarray.DataArray
        List of GRACE solution DataArrays
    precip_data : xarray.DataArray
        Monthly precipitation data
    aquifer_gdf : GeoDataFrame
        GeoDataFrame with aquifer boundaries
    threshold_percentile : float, default=0.95
        Percentile threshold for extreme precipitation
    precip_floor : float, default=10.0
        Minimum precipitation value (mm) for extreme events
    exclude_years : list, default=[2017, 2018]
        Years to exclude from analysis
    id_col : str, default="subbasin_id"
        Column name for aquifer ID
    aq_name_col : str, default="aq_name"
        Column name for aquifer name
    decompose_grace : bool, default=False
        If True, decompose GRACE to residual
    pixel_sum : bool, default=True
        If True, aggregate valid events per pixel (sum precip, sum response, sum std).
        If False, plot all individual events from all pixels.
    one_plot : bool, default=True
        If True, create a single combined plot and statistics for all aquifers.
        If False, create separate plots for each aquifer.
    aquifer_ids : list, optional
        List of aquifer IDs to process. If None, processes all aquifers in aquifer_gdf.
        Filters based on id_col (e.g., subbasin_id).
    annotations : bool, default=False
        If True, annotate outlier points (drifting points) with lat/lon coordinates
        on the relationship plots.
    save_dir : str, optional
        Directory to save relationship plots
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'aquifer_results': dict keyed by aquifer_id, each containing:
            - 'aq_name': str, aquifer name
            - 'pixel_results': dict from analyze_grace_response_by_pixel
            - 'n_events': int, number of valid events
            - 'r': float, correlation coefficient
            - 'r_squared': float, R-squared value
            - 'p_value': float, p-value
            - 'slope': float, regression slope
            - 'intercept': float, regression intercept
        - 'relationship_df': pandas.DataFrame with all relationship data for all aquifers
            Columns: aquifer_id, aquifer_name, total_precip, valid_response, valid_std, 
                     lat, lon, cluster_start, cluster_end
    """
    from grace_analysis_utils import _geometry_to_clip_format
    
    # Prepare GRACE solutions and precipitation
    grace_solutions_prepared = []
    for grace in grace_solutions:
        grace_prep = grace.rio.write_crs("EPSG:4326")
        grace_prep = grace_prep.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
        grace_solutions_prepared.append(grace_prep)
    
    precip_data_prep = precip_data.rio.write_crs("EPSG:4326")
    precip_data_prep = precip_data_prep.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
    
    aquifer_gdf = aquifer_gdf.to_crs("EPSG:4326")
    
    # Filter aquifers by ID list if provided
    if aquifer_ids is not None:
        aquifer_gdf = aquifer_gdf[aquifer_gdf[id_col].isin(aquifer_ids)].copy()
        if len(aquifer_gdf) == 0:
            _note(
                f"No aquifers found with IDs {aquifer_ids} "
                f"(column {id_col!r}); returning empty results"
            )
            return {}, pd.DataFrame()
        _note(f"Processing {len(aquifer_gdf)} aquifer(s) from filtered list: {aquifer_ids}")
    
    aquifer_results = {}
    all_relationship_data = []
    n_skipped = 0
    skipped_examples = []
    
    # Process each aquifer
    for row in tqdm(aquifer_gdf.itertuples(), total=len(aquifer_gdf), desc="Processing aquifers"):
        aquifer_id = getattr(row, id_col, None)
        aquifer_name = getattr(row, aq_name_col, f"Aquifer {aquifer_id}")
        
        try:
            geom_clip = _geometry_to_clip_format(row.geometry)
            
            # Clip GRACE solutions
            grace_clipped = []
            for grace_prep in grace_solutions_prepared:
                grace_c = grace_prep.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
                grace_c = grace_c.rio.write_crs("EPSG:4326")
                grace_c = grace_c.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
                grace_clipped.append(grace_c)
            
            # Clip precipitation
            precip_clipped = precip_data_prep.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
            precip_clipped = precip_clipped.rio.write_crs("EPSG:4326")
            precip_clipped = precip_clipped.rio.set_spatial_dims(x_dim="lon", y_dim="lat")
            
            # Run pixel analysis
            pixel_results = analyze_grace_response_by_pixel(
                grace_solutions=grace_clipped,
                precip_data=precip_clipped,
                threshold_percentile=threshold_percentile,
                precip_floor=precip_floor,
                exclude_years=exclude_years,
                aoi_geometry=None,
                decompose_grace=decompose_grace,
                grace_threshold=grace_threshold
            )
            
            events_df = pixel_results.get('events_dataframe', pd.DataFrame())
            
            if events_df.empty or 'is_valid' not in events_df.columns:
                n_skipped += 1
                skipped_examples.append(aquifer_id)
                continue
            
            valid_events_df = events_df[events_df['is_valid'] == True].copy()
            
            if len(valid_events_df) < 2:
                n_skipped += 1
                skipped_examples.append(aquifer_id)
                continue
            
            if pixel_sum:
                # Aggregate per pixel: sum precip, sum response, sum std in quadrature for each pixel
                # Sum std in quadrature (square root of sum of squares) since these are independent uncertainties
                pixel_summary = valid_events_df.groupby(['lat', 'lon']).agg({
                    'precip_sum': 'sum',
                    'diff_mean': 'sum',
                    'diff_std': lambda x: np.sqrt(np.nansum(x**2))  # Quadrature sum
                }).reset_index()
                
                total_precip_valid = pixel_summary['precip_sum'].values
                valid_response_valid = pixel_summary['diff_mean'].values
                valid_std_valid = pixel_summary['diff_std'].values
                
                # Store aggregated data for relationship_df
                plot_df = pixel_summary.copy()
                plot_df['aquifer_id'] = aquifer_id
                plot_df['aquifer_name'] = aquifer_name
                plot_df.rename(columns={'diff_mean': 'valid_response', 'diff_std': 'valid_std', 
                                       'precip_sum': 'total_precip'}, inplace=True)
                plot_df['cluster_start'] = None
                plot_df['cluster_end'] = None
                
            else:
                # Use all individual events
                total_precip_valid = valid_events_df['precip_sum'].values
                valid_response_valid = valid_events_df['diff_mean'].values
                valid_std_valid = valid_events_df['diff_std'].values
                
                # Store all events for relationship_df
                plot_df = valid_events_df[['lat', 'lon', 'precip_sum', 'diff_mean', 'diff_std',
                                          'cluster_start', 'cluster_end']].copy()
                plot_df['aquifer_id'] = aquifer_id
                plot_df['aquifer_name'] = aquifer_name
                plot_df.rename(columns={'diff_mean': 'valid_response', 'diff_std': 'valid_std',
                                       'precip_sum': 'total_precip'}, inplace=True)
            
            valid_mask = ~(np.isnan(total_precip_valid) | np.isnan(valid_response_valid))
            total_precip_valid = total_precip_valid[valid_mask]
            valid_response_valid = valid_response_valid[valid_mask]
            valid_std_valid = valid_std_valid[valid_mask]
            
            # Extract corresponding lat/lon for valid points if annotations requested
            lats_valid = None
            lons_valid = None
            if annotations:
                lats_valid = plot_df['lat'].values[valid_mask] if 'lat' in plot_df.columns else None
                lons_valid = plot_df['lon'].values[valid_mask] if 'lon' in plot_df.columns else None
            
            if len(total_precip_valid) < 2:
                continue
            
            # Calculate statistics
            r_val, p_val = pearsonr(total_precip_valid, valid_response_valid)
            slope, intercept, _, _, _ = linregress(total_precip_valid, valid_response_valid)
            r_squared = r_val ** 2
            
            # Store results
            aquifer_results[aquifer_id] = {
                'aq_name': aquifer_name,
                'pixel_results': pixel_results,
                'n_events': len(total_precip_valid),
                'r': r_val,
                'r_squared': r_squared,
                'p_value': p_val,
                'slope': slope,
                'intercept': intercept
            }
            
            # Store for relationship_df
            all_relationship_data.append(plot_df)
            
            # Plot relationship based on one_plot flag
            if not one_plot:
                # Plot individual relationship for this aquifer
                _plot_aquifer_relationship(
                    total_precip_valid, valid_response_valid, valid_std_valid,
                    aquifer_name, threshold_percentile,
                    r_val, r_squared, slope, p_val, len(total_precip_valid),
                    save_dir=save_dir, aquifer_id=aquifer_id,
                    annotations=annotations, lats=lats_valid, lons=lons_valid, annotation_threshold=annotation_threshold
                )
            # If one_plot=True, skip individual plots (will create combined plot after loop)
            
        except Exception as e:
            n_skipped += 1
            skipped_examples.append(aquifer_id)
            _note(f"[Aquifer {aquifer_id} ({aquifer_name})] Error: {e}")
            continue

    _summarize_skipped(
        "aquifers (no/insufficient valid events or error)",
        n_skipped,
        len(aquifer_gdf),
        examples=skipped_examples,
    )
    
    if all_relationship_data:
        relationship_df = pd.concat(all_relationship_data, ignore_index=True)
    else:
        relationship_df = pd.DataFrame()
        if len(aquifer_gdf) > 0 and len(aquifer_results) == 0:
            _note(
                "analyze_grace_response_by_aquifer_pixel: no aquifer produced "
                "relationship data (need >=2 valid events per aquifer)"
            )
    
    # Create combined plot if one_plot=True
    if one_plot and len(aquifer_results) > 0:
        # Collect all data from all aquifers
        all_precip = []
        all_response = []
        all_std = []
        all_aquifer_names = []
        all_aquifer_ids = []
        all_lats = []
        all_lons = []
        
        for aq_id, aq_data in aquifer_results.items():
            # Get data from relationship_df for this aquifer
            aq_df = relationship_df[relationship_df['aquifer_id'] == aq_id].copy()
            if len(aq_df) == 0:
                continue
            
            valid_mask_aq = ~(np.isnan(aq_df['total_precip'].values) | 
                             np.isnan(aq_df['valid_response'].values))
            
            if np.sum(valid_mask_aq) < 2:
                continue
            
            all_precip.extend(aq_df['total_precip'].values[valid_mask_aq])
            all_response.extend(aq_df['valid_response'].values[valid_mask_aq])
            all_std.extend(aq_df['valid_std'].values[valid_mask_aq])
            all_aquifer_names.extend([aq_data['aq_name']] * np.sum(valid_mask_aq))
            all_aquifer_ids.extend([aq_id] * np.sum(valid_mask_aq))
            
            if 'lat' in aq_df.columns and 'lon' in aq_df.columns:
                all_lats.extend(aq_df['lat'].values[valid_mask_aq])
                all_lons.extend(aq_df['lon'].values[valid_mask_aq])
        
        if len(all_precip) >= 2:
            all_precip = np.array(all_precip)
            all_response = np.array(all_response)
            all_std = np.array(all_std)
            
            # Calculate combined statistics
            r_combined, p_combined = pearsonr(all_precip, all_response)
            slope_combined, intercept_combined, _, _, _ = linregress(all_precip, all_response)
            r_squared_combined = r_combined ** 2
            
            # Get lat/lon for annotations if requested
            lats_combined = np.array(all_lats) if len(all_lats) == len(all_precip) else None
            lons_combined = np.array(all_lons) if len(all_lons) == len(all_precip) else None
            
            # Create combined plot
            combined_title = f"All Aquifers (Combined)"
            _plot_aquifer_relationship(
                all_precip, all_response, all_std,
                combined_title, threshold_percentile,
                r_combined, r_squared_combined, slope_combined, p_combined, len(all_precip),
                save_dir=save_dir, aquifer_id=None,
                annotations=annotations, lats=lats_combined, lons=lons_combined, 
                annotation_threshold=annotation_threshold
            )
    
    return aquifer_results, relationship_df


def _plot_aquifer_relationship(
    total_precip, valid_response, valid_std, aquifer_name, threshold_percentile,
    r, r_squared, slope, p_value, n,
    save_dir=None, save_path=None, aquifer_id=None,
    annotations=False, lats=None, lons=None, annotation_threshold=1.5
):
    """Plot relationship between extreme precipitation and valid GRACE response for one aquifer."""
    fig, ax = plt.subplots(figsize=(6, 4), gridspec_kw={'wspace': 0, 'hspace': 0.15})
    
    # Calculate vmin/vmax for uncertainty colormap to ensure full color range is used
    valid_std_values = valid_std[~np.isnan(valid_std)]
    if len(valid_std_values) > 0:
        std_vmin = 0  # Always start from 0 for uncertainty
        std_vmax = np.nanmax(valid_std_values)
    else:
        std_vmin = 0
        std_vmax = 1
    
    scatter = ax.scatter(total_precip, valid_response, s=30, c=valid_std, 
                        cmap='Blues', edgecolors='black', linewidth=0.5,
                        vmin=std_vmin, vmax=std_vmax)
    
    # Regression line
    if not np.isnan(r) and len(total_precip) >= 2:
        slope, intercept, _, _, _ = linregress(total_precip, valid_response)
        x_line = np.linspace(total_precip.min(), total_precip.max(), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, 'r-', lw=2)
        
        # Annotate outliers (drifting points) if requested
        if annotations and lats is not None and lons is not None:
            predicted = slope * total_precip + intercept
            residuals = valid_response - predicted
            residual_std = np.nanstd(residuals)
            
            # Identify outliers: points with residuals > 1.5 * std (points that don't fit well)
            outlier_mask = np.abs(residuals) > annotation_threshold
            
            # Ensure lat/lon arrays match the data length
            if len(lats) == len(total_precip) and len(lons) == len(total_precip):
                for i in range(len(total_precip)):
                    if outlier_mask[i] and not (np.isnan(lats[i]) or np.isnan(lons[i])):
                        ax.annotate(
                            f"({lats[i]:.1f},{lons[i]:.1f})",
                            xy=(total_precip[i], valid_response[i]),
                            xytext=(5, 5),
                            textcoords='offset points',
                            fontsize=8,
                            alpha=0.7,
                            bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.5, edgecolor='black', linewidth=0.5),
                            arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0', lw=0.5, alpha=0.7)
                        )
    
    # Set x and y limits with 10% padding based on max value
    x_min, x_max = np.nanmin(total_precip), np.nanmax(total_precip)
    y_min, y_max = np.nanmin(valid_response), np.nanmax(valid_response)
    
    # Increase limits by 10% of the max value
    x_padding = 0.2 * abs(x_max) if abs(x_max) > 0 else 1.0
    y_padding = 0.4 * abs(y_max) if abs(y_max) > 0 else 1.0
    
    ax.set_xlim(x_min - x_padding, x_max + x_padding)
    ax.set_ylim(y_min - y_padding, y_max + y_padding)
    
    ax.set_xlabel('Cumulative EPEs (cm)', fontsize=11)
    ax.set_ylabel('Cumulative GRACE GWS (cm)', fontsize=11)
    ax.set_title(aquifer_name, fontsize=12)
    
    stats_text = f"r = {r:.2f}\nR² = {r_squared:.2f}\nslope = {slope:.2f}\np = {format_pvalue(p_value)}\nn = {n}"
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray'))
    
    cbar = plt.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label('Uncertainty (cm)', fontsize=10)
    
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')
    
    plt.tight_layout()
    
    # Save with priority: save_path > save_dir
    if save_path:
        save_path_obj = Path(save_path)
        save_path_obj.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path_obj, dpi=500, bbox_inches='tight', format='jpeg')
    elif save_dir:
        save_dir_obj = Path(save_dir)
        save_dir_obj.mkdir(parents=True, exist_ok=True)
        filename = f"aquifer_{aquifer_id}_relationship_{int(threshold_percentile*100)}th.jpeg"
        if aquifer_id is None:
            filename = f"{aquifer_name.replace(' ', '_')}_relationship_{int(threshold_percentile*100)}th.jpeg"
        plt.savefig(save_dir_obj / filename, dpi=500, bbox_inches='tight', format='jpeg')
    
    plt.show()


def _efficiency_classify_bounds_and_labels(efficiency_max, n_classes, data_array):
    """Equal classes in [0, efficiency_max] plus one overflow class (> max)."""
    n_classes = int(n_classes)
    emax = float(efficiency_max)
    inner_bounds = np.linspace(0.0, emax, n_classes)
    vals = np.asarray(data_array, dtype=float)
    vals = vals[np.isfinite(vals)]
    data_hi = float(np.nanmax(vals)) if vals.size else emax
    overflow_upper = max(data_hi * 1.01, emax * 1.01, emax + 1e-6)
    bounds = np.append(inner_bounds, overflow_upper)
    labels = []
    tick_locs = []
    for i in range(n_classes - 1):
        lo, hi = inner_bounds[i], inner_bounds[i + 1]
        labels.append(f"{lo:g}-{hi:g}")
        tick_locs.append((lo + hi) / 2.0)
    labels.append(f">{emax:g}")
    tick_locs.append((inner_bounds[-1] + overflow_upper) / 2.0)
    return bounds, labels, tick_locs


def _efficiency_class_counts(values, efficiency_max, n_classes):
    """Count pixels per efficiency class (matches classified map bins)."""
    n_classes = int(n_classes)
    emax = float(efficiency_max)
    vals = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(vals) & (vals > 0)
    n_valid = int(valid.sum())
    bounds, labels, _ = _efficiency_classify_bounds_and_labels(
        emax, n_classes, vals[valid] if n_valid else vals
    )
    rows = []
    for i in range(n_classes):
        lo, hi = float(bounds[i]), float(bounds[i + 1])
        if i < n_classes - 1:
            mask = valid & (vals >= lo) & (vals < hi)
        else:
            mask = valid & (vals >= lo) & (vals <= hi)
        n = int(mask.sum())
        pct = (100.0 * n / n_valid) if n_valid > 0 else np.nan
        rows.append(
            {
                "class": i,
                "label": labels[i],
                "lower": lo,
                "upper": hi if i < n_classes - 1 else np.inf,
                "n_pixels": n,
                "pct": pct,
            }
        )
    return pd.DataFrame(rows)


def _prepare_pixel_analysis_arrays(pixel_results, aoi_geometry=None):
    """
    AOI-mask pixel grids and derived fields used by ``plot_pixel_analysis_maps``.

    Returns arrays **before** the ``> 0`` plot filters so catalogs retain zeros
    and pixels valid for only some panels.
    """
    total_precip = pixel_results['total_precip']
    valid_response = pixel_results['valid_response_sum']
    valid_std = pixel_results['valid_std_sum']

    recharge_efficiency = xr.where(
        total_precip > 0,
        (valid_response / total_precip) * 100,
        np.nan,
    )
    recharge_efficiency.name = 'recharge_efficiency'
    recharge_efficiency = recharge_efficiency.assign_coords(
        lat=total_precip.lat, lon=total_precip.lon
    )

    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf_mask = gpd.GeoDataFrame(
                geometry=aoi_geometry, crs=aoi_geometry.crs
            ).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf_mask = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf_mask = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")

        ref_data = total_precip.copy()
        ref_data.rio.write_crs("EPSG:4326", inplace=True)
        ref_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

        transform = ref_data.rio.transform()
        mask_array = rasterize(
            [mapping(geom) for geom in aoi_gdf_mask.geometry],
            out_shape=(len(ref_data.lat), len(ref_data.lon)),
            transform=transform,
            fill=0,
            default_value=1,
            dtype=np.uint8,
        )

        mask_da = xr.DataArray(
            mask_array.astype(bool),
            coords={'lat': ref_data.lat, 'lon': ref_data.lon},
            dims=['lat', 'lon'],
        )

        total_precip = total_precip.where(mask_da)
        valid_response = valid_response.where(mask_da)
        valid_std = valid_std.where(mask_da)
        recharge_efficiency = recharge_efficiency.where(mask_da)

    uncertainty_relative_pct = xr.where(
        valid_response > 0,
        (valid_std / valid_response) * 100.0,
        np.nan,
    )
    uncertainty_relative_pct = uncertainty_relative_pct.assign_coords(
        lat=total_precip.lat, lon=total_precip.lon
    )
    uncertainty_relative_pct.name = 'uncertainty_relative_pct'

    return {
        'total_precip': total_precip,
        'valid_response': valid_response,
        'valid_std': valid_std,
        'recharge_efficiency': recharge_efficiency,
        'uncertainty_relative_pct': uncertainty_relative_pct,
    }


def summarize_arid_response_coverage(pixel_results, aoi_geometry):
    """
    Share of arid-AOI GRACE cells with a detectable valid GRACE response.

    Denominator = all GRACE grid cells inside the arid AOI polygon (rasterized),
    including cells with zero EPE / zero response. Numerator = cells with
    ``valid_response_sum > 0``.

    Parameters
    ----------
    pixel_results : dict
        Output of ``analyze_grace_response_by_pixel``.
    aoi_geometry : GeoSeries, GeoDataFrame, or shapely geometry
        Arid-region boundary used to mask the GRACE grid.

    Returns
    -------
    dict
        ``n_resp``, ``n_tot``, ``pct`` (rounded percent).
    """
    if aoi_geometry is None:
        raise ValueError("aoi_geometry is required for arid-boundary coverage.")

    arrays = _prepare_pixel_analysis_arrays(pixel_results, aoi_geometry)
    vr = np.asarray(arrays["valid_response"].values, dtype=float)
    n_tot = int(np.isfinite(vr).sum())
    n_resp = int(np.nansum(vr > 0))
    pct = int(round(100.0 * n_resp / n_tot)) if n_tot else 0
    _announce(
        f"{pct}% of arid GRACE pixels show a detectable response ({n_resp}/{n_tot})"
    )
    return {"n_resp": n_resp, "n_tot": n_tot, "pct": pct}


def _build_pixel_analysis_catalog(
    total_precip,
    valid_response,
    valid_std,
    uncertainty_relative_pct,
    *,
    recharge_efficiency=None,
):
    """Long-form per-pixel table (lat/lon + cumulative EPE, GRACE, uncertainty)."""
    tp, rs, su, urp = xr.align(
        total_precip,
        valid_response,
        valid_std,
        uncertainty_relative_pct,
        join='inner',
    )
    data_vars = {
        'cumulative_epe_cm': tp,
        'cumulative_grace_cm': rs,
        'cumulative_uncertainty_cm': su,
        'uncertainty_relative_pct': urp,
    }
    if recharge_efficiency is not None:
        re, _ = xr.align(recharge_efficiency, tp, join='inner')
        data_vars['recharge_efficiency_pct'] = re

    df = xr.Dataset(data_vars).to_dataframe().reset_index()
    main_cols = [
        'cumulative_epe_cm',
        'cumulative_grace_cm',
        'cumulative_uncertainty_cm',
    ]
    df = df.dropna(subset=main_cols, how='all').copy()
    if df.empty:
        _note(
            "pixel analysis catalog empty after dropping all-NaN rows "
            "(check pixel_results totals / AOI mask)"
        )
        return df

    lat_vals = np.asarray(tp.lat.values, dtype=float)
    lon_vals = np.asarray(tp.lon.values, dtype=float)

    def _nearest_index(values, x):
        return int(np.argmin(np.abs(values - float(x))))

    df['i'] = df['lat'].map(lambda v: _nearest_index(lat_vals, v))
    df['j'] = df['lon'].map(lambda v: _nearest_index(lon_vals, v))
    df['cell_id'] = df.apply(
        lambda row: f"P{int(row['i']):05d}_{int(row['j']):05d}",
        axis=1,
    )

    col_order = [
        'cell_id', 'i', 'j', 'lat', 'lon',
        'cumulative_epe_cm', 'cumulative_grace_cm', 'cumulative_uncertainty_cm',
        'uncertainty_relative_pct',
    ]
    if 'recharge_efficiency_pct' in df.columns:
        col_order.append('recharge_efficiency_pct')
    return df[[c for c in col_order if c in df.columns]]


# Lake/reservoir std-ratio overlay: white->blue, three classes (low = white).
_LAKE_STD_COLORS = ("#ffffff", "#6baed6", "#08306b")
# Fixed mid/high class boundaries (%); the lowest bound is the filter threshold.
_LAKE_STD_MID = 50.0
_LAKE_STD_HIGH = 100.0


def _lake_std_categories(min_pct=20.0):
    """Return (bounds, colors, labels) for the lake std-ratio classes.

    Three classes with fixed mid/high edges (50%, 100%); the lowest edge is the
    filter threshold ``min_pct``. Labels adapt to the threshold, e.g.
    ``min_pct=25`` -> ('25–50%', '50–100%', '>100%').
    """
    lo = float(min_pct)
    bounds = (lo, _LAKE_STD_MID, _LAKE_STD_HIGH)
    labels = (
        f"{lo:g}-{_LAKE_STD_MID:g}%",
        f"{_LAKE_STD_MID:g}-{_LAKE_STD_HIGH:g}%",
        f">{_LAKE_STD_HIGH:g}%",
    )
    return bounds, _LAKE_STD_COLORS, labels


def _prepare_lake_std_points(lake_points, value_col="std_pct_gr", min_pct=20.0):
    """
    Load and classify lake/reservoir std-ratio points for overlay.

    Points are reprojected to EPSG:4326, filtered to ``value_col > min_pct``
    (% of GRACE std), and binned into three classes: ``(min_pct, 50)`` -> 0,
    ``[50, 100)`` -> 1, ``>= 100`` -> 2. Points at or below ``min_pct`` or with
    non-finite values are dropped.

    Returns a GeoDataFrame with an added integer ``_lake_cat`` column, or
    ``None`` when ``lake_points`` is ``None``.
    """
    if lake_points is None:
        return None

    if isinstance(lake_points, (str, Path)):
        gdf = gpd.read_file(str(lake_points))
    elif isinstance(lake_points, gpd.GeoDataFrame):
        gdf = lake_points.copy()
    elif isinstance(lake_points, gpd.GeoSeries):
        gdf = gpd.GeoDataFrame(geometry=lake_points, crs=lake_points.crs)
    else:
        raise TypeError(
            "lake_points must be a shapefile path, GeoDataFrame, or GeoSeries."
        )

    if value_col not in gdf.columns:
        raise ValueError(
            f"lake_points is missing the value column {value_col!r}. "
            f"Available columns: {list(gdf.columns)}"
        )

    if gdf.crs is None:
        gdf = gdf.set_crs("EPSG:4326")
    gdf = gdf.to_crs("EPSG:4326")

    (lo, mid, hi), _, _ = _lake_std_categories(min_pct)
    vals = pd.to_numeric(gdf[value_col], errors="coerce").to_numpy(dtype=float)
    cat = np.full(len(gdf), -1, dtype=int)
    cat[(vals > lo) & (vals < mid)] = 0
    cat[(vals >= mid) & (vals < hi)] = 1
    cat[vals >= hi] = 2

    gdf = gdf.assign(_lake_cat=cat)
    gdf = gdf[gdf["_lake_cat"] >= 0]
    return gdf.reset_index(drop=True)


def build_pixel_analysis_catalog(
    pixel_results,
    aoi_geometry=None,
    *,
    require_positive_re: bool = False,
) -> pd.DataFrame:
    """
    Build a long-form per-pixel catalog from ``analyze_grace_response_by_pixel`` output.

    Columns include lat/lon, cumulative EPE/response/uncertainty, and
    ``recharge_efficiency_pct``. When ``require_positive_re=True``, keep only
    pixels with finite RE > 0 (same validity used for the RE map).
    """
    arrays = _prepare_pixel_analysis_arrays(pixel_results, aoi_geometry)
    catalog = _build_pixel_analysis_catalog(
        arrays["total_precip"],
        arrays["valid_response"],
        arrays["valid_std"],
        arrays["uncertainty_relative_pct"],
        recharge_efficiency=arrays["recharge_efficiency"],
    )
    if catalog.empty:
        _note(
            "build_pixel_analysis_catalog: no valid pixels "
            "(relax AOI / require_positive_re, or check analysis outputs)"
        )
        return catalog
    if require_positive_re and "recharge_efficiency_pct" in catalog.columns:
        re = pd.to_numeric(catalog["recharge_efficiency_pct"], errors="coerce")
        catalog = catalog[np.isfinite(re) & (re > 0)].copy()
        if catalog.empty:
            _note(
                "build_pixel_analysis_catalog: empty after require_positive_re "
                "(no pixels with finite recharge efficiency > 0)"
            )
    return catalog.reset_index(drop=True)


def join_lake_points_to_recharge_catalog(
    lake_points: Union[str, Path, gpd.GeoDataFrame],
    pixel_results=None,
    *,
    pixel_catalog: Optional[pd.DataFrame] = None,
    aoi_geometry=None,
    lake_value_col: str = "std_pct_gr",
    lake_min_pct: float = 20.0,
    coord_decimals: int = 1,
) -> Tuple[gpd.GeoDataFrame, Dict[str, int]]:
    """
    Join notebook-02 SWS lake/pixel points to valid RE analysis pixels.

    Matching uses rounded GRACE cell centers (``grace_lat``/``grace_lon`` or
    point geometry vs catalog ``lat``/``lon``). Only pixels with finite
    recharge efficiency > 0 are kept (Fig S11 / Table S3 population).

    Returns
    -------
    joined : GeoDataFrame
        Lake attributes plus RE fields.
    counts : dict
        ``n_sws``, ``n_on_re``, ``n_dropped``.
    """
    lakes = _prepare_lake_std_points(
        lake_points, value_col=lake_value_col, min_pct=lake_min_pct
    )
    if lakes is None or lakes.empty:
        raise ValueError("No lake points available after std-ratio filter")

    n_sws = int(len(lakes))
    if pixel_catalog is None:
        if pixel_results is None:
            raise ValueError("Provide pixel_results or pixel_catalog")
        pixel_catalog = build_pixel_analysis_catalog(
            pixel_results, aoi_geometry=aoi_geometry, require_positive_re=True
        )
    else:
        pixel_catalog = pixel_catalog.copy()
        if "recharge_efficiency_pct" in pixel_catalog.columns:
            re = pd.to_numeric(pixel_catalog["recharge_efficiency_pct"], errors="coerce")
            pixel_catalog = pixel_catalog[np.isfinite(re) & (re > 0)].copy()

    if pixel_catalog.empty:
        raise ValueError("RE pixel catalog is empty after validity filter")

    lakes = lakes.copy()
    if "grace_lat" in lakes.columns and "grace_lon" in lakes.columns:
        lakes["_join_lat"] = pd.to_numeric(lakes["grace_lat"], errors="coerce")
        lakes["_join_lon"] = pd.to_numeric(lakes["grace_lon"], errors="coerce")
    else:
        lakes["_join_lat"] = lakes.geometry.y
        lakes["_join_lon"] = lakes.geometry.x

    lakes["_k_lat"] = lakes["_join_lat"].round(coord_decimals)
    lakes["_k_lon"] = lakes["_join_lon"].round(coord_decimals)

    cat = pixel_catalog.copy()
    cat["_k_lat"] = pd.to_numeric(cat["lat"], errors="coerce").round(coord_decimals)
    cat["_k_lon"] = pd.to_numeric(cat["lon"], errors="coerce").round(coord_decimals)

    re_cols = [
        c
        for c in (
            "cumulative_epe_cm",
            "cumulative_grace_cm",
            "cumulative_uncertainty_cm",
            "uncertainty_relative_pct",
            "recharge_efficiency_pct",
            "cell_id",
        )
        if c in cat.columns
    ]
    cat_join = cat[["_k_lat", "_k_lon"] + re_cols].drop_duplicates(
        subset=["_k_lat", "_k_lon"], keep="first"
    )

    joined = lakes.merge(cat_join, on=["_k_lat", "_k_lon"], how="inner")
    n_on_re = int(len(joined))
    counts = {
        "n_sws": n_sws,
        "n_on_re": n_on_re,
        "n_dropped": int(n_sws - n_on_re),
    }
    _announce(
        f"Fig S11 / Table S3: {n_on_re} of {n_sws} SWS pixels on valid RE "
        f"(dropped {counts['n_dropped']})"
    )
    joined = joined.drop(
        columns=[c for c in ("_join_lat", "_join_lon", "_k_lat", "_k_lon") if c in joined.columns],
        errors="ignore",
    )
    return joined.reset_index(drop=True), counts


_TABLE_S3_RENAME = {
    "grace_lat": "GRACE pixel latitude (deg)",
    "grace_lon": "GRACE pixel longitude (deg)",
    "n_lakes": "Number of lakes in pixel",
    "lake_ids": "HydroLAKES IDs",
    "lake_names": "Lake names",
    "lake_id": "HydroLAKES ID",
    "lake_name": "Lake name",
    "country": "Country",
    "area_km2": "Total lake area (km2)",
    "completeness_pct": "Lake record completeness (%)",
    "complete": "Lake record completeness (%)",
    "std_pct_gr": "Lake std as % of GRACE std",
    "lake_std_pct_of_grace": "Lake std as % of GRACE std",
    "lake_std_cm": "Lake storage std (cm WE)",
    "grace_std_cm": "GRACE TWSA std (cm WE)",
    "std_cm_res": "Residual lake storage std (cm WE)",
    "gstd_cm_re": "Residual GRACE TWSA std (cm WE)",
    "std_pct_re": "Residual lake std as % of GRACE",
    "lake_std_pct_of_grace_residual": "Residual lake std as % of GRACE",
    "lake_std_cm_residual": "Residual lake storage std (cm WE)",
    "grace_std_cm_residual": "Residual GRACE TWSA std (cm WE)",
    "ltrend_cm": "Lake trend (cm/yr)",
    "gtrend_cm": "GRACE trend (cm/yr)",
    "lake_trend_cm_yr": "Lake trend (cm/yr)",
    "grace_trend_cm_yr": "GRACE trend (cm/yr)",
    "n_months": "Overlap months",
    "n_overlap_months": "Overlap months",
    "window_deg": "GRACE window (deg)",
    "grace_km2": "GRACE window area (km2)",
    "grace_window_area_km2": "GRACE window area (km2)",
    "dist_deg": "Distance to GRACE cell (deg)",
    "haversine_distance_deg": "Distance to GRACE cell (deg)",
    "cumulative_epe_cm": "Cumulative EPE precipitation (cm)",
    "cumulative_grace_cm": "Cumulative GRACE residual response (cm)",
    "cumulative_uncertainty_cm": "Cumulative uncertainty (cm)",
    "uncertainty_relative_pct": "Relative uncertainty (% of response)",
    "recharge_efficiency_pct": "Recharge efficiency (%)",
    "cell_id": "RE cell ID",
}

_TABLE_S3_DROP = {
    "geometry",
    "_lake_cat",
    "_ratio",
    "plot_lat",
    "plot_lon",
    "record_type",
    "grid_assignment",
    "lat",
    "lon",
    "i",
    "j",
}


def _excel_force_text_cell(value) -> str:
    """
    Format a CSV cell so Excel treats it as text (not a number).

    Comma-separated HydroLAKES IDs are otherwise parsed as one huge number and
    truncated past ~15 digits (trailing zeros). The ``="..."`` form displays the
    original string in Excel while remaining readable in text editors.
    """
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return ""
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if not text:
        return ""
    if text.startswith('="') and text.endswith('"'):
        return text
    return f'="{text.replace(chr(34), chr(34)+chr(34))}"'


def export_table_s3_lake_recharge(
    joined_gdf: Union[pd.DataFrame, gpd.GeoDataFrame],
    save_path: Union[str, Path],
) -> pd.DataFrame:
    """
    Clean and write Table S3 (SWS lakes on valid RE pixels) as UTF-8 CSV.

    Expects the GeoDataFrame/DataFrame from
    ``join_lake_points_to_recharge_catalog``.

    HydroLAKES ID columns are written Excel-safe (``="..."``) so comma-separated
    ID lists are not coerced to truncated numbers when opened in Excel.
    """
    df = pd.DataFrame(joined_gdf).copy()
    df = df.drop(columns=[c for c in _TABLE_S3_DROP if c in df.columns], errors="ignore")

    ordered = [c for c in _TABLE_S3_RENAME if c in df.columns]
    leftovers = [c for c in df.columns if c not in ordered]
    df = df[ordered + leftovers]
    df = df.rename(columns={k: v for k, v in _TABLE_S3_RENAME.items() if k in df.columns})

    for col in df.columns:
        if any(tok in col.lower() for tok in ("latitude", "longitude")):
            df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
        elif "area (km2)" in col.lower():
            df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
        elif any(
            tok in col.lower()
            for tok in ("std", "%", "trend", "completeness", "epe", "response", "uncertainty", "efficiency")
        ):
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().any():
                df[col] = num.round(2)

    # Force HydroLAKES ID fields to text for Excel (comma-separated lists).
    for col in df.columns:
        if "hydrolakes id" in col.lower():
            df[col] = df[col].map(_excel_force_text_cell)

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig helps Excel recognize UTF-8 (lake names); IDs use ="..." above.
    df.to_csv(save_path, index=False, encoding="utf-8-sig")
    _item(_rel(save_path), "ok")
    return df.reset_index(drop=True)


def plot_pixel_analysis_maps(
    pixel_results,
    aoi_geometry=None,
    figsize=(16, 12),
    ncols=2,
    cbar_orientation='vertical',
    cmaps=None,
    save_path=None,
    epe_max=600.0,
    grace_max=None,
    efficiency_max=15.0,
    uncertainty_display='absolute',
    uncertainty_pct_max=100.0,
    *,
    layout="full",
    efficiency_classify=False,
    efficiency_n_classes=5,
    return_pixel_catalog=False,
    save_raster=False,
    saved_rasters_path="outputs/rasters",
    raster_tags=None,
    lake_points=None,
    lake_value_col="std_pct_gr",
    lake_min_pct=20.0,
    lake_marker_size=None,
    lake_legend_title="Reservoir/Lake (%)",
):
    """
    Plot pixel-based analysis maps in a flexible grid.

    Panel order (when ``layout='full'``):

    1. Cumulative EPEs (``total_precip``)
    2. Cumulative GRACE GWS (``valid_response_sum``)
    3. Recharge efficiency (%) = (GRACE / precip) * 100
    4. Uncertainty (see ``uncertainty_display``)

    Parameters
    ----------
    layout : {'full', 'efficiency', 'precip_grace_uncertainty'}, default='full'
        - ``full``: all four panels (default, backward compatible).
        - ``efficiency``: only the recharge-efficiency map.
        - ``precip_grace_uncertainty``: EPE, GRACE response, and uncertainty only
          (no efficiency panel).

    pixel_results : dict
        Dictionary containing 'total_precip', 'valid_response_sum', 'valid_std_sum'
    aoi_geometry : GeoSeries, GeoDataFrame, or geometry, optional
        Area of interest geometry to mask pixels
    figsize : tuple, default=(16, 12)
        Figure size (width, height) in inches
    ncols : int, default=2
        Number of columns in the subplot grid (e.g., 1 for vertical stack,
        2 for 2x2 grid, 4 for single row).
    cbar_orientation : str, default='vertical'
        Colorbar orientation: 'vertical' or 'horizontal'.
    cmaps : list, optional
        Colormap names; length must match the number of panels for the chosen
        ``layout`` (1, 3, or 4). If None, sensible defaults are chosen per panel.
    save_path : str, optional
        Path to save the figure
    epe_max : float, default=600.0
        Maximum value (cap) for Cumulative EPEs colormap (in cm)
    grace_max : float, optional
        If set, cumulative GRACE GWS uses a fixed colormap ``vmin=0``, ``vmax=grace_max``
        (cm), matching the style of ``epe_max``. If ``None`` (default), limits follow the
        data-driven common scale (see ``uncertainty_display``). When
        ``uncertainty_display='absolute'``, the uncertainty panel still uses that common
        scale with ``valid_std``; ``grace_max`` only caps the GRACE panel, so the two panels
        may differ if ``grace_max`` is below the shared data maximum.
    efficiency_max : float, default=15.0
        Maximum value (cap) for Recharge Efficiency colormap (in %). Also used as the
        top of the in-range ladder when ``efficiency_classify=True``.
    efficiency_classify : bool, default=False
        If ``True`` (requires ``layout='efficiency'``), map recharge efficiency with
        discrete classes instead of a continuous colormap. ``efficiency_n_classes - 1``
        equal-width bins cover ``[0, efficiency_max]``; the last class is ``> efficiency_max``.
        Example: ``efficiency_max=15``, ``efficiency_n_classes=4`` → 0–5, 5–10, 10–15, >15.
    efficiency_n_classes : int, default=5
        Total number of classes when ``efficiency_classify=True`` (in-range + overflow).
        Must be ``>= 2``.
    uncertainty_display : str, default='absolute'
        Uncertainty panel: 'absolute' — valid_std_sum (cm), same color scale as GRACE
        when ``relative_pct`` is not used; 'relative_pct' — (valid_std / response) * 100.
    uncertainty_pct_max : float, default=100.0
        Colorbar cap (%) for uncertainty when uncertainty_display='relative_pct'.
    return_pixel_catalog : bool, default=False
        If ``True``, include ``pixel_catalog`` in ``pixel_counts`` (a per-pixel
        DataFrame). Default ``False`` avoids building a large table when only
        plotting; use :func:`build_pixel_analysis_pixel_catalog` instead.
    save_raster : bool, default=False
        If True, export each mapped panel as GeoTIFF under ``saved_rasters_path``.
    saved_rasters_path : str, default=.../tiff_files
        Output directory for GeoTIFF exports when ``save_raster=True``.
    raster_tags : list of str, optional
        Extra filename tokens appended to each exported raster stem.
    lake_points : str, GeoDataFrame, or GeoSeries, optional
        Point layer of lake/reservoir std ratio (percent of GRACE std). When
        provided, points are overlaid on every panel, colored white->blue by
        ``lake_value_col`` in three classes (20–50%, 50–100%, >100%; white is
        the lowest class). A clean legend is drawn on the lower left of the
        first panel.
    lake_value_col : str, default='std_pct_gr'
        Column in ``lake_points`` with the std-ratio percentage.
    lake_min_pct : float, default=20.0
        Only plot lakes with ``lake_value_col > lake_min_pct``. This value also
        becomes the lower edge of the first class, so the three classes are
        ``(lake_min_pct, 50)``, ``[50, 100)``, ``>= 100`` (mid/high edges fixed).
    lake_marker_size : float, optional
        Marker size (points^2) for the overlaid points. If ``None`` (default),
        it is auto-sized to approximately 1.5 grid cells (diameter) of the
        pixel data. Points are drawn with a red edge.
    lake_legend_title : str, default='Reservoir/Lake (%)'
        Title for the lake/reservoir overlay legend (rendered bold).

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes
    pixel_counts : dict
        Number of finite, plottable pixels per variable after AOI masking and ``> 0``
        filters: keys ``epe``, ``grace``, ``efficiency``, ``uncertainty``. Panels in the
        current ``layout`` use the matching entry; e.g. ``layout='efficiency'`` plots
        ``pixel_counts['efficiency']`` pixels. When ``efficiency_classify=True``, also
        includes ``efficiency_by_class`` (DataFrame with ``class``, ``label``,
        ``lower``, ``upper``, ``n_pixels``, ``pct`` per class).
        When ``return_pixel_catalog=True``, also includes ``pixel_catalog``
        (DataFrame): one row per AOI pixel with ``lat``, ``lon``,
        ``cumulative_epe_cm``, ``cumulative_grace_cm``,
        ``cumulative_uncertainty_cm``, and ``uncertainty_relative_pct`` (see
        ``build_pixel_analysis_pixel_catalog``).
        When ``save_raster=True``, also includes ``saved_rasters`` (list of
        exported GeoTIFF paths).
    """
    allowed_unc = ('absolute', 'relative_pct')
    if uncertainty_display not in allowed_unc:
        raise ValueError(f"uncertainty_display must be one of {allowed_unc}, got {uncertainty_display!r}")
    allowed_layout = ('full', 'efficiency', 'precip_grace_uncertainty')
    if layout not in allowed_layout:
        raise ValueError(f"layout must be one of {allowed_layout}, got {layout!r}")
    if grace_max is not None:
        gmx = float(grace_max)
        if not np.isfinite(gmx) or gmx <= 0.0:
            raise ValueError("grace_max must be a positive finite float when set")
    if efficiency_classify:
        if layout != 'efficiency':
            raise ValueError("efficiency_classify=True requires layout='efficiency'")
        nc = int(efficiency_n_classes)
        if nc < 2:
            raise ValueError("efficiency_n_classes must be >= 2 when efficiency_classify=True")
    arrays = _prepare_pixel_analysis_arrays(pixel_results, aoi_geometry)
    total_precip = arrays['total_precip']
    valid_response = arrays['valid_response']
    valid_std = arrays['valid_std']
    recharge_efficiency = arrays['recharge_efficiency']
    uncertainty_relative_pct = arrays['uncertainty_relative_pct']

    pixel_catalog = None
    if return_pixel_catalog:
        pixel_catalog = _build_pixel_analysis_catalog(
            total_precip,
            valid_response,
            valid_std,
            uncertainty_relative_pct,
            recharge_efficiency=recharge_efficiency,
        )

    # Only plot pixels with values > 0 (mask out zeros and negatives)
    total_precip = total_precip.where(total_precip > 0)
    valid_response = valid_response.where(valid_response > 0)
    valid_std = valid_std.where(valid_response > 0)
    recharge_efficiency = recharge_efficiency.where(recharge_efficiency > 0)
    uncertainty_relative_pct = uncertainty_relative_pct.where(valid_response > 0)
    
    if uncertainty_display == 'relative_pct':
        unc_title = "Uncertainty (% of response)"
        unc_data = uncertainty_relative_pct
    else:
        unc_title = "Uncertainty (cm)"
        unc_data = valid_std

    full_plot_specs = [
        {'kind': 'epe', 'data': total_precip, 'title': 'Cumulative EPEs (cm)'},
        {'kind': 'grace', 'data': valid_response, 'title': 'Cumulative GRACE GWS (cm)'},
        {'kind': 'efficiency', 'data': recharge_efficiency, 'title': 'Recharge Efficiency (%)'},
        {'kind': 'uncertainty', 'data': unc_data, 'title': unc_title},
    ]
    if layout == 'full':
        plot_specs = full_plot_specs
    elif layout == 'efficiency':
        plot_specs = [full_plot_specs[2]]
    else:
        plot_specs = [full_plot_specs[0], full_plot_specs[1], full_plot_specs[3]]

    def _count_finite_pixels(da):
        return int(np.isfinite(np.asarray(da.values, dtype=float)).sum())

    pixel_counts = {
        'epe': _count_finite_pixels(total_precip),
        'grace': _count_finite_pixels(valid_response),
        'efficiency': _count_finite_pixels(recharge_efficiency),
        'uncertainty': _count_finite_pixels(unc_data),
    }
    if return_pixel_catalog:
        pixel_counts['pixel_catalog'] = pixel_catalog
    _count_labels = {
        'epe': 'EPE',
        'grace': 'GRACE response',
        'efficiency': 'RE',
        'uncertainty': 'uncertainty',
    }
    plotted_kinds = [s['kind'] for s in plot_specs]
    count_bits = [
        f"{_count_labels[kind]}={pixel_counts[kind]}" for kind in plotted_kinds
    ]
    _announce(f"pixel maps: {'; '.join(count_bits)}")

    if efficiency_classify and layout == 'efficiency':
        class_df = _efficiency_class_counts(
            recharge_efficiency.values, efficiency_max, efficiency_n_classes
        )
        pixel_counts['efficiency_by_class'] = class_df
        print(
            class_df[["class", "label", "n_pixels", "pct"]]
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

    n_panels = len(plot_specs)
    # Adaptive grid layout based on ncols
    nrows = int(np.ceil(n_panels / ncols))
    
    # Subplot gaps aligned with plot_multiple_maps_with_balanced_colorbar (wspace=0, hspace=0.01)
    gs_kw = {'wspace': _MAP_SUBPLOT_WSPACE, 'hspace': _MAP_SUBPLOT_HSPACE}
    
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize,
                            subplot_kw={'projection': ccrs.PlateCarree()},
                            gridspec_kw=gs_kw)
    if n_panels == 1:
        axes = np.array([axes])
    axes = np.atleast_1d(axes).flatten()
    
    # Hide any unused axes when n_panels doesn't fill the grid
    for extra_idx in range(n_panels, len(axes)):
        axes[extra_idx].set_visible(False)
    
    minx = float(total_precip.lon.min())
    maxx = float(total_precip.lon.max())
    miny = float(total_precip.lat.min())
    maxy = float(total_precip.lat.max())
    extent = [minx - 1, maxx + 1, miny - 1, maxy + 1]
    
    # Fixed graticule: 40°S, 20°S, Equator, 20°N, 40°N; meridians 150°W…150°E every 30°
    _lat_graticule = np.array([-40.0, -20.0, 0.0, 20.0, 40.0], dtype=float)
    _lon_graticule = np.array(
        [-150.0, -120.0, -90.0, -60.0, -30.0, 0.0, 30.0, 60.0, 90.0, 120.0, 150.0],
        dtype=float,
    )
    margin_deg = 0.5
    lat_graticule = _lat_graticule[
        (_lat_graticule >= extent[2] - margin_deg) & (_lat_graticule <= extent[3] + margin_deg)
    ]
    lon_graticule = _lon_graticule[
        (_lon_graticule >= extent[0] - margin_deg) & (_lon_graticule <= extent[1] + margin_deg)
    ]
    if lat_graticule.size == 0:
        lat_graticule = _lat_graticule
    if lon_graticule.size == 0:
        # Longitudes may be 0–360°: match equivalent meridians
        lon_0360 = np.mod(_lon_graticule, 360.0)
        in_view = (lon_0360 >= extent[0] - margin_deg) & (lon_0360 <= extent[1] + margin_deg)
        if in_view.any():
            lon_graticule = lon_0360[in_view]
        else:
            lon_graticule = _lon_graticule
    
    default_cmaps_by_kind = {
        'epe': 'YlOrRd',
        'grace': 'Blues',
        'efficiency': 'RdYlGn',
        'uncertainty': 'Blues',
    }
    if cmaps is None:
        cmaps = [default_cmaps_by_kind[s['kind']] for s in plot_specs]
    elif len(cmaps) != n_panels:
        raise ValueError(
            f"cmaps must have length {n_panels} for layout={layout!r}, got {len(cmaps)}"
        )
    
    # Common scale for GRACE response and absolute uncertainty only (panel 1 & 4)
    valid_response_values = valid_response.values[~np.isnan(valid_response.values)]
    valid_std_values = valid_std.values[~np.isnan(valid_std.values)]
    
    if uncertainty_display == 'absolute':
        if len(valid_response_values) > 0 and len(valid_std_values) > 0:
            vmin_common = min(np.nanmin(valid_response_values), np.nanmin(valid_std_values))
            vmax_common = max(np.nanmax(valid_response_values), np.nanmax(valid_std_values))
            vmin_common = max(0, vmin_common) if vmin_common >= 0 else vmin_common
            vmax_common = max(vmax_common, 0)
        else:
            vmin_common = None
            vmax_common = None
    else:
        if len(valid_response_values) > 0:
            vmin_common = np.nanmin(valid_response_values)
            vmax_common = np.nanmax(valid_response_values)
            vmin_common = max(0, vmin_common) if vmin_common >= 0 else vmin_common
            vmax_common = max(vmax_common, 0)
        else:
            vmin_common = None
            vmax_common = None
    
    # Colorbar spacing aligned with plot_aridity_raster (pad=0.01, fraction=0.05)
    cbar_horizontal = cbar_orientation.lower() == 'horizontal'
    cbar_kw_base = {
        'orientation': cbar_orientation.lower(),
        'fraction': _MAP_CBAR_FRACTION,
        'pad': _MAP_CBAR_PAD,
        'shrink': _MAP_CBAR_SHRINK_H if cbar_horizontal else _MAP_CBAR_SHRINK_V,
    }
    
    # Prepare AOI overlay GeoDataFrame once
    aoi_gdf_plot = None
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf_plot = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf_plot = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf_plot = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")

    # Prepare lake/reservoir std-ratio overlay points once
    lake_points_gdf = _prepare_lake_std_points(
        lake_points, value_col=lake_value_col, min_pct=lake_min_pct
    )
    _, lake_colors, lake_labels = _lake_std_categories(lake_min_pct)

    for idx, (ax, spec, cmap) in enumerate(zip(axes[:n_panels], plot_specs, cmaps)):
        data = spec['data']
        title = spec['title']
        kind = spec['kind']
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor='lightgrey')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
        ax.coastlines()
        
        plot_vmin = None
        plot_vmax = None
        
        if kind == 'epe':
            plot_vmax = epe_max
            plot_vmin = 0
        elif kind == 'grace':
            if grace_max is not None:
                plot_vmin = 0
                plot_vmax = float(grace_max)
            elif vmin_common is not None and vmax_common is not None:
                plot_vmin = vmin_common
                plot_vmax = vmax_common
        elif kind == 'efficiency':
            plot_vmax = efficiency_max
            plot_vmin = 0
        elif kind == 'uncertainty':
            if uncertainty_display == 'relative_pct':
                plot_vmin = 0
                plot_vmax = uncertainty_pct_max
            elif vmin_common is not None and vmax_common is not None:
                plot_vmin = vmin_common
                plot_vmax = vmax_common
        
        cbar_kw = {**cbar_kw_base, 'label': title}

        use_efficiency_classify = (
            efficiency_classify and layout == 'efficiency' and kind == 'efficiency'
        )
        if use_efficiency_classify:
            # Discrete in-range bins [0, efficiency_max] with an extended
            # ">max" triangle; ticks sit at the bin boundaries (e.g. 0,5,10,15).
            emax = float(efficiency_max)
            inner_bounds = np.linspace(0.0, emax, int(efficiency_n_classes))
            n_in = len(inner_bounds) - 1
            cmap_full = plt.get_cmap(cmap, n_in + 1)
            seg_colors = [cmap_full(k) for k in range(n_in + 1)]
            listed_cmap = mcolors.ListedColormap(seg_colors[:n_in])
            listed_cmap.set_over(seg_colors[n_in])
            norm = mcolors.BoundaryNorm(inner_bounds, ncolors=n_in)
            quad = data.plot(
                ax=ax,
                transform=ccrs.PlateCarree(),
                cmap=listed_cmap,
                norm=norm,
                add_colorbar=False,
            )
            cbar = fig.colorbar(quad, ax=ax, extend='max', **cbar_kw)
            cbar.set_ticks(inner_bounds)
            cbar.set_ticklabels([f"{b:g}" for b in inner_bounds])
        else:
            plot_kwargs = dict(ax=ax, transform=ccrs.PlateCarree(), cmap=cmap,
                               add_colorbar=True, cbar_kwargs=cbar_kw)
            if plot_vmin is not None or plot_vmax is not None:
                plot_kwargs.update(vmin=plot_vmin, vmax=plot_vmax)
            data.plot(**plot_kwargs)

        if aoi_gdf_plot is not None:
            aoi_gdf_plot.plot(ax=ax, facecolor='none', edgecolor='black',
                            linewidth=0.2, transform=ccrs.PlateCarree())

        if lake_points_gdf is not None and len(lake_points_gdf):
            cats = lake_points_gdf["_lake_cat"].to_numpy()
            point_colors = [lake_colors[c] for c in cats]

            # Auto-size markers to ~1.5 grid cells (diameter) when not set explicitly
            if lake_marker_size is None:
                lon_vals = np.asarray(total_precip.lon.values, dtype=float)
                lat0 = float(np.asarray(total_precip.lat.values, dtype=float)[0])
                lon0 = float(lon_vals[0])
                cell_deg = float(abs(np.mean(np.diff(lon_vals)))) if lon_vals.size > 1 else 1.0
                p0 = ax.transData.transform((lon0, lat0))
                p1 = ax.transData.transform((lon0 + cell_deg, lat0))
                px = float(np.hypot(p1[0] - p0[0], p1[1] - p0[1]))
                diameter_pts = px * 72.0 / fig.dpi
                marker_s = max((1.5 * diameter_pts) ** 2, 4.0)
            else:
                marker_s = lake_marker_size

            ax.scatter(
                lake_points_gdf.geometry.x.values,
                lake_points_gdf.geometry.y.values,
                c=point_colors,
                s=marker_s,
                edgecolors="red",
                linewidths=0.7,
                transform=ccrs.PlateCarree(),
                zorder=20,
            )
            if idx == 0:
                from matplotlib.lines import Line2D

                present = sorted(set(int(c) for c in cats))
                legend_handles = [
                    Line2D(
                        [0], [0],
                        marker="o",
                        linestyle="none",
                        markerfacecolor=lake_colors[c],
                        markeredgecolor="red",
                        markeredgewidth=0.7,
                        markersize=8,
                        label=lake_labels[c],
                    )
                    for c in present
                ]
                lake_legend = ax.legend(
                    handles=legend_handles,
                    loc="lower left",
                    title=lake_legend_title,
                    fontsize=8,
                    title_fontsize=8,
                    framealpha=0.6,
                    facecolor="white",
                    edgecolor="0.7",
                    borderpad=0.6,
                    labelspacing=0.4,
                    handletextpad=0.5,
                )
                lake_legend.get_title().set_fontweight("bold")
                lake_legend.set_zorder(30)
                ax.add_artist(lake_legend)

        if layout != 'efficiency':
            panel_lbl = chr(ord("a") + idx)
            ax.text(
                0.5,
                0.95,
                panel_lbl,
                transform=ax.transAxes,
                ha="center",
                va="top",
                fontsize=12,
                fontweight="bold",
                zorder=25,
                bbox={
                    "boxstyle": "round,pad=0.32",
                    "facecolor": "white",
                    "edgecolor": "0.35",
                    "linewidth": 0.9,
                    "alpha": 0.94,
                },
            )
        
        # Gridlines — adaptive label visibility based on grid position
        row_idx = idx // ncols
        col_idx = idx % ncols
        last_row = (row_idx == nrows - 1) or (idx + ncols >= n_panels)
        first_col = (col_idx == 0)
        
        gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray',
                         alpha=0.5, linestyle='--')
        gl.top_labels = False
        gl.right_labels = False
        gl.bottom_labels = last_row
        gl.left_labels = first_col
        
        gl.ylocator = mticker.FixedLocator(lat_graticule)
        gl.xlocator = mticker.FixedLocator(lon_graticule)
        gl.xlabel_style = {'rotation': 0, 'size': _MAP_GRID_LABEL_FONTSIZE}
        gl.ylabel_style = {'rotation': 90, 'size': _MAP_GRID_LABEL_FONTSIZE}
    
    # Cartopy GeoAxes are not fully supported by tight_layout; spacing still OK with
    # savefig(bbox_inches='tight'). Filter the known UserWarning.
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches='tight')
        _item(_rel(save_path), "ok")

    if save_raster:
        from grace_analysis_utils import (
            _compose_raster_output_path,
            _export_geotiff_dataarray,
        )

        # Continuous panel values (not display class IDs). Resolve relative dirs
        # against the repo root so cwd=notebooks/ does not write notebooks/outputs/.
        raster_dir = Path(saved_rasters_path)
        if not raster_dir.is_absolute():
            raster_dir = Path(__file__).resolve().parent.parent / raster_dir

        raster_stems = {
            "epe": "pixel_cumulative_epe_cm",
            "grace": "pixel_cumulative_grace_cm",
            "efficiency": "pixel_recharge_efficiency_pct",
            "uncertainty": (
                "pixel_uncertainty_relative_pct"
                if uncertainty_display == "relative_pct"
                else "pixel_uncertainty_cm"
            ),
        }
        base_tags = list(raster_tags or [])
        saved_raster_paths = []
        for spec in plot_specs:
            kind = spec["kind"]
            stem = raster_stems[kind]
            out_path = _compose_raster_output_path(
                str(raster_dir), stem, base_tags
            )
            saved_raster_paths.append(
                _export_geotiff_dataarray(spec["data"], out_path)
            )
        pixel_counts["saved_rasters"] = saved_raster_paths
    
    plt.show()

    return fig, axes, pixel_counts


def _mask_pixel_results_by_geometry(pixel_results, geometry):
    """Return pixel_results dict masked to pixels inside ``geometry``."""
    tp = pixel_results["total_precip"]
    rs = pixel_results["valid_response_sum"]
    su = pixel_results["valid_std_sum"]
    tp, rs, su = xr.align(tp, rs, su, join="inner")

    if isinstance(geometry, gpd.GeoSeries):
        geom_gdf = gpd.GeoDataFrame(geometry=geometry, crs=geometry.crs).to_crs("EPSG:4326")
    elif isinstance(geometry, gpd.GeoDataFrame):
        geom_gdf = geometry.to_crs("EPSG:4326")
    else:
        geom_gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

    ref_data = tp.copy()
    ref_data.rio.write_crs("EPSG:4326", inplace=True)
    ref_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    transform = ref_data.rio.transform()
    mask_array = rasterize(
        [mapping(geom) for geom in geom_gdf.geometry],
        out_shape=(len(ref_data.lat), len(ref_data.lon)),
        transform=transform,
        fill=0,
        default_value=1,
        dtype=np.uint8,
    )
    mask_da = xr.DataArray(
        mask_array.astype(bool),
        coords={"lat": ref_data.lat, "lon": ref_data.lon},
        dims=["lat", "lon"],
    )
    return {
        "total_precip": tp.where(mask_da),
        "valid_response_sum": rs.where(mask_da),
        "valid_std_sum": su.where(mask_da),
    }


def _epe_values_to_cm(values, epe_input_unit="mm"):
    """Convert EPE totals to cm; accepts mm/cm aliases used by scatter helpers."""
    u = str(epe_input_unit).strip().lower()
    arr = np.asarray(values, dtype=float)
    if u in ("mm", "millimeter", "millimeters"):
        return arr * 0.1
    if u in ("cm", "centimeter", "centimeters"):
        return arr
    raise ValueError("epe_input_unit must be 'mm' or 'cm'")


def _pixel_epe_grace_arrays_from_results(
    pixel_results,
    *,
    epe_input_unit="mm",
    grace_threshold=0.0,
    min_cumulative_epe_cm=0.0,
    max_cumulative_epe_cm=None,
    max_cumulative_grace_cm=None,
    min_abs_response_cm=None,
    vmax_uncertainty_pct=None,
):
    """Filter pixel arrays for EPE–GRACE scatter; returns x, y, pct, vmax_c."""
    tp = pixel_results["total_precip"]
    rs = pixel_results["valid_response_sum"]
    su = pixel_results["valid_std_sum"]
    tp, rs, su = xr.align(tp, rs, su, join="inner")
    x_raw = tp.values.ravel().astype(float)
    y = rs.values.ravel().astype(float)
    s = su.values.ravel().astype(float)

    x = _epe_values_to_cm(x_raw, epe_input_unit)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    m &= x > float(min_cumulative_epe_cm)
    if max_cumulative_epe_cm is not None:
        m &= x <= float(max_cumulative_epe_cm)
    m &= y > float(grace_threshold)
    if min_abs_response_cm is not None:
        m &= np.abs(y) >= float(min_abs_response_cm)

    x, y, s = x[m], y[m], s[m]
    if max_cumulative_grace_cm is not None:
        keep_cap = y <= float(max_cumulative_grace_cm)
        x, y, s = x[keep_cap], y[keep_cap], s[keep_cap]
    eps = 1e-6
    denom = np.maximum(np.abs(y), eps)
    pct = 100.0 * s / denom

    if vmax_uncertainty_pct is not None:
        cap = float(vmax_uncertainty_pct)
        keep = pct <= cap
        x, y, s, pct = x[keep], y[keep], s[keep], pct[keep]
        vmax_c = cap
    else:
        vmax_c = float(np.nanmax(pct)) if len(pct) else 1.0
    if len(x) == 0:
        _note(
            "EPE–GRACE pixel arrays empty after filters "
            "(check grace_threshold / EPE caps / finite mask)"
        )
    return x, y, pct, vmax_c


def _mask_events_dataframe_by_geometry(events_df, geometry):
    """Return event rows whose (lon, lat) fall inside ``geometry``."""
    if events_df is None or len(events_df) == 0:
        return events_df
    if "lat" not in events_df.columns or "lon" not in events_df.columns:
        raise ValueError("events_dataframe must contain 'lat' and 'lon' columns")

    if isinstance(geometry, gpd.GeoSeries):
        geom_gdf = gpd.GeoDataFrame(geometry=geometry, crs=geometry.crs).to_crs("EPSG:4326")
    elif isinstance(geometry, gpd.GeoDataFrame):
        geom_gdf = geometry.to_crs("EPSG:4326")
    else:
        geom_gdf = gpd.GeoDataFrame(geometry=[geometry], crs="EPSG:4326")

    pts = gpd.GeoDataFrame(
        events_df.copy(),
        geometry=gpd.points_from_xy(events_df["lon"], events_df["lat"]),
        crs="EPSG:4326",
    )
    joined = gpd.sjoin(pts, geom_gdf[["geometry"]], how="inner", predicate="within")
    drop_cols = [c for c in ("index_right", "geometry") if c in joined.columns]
    return joined.drop(columns=drop_cols, errors="ignore")


def _pixel_epe_grace_arrays_from_events(
    events_df,
    *,
    epe_input_unit="mm",
    grace_threshold=0.0,
    min_cumulative_epe_cm=0.0,
    max_cumulative_epe_cm=None,
    max_cumulative_grace_cm=None,
    min_abs_response_cm=None,
    vmax_uncertainty_pct=None,
):
    """Filter valid event rows for EPE–GRACE scatter; returns x, y, pct, vmax_c."""
    if events_df is None or len(events_df) == 0:
        _note(
            "EPE–GRACE event arrays empty: no rows "
            "(empty events_dataframe or none marked is_valid)"
        )
        return np.array([]), np.array([]), np.array([]), 1.0

    x_raw = events_df["precip_sum"].values.astype(float)
    y = events_df["diff_mean"].values.astype(float)
    s = events_df["diff_std"].values.astype(float)

    x = _epe_values_to_cm(x_raw, epe_input_unit)

    m = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    m &= x > float(min_cumulative_epe_cm)
    if max_cumulative_epe_cm is not None:
        m &= x <= float(max_cumulative_epe_cm)
    m &= y > float(grace_threshold)
    if min_abs_response_cm is not None:
        m &= np.abs(y) >= float(min_abs_response_cm)

    x, y, s = x[m], y[m], s[m]
    if max_cumulative_grace_cm is not None:
        keep_cap = y <= float(max_cumulative_grace_cm)
        x, y, s = x[keep_cap], y[keep_cap], s[keep_cap]
    eps = 1e-6
    denom = np.maximum(np.abs(y), eps)
    pct = 100.0 * s / denom

    if vmax_uncertainty_pct is not None:
        cap = float(vmax_uncertainty_pct)
        keep = pct <= cap
        x, y, s, pct = x[keep], y[keep], s[keep], pct[keep]
        vmax_c = cap
    else:
        vmax_c = float(np.nanmax(pct)) if len(pct) else 1.0
    if len(x) == 0:
        _note(
            "EPE–GRACE event arrays empty after filters "
            "(check is_valid / grace_threshold / EPE caps)"
        )
    return x, y, pct, vmax_c


def _pixel_epe_grace_arrays(
    pixel_results,
    aggregation="pixel",
    geometry=None,
    **filter_kw,
):
    """Dispatch EPE–GRACE scatter arrays for pixel- or event-level aggregation."""
    agg = str(aggregation).strip().lower()
    if agg not in {"pixel", "event"}:
        raise ValueError("aggregation must be 'pixel' or 'event'")

    if agg == "pixel":
        pr = pixel_results
        if geometry is not None:
            pr = _mask_pixel_results_by_geometry(pixel_results, geometry)
        return _pixel_epe_grace_arrays_from_results(pr, **filter_kw)

    events_df = pixel_results.get("events_dataframe", pd.DataFrame())
    if events_df is None or len(events_df) == 0:
        _note(
            "EPE–GRACE event aggregation: events_dataframe empty or missing; "
            "returning empty arrays"
        )
        return np.array([]), np.array([]), np.array([]), 1.0
    if "is_valid" in events_df.columns:
        events_df = events_df.loc[events_df["is_valid"] == True].copy()
    if geometry is not None:
        events_df = _mask_events_dataframe_by_geometry(events_df, geometry)
    return _pixel_epe_grace_arrays_from_events(events_df, **filter_kw)


def _sanitize_domain_path_token(label):
    return "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))


def _uncertainty_pct_cbar_ticks(vmax_c):
    """Tick locations for uncertainty (%) colorbar from 0 to vmax."""
    vmax = float(vmax_c)
    if not np.isfinite(vmax) or vmax <= 0:
        return np.array([0.0])
    step = 5.0 if vmax <= 25.0 else 10.0
    return np.arange(0.0, vmax + step * 0.5, step)


def _domain_save_path(save_path, domain_label):
    """Return per-region save path with sanitized domain inserted before the extension."""
    if save_path is None:
        return None
    token = _sanitize_domain_path_token(domain_label)
    if "{domain}" in save_path:
        return save_path.format(domain=token)
    p = Path(save_path)
    return str(p.with_name(f"{p.stem}_{token}{p.suffix}"))


def plot_event_cluster_distribution_and_relationship(
    pixel_results,
    *,
    epe_input_unit="mm",
    bins=40,
    cmap="viridis_r",
    max_uncertainty_pct=None,
    vmax_uncertainty_pct=None,
    figsize=(11.0, 4.5),
    save_path=None,
    dpi=150,
):
    """
    Plot all finite event clusters from ``pixel_results['events_dataframe']``.

    Panel a shows the distribution of event-cluster GRACE response
    (``diff_mean``), including negative values. Panel b shows event-cluster
    precipitation (``precip_sum``) vs GRACE response, colored by relative
    uncertainty ``100 * diff_std / abs(diff_mean)``.

    Unlike ``plot_pixel_epe_grace_relationship(..., aggregation='event')``,
    this diagnostic does not require ``is_valid=True`` and does not apply a
    positive GRACE-response threshold. It keeps rows with finite precipitation,
    response, and uncertainty.

    Parameters
    ----------
    max_uncertainty_pct : float, optional
        Filter: remove points with uncertainty above this value before plotting
        and fitting. ``None`` keeps all points.
    vmax_uncertainty_pct : float, optional
        Colorbar cap: scale is 0–vmax. Points with uncertainty above the cap are
        still plotted (darkest color) and included in the fit; the colorbar uses
        ``extend='max'`` so values above the cap are visible as out-of-range.
        ``None`` auto-scales to data max (no extend).

    Returns
    -------
    fig, axes, summary_df
    """
    from matplotlib.colors import Normalize

    events_df = pixel_results.get("events_dataframe", pd.DataFrame())
    if events_df is None or len(events_df) == 0:
        raise ValueError("pixel_results['events_dataframe'] is empty or missing.")

    required = {"precip_sum", "diff_mean", "diff_std"}
    missing = sorted(required - set(events_df.columns))
    if missing:
        raise ValueError(f"events_dataframe missing required columns: {missing}")

    x_raw = events_df["precip_sum"].values.astype(float)
    y = events_df["diff_mean"].values.astype(float)
    s = events_df["diff_std"].values.astype(float)

    x = _epe_values_to_cm(x_raw, epe_input_unit)

    finite = np.isfinite(x) & np.isfinite(y) & np.isfinite(s)
    x, y, s = x[finite], y[finite], s[finite]
    if len(x) == 0:
        raise ValueError("No finite event clusters with precipitation, response, and uncertainty.")

    eps = 1e-6
    pct = 100.0 * s / np.maximum(np.abs(y), eps)
    if max_uncertainty_pct is not None:
        keep = pct <= float(max_uncertainty_pct)
        x, y, s, pct = x[keep], y[keep], s[keep], pct[keep]
    if len(x) == 0:
        raise ValueError("No event clusters remaining after uncertainty filter.")
    if vmax_uncertainty_pct is not None:
        vmax_c = float(vmax_uncertainty_pct)
    else:
        vmax_c = float(np.nanmax(pct)) if len(pct) else 1.0

    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    ax_hist, ax_scatter = axes

    def _panel_letter(ax, letter):
        ax.text(
            0.02,
            0.98,
            letter,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            fontweight="bold",
            zorder=10,
            bbox=dict(boxstyle="round", facecolor="white", edgecolor="black", alpha=0.9),
        )

    y_mean = float(np.nanmean(y))
    y_median = float(np.nanmedian(y))
    x_mean = float(np.nanmean(x))
    x_median = float(np.nanmedian(x))

    ax_hist.hist(y, bins=bins, color="#4c78a8", alpha=0.75, edgecolor="white")
    ax_hist.text(
        0.97,
        0.97,
        f"n = {len(y)}\nmean = {y_mean:.1f}\nmedian = {y_median:.1f}",
        transform=ax_hist.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )
    ax_hist.set_xlabel("GRACE response (cm)", fontsize=11)
    ax_hist.set_ylabel("Count", fontsize=11)
    ax_hist.grid(axis="y", alpha=0.3)
    _panel_letter(ax_hist, "a")

    # Cap color scale at vmax_c; clip=True maps pct > vmax to the end color so
    # high-uncertainty points stay visible (darkest). extend='max' marks ">vmax".
    capped_scale = vmax_uncertainty_pct is not None
    norm = Normalize(vmin=0.0, vmax=max(vmax_c, 1e-9), clip=True)
    sc = ax_scatter.scatter(
        x,
        y,
        c=pct,
        s=18,
        cmap=cmap,
        norm=norm,
        alpha=0.85,
        edgecolors="none",
    )
    ax_scatter.axhline(0.0, color="0.35", lw=0.8, ls="--", alpha=0.4)
    ax_scatter.set_xlim(0, float(np.nanmax(x)) * 1.05 if np.nanmax(x) > 0 else 1.0)
    y_abs = float(np.nanmax(np.abs(y)))
    ax_scatter.set_ylim(-1.05 * y_abs, 1.05 * y_abs) if y_abs > 0 else ax_scatter.set_ylim(-1, 1)

    slope, intercept, r_val, p_val, _ = linregress(x, y)
    x_fit = np.linspace(0.0, ax_scatter.get_xlim()[1], 200)
    ax_scatter.plot(x_fit, slope * x_fit + intercept, color="crimson", lw=1.8)

    def _p_txt(p):
        if p < 0.01:
            return "p < 0.01"
        return f"p = {p:.3f}"

    ax_scatter.text(
        0.97,
        0.97,
        (
            f"n = {len(x)}\n"
            f"slope = {slope:.2f}\n"
            f"r = {r_val:.2f}\n"
            f"{_p_txt(p_val)}"
        ),
        transform=ax_scatter.transAxes,
        ha="right",
        va="top",
        fontsize=11,
        bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
    )
    ax_scatter.set_xlabel("EPE (cm)", fontsize=11)
    ax_scatter.set_ylabel("GRACE response (cm)", fontsize=11)
    ax_scatter.grid(True, alpha=0.3)
    _panel_letter(ax_scatter, "b")

    cbar = fig.colorbar(
        sc,
        ax=ax_scatter,
        pad=0.02,
        extend="max" if capped_scale else "neither",
    )
    cbar.set_label("Uncertainty (%)")
    cbar.set_ticks(_uncertainty_pct_cbar_ticks(vmax_c))

    summary_df = pd.DataFrame(
        [
            {
                "n_events": int(len(x)),
                "grace_response_mean_cm": y_mean,
                "grace_response_median_cm": y_median,
                "grace_response_min_cm": float(np.nanmin(y)),
                "grace_response_max_cm": float(np.nanmax(y)),
                "epe_mean_cm": x_mean,
                "epe_median_cm": x_median,
                "uncertainty_pct_median": float(np.nanmedian(pct)),
                "slope_dimensionless": float(slope),
                "intercept_cm": float(intercept),
                "pearson_r": float(r_val),
                "r_squared": float(r_val ** 2),
                "p_value": float(p_val),
            }
        ]
    )

    fig.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    plt.show()
    return fig, axes, summary_df


def _draw_pixel_epe_grace_on_ax(
    ax,
    x,
    y,
    pct,
    vmax_c,
    *,
    model="linear",
    title=None,
    domain_label=None,
    domain_label_align="center",
    domain_label_fontsize=9,
    xlabel="Cumulative EPEs (cm)",
    ylabel="Cumulative GRACE Response (cm)",
    cmap="viridis_r",
    show_xlabel=True,
    show_ylabel=True,
    empty_msg="No points after filters",
):
    """Draw EPE–GRACE scatter on an existing axis. Returns (scatter_artist, stats_df)."""
    from matplotlib.colors import Normalize

    def _p_txt(p_val: float) -> str:
        if p_val < 0.01:
            return "p < 0.01"
        return f"p = {p_val:.3f}"

    def _draw_domain_label(axis):
        if domain_label is None:
            return
        align = str(domain_label_align).strip().lower()
        if align == "left":
            x_pos, ha = 0.03, "left"
        else:
            x_pos, ha = 0.5, "center"
        axis.text(
            x_pos,
            0.98,
            domain_label,
            transform=axis.transAxes,
            ha=ha,
            va="top",
            fontsize=float(domain_label_fontsize),
            fontweight="bold",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.9,
                "alpha": 0.94,
            },
        )

    if len(x) < 1:
        _note(
            f"{empty_msg} — check filters (grace_threshold, EPE caps, domain mask)"
        )
        ax.text(
            0.5,
            0.5,
            empty_msg,
            ha="center",
            va="center",
            transform=ax.transAxes,
        )
        _draw_domain_label(ax)
        if title is not None:
            ax.set_title(title, fontsize=11)
        ax.set_xlabel(xlabel if show_xlabel else "")
        ax.set_ylabel(ylabel if show_ylabel else "")
        ax.grid(True, alpha=0.35)
        return None, None

    norm = Normalize(vmin=0.0, vmax=max(vmax_c, 1e-9))
    sc = ax.scatter(
        x, y, c=pct, s=18, cmap=cmap, norm=norm, alpha=0.85, edgecolors="none"
    )

    x_max = float(np.nanmax(x))
    y_max = float(np.nanmax(y))
    x_hi = x_max * 1.5 if x_max > 0 else 1.0
    y_hi = y_max * 1.5 if y_max > 0 else 1.0
    ax.set_xlim(0, x_hi)
    ax.set_ylim(0, y_hi)

    _draw_domain_label(ax)

    mdl = str(model).strip().lower()
    if mdl not in {"linear", "power", "compare"}:
        raise ValueError("model must be one of {'linear','power','compare'}")

    stats_df = None
    if len(x) >= 2:
        if mdl == "compare":
            cmp = compare_models(x, y)
            lin = cmp["linear"]
            powm = cmp["power"]
            print("Model comparison (common subset x>0,y>0):")
            print(
                f"  linear: R2={lin['r2']:.3f}, RMSE={lin['rmse']:.3g}, AIC={lin['aic']:.2f}"
            )
            print(
                f"  power : R2={powm['r2']:.3f}, RMSE={powm['rmse']:.3g}, AIC={powm['aic']:.2f}"
            )
            print(f"  preferred (AIC): {cmp['preferred']}")

            xs = np.linspace(0, x_hi, 200)
            ax.plot(
                xs,
                lin["predict"](xs),
                color="crimson",
                lw=1.8,
                label=f"Linear fit (AIC={lin['aic']:.1f})",
            )
            xs_p = xs[xs > 0]
            ax.plot(
                xs_p,
                powm["predict"](xs_p),
                color="#1f77b4",
                lw=1.8,
                label=f"Power fit (AIC={powm['aic']:.1f})",
            )
            txt = (
                f"Preferred: {cmp['preferred']}\n"
                f"r_lin = {lin['r']:.2f}\n"
                f"r_pow = {powm['r']:.2f}"
            )
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
            ax.text(
                0.97,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
            )
            stats_df = pd.DataFrame(
                [
                    {
                        "n_pixels": len(x),
                        "linear_slope": lin["params"]["a"],
                        "linear_intercept": lin["params"]["b"],
                        "linear_r": lin["r"],
                        "linear_p": lin["p"],
                        "linear_r2": lin["r2"],
                        "linear_aic": lin["aic"],
                        "power_a": powm["params"]["a"],
                        "power_b": powm["params"]["b"],
                        "power_r_log": powm["r"],
                        "power_p": powm["p"],
                        "power_r2": powm["r2"],
                        "power_aic": powm["aic"],
                        "preferred": cmp["preferred"],
                    }
                ]
            )

        elif mdl == "power":
            powm = fit_model(x, y, model="power")
            xs = np.linspace(0, x_hi, 200)
            xs_p = xs[xs > 0]
            ax.plot(
                xs_p,
                powm["predict"](xs_p),
                color="#1f77b4",
                lw=1.8,
                label="Power-law fit",
            )
            a_hat = powm["params"]["a"]
            b_hat = powm["params"]["b"]
            txt = (
                f"n = {powm['n']}\n"
                f"y = {a_hat:.2f} x^{b_hat:.2f}\n"
                f"r(log) = {powm['r']:.2f}\n"
                f"{_p_txt(powm['p'])}"
            )
            ax.legend(loc="upper left", fontsize=9, framealpha=0.95)
            ax.text(
                0.97,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
            )
            x_pp = x[(x > 0) & (y > 0)]
            y_pp = y[(x > 0) & (y > 0)]
            if len(x_pp) >= 2:
                yhat_pp = powm["predict"](x_pp)
                rss = float(np.sum((y_pp - yhat_pp) ** 2))
                tss = float(np.sum((y_pp - np.mean(y_pp)) ** 2))
                power_r2 = 1.0 - (rss / tss) if tss > 0 else np.nan
            else:
                power_r2 = np.nan
            stats_df = pd.DataFrame(
                [
                    {
                        "n_pixels": len(x),
                        "power_a": a_hat,
                        "power_b": b_hat,
                        "power_r_log": powm["r"],
                        "power_r2": power_r2,
                        "power_p": powm["p"],
                    }
                ]
            )

        else:
            lin = fit_model(x, y, model="linear")
            slope = lin["params"]["a"]
            intercept = lin["params"]["b"]
            r_val = lin["r"]
            p_val = lin["p"]
            xs = np.linspace(0, x_hi, 200)
            ax.plot(xs, lin["predict"](xs), color="crimson", lw=1.8)
            slope_str = f"{slope:.2f}"
            txt = (
                f"n = {len(x)}\n"
                f"slope = {slope_str}\n"
                f"r = {r_val:.2f}\n"
                f"{_p_txt(p_val)}"
            )
            ax.text(
                0.97,
                0.97,
                txt,
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=9,
                bbox=dict(boxstyle="round", facecolor="white", edgecolor="0.7", alpha=0.95),
            )
            stats_df = pd.DataFrame(
                [
                    {
                        "n_pixels": len(x),
                        "slope_dimensionless": slope,
                        "intercept_cm": intercept,
                        "pearson_r": r_val,
                        "r_squared": float(r_val ** 2),
                        "p_value": p_val,
                    }
                ]
            )

    ax.set_xlabel(xlabel if show_xlabel else "")
    ax.set_ylabel(ylabel if show_ylabel else "")
    if title is not None:
        ax.set_title(title, fontsize=11)
    ax.grid(True, alpha=0.35)
    return sc, stats_df


def _pixel_epe_grace_single_figure(
    x,
    y,
    pct,
    vmax_c,
    *,
    model="linear",
    title=None,
    domain_label=None,
    xlabel="Cumulative EPEs (cm)",
    ylabel="Cumulative GRACE Response (cm)",
    cbar_label="Uncertainty (%)",
    figsize=(7.0, 6.0),
    cmap="viridis_r",
    dpi=150,
    save_path=None,
):
    """Draw one EPE–GRACE scatter (+ optional residual panel). Returns fig, ax, stats_df."""
    from matplotlib.colors import Normalize

    mdl = str(model).strip().lower()
    want_residuals = mdl in {"power", "compare"}

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    sc, stats_df = _draw_pixel_epe_grace_on_ax(
        ax,
        x,
        y,
        pct,
        vmax_c,
        model=model,
        title=title,
        domain_label=domain_label,
        xlabel=xlabel,
        ylabel=ylabel,
        cmap=cmap,
        show_xlabel=True,
        show_ylabel=True,
    )

    if sc is not None:
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
        cbar.set_ticks(_uncertainty_pct_cbar_ticks(vmax_c))

    if want_residuals and len(x) >= 2 and sc is not None:
        fig.clf()
        gs = fig.add_gridspec(2, 1, height_ratios=[3.2, 1.2], hspace=0.08)
        ax = fig.add_subplot(gs[0, 0])
        ax_res = fig.add_subplot(gs[1, 0], sharex=ax)
        sc, stats_df = _draw_pixel_epe_grace_on_ax(
            ax,
            x,
            y,
            pct,
            vmax_c,
            model=model,
            title=title,
            domain_label=domain_label,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            show_xlabel=False,
            show_ylabel=True,
        )
        cbar = fig.colorbar(sc, ax=ax, pad=0.02)
        cbar.set_label(cbar_label)
        cbar.set_ticks(_uncertainty_pct_cbar_ticks(vmax_c))
        ax_res.set_xlabel(xlabel)

        if mdl == "compare":
            cmp = compare_models(x, y)
            powm = cmp["power"]
            mp = (x > 0) & (y > 0)
            x_p = x[mp]
            y_p = y[mp]
            yhat_p = powm["predict"](x_p)
            res = y_p - yhat_p
            ax_res.axhline(0.0, color="0.35", lw=1.0)
            ax_res.scatter(x_p, res, s=10, alpha=0.55, color="#1f77b4", edgecolors="none")
            ax_res.set_ylabel("Residual", fontsize=9)
            ax_res.grid(True, alpha=0.25)
            e2 = res**2
            X = np.column_stack([np.ones_like(x_p), x_p])
            beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
            e2_hat = X @ beta
            rss = float(np.sum((e2 - e2_hat) ** 2))
            tss = float(np.sum((e2 - np.mean(e2)) ** 2))
            r2_aux = 1.0 - (rss / tss) if tss > 0 else np.nan
            lm = float(len(e2) * r2_aux) if np.isfinite(r2_aux) else np.nan
            p_bp = float(chi2.sf(lm, df=1)) if np.isfinite(lm) else np.nan
            print(f"Breusch–Pagan (power residuals): LM={lm:.3f}, p={p_bp:.3g}")

        elif mdl == "power":
            powm = fit_model(x, y, model="power")
            mp = (x > 0) & (y > 0)
            x_p = x[mp]
            y_p = y[mp]
            yhat_p = powm["predict"](x_p)
            res = y_p - yhat_p
            ax_res.axhline(0.0, color="0.35", lw=1.0)
            ax_res.scatter(x_p, res, s=10, alpha=0.55, color="#1f77b4", edgecolors="none")
            ax_res.set_ylabel("Residual", fontsize=9)
            ax_res.grid(True, alpha=0.25)
            e2 = res**2
            X = np.column_stack([np.ones_like(x_p), x_p])
            beta, *_ = np.linalg.lstsq(X, e2, rcond=None)
            e2_hat = X @ beta
            rss = float(np.sum((e2 - e2_hat) ** 2))
            tss = float(np.sum((e2 - np.mean(e2)) ** 2))
            r2_aux = 1.0 - (rss / tss) if tss > 0 else np.nan
            lm = float(len(e2) * r2_aux) if np.isfinite(r2_aux) else np.nan
            p_bp = float(chi2.sf(lm, df=1)) if np.isfinite(lm) else np.nan
            print(f"Breusch–Pagan (power residuals): LM={lm:.3f}, p={p_bp:.3g}")

    plt.tight_layout()
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
    return fig, ax, stats_df


def _pixel_epe_grace_collage_figure(
    region_data,
    *,
    collage_ncols,
    vmax_c,
    model="linear",
    xlabel="Cumulative EPEs (cm)",
    ylabel="Cumulative GRACE Response (cm)",
    cbar_label="Uncertainty (%)",
    figsize=(7.0, 6.0),
    cmap="viridis_r",
    dpi=150,
    save_path=None,
):
    """Build multi-region collage with one uncertainty colorbar per row."""
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    n_regions = len(region_data)
    ncols = int(collage_ncols)
    if ncols < 1:
        raise ValueError("collage_ncols must be >= 1")
    nrows = int(np.ceil(n_regions / ncols))

    fw = figsize[0] * ncols * 1.04
    fh = figsize[1] * nrows
    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    gs = GridSpec(
        nrows,
        ncols,
        figure=fig,
        wspace=0.14,
        hspace=0.12,
    )

    axes = np.empty((nrows, ncols), dtype=object)
    norm = Normalize(vmin=0.0, vmax=max(vmax_c, 1e-9))
    stats_rows = []

    for i, item in enumerate(region_data):
        row = i // ncols
        col = i % ncols
        ax = fig.add_subplot(gs[row, col])
        axes[row, col] = ax

        show_ylabel = col == 0
        show_xlabel = row == nrows - 1

        sc, stats_df = _draw_pixel_epe_grace_on_ax(
            ax,
            item["x"],
            item["y"],
            item["pct"],
            vmax_c,
            model=model,
            domain_label=item["domain_label"],
            domain_label_align="left",
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            show_xlabel=show_xlabel,
            show_ylabel=show_ylabel,
        )
        if stats_df is not None:
            row_df = stats_df.copy()
            row_df.insert(0, "domain", item["domain_label"])
            stats_rows.append(row_df)

    for row in range(nrows):
        for col in range(ncols):
            idx = row * ncols + col
            if idx >= n_regions and axes[row, col] is not None:
                axes[row, col].set_visible(False)

    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    for row in range(nrows):
        last_col = -1
        for col in range(ncols):
            if row * ncols + col < n_regions:
                last_col = col
        if last_col < 0:
            continue
        right_ax = axes[row, last_col]
        divider = make_axes_locatable(right_ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.04)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label)
        cbar.set_ticks(_uncertainty_pct_cbar_ticks(vmax_c))

    fig.tight_layout(pad=0.12, w_pad=0.08, h_pad=0.04)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)

    stats_df = pd.concat(stats_rows, ignore_index=True) if stats_rows else None
    return fig, axes, stats_df




def plot_pixel_epe_grace_relationship(
    pixel_results,
    *,
    model="linear",
    aggregation="pixel",
    domain="all",
    domain_gdf=None,
    domain_name_col="Domain",
    collage=False,
    collage_ncols=2,
    grace_threshold=0.0,
    min_cumulative_epe_cm=0.0,
    max_cumulative_epe_cm=None,
    max_cumulative_grace_cm=None,
    min_abs_response_cm=None,
    title=None,
    xlabel="Cumulative EPEs (cm)",
    ylabel="Cumulative GRACE Response (cm)",
    cbar_label="Uncertainty (%)",
    epe_input_unit="mm",
    figsize=(7.0, 6.0),
    cmap="viridis_r",
    vmax_uncertainty_pct=None,
    save_path=None,
    dpi=150,
):
    """
    Scatter EPE vs GRACE from ``analyze_grace_response_by_pixel``.

    ``aggregation`` : {'pixel', 'event'}
        - ``'pixel'`` (default): one point per grid cell using ``total_precip``,
          ``valid_response_sum``, and ``valid_std_sum`` (cumulative per pixel).
        - ``'event'``: one point per valid cluster in ``events_dataframe``
          (``precip_sum``, ``diff_mean``, ``diff_std``; ``is_valid=True`` only).
          With default ``xlabel``/``ylabel``, axes use ``EPE (cm)`` and
          ``GRACE Response (cm)`` (not the cumulative pixel labels).

    EPE totals are usually stored in **mm**; with ``epe_input_unit='mm'`` (default)
    they are multiplied by 0.1 so the x-axis is **cm**, consistent with GRACE (cm).

    Uncertainty (color) is ``100 * std / max(|y|, eps)`` (inter-solution spread
    relative to response magnitude). Filter kwargs named ``min_cumulative_*`` apply
    to per-pixel totals or per-event magnitudes depending on ``aggregation``.

    In pixel mode, ``total_precip`` sums precip from all clusters at a pixel;
    ``valid_response_sum`` sums only valid GRACE responses. Event mode uses valid
    clusters only for both axes.

    ``domain`` : {'all', 'regions', 'regions_and_all'}
        - ``'all'``: one scatter using all pixels in ``pixel_results``.
        - ``'regions'``: one panel per polygon in ``domain_gdf``.
        - ``'regions_and_all'``: prepends a **Global** panel (all pixels, same
          as ``'all'``) then each region; Global is labeled top-left in collage mode.

    ``collage`` : bool
        If ``True`` (requires ``domain='regions'`` or ``'regions_and_all'``),
        arrange panels in one figure grid with ``collage_ncols`` columns, one
        uncertainty colorbar per row on the right, y-label on the left column
        only, x-label on the bottom row only. Only ``model='linear'`` is supported.

    Returns
    -------
    fig, ax, stats_df or figs, axes, stats_df
        Single figure when ``domain='all'`` or ``collage=True``; list of figures
        when ``domain`` is ``'regions'`` or ``'regions_and_all'`` and
        ``collage=False``.
    """
    dom = str(domain).strip().lower()
    if dom not in {"all", "regions", "regions_and_all"}:
        raise ValueError("domain must be 'all', 'regions', or 'regions_and_all'")
    agg = str(aggregation).strip().lower()
    if agg not in {"pixel", "event"}:
        raise ValueError("aggregation must be 'pixel' or 'event'")
    if collage and dom not in {"regions", "regions_and_all"}:
        raise ValueError("collage=True requires domain='regions' or 'regions_and_all'")
    if collage and str(model).strip().lower() != "linear":
        raise ValueError("collage=True requires model='linear'")

    if agg == "event":
        if xlabel == "Cumulative EPEs (cm)":
            xlabel = "EPE (cm)"
        if ylabel == "Cumulative GRACE Response (cm)":
            ylabel = "GRACE Response (cm)"

    filter_kw = dict(
        epe_input_unit=epe_input_unit,
        grace_threshold=grace_threshold,
        min_cumulative_epe_cm=min_cumulative_epe_cm,
        max_cumulative_epe_cm=max_cumulative_epe_cm,
        max_cumulative_grace_cm=max_cumulative_grace_cm,
        min_abs_response_cm=min_abs_response_cm,
        vmax_uncertainty_pct=vmax_uncertainty_pct,
    )
    draw_kw = dict(
        model=model,
        title=title,
        xlabel=xlabel,
        ylabel=ylabel,
        cbar_label=cbar_label,
        figsize=figsize,
        cmap=cmap,
        dpi=dpi,
    )

    if dom == "all":
        x, y, pct, vmax_c = _pixel_epe_grace_arrays(
            pixel_results, aggregation=agg, geometry=None, **filter_kw
        )
        fig, ax, stats_df = _pixel_epe_grace_single_figure(
            x, y, pct, vmax_c, save_path=save_path, **draw_kw
        )
        plt.show()
        return fig, ax, stats_df

    if dom in {"regions", "regions_and_all"} and domain_gdf is None:
        raise ValueError("domain_gdf is required when domain='regions' or 'regions_and_all'")

    region_data = []
    if dom == "regions_and_all":
        x, y, pct, vmax_c = _pixel_epe_grace_arrays(
            pixel_results, aggregation=agg, geometry=None, **filter_kw
        )
        region_data.append(
            {
                "domain_label": "Global",
                "x": x,
                "y": y,
                "pct": pct,
                "vmax_c": vmax_c,
            }
        )
    for _, row in domain_gdf.iterrows():
        domain_label = row[domain_name_col]
        x, y, pct, vmax_c = _pixel_epe_grace_arrays(
            pixel_results,
            aggregation=agg,
            geometry=row.geometry,
            **filter_kw,
        )
        region_data.append(
            {
                "domain_label": domain_label,
                "x": x,
                "y": y,
                "pct": pct,
                "vmax_c": vmax_c,
            }
        )

    if collage:
        if vmax_uncertainty_pct is not None:
            shared_vmax = float(vmax_uncertainty_pct)
        else:
            pcts = [d["vmax_c"] for d in region_data if len(d["pct"])]
            shared_vmax = float(max(pcts)) if pcts else 1.0

        collage_kw = {k: v for k, v in draw_kw.items() if k != "title"}
        fig, axes, stats_df = _pixel_epe_grace_collage_figure(
            region_data,
            collage_ncols=collage_ncols,
            vmax_c=shared_vmax,
            save_path=save_path,
            **collage_kw,
        )
        plt.show()
        return fig, axes, stats_df

    figs = []
    axes_list = []
    stats_rows = []
    for item in region_data:
        region_path = _domain_save_path(save_path, item["domain_label"])
        fig, ax, stats_df = _pixel_epe_grace_single_figure(
            item["x"],
            item["y"],
            item["pct"],
            item["vmax_c"],
            domain_label=item["domain_label"],
            save_path=region_path,
            **draw_kw,
        )
        figs.append(fig)
        axes_list.append(ax)
        if stats_df is not None:
            row_df = stats_df.copy()
            row_df.insert(0, "domain", item["domain_label"])
            stats_rows.append(row_df)
        plt.show()

    stats_df = pd.concat(stats_rows, ignore_index=True) if stats_rows else None
    return figs, axes_list, stats_df


def plot_epe_grace_agg_uncertainty_collage(
    pixel_results,
    *,
    aggregations=("pixel", "event", "pixel", "event"),
    vmax_uncertainty_pcts=(50, 50, 20, 20),
    panel_labels=("a", "b", "c", "d"),
    grace_threshold=1.5,
    min_cumulative_epe_cm=0.0,
    max_cumulative_epe_cm=None,
    max_cumulative_grace_cm=None,
    min_abs_response_cm=None,
    epe_input_unit="cm",
    model="linear",
    cmap="viridis_r",
    cbar_label="Uncertainty (%)",
    panel_figsize=(7.0, 6.0),
    dpi=150,
    save_path=None,
):
    """
    Fig S8-style 2x2 collage: aggregation (pixel/event) x uncertainty cap.

    Default panel order (row-major), matching the manuscript collage:

    - **a** ``pixel``, vmax=50
    - **b** ``event``, vmax=50
    - **c** ``pixel``, vmax=20
    - **d** ``event``, vmax=20

    Each row has its own uncertainty colorbar (scale = that row's vmax).
    ``vmax`` also filters points to ``uncertainty_pct <= vmax``, same as
    ``plot_pixel_epe_grace_relationship``.

    Returns
    -------
    fig, axes, stats_df
        ``axes`` is a (2, 2) object array; ``stats_df`` has columns
        ``panel``, ``aggregation``, ``vmax_uncertainty_pct``, plus fit stats.
    """
    from matplotlib.colors import Normalize
    from matplotlib.gridspec import GridSpec
    from matplotlib.cm import ScalarMappable
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    aggs = tuple(aggregations)
    vmxs = tuple(float(v) for v in vmax_uncertainty_pcts)
    labels = tuple(panel_labels)
    n = len(aggs)
    if not (n == len(vmxs) == len(labels) == 4):
        raise ValueError(
            "aggregations, vmax_uncertainty_pcts, and panel_labels must each have length 4"
        )
    if str(model).strip().lower() != "linear":
        raise ValueError("plot_epe_grace_agg_uncertainty_collage requires model='linear'")

    nrows, ncols = 2, 2
    fw = float(panel_figsize[0]) * ncols * 1.04
    fh = float(panel_figsize[1]) * nrows
    fig = plt.figure(figsize=(fw, fh), dpi=dpi)
    gs = GridSpec(nrows, ncols, figure=fig, wspace=0.16, hspace=0.18)

    axes = np.empty((nrows, ncols), dtype=object)
    stats_rows = []
    filter_base = dict(
        epe_input_unit=epe_input_unit,
        grace_threshold=grace_threshold,
        min_cumulative_epe_cm=min_cumulative_epe_cm,
        max_cumulative_epe_cm=max_cumulative_epe_cm,
        max_cumulative_grace_cm=max_cumulative_grace_cm,
        min_abs_response_cm=min_abs_response_cm,
    )

    for i, (agg, vmax_c, letter) in enumerate(zip(aggs, vmxs, labels)):
        row, col = divmod(i, ncols)
        ax = fig.add_subplot(gs[row, col])
        axes[row, col] = ax

        agg_l = str(agg).strip().lower()
        if agg_l == "event":
            xlabel, ylabel = "EPE (cm)", "GRACE Response (cm)"
        elif agg_l == "pixel":
            xlabel, ylabel = "Cumulative EPEs (cm)", "Cumulative GRACE Response (cm)"
        else:
            raise ValueError("aggregation must be 'pixel' or 'event'")

        x, y, pct, vmax_used = _pixel_epe_grace_arrays(
            pixel_results,
            aggregation=agg_l,
            geometry=None,
            vmax_uncertainty_pct=vmax_c,
            **filter_base,
        )

        sc, stats_df = _draw_pixel_epe_grace_on_ax(
            ax,
            x,
            y,
            pct,
            vmax_used,
            model=model,
            domain_label=str(letter),
            domain_label_align="center",
            domain_label_fontsize=14,
            xlabel=xlabel,
            ylabel=ylabel,
            cmap=cmap,
            show_xlabel=(row == nrows - 1),
            show_ylabel=True,
        )
        if stats_df is not None:
            row_df = stats_df.copy()
            row_df.insert(0, "panel", str(letter))
            row_df.insert(1, "aggregation", agg_l)
            row_df.insert(2, "vmax_uncertainty_pct", float(vmax_c))
            stats_rows.append(row_df)

    # One colorbar per row; scale matches that row's vmax (panels share vmax within a row).
    for row in range(nrows):
        vmax_row = float(vmxs[row * ncols])
        right_ax = axes[row, ncols - 1]
        sm = ScalarMappable(
            norm=Normalize(vmin=0.0, vmax=max(vmax_row, 1e-9)),
            cmap=cmap,
        )
        sm.set_array([])
        divider = make_axes_locatable(right_ax)
        cax = divider.append_axes("right", size="3.5%", pad=0.04)
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(cbar_label)
        cbar.set_ticks(_uncertainty_pct_cbar_ticks(vmax_row))

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
            category=UserWarning,
        )
        fig.tight_layout(pad=0.12, w_pad=0.10, h_pad=0.08)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, bbox_inches="tight", dpi=dpi)
        _item(_rel(save_path), "ok")

    plt.show()
    stats_out = pd.concat(stats_rows, ignore_index=True) if stats_rows else None
    return fig, axes, stats_out


def _diag_closest_nice_bin_width(w_raw):
    """Pick closest 'nice' bin width to w_raw from (1, 2, 2.5, 4, 5, 10) × 10^k."""
    w_raw = float(w_raw)
    if not np.isfinite(w_raw) or w_raw <= 0.0:
        return 1.0
    exp = int(np.floor(np.log10(w_raw)))
    candidates = [m * (10.0 ** exp) for m in (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)]
    exp1 = exp + 1
    candidates += [m * (10.0 ** exp1) for m in (1.0, 2.0, 2.5, 4.0, 5.0)]
    return min(candidates, key=lambda w: abs(np.log(w_raw / w)))


def _diag_next_coarser_nice_width(w):
    """Smallest nice bin width strictly greater than ``w``."""
    w = float(w)
    if not np.isfinite(w) or w <= 0.0:
        return 1.0
    exp = int(np.floor(np.log10(w)))
    m = w / (10.0 ** exp)
    ladder = (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)
    for m2 in ladder:
        if m2 > m * (1.0 + 1e-12):
            return m2 * (10.0 ** exp)
    return ladder[0] * (10.0 ** (exp + 1))


def _diag_prev_finer_nice_width(w):
    """Largest nice bin width strictly less than ``w``."""
    w = float(w)
    if not np.isfinite(w) or w <= 0.0:
        return w * 0.5
    exp = int(np.floor(np.log10(w)))
    m = w / (10.0 ** exp)
    ladder = (1.0, 2.0, 2.5, 4.0, 5.0, 10.0)
    for m2 in reversed(ladder):
        if m2 < m * (1.0 - 1e-12):
            return m2 * (10.0 ** exp)
    return 5.0 * (10.0 ** (exp - 1))


def _diag_nbins_for_width(vmin, vmax, w):
    w = float(w)
    if w <= 0.0 or not np.isfinite(w):
        return 10**9
    start = np.floor(float(vmin) / w) * w
    n_intervals = int(np.ceil((float(vmax) - start) / w))
    return max(1, n_intervals)


def _diag_edges_from_width(vmin, vmax, w):
    w = float(w)
    vmin, vmax = float(vmin), float(vmax)
    start = np.floor(vmin / w) * w
    n_intervals = int(np.ceil((vmax - start) / w))
    if n_intervals < 1:
        n_intervals = 1
    edges = start + np.arange(n_intervals + 1, dtype=float) * w
    if edges[-1] < vmax - 1e-9 * max(1.0, abs(vmax)):
        n_intervals += 1
        edges = start + np.arange(n_intervals + 1, dtype=float) * w
    return edges


def _diag_target_nbins(span):
    """Prefer ~5 bins over short spans (e.g. 0–20); more bins for long spans."""
    span = float(span)
    if not np.isfinite(span) or span <= 0.0:
        return 20
    if span <= 30.0:
        return max(6, min(10, int(round(span / 4.0))))
    if span <= 100.0:
        return max(12, min(24, int(round(span / 5.0))))
    return int(np.clip(round(span / 12.0), 30, 80))


def _diag_histogram_edges_auto_nice(data, max_bins=250, min_bins=6):
    """
    Deterministic histogram edges: nice bin width w from (1,2,2.5,4,5)×10^k,
    aligned to multiples of w. Chooses w so bin count is near a span-based target
    and stays within [min_bins, max_bins] when possible (otherwise respects max_bins).
    """
    data = np.asarray(data, dtype=float).ravel()
    data = data[np.isfinite(data)]
    if data.size == 0:
        return np.array([0.0, 1.0], dtype=float)
    vmin = float(np.min(data))
    vmax = float(np.max(data))
    if vmin == vmax:
        return np.array([vmin, vmin + 1.0], dtype=float)

    span = vmax - vmin
    nb_target = _diag_target_nbins(span)
    w_raw = span / float(max(nb_target, 1))
    w = _diag_closest_nice_bin_width(w_raw)

    for _ in range(64):
        nb = _diag_nbins_for_width(vmin, vmax, w)
        if nb > max_bins:
            w2 = _diag_next_coarser_nice_width(w)
            if w2 <= w * (1.0 + 1e-12):
                break
            w = w2
            continue
        if nb < min_bins:
            w2 = _diag_prev_finer_nice_width(w)
            if w2 >= w * (1.0 - 1e-12):
                break
            w = w2
            continue
        return _diag_edges_from_width(vmin, vmax, w)

    return _diag_edges_from_width(vmin, vmax, w)


def plot_pixel_results_distribution_diagnostics(
    pixel_results,
    *,
    quantiles=(0.99,),
    uncertainty_mode="relative_pct",
    figsize=(10.0, 9.0),
    bins="auto",
    save_path=None,
    dpi=120,
    min_grace_response_cm=None,
):
    """
    Diagnostic 2×2 histograms of pixel-level summaries from ``pixel_results``.

    Quantiles are probabilities in (0, 1); e.g. ``0.99`` is the 99th percentile.

    Panels match ``plot_pixel_analysis_maps`` definitions, with **cumulative EPE**
    assumed to be stored in **cm** in ``total_precip`` (native units used as-is).

    - **Cumulative EPE**: ``total_precip`` (**cm**).
    - **GRACE response**: ``valid_response_sum`` (**cm**).
    - **Uncertainty**: ``relative_pct`` — ``(valid_std / valid_response) * 100`` where
      response > 0 (same as ``plot_pixel_analysis_maps``; ``std == 0`` yields 0%);
      or ``absolute`` — ``valid_std_sum`` (**cm**).
    - **Efficiency**: ``(valid_response / total_precip) * 100`` where ``total_precip > 0``
      with both GRACE and EPE in **cm** (same formula as maps when inputs share that convention).

    Parameters
    ----------
    pixel_results : dict
        Must contain ``total_precip``, ``valid_response_sum``, ``valid_std_sum``.
    quantiles : iterable of float
        Values strictly between 0 and 1; vertical lines and labels on each subplot.
    uncertainty_mode : {'relative_pct', 'absolute'}
    min_grace_response_cm : float, optional
        If set (e.g. ``1.5``), pixels with non-finite ``valid_response_sum`` or cumulative
        GRACE response **less than or equal to** this value (cm) are set to NaN in precip,
        GRACE, and std before all four histograms. Pixels kept therefore satisfy
        ``valid_response_sum > min_grace_response_cm``, consistent with
        ``grace_threshold`` in ``analyze_grace_response_by_pixel`` (per-stack validity uses
        ensemble mean **>** ``grace_threshold``). Default ``None`` keeps all finite values
        (legacy behavior).
    figsize : tuple
    bins : int, str, or array-like
        If an ``int``, passed to ``hist`` as the bin count (numpy/matplotlib behavior).
        If ``\"auto\"`` (case-insensitive), bin edges are deterministic: a target bin count
        from the data span picks an initial width, rounded to the nearest ``(1, 2, 2.5, 4, 5)
        × 10^k`` cm (or % for efficiency), then coarsened or refined so the count stays within
        a modest range (default about 6–250 bins), keeping short spans at roughly five bins
        across ~20 units when the span allows.
        Otherwise (e.g. explicit edge sequence), passed through to ``hist``.
    save_path : str, optional
    dpi : int

    Returns
    -------
    fig : matplotlib.figure.Figure
    axes : ndarray of Axes, shape (2, 2)
    summary : pandas.DataFrame
        Columns ``variable``, ``quantile``, ``value`` (quantile values rounded to nearest integer).

    Notes
    -----
    The number of finite values, mean, and median for each panel are printed to stdout
    as one summary table (after plotting). With ``min_grace_response_cm`` set, EPE/GRACE/
    efficiency/uncertainty share the same response filter, so their ``n`` match
    ``plot_pixel_analysis_maps`` GRACE/uncertainty counts for that threshold (maps without
    the threshold still show a larger EPE ``n``).
    """
    from matplotlib.lines import Line2D

    tp = pixel_results["total_precip"]
    rs = pixel_results["valid_response_sum"]
    su = pixel_results["valid_std_sum"]
    tp, rs, su = xr.align(tp, rs, su, join="inner")

    qs = [float(q) for q in quantiles]
    for q in qs:
        if not (0.0 < q < 1.0):
            raise ValueError(f"quantiles must be in (0, 1), got {q!r}")

    umode = str(uncertainty_mode).strip().lower()
    if umode not in {"relative_pct", "absolute"}:
        raise ValueError("uncertainty_mode must be 'relative_pct' or 'absolute'")

    precip_cm = tp.values.ravel().astype(float)
    grace_cm = rs.values.ravel().astype(float)
    std_cm = su.values.ravel().astype(float)

    if min_grace_response_cm is not None:
        t = float(min_grace_response_cm)
        if not np.isfinite(t) or t < 0.0:
            raise ValueError("min_grace_response_cm must be a non-negative finite float")
        bad = ~np.isfinite(grace_cm) | (grace_cm <= t)
        nan = np.nan
        precip_cm = np.where(bad, nan, precip_cm)
        grace_cm = np.where(bad, nan, grace_cm)
        std_cm = np.where(bad, nan, std_cm)

    if umode == "relative_pct":
        # Match plot_pixel_analysis_maps / _prepare_pixel_analysis_arrays:
        # (std / response) * 100 wherever response > 0 (std == 0 -> 0%).
        unc = np.full_like(grace_cm, np.nan, dtype=float)
        m_u = (grace_cm > 0) & np.isfinite(grace_cm) & np.isfinite(std_cm)
        unc[m_u] = (std_cm[m_u] / grace_cm[m_u]) * 100.0
        unc_title = "Uncertainty (% of response)"
        unc_xlabel = "Uncertainty (%)"
    else:
        unc = std_cm.copy()
        unc_title = "Uncertainty (inter-solution spread)"
        unc_xlabel = "Uncertainty (cm)"

    eff = np.full_like(grace_cm, np.nan, dtype=float)
    m_e = (precip_cm > 0) & np.isfinite(precip_cm) & np.isfinite(grace_cm)
    eff[m_e] = (grace_cm[m_e] / precip_cm[m_e]) * 100.0

    panels = [
        (
            precip_cm,
            "Cumulative EPE",
            "Cumulative EPE (cm)",
            "Count",
        ),
        (
            grace_cm,
            "GRACE response",
            "Cumulative GRACE response (cm)",
            "Count",
        ),
        (
            unc,
            unc_title,
            unc_xlabel,
            "Count",
        ),
        (
            eff,
            "Recharge efficiency",
            "Recharge Efficiency (%)",
            "Count",
        ),
    ]

    quantile_colors = ["#d62728", "#ff7f0e", "#2ca02c", "#9467bd", "#8c564b"]

    def _finite_hist_samples(arr):
        a = np.asarray(arr, dtype=float).ravel()
        return a[np.isfinite(a)]

    use_auto_bins = isinstance(bins, str) and str(bins).strip().lower() == "auto"

    fig, axes = plt.subplots(2, 2, figsize=figsize, dpi=dpi, constrained_layout=True)
    axes_flat = axes.ravel()
    summary_rows = []
    stats_rows = []

    for i, (ax, (data_raw, title_short, xlabel, ylabel)) in enumerate(zip(axes_flat, panels)):
        data = _finite_hist_samples(data_raw)
        if data.size == 0:
            mean_v = np.nan
            median_v = np.nan
        else:
            mean_v = float(np.mean(data))
            median_v = float(np.median(data))
        stats_rows.append(
            {
                "variable": title_short,
                "n": int(data.size),
                "mean": mean_v,
                "median": median_v,
            }
        )
        ax.set_xlabel(xlabel, fontsize=10)
        if i % 2 == 0:
            ax.set_ylabel(ylabel, fontsize=10)
        else:
            ax.set_ylabel("")
        ax.grid(True, alpha=0.3, linestyle="-")

        panel_lbl = chr(ord("a") + i)
        ax.text(
            0.5,
            0.95,
            panel_lbl,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=12,
            fontweight="bold",
            zorder=10,
            bbox={
                "boxstyle": "round,pad=0.32",
                "facecolor": "white",
                "edgecolor": "0.35",
                "linewidth": 0.9,
                "alpha": 0.94,
            },
        )

        if data.size == 0:
            ax.text(
                0.5,
                0.5,
                "No finite values",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=10,
                color="0.45",
            )
            continue

        hist_bins = _diag_histogram_edges_auto_nice(data) if use_auto_bins else bins
        n, edges, patches = ax.hist(
            data,
            bins=hist_bins,
            color="#4c72b0",
            edgecolor="white",
            linewidth=0.6,
            alpha=0.88,
        )

        legend_handles = []
        for k, q in enumerate(qs):
            v = float(np.quantile(data, q))
            pname = int(round(100 * q))
            var_key = title_short
            summary_rows.append(
                {"variable": var_key, "quantile": q, "value": int(round(v)) if np.isfinite(v) else np.nan}
            )
            c = quantile_colors[k % len(quantile_colors)]
            ax.axvline(v, color=c, linestyle="--", linewidth=1.6, zorder=4)
            val_str = (
                str(int(round(v)))
                if np.isfinite(v)
                else "nan"
            )
            legend_handles.append(
                Line2D(
                    [0],
                    [0],
                    color=c,
                    linestyle="--",
                    linewidth=1.6,
                    label=f"{pname}th pct: {val_str}",
                )
            )

        ax.legend(
            handles=legend_handles,
            loc="upper right",
            fontsize=11,
            framealpha=0.92,
            handlelength=2.2,
        )

    summary = pd.DataFrame(summary_rows)
    stats_df = pd.DataFrame(stats_rows)
    _fmt = lambda v: "nan" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.6g}"
    print("plot_pixel_results_distribution_diagnostics: distribution summary:")
    print(
        stats_df.to_string(
            index=False,
            formatters={"mean": _fmt, "median": _fmt},
        )
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

    return fig, axes, summary


