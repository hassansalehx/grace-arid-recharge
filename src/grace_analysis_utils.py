"""
GRACE data analysis utilities for arid-region notebooks.

Notebook-facing surface (see ``__all__``): load/process GRACE and predictors,
TWSA-CPA correlation maps, and shared plotting helpers used by notebook 03.

Residual note:
  TWSA-CPA correlation uses calendar-locked ``_decompose_grace_calendar``.
  Pixel EPE/RE analysis in ``grace_analysis_pixel`` currently uses index-based
  ``decompose_grace_sin_cosin`` after gap years are dropped. Treat those
  residual products as related but not identical.
"""

import numpy as np
from scipy.stats import linregress, spearmanr, pearsonr
import xarray as xr
import geopandas as gpd
import rioxarray
import rasterio
import pandas as pd
from cftime import num2date
import os
import gc
from pathlib import Path

# Plotting
import matplotlib.pyplot as plt
import cartopy.feature as cfeature
from matplotlib import cm
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap, BoundaryNorm
import dask.array as da
from matplotlib.ticker import FormatStrFormatter
import cartopy.crs as ccrs
import matplotlib.dates as mdates
import matplotlib.ticker as mticker
from IPython.display import display  # Used in display_comparison_table

__all__ = [
    "format_pvalue",
    "process_grace_data",
    "process_predictor_fine",
    "plot_aridity_raster",
    "calculate_grace_precip_correlation_per_pixel",
    "plot_grace_correlation_map",
    "summarize_grace_correlation_outputs",
    "plot_grace_precip_correlation_interactive_map",
    "decompose_grace_sin_cosin",
]


from status_io import (  # noqa: E402
    announce as _announce,
    item as _item,
    note as _note,
    raise_ctx as _raise_ctx,
    rel as _rel,
    summarize_skipped as _summarize_skipped,
)


def format_pvalue(p_val):
    """
    Format p-value for display in plots.
    
    Parameters:
    -----------
    p_val : float
        P-value to format
        
    Returns:
    --------
    str
        Formatted p-value string:
        - "< 0.01" if p < 0.01
        - "< 0.05" if 0.01 <= p < 0.05
        - Actual value if p >= 0.05
    """
    if pd.isna(p_val) or np.isnan(p_val):
        return "N/A"
    if p_val < 0.01:
        return "< 0.01"
    elif p_val < 0.05:
        return "< 0.05"
    else:
        return f"{p_val:.2f}"
# OPTIMIZED: Removed unused imports (cupy, folium, plotly, branca, json, base64, BytesIO, PIL)
# These were not used in the actual code

from shapely.geometry import mapping
from tqdm import tqdm
import statsmodels.api as sm

# Map style contract (use these for any new Cartopy map so figures match):
# - gridspec: wspace=_MAP_SUBPLOT_WSPACE, hspace=_MAP_SUBPLOT_HSPACE
# - vertical colorbar: pad=_MAP_CBAR_PAD, fraction=_MAP_CBAR_FRACTION, shrink=_MAP_CBAR_SHRINK_V, extend as needed
# - plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)
# - grid label fontsize: _MAP_GRID_LABEL_FONTSIZE on both gl.xlabel_style and gl.ylabel_style
# - AOI outline: linewidth=_MAP_AOI_LINEWIDTH (pixel maps in grace_analysis_pixel may use thinner lines)
# See plot_aridity_raster, plot_pixel_analysis_maps (grace_analysis_pixel), analyze_lsm_outputs.
_MAP_SUBPLOT_WSPACE = 0.0
_MAP_SUBPLOT_HSPACE = 0.02
_MAP_CBAR_PAD = 0.01
_MAP_CBAR_FRACTION = 0.05
_MAP_CBAR_SHRINK_V = 0.95
_MAP_CBAR_SHRINK_H = 0.95
_MAP_TIGHT_LAYOUT_PAD = 0.2
# Cartopy gridline tick labels: set both x and y or lon/lat can render at different default sizes
_MAP_GRID_LABEL_FONTSIZE = 9
# AOI outline on PlateCarree maps (match across plot_aridity_raster, trend/correlation, analyze_lsm_outputs)
_MAP_AOI_LINEWIDTH = 0.5
# Aridity map AOI overlay (plot_aridity_raster only; other maps keep _MAP_AOI_LINEWIDTH)
_ARIDITY_AOI_LINEWIDTH = 1.0
# Colorblind-friendly boundary hues for viridis aridity backgrounds (Paul Tol–style)
_ARIDITY_DOMAIN_BOUNDARY_COLORS = (
    "#CC3311",
    "#0077BB",
    "#EE7733",
    "#33BBEE",
    "#EE3377",
    "#228833",
    "#AA3377",
    "#BBBB44",
)


def _aridity_domain_boundary_color_map(labels):
    """Map sorted unique labels to stable boundary colors."""
    unique = sorted({str(l) for l in labels})
    cmap = {}
    palette = _ARIDITY_DOMAIN_BOUNDARY_COLORS
    tab = plt.get_cmap("tab10")
    for i, lab in enumerate(unique):
        if i < len(palette):
            cmap[lab] = palette[i]
        else:
            cmap[lab] = tab((i - len(palette)) % 10)
    return cmap


def _aoi_geometry_to_gdf_plot(aoi_geometry):
    """Normalize AOI input to GeoDataFrame in EPSG:4326 for boundary plotting."""
    if isinstance(aoi_geometry, gpd.GeoSeries):
        return gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs(
            "EPSG:4326"
        )
    if isinstance(aoi_geometry, gpd.GeoDataFrame):
        return aoi_geometry.to_crs("EPSG:4326")
    return gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")


def _aridity_aoi_boundary_color_map(aoi_geometry, boundary_col=None):
    """Return stable label -> edge color dict for AOI features."""
    gdf = _aoi_geometry_to_gdf_plot(aoi_geometry)
    name_col = None
    if boundary_col is not None and boundary_col in gdf.columns:
        name_col = boundary_col
    elif "Domain" in gdf.columns:
        name_col = "Domain"

    labels = []
    for i, (_, row) in enumerate(gdf.iterrows()):
        if name_col is not None:
            labels.append(str(row[name_col]))
        else:
            labels.append(f"Feature_{i}")
    return _aridity_domain_boundary_color_map(labels)


def _plot_aridity_aoi_boundaries(
    ax,
    aoi_geometry,
    *,
    boundary_col=None,
    color_map=None,
    linewidth=_ARIDITY_AOI_LINEWIDTH,
):
    """Draw per-feature AOI boundaries with distinct colors (no legend)."""
    gdf = _aoi_geometry_to_gdf_plot(aoi_geometry)
    if color_map is None:
        color_map = _aridity_aoi_boundary_color_map(aoi_geometry, boundary_col)

    name_col = None
    if boundary_col is not None and boundary_col in gdf.columns:
        name_col = boundary_col
    elif "Domain" in gdf.columns:
        name_col = "Domain"

    for i, (_, row) in enumerate(gdf.iterrows()):
        if name_col is not None:
            lab = str(row[name_col])
        else:
            lab = f"Feature_{i}"
        gpd.GeoSeries([row.geometry], crs=gdf.crs).plot(
            ax=ax,
            facecolor="none",
            edgecolor=color_map[lab],
            linewidth=linewidth,
            transform=ccrs.PlateCarree(),
        )
    return color_map



def _ensure_aridity_display_raster(path, max_pixels=1_000_000):
    """
    Return a downsampled GeoTIFF suitable for quick map display.

    The source Zomer AI GeoTIFF has no overviews (~485 MB / ~900M cells), so a
    windowed ``out_shape`` read still scans the full window (~10s on WSL). Build a
    ~max_pixels display cache once under ``data/interim/aridity/`` and reuse it.
    """
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.transform import Affine

    path = Path(path)
    here = Path(__file__).resolve().parent
    repo = here.parent if here.name == "src" else Path.cwd()
    cache_dir = repo / "data" / "interim" / "aridity"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{path.stem}_display.tif"
    if cache.is_file() and cache.stat().st_size > 0:
        return str(cache)

    _note(f"building aridity display cache (one-time, ~max_pixels={max_pixels:,})")
    with rasterio.open(path) as src:
        total = src.height * src.width
        if max_pixels and total > max_pixels:
            scale = np.sqrt(total / float(max_pixels))
            out_h = max(1, int(round(src.height / scale)))
            out_w = max(1, int(round(src.width / scale)))
        else:
            out_h, out_w = src.height, src.width
        data = src.read(1, out_shape=(out_h, out_w), resampling=Resampling.nearest)
        transform = src.transform * Affine.scale(src.width / out_w, src.height / out_h)
        profile = src.profile.copy()
        profile.update(
            height=out_h,
            width=out_w,
            transform=transform,
            compress="lzw",
            tiled=True,
            blockxsize=256,
            blockysize=256,
            count=1,
            dtype=data.dtype,
        )
        profile.pop("overviews", None)
        with rasterio.open(cache, "w", **profile) as dst:
            dst.write(data, 1)
    _item(_rel(cache), "ok")
    return str(cache)


def plot_aridity_raster(
    path,
    cmap='viridis',
    label="Aridity Index",
    vmin=0.0,
    vmax=1.0,
    dpi=300,
    figsize=(12, 6),
    save_path=None,
    aoi_geometry=None,
    clip_aoi=True,
    max_pixels=1_000_000,
    color_boundaries=False,
    aoi_boundary_col=None,
):
    """
    Plot aridity raster map from Zomer dataset.
    
    Parameters:
    -----------
    path : str
        Path to aridity raster file
    cmap : str, default='viridis'
        Colormap name
    label : str, default="Aridity Index"
        Colorbar label
    vmin : float, default=0.0
        Minimum value for colormap
    vmax : float, default=1.0
        Maximum value for colormap
    dpi : int, default=300
        Resolution for saved figure
    save_path : str, optional
        Path to save the figure
    aoi_geometry : GeoSeries, GeoDataFrame, or geometry, optional
        Area of interest geometry to clip the aridity data and overlay boundaries
    clip_aoi : bool, default True
        If True, clip raster to AOI. If False, full raster is shown (AOI outline only);
        the raster is downsampled to avoid loading a huge array into memory.
    max_pixels : int, default 1_000_000
        Target pixel budget for the displayed array. A one-time display GeoTIFF
        (~max_pixels) is written under ``data/interim/aridity/`` and reused on
        later calls. Without that cache, a windowed read of the full-res source
        (no overviews) is ~10s+ on WSL.
    color_boundaries : bool, default False
        If False, draw a single black outline for the full AOI. If True, draw each
        feature with a distinct color (use GeoDataFrame + ``aoi_boundary_col`` for names).
    aoi_boundary_col : str, optional
        Column in ``aoi_geometry`` for feature labels when ``color_boundaries=True``
        (assigns colors per feature; no legend is drawn).
        Defaults to ``'Domain'`` if that column exists.
    """
    # The Zomer AI GeoTIFF is ~1 km global (~485 MB / ~900M cells) with no overviews.
    # Use a one-time display cache (~max_pixels), then window + polygon-mask.
    path = _ensure_aridity_display_raster(path, max_pixels=max_pixels)
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.windows import from_bounds as _window_from_bounds
    from rasterio.transform import array_bounds as _array_bounds
    from rasterio import features as _rio_features

    aoi_gdf = None
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs)
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.copy()
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")

    with rasterio.open(path) as src:
        src_crs = src.crs or "EPSG:4326"
        if aoi_gdf is not None:
            if aoi_gdf.crs is None:
                aoi_gdf = aoi_gdf.set_crs(src_crs)
            else:
                aoi_gdf = aoi_gdf.to_crs(src_crs)
            minx, miny, maxx, maxy = aoi_gdf.total_bounds
            pad = 0.05
            minx, miny, maxx, maxy = minx - pad, miny - pad, maxx + pad, maxy + pad
            window = _window_from_bounds(minx, miny, maxx, maxy, transform=src.transform)
            window = window.round_offsets().round_lengths()
            win_h = max(int(window.height), 1)
            win_w = max(int(window.width), 1)
            total = win_h * win_w
            if max_pixels and total > max_pixels:
                scale = np.sqrt(total / float(max_pixels))
                out_h = max(1, int(round(win_h / scale)))
                out_w = max(1, int(round(win_w / scale)))
            else:
                out_h, out_w = win_h, win_w
            data = src.read(
                1,
                window=window,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
                boundless=True,
                fill_value=0 if src.nodata is None else src.nodata,
            ).astype(float)
            transform = src.window_transform(window) * src.transform.scale(
                (win_w / out_w) if out_w else 1.0,
                (win_h / out_h) if out_h else 1.0,
            )
            nodata = src.nodata
            minx, miny, maxx, maxy = _array_bounds(out_h, out_w, transform)
        else:
            total = src.height * src.width
            if max_pixels and total > max_pixels:
                scale = np.sqrt(total / float(max_pixels))
                out_h = max(1, int(round(src.height / scale)))
                out_w = max(1, int(round(src.width / scale)))
            else:
                out_h, out_w = src.height, src.width
            data = src.read(
                1,
                out_shape=(out_h, out_w),
                resampling=Resampling.nearest,
            ).astype(float)
            transform = src.transform * src.transform.scale(
                (src.width / out_w) if out_w else 1.0,
                (src.height / out_h) if out_h else 1.0,
            )
            nodata = src.nodata
            minx, miny, maxx, maxy = _array_bounds(out_h, out_w, transform)

    extent = [minx, maxx, miny, maxy]

    if aoi_gdf is not None and clip_aoi:
        poly_mask = _rio_features.rasterize(
            ((geom, 1) for geom in aoi_gdf.geometry if geom is not None and not geom.is_empty),
            out_shape=data.shape,
            transform=transform,
            fill=0,
            dtype=np.uint8,
            all_touched=True,
        )
        data = np.where(poly_mask.astype(bool), data, np.nan)

    # Mask invalid and 0 values (avoid comparing to nodata when it is None/NaN)
    mask_bad = (data == 0) | (data < 1)
    if nodata is not None:
        if isinstance(nodata, (float, np.floating)) and np.isnan(nodata):
            mask_bad = mask_bad | np.isnan(data)
        else:
            mask_bad = mask_bad | (data == nodata)
    data = np.ma.masked_where(mask_bad, data)
    data /= 10000  # Scale to 0–1+

    # Apply lower cap using vmin (default 0); keep values above vmax so they render with over-color
    cap_min = vmin if vmin is not None else 0.0
    cap_max = vmax if vmax is not None else 1.0
    capped_data = np.clip(data.filled(np.nan), cap_min, None)

    # Plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set extent with padding if clipped
    map_extent = None
    if aoi_geometry is not None:
        map_extent = [
            extent[0] - 1,
            extent[1] + 1,
            extent[2] - 1,
            extent[3] + 1,
        ]
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
    
    im = ax.imshow(
        capped_data,
        extent=extent,
        transform=ccrs.PlateCarree(),
        cmap=plt.get_cmap(cmap),
        vmin=cap_min,
        vmax=cap_max,
        interpolation='nearest'
    )

    # Map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'rotation': 0, 'size': _MAP_GRID_LABEL_FONTSIZE}
    gl.ylabel_style = {'rotation': 90, 'size': _MAP_GRID_LABEL_FONTSIZE}
    # Overlay AOI boundary if provided
    if aoi_geometry is not None:
        if color_boundaries:
            _plot_aridity_aoi_boundaries(
                ax,
                aoi_geometry,
                boundary_col=aoi_boundary_col,
                linewidth=_ARIDITY_AOI_LINEWIDTH,
            )
        else:
            aoi_gdf_plot = _aoi_geometry_to_gdf_plot(aoi_geometry)
            aoi_gdf_plot.plot(
                ax=ax,
                facecolor="none",
                edgecolor="black",
                linewidth=_MAP_AOI_LINEWIDTH,
                transform=ccrs.PlateCarree(),
            )
        # Per-polygon gpd.plot can alter Cartopy axes limits; restore locked extent
        ax.set_extent(map_extent, crs=ccrs.PlateCarree())
        ax.set_autoscale_on(False)

    # Colorbar with arrow tip to indicate values > 1
    cbar = fig.colorbar(
        im, ax=ax, orientation='vertical',
        pad=_MAP_CBAR_PAD, fraction=_MAP_CBAR_FRACTION,
        extend='neither', shrink=_MAP_CBAR_SHRINK_V,
    )
    cbar.set_label(label, fontsize=12)
    cbar.set_ticks(np.linspace(cap_min, cap_max, 6))
    cbar.ax.tick_params(labelsize=10)

    # Title
    #ax.set_title('Global Aridity Index (Zomer et al. 2022)', fontsize=14)
    plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)
    if save_path:
        plt.savefig(save_path, dpi=300)
    plt.show()


def _decompose_grace_calendar(grace_ts_array, time_coords):
    """Return harmonic residual using calendar elapsed months as time axis.

    Fits  y = a*t + b + annual(sin/cos) + semi-annual(sin/cos)
    where *t* is elapsed months from the first timestamp (float), so the
    annual/semi-annual harmonics stay phase-locked to the real calendar even
    when months are missing (e.g. 2017-2018 GRACE gap).

    Parameters
    ----------
    grace_ts_array : 1-D numpy array
        GRACE values aligned to *time_coords*.
    time_coords : 1-D numpy array of datetime64
        Corresponding timestamps (same length as *grace_ts_array*).

    Returns
    -------
    1-D numpy array  (same length as input) with residuals; NaN where input
    was NaN or where the fit failed.
    """
    n = len(grace_ts_array)
    out = np.full(n, np.nan)

    dates = pd.to_datetime(time_coords)
    valid = ~np.isnan(grace_ts_array)
    if valid.sum() < 12:
        return out

    t0 = dates.min()
    sec_per_month = 365.25 / 12 * 24 * 3600
    t_all = (dates - t0).total_seconds().values / sec_per_month
    y_all = np.asarray(grace_ts_array, dtype=float)

    X = np.column_stack([
        t_all,
        np.ones(n),
        np.cos(2 * np.pi * t_all / 12), np.sin(2 * np.pi * t_all / 12),
        np.cos(2 * np.pi * t_all / 6),  np.sin(2 * np.pi * t_all / 6),
    ])

    try:
        coeffs, _, _, _ = np.linalg.lstsq(X[valid], y_all[valid], rcond=None)
    except Exception:
        return out

    fitted = X @ coeffs
    residual = y_all - fitted
    out[valid] = residual[valid]
    return out


def _calculate_lag_correlation_pixel(
    grace_ts, precip_ts, max_lag, min_points, corr_method='spearman',
):
    """Compute Spearman/Pearson at lags 0..max_lag (precip leads GRACE).

    At lag *L* the correlation is between ``precip[:-L]`` and ``grace[L:]``
    (precipitation *L* months earlier compared with GRACE at the current
    month).  Lag 0 compares the same month.

    Returns
    -------
    tuple (best_r, best_p, best_lag, r_lag0, p_lag0)
        *best_** refer to the lag with the highest positive correlation.
        If no valid lag is found all values are NaN.
    """
    from scipy.stats import spearmanr, pearsonr

    method = str(corr_method).lower()
    if method == 'spearman':
        corr_func = spearmanr
    elif method == 'pearson':
        corr_func = pearsonr
    else:
        raise ValueError("corr_method must be 'spearman' or 'pearson'")

    best_r, best_p, best_lag = np.nan, np.nan, np.nan
    r_lag0, p_lag0 = np.nan, np.nan

    for lag in range(0, max_lag + 1):
        if lag == 0:
            g, p_arr = grace_ts, precip_ts
        else:
            if len(grace_ts) <= lag:
                continue
            g = grace_ts[lag:]
            p_arr = precip_ts[:-lag]

        mask = ~(np.isnan(g) | np.isnan(p_arr))
        n_ok = int(mask.sum())
        if n_ok < min_points:
            continue
        if np.std(g[mask]) == 0 or np.std(p_arr[mask]) == 0:
            continue

        try:
            r, p = corr_func(g[mask], p_arr[mask])
        except Exception:
            continue
        if np.isnan(r) or np.isnan(p):
            continue

        if lag == 0:
            r_lag0, p_lag0 = r, p

        if np.isnan(best_r) or r > best_r:
            best_r, best_p, best_lag = r, p, float(lag)

    return best_r, best_p, best_lag, r_lag0, p_lag0


def calculate_grace_precip_correlation_per_pixel(
    grace_data, precip_data, use_residual=True, exclude_years=[2017, 2018],
    client=None, aoi_geometry=None, corr_method='spearman',
    max_lag_months=12, min_common_dates=24, precip_mode='anomaly',
):
    """
    Calculate correlation between GRACE and precipitation for each pixel,
    including lag analysis where precipitation leads GRACE.

    Uses calendar-based harmonic decomposition (elapsed months from the first
    timestamp) so that the annual/semi-annual fit stays phase-locked to the
    real calendar even when gap years are excluded.

    Parameters
    ----------
    grace_data : xarray.DataArray
        GRACE data (e.g., grace_mean) with 'time', 'lat', 'lon' dimensions.
    precip_data : xarray.DataArray
        Precipitation data with 'time', 'lat', 'lon' dimensions.
        Will be automatically coarsened to match GRACE resolution if needed.
    use_residual : bool, default=True
        If True, decompose GRACE to residual before correlation.
        If False, use TWS directly.
    exclude_years : list, default=[2017, 2018]
        Years to exclude from GRACE and from correlation pairing (e.g., GRACE gap
        years). For ``precip_mode='cumsum'``, excluded years are removed from the
        paired time axis only after cumulative anomaly is computed along the full
        precipitation timeline on the GRACE grid, so post-gap months retain the
        integrated effect of those years.
    client : dask.distributed.Client, optional
        Dask client for parallel processing. Pass the notebook ``client`` from
        ``LocalCluster`` to use all workers; if None, uses the default scheduler.
    aoi_geometry : GeoSeries, GeoDataFrame, or geometry, optional
        Area of interest geometry to clip both GRACE and precipitation data
        before analysis.
    corr_method : {'spearman', 'pearson'}, default='spearman'
        Correlation method used for per-pixel correlation and p-value.
    max_lag_months : int, default=12
        Maximum lag to test (in months).  Positive lag means precipitation
        leads GRACE (precip at time *t* compared with GRACE at *t + lag*).
    min_common_dates : int, default=24
        Minimum number of overlapping non-NaN months required for a valid
        correlation at any lag.
    precip_mode : {'month_to_month', 'anomaly', 'cumsum'}, default='anomaly'
        Precipitation predictor used in correlation:
        - ``month_to_month``: raw monthly precipitation P(t)
        - ``anomaly``: monthly anomaly P(t) - climatology(month)
        - ``cumsum``: cumulative anomaly cumsum(P(t) - climatology(month)),
          integrated along all overlapping precip months on the GRACE grid before
          applying ``exclude_years`` to align with GRACE.

    Returns
    -------
    tuple : (correlation_da, pvalue_da, lag_da, correlation_lag0_da)
        correlation_da   : optimal-lag Spearman/Pearson ρ per pixel
        pvalue_da        : p-value at the optimal lag per pixel
        lag_da           : optimal lag (months) per pixel
        correlation_lag0_da : zero-lag ρ per pixel
    """
    from shapely.geometry import mapping

    # ------------------------------------------------------------------
    # 1. Spatial metadata
    # ------------------------------------------------------------------
    if not grace_data.rio.crs:
        grace_data.rio.write_crs("EPSG:4326", inplace=True)
    if not precip_data.rio.crs:
        precip_data.rio.write_crs("EPSG:4326", inplace=True)
    try:
        grace_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        precip_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    except Exception:
        pass

    # ------------------------------------------------------------------
    # 2. Clip to AOI
    # ------------------------------------------------------------------
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")

        geom_clip = [mapping(geom.buffer(0)) for geom in aoi_gdf.geometry]
        try:
            grace_data = grace_data.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
            precip_data = precip_data.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
        except Exception as e:
            _note(
                f"AOI clip failed ({e}); continuing unclipped. "
                "Hint: ensure DataArray CRS is EPSG:4326 and spatial dims are lon/lat."
            )

    # ------------------------------------------------------------------
    # 3. Coarsen precipitation to GRACE grid
    # ------------------------------------------------------------------
    if ('lat' in precip_data.dims and 'lon' in precip_data.dims
            and 'lat' in grace_data.dims and 'lon' in grace_data.dims):
        precip_data = precip_data.interp(lat=grace_data.lat, lon=grace_data.lon, method='nearest')

    # ------------------------------------------------------------------
    # 4–5. Temporal alignment, year exclusion, precipitation predictor
    # ------------------------------------------------------------------
    _mode = str(precip_mode).strip().lower()
    if _mode not in {'month_to_month', 'anomaly', 'cumsum'}:
        raise ValueError(
            "precip_mode must be one of {'month_to_month', 'anomaly', 'cumsum'}"
        )

    if _mode == 'cumsum':
        # Cumulative anomaly along the full precip timeline on the GRACE grid so gap
        # years contribute to the integral; pair only GRACE times after exclude_years.
        climatology_full = _calculate_precip_climatology(precip_data)
        precip_monthly_anomaly_full = (
            precip_data.groupby("time.month") - climatology_full
        )
        precip_cumsum_full = precip_monthly_anomaly_full.cumsum(dim="time")

        if exclude_years:
            grace_aligned = grace_data.sel(
                time=~grace_data.time.dt.year.isin(exclude_years))
        else:
            grace_aligned = grace_data

        grace_aligned, precip_predictor = xr.align(
            grace_aligned, precip_cumsum_full, join="inner"
        )
    else:
        grace_aligned, precip_aligned = xr.align(grace_data, precip_data, join="inner")
        if len(grace_aligned.time) == 0:
            raise ValueError(
                "No overlapping time periods between GRACE and precipitation data "
                "after alignment"
            )
        if grace_aligned.shape[1] == 0 or grace_aligned.shape[2] == 0:
            raise ValueError(
                "No overlapping spatial coverage between GRACE and precipitation "
                "data after alignment"
            )

        if exclude_years:
            grace_aligned = grace_aligned.sel(
                time=~grace_aligned.time.dt.year.isin(exclude_years))
            precip_aligned = precip_aligned.sel(
                time=~precip_aligned.time.dt.year.isin(exclude_years))
        grace_aligned, precip_aligned = xr.align(
            grace_aligned, precip_aligned, join="inner"
        )

        if _mode == 'month_to_month':
            precip_predictor = precip_aligned
        else:  # anomaly
            climatology = _calculate_precip_climatology(precip_aligned)
            precip_monthly_anomaly = precip_aligned.groupby("time.month") - climatology
            precip_predictor = precip_monthly_anomaly

    if len(grace_aligned.time) == 0:
        raise ValueError(
            "No overlapping time periods between GRACE and precipitation data "
            "after alignment"
        )
    if grace_aligned.shape[1] == 0 or grace_aligned.shape[2] == 0:
        raise ValueError(
            "No overlapping spatial coverage between GRACE and precipitation "
            "data after alignment"
        )

    # ------------------------------------------------------------------
    # 6. Rechunk for parallel apply_ufunc (time in one chunk; split lat/lon)
    # ------------------------------------------------------------------
    def _chunk_for_pixel_ufunc(da, lat_chunk=40, lon_chunk=40):
        """Ensure dask-backed chunks suitable for per-pixel ufuncs."""
        import dask.array as dask_array
        if isinstance(da.data, dask_array.Array):
            if da.chunks:
                return da.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})
        return da.chunk({'time': -1, 'lat': lat_chunk, 'lon': lon_chunk})

    grace_aligned = _chunk_for_pixel_ufunc(grace_aligned)
    precip_predictor = _chunk_for_pixel_ufunc(precip_predictor)

    # ------------------------------------------------------------------
    # 7. GRACE decomposition  (calendar elapsed-month harmonic)
    # ------------------------------------------------------------------
    time_values = grace_aligned.time.values  # datetime64 array shared by all pixels

    if use_residual:
        def _decompose_pixel_calendar(grace_ts_array):
            return _decompose_grace_calendar(grace_ts_array, time_values)

        grace_processed = xr.apply_ufunc(
            _decompose_pixel_calendar,
            grace_aligned,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[float],
            dask_gufunc_kwargs={
                'output_sizes': {'time': len(grace_aligned.time)},
                'allow_rechunk': True,
            },
        )
    else:
        grace_processed = grace_aligned

    if hasattr(grace_processed, 'chunks') and grace_processed.chunks:
        grace_processed = grace_processed.chunk({'time': -1, 'lat': 'auto', 'lon': 'auto'})

    # ------------------------------------------------------------------
    # 8. Lag correlation (precip leads GRACE)
    # ------------------------------------------------------------------
    _max_lag = int(max_lag_months)
    _min_pts = int(min_common_dates)

    def _lag_correlation_all(grace_ts, precip_ts):
        best_r, best_p, best_lag, r_lag0, _p_lag0 = _calculate_lag_correlation_pixel(
            grace_ts, precip_ts, _max_lag, _min_pts, corr_method)
        return best_r, best_p, best_lag, r_lag0

    _ufunc_kw = dict(
        input_core_dims=[['time'], ['time']],
        output_core_dims=[[], [], [], []],
        vectorize=True,
        dask='parallelized',
        output_dtypes=[float, float, float, float],
        dask_gufunc_kwargs={'allow_rechunk': True},
    )

    correlation_da, pvalue_da, lag_da, corr_lag0_da = xr.apply_ufunc(
        _lag_correlation_all,
        grace_processed,
        precip_predictor,
        **_ufunc_kw,
    )

    # ------------------------------------------------------------------
    # 9. Compute (dask → numpy) — single fused graph
    # ------------------------------------------------------------------
    import dask
    all_das = [correlation_da, pvalue_da, lag_da, corr_lag0_da]
    compute_kw = {'scheduler': client} if client is not None else {}
    try:
        from dask.diagnostics import ProgressBar
        with ProgressBar():
            correlation_da, pvalue_da, lag_da, corr_lag0_da = dask.compute(
                *all_das, **compute_kw)
    except Exception:
        correlation_da, pvalue_da, lag_da, corr_lag0_da = dask.compute(
            *all_das, **compute_kw)

    n_processed = int(np.sum(~np.isnan(correlation_da.values)))
    grace_label = "residual" if use_residual else "anomaly"
    method_label = str(corr_method).strip().capitalize()
    print(
        f"TWSA-CPA {grace_label}/{_mode}: {n_processed} valid pixels "
        f"({method_label}, lag {int(max_lag_months)})"
    )

    # ------------------------------------------------------------------
    # 10. Names, CRS, spatial dims
    # ------------------------------------------------------------------
    correlation_da.name = 'correlation'
    pvalue_da.name = 'pvalue'
    lag_da.name = 'optimal_lag'
    corr_lag0_da.name = 'correlation_lag0'

    for da in (correlation_da, pvalue_da, lag_da, corr_lag0_da):
        try:
            crs = grace_aligned.rio.crs if grace_aligned.rio.crs else "EPSG:4326"
            da.rio.write_crs(crs, inplace=True)
            da.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        except Exception:
            if not da.rio.crs:
                da.rio.write_crs("EPSG:4326", inplace=True)

    for da in (correlation_da, pvalue_da, lag_da, corr_lag0_da):
        if 'lon' not in da.coords or 'lat' not in da.coords:
            raise ValueError(f"Failed to set coordinates on {da.name}. "
                             f"coords: {list(da.coords.keys())}")

    return correlation_da, pvalue_da, lag_da, corr_lag0_da



def _correlation_classify_bounds_labels(thresholds, vmin, vmax):
    """Build BoundaryNorm bounds and colorbar tick labels (low to high on colorbar)."""
    thresholds = sorted([float(t) for t in thresholds], reverse=True)
    vmin, vmax = float(vmin), float(vmax)
    if len(thresholds) < 1:
        raise ValueError("classify_thresholds must contain at least one value")
    for t in thresholds:
        if not (vmin <= t <= vmax):
            raise ValueError(
                f"classify threshold {t:g} must lie within plot limits [{vmin:g}, {vmax:g}]"
            )
    inner = list(reversed(thresholds))
    bounds = np.array([vmin] + inner + [vmax], dtype=float)
    cbar_labels = []
    tick_locs = []
    cbar_labels.append(f"< {thresholds[-1]:g}")
    tick_locs.append((vmin + thresholds[-1]) / 2.0)
    for i in range(len(thresholds) - 1, 0, -1):
        lo, hi = thresholds[i], thresholds[i - 1]
        cbar_labels.append(f"{lo:g}–{hi:g}")
        tick_locs.append((lo + hi) / 2.0)
    cbar_labels.append(f"≥ {thresholds[0]:g}")
    tick_locs.append((thresholds[0] + vmax) / 2.0)
    return bounds, cbar_labels, np.asarray(tick_locs, dtype=float)


def _correlation_class_counts(values, thresholds, vmin, vmax):
    """Count pixels per correlation class (class 0 = highest tier)."""
    thresholds = sorted([float(t) for t in thresholds], reverse=True)
    vmin, vmax = float(vmin), float(vmax)
    vals = np.asarray(values, dtype=float).ravel()
    valid = np.isfinite(vals)
    n_valid = int(valid.sum())
    rows = []

    def _append(class_idx, label, lower, upper, mask):
        n = int(mask.sum())
        pct = (100.0 * n / n_valid) if n_valid > 0 else np.nan
        rows.append(
            {
                "class": class_idx,
                "label": label,
                "lower": lower,
                "upper": upper,
                "n_pixels": n,
                "pct": pct,
            }
        )

    class_idx = 0
    _append(
        class_idx,
        f"≥ {thresholds[0]:g}",
        thresholds[0],
        np.inf,
        valid & (vals >= thresholds[0]),
    )
    class_idx += 1
    for i in range(len(thresholds) - 1):
        hi, lo = thresholds[i], thresholds[i + 1]
        _append(
            class_idx,
            f"{lo:g}–{hi:g}",
            lo,
            hi,
            valid & (vals >= lo) & (vals < hi),
        )
        class_idx += 1
    _append(
        class_idx,
        f"< {thresholds[-1]:g}",
        -np.inf,
        thresholds[-1],
        valid & (vals < thresholds[-1]),
    )
    return pd.DataFrame(rows)


def _sanitize_raster_token(value):
    """Filesystem-safe token for GeoTIFF filenames."""
    token = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(value).strip())
    return token.strip("_") or "untagged"


def _compose_raster_output_path(saved_rasters_path, stem, tags=()):
    """Build ``<saved_rasters_path>/<stem>_<tag1>_<tag2>.tif``."""
    parts = [_sanitize_raster_token(stem)]
    for tag in tags:
        t = _sanitize_raster_token(tag)
        if t and t not in parts:
            parts.append(t)
    filename = "_".join(parts) + ".tif"
    return str(Path(saved_rasters_path) / filename)


def _export_geotiff_dataarray(da, out_path, *, nodata=np.nan):
    """Write a lat/lon DataArray to GeoTIFF (EPSG:4326, north-up).

    Invalid/masked cells remain NaN in the file (float32, nodata=NaN) so GIS
    software does not display them as zero.
    """
    from rasterio.transform import from_bounds

    export_da = da.copy()
    if export_da.ndim > 2:
        export_da = export_da.squeeze(drop=True)
    if "lat" not in export_da.dims or "lon" not in export_da.dims:
        raise ValueError("DataArray must have lat/lon dimensions for GeoTIFF export")

    export_da = export_da.sortby("lat")
    if "lon" in export_da.dims:
        export_da = export_da.sortby("lon")
    lats = np.asarray(export_da.lat.values, dtype=np.float64)
    lons = np.asarray(export_da.lon.values, dtype=np.float64)

    # float32 + explicit NaN: never fill masked pixels with 0
    arr = np.asarray(export_da.values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, np.nan).astype(np.float32)

    # Matplotlib maps use origin='lower' (row 0 = south). GeoTIFF row 0 = north.
    if len(lats) > 1 and lats[0] < lats[-1]:
        arr = np.flipud(arr)
        lats = lats[::-1]
    if len(lons) > 1 and lons[0] > lons[-1]:
        arr = np.fliplr(arr)
        lons = lons[::-1]

    if len(lons) > 1:
        lon_res = float(np.diff(lons).mean())
    else:
        lon_res = 1.0
    if len(lats) > 1:
        lat_res = float(np.diff(lats).mean())
    else:
        lat_res = 1.0

    west = float(lons.min()) - lon_res / 2.0
    east = float(lons.max()) + lon_res / 2.0
    south = float(lats.min()) - abs(lat_res) / 2.0
    north = float(lats.max()) + abs(lat_res) / 2.0
    transform = from_bounds(west, south, east, north, len(lons), len(lats))

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "nodata": np.float32(np.nan),
        "width": len(lons),
        "height": len(lats),
        "count": 1,
        "crs": "EPSG:4326",
        "transform": transform,
    }
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr, 1)

    _item(_rel(out_path), "ok")
    return out_path



def plot_grace_correlation_map(
    correlation_data, pvalue_data=None, cmap='RdBu_r', 
    label=r"Correlation Coefficient (r)", 
    vmin=None, vmax=None, dpi=300, figsize=(10, 4), save_path=None, 
    aoi_geometry=None, EPE95_sorted=None, mask_non_significant=True, 
    significance_level=0.05,
    cbar_orientation='vertical',
    stipple_non_significant=False,
    stipple_size=5,
    stipple_alpha=0.9,
    divergent=True,
    cbar_ticks=None,
    *,
    classify_thresholds=None,
    classify_values=False,
    save_raster=False,
    saved_rasters_path="outputs/rasters",
    raster_tags=None,
):
    """
    Plot GRACE-precipitation correlation map with optional p-value masking.
    
    Parameters:
    -----------
    correlation_data : xarray.DataArray
        Correlation coefficient data from calculate_grace_precip_correlation_per_pixel
    pvalue_data : xarray.DataArray, optional
        P-value data from calculate_grace_precip_correlation_per_pixel.
        If provided and mask_non_significant=True, non-significant pixels will be masked.
    cmap : str, default='RdBu_r'
        Colormap name (divergent recommended for correlations)
    label : str, default=r"Correlation Coefficient"
        Colorbar label
    vmin : float, optional
        Minimum value for colormap (centered at 0 if None)
    vmax : float, optional
        Maximum value for colormap (centered at 0 if None)
    dpi : int, default=500
        Resolution for saved figure
    figsize : tuple, default=(10, 5)
        Figure size
    save_path : str, optional
        Path to save the figure
    aoi_geometry : GeoSeries, GeoDataFrame, or geometry, optional
        Area of interest geometry to clip the data and overlay boundaries
    EPE95_sorted : GeoDataFrame, optional
        GeoDataFrame to overlay on the map
    mask_non_significant : bool, default=True
        If True and pvalue_data is provided, mask non-significant pixels (p >= significance_level)
    significance_level : float, default=0.05
        Significance level for masking non-significant correlations
    cbar_orientation : {'vertical', 'horizontal'}, default='vertical'
        Colorbar placement. Vertical matches ``plot_aridity_raster``, ``analyze_lsm_outputs``,
        and ``plot_pixel_analysis_maps`` (``_MAP_CBAR_*`` layout).
    stipple_non_significant : bool, default=False
        If True and ``pvalue_data`` is provided, overlay non-significant pixels (p >= significance_level)
        as stippling markers.
    stipple_size : float, default=10
        Marker size for non-significant stippling.
    stipple_alpha : float, default=0.6
        Marker alpha for non-significant stippling.
    divergent : bool, default=True
        If True, use ``TwoSlopeNorm`` centered at 0 (appropriate for
        correlation maps).  If False, use standard ``Normalize`` (appropriate
        for lag or other sequential maps).
    cbar_ticks : array-like, optional
        Explicit colorbar tick values.  If None and ``divergent`` is True,
        defaults to ``[-1, -0.5, 0, 0.5, 1]``. Ignored when ``classify_values``
        is True.
    classify_thresholds : list of float, optional
        Thresholds defining correlation classes (e.g. ``[0.5, 0.2]`` → three
        classes: ``≥ 0.5``, ``0.2–0.5``, ``< 0.2``). Required when
        ``classify_values=True``.
    classify_values : bool, default=False
        If True, bin correlations into discrete classes using
        ``classify_thresholds``. Uses a discrete colormap and class-center
        colorbar ticks. ``divergent`` and ``cbar_ticks`` are ignored in this
        mode. Negative correlations fall in the lowest class.
    save_raster : bool, default=False
        If True, export the continuous correlation grid (ρ) as GeoTIFF under
        ``saved_rasters_path``. Classification is display-only and is not written
        into the raster values or filename.
    saved_rasters_path : str, default="outputs/rasters"
        Output directory for GeoTIFF exports when ``save_raster=True``. Relative
        paths are resolved against the repository root (``…/github``), not the
        process cwd. Prefer ``str(OUTPUT_DIR / "rasters")`` from the notebook.
    raster_tags : list of str, optional
        Extra filename tokens (e.g. ``precip_mode``, ``residual``, ``0lag``).

    Returns
    -------
    fig : matplotlib.figure.Figure
    ax : cartopy.mpl.geoaxes.GeoAxes
    class_stats : pandas.DataFrame or None
        Per-class pixel counts and percentages when ``classify_values`` is True;
        otherwise ``None``.
    """
    # Verify input has required coordinates
    if 'lon' not in correlation_data.coords or 'lat' not in correlation_data.coords:
        raise ValueError(f"correlation_data must have 'lon' and 'lat' as coordinates. "
                        f"Available coords: {list(correlation_data.coords.keys())}, "
                        f"Available dims: {list(correlation_data.dims)}")
    
    # Ensure CRS and spatial dimensions are set
    if not correlation_data.rio.crs:
        correlation_data.rio.write_crs("EPSG:4326", inplace=True)
    
    try:
        x_dim = correlation_data.rio.x_dim
        y_dim = correlation_data.rio.y_dim
    except AttributeError:
        if 'lon' in correlation_data.dims and 'lat' in correlation_data.dims:
            correlation_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        else:
            correlation_data.rio.set_spatial_dims(x_dim='x', y_dim='lat', inplace=True)
    
    # Mask non-significant pixels if requested
    if mask_non_significant and pvalue_data is not None:
        # Ensure pvalue_data has same CRS and spatial dims
        if not pvalue_data.rio.crs:
            pvalue_data.rio.write_crs("EPSG:4326", inplace=True)
        try:
            pvalue_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        except:
            pass
        
        # Ensure correlation_data has coordinates before masking
        if 'lon' not in correlation_data.coords or 'lat' not in correlation_data.coords:
            raise ValueError("correlation_data must have 'lon' and 'lat' coordinates. "
                           f"Available coords: {list(correlation_data.coords.keys())}")
        
        # Create mask for significant correlations
        significant_mask = pvalue_data < significance_level
        correlation_data = correlation_data.where(significant_mask)
        
        # Ensure coordinates are preserved after masking
        if 'lon' not in correlation_data.coords or 'lat' not in correlation_data.coords:
            raise ValueError("Coordinates were lost after masking. This should not happen.")
    
    # Calculate extent from coordinates (coordinates are always valid even if data is NaN)
    # Get coordinate arrays directly - check multiple ways to access coordinates
    try:
        if 'lon' in correlation_data.coords:
            lon_coords = correlation_data.lon.values
        elif 'lon' in correlation_data.dims:
            lon_coords = correlation_data.indexes['lon'].values
        else:
            raise ValueError("Cannot find 'lon' coordinate in correlation_data")
            
        if 'lat' in correlation_data.coords:
            lat_coords = correlation_data.lat.values
        elif 'lat' in correlation_data.dims:
            lat_coords = correlation_data.indexes['lat'].values
        else:
            raise ValueError("Cannot find 'lat' coordinate in correlation_data")
    except (KeyError, AttributeError) as e:
        raise ValueError(f"Cannot determine extent: correlation_data has no valid coordinates. Error: {e}\n"
                        f"Available coords: {list(correlation_data.coords.keys())}, dims: {list(correlation_data.dims)}")
    
    if len(lon_coords) == 0 or len(lat_coords) == 0:
        raise ValueError("Cannot determine extent: correlation_data has empty coordinate arrays")
    
    # Get min/max from coordinate arrays (these should always be valid)
    lon_min = float(np.min(lon_coords))
    lon_max = float(np.max(lon_coords))
    lat_min = float(np.min(lat_coords))
    lat_max = float(np.max(lat_coords))
    
    # Account for pixel size (half pixel on each side)
    if len(lon_coords) > 1:
        lon_res = float(np.diff(lon_coords).mean())
    else:
        lon_res = 1.0
    if len(lat_coords) > 1:
        lat_res = float(np.diff(lat_coords).mean())
    else:
        lat_res = 1.0
    minx = lon_min - lon_res / 2
    maxx = lon_max + lon_res / 2
    miny = lat_min - lat_res / 2
    maxy = lat_max + lat_res / 2
    extent = [minx, maxx, miny, maxy]
    
    # Clip to AOI if provided (using mask approach if rio.clip fails)
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs)
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.copy()
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        
        # Ensure CRS matches
        aoi_gdf = aoi_gdf.to_crs("EPSG:4326")
        
        # Try rio.clip, but if it fails, we'll just overlay the boundary
        try:
            correlation_data = correlation_data.rio.clip(aoi_gdf.geometry, crs="EPSG:4326", drop=False)
        except Exception:
            # If clipping fails, we'll just show the boundary overlay
            pass
    
    # Get data values
    data = correlation_data.values
    
    # Determine vmin/vmax
    data_valid = data[~np.isnan(data)]
    use_classify = bool(classify_values)
    if use_classify:
        if classify_thresholds is None:
            raise ValueError("classify_thresholds is required when classify_values=True")
    if use_classify:
        if divergent:
            if vmin is None:
                vmin = -1.0
            if vmax is None:
                vmax = 1.0
        else:
            if len(data_valid) > 0:
                vmin = vmin if vmin is not None else float(data_valid.min())
                vmax = vmax if vmax is not None else float(data_valid.max())
            else:
                vmin = vmin if vmin is not None else 0
                vmax = vmax if vmax is not None else 1
        bounds, class_cbar_labels, class_tick_locs = _correlation_classify_bounds_labels(
            classify_thresholds, vmin, vmax
        )
        n_classes = len(classify_thresholds) + 1
        cmap_obj = plt.get_cmap(cmap, int(n_classes))
        norm = mcolors.BoundaryNorm(boundaries=bounds, ncolors=int(n_classes))
        class_stats = None
    elif divergent:
        # Keep correlation maps visually consistent across runs.
        # If user did not request custom limits, default to full correlation range.
        if vmin is None:
            vmin = -1.0
        if vmax is None:
            vmax = 1.0
        norm = mcolors.TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        class_stats = None
        bounds = class_cbar_labels = class_tick_locs = None
    else:
        if len(data_valid) > 0:
            vmin = vmin if vmin is not None else float(data_valid.min())
            vmax = vmax if vmax is not None else float(data_valid.max())
        else:
            vmin = vmin if vmin is not None else 0
            vmax = vmax if vmax is not None else 1
        norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
        cmap_obj = plt.get_cmap(cmap)
        class_stats = None
        bounds = class_cbar_labels = class_tick_locs = None
    
    # Plot
    fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': ccrs.PlateCarree()})
    
    # Set extent with padding if clipped
    if aoi_geometry is not None:
        ax.set_extent([extent[0] - 1, extent[1] + 1, extent[2] - 1, extent[3] + 1], 
                      crs=ccrs.PlateCarree())
    
    # Ensure lat is sorted in ascending order (low to high)
    correlation_data_sorted = correlation_data.sortby('lat')

    if use_classify:
        class_stats = _correlation_class_counts(
            correlation_data_sorted.values, classify_thresholds, vmin, vmax
        )
        print("Correlation class summary (valid mapped pixels):")
        print(
            class_stats[["class", "label", "n_pixels", "pct"]]
            .to_string(index=False, float_format=lambda x: f"{x:.2f}")
        )

    # Calculate extent for imshow from coordinates (consistent with above)
    lon_coords_sorted = correlation_data_sorted.lon.values
    lat_coords_sorted = correlation_data_sorted.lat.values
    
    lon_min_sorted = float(np.min(lon_coords_sorted))
    lon_max_sorted = float(np.max(lon_coords_sorted))
    lat_min_sorted = float(np.min(lat_coords_sorted))
    lat_max_sorted = float(np.max(lat_coords_sorted))
    
    if len(lon_coords_sorted) > 1:
        lon_res_sorted = float(np.diff(lon_coords_sorted).mean())
    else:
        lon_res_sorted = 1.0
    if len(lat_coords_sorted) > 1:
        lat_res_sorted = float(np.diff(lat_coords_sorted).mean())
    else:
        lat_res_sorted = 1.0
    minx_imshow = lon_min_sorted - lon_res_sorted / 2
    maxx_imshow = lon_max_sorted + lon_res_sorted / 2
    miny_imshow = lat_min_sorted - lat_res_sorted / 2
    maxy_imshow = lat_max_sorted + lat_res_sorted / 2
    extent_imshow = [minx_imshow, maxx_imshow, miny_imshow, maxy_imshow]
    
    # Use imshow with origin='lower' to fix flipped orientation
    im = ax.imshow(
        correlation_data_sorted.values,
        extent=extent_imshow,
        transform=ccrs.PlateCarree(),
        cmap=cmap_obj,
        norm=norm,
        interpolation='nearest',
        origin='lower'  # Fix flipped orientation
    )

    # Optional stippling for non-significant pixels (p >= alpha threshold)
    if stipple_non_significant and pvalue_data is not None:
        pvalue_sorted = pvalue_data.sortby('lat')
        _, pvalue_sorted = xr.align(correlation_data_sorted, pvalue_sorted, join='inner')
        nonsig_mask = (pvalue_sorted.values >= significance_level)
        if np.any(nonsig_mask):
            lon_mesh, lat_mesh = np.meshgrid(correlation_data_sorted.lon.values, correlation_data_sorted.lat.values)
            ax.scatter(
                lon_mesh[nonsig_mask],
                lat_mesh[nonsig_mask],
                s=stipple_size,
                c='k',
                alpha=stipple_alpha,
                marker='o',
                linewidths=0,
                transform=ccrs.PlateCarree(),
                zorder=4,
            )
    
    # Map features
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)
    ax.add_feature(cfeature.LAND, facecolor='lightgray')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    
    # Add gridlines
    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color='gray', 
                     alpha=0.5, linestyle='--')
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {'rotation': 0, 'size': _MAP_GRID_LABEL_FONTSIZE}
    gl.ylabel_style = {'rotation': 90, 'size': _MAP_GRID_LABEL_FONTSIZE}
    
    # Overlay AOI boundary if provided
    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf_plot = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf_plot = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf_plot = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        
        aoi_gdf_plot.plot(ax=ax, facecolor='none', edgecolor='black', 
                         linewidth=_MAP_AOI_LINEWIDTH, transform=ccrs.PlateCarree())
    
    if EPE95_sorted is not None:
        EPE95_sorted.plot(ax=ax, facecolor='none', edgecolor='blue', 
                         linewidth=1, linestyle='-', transform=ccrs.PlateCarree())
    # Colorbar: match project map style (_MAP_CBAR_* + analyze_lsm_outputs)
    orient = str(cbar_orientation).strip().lower()
    if orient not in ('vertical', 'horizontal'):
        raise ValueError("cbar_orientation must be 'vertical' or 'horizontal'")
    cbar_kw = {
        'orientation': orient,
        'pad': _MAP_CBAR_PAD,
        'fraction': _MAP_CBAR_FRACTION,
        'extend': 'neither',
        'shrink': 0.88 if orient == 'vertical' else _MAP_CBAR_SHRINK_H,
    }
    if use_classify:
        cbar = fig.colorbar(
            im,
            ax=ax,
            boundaries=bounds,
            ticks=class_tick_locs,
            **cbar_kw,
        )
        cbar.set_ticklabels(class_cbar_labels)
        cbar.minorticks_off()
        tick_axis = cbar.ax.xaxis if orient == 'horizontal' else cbar.ax.yaxis
        tick_axis.set_minor_locator(mticker.NullLocator())
    else:
        cbar = fig.colorbar(im, ax=ax, **cbar_kw)
        if cbar_ticks is not None:
            tick_vals = np.asarray(cbar_ticks, dtype=float)
        elif divergent:
            tick_vals = np.array([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
        else:
            tick_vals = None
        if tick_vals is not None:
            all_int_like = np.all(np.isclose(tick_vals, np.round(tick_vals)))
            tick_formatter = FormatStrFormatter('%.0f' if all_int_like else '%.1f')
            if orient == 'vertical':
                cbar.ax.yaxis.set_major_locator(mticker.FixedLocator(tick_vals))
                cbar.ax.yaxis.set_major_formatter(tick_formatter)
            else:
                cbar.ax.xaxis.set_major_locator(mticker.FixedLocator(tick_vals))
                cbar.ax.xaxis.set_major_formatter(tick_formatter)
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"Correlation map saved to: {save_path}")

    if save_raster:
        # Export continuous correlation (Spearman/Pearson ρ), not map class IDs.
        # Callers pass analysis tags via raster_tags (e.g. precip_mode, residual/anomaly, 0lag).
        raster_dir = Path(saved_rasters_path)
        if not raster_dir.is_absolute():
            # Resolve relative paths against the repo root (…/github), not the cwd
            repo_root = Path(__file__).resolve().parent.parent
            raster_dir = repo_root / raster_dir
        raster_tag_list = list(raster_tags or [])
        if mask_non_significant:
            raster_tag_list.append("sig_masked")
        label_l = str(label).lower()
        if "spearman" in label_l:
            raster_tag_list.append("spearman")
        elif "pearson" in label_l:
            raster_tag_list.append("pearson")
        raster_out = _compose_raster_output_path(
            str(raster_dir),
            "grace_precip_correlation",
            raster_tag_list,
        )
        _export_geotiff_dataarray(correlation_data_sorted, raster_out)

    plt.show()
    return fig, ax, class_stats


def summarize_grace_correlation_outputs(
    correlation_da,
    pvalue_da,
    lag_da,
    significance_level=0.05,
    dist_figsize=None,
    show_stats_box=True,
    stats_box_fields=('median', 'q05', 'q95'),
    significant_only=False,
    hist_use_map_cmap=False,
    show_lag_panel=True,
    correlation_ylim_max=None,
    lag_ylim_max=None,
    dpi=300,
    save_prefix=None,
):
    """
    Summarize correlation/lag outputs and generate distribution-only figure.

    This utility computes summary statistics and plots distributions for
    correlation and lag outputs. Map plotting has been removed; use
    ``plot_grace_correlation_map(...)`` directly when map visuals are needed.

    Parameters
    ----------
    correlation_da : xarray.DataArray
        Pixelwise optimal-lag correlation.
    pvalue_da : xarray.DataArray
        Pixelwise p-values corresponding to ``correlation_da``.
    lag_da : xarray.DataArray
        Pixelwise optimal lag (months).
    significance_level : float, default=0.05
        Significance threshold used for summary counts.
    dist_figsize : tuple, optional
        Figure size. Defaults to ``(13, 4.8)`` with lag panel, ``(6.5, 4.8)``
        without.
    show_stats_box : bool, default=True
        If True, draw per-histogram summary box.
    stats_box_fields : tuple, default=('median', 'q05', 'q95')
        Statistic fields to show in each histogram box.
    significant_only : bool, default=False
        If True, distributions are plotted only for significant grid cells
        (``pvalue_da < significance_level``). In this mode, the histogram
        stats box ``N grid cells`` reflects the significant subset size.
    hist_use_map_cmap : bool, default=False
        If True, histogram bars use map-matched colormaps:
        correlation uses ``RdBu`` over [-1, 1], lag uses ``viridis_r`` over [0, 12].
    show_lag_panel : bool, default=True
        If False, plot only the correlation histogram.
    correlation_ylim_max : float, optional
        Optional upper y-limit for the correlation histogram (grid-cell count axis).
    lag_ylim_max : float, optional
        Optional upper y-limit for the lag histogram (grid-cell count axis).
    dpi : int, default=300
        Saved figure resolution.
    save_prefix : str, optional
        If provided, saves ``{save_prefix}_distributions.png``.

    Returns
    -------
    dict
        {
          'stats': nested statistics dictionary,
          'fig_distributions': matplotlib Figure
        }
    """
    def _valid_vals(da):
        vals = np.asarray(da.values, dtype=float).ravel()
        return vals[np.isfinite(vals)]

    def _stats_box_text(metric_stats, n_valid):
        field_label = {
            'mean': 'Mean',
            'median': 'Median',
            'std': 'Std',
            'q05': 'Q05',
            'q95': 'Q95',
            'min': 'Min',
            'max': 'Max',
        }
        lines = [f"N grid cells = {n_valid:,}"]
        for fld in stats_box_fields:
            if fld not in metric_stats:
                continue
            fmt = "{:.2f}" if metric_stats is not stats['lag_months'] else "{:.1f}"
            lines.append(f"{field_label.get(fld, fld)} = {fmt.format(metric_stats[fld])}")
        lines.append("Median line: solid")
        lines.append("Q05/Q95 lines: dashed")
        return "\n".join(lines)

    def _plot_single_hist(
        ax, vals, bins, color, xlabel, metric_key,
        box_side='left', show_ylabel=True, use_cmap=False, cmap_name=None, cmap_norm=None
    ):
        if use_cmap and cmap_name is not None and cmap_norm is not None:
            counts, edges = np.histogram(vals, bins=bins)
            centers = 0.5 * (edges[:-1] + edges[1:])
            widths = np.diff(edges)
            cmap_obj = plt.get_cmap(cmap_name)
            bar_colors = [cmap_obj(cmap_norm(c)) for c in centers]
            ax.bar(
                edges[:-1], counts, width=widths, align='edge',
                color=bar_colors, alpha=0.9, edgecolor='white', linewidth=0.5
            )
        else:
            ax.hist(vals, bins=bins, color=color, alpha=0.85, edgecolor='white', linewidth=0.5)
        ax.set_xlabel(xlabel, fontsize=13)
        ax.set_ylabel("Grid cell count", fontsize=13 if show_ylabel else 0)
        if not show_ylabel:
            ax.set_ylabel("")
        ax.grid(True, linestyle='--', alpha=0.35)
        ax.tick_params(labelsize=12)

        # Vertical guides for manuscript readability.
        med = stats[metric_key]['median']
        q05 = stats[metric_key]['q05']
        q95 = stats[metric_key]['q95']
        ln_med = ax.axvline(med, color='black', linestyle='--', linewidth=1.5, alpha=0.95, label=f"Median = {med:.2f}" if metric_key != 'lag_months' else f"Median = {med:.1f}")
        #ln_q05 = ax.axvline(q05, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f"Q05 = {q05:.2f}" if metric_key != 'lag_months' else f"Q05 = {q05:.1f}")
        #ln_q95 = ax.axvline(q95, color='black', linestyle='--', linewidth=1.2, alpha=0.8, label=f"Q95 = {q95:.2f}" if metric_key != 'lag_months' else f"Q95 = {q95:.1f}")

        if show_stats_box:
            from matplotlib.lines import Line2D
            n_item = Line2D([], [], linestyle='None', marker=None, color='none', label=f"N grid cells = {len(vals):,}")
            handles = [n_item, ln_med]
            if str(box_side).lower() == 'right':
                legend_loc = 'upper right'
            else:
                legend_loc = 'upper left'
            ax.legend(
                handles=handles,
                loc=legend_loc,
                frameon=True,
                framealpha=0.9,
                facecolor='white',
                edgecolor='0.6',
                fontsize=12,
            )

    corr_vals_all = _valid_vals(correlation_da)
    p_vals = _valid_vals(pvalue_da)
    lag_vals_all = _valid_vals(lag_da)

    if corr_vals_all.size == 0:
        raise ValueError("correlation_da has no finite values to summarize.")
    if p_vals.size == 0:
        raise ValueError("pvalue_da has no finite values to summarize.")
    if lag_vals_all.size == 0:
        raise ValueError("lag_da has no finite values to summarize.")

    sig_mask = np.isfinite(pvalue_da.values) & (pvalue_da.values < significance_level)
    total_valid = int(np.isfinite(pvalue_da.values).sum())
    n_sig = int(sig_mask.sum())
    pct_sig = (100.0 * n_sig / total_valid) if total_valid > 0 else np.nan

    stats = {
        'n_pixels': {
            'valid_total': total_valid,
            'significant': n_sig,
            'non_significant': int(total_valid - n_sig),
            'significant_pct': float(pct_sig),
        },
        'correlation': {
            'mean': float(np.nanmean(corr_vals_all)),
            'median': float(np.nanmedian(corr_vals_all)),
            'std': float(np.nanstd(corr_vals_all)),
            'min': float(np.nanmin(corr_vals_all)),
            'max': float(np.nanmax(corr_vals_all)),
            'q05': float(np.nanpercentile(corr_vals_all, 5)),
            'q95': float(np.nanpercentile(corr_vals_all, 95)),
        },
        'pvalue': {
            'mean': float(np.nanmean(p_vals)),
            'median': float(np.nanmedian(p_vals)),
            'std': float(np.nanstd(p_vals)),
            'min': float(np.nanmin(p_vals)),
            'max': float(np.nanmax(p_vals)),
            'q05': float(np.nanpercentile(p_vals, 5)),
            'q95': float(np.nanpercentile(p_vals, 95)),
        },
        'lag_months': {
            'mean': float(np.nanmean(lag_vals_all)),
            'median': float(np.nanmedian(lag_vals_all)),
            'std': float(np.nanstd(lag_vals_all)),
            'min': float(np.nanmin(lag_vals_all)),
            'max': float(np.nanmax(lag_vals_all)),
            'q05': float(np.nanpercentile(lag_vals_all, 5)),
            'q95': float(np.nanpercentile(lag_vals_all, 95)),
        },
    }

    if significant_only:
        corr_arr = np.asarray(correlation_da.values, dtype=float)
        lag_arr = np.asarray(lag_da.values, dtype=float)
        plot_mask = sig_mask & np.isfinite(corr_arr) & np.isfinite(lag_arr)
        corr_vals = corr_arr[plot_mask]
        lag_vals = lag_arr[plot_mask]
        if corr_vals.size == 0 or lag_vals.size == 0:
            raise ValueError(
                "No significant grid cells available to plot "
                f"(p < {significance_level})."
            )
    else:
        corr_vals = corr_vals_all
        lag_vals = lag_vals_all

    dist_specs = [
        (
            corr_vals, np.linspace(-1, 1, 31), "#3b82f6", r"Spearman $\rho$", 'correlation',
            'RdBu', mcolors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=1)
        ),
    ]
    if show_lag_panel:
        dist_specs.append(
            (
                lag_vals, np.arange(-0.5, 13.5, 1.0), "#10b981", "Optimal lag (months)", 'lag_months',
                'viridis_r', mcolors.Normalize(vmin=0, vmax=12)
            )
        )

    n_panels = len(dist_specs)
    if dist_figsize is None:
        dist_figsize = (13, 4.8) if n_panels > 1 else (6.5, 4.8)
    fig_dist, axd = plt.subplots(1, n_panels, figsize=dist_figsize)
    if n_panels == 1:
        axd = [axd]
    for idx, (ax, ds) in enumerate(zip(axd, dist_specs)):
        vals, bins, color, title, metric_key, cmap_name, cmap_norm = ds
        _plot_single_hist(
            ax, vals, bins, color, title, metric_key,
            box_side=('left' if idx == 0 else 'right'),
            show_ylabel=(metric_key == 'correlation'),
            use_cmap=bool(hist_use_map_cmap),
            cmap_name=cmap_name,
            cmap_norm=cmap_norm,
        )
        if metric_key == 'correlation' and correlation_ylim_max is not None:
            ax.set_ylim(0, float(correlation_ylim_max))
        if metric_key == 'lag_months' and lag_ylim_max is not None:
            ax.set_ylim(0, float(lag_ylim_max))
    fig_dist.tight_layout(rect=[0, 0, 1, 1], pad=0.8)

    if save_prefix:
        dist_path = f"{save_prefix}"
        Path(dist_path).parent.mkdir(parents=True, exist_ok=True)
        fig_dist.savefig(dist_path, dpi=dpi, bbox_inches='tight')
        print(f"Saved: {dist_path}")

    return {
        'stats': stats,
        'fig_distributions': fig_dist,
    }


def _encode_pixel_cell_id(i, j):
    """Return stable grid-cell identifier from lat/lon indices."""
    return f"P{int(i):05d}_{int(j):05d}"


def _plotly_colorscale_from_matplotlib_cmap(cmap='RdBu'):
    """
    Map a matplotlib diverging cmap name to Plotly colorscale settings.

    Default ``RdBu`` gives blue for positive values and red for negative.
    ``RdBu_r`` reverses that (red positive, blue negative).
    """
    name = str(cmap).strip()
    lower = name.lower()
    if lower in {'rdbu', 'rdbu_r'}:
        return 'RdBu', lower.endswith('_r')
    if lower in {'rdylbu', 'rdylbu_r'}:
        return 'RdYlBu', lower.endswith('_r')
    # Generic fallback: use RdBu and infer reversal from _r suffix
    return 'RdBu', lower.endswith('_r')


def _correlation_label_for_method(corr_method='spearman'):
    """Axis / annotation label for Pearson or Spearman correlation."""
    method = str(corr_method).strip().lower()
    if method == 'pearson':
        return r"Pearson r"
    if method == 'spearman':
        return r"Spearman ρ"
    raise ValueError("corr_method must be 'spearman' or 'pearson'")


def _axis_ylim_with_padding(
    values,
    *,
    pad_fraction=2.0,
    floor_zero=False,
    symmetric=False,
):
    """
    Y-limits with ``pad_fraction`` padding (default 200%).

    If ``symmetric`` is True, use the peak absolute value and return
    ``[-bound, bound]`` so the series is centered on zero::

        bound = abs_peak + abs_peak * pad_fraction

    Otherwise extend independently above max and below min.
    """
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return (0.0, 1.0) if floor_zero else (-1.0, 1.0)

    if symmetric:
        abs_peak = float(np.nanmax(np.abs(v)))
        if abs_peak == 0.0:
            abs_peak = 0.5
        bound = abs_peak + abs_peak * pad_fraction
        return -bound, bound

    vmin, vmax = float(np.min(v)), float(np.max(v))
    ymax = vmax + abs(vmax) * pad_fraction
    ymin = vmin - abs(vmin) * pad_fraction
    if floor_zero:
        ymin = max(0.0, ymin)
    if ymin >= ymax:
        ymax += 1.0
    return ymin, ymax


def _prepare_grace_precip_correlation_inputs(
    grace_data,
    precip_data,
    *,
    use_residual=True,
    exclude_years=None,
    precip_mode='anomaly',
    aoi_geometry=None,
    compute=True,
):
    """Align GRACE/precip arrays the same way as per-pixel correlation analysis."""
    if exclude_years is None:
        exclude_years = [2017, 2018]

    if not grace_data.rio.crs:
        grace_data = grace_data.rio.write_crs("EPSG:4326")
    if not precip_data.rio.crs:
        precip_data = precip_data.rio.write_crs("EPSG:4326")
    try:
        grace_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
        precip_data.rio.set_spatial_dims(x_dim='lon', y_dim='lat', inplace=True)
    except Exception:
        pass

    if aoi_geometry is not None:
        if isinstance(aoi_geometry, gpd.GeoSeries):
            aoi_gdf = gpd.GeoDataFrame(geometry=aoi_geometry, crs=aoi_geometry.crs).to_crs("EPSG:4326")
        elif isinstance(aoi_geometry, gpd.GeoDataFrame):
            aoi_gdf = aoi_geometry.to_crs("EPSG:4326")
        else:
            aoi_gdf = gpd.GeoDataFrame(geometry=[aoi_geometry], crs="EPSG:4326")
        geom_clip = [mapping(geom.buffer(0)) for geom in aoi_gdf.geometry]
        try:
            grace_data = grace_data.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
            precip_data = precip_data.rio.clip(geom_clip, crs="EPSG:4326", drop=True)
        except Exception as e:
            _note(
                f"AOI clip failed ({e}); continuing unclipped. "
                "Hint: ensure DataArray CRS is EPSG:4326 and spatial dims are lon/lat."
            )

    if ('lat' in precip_data.dims and 'lon' in precip_data.dims
            and 'lat' in grace_data.dims and 'lon' in grace_data.dims):
        grace_lat_res = abs(float(grace_data.lat[1] - grace_data.lat[0])) if len(grace_data.lat) > 1 else 1.0
        precip_lat_res = abs(float(precip_data.lat[1] - precip_data.lat[0])) if len(precip_data.lat) > 1 else 1.0
        if precip_lat_res < grace_lat_res * 0.9:
            precip_data = precip_data.interp(lat=grace_data.lat, lon=grace_data.lon, method='nearest')

    _mode = str(precip_mode).strip().lower()
    if _mode not in {'month_to_month', 'anomaly', 'cumsum'}:
        raise ValueError("precip_mode must be one of {'month_to_month', 'anomaly', 'cumsum'}")

    if _mode == 'cumsum':
        climatology_full = _calculate_precip_climatology(precip_data)
        precip_monthly_anomaly_full = precip_data.groupby("time.month") - climatology_full
        precip_cumsum_full = precip_monthly_anomaly_full.cumsum(dim="time")
        if exclude_years:
            grace_aligned = grace_data.sel(time=~grace_data.time.dt.year.isin(exclude_years))
        else:
            grace_aligned = grace_data
        grace_aligned, precip_predictor = xr.align(grace_aligned, precip_cumsum_full, join="inner")
        precip_raw_aligned, _ = xr.align(precip_data, grace_aligned, join="inner")
    else:
        grace_aligned, precip_aligned = xr.align(grace_data, precip_data, join="inner")
        if exclude_years:
            grace_aligned = grace_aligned.sel(time=~grace_aligned.time.dt.year.isin(exclude_years))
            precip_aligned = precip_aligned.sel(time=~precip_aligned.time.dt.year.isin(exclude_years))
        grace_aligned, precip_aligned = xr.align(grace_aligned, precip_aligned, join="inner")
        precip_raw_aligned = precip_aligned
        if _mode == 'month_to_month':
            precip_predictor = precip_aligned
        else:
            climatology = _calculate_precip_climatology(precip_aligned)
            precip_predictor = precip_aligned.groupby("time.month") - climatology

    if len(grace_aligned.time) == 0:
        raise ValueError("No overlapping time periods after alignment.")
    if grace_aligned.shape[1] == 0 or grace_aligned.shape[2] == 0:
        raise ValueError("No overlapping spatial coverage after alignment.")

    time_values = grace_aligned.time.values
    if use_residual:
        def _decompose_pixel_calendar(grace_ts_array):
            return _decompose_grace_calendar(grace_ts_array, time_values)

        grace_processed = xr.apply_ufunc(
            _decompose_pixel_calendar,
            grace_aligned,
            input_core_dims=[['time']],
            output_core_dims=[['time']],
            vectorize=True,
        )
    else:
        grace_processed = grace_aligned

    climatology_plot = _calculate_precip_climatology(precip_raw_aligned)
    precip_monthly_anomaly = precip_raw_aligned.groupby("time.month") - climatology_plot
    precip_cumsum_anomaly = precip_monthly_anomaly.cumsum(dim="time")

    if compute:
        import dask
        to_compute = [grace_processed, precip_predictor, precip_raw_aligned, precip_cumsum_anomaly]
        try:
            grace_processed, precip_predictor, precip_raw_aligned, precip_cumsum_anomaly = dask.compute(*to_compute)
        except Exception:
            grace_processed = grace_processed.load()
            precip_predictor = precip_predictor.load()
            precip_raw_aligned = precip_raw_aligned.load()
            precip_cumsum_anomaly = precip_cumsum_anomaly.load()

    return {
        'grace_processed': grace_processed,
        'precip_predictor': precip_predictor,
        'precip_raw': precip_raw_aligned,
        'precip_cumsum_anomaly': precip_cumsum_anomaly,
        'time_values': time_values,
        'precip_mode': _mode,
        'use_residual': bool(use_residual),
        'exclude_years': list(exclude_years) if exclude_years else [],
    }


def _build_correlation_pixel_catalog(
    correlation_da,
    pvalue_da=None,
    lag_da=None,
    *,
    mask_non_significant=False,
    significance_level=0.05,
):
    """Tabular catalog of valid correlation grid cells with stable cell ids."""
    lats = np.asarray(correlation_da.lat.values, dtype=float)
    lons = np.asarray(correlation_da.lon.values, dtype=float)
    corr = np.asarray(correlation_da.values, dtype=float)
    pvals = np.asarray(pvalue_da.values, dtype=float) if pvalue_da is not None else None
    lags = np.asarray(lag_da.values, dtype=float) if lag_da is not None else None

    rows = []
    for i, lat in enumerate(lats):
        for j, lon in enumerate(lons):
            r = corr[i, j]
            if not np.isfinite(r):
                continue
            p = pvals[i, j] if pvals is not None else np.nan
            if mask_non_significant and pvals is not None:
                if not (np.isfinite(p) and p < float(significance_level)):
                    continue
            lag = lags[i, j] if lags is not None else np.nan
            rows.append({
                'cell_id': _encode_pixel_cell_id(i, j),
                'i': int(i),
                'j': int(j),
                'lat': float(lat),
                'lon': float(lon),
                'correlation': float(r),
                'pvalue': float(p) if np.isfinite(p) else np.nan,
                'optimal_lag_months': float(lag) if np.isfinite(lag) else np.nan,
            })
    catalog = pd.DataFrame(rows)
    if catalog.empty:
        raise ValueError("No valid correlation grid cells to plot.")
    return catalog


def _extract_pixel_timeseries_bundle(prepared_inputs, i, j):
    """Extract per-pixel time series used for interactive/publication plots."""
    grace_proc = prepared_inputs['grace_processed']
    precip_raw = prepared_inputs['precip_raw']
    precip_cum = prepared_inputs['precip_cumsum_anomaly']
    precip_pred = prepared_inputs['precip_predictor']

    grace_vals = np.asarray(grace_proc.isel(lat=i, lon=j).values, dtype=float)
    precip_monthly = np.asarray(precip_raw.isel(lat=i, lon=j).values, dtype=float)
    precip_cum_vals = np.asarray(precip_cum.isel(lat=i, lon=j).values, dtype=float)
    precip_pred_vals = np.asarray(precip_pred.isel(lat=i, lon=j).values, dtype=float)
    times = pd.to_datetime(grace_proc.time.values)

    grace_series = pd.Series(grace_vals, index=times, name='grace_residual')
    precip_monthly_series = pd.Series(precip_monthly, index=times, name='precip_monthly')
    precip_cum_series = pd.Series(precip_cum_vals, index=times, name='precip_cumsum_anomaly')
    precip_pred_series = pd.Series(precip_pred_vals, index=times, name='precip_predictor')

    return {
        'grace_residual': grace_series,
        'precip_monthly': precip_monthly_series,
        'precip_cumsum_anomaly': precip_cum_series,
        'precip_predictor': precip_pred_series,
        'lat': float(grace_proc.lat.values[i]),
        'lon': float(grace_proc.lon.values[j]),
    }


def _pixel_timeseries_title(row, corr_label=r"Spearman ρ"):
    """Single-line title: lat, lon, cell id, correlation, p-value."""
    parts = [
        f"Lat {row['lat']:.2f}°",
        f"Lon {row['lon']:.2f}°",
        str(row['cell_id']),
    ]
    if np.isfinite(row.get('correlation', np.nan)):
        parts.append(f"{corr_label} = {row['correlation']:.2f}")
    if np.isfinite(row.get('pvalue', np.nan)):
        p = row['pvalue']
        parts.append(f"p = {p:.3f}" if p >= 0.001 else "p < 0.001")
    return ", ".join(parts)


def _build_interactive_pixel_timeseries_figure(
    row,
    ts_bundle,
    *,
    corr_label=r"Spearman ρ",
    title=None,
    ts_figsize=(10, 4),
    pad_fraction=2.0,
):
    """Plotly figure matching publication triple-axis time-series style."""
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError(
            "Plotly is required for interactive pixel time series. "
            "Install with `pip install plotly`."
        ) from exc

    grace_color = '#1f4e79'
    cpa_color = '#b22222'
    precip_color = '#5b9bd5'

    grace = ts_bundle['grace_residual']
    precip_cum = ts_bundle['precip_cumsum_anomaly']
    precip_mon = ts_bundle['precip_monthly']
    valid_precip = precip_mon.dropna()

    if title is None:
        title = _pixel_timeseries_title(row, corr_label)

    ymin_g, ymax_g = _axis_ylim_with_padding(
        grace.values, pad_fraction=pad_fraction, symmetric=True
    )
    ymin_c, ymax_c = _axis_ylim_with_padding(
        precip_cum.values, pad_fraction=pad_fraction, symmetric=True
    )
    ymin_p, ymax_p = _axis_ylim_with_padding(
        valid_precip.values, pad_fraction=pad_fraction, floor_zero=True
    )

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=grace.index,
        y=grace.values,
        mode='lines',
        name='GRACE residual',
        line=dict(color=grace_color, width=1.8),
        yaxis='y',
        hovertemplate="%{x|%Y-%m}<br>GRACE: %{y:.2f} cm<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=precip_cum.index,
        y=precip_cum.values,
        mode='lines',
        name='CPA',
        line=dict(color=cpa_color, width=1.8),
        yaxis='y2',
        hovertemplate="%{x|%Y-%m}<br>CPA: %{y:.2f} mm<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=precip_mon.index,
        y=precip_mon.values,
        name='Monthly precipitation',
        marker=dict(color=f'rgba(91, 155, 213, 0.72)', line=dict(width=0)),
        yaxis='y3',
        hovertemplate="%{x|%Y-%m}<br>Precip: %{y:.2f} mm<extra></extra>",
    ))

    # x domain 0–0.88 leaves room for outer right y-axis (matches publication layout)
    plot_x_center = 0.44
    ts_w = int(float(ts_figsize[0]) * 100)
    ts_h = int(float(ts_figsize[1]) * 100)

    fig.update_layout(
        title=dict(
            text=title,
            x=plot_x_center,
            xanchor='center',
            y=0.98,
            yanchor='top',
            font=dict(size=11),
            pad=dict(t=0, b=0),
        ),
        template='plotly_white',
        width=ts_w,
        height=ts_h,
        autosize=False,
        hovermode='x unified',
        legend=dict(
            x=0.01,
            y=0.98,
            xanchor='left',
            yanchor='top',
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.92)',
        ),
        margin=dict(l=58, r=105, t=30, b=40),
        xaxis=dict(
            title='',
            domain=[0.0, 0.88],
            tickformat='%Y',
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text='GRACE residual (cm)', font=dict(color=grace_color, size=11)),
            tickfont=dict(color=grace_color, size=10),
            range=[ymin_g, ymax_g],
            showgrid=True,
            gridcolor='rgba(0,0,0,0.08)',
            zeroline=True,
            zerolinecolor='rgba(0,0,0,0.25)',
        ),
        yaxis2=dict(
            title=dict(text='CPA (mm)', font=dict(color=cpa_color, size=11)),
            tickfont=dict(color=cpa_color, size=10),
            overlaying='y',
            side='right',
            range=[ymin_c, ymax_c],
            showgrid=False,
            zeroline=False,
        ),
        yaxis3=dict(
            title=dict(text='Monthly precip (mm)', font=dict(color=precip_color, size=11)),
            tickfont=dict(color=precip_color, size=10),
            overlaying='y',
            side='right',
            anchor='free',
            position=0.98,
            range=[ymin_p, ymax_p],
            showgrid=False,
            zeroline=False,
        ),
    )
    return fig


def plot_grace_precip_correlation_interactive_map(
    grace_data,
    precip_data,
    correlation_da,
    pvalue_da=None,
    lag_da=None,
    *,
    use_residual=True,
    exclude_years=None,
    precip_mode='anomaly',
    aoi_geometry=None,
    vmin=-1.0,
    vmax=1.0,
    cmap='RdBu',
    corr_method='spearman',
    label=None,
    mask_non_significant=False,
    significance_level=0.05,
    title=None,
    map_figsize=(10, 4),
    ts_figsize=(10, 4),
    save_map_path=None,
    save_ts_dir=None,
    compute_inputs=True,
):
    """
    Interactive GRACE–precipitation correlation map (no classification).

    Hover shows cell id, coordinates, correlation, p-value, and optimal lag.
    Click a grid cell to update a single interactive time-series figure below
    the map (GRACE residual, cumulative precipitation anomaly, and monthly
    precipitation bars on one panel with three y-axes).

    Requires Plotly and, for click events in Jupyter, ``ipywidgets``.

    Parameters
    ----------
    grace_data, precip_data : xarray.DataArray
        Original GRACE and precipitation cubes used for correlation.
    correlation_da : xarray.DataArray
        Per-pixel correlation grid (e.g. from
        ``calculate_grace_precip_correlation_per_pixel``).
    pvalue_da, lag_da : xarray.DataArray, optional
        Matching p-value and optimal-lag grids.
    use_residual, exclude_years, precip_mode, aoi_geometry
        Must match the settings used to compute ``correlation_da`` so that
        click-through time series are consistent with the map.
    vmin, vmax : float
        Correlation color scale limits (default ±1).
    cmap : str, default='RdBu'
        Diverging matplotlib-style colormap name. ``RdBu`` maps blue to
        positive correlations and red to negative; append ``_r`` to reverse.
    corr_method : {'spearman', 'pearson'}, default='spearman'
        Correlation method used to compute ``correlation_da``. Sets the
        default colorbar / annotation label when ``label`` is None.
    label : str, optional
        Colorbar and annotation label. Defaults to Spearman ρ or Pearson r
        from ``corr_method``.
    map_figsize : tuple, default=(10, 4)
        Map figure size in inches (scaled ×100 for Plotly pixels).
    ts_figsize : tuple, default=(10, 4)
        Time-series figure size in inches when a cell is clicked.
    mask_non_significant : bool, default=False
        If True, only plot cells with ``pvalue_da < significance_level``.
    save_map_path : str, optional
        If set, save the map as HTML (and PNG if kaleido is available).
    save_ts_dir : str, optional
        Directory where clicked time-series figures are saved as HTML.
    compute_inputs : bool, default=True
        If True, load aligned GRACE/precip arrays into memory once for fast
        click-through time series.

    Returns
    -------
    dict
        ``fig`` (Plotly FigureWidget or Figure), ``ts_output`` (ipywidgets
        Output slot), ``pixel_catalog`` (DataFrame), ``prepared_inputs``,
        and ``meta``.
    """
    try:
        import plotly.graph_objects as go
    except Exception as exc:
        raise ImportError(
            "Plotly is required for the interactive correlation map. "
            "Install with `pip install plotly`."
        ) from exc

    if exclude_years is None:
        exclude_years = [2017, 2018]

    method = str(corr_method).strip().lower()
    if method not in {'spearman', 'pearson'}:
        raise ValueError("corr_method must be 'spearman' or 'pearson'")
    if label is None:
        label = _correlation_label_for_method(method)

    prepared_inputs = _prepare_grace_precip_correlation_inputs(
        grace_data,
        precip_data,
        use_residual=use_residual,
        exclude_years=exclude_years,
        precip_mode=precip_mode,
        aoi_geometry=aoi_geometry,
        compute=compute_inputs,
    )

    pixel_catalog = _build_correlation_pixel_catalog(
        correlation_da,
        pvalue_da=pvalue_da,
        lag_da=lag_da,
        mask_non_significant=mask_non_significant,
        significance_level=significance_level,
    )

    lons = pixel_catalog['lon'].values
    lats = pixel_catalog['lat'].values
    corr_vals = pixel_catalog['correlation'].values
    custom = np.column_stack([
        pixel_catalog['cell_id'].values,
        pixel_catalog['lat'].values,
        pixel_catalog['lon'].values,
        pixel_catalog['pvalue'].values,
        pixel_catalog['optimal_lag_months'].values,
    ])

    lon_span = float(np.nanmax(lons) - np.nanmin(lons)) if len(lons) else 10.0
    lat_span = float(np.nanmax(lats) - np.nanmin(lats)) if len(lats) else 10.0
    marker_size = max(6, min(14, 220 / max(len(pixel_catalog), 1) ** 0.35))
    colorscale_name, reversescale = _plotly_colorscale_from_matplotlib_cmap(cmap)

    map_w = int(float(map_figsize[0]) * 100)
    map_h = int(float(map_figsize[1]) * 100)
    # Geo ends at x=0.94; colorbar starts immediately to its right (paper coords)
    _geo_x_hi = 0.94
    _cbar_x = _geo_x_hi

    scatter = go.Scattergeo(
        lon=lons,
        lat=lats,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=corr_vals,
            cmin=float(vmin),
            cmax=float(vmax),
            cmid=0.0,
            colorscale=colorscale_name,
            reversescale=reversescale,
            colorbar=dict(
                title=dict(text=label, side='right'),
                orientation='v',
                len=0.88,
                y=0.47,
                yanchor='middle',
                thickness=12,
                x=_cbar_x,
                xanchor='left',
                xpad=0,
                outlinewidth=0,
            ),
            line=dict(width=0.3, color='rgba(0,0,0,0.35)'),
        ),
        customdata=custom,
        hovertemplate=(
            "Cell ID: %{customdata[0]}<br>"
            "Lat: %{customdata[1]:.2f}°<br>"
            "Lon: %{customdata[2]:.2f}°<br>"
            f"{label}: %{{marker.color:.2f}}<br>"
            "p: %{customdata[3]:.3f}<br>"
            "Lag: %{customdata[4]:.0f} mo<extra></extra>"
        ),
        name='Grid cells',
    )

    map_title = title or "GRACE–precipitation correlation (click a cell for time series)"
    layout = go.Layout(
        title=dict(
            text=map_title,
            x=0.5,
            xanchor='center',
            y=0.98,
            yanchor='top',
            font=dict(size=12),
            pad=dict(t=0, b=0),
        ),
        template='plotly_white',
        width=map_w,
        height=map_h,
        autosize=False,
        geo=dict(
            projection_type='natural earth',
            showcountries=True,
            showland=True,
            landcolor='#e8e8e8',
            oceancolor='#d6eef8',
            lonaxis=dict(range=[float(np.nanmin(lons) - 1), float(np.nanmax(lons) + 1)]),
            lataxis=dict(range=[float(np.nanmin(lats) - 1), float(np.nanmax(lats) + 1)]),
            fitbounds='locations',
            domain=dict(x=[0.0, _geo_x_hi], y=[0.0, 0.94]),
        ),
        margin=dict(l=0, r=52, t=28, b=0),
    )

    widget_mode = True
    try:
        fig = go.FigureWidget(data=[scatter], layout=layout)
    except Exception:
        widget_mode = False
        fig = go.Figure(data=[scatter], layout=layout)

    # Stack map + time series with minimal vertical gap (ipywidgets Output padding)
    ts_output = None
    ts_fig_widget = {'fig': None}
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display

        zero_layout = widgets.Layout(width=f'{map_w}px', margin='0px', padding='0px')
        map_output = widgets.Output(layout=zero_layout)
        ts_output = widgets.Output(layout=widgets.Layout(width=f'{map_w}px', margin='0px', padding='0px'))
        panel = widgets.VBox(
            [map_output, ts_output],
            layout=widgets.Layout(width='100%', margin='0px', padding='0px'),
        )
        with map_output:
            display(fig)
        display(panel)
    except Exception:
        from IPython.display import display, clear_output
        display(fig)
        clear_output = None

    save_ts_path = Path(save_ts_dir) if save_ts_dir else None
    if save_ts_path is not None:
        save_ts_path.mkdir(parents=True, exist_ok=True)

    def _show_timeseries_for_index(idx):
        row = pixel_catalog.iloc[int(idx)]
        ts_bundle = _extract_pixel_timeseries_bundle(
            prepared_inputs, int(row['i']), int(row['j'])
        )
        ts_fig = _build_interactive_pixel_timeseries_figure(
            row, ts_bundle, corr_label=label, ts_figsize=ts_figsize
        )
        if save_ts_path is not None:
            out_html = save_ts_path / f"{row['cell_id']}_timeseries.html"
            ts_fig.write_html(str(out_html))
            print(f"Saved time series: {out_html}")
            try:
                out_png = save_ts_path / f"{row['cell_id']}_timeseries.png"
                ts_fig.write_image(str(out_png), scale=2)
                print(f"Saved time series: {out_png}")
            except Exception:
                pass

        if ts_output is not None and clear_output is not None:
            with ts_output:
                clear_output(wait=True)
                try:
                    fw = go.FigureWidget(ts_fig)
                    ts_fig_widget['fig'] = fw
                    display(fw)
                except Exception:
                    display(ts_fig)
        elif ts_fig_widget['fig'] is not None and isinstance(ts_fig_widget['fig'], go.FigureWidget):
            with ts_fig_widget['fig'].batch_update():
                ts_fig_widget['fig'].data = ()
                for tr in ts_fig.data:
                    ts_fig_widget['fig'].add_trace(tr)
                ts_fig_widget['fig'].layout.update(ts_fig.layout)
        else:
            try:
                fw = go.FigureWidget(ts_fig)
                ts_fig_widget['fig'] = fw
                display(fw)
            except Exception:
                display(ts_fig)
        return ts_fig

    def _show_timeseries_for_cell_id(cid):
        match = pixel_catalog.loc[pixel_catalog['cell_id'] == str(cid)]
        if match.empty:
            raise ValueError(f"cell_id {cid!r} not found in pixel_catalog.")
        row_pos = pixel_catalog.index.get_loc(match.index[0])
        if isinstance(row_pos, slice):
            row_pos = row_pos.start
        elif isinstance(row_pos, np.ndarray):
            row_pos = int(row_pos[0])
        return _show_timeseries_for_index(int(row_pos))

    if widget_mode:
        def _on_click(trace, points, selector):
            if not points.point_inds:
                return
            _show_timeseries_for_index(points.point_inds[0])

        fig.data[0].on_click(_on_click)
    else:
        print(
            "FigureWidget unavailable (install ipywidgets). "
            "Use plot_grace_precip_pixel_timeseries_publication(cell_id, ...) "
            "with a cell id from pixel_catalog."
        )

    if save_map_path:
        out = Path(save_map_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if out.suffix.lower() == '.html':
            fig.write_html(str(out))
        else:
            fig.write_html(str(out.with_suffix('.html')))
            try:
                fig.write_image(str(out), scale=2)
            except Exception as exc:
                print(f"Map PNG save skipped ({exc}). HTML was written instead.")
        print(f"Saved interactive map: {out}")

    meta = {
        'n_cells': int(len(pixel_catalog)),
        'widget_mode': bool(widget_mode),
        'vmin': float(vmin),
        'vmax': float(vmax),
        'cmap': cmap,
        'corr_method': method,
        'label': label,
        'precip_mode': precip_mode,
        'use_residual': bool(use_residual),
    }
    return {
        'fig': fig,
        'ts_output': ts_output,
        'ts_fig_widget': ts_fig_widget.get('fig'),
        'pixel_catalog': pixel_catalog,
        'prepared_inputs': prepared_inputs,
        'meta': meta,
        'show_timeseries_for_cell_id': _show_timeseries_for_cell_id,
    }


def plot_arid_watersheds(arid_watersheds, value_col='ari_sav', n_labels=10, 
id_col="subbasin_id", path=None, figsize=(12, 6), label=r"Value (mm yr$^{-1}$)"):
    """Plot arid watersheds with specified value column."""
    # Ensure WGS84 projection
    # OPTIMIZED: Removed unnecessary .copy() - .to_crs() already returns a copy
    gdf = arid_watersheds.to_crs("EPSG:4326")
    # Multiply values by 10 to convert to mm
    gdf[value_col] = gdf[value_col] * 10
    
    # Get extent with 2° buffer
    bounds = gdf.total_bounds  # [minx, miny, maxx, maxy]
    buffered_extent = [
        bounds[0] - 1, bounds[2] + 1,  # min lon, max lon
        bounds[1] - 2, bounds[3] + 2   # min lat, max lat
    ]

    # Color normalization - divergent centered at 0
    vmin = gdf[value_col].min()
    vmax = gdf[value_col].max()
    # Center at 0 for divergent colormap
    vcenter = 0.0
    vmax_abs = max(abs(vmin), abs(vmax))
    
    # Use TwoSlopeNorm for divergent colormap centered at 0
    norm = mcolors.TwoSlopeNorm(vmin=-vmax_abs, vcenter=vcenter, vmax=vmax_abs)
    cmap = plt.cm.RdYlBu_r  # Divergent colormap (red for negative, blue for positive)

    # Plot
    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    ax.set_extent(buffered_extent, crs=ccrs.PlateCarree())

    ax.set_extent(buffered_extent, crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.LAND, facecolor='lightgrey')
    ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
    ax.add_feature(cfeature.BORDERS, linestyle='--', edgecolor='black', linewidth=0.5)
    ax.coastlines()

    gdf.plot(
        column=value_col,
        cmap=cmap,
        norm=norm,
        ax=ax,
        edgecolor='black',
        linewidth=0.5,
    )

    # Add gridlines
    gl = ax.gridlines(
    draw_labels=True,
    linewidth=0.5,
    color='gray',
    alpha=0.7,
    linestyle='--'
    )
    
    # Format ticks
    gl.top_labels = False   # Hide top labels
    gl.right_labels = False # Hide right labels
    gl.xlocator = mticker.FixedLocator(np.arange(-180, 181, 20))  # lon every 20°
    gl.ylocator = mticker.FixedLocator(np.arange(-90, 91, 20))    # lat every 10°
    # Make latitude labels horizontal to save space
    gl.xlabel_style = {'rotation': 0, 'size': _MAP_GRID_LABEL_FONTSIZE}
    gl.ylabel_style = {'rotation': 90, 'size': _MAP_GRID_LABEL_FONTSIZE}

    # Add labels to largest N aquifers
    largest = gdf.nlargest(n_labels, value_col)

    # OPTIMIZED: Use itertuples() instead of iterrows() (small performance gain for plotting)
    for row in largest.itertuples():
        centroid = row.geometry.centroid
        subbasin_id = getattr(row, id_col, None)
        ax.text(
            centroid.x,
            centroid.y,
            str(subbasin_id),
            fontsize=7,
            fontweight='bold',
            ha='center',
            va='center',
            transform=ccrs.PlateCarree(),
            bbox=dict(boxstyle="round",pad=0.2, fc="white", ec="black", lw=0.3)
        )
        
    # Colorbar (using vmin and vmax calculated from value_col above)
    sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    sm._A = []
    cbar = fig.colorbar(
        sm, ax=ax, orientation='vertical',
        pad=_MAP_CBAR_PAD, fraction=_MAP_CBAR_FRACTION,
        shrink=_MAP_CBAR_SHRINK_V, extend="max",
    )
    cbar.set_label(label, fontsize=10)
    # Let matplotlib automatically determine ticks based on vmin/vmax
    cbar.ax.tick_params(labelsize=10)
    cbar.ax.grid(True, linestyle="--", alpha=0.5)

    plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)
    if path:
        plt.savefig(path, dpi=500)
    plt.show()


def sort_and_index_by_area(gdf, area_column='aq_km2'):
    """Sort GeoDataFrame by area column and add subbasin_id index."""
    gdf = gdf.sort_values(by=area_column, ascending=False).reset_index(drop=True)
    gdf["subbasin_id"] = gdf["aq_id"].astype(int) + 1
    gdf=gdf.drop(columns=['aq_id'])
    return gdf


def process_grace_data(grace_file, aoi_geometry, time_range, 
                variable_name='lwe_thickness', exclude_years=[2017, 2018], 
                land_mask_file=None, apply_scaling_factor=False):
    """
    Process GRACE data: load, apply land mask, coarsen, clip, and interpolate.
    
    Parameters
    ----------
    grace_file : str
        Path to GRACE .nc file
    aoi_geometry : geometry
        Area of interest for clipping
    time_range : pd.DatetimeIndex
        Time range for interpolation
    variable_name : str, default='lwe_thickness'
        Variable name in the GRACE file
    exclude_years : list, default=[2017, 2018]
        Years to exclude from the time series
    land_mask_file : str, optional
        Path to separate land mask file (e.g., for CSR). If None, looks for 'land_mask' variable in grace_file.
    apply_scaling_factor : bool, default=False
        Apply scaling factor to the GRACE data
    Returns
    -------
    xr.DataArray
        Processed GRACE data
    """
    grace_path = Path(grace_file)
    if not grace_path.is_file():
        _raise_ctx(FileNotFoundError, f"GRACE file not found: {_rel(grace_file)}")
    _note(f"GRACE: {_rel(grace_file)}")
    # Load GRACE dataset
    grace_data = xr.open_dataset(grace_file)
    if variable_name not in grace_data:
        available = list(grace_data.data_vars)
        _raise_ctx(
            KeyError,
            f"Variable {variable_name!r} not in GRACE dataset {_rel(grace_file)}. "
            f"Available: {available}",
        )

    # Fix Time Format (Handles both "Units" and "units")
    time_attrs = grace_data['time'].attrs
    time_units = time_attrs.get('Units', time_attrs.get('units', None))  # Handles both cases

    if time_units and 'days since' in time_units:
        converted_time = num2date(grace_data['time'].values, time_units)
        grace_data['time'] = xr.DataArray(
            np.array(converted_time, dtype="datetime64[ns]"),  # Ensures numpy datetime64 format
            dims=["time"]
        )
    else:
        # Check if time is in YYYYMM format (numeric, like 200301. for January 2003)
        time_values = grace_data['time'].values
        if time_values.dtype.kind in 'fiu':  # numeric types
            # Check if values are in reasonable YYYYMM range (190000 to 210000)
            time_int = np.array([int(t) for t in time_values])
            if np.all((time_int >= 190000) & (time_int <= 210000)) and np.all((time_int % 100 >= 1) & (time_int % 100 <= 12)):
                # Convert YYYYMM format to datetime64 end-of-month dates
                years = time_int // 100
                months = time_int % 100
                dates = pd.to_datetime({'year': years, 'month': months, 'day': 1})
                end_of_month = dates + pd.offsets.MonthEnd(0)
                datetime_times = end_of_month.values.astype('datetime64[ns]')
                # Update the time coordinate in the dataset
                grace_data = grace_data.assign_coords(time=datetime_times)
    
    grace_lwe = grace_data[variable_name]  # Reload after time fix
    
    # ============================================================================
    # STEP: Apply land mask BEFORE coarsening
    # ============================================================================
    land_mask = None
    
    # Try to load land mask from separate file (e.g., CSR)
    if land_mask_file is not None:
        try:
            _note(f"land mask: {_rel(land_mask_file)}")
            land_mask_data = xr.open_dataset(land_mask_file)
            # Check for different possible land mask variable names
            if 'LO_val' in land_mask_data:  # CSR land mask variable
                land_mask = land_mask_data['LO_val']
                _note("found land mask variable: LO_val")
            elif 'land_mask' in land_mask_data:
                land_mask = land_mask_data['land_mask']
                _note("found land mask variable: land_mask")
            else:
                print(f"  Warning: Land mask variable not found. Available variables: {list(land_mask_data.data_vars)}")
        except Exception as e:
            print(f"  Warning: Could not load land mask file: {e}")
    
    # If no separate file, try to get land_mask from the main GRACE file
    if land_mask is None and 'land_mask' in grace_data:
        print("Loading land mask from GRACE data file")
        land_mask = grace_data['land_mask']
    
    # Apply land mask: mask ocean pixels (land_mask==0 or NaN) before coarsening
    if land_mask is not None:
        print(f"  Applying land mask to exclude ocean pixels...")
        # Ensure land_mask has same lat/lon dims as grace_lwe
        if land_mask.lat.shape == grace_lwe.lat.shape and land_mask.lon.shape == grace_lwe.lon.shape:
            # Mask where land_mask is 0 or NaN (ocean)
            grace_lwe = grace_lwe.where(land_mask > 0)
            n_masked = (land_mask == 0).sum().item() if hasattr((land_mask == 0).sum(), 'item') else 0
            print(f"  Masked {n_masked} ocean pixels (set to NaN)")
        else:
            print(f"  Warning: Land mask dimensions don't match GRACE data. Skipping masking.")
    else:
        print("  Warning: No land mask found. Proceeding without masking.")

    # Ensure latitude and longitude are evenly spaced before coarsening
    grace_lwe = grace_lwe.sortby("lat")  # Ensure lat is sorted in increasing order    
    grace_lwe = grace_lwe.sortby("lon")  # Ensure lon is sorted in increasing order    

    n_lat, n_lon = len(grace_lwe.lat), len(grace_lwe.lon)
    if n_lat < 2 or n_lon < 2:
        _raise_ctx(
            ValueError,
            f"GRACE lat/lon must have length >= 2 to estimate resolution "
            f"(got lat={n_lat}, lon={n_lon})",
        )

    # Determine spatial resolution (difference between consecutive lat/lon values)
    delta_lat = np.abs(grace_lwe.lat.values[1] - grace_lwe.lat.values[0])
    delta_lon = np.abs(grace_lwe.lon.values[1] - grace_lwe.lon.values[0])

    # Determine coarsening factor for 1-degree resolution
    if delta_lat == 0.5 and delta_lon == 0.5:
        coarsen_factor = 2  # Coarsen by 2x2 to get 1-degree
    elif delta_lat == 0.25 and delta_lon == 0.25:
        coarsen_factor = 4  # Coarsen by 4x4 to get 1-degree
    elif delta_lat == 1 and delta_lon == 1:
        coarsen_factor = 1  # Coarsen by 1x1 to get 1-degree
    else:
        print(f"⚠️ Warning: Unexpected resolution Δlat={delta_lat}, Δlon={delta_lon}. Defaulting to interpolation.")
        coarsen_factor = None  # No coarsening, will use interpolation

    # Apply coarsening if determined
    # CRITICAL: Exclude coarsened pixels that contain ANY ocean/NaN pixel
    if coarsen_factor is not None and coarsen_factor > 1:
        print(f"  Coarsening by {coarsen_factor}×{coarsen_factor} to 1-degree resolution...")
        
        # Strategy: Count valid pixels per coarse cell, then mask where count < expected
        # This ensures any cell with even ONE ocean/NaN pixel is excluded
        
        # For a representative time slice, count valid pixels per coarse cell
        # We use the first time step as reference (land/ocean mask is constant)
        valid_mask = grace_lwe.isel(time=0).notnull().astype(float)
        valid_count = valid_mask.coarsen(
            lat=coarsen_factor, 
            lon=coarsen_factor, 
            boundary="trim"
        ).sum()
        
        # Expected count if all pixels in the coarse cell are valid
        expected_count = coarsen_factor ** 2
        
        # Create mask: True where ALL pixels in the coarse cell are valid (no ocean contamination)
        all_land_mask = valid_count == expected_count
        
        # Apply standard mean coarsening to data
        grace_lwe_coarse = grace_lwe.coarsen(
            lat=coarsen_factor, 
            lon=coarsen_factor, 
            boundary="trim"
        ).mean()
        
        # Apply the "all land" mask: set ocean-contaminated pixels to NaN
        grace_lwe_coarse = grace_lwe_coarse.where(all_land_mask)
        
        n_total_coarse = all_land_mask.size
        n_land_only = all_land_mask.sum().item()
        n_contaminated = n_total_coarse - n_land_only
        print(f"  Coarsened grid: {n_total_coarse} total pixels, {n_land_only} pure-land, {n_contaminated} ocean-contaminated (excluded)")
        
        grace_lwe = grace_lwe_coarse
    
    # Adjust longitude to -180 to 180 (if necessary)
    if grace_lwe.lon.max() > 180:
        grace_lwe = grace_lwe.assign_coords(
            lon=((grace_lwe.lon + 180) % 360 - 180).round(3)
        ).sortby('lon')

    if apply_scaling_factor:
        scaling_factor = grace_data.get('scale_factor', None)
    else:
        scaling_factor = None

    if scaling_factor is not None:
        # Ensure longitude adjustment
        if scaling_factor.lon.max() > 180:
            scaling_factor = scaling_factor.assign_coords(
                lon=((scaling_factor.lon + 180) % 360 - 180).round(3)
            ).sortby('lon')

        # Apply same coarsening to scaling factor
        if coarsen_factor is not None:
            scaling_factor = scaling_factor.coarsen(lat=coarsen_factor, lon=coarsen_factor, boundary="trim").mean()

        # Apply scaling factor
        grace_lwe_scaled = grace_lwe * scaling_factor
    else:
        grace_lwe_scaled = grace_lwe  # If no scaling factor, use original data

    # Set CRS and spatial dimensions
    grace_lwe_scaled.rio.write_crs("EPSG:4326", inplace=True)
    grace_lwe_scaled.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)

    # Clip GRACE data to the AOI region (failures still raise; clearer context)
    try:
        grace_clipped = grace_lwe_scaled.rio.clip(aoi_geometry, crs="EPSG:4326", drop=True)
    except Exception as e:
        _raise_ctx(
            ValueError,
            f"Failed to clip GRACE to AOI ({e}). "
            "Hint: ensure CRS is EPSG:4326 and spatial dims are lon/lat.",
            cause=e,
        )
    grace_clipped.rio.write_crs("EPSG:4326", inplace=True)

    # Interpolate to regular monthly intervals
    grace_regular = grace_clipped.interp(time=time_range)
    grace_regular = grace_regular.interpolate_na(dim="time", method="linear")

    # Exclude specific years
    grace_filtered = grace_regular.sel(time=~grace_regular.time.dt.year.isin(exclude_years))

    return grace_filtered


def plot_timeseries_with_precip(
    series_list,
    labels,
    colors=None,
    rainfall=None,
    plot_precip=False,
    title="TWS and Rainfall Time Series",
    ylabel="TWS (cm)",
    precip_label="Rainfall (mm)",
    figsize=(10, 4)
):
    """Plot multiple time series with optional rainfall on a shared time axis."""
    # Convert inputs to pandas Series
    series_list = [s.to_series() if hasattr(s, "to_series") else s for s in series_list]
    rainfall_series = rainfall.mean(dim=("lat", "lon")).to_series() if hasattr(rainfall, "mean") else rainfall

    if colors is None:
        colors = plt.cm.tab10.colors  # Default color cycle

    # Create figure and axis
    fig, ax1 = plt.subplots(figsize=figsize)

    # Plot each TWS time series
    lines = []
    for i, (series, label) in enumerate(zip(series_list, labels)):
        line, = ax1.plot(series.index, series.values, label=label, color=colors[i % len(colors)], linewidth=2)
        lines.append(line)

    # Axis formatting
    ax1.set_title(title, fontsize=14)
    ax1.set_ylabel(ylabel, fontsize=14)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    plt.xticks(rotation=45)

    # Set X limits based on rainfall if present, else based on first series
    if plot_precip and rainfall_series is not None:
        x_min = rainfall_series.index.min() - pd.DateOffset(years=1)
        x_max = rainfall_series.index.max() + pd.DateOffset(years=1)
        ax1.set_ylim(series_list[0].min() - 2, series_list[0].max() + 5)
    else:
        x_min = series_list[0].index.min() - pd.DateOffset(years=1)
        x_max = series_list[0].index.max() + pd.DateOffset(years=1)

    ax1.set_xlim(x_min, x_max)

    # Add rainfall bars if enabled
    if plot_precip and rainfall_series is not None:
        ax2 = ax1.twinx()
        bars = ax2.bar(
            rainfall_series.index,
            rainfall_series.values,
            label=precip_label,
            color="#c9630a",
            alpha=0.9,
            width=40
        )
        ax2.set_ylim(rainfall_series.min(), rainfall_series.max() + 100)
        ax2.invert_yaxis()
        ax2.set_ylabel(precip_label, fontsize=14)
        lines.append(bars)

    # Combine legend entries
    legend_labels = labels + ([precip_label] if plot_precip and rainfall_series is not None else [])
    ax1.legend(lines, legend_labels, fontsize=12, loc='lower left')

    plt.tight_layout()
    plt.show()


def process_predictor_fine(precip_fine, data_path, variable_name, aoi_geometry, time_range, exclude_years=[2017, 2018]):
    """Process fine-resolution predictor data (e.g., precipitation)."""
    if exclude_years is not None and len(exclude_years) > 0:
        if not getattr(process_predictor_fine, "_exclude_years_noted", False):
            _note(
                "process_predictor_fine: exclude_years is currently unused "
                f"(received {list(exclude_years)}); years are not filtered here"
            )
            process_predictor_fine._exclude_years_noted = True
    # Load data lazily
    data = xr.open_dataset(data_path, engine="zarr", chunks={})  # no eager chunks

    # Rename dimensions
    rename_dict = {}
    if "X" in data.dims:
        rename_dict["X"] = "lon"
    if "Y" in data.dims:
        rename_dict["Y"] = "lat"
    if rename_dict:
        data = data.rename(rename_dict)

    predictor = data[variable_name]

    # Interpolate spatially to match GRACE grid
    predictor = predictor.interp(lat=precip_fine.lat, lon=precip_fine.lon)
    predictor = predictor.transpose("time", "lat", "lon")

    # Clip using AOI
    predictor.rio.write_crs("EPSG:4326", inplace=True)
    predictor.rio.set_spatial_dims(y_dim="lat", x_dim="lon", inplace=True)
    predictor = predictor.rio.clip(aoi_geometry, crs="EPSG:4326", drop=True)

    # Now chunk the array to allow interpolation to work efficiently
    predictor = predictor.chunk({"time": 100, "lat": 100, "lon": 100})

    predictor = predictor.resample(time='ME').mean().interp(time=time_range, method="linear")
    predictor = predictor.interpolate_na(dim="time", method="linear")
    gc.collect()

    predictor_full = predictor.persist()

    return predictor_full

def subbasin_trend_analysis(dataarrays, labels, gdf, id_column="subbasin_id"):
    """Calculate trend analysis for each subbasin."""
    # OPTIMIZED: Removed unnecessary .copy(), only copy if modifying original
    # OPTIMIZED: Move CRS operations outside loop (done once per dataarray)
    for var in dataarrays:
        var.rio.write_crs("EPSG:4326", inplace=True)

    for da, label in zip(dataarrays, labels):
        trend_list = []
        n_failed = 0

        # OPTIMIZED: Use enumerate with geometry iterator instead of iterrows()
        for idx, geom in enumerate(tqdm(gdf.geometry, desc=f"Processing {label}", total=len(gdf))):
            try:
                # OPTIMIZED: Use helper function for geometry conversion
                da_clipped = da.rio.clip(_geometry_to_clip_format(geom), all_touched=False, drop=False)
                ts_mean = da_clipped.mean(dim=["lat", "lon"], skipna=True)

                if np.all(np.isnan(ts_mean)):
                    trend = np.nan
                else:
                    time_index = np.arange(ts_mean.shape[0])
                    trend = linregress(time_index, ts_mean.values).slope

                trend_list.append(trend)

            except Exception:
                n_failed += 1
                trend_list.append(np.nan)

        _summarize_skipped(f"{label} subbasin trends", n_failed, len(gdf))

        # Add trend column to GeoDataFrame
        gdf[f"{label}_trend"] = trend_list

    return gdf


def add_average_annual_precipitation(gdf, rainfall_da, date_dim="time", id_col="subbasin_id", new_col="avg_annual_precip"):
    """Add average annual precipitation to GeoDataFrame."""
    # OPTIMIZED: Removed unnecessary .copy() calls - only copy if we modify original
    # OPTIMIZED: Move CRS operation outside loop
    rainfall_da.rio.write_crs("EPSG:4326", inplace=True).persist()  # Ensure CRS is set

    avg_annual_precip_list = []
    n_failed = 0

    # OPTIMIZED: Use itertuples() instead of iterrows() for 10-50x speedup
    for row in tqdm(gdf.itertuples(), total=len(gdf), desc="Calculating Annual Precipitation"):
        geom = row.geometry

        try:
            # OPTIMIZED: Use helper function for geometry conversion
            clipped = rainfall_da.rio.clip(_geometry_to_clip_format(geom), all_touched=False, drop=False)
            ts = clipped.mean(dim=["lat", "lon"], skipna=True)  # Time series per feature

            # Convert time coordinate to datetime
            time_index = pd.to_datetime(ts[date_dim].values)
            df = pd.DataFrame({ "precip": ts.values }, index=time_index)

            # Group by year and get annual sum
            annual_sum = df.resample("YE").sum()
            avg_annual_precip = annual_sum.mean().values[0]  # Average annual precipitation (mm/year)
        except Exception:
            n_failed += 1
            avg_annual_precip = np.nan

        avg_annual_precip_list.append(avg_annual_precip)

    _summarize_skipped("avg annual precip features", n_failed, len(gdf))
    gdf[new_col] = avg_annual_precip_list
    return gdf


def plot_multiple_maps_with_balanced_colorbar(gdf, value_cols, titles=None, cmap="RdBu", figsize=(12, 6),
    col_wrap=2, vmin_vmax_dict=None, save=None):
    """Plot multiple maps with balanced colorbars."""
    # OPTIMIZED: Removed unnecessary .copy() - .to_crs() already returns a copy
    gdf = gdf.to_crs("EPSG:4326")
    titles = titles or value_cols
    n = len(value_cols)
    ncols = col_wrap
    nrows = (n + ncols - 1) // ncols

    fig, axs = plt.subplots(
        nrows=nrows, ncols=ncols, figsize=figsize,
        subplot_kw={'projection': ccrs.PlateCarree()},
        gridspec_kw={'wspace': _MAP_SUBPLOT_WSPACE, 'hspace': _MAP_SUBPLOT_HSPACE},
    )
    axs = axs.flatten()

    minx, miny, maxx, maxy = gdf.total_bounds
    extent = [minx - 5, maxx + 5, miny - 5, maxy + 5]

    for i, (col, title) in enumerate(zip(value_cols, titles)):
        ax = axs[i]
        ax.set_extent(extent)
        ax.add_feature(cfeature.LAND, facecolor='darkgrey')
        ax.add_feature(cfeature.OCEAN, facecolor='lightblue')
        ax.add_feature(cfeature.BORDERS, linestyle=':', edgecolor='black', linewidth=0.4)
        ax.grid()
        ax.coastlines()
        gl = ax.gridlines(draw_labels=True, linewidth=0.3, color="gray", alpha=0.5)
        
        gl.right_labels = False   # never show right labels
        if i == 0:                # top subplot
            gl.bottom_labels = False
            gl.top_labels = True
            gl.left_labels = True
        elif i == 1:              # middle subplot
            gl.top_labels = False
            gl.bottom_labels = False
            gl.left_labels = True
        elif i == 2:              # bottom subplot
            gl.top_labels = False
            gl.bottom_labels = True
            gl.left_labels = True

        gl.xlabel_style = {'rotation': 0, 'size': _MAP_GRID_LABEL_FONTSIZE}
        gl.ylabel_style = {'rotation': 90, 'size': _MAP_GRID_LABEL_FONTSIZE}

        # Special handling for categorical significance map
        if col == "significance":
            use_cmap = ListedColormap(["#a6dba0", "#d73027"])  # 0 = not sig, 1 = sig
            norm = BoundaryNorm([-0.5, 0.5, 1.5], use_cmap.N)
            extend = 'neither'
        
        else:
            # Determine vmin/vmax from dict or data
            if vmin_vmax_dict and col in vmin_vmax_dict:
                vmin, vmax = vmin_vmax_dict[col]
            else:
                vmin = gdf[col].min().round(1)
                vmax = gdf[col].max().round(1)
        
            # Decide on color scale based on vmin
            if vmin >= 0:
                # Use sequential colormap (positive-only data)
                use_cmap = "plasma_r"
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                extend = 'max' if gdf[col].max() > vmax else 'neither'
            else:
                # Use diverging colormap centered around zero
                use_cmap = cmap
                norm = mcolors.Normalize(vmin=vmin, vmax=vmax)
                extend = 'both' if (gdf[col].min() < vmin or gdf[col].max() > vmax) else 'neither'

        sm = cm.ScalarMappable(norm=norm, cmap=use_cmap)

        gdf.plot(column=col, ax=ax, cmap=use_cmap, edgecolor='black', linewidth=0.4, norm=norm)
        
        ax.set_extent(extent, crs=ccrs.PlateCarree())
        if col == "significance":
            cbar = fig.colorbar(
                sm, ax=ax, orientation='vertical',
                fraction=_MAP_CBAR_FRACTION, pad=_MAP_CBAR_PAD,
                shrink=_MAP_CBAR_SHRINK_V,
            )
            cbar.set_ticks([0.2, 0.8])  # midpoints of 0–1 bins
            cbar.set_ticklabels(["Not Significant", "Significant"])
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label("Statistical Significance (p < 0.05)", fontsize=10)
        else:
            cbar = fig.colorbar(
                sm, ax=ax, orientation='vertical',
                fraction=_MAP_CBAR_FRACTION, pad=_MAP_CBAR_PAD,
                extend=extend, shrink=_MAP_CBAR_SHRINK_V,
            )
            cbar.set_ticks(np.linspace(vmin, vmax, 5))
            cbar.ax.tick_params(labelsize=8)
            cbar.set_label(f"{title}", fontsize=10)

    # Remove unused subplots
    for j in range(i + 1, len(axs)):
        fig.delaxes(axs[j])

    plt.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)
    if save:
        plt.savefig(save, dpi=500)
    plt.show()


def plot_timeseries(
    series_list,
    labels,
    colors=None,
    rainfall=None,
    plot_precip=True,
    title="TWS and Rainfall Time Series",
    ylabel="TWS (cm)",
    precip_label="Rainfall (mm)",
    figsize=(10, 4),
    ax=None,
    ylim=None,
    show_ylabel=True
):
    """Plot time series with optional rainfall."""
    series_list = [s.to_series() if hasattr(s, "to_series") else s for s in series_list]
    rainfall_series = rainfall.mean(dim=("lat", "lon")).to_series() if hasattr(rainfall, "mean") else rainfall

    if colors is None:
        colors = plt.cm.tab10.colors

    if ax is None:
        fig, ax1 = plt.subplots(figsize=figsize)
    else:
        ax1 = ax

    lines = []
    for i, (series, label) in enumerate(zip(series_list, labels)):
        line, = ax1.plot(series.index, series.values, label=label, color=colors[i % len(colors)], linewidth=2)
        lines.append(line)

    ax1.set_title(title, fontsize=14)
    if show_ylabel:
        ax1.set_ylabel(ylabel, fontsize=12)
    ax1.grid(True, linestyle="--", alpha=0.7)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    ax1.xaxis.set_major_locator(mdates.YearLocator(1))
    ax1.tick_params(axis='x', rotation=45)

    x_min = series_list[0].index.min() - pd.DateOffset(years=1)
    x_max = series_list[0].index.max() + pd.DateOffset(years=1)
    ax1.set_xlim(x_min, x_max)

    if ylim is not None:
        ax1.set_ylim(*ylim)

    if plot_precip and rainfall_series is not None:
        ax2 = ax1.twinx()
        bars = ax2.bar(
            rainfall_series.index,
            rainfall_series.values,
            label=precip_label,
            color="#c9630a",
            alpha=0.9,
            width=40
        )
        ax2.set_ylim(rainfall_series.min(), rainfall_series.max() + 100)
        ax2.invert_yaxis()
        if show_ylabel:
            ax2.set_ylabel(precip_label, fontsize=12)
        lines.append(bars)

    legend_labels = labels + ([precip_label] if plot_precip and rainfall_series is not None else [])
    ax1.legend(lines, legend_labels, fontsize=10, loc='lower left')


def plot_subbasin_time_series_all(
    gdf, dataarrays, rainfall, labels,
    id_col="aq_name", ids_to_plot=None, cols=2, uniform_ylim=False, save=None
):
    """Plot time series for all subbasins."""
    if ids_to_plot is not None:
        gdf = gdf[gdf[id_col].isin(ids_to_plot)]

    n = len(gdf)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 8, rows * 3), squeeze=False)

    # Pre-compute global ylim if needed
    global_min = np.inf
    global_max = -np.inf

    # OPTIMIZED: Use itertuples() instead of iterrows() for 10-50x speedup
    if uniform_ylim:
        for row in gdf.itertuples():
            geom = row.geometry
            for da in dataarrays:
                geom_clip = _geometry_to_clip_format(geom)
                da_clipped = da.rio.clip(geom_clip, all_touched=False, drop=False)
                ts = da_clipped.mean(dim=["lat", "lon"], skipna=True).values
                global_min = min(global_min, np.nanmin(ts))
                global_max = max(global_max, np.nanmax(ts))
        global_ylim = (global_min - 2, global_max + 5)
    else:
        global_ylim = None

    # OPTIMIZED: Use itertuples() instead of iterrows() and enumerate for faster iteration
    for idx, row in enumerate(tqdm(list(gdf.itertuples()), total=n, desc="Plotting Subbasins")):
        ax = axes.flatten()[idx]
        geom = row.geometry
        sub_id = getattr(row, id_col, None)

        series_list = []
        # OPTIMIZED: Use helper function for geometry conversion
        geom_clip = _geometry_to_clip_format(geom)
        for da in dataarrays:
            da_clipped = da.rio.clip(geom_clip, all_touched=False, drop=False)
            mean_ts = da_clipped.mean(dim=["lat", "lon"], skipna=True)
            series_list.append(mean_ts)

        rainfall_clipped = rainfall.rio.clip(geom_clip, all_touched=False, drop=False)

        show_ylabel = (idx % cols == 0)

        if not uniform_ylim:
            ts_all = np.concatenate([s.values.flatten() for s in series_list])
            local_min = np.nanmin(ts_all)
            local_max = np.nanmax(ts_all)
            local_ylim = (local_min - 2, local_max + 10)
        else:
            local_ylim = global_ylim

        plot_timeseries(
            series_list=series_list,
            labels=labels,
            colors=["#1f77b4", "#d62728", "#2ca02c"],
            rainfall=rainfall_clipped,
            plot_precip=True,
            title=f" {sub_id}",
            ax=ax,
            ylim=local_ylim,
            show_ylabel=show_ylabel
        )

    # Hide unused subplots
    for ax in axes.flatten()[n:]:
        ax.set_visible(False)

    plt.tight_layout()
    if save:
        plt.savefig(save, dpi=500)
    plt.show()


def decompose_grace_sin_cosin(grace_ts, time):
    """Decompose GRACE time series into trend, seasonal, and residual components.

    Returns three 1-D arrays (trend, season, residual) whose length matches the
    series after ``dropna()`` — the same convention as successful OLS fits.
    When too few points remain for a stable 6-parameter fit (<12, mirroring
    ``_decompose_grace_calendar``), returns NaN arrays of that length.
    """
    ts = grace_ts.to_pandas().dropna()
    n = len(ts)
    # Intercept + trend + annual sin/cos + semi-annual sin/cos (=6 params);
    # mirror calendar residual path which requires >=12 valid points.
    if n < 12:
        if not getattr(decompose_grace_sin_cosin, "_short_noted", False):
            _note(
                f"decompose_grace_sin_cosin: series too short for OLS "
                f"({n} points after dropna; need >=12); returning NaN components"
            )
            decompose_grace_sin_cosin._short_noted = True
        nan = np.full(n, np.nan, dtype=float)
        return nan, nan, nan

    t = np.arange(n)  # time index (0..N-1)
    
    X = pd.DataFrame({
        "trend": t,
        "sin1": np.sin(2 * np.pi * t / 12),  # annual
        "cos1": np.cos(2 * np.pi * t / 12),
        "sin2": np.sin(2 * np.pi * t / 6),   # semiannual
        "cos2": np.cos(2 * np.pi * t / 6),
    })
    X = sm.add_constant(X)  # intercept
    
    # regression
    model = sm.OLS(ts.values, X).fit()
    
    # fitted components
    fit = model.fittedvalues
    trend = model.params["trend"] * t + model.params["const"]
    season = (
        model.params["sin1"] * X["sin1"] +
        model.params["cos1"] * X["cos1"] +
        model.params["sin2"] * X["sin2"] +
        model.params["cos2"] * X["cos2"]
    )
    residual = ts.values - (trend + season)

    return trend, season, residual


# --- Helper functions for common operations ---
def _geometry_to_clip_format(geom):
    """Convert geometry to format suitable for rioxarray clipping (OPTIMIZED: helper to reduce duplication)."""
    if isinstance(geom, gpd.GeoDataFrame):
        geom = geom.geometry.iloc[0]
    elif isinstance(geom, gpd.GeoSeries):
        geom = geom.iloc[0]
    return [mapping(geom.buffer(0))]

def _calculate_precip_climatology(precip_ts):
    """Calculate monthly precipitation climatology (OPTIMIZED: helper to reduce duplication)."""
    return precip_ts.groupby("time.month").mean(dim="time")

def _calculate_precip_anomaly(precip_ts, climatology=None):
    """Calculate cumulative precipitation anomaly (OPTIMIZED: helper to reduce duplication)."""
    if climatology is None:
        climatology = _calculate_precip_climatology(precip_ts)
    anomaly = precip_ts.groupby("time.month") - climatology
    return anomaly.cumsum(dim="time")

# --- Helper functions for plot_grace_precip_extremes ---
def _mask_gap_years(series_or_df, gap_years):
    """Set values to NaN for specified gap years."""
    if isinstance(series_or_df, pd.DataFrame):
        series_or_df.loc[series_or_df.index.to_series().dt.year.isin(gap_years)] = np.nan
    elif isinstance(series_or_df, pd.Series):
        series_or_df.loc[series_or_df.index.to_series().dt.year.isin(gap_years)] = np.nan
    return series_or_df

def _create_full_monthly_range(series):
    """Create full monthly date range from series index."""
    if len(series) == 0:
        return pd.DatetimeIndex([])
    first_date = series.index[0]
    last_date = series.index[-1]
    return pd.date_range(start=first_date, end=last_date, freq='M')

def _calculate_axis_limits(mean_series, std_series=None, ymin_offset=-1, ymax_offset=5, 
                          default_min=-10, default_max=10):
    """Calculate axis limits with NaN handling."""
    if std_series is None:
        ymin = np.nanmin(mean_series.values) + ymin_offset
        ymax = np.nanmax(mean_series.values) + ymax_offset
    else:
        ymin = np.nanmin((mean_series - std_series).values) + ymin_offset
        ymax = np.nanmax((mean_series + std_series).values) + ymax_offset
    if not np.isfinite(ymin):
        ymin = default_min
    if not np.isfinite(ymax):
        ymax = default_max
    return ymin, ymax

def _calculate_trend_date_based(series, fill_na=True):
    """Calculate linear trend using actual dates (months since start)."""
    valid_mask = ~np.isnan(series.values)
    if np.sum(valid_mask) <= 1:
        return None, None, None, None
    
    if fill_na:
        series_filled = series.interpolate(method='linear', limit_direction='both')
        y_values = series_filled.values
    else:
        series_filled = series
        y_values = series[valid_mask].values
    
    first_date = series_filled.index[0]
    # OPTIMIZED: Vectorized calculation instead of list comprehension (2-3x speedup)
    # Average days per month (accounts for leap years)
    DAYS_PER_MONTH = 30.44
    # Use pandas Timedelta for vectorized calculation
    if isinstance(series_filled.index, pd.DatetimeIndex):
        months_since_start = (series_filled.index - first_date).days / DAYS_PER_MONTH
        if hasattr(months_since_start, 'values'):
            months_since_start = months_since_start.values
        else:
            months_since_start = np.array(months_since_start)
    else:
        # Fallback for non-DatetimeIndex
        months_since_start = np.array([(d - first_date).days / DAYS_PER_MONTH for d in series_filled.index])
    
    if fill_na:
        slope, intercept, _, _, _ = linregress(months_since_start, y_values)
    else:
        slope, intercept, _, _, _ = linregress(months_since_start[valid_mask], y_values)
    
    trendline_values = slope * months_since_start + intercept
    slope_per_year = slope * 12
    
    return slope_per_year, trendline_values, intercept, series_filled.index

def _calculate_precip_ylim(p_ser, default_max=200, min_buffer=60, buffer_frac=0.30):
    """Calculate precipitation y-axis limits."""
    precip_max = np.nanmax(p_ser.values) if len(p_ser) > 0 else 100
    precip_buffer = max(min_buffer, buffer_frac * precip_max) if np.isfinite(precip_max) else min_buffer
    precip_ymax = precip_max + precip_buffer
    if not np.isfinite(precip_ymax):
        precip_ymax = default_max
    return precip_ymax

def plot_grace_precip_extremes(
    grace_data=None,
    rainfall_monthly=None,
    geom=None,
    area_km2=None,
    threshold_percentile=95,
    precip_floor=10,
    title=None,
    save_path=None,
    grace_solutions=None,
    trend=False,
    all_components=False,
    gap_years=[2017, 2018],
    fig_size=(10, 10),
):
    """
    Plot GRACE TWS residuals and Precipitation with EPE markers.

    Supports either a single GRACE dataset via `grace_data` (backward compatible)
    or multiple solutions via `grace_solutions` (iterable of datasets). When
    multiple solutions are provided, the plot shows the mean residual time series
    and an error band of ±1σ (standard deviation across solutions). Differences
    printed and annotated include ± the standard deviation of per-solution
    before/after differences within the cluster window.
    
    Parameters:
    -----------
    all_components : bool, default=False
        If True, creates 3 subplots:
        - Top: GRACE TWS (not residual) vs precipitation with linear trend (cm/year)
        - Middle: Seasonal component
        - Bottom: Residual vs precipitation (current behavior)
        X-axis labels only on bottom subplot, no space between subplots,
        only ticks (no labels) for first and second subplots.
    gap_years : list, default=[2017, 2018]
        Years to mask as NaN in GRACE and precipitation data for gap visualization.
    """
    # OPTIMIZED: Use helper function for geometry conversion
    geom = _geometry_to_clip_format(geom)

    def _rio_ready_for_clip(da: xr.DataArray) -> xr.DataArray:
        """Ensure rioxarray has CRS + lon/lat dims before clip (avoids MissingCRS)."""
        if not isinstance(da, xr.DataArray):
            return da
        out = da
        if out.rio.crs is None:
            out = out.rio.write_crs("EPSG:4326")
        try:
            out.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        except Exception:
            pass
        return out

    # Allow passing multiple solutions via grace_data for backward compatibility
    if grace_solutions is None and isinstance(grace_data, (list, tuple)) and len(grace_data) > 0:
        grace_solutions = list(grace_data)

    # --- Clip and average spatially (GRACE and precipitation) ---
    # Calculate precipitation once (optimization: avoid N redundant operations)
    rainfall_monthly = _rio_ready_for_clip(rainfall_monthly)
    p_mean_m = rainfall_monthly.rio.clip(geom, crs="EPSG:4326", drop=True).mean(
        dim=["lat", "lon"], skipna=True
    )
    
    if grace_solutions is not None and len(grace_solutions) > 0:
        # Process each GRACE solution → residual volume series
        residual_series_list = []
        tws_series_list = []
        seasonal_series_list = []
        common_index = None
        for ds in grace_solutions:
            ds = _rio_ready_for_clip(ds)
            g_mean_i = ds.rio.clip(geom, crs="EPSG:4326", drop=True).mean(
                dim=["lat", "lon"], skipna=True
            )
            g_mean_i, p_mean = xr.align(g_mean_i, p_mean_m, join="inner")
            time_i = p_mean_m.time
            trend_i, season_i, residual_i = decompose_grace_sin_cosin(g_mean_i, time_i)
            g_ser_i = g_mean_i.to_series().dropna()
            # align precipitation index to GRACE
            if common_index is None:
                common_index = g_ser_i.index
            else:
                common_index = common_index.intersection(g_ser_i.index)
            
            # convert residual to series on same index as g_ser_i
            # The decomposition arrays (trend_i, season_i, residual_i) have length equal to len(g_ser_i)
            # because decompose_grace_sin_cosin uses dropna() internally
            # Create Series by directly assigning values to avoid index alignment issues
            residual_arr = np.asarray(residual_i)
            g_res_i = pd.Series(index=g_ser_i.index, dtype=float)
            g_res_i.values[:] = residual_arr
            # convert cm → km³
            #g_res_i = g_res_i * (area_km2 * 0.00001)
            residual_series_list.append(g_res_i)
            
            # Store TWS (trend + season + residual) and seasonal if all_components
            if all_components:
                tws_values = trend_i + season_i + residual_i
                tws_arr = np.asarray(tws_values)
                tws_i = pd.Series(index=g_ser_i.index, dtype=float)
                tws_i.values[:] = tws_arr
                    
                season_arr = np.asarray(season_i)
                season_ser_i = pd.Series(index=g_ser_i.index, dtype=float)
                season_ser_i.values[:] = season_arr
                tws_series_list.append(tws_i)
                seasonal_series_list.append(season_ser_i)

        # Align all residual series to common index and compute mean/std
        residual_df = pd.concat([s.reindex(common_index) for s in residual_series_list], axis=1)
        residual_df.columns = [f"sol_{i+1}" for i in range(residual_df.shape[1])]
        # Mask gap years in residual data
        _mask_gap_years(residual_df, gap_years)
        g_ser_mean = residual_df.mean(axis=1)
        g_ser_std = residual_df.std(axis=1, ddof=1)
        g_ser_mean.name = "GRACE TWS Residual (cm)"
        
        # Create full monthly date range and reindex for proper gap visualization
        if len(g_ser_mean) > 0:
            full_date_range = _create_full_monthly_range(g_ser_mean)
            # Reindex to include all months (missing dates will be NaN)
            g_ser_mean = g_ser_mean.reindex(full_date_range)
            if g_ser_std is not None:
                g_ser_std = g_ser_std.reindex(full_date_range)
            if residual_df is not None:
                residual_df = residual_df.reindex(full_date_range)
        
        # If all_components, compute mean/std for TWS and seasonal
        if all_components:
            tws_df = pd.concat([s.reindex(common_index) for s in tws_series_list], axis=1)
            tws_df.columns = [f"sol_{i+1}" for i in range(tws_df.shape[1])]
            # Mask gap years in TWS data
            _mask_gap_years(tws_df, gap_years)
            g_tws_mean = tws_df.mean(axis=1)
            g_tws_std = tws_df.std(axis=1, ddof=1)
            g_tws_mean.name = "GRACE TWS (cm)"
            
            # Reindex TWS to full monthly date range (use same index as residual)
            if len(g_tws_mean) > 0 and len(g_ser_mean) > 0:
                g_tws_mean = g_tws_mean.reindex(g_ser_mean.index)
                if g_tws_std is not None:
                    g_tws_std = g_tws_std.reindex(g_ser_mean.index)
                tws_df = tws_df.reindex(g_ser_mean.index)
            
            seasonal_df = pd.concat([s.reindex(common_index) for s in seasonal_series_list], axis=1)
            seasonal_df.columns = [f"sol_{i+1}" for i in range(seasonal_df.shape[1])]
            # Mask gap years in seasonal data
            _mask_gap_years(seasonal_df, gap_years)
            g_seasonal_mean = seasonal_df.mean(axis=1)
            g_seasonal_std = seasonal_df.std(axis=1, ddof=1)
            g_seasonal_mean.name = "GRACE Seasonal (cm)"
            
            # Reindex seasonal to full monthly date range (use same index as residual)
            if len(g_seasonal_mean) > 0 and len(g_ser_mean) > 0:
                g_seasonal_mean = g_seasonal_mean.reindex(g_ser_mean.index)
                if g_seasonal_std is not None:
                    g_seasonal_std = g_seasonal_std.reindex(g_ser_mean.index)
        else:
            tws_df = None
            g_tws_mean = None
            g_tws_std = None
            g_seasonal_mean = None
            g_seasonal_std = None

        # Prepare precipitation series aligned to mean index (will include NaN for gap years)
        p_ser = p_mean_m.to_series().reindex(g_ser_mean.index)
        # Mask gap years in precipitation
        _mask_gap_years(p_ser, gap_years)
        p_mean = p_mean_m.reindex(time=g_ser_mean.index)  # Align to common index for quantile calculation
    else:
        # Backward-compatible single GRACE dataset path
        grace_data = _rio_ready_for_clip(grace_data)
        g_mean = grace_data.rio.clip(geom, crs="EPSG:4326", drop=True).mean(
            dim=["lat", "lon"], skipna=True
        )
        g_mean, p_mean = xr.align(g_mean, p_mean_m, join="inner")

        # --- Decompose GRACE ---
        time = p_mean_m.time
        trend, season, residual = decompose_grace_sin_cosin(g_mean, time)

        # Convert to pandas
        g_ser_orig = g_mean.to_series().dropna()
        p_ser = p_mean.to_series().reindex(g_ser_orig.index)
        # The decomposition arrays have length equal to len(g_ser_orig) because decompose_grace_sin_cosin uses dropna() internally
        # Create Series by directly assigning values to avoid index alignment issues
        residual_arr = np.asarray(residual)
        g_ser = pd.Series(index=g_ser_orig.index, name="GRACE TWS Residual", dtype=float)
        g_ser.values[:] = residual_arr
       
        # --- Convert cm → km³ ---
        #g_ser = g_ser * (area_km2 * 0.00001)
        g_ser.name = "GRACE TWS Residual (cm)"
        # Mask gap years in GRACE residual
        _mask_gap_years(g_ser, gap_years)
        
        # Create full monthly date range and reindex for proper gap visualization
        if len(g_ser) > 0:
            full_date_range = _create_full_monthly_range(g_ser)
            # Reindex to include all months (missing dates will be NaN)
            g_ser = g_ser.reindex(full_date_range)
            # Also reindex precipitation to match
            p_ser = p_ser.reindex(full_date_range)
            # Mask gap years in precipitation
            _mask_gap_years(p_ser, gap_years)
        
        g_ser_mean = g_ser
        g_ser_std = None
        residual_df = None  # Not available for single dataset
        tws_df = None  # Not available for single dataset
        
        # Store TWS and seasonal if all_components
        if all_components:
            # Use original index before reindexing (from g_ser_orig)
            tws_values = trend + season + residual
            tws_arr = np.asarray(tws_values)
            g_tws_mean = pd.Series(index=g_ser_orig.index, name="GRACE TWS (cm)", dtype=float)
            g_tws_mean.values[:] = tws_arr
            # Mask gap years in TWS
            _mask_gap_years(g_tws_mean, gap_years)
            # Reindex to full date range (already done for g_ser above)
            g_tws_mean = g_tws_mean.reindex(g_ser_mean.index)
            
            season_arr = np.asarray(season)
            g_seasonal_mean = pd.Series(index=g_ser_orig.index, name="GRACE Seasonal (cm)", dtype=float)
            g_seasonal_mean.values[:] = season_arr
            # Mask gap years in seasonal
            _mask_gap_years(g_seasonal_mean, gap_years)
            # Reindex to full date range (already done for g_ser above)
            g_seasonal_mean = g_seasonal_mean.reindex(g_ser_mean.index)
            g_tws_std = None
            g_seasonal_std = None
        else:
            g_tws_mean = None
            g_tws_std = None
            g_seasonal_mean = None
            g_seasonal_std = None
    
    # --- Compute precipitation threshold ---
    q = p_mean_m.quantile(threshold_percentile).compute()
    if np.isnan(q):
        print(f"Skipping {title}: rainfall quantile is NaN")
        return
    threshold = max(float(q.item()), precip_floor)
    is_extreme = p_ser > threshold
    extreme_dates = p_ser.index[is_extreme].sort_values()

    # --- Cluster events within 12 months ---
    clusters = []
    if len(extreme_dates) > 0:
        cluster_start = extreme_dates[0]
        cluster_end = extreme_dates[0]
        for d in extreme_dates[1:]:
            if (d - cluster_end) <= pd.Timedelta(days=365):
                # extend cluster
                cluster_end = d
            else:
                clusters.append((cluster_start, cluster_end))
                cluster_start, cluster_end = d, d
        clusters.append((cluster_start, cluster_end))

    # --- Helper function to plot clusters ---
    def plot_clusters(ax, grace_series, grace_std, clusters, residual_df=None, use_residual_df=False):
        """Plot cluster markers and annotations on an axis."""
        for (start_evt, end_evt) in clusters:
            # vertical markers at event cluster
            ax.axvline(start_evt, color="red", linestyle="--", lw=0.8, alpha=0.6)
            ax.axvline(end_evt,   color="red", linestyle="--", lw=0.8, alpha=0.6)

            # windows for before and after
            start_before = start_evt - pd.DateOffset(months=12)
            end_after    = end_evt + pd.DateOffset(months=12)

            before_vals = grace_series.loc[start_before:start_evt]
            after_vals  = grace_series.loc[end_evt:end_after]

            if len(before_vals) > 0 and len(after_vals) > 0:
                # Use nanmean to handle NaN values
                avg_before = np.nanmean(before_vals.values)
                avg_after  = np.nanmean(after_vals.values)
                change     = avg_after - avg_before
                
                # Skip if either average is NaN
                if np.isnan(avg_before) or np.isnan(avg_after):
                    continue

                # If multiple solutions, compute per-solution diff std for this cluster
                change_std = None
                if grace_std is not None and use_residual_df and residual_df is not None:
                    diffs = []
                    for col in residual_df.columns:
                        s = residual_df[col]
                        b = s.loc[start_before:start_evt]
                        a = s.loc[end_evt:end_after]
                        if len(b) > 0 and len(a) > 0:
                            diffs.append(a.mean() - b.mean())
                    if len(diffs) >= 2:
                        change_std = float(np.std(diffs, ddof=1))

                # OPTIMIZED: Filter for detectable changes only (same logic as summarize_results_dict)
                # Only show changes where: change > 0 AND change > std (if std available)
                is_detectable_change = False
                if change_std is not None:
                    # For multiple solutions: change must be > 0 AND > std
                    is_detectable_change = (change > 0) and (change > change_std)
                else:
                    # For single solution: only require change > 0
                    is_detectable_change = (change > 0)

                # Plot dashed red lines (always plot for visibility, but only annotate detectable changes)
                if use_residual_df:

                    # Print summary only for detectable changes (include ±σ if available)
                    if is_detectable_change:
                        ax.hlines(avg_before, xmin=start_before, xmax=start_evt,
                                colors='black', linestyles='--', lw=1.5)
                        ax.hlines(avg_after, xmin=end_evt, xmax=end_after,
                                colors='black', linestyles='--', lw=1.5)                        
                        if change_std is not None:
                            print(f"{title} | Cluster {start_evt.date()}–{end_evt.date()} | "
                                f"Before = {avg_before:.1f}, After = {avg_after:.1f}, Change = {change:.1f} ± {change_std:.2f} cm")
                        else:
                            print(f"{title} | Cluster {start_evt.date()}–{end_evt.date()} | "
                                f"Before = {avg_before:.1f}, After = {avg_after:.1f}, Change = {change:.1f} cm")

                # Annotate change near the event on the plot (only for residual panel and detectable changes)
                if use_residual_df and is_detectable_change:
                    try:
                        if change_std is not None:
                            text = f"ΔGWS={change:.1f} ± {change_std:.1f}"
                        else:
                            text = f"ΔGWS={change:.1f}"
                        ax.annotate(
                            text,
                            xy=(end_evt, avg_after),
                            xycoords='data',
                            xytext=(-30, 40),
                            textcoords='offset points',
                            fontsize=14,
                            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='gray', alpha=0.9),
                            arrowprops=dict(arrowstyle='->', color='gray', lw=0.8, alpha=0.8)
                        )
                    except Exception:
                        pass

    # --- Plotting ---
    if all_components:
        # Create 3 subplots when all_components=True
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=fig_size, sharex=True)
        fig.subplots_adjust(hspace=0)  # No space between subplots
        
        # --- Top panel: TWS vs Precipitation ---
        # Plot GRACE TWS (will show NaN gaps naturally)
        ax1.plot(g_tws_mean.index, g_tws_mean.values, color='blue', lw=1.5,
                 label='GRACE TWS')
        if g_tws_std is not None:
            ax1.fill_between(
                g_tws_mean.index,
                (g_tws_mean - g_tws_std).values,
                (g_tws_mean + g_tws_std).values,
                color='blue', alpha=0.2, label='±1σ (Ensemble Mean)'
            )
        
        # Add linear trend for TWS
        if tws_df is not None and tws_df.shape[1] > 1:
            # Multiple solutions: compute trend for each solution
            slopes_per_year = []
            intercepts = []
            for col in tws_df.columns:
                series = tws_df[col]
                slope_per_year, _, intercept, _ = _calculate_trend_date_based(series, fill_na=True)
                if slope_per_year is not None:
                    slopes_per_year.append(slope_per_year)
                    intercepts.append(intercept)
            
            if len(slopes_per_year) > 0:
                slope_mean_per_year = np.mean(slopes_per_year)
                slope_std_per_year = np.std(slopes_per_year, ddof=1) if len(slopes_per_year) > 1 else 0.0
                
                # Recalculate intercept using mean slope and mean series
                first_date = g_tws_mean.index[0]
                DAYS_PER_MONTH = 30.44
                months_since_start = np.array([(d - first_date).days / DAYS_PER_MONTH for d in g_tws_mean.index])
                # Use mean of g_tws_mean to calculate intercept
                g_tws_mean_filled = g_tws_mean.interpolate(method='linear', limit_direction='both')
                slope_mean_per_month = slope_mean_per_year / 12
                # Calculate intercept: y_mean = slope * x + intercept => intercept = y_mean - slope * x_mean
                y_mean = np.nanmean(g_tws_mean_filled.values)
                x_mean = np.nanmean(months_since_start)
                intercept_mean = y_mean - slope_mean_per_month * x_mean
                
                trendline_values = slope_mean_per_month * months_since_start + intercept_mean
                
                if np.isfinite(slope_mean_per_year):
                    label_text = (f'Trend ({slope_mean_per_year:.2f} ± {slope_std_per_year:.2f} cm/year)' 
                                 if slope_std_per_year > 0 
                                 else f'Trend ({slope_mean_per_year:.2f} cm/year)')
                    ax1.plot(g_tws_mean.index, trendline_values, color='red', lw=2, 
                            linestyle='-', alpha=0.8, label=label_text, zorder=5)
        else:
            # Single solution: use mean series
            slope_per_year, trendline_values, _, trend_index = _calculate_trend_date_based(g_tws_mean, fill_na=True)
            if slope_per_year is not None and np.isfinite(slope_per_year):
                ax1.plot(trend_index, trendline_values, color='red', lw=2, 
                        linestyle='-', alpha=0.8, label=f'Trend ({slope_per_year:.2f} cm/year)', zorder=5)
        
        # Precipitation on secondary axis
        ax1_precip = ax1.twinx()
        # Only plot precipitation where values are not NaN (exclude 2017-2018 gap)
        p_ser_valid = p_ser.dropna()
        if len(p_ser_valid) > 0:
            ax1_precip.bar(p_ser_valid.index, p_ser_valid.values, width=35,
                    color="tab:blue", alpha=0.6, label="Precipitation (mm)")
        precip_ymax = _calculate_precip_ylim(p_ser)
        ax1_precip.set_ylim(precip_ymax, 0)  # inverted
        ax1_precip.set_ylabel("Precipitation (mm)", fontsize=14)
        ax1_precip.axhline(threshold, color="tab:orange", linestyle="--", lw=1.2,
                    label=f"{int(threshold_percentile*100)}th perc. ({threshold:.1f} mm)")
        
        ax1.set_ylabel("GRACE TWS (cm)", fontsize=14)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ymin_tws, ymax_tws = _calculate_axis_limits(g_tws_mean, g_tws_std, ymin_offset=-1, ymax_offset=5)
        ax1.set_ylim(ymin_tws, ymax_tws)
        ax1.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # Only ticks, no labels
        plt.yticks(fontsize=12)
        
        # Plot clusters on top panel
        plot_clusters(ax1, g_tws_mean, g_tws_std, clusters, residual_df=None, use_residual_df=False)
        
        # Legends for top panel
        ax1.legend(loc="lower right", fontsize=12)
        ax1_precip.legend(loc="best", fontsize=12)
    
        
        # --- Middle panel: Seasonal component ---
        ax2.plot(g_seasonal_mean.index, g_seasonal_mean.values, color='green', lw=1.5,
                 label='GRACE Seasonal')
        if g_seasonal_std is not None:
            ax2.fill_between(
                g_seasonal_mean.index,
                (g_seasonal_mean - g_seasonal_std).values,
                (g_seasonal_mean + g_seasonal_std).values,
                color='green', alpha=0.2, label='±1σ (Ensemble Mean)'
            )
        
        ax2.set_ylabel("GRACE Seasonal (cm)", fontsize=14)
        ax2.grid(True, linestyle='--', alpha=0.6)
        ymin_season, ymax_season = _calculate_axis_limits(g_seasonal_mean, g_seasonal_std, 
                                                          ymin_offset=-1, ymax_offset=1, 
                                                          default_min=-5, default_max=5)
        ax2.set_ylim(ymin_season, ymax_season)
        ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False)  # Only ticks, no labels
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        ax2.legend(loc="best", fontsize=12)
        
        # --- Bottom panel: Residual vs Precipitation (current behavior) ---
        ax3.plot(g_ser_mean.index, g_ser_mean.values, color='blue', lw=1.5,
                 label='GRACE TWS Residual')
        if g_ser_std is not None:
            ax3.fill_between(
                g_ser_mean.index,
                (g_ser_mean - g_ser_std).values,
                (g_ser_mean + g_ser_std).values,
                color='blue', alpha=0.2, label='±1σ (Ensemble Mean)'
            )
        
        ax3.set_ylabel("GRACE TWS Residual (cm)", fontsize=14)
        ax3.grid(True, linestyle='--', alpha=0.6)
        ymin_res, ymax_res = _calculate_axis_limits(g_ser_mean, g_ser_std, ymin_offset=-1, ymax_offset=5)
        ax3.set_ylim(ymin_res, ymax_res)
        
        # X-axis labels only on bottom subplot
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax3.xaxis.set_major_locator(mdates.YearLocator(2))
        ax3.xaxis.set_minor_locator(mdates.YearLocator(1))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right',fontsize=12)
        plt.xticks(fontsize=12)
        plt.yticks(fontsize=12)
        
        # Plot clusters on bottom panel
        plot_clusters(ax3, g_ser_mean, g_ser_std, clusters, 
                     residual_df=residual_df if grace_solutions is not None else None, 
                     use_residual_df=True)
        
        # Precipitation on secondary axis for bottom panel
        ax3_precip = ax3.twinx()
        # Only plot precipitation where values are not NaN (exclude 2017-2018 gap)
        p_ser_valid = p_ser.dropna()
        if len(p_ser_valid) > 0:
            ax3_precip.bar(p_ser_valid.index, p_ser_valid.values, width=35,
                    color="tab:blue", alpha=0.6, label="Precipitation (mm)")
        ax3_precip.set_ylim(precip_ymax, 0)  # inverted
        ax3_precip.set_ylabel("Precipitation (mm)", fontsize=14)
        ax3_precip.axhline(threshold, color="tab:orange", linestyle="--", lw=1.2,
                    )
        
        # Legends for bottom panel
        ax3.legend(loc="lower right", fontsize=12)
        ax3_precip.legend(loc="best", fontsize=12)
        
        # Set title on top subplot
        #ax1.set_title(title, fontsize=13, pad=8)
    else:
        # Original single subplot behavior
        fig, ax1 = plt.subplots(figsize=(10, 4))

        # GRACE residual line (mean) and optional error band
        ax1.plot(g_ser_mean.index, g_ser_mean.values, color='blue', lw=1.5,
                 label='GRACE TWS Residual')
        if g_ser_std is not None:
            ax1.fill_between(
                g_ser_mean.index,
                (g_ser_mean - g_ser_std).values,
                (g_ser_mean + g_ser_std).values,
                color='blue', alpha=0.2, label='±1σ (Ensemble Mean)'
            )
        
        # Add trendline if requested
        if trend:
            slope_per_year, trendline_values, _, trend_index = _calculate_trend_date_based(g_ser_mean, fill_na=True)
            if slope_per_year is not None and np.isfinite(slope_per_year):
                ax1.plot(trend_index, trendline_values, color='red', lw=2, 
                        linestyle='-', alpha=0.8, label=f'Trend ({slope_per_year:.2f} cm/year)')
        
        ax1.set_ylabel("GRACE TWS Residual (cm)", fontsize=14)
        #ax1.set_title(title, fontsize=13, pad=8)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ymin, ymax = _calculate_axis_limits(g_ser_mean, g_ser_std, ymin_offset=-1, ymax_offset=5)
        ax1.set_ylim(ymin, ymax)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
        ax1.xaxis.set_major_locator(mdates.YearLocator(2))
        plt.xticks(rotation=45,fontsize=12)
        plt.yticks(fontsize=12)

        # Plot clusters
        plot_clusters(ax1, g_ser_mean, g_ser_std, clusters, 
                     residual_df=residual_df if grace_solutions is not None else None, 
                     use_residual_df=True)

        # --- Precipitation (secondary axis) ---
        ax2 = ax1.twinx()
        ax2.tick_params(axis='x', which='both', bottom=True, top=False, labelbottom=False,size=12)  # Only ticks, no labels
        # Only plot precipitation where values are not NaN (exclude 2017-2018 gap)
        p_ser_valid = p_ser.dropna()
        if len(p_ser_valid) > 0:
            ax2.bar(p_ser_valid.index, p_ser_valid.values, width=35,
                    color="tab:blue", alpha=0.6, label="Precipitation (mm)")

        precip_ymax = _calculate_precip_ylim(p_ser)
        ax2.set_ylim(precip_ymax, 0)  # inverted
        ax2.set_ylabel("Precipitation (mm)", fontsize=14)

        # Threshold line
        ax2.axhline(threshold, color="tab:orange", linestyle="--", lw=1.2,
                    label=f"{int(threshold_percentile*100)}th perc. ({threshold:.1f} mm)")

        # Legends
        ax1.legend(loc="lower right", fontsize=12)
        ax2.legend(loc="best", fontsize=12)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=500, bbox_inches="tight")
    plt.show()


def summarize_responses_clustered_simple(
    responses,
    grace_solution_name=None,
    threshold_percentile=None,
    aquifer_boundary=None,
    gdf_reference=None
):
    """
    Simplified summary function that returns only essential columns for comparison across loop iterations.
    
    Parameters:
    -----------
    responses : dict
        The responses_clus dictionary from analyze_grace_response_by_subbasin_clustered
    grace_solution_name : str, optional
        Name of the GRACE solution (e.g., 'GRACE_CSR')
    threshold_percentile : float, optional
        Threshold percentile used (e.g., 0.95)
    aquifer_boundary : str, optional
        Aquifer boundary type (e.g., 'Full', 'EPE95', 'EPE95_CN75')
    gdf_reference : GeoDataFrame, optional
        Reference GeoDataFrame to get aq_name mapping if subbasin_id is not the same as aq_name
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'detailed': Simplified DataFrame with cluster-level data
        - 'stats': Simplified DataFrame with summary statistics per subbasin
    """
    results = {}
    
    # Flatten the nested structure into a detailed DataFrame
    detailed_records = []
    
    for subbasin_id, data in responses.items():
        response_month = data.get("response_month", [])
        extreme_precip_month = data.get("extreme_precip_month", [])
        
        # Create a mapping of cluster keys to combine GRACE and precip data
        cluster_data = {}
        
        # Add GRACE response data
        for item in response_month:
            cluster_key = (item["cluster_start"], item["cluster_end"])
            cluster_data[cluster_key] = {
                "subbasin_id": subbasin_id,
                "cluster_start": pd.to_datetime(item["cluster_start"]),
                "cluster_end": pd.to_datetime(item["cluster_end"]),
                "area_km2": item.get("area_km2", np.nan),
                "before_avg": item.get("before_avg", np.nan),
                "after_avg": item.get("after_avg", np.nan),
                "grace_diff": item.get("diff", np.nan),
            }
        
        # Add precipitation data
        for item in extreme_precip_month:
            cluster_key = (item["cluster_start"], item["cluster_end"])
            if cluster_key in cluster_data:
                cluster_data[cluster_key].update({
                    "precip_sum": item.get("sum", np.nan),
                    "precip_max": item.get("max", np.nan),
                    "precip_count": item.get("count", np.nan),
                })
            else:
                # If cluster only has precip data (shouldn't happen, but handle it)
                cluster_data[cluster_key] = {
                    "subbasin_id": subbasin_id,
                    "cluster_start": pd.to_datetime(item["cluster_start"]),
                    "cluster_end": pd.to_datetime(item["cluster_end"]),
                    "area_km2": np.nan,
                    "before_avg": np.nan,
                    "after_avg": np.nan,
                    "grace_diff": np.nan,
                    "precip_sum": item.get("sum", np.nan),
                    "precip_max": item.get("max", np.nan),
                    "precip_count": item.get("count", np.nan),
                }
        
        # Add cluster duration and metadata
        for cluster_key, record in cluster_data.items():
            record["cluster_duration_days"] = (record["cluster_end"] - record["cluster_start"]).days
            # Add metadata columns
            if grace_solution_name:
                record["grace_solution"] = grace_solution_name
            if threshold_percentile is not None:
                record["threshold_perc"] = threshold_percentile * 100
            if aquifer_boundary:
                record["aquifer_boundary"] = aquifer_boundary
            detailed_records.append(record)
    
    # Create detailed DataFrame with only requested columns
    if detailed_records:
        df_detailed = pd.DataFrame(detailed_records)
        # Select only the columns requested by user
        cols_to_keep = ["cluster_start", "cluster_end", "area_km2", "before_avg", "after_avg", 
                       "grace_diff", "precip_sum", "precip_max", "precip_count", "cluster_duration_days"]
        # Add metadata columns if provided
        metadata_cols = []
        if grace_solution_name:
            metadata_cols.append("grace_solution")
        if threshold_percentile is not None:
            metadata_cols.append("threshold_perc")
        if aquifer_boundary:
            metadata_cols.append("aquifer_boundary")
        if "subbasin_id" not in cols_to_keep:
            cols_to_keep.insert(0, "subbasin_id")
        
        df_detailed = df_detailed[cols_to_keep + metadata_cols]
        df_detailed = df_detailed.sort_values(by=["subbasin_id", "cluster_start"])
    else:
        cols_to_keep = ["subbasin_id", "cluster_start", "cluster_end", "area_km2", "before_avg", 
                        "after_avg", "grace_diff", "precip_sum", "precip_max", "precip_count", 
                        "cluster_duration_days"]
        metadata_cols = []
        if grace_solution_name:
            metadata_cols.append("grace_solution")
        if threshold_percentile is not None:
            metadata_cols.append("threshold_perc")
        if aquifer_boundary:
            metadata_cols.append("aquifer_boundary")
        df_detailed = pd.DataFrame(columns=cols_to_keep + metadata_cols)
    
    results['detailed'] = df_detailed
    
    # Create simplified summary statistics
    if len(df_detailed) > 0:
        # Get aq_name mapping if gdf_reference is provided
        aq_name_map = None
        if gdf_reference is not None and 'aq_name' in gdf_reference.columns:
            aq_name_map = gdf_reference.set_index('subbasin_id')['aq_name'].to_dict()
        
        stats_by_subbasin = df_detailed.groupby('subbasin_id').agg({
            'grace_diff': ['count', 'mean', 'std', 'min', 'max', 'sum'],
            'precip_sum': ['max'],
            'cluster_duration_days': ['mean', 'min', 'max'],
            'area_km2': 'first'
        }).round(2)
        
        # Flatten column names
        stats_by_subbasin.columns = ['_'.join(col).strip() for col in stats_by_subbasin.columns.values]
        stats_by_subbasin = stats_by_subbasin.reset_index()
        
        # Calculate sum of positive grace_diff values (mean > 0)
        positive_sums = df_detailed.groupby('subbasin_id').apply(
            lambda x: x[x['grace_diff'] > 0]['grace_diff'].sum() if len(x[x['grace_diff'] > 0]) > 0 else 0
        ).round(2)
        positive_sums.name = 'grace_diff_sum_pos'
        stats_by_subbasin = stats_by_subbasin.merge(positive_sums, left_on='subbasin_id', right_index=True, how='left')
        stats_by_subbasin['grace_diff_sum_pos'] = stats_by_subbasin['grace_diff_sum_pos'].fillna(0)
        
        # Add aq_name if mapping is available
        if aq_name_map:
            stats_by_subbasin['aq_name'] = stats_by_subbasin['subbasin_id'].map(aq_name_map)
        
        # Add metadata columns
        if grace_solution_name:
            stats_by_subbasin['grace_solution'] = grace_solution_name
        if threshold_percentile is not None:
            stats_by_subbasin['threshold_perc'] = threshold_percentile * 100
        if aquifer_boundary:
            stats_by_subbasin['aquifer_boundary'] = aquifer_boundary
        
        # Reorder columns as requested by user
        col_order = ['aquifer_boundary', 'grace_solution', 'threshold_perc', 'subbasin_id', 'aq_name',
                    'grace_diff_count', 'grace_diff_mean', 'grace_diff_std', 'grace_diff_min', 
                    'grace_diff_max', 'grace_diff_sum', 'grace_diff_sum_pos', 'precip_sum_max',
                    'cluster_duration_days_mean', 'cluster_duration_days_min', 'cluster_duration_days_max',
                    'area_km2_first']
        
        # Only include columns that exist
        existing_cols = [col for col in col_order if col in stats_by_subbasin.columns]
        other_cols = [col for col in stats_by_subbasin.columns if col not in col_order]
        stats_by_subbasin = stats_by_subbasin[existing_cols + other_cols]
        
        results['stats'] = stats_by_subbasin
    else:
        # Return empty DataFrame with expected columns
        results['stats'] = pd.DataFrame(columns=[
            'aquifer_boundary', 'grace_solution', 'threshold_perc', 'subbasin_id', 'aq_name',
            'grace_diff_count', 'grace_diff_mean', 'grace_diff_std', 'grace_diff_min', 
            'grace_diff_max', 'grace_diff_sum', 'grace_diff_sum_pos', 'precip_sum_max',
            'cluster_duration_days_mean', 'cluster_duration_days_min', 'cluster_duration_days_max',
            'area_km2_first'
        ])
    
    return results


def summarize_results_dict(results_dict, save_csv=False, csv_prefix="all_iterations"):
    """
    Process all entries in results_dict and combine them into comprehensive DataFrames.
    
    This function extracts data from results_dict (which already contains all loop iterations)
    and creates simplified DataFrames for comparison.
    
    For GRACE_Mean, it also calculates the standard deviation of the three solution values
    (CSR, JPL, GSFC) for each cluster and sums them.
    
    Parameters:
    -----------
    results_dict : dict
        Dictionary with keys like 'GRACE_CSR_95th_Full' containing:
        - 'responses': responses_clus dictionary
        - 'grace_solution': name of GRACE solution
        - 'threshold': threshold percentile
        - 'aquifer_boundary': boundary type
        - 'aquifer_gdf': GeoDataFrame for aq_name mapping
    save_csv : bool, default=False
        If True, save combined DataFrames as CSV files
    csv_prefix : str, default="all_iterations"
        Prefix for CSV filenames
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'all_detailed': Combined detailed DataFrame from all iterations
        - 'all_stats': Combined stats DataFrame from all iterations
        - 'pivot_tables': Dictionary of pivot tables for comparison
    """
    all_summarized_results = []
    
    # Process each entry in results_dict
    for key, data in results_dict.items():
        responses = data.get('responses')
        if responses is None:
            continue
        
        # Extract metadata from results_dict
        grace_solution_name = data.get('grace_solution')
        threshold_pct = data.get('threshold')
        aquifer_boundary = data.get('aquifer_boundary')
        aquifer_gdf = data.get('aquifer_gdf')
        
        # Summarize this iteration
        summary_result = summarize_responses_clustered_simple(
            responses=responses,
            grace_solution_name=grace_solution_name,
            threshold_percentile=threshold_pct,
            aquifer_boundary=aquifer_boundary,
            gdf_reference=aquifer_gdf
        )
        all_summarized_results.append(summary_result)
    
    # Combine all results first
    combined_all = combine_all_iterations(all_summarized_results, save_csv=False, csv_prefix=csv_prefix)
    
    # For GRACE_Mean entries, calculate std of the three solutions and add to stats
    if len(combined_all['all_stats']) > 0:
        # OPTIMIZED: Only copy if we need to modify (avoid unnecessary copy)
        all_stats = combined_all['all_stats'].copy()
        
        # OPTIMIZED: Find GRACE_Mean entries (no need to copy if not modifying)
        mean_entries = all_stats[all_stats['grace_solution'] == 'GRACE_Mean']
        
        if len(mean_entries) > 0:
            # OPTIMIZED: Create multi-index for faster lookups (5-10x speedup)
            try:
                all_stats_indexed = all_stats.set_index(['grace_solution', 'threshold_perc', 'aquifer_boundary', 'subbasin_id'])
                detailed_indexed = combined_all['all_detailed'].set_index(['grace_solution', 'threshold_perc', 'aquifer_boundary', 'subbasin_id'])
                use_index = True
            except Exception:
                # Fallback to original method if indexing fails
                use_index = False
            
            # OPTIMIZED: Use itertuples() instead of iterrows() for faster iteration
            for mean_row in mean_entries.itertuples():
                idx = mean_row.Index
                threshold_perc = mean_row.threshold_perc
                aquifer_bound = mean_row.aquifer_boundary
                subbasin_id = mean_row.subbasin_id
                
                # OPTIMIZED: Use multi-index lookup if available, otherwise use filtering
                if use_index:
                    try:
                        csr_entry = all_stats_indexed.loc[('GRACE_CSR', threshold_perc, aquifer_bound, subbasin_id)]
                        jpl_entry = all_stats_indexed.loc[('GRACE_JPL', threshold_perc, aquifer_bound, subbasin_id)]
                        gsfc_entry = all_stats_indexed.loc[('GRACE_GSFC', threshold_perc, aquifer_bound, subbasin_id)]
                        
                        # Convert single row Series to DataFrame if needed
                        if isinstance(csr_entry, pd.Series):
                            csr_entry = csr_entry.to_frame().T.reset_index()
                        else:
                            csr_entry = csr_entry.reset_index()
                        if isinstance(jpl_entry, pd.Series):
                            jpl_entry = jpl_entry.to_frame().T.reset_index()
                        else:
                            jpl_entry = jpl_entry.reset_index()
                        if isinstance(gsfc_entry, pd.Series):
                            gsfc_entry = gsfc_entry.to_frame().T.reset_index()
                        else:
                            gsfc_entry = gsfc_entry.reset_index()
                        
                        detailed_subset = detailed_indexed.loc[
                            (slice(None), threshold_perc, aquifer_bound, subbasin_id)
                        ].reset_index()
                    except (KeyError, IndexError):
                        # Fallback to filtering method
                        csr_entry = all_stats[
                            (all_stats['grace_solution'] == 'GRACE_CSR') &
                            (all_stats['threshold_perc'] == threshold_perc) &
                            (all_stats['aquifer_boundary'] == aquifer_bound) &
                            (all_stats['subbasin_id'] == subbasin_id)
                        ]
                        jpl_entry = all_stats[
                            (all_stats['grace_solution'] == 'GRACE_JPL') &
                            (all_stats['threshold_perc'] == threshold_perc) &
                            (all_stats['aquifer_boundary'] == aquifer_bound) &
                            (all_stats['subbasin_id'] == subbasin_id)
                        ]
                        gsfc_entry = all_stats[
                            (all_stats['grace_solution'] == 'GRACE_GSFC') &
                            (all_stats['threshold_perc'] == threshold_perc) &
                            (all_stats['aquifer_boundary'] == aquifer_bound) &
                            (all_stats['subbasin_id'] == subbasin_id)
                        ]
                        detailed_subset = combined_all['all_detailed'][
                            (combined_all['all_detailed']['threshold_perc'] == threshold_perc) &
                            (combined_all['all_detailed']['aquifer_boundary'] == aquifer_bound) &
                            (combined_all['all_detailed']['subbasin_id'] == subbasin_id)
                        ]
                else:
                    # Original filtering method
                    csr_entry = all_stats[
                        (all_stats['grace_solution'] == 'GRACE_CSR') &
                        (all_stats['threshold_perc'] == threshold_perc) &
                        (all_stats['aquifer_boundary'] == aquifer_bound) &
                        (all_stats['subbasin_id'] == subbasin_id)
                    ]
                    jpl_entry = all_stats[
                        (all_stats['grace_solution'] == 'GRACE_JPL') &
                        (all_stats['threshold_perc'] == threshold_perc) &
                        (all_stats['aquifer_boundary'] == aquifer_bound) &
                        (all_stats['subbasin_id'] == subbasin_id)
                    ]
                    gsfc_entry = all_stats[
                        (all_stats['grace_solution'] == 'GRACE_GSFC') &
                        (all_stats['threshold_perc'] == threshold_perc) &
                        (all_stats['aquifer_boundary'] == aquifer_bound) &
                        (all_stats['subbasin_id'] == subbasin_id)
                    ]
                    detailed_subset = combined_all['all_detailed'][
                        (combined_all['all_detailed']['threshold_perc'] == threshold_perc) &
                        (combined_all['all_detailed']['aquifer_boundary'] == aquifer_bound) &
                        (combined_all['all_detailed']['subbasin_id'] == subbasin_id)
                    ]
                
                if len(csr_entry) > 0 and len(jpl_entry) > 0 and len(gsfc_entry) > 0:
                    # Get grace_diff values for each cluster from the three solutions
                    csr_detailed = detailed_subset[detailed_subset['grace_solution'] == 'GRACE_CSR']
                    jpl_detailed = detailed_subset[detailed_subset['grace_solution'] == 'GRACE_JPL']
                    gsfc_detailed = detailed_subset[detailed_subset['grace_solution'] == 'GRACE_GSFC']
                    
                    # Match clusters by cluster_start and cluster_end
                    # Store both mean differences and std values
                    cluster_means_all = []  # All mean differences (mean of three solutions)
                    cluster_means_pos = []  # Mean differences where filtered (mean > 0 and mean > std)
                    cluster_stds_all = []   # All std values (error/uncertainty)
                    cluster_stds_pos = []   # Std values for filtered clusters
                    
                    # OPTIMIZED: Get unique cluster combinations (start, end) - use itertuples
                    csr_clusters = csr_detailed[['cluster_start', 'cluster_end']].drop_duplicates()
                    
                    # OPTIMIZED: Use itertuples() instead of iterrows() for faster iteration
                    for cluster_info in csr_clusters.itertuples():
                        cluster_start = cluster_info.cluster_start
                        cluster_end = cluster_info.cluster_end
                        
                        # Find matching clusters in all three solutions
                        csr_cluster = csr_detailed[
                            (csr_detailed['cluster_start'] == cluster_start) &
                            (csr_detailed['cluster_end'] == cluster_end)
                        ]
                        jpl_cluster = jpl_detailed[
                            (jpl_detailed['cluster_start'] == cluster_start) &
                            (jpl_detailed['cluster_end'] == cluster_end)
                        ]
                        gsfc_cluster = gsfc_detailed[
                            (gsfc_detailed['cluster_start'] == cluster_start) &
                            (gsfc_detailed['cluster_end'] == cluster_end)
                        ]
                        
                        if len(csr_cluster) > 0 and len(jpl_cluster) > 0 and len(gsfc_cluster) > 0:
                            # Get grace_diff values from the three solutions
                            grace_diffs = [
                                csr_cluster.iloc[0]['grace_diff'],
                                jpl_cluster.iloc[0]['grace_diff'],
                                gsfc_cluster.iloc[0]['grace_diff']
                            ]
                            
                            # Calculate mean and std of the three solutions
                            if all(not pd.isna(x) for x in grace_diffs):
                                mean_diff = np.mean(grace_diffs)
                                cluster_std = np.std(grace_diffs, ddof=1)
                                
                                # Store all values
                                cluster_means_all.append(mean_diff)
                                cluster_stds_all.append(cluster_std)
                                
                                # Filter: mean > 0 and mean > std (only include reliable positive changes)
                                if mean_diff > 0 and mean_diff > cluster_std:
                                    cluster_means_pos.append(mean_diff)
                                    cluster_stds_pos.append(cluster_std)
                    
                    # Calculate sums
                    # Sum of all mean differences (for all clusters)
                    mean_sum_all = sum(cluster_means_all) if cluster_means_all else 0
                    # Sum of mean differences where filtered (this is what user wants)
                    mean_sum_pos = sum(cluster_means_pos) if cluster_means_pos else 0
                    # Sum of all std values (total uncertainty across all clusters)
                    std_sum_all = sum(cluster_stds_all) if cluster_stds_all else 0
                    # Sum of std values for filtered clusters (uncertainty of reliable positive changes)
                    std_sum_pos = sum(cluster_stds_pos) if cluster_stds_pos else 0
                    
                    # Add to the stats DataFrame
                    # Update the sum values with actual means (override what was calculated per solution)
                    all_stats.loc[idx, 'grace_diff_sum'] = round(mean_sum_all, 2)
                    all_stats.loc[idx, 'grace_diff_sum_pos'] = round(mean_sum_pos, 2)
                    # Add std columns for error reporting
                    all_stats.loc[idx, 'grace_diff_std_sum_all'] = round(std_sum_all, 2)
                    all_stats.loc[idx, 'grace_diff_std_sum_pos'] = round(std_sum_pos, 2)
            
            # Update the combined_all with the updated stats
            combined_all['all_stats'] = all_stats
            
            # Recreate pivot tables to include the new std columns (combined)
            # This must happen AFTER the std columns are added
            if len(all_stats) > 0:
                pivot_tables = {}
                
                # Check if std columns have any non-null values
                has_std_all = 'grace_diff_std_sum_all' in all_stats.columns and all_stats['grace_diff_std_sum_all'].notna().any()
                has_std_pos = 'grace_diff_std_sum_pos' in all_stats.columns and all_stats['grace_diff_std_sum_pos'].notna().any()
                
                # Pivot: grace_diff_sum with std_sum_all (combined)
                if all(col in all_stats.columns for col in ['grace_solution', 'threshold_perc', 'aquifer_boundary', 'grace_diff_sum']):
                    pivot_values = ['grace_diff_sum']
                    if has_std_all:
                        pivot_values.append('grace_diff_std_sum_all')
                    
                    try:
                        pivot_tables['grace_diff_sum'] = all_stats.pivot_table(
                            values=pivot_values,
                            index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                            columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                            aggfunc='first'
                        )
                    except Exception as e:
                        print(f"Warning: Could not create grace_diff_sum pivot table: {e}")
                        # Fallback to single value
                        pivot_tables['grace_diff_sum'] = all_stats.pivot_table(
                            values='grace_diff_sum',
                            index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                            columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                            aggfunc='first'
                        )
                
                # Pivot: grace_diff_sum_pos with std_sum_pos (combined)
                if 'grace_diff_sum_pos' in all_stats.columns:
                    pivot_values = ['grace_diff_sum_pos']
                    if has_std_pos:
                        pivot_values.append('grace_diff_std_sum_pos')
                    
                    try:
                        pivot_tables['grace_diff_sum_pos'] = all_stats.pivot_table(
                            values=pivot_values,
                            index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                            columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                            aggfunc='first'
                        )
                    except Exception as e:
                        print(f"Warning: Could not create grace_diff_sum_pos pivot table: {e}")
                        # Fallback to single value
                        pivot_tables['grace_diff_sum_pos'] = all_stats.pivot_table(
                            values='grace_diff_sum_pos',
                            index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                            columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                            aggfunc='first'
                        )
                
                # Keep other existing pivot tables
                if 'pivot_tables' in combined_all:
                    for key in ['grace_diff_mean', 'grace_diff_count']:
                        if key in combined_all['pivot_tables']:
                            pivot_tables[key] = combined_all['pivot_tables'][key]
                
                combined_all['pivot_tables'] = pivot_tables
    
    # Save if requested
    if save_csv:
        import os
        os.makedirs(os.path.dirname(csv_prefix) if os.path.dirname(csv_prefix) else '.', exist_ok=True)
        
        if len(combined_all['all_detailed']) > 0:
            combined_all['all_detailed'].to_excel(f"{csv_prefix}_detailed.xlsx", index=False)
        
        if len(combined_all['all_stats']) > 0:
            combined_all['all_stats'].to_excel(f"{csv_prefix}_stats.xlsx", index=False)
        
        if 'pivot_tables' in combined_all:
            for name, pivot_df in combined_all['pivot_tables'].items():
                pivot_df.to_excel(f"{csv_prefix}_pivot_{name}.xlsx")
        
        print(f"Excel files saved with prefix: {csv_prefix}")
    
    return combined_all


def combine_all_iterations(results_list, save_csv=False, csv_prefix="all_iterations"):
    """
    Combine results from multiple loop iterations into comprehensive DataFrames.
    
    Parameters:
    -----------
    results_list : list of dict
        List of results dictionaries from summarize_responses_clustered_simple
    save_csv : bool, default=False
        If True, save combined DataFrames as CSV files
    csv_prefix : str, default="all_iterations"
        Prefix for CSV filenames
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'all_detailed': Combined detailed DataFrame from all iterations
        - 'all_stats': Combined stats DataFrame from all iterations
        - 'pivot_tables': Dictionary of pivot tables for comparison
    """
    combined_results = {}
    
    # Combine all detailed DataFrames
    detailed_dfs = [r['detailed'] for r in results_list if 'detailed' in r and len(r['detailed']) > 0]
    if detailed_dfs:
        all_detailed = pd.concat(detailed_dfs, ignore_index=True)
        all_detailed = all_detailed.sort_values(by=['grace_solution', 'threshold_perc', 'aquifer_boundary', 
                                                   'subbasin_id', 'cluster_start'])
    else:
        all_detailed = pd.DataFrame()
    
    combined_results['all_detailed'] = all_detailed
    
    # Combine all stats DataFrames
    stats_dfs = [r['stats'] for r in results_list if 'stats' in r and len(r['stats']) > 0]
    if stats_dfs:
        all_stats = pd.concat(stats_dfs, ignore_index=True)
        all_stats = all_stats.sort_values(by=['grace_solution', 'threshold_perc', 'aquifer_boundary', 'subbasin_id'])
    else:
        all_stats = pd.DataFrame()
    
    combined_results['all_stats'] = all_stats
    
    # Create pivot tables for comparison
    if len(all_stats) > 0:
        pivot_tables = {}
        
        # Pivot: grace_diff_sum by solution, threshold, and boundary
        # Include std_sum_all for GRACE_Mean (combined pivot table)
        if all(col in all_stats.columns for col in ['grace_solution', 'threshold_perc', 'aquifer_boundary', 'grace_diff_sum']):
            # Create pivot with both sum and std_sum_all values
            pivot_values = ['grace_diff_sum']
            if 'grace_diff_std_sum_all' in all_stats.columns:
                pivot_values.append('grace_diff_std_sum_all')
            
            pivot_tables['grace_diff_sum'] = all_stats.pivot_table(
                values=pivot_values,
                index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                aggfunc='first'
            )
        
        # Pivot: grace_diff_sum_pos by solution, threshold, and boundary
        # Include std_sum_pos for GRACE_Mean (combined pivot table)
        if 'grace_diff_sum_pos' in all_stats.columns:
            # Create pivot with both sum_pos and std_sum_pos values
            pivot_values = ['grace_diff_sum_pos']
            if 'grace_diff_std_sum_pos' in all_stats.columns:
                pivot_values.append('grace_diff_std_sum_pos')
            
            pivot_tables['grace_diff_sum_pos'] = all_stats.pivot_table(
                values=pivot_values,
                index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                aggfunc='first'
            )
        
        # Pivot: grace_diff_mean by solution, threshold, and boundary
        if 'grace_diff_mean' in all_stats.columns:
            pivot_tables['grace_diff_mean'] = all_stats.pivot_table(
                values='grace_diff_mean',
                index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                aggfunc='first'
            )
        
        # Pivot: grace_diff_count (number of clusters) by solution, threshold, and boundary
        if 'grace_diff_count' in all_stats.columns:
            pivot_tables['grace_diff_count'] = all_stats.pivot_table(
                values='grace_diff_count',
                index=['subbasin_id', 'aq_name'] if 'aq_name' in all_stats.columns else 'subbasin_id',
                columns=['grace_solution', 'threshold_perc', 'aquifer_boundary'],
                aggfunc='first'
            )
        
        combined_results['pivot_tables'] = pivot_tables
    
    # Save to CSV if requested
    if save_csv:
        import os
        os.makedirs(os.path.dirname(csv_prefix) if os.path.dirname(csv_prefix) else '.', exist_ok=True)
        
        if len(all_detailed) > 0:
            all_detailed.to_excel(f"{csv_prefix}_detailed.xlsx", index=False)
        
        if len(all_stats) > 0:
            all_stats.to_excel(f"{csv_prefix}_stats.xlsx", index=False)
        
        if 'pivot_tables' in combined_results:
            for name, pivot_df in combined_results['pivot_tables'].items():
                pivot_df.to_excel(f"{csv_prefix}_pivot_{name}.xlsx")
        
        print(f"Excel files saved with prefix: {csv_prefix}")
    
    return combined_results


def extract_stats_to_dataframe(results_dict):
    """
    Extract statistics from results_dict into a comprehensive DataFrame.
    Updated to handle aquifer boundaries in the analysis.
    
    Parameters:
    - results_dict: Dictionary containing analysis results from multiple GRACE solutions, thresholds, and aquifer boundaries
    
    Returns:
    - comparison_df: DataFrame with all statistics for easy comparison
    """
    stats_records = []

    for key, data in results_dict.items():
        stats_df = data['stats']
        
        if len(stats_df) > 0:  # Check if we have statistics
            # Get the single row (since we used multiple=False)
            stats_row = stats_df.iloc[0]
            
            record = {
                'GRACE_Solution': data['grace_solution'],
                'Threshold_Percentile': f"{data['threshold']*100:.0f}th",
                'Threshold_Value': data['threshold'],
                'N_Events': stats_row['n_events'],
                'Trend_Slope': stats_row['Trend (Slope)'],
                'Intercept': stats_row['Intercept'],
                'R_Squared': stats_row['R²'],
                'P_Value_Linear': stats_row['p-value (Linear)'],
                'Kendall_Tau': stats_row['Kendall Tau'],
                'P_Value_Kendall': stats_row['p-value (Kendall)'],
                'Spearman_R': stats_row['Spearman R'],
                'P_Value_Spearman': stats_row['p-value (Spearman)'],
                'Trend_Significant': stats_row['Trend Sig?']
            }
            
            # Add aquifer boundary information if available
            if 'aquifer_boundary' in data:
                record['Aquifer_Boundary'] = data['aquifer_boundary']
            
            stats_records.append(record)
        else:
            print(f"Warning: No statistics found for {key}")

    # Create comprehensive DataFrame
    comparison_df = pd.DataFrame(stats_records)

    # Sort by GRACE solution, threshold, and aquifer boundary for better readability
    if 'Aquifer_Boundary' in comparison_df.columns:
        comparison_df = comparison_df.sort_values(['GRACE_Solution', 'Threshold_Value', 'Aquifer_Boundary']).reset_index(drop=True)
    else:
        comparison_df = comparison_df.sort_values(['GRACE_Solution', 'Threshold_Value']).reset_index(drop=True)
    
    print(f"Total combinations analyzed: {len(comparison_df)}")
    
    return comparison_df


def display_comparison_table(comparison_df):
    """
    Display the comparison table with proper formatting.
    
    Parameters:
    - comparison_df: DataFrame with statistics from extract_stats_to_dataframe()
    """
    # Display with proper formatting
    display_df = comparison_df.copy()

    # Format numerical columns for better readability
    display_df['Trend_Slope'] = display_df['Trend_Slope'].round(2)
    display_df['R_Squared'] = display_df['R_Squared'].round(2)
    display_df['P_Value_Linear'] = display_df['P_Value_Linear'].apply(format_pvalue)
    display_df['Kendall_Tau'] = display_df['Kendall_Tau'].round(2)
    display_df['Spearman_R'] = display_df['Spearman_R'].round(2)
    display_df['P_Value_Kendall'] = display_df['P_Value_Kendall'].apply(format_pvalue)
    display_df['P_Value_Spearman'] = display_df['P_Value_Spearman'].apply(format_pvalue)

    # Display the table
    display(display_df)

def generate_analysis_insights(comparison_df):
    """
    Generate detailed analysis insights from the comparison DataFrame.
    Updated to handle aquifer boundaries in the analysis.
    
    Parameters:
    - comparison_df: DataFrame with statistics from extract_stats_to_dataframe()
    """
    
    # Check if we have aquifer boundaries in the data
    has_aquifer_boundaries = 'Aquifer_Boundary' in comparison_df.columns

    # 1. Best performing GRACE solution
    best_r2_idx = comparison_df['R_Squared'].idxmax()
    best_combo = comparison_df.loc[best_r2_idx]
    print(f"🏆 Best R² Performance:")
    print(f"   GRACE Solution: {best_combo['GRACE_Solution']}")
    print(f"   Threshold: {best_combo['Threshold_Percentile']}")
    if has_aquifer_boundaries:
        print(f"   Aquifer Boundary: {best_combo['Aquifer_Boundary']}")
    print(f"   R² = {best_combo['R_Squared']:.2f}")
    print(f"   P-value = {format_pvalue(best_combo['P_Value_Linear'])}")
    print(f"   Significant: {best_combo['Trend_Significant']}")

    print("\n" + "-" * 50)

    # 2. Significant trends summary
    significant_df = comparison_df[comparison_df['Trend_Significant'] == 'Yes']
    print(f"📊 Significant Trends Summary:")
    print(f"   Total significant combinations: {len(significant_df)}/{len(comparison_df)}")

    if len(significant_df) > 0:
        print(f"   Top significant combinations:")
        # Sort by R² and show top 10 (OPTIMIZED: use itertuples for small performance gain)
        top_significant = significant_df.nlargest(10, 'R_Squared')
        for row in top_significant.itertuples():
            # OPTIMIZED: Use getattr for namedtuple access (itertuples returns namedtuple)
            grace_sol = getattr(row, 'GRACE_Solution', None)
            threshold = getattr(row, 'Threshold_Percentile', None)
            aquifer_bound = getattr(row, 'Aquifer_Boundary', None) if has_aquifer_boundaries else None
            r_squared = getattr(row, 'R_Squared', None)
            
            if has_aquifer_boundaries:
                print(f"     - {grace_sol} ({threshold}, {aquifer_bound}): R²={r_squared:.2f}")
            else:
                print(f"     - {grace_sol} ({threshold}): R²={r_squared:.2f}")

    print("\n" + "-" * 50)

    # 3. Aquifer boundary comparison (if available)
    if has_aquifer_boundaries:
        print("🏞️ Aquifer Boundary Comparison (Average R²):")
        aquifer_avg_r2 = comparison_df.groupby('Aquifer_Boundary')['R_Squared'].mean().sort_values(ascending=False)
        for i, (aquifer, avg_r2) in enumerate(aquifer_avg_r2.items(), 1):
            print(f"   {i}. {aquifer}: R²={avg_r2:.2f}")
        
        print("\n" + "-" * 50)

    # 4. Threshold comparison
    print("📈 Threshold Comparison (95th vs 99th):")
    for grace_sol in comparison_df['GRACE_Solution'].unique():
        sol_data = comparison_df[comparison_df['GRACE_Solution'] == grace_sol]
        if has_aquifer_boundaries:
            # Average across aquifer boundaries for threshold comparison
            sol_data_avg = sol_data.groupby('Threshold_Percentile')['R_Squared'].mean()
            if '95th' in sol_data_avg.index and '99th' in sol_data_avg.index:
                r2_diff = sol_data_avg['99th'] - sol_data_avg['95th']
                print(f"   {grace_sol}:")
                print(f"     95th: R²={sol_data_avg['95th']:.2f}, 99th: R²={sol_data_avg['99th']:.2f}, Δ={r2_diff:+.2f}")
        else:
            if len(sol_data) == 2:
                p95 = sol_data[sol_data['Threshold_Percentile'] == '95th'].iloc[0]
                p99 = sol_data[sol_data['Threshold_Percentile'] == '99th'].iloc[0]
                r2_diff = p99['R_Squared'] - p95['R_Squared']
                print(f"   {grace_sol}:")
                print(f"     95th: R²={p95['R_Squared']:.2f}, 99th: R²={p99['R_Squared']:.2f}, Δ={r2_diff:+.2f}")

    print("\n" + "-" * 50)

    # 5. GRACE solution ranking
    print("🥇 GRACE Solution Ranking (by average R²):")
    grace_avg_r2 = comparison_df.groupby('GRACE_Solution')['R_Squared'].mean().sort_values(ascending=False)
    for i, (grace_sol, avg_r2) in enumerate(grace_avg_r2.items(), 1):
        print(f"   {i}. {grace_sol}: R²={avg_r2:.2f}")

    print("\n" + "=" * 50)
    print("✅ Analysis complete! Use 'comparison_df' for further analysis.")


def create_comparison_pivot_tables(comparison_df):
    """
    Create pivot tables for easier comparison of statistics.
    Updated to handle aquifer boundaries in the analysis.
    
    Parameters:
    - comparison_df: DataFrame with statistics from extract_stats_to_dataframe()
    
    Returns:
    - Dictionary containing all pivot tables
    """
    print("📊 Pivot Tables for Easy Comparison:")
    print("=" * 60)

    # Check if we have aquifer boundaries in the data
    has_aquifer_boundaries = 'Aquifer_Boundary' in comparison_df.columns
    
    if has_aquifer_boundaries:
        print("\n🎯 R² Values Comparison (by Aquifer Boundary):")
        
        # Create pivot table with aquifer boundary as additional dimension
        r2_pivot = comparison_df.pivot_table(
            index=['GRACE_Solution', 'Aquifer_Boundary'], 
            columns='Threshold_Percentile', 
            values='R_Squared',
            aggfunc='mean'  # In case of duplicates, take mean
        ).round(2)
        display(r2_pivot)

        print("\n📈 P-values Comparison (by Aquifer Boundary):")
        pval_pivot = comparison_df.pivot_table(
            index=['GRACE_Solution', 'Aquifer_Boundary'], 
            columns='Threshold_Percentile', 
            values='P_Value_Linear',
            aggfunc='mean'
        )
        display(pval_pivot)

        print("\n✅ Trend Significance (by Aquifer Boundary):")
        sig_pivot = comparison_df.pivot_table(
            index=['GRACE_Solution', 'Aquifer_Boundary'], 
            columns='Threshold_Percentile', 
            values='Trend_Significant',
            aggfunc=lambda x: x.iloc[0] if len(x) > 0 else 'No'  # Take first value for significance
        )
        display(sig_pivot)

        print("\n📊 Number of Events (by Aquifer Boundary):")
        events_pivot = comparison_df.pivot_table(
            index=['GRACE_Solution', 'Aquifer_Boundary'], 
            columns='Threshold_Percentile', 
            values='N_Events',
            aggfunc='mean'
        ).round(0)
        display(events_pivot)
        
        # Also create simplified pivot tables without aquifer boundary for comparison
        print("\n" + "="*60)
        print("📊 Simplified Comparison (Averaged Across Aquifer Boundaries):")
        print("="*60)
        
        # Average across aquifer boundaries
        simplified_df = comparison_df.groupby(['GRACE_Solution', 'Threshold_Percentile']).agg({
            'R_Squared': 'mean',
            'P_Value_Linear': 'mean', 
            'Trend_Significant': lambda x: 'Yes' if x.value_counts().get('Yes', 0) > len(x)/2 else 'No',
            'N_Events': 'mean'
        }).reset_index()
        
        print("\n🎯 R² Values (Averaged):")
        r2_pivot_simple = simplified_df.pivot(index='GRACE_Solution', columns='Threshold_Percentile', values='R_Squared').round(2)
        display(r2_pivot_simple)
        
    else:
        # Original pivot tables for backward compatibility
        print("\n🎯 R² Values Comparison:")
        r2_pivot = comparison_df.pivot(index='GRACE_Solution', columns='Threshold_Percentile', values='R_Squared')
        r2_pivot = r2_pivot.round(2)
        display(r2_pivot)

        print("\n📈 P-values Comparison:")
        pval_pivot = comparison_df.pivot(index='GRACE_Solution', columns='Threshold_Percentile', values='P_Value_Linear')
        display(pval_pivot)

        print("\n✅ Trend Significance:")
        sig_pivot = comparison_df.pivot(index='GRACE_Solution', columns='Threshold_Percentile', values='Trend_Significant')
        display(sig_pivot)

        print("\n📊 Number of Events:")
        events_pivot = comparison_df.pivot(index='GRACE_Solution', columns='Threshold_Percentile', values='N_Events')
        display(events_pivot)
  
    # Return pivot tables for further use
    pivot_tables = {
        'r2_values': r2_pivot,
        'p_values': pval_pivot,
        'significance': sig_pivot,
        'n_events': events_pivot
    }
    
    if has_aquifer_boundaries:
        pivot_tables['simplified_r2'] = r2_pivot_simple
    
    return pivot_tables


def analyze_results_comprehensive(results_dict):
    """
    Complete analysis pipeline that extracts stats, displays tables, and provides insights.
    
    Parameters:
    - results_dict: Dictionary containing analysis results from multiple GRACE solutions and thresholds
    
    Returns:
    - comparison_df: DataFrame with all statistics
    - pivot_tables: Dictionary containing pivot tables
    """
    
    # Step 1: Extract statistics to DataFrame
    comparison_df = extract_stats_to_dataframe(results_dict)
    
    # Step 2: Display formatted table
    display_comparison_table(comparison_df)
    
    # Step 3: Generate insights
    generate_analysis_insights(comparison_df)
    
    # Step 4: Create pivot tables
    pivot_tables = create_comparison_pivot_tables(comparison_df)

    
    return comparison_df, pivot_tables


def plot_grace_precip_correlation_temporal(
    gdf, grace_data, precip_data, aquifer_boundaries=None, aquifer_ids=None, 
    figsize=(12, 6), save_dir=None, max_lag=12, plot_TWS=False, annotate_max=True
):
    """
    Plot GRACE and cumulative precipitation anomaly time series for each aquifer.
    Loops through different aquifer boundaries and calculates correlation for residual (and optionally TWS).
    Calculates correlation at lag 0 and maximum correlation across lags.
    
    Parameters:
    -----------
    gdf : GeoDataFrame
        GeoDataFrame containing aquifer geometries and metadata (base/Full boundary)
    grace_data : xarray.DataArray
        GRACE TWS data
    precip_data : xarray.DataArray  
        Precipitation data
    aquifer_boundaries : dict, optional
        Dictionary with boundary names as keys and GeoDataFrames as values.
        If None, uses only the provided gdf
    aquifer_ids : list, optional
        List of aquifer IDs to plot. If None, plots all aquifers
    figsize : tuple
        Figure size for the plot
    save_dir : str, optional
        Directory to save plots and Excel file
    max_lag : int, default=12
        Maximum lag (in months) to test for correlation.
        Positive lag means precipitation leads (precip happens first, GRACE responds later).
        Negative lag means GRACE leads (GRACE changes first, precip happens later).
        Example: lag=2 means precip at month t correlates with GRACE at month t+2.
        Example: lag=-2 means GRACE at month t correlates with precip at month t+2.
    plot_TWS : bool, default=False
        If True, also plots TWS in addition to Residual. If False, only plots Residual.
    annotate_max : bool, default=True
        If True, shows both "Lag 0: # (p=...)" and "Max: # (p=...)" annotations.
        If False, shows only "R: # (p=...)" annotation.
        
    Returns:
    --------
    dict : Dictionary containing:
        - 'correlation_stats': Dictionary with detailed correlation statistics
        - 'correlation_df': DataFrame with lag 0 and max correlation for each case
    """
    
    # OPTIMIZED: Move CRS operations before loops, remove unnecessary .copy()
    gdf = gdf.to_crs("EPSG:4326")
    grace_data.rio.write_crs("EPSG:4326", inplace=True)
    precip_data.rio.write_crs("EPSG:4326", inplace=True)
    
    # Set up boundaries - if not provided, use gdf as 'Full'
    if aquifer_boundaries is None:
        aquifer_boundaries = {'Full': gdf}
    else:
        # OPTIMIZED: Removed unnecessary .copy() calls
        # Ensure all boundaries have correct CRS (done once, not in loop)
        for name, boundary_gdf in aquifer_boundaries.items():
            aquifer_boundaries[name] = boundary_gdf.to_crs("EPSG:4326")
    
    # Filter aquifer IDs if specified
    if aquifer_ids is None:
        # Get aquifer IDs from the base gdf
        aquifer_ids = gdf['subbasin_id'].tolist()
    else:
        aquifer_ids = [aid for aid in aquifer_ids if aid in gdf['subbasin_id'].values]
    
    print(f"Plotting temporal correlations for aquifers: {aquifer_ids}")
    print(f"Boundaries: {list(aquifer_boundaries.keys())}")
    
    correlation_stats = {}
    
    # Loop through each boundary
    for boundary_name, boundary_gdf in aquifer_boundaries.items():
        print(f"\n{'='*60}")
        print(f"Processing boundary: {boundary_name}")
        print(f"{'='*60}")
        
        # Loop through each aquifer
        for aquifer_id in aquifer_ids:
            # Check if this aquifer exists in this boundary
            if aquifer_id not in boundary_gdf['subbasin_id'].values:
                continue
            
            # OPTIMIZED: Use query or boolean indexing more efficiently
            boundary_subset = boundary_gdf[boundary_gdf['subbasin_id'] == aquifer_id]
            if len(boundary_subset) == 0:
                continue
            row = boundary_subset.iloc[0]
            geom = [row.geometry.__geo_interface__]
            
            # Get aquifer name (prefer aq_name, fallback to other names)
            aquifer_name = row.get('aq_name', row.get('Aquifer_sy', f'Aquifer {aquifer_id}'))
            
            try:
                # Clip data to aquifer
                grace_clip = grace_data.rio.clip(geom, drop=True)
                precip_clip = precip_data.rio.clip(geom, drop=True)
                
                # Calculate spatial means
                grace_ts = grace_clip.mean(dim=["lat", "lon"], skipna=True)
                precip_ts = precip_clip.mean(dim=["lat", "lon"], skipna=True)
                
                # Align time series
                grace_ts, precip_ts = xr.align(grace_ts, precip_ts, join="inner")
                
                # Decompose GRACE to get residual
                time = grace_ts.time
                trend, season, residual = decompose_grace_sin_cosin(grace_ts, time)
                
                # OPTIMIZED: Calculate precipitation anomaly using helper function
                precip_anomaly = _calculate_precip_anomaly(precip_ts)
                
                # Process Residual (always) and TWS (if plot_TWS is True)
                grace_types_to_plot = [('Residual', residual)]
                if plot_TWS:
                    grace_types_to_plot.append(('TWS', grace_ts))
                
                for grace_type, grace_series_data in grace_types_to_plot:
                    # Convert to pandas for easier handling
                    if grace_type == 'TWS':
                        grace_series = pd.Series(grace_series_data.values, 
                                                index=pd.to_datetime(time.values), 
                                                name="GRACE TWS")
                        ylabel = "GRACE TWS (cm)"
                        color = 'blue'
                    else:
                        grace_series = pd.Series(grace_series_data.values, 
                                                index=pd.to_datetime(time.values), 
                                                name="GRACE Residual")
                        ylabel = "GRACE TWS Residual (cm)"
                        color = 'blue'
                    
                    precip_series = pd.Series(precip_anomaly.values, 
                                            index=pd.to_datetime(time.values), 
                                            name="Precip Anomaly")
                    
                    # Calculate correlation at lag 0
                    from scipy.stats import pearsonr
                    corr_lag0, p_value_lag0 = pearsonr(grace_series.values, precip_series.values)
                    
                    # Calculate correlations at different lags to find maximum
                    grace_values = grace_series.values
                    precip_values = precip_series.values
                    
                    best_corr = corr_lag0
                    best_p = p_value_lag0
                    best_lag = 0
                    
                    lags = range(-max_lag, max_lag + 1)
                    for lag in lags:
                        if lag == 0:
                            continue
                        
                        if lag > 0:
                            # Positive lag: Precip leads (precip happens first, GRACE responds later)
                            # Compares precip[t] with grace[t+lag]
                            # Example: lag=2 means precip at month t correlates with GRACE at month t+2
                            shifted_precip = precip_values[:-lag]
                            shifted_grace = grace_values[lag:]
                        else:
                            # Negative lag: GRACE leads (GRACE changes first, precip happens later)
                            # Compares precip[t-lag] with grace[t]
                            # Example: lag=-2 means GRACE at month t correlates with precip at month t+2
                            shifted_precip = precip_values[-lag:]
                            shifted_grace = grace_values[:lag]
                        
                        # Remove NaN values
                        valid = ~(np.isnan(shifted_precip) | np.isnan(shifted_grace))
                        if np.sum(valid) < 5:
                            continue
                        
                        r, p = pearsonr(shifted_precip[valid], shifted_grace[valid])
                        
                        # Check if this is the best correlation (by absolute value)
                        if r > best_corr:
                            best_corr = r
                            best_p = p
                            best_lag = lag
                    
                    # Store statistics
                    key = f"{boundary_name}_{aquifer_id}_{grace_type}"
                    correlation_stats[key] = {
                        'boundary': boundary_name,
                        'aquifer_id': aquifer_id,
                        'aquifer_name': aquifer_name,
                        'grace_type': grace_type,
                        'correlation_lag0': corr_lag0,
                        'p_value_lag0': p_value_lag0,
                        'correlation_max': best_corr,
                        'p_value_max': best_p,
                        'lag_max_corr': best_lag,
                        'n_observations': len(grace_series)
                    }
                    
                    # Use lag 0 correlation for display
                    corr_coef = corr_lag0
                    p_value = p_value_lag0
                    
                    # Create standardized plot
                    fig, ax1 = plt.subplots(figsize=figsize)
                    
                    # Calculate y-axis limits with 2 cm padding for GRACE
                    grace_min = np.nanmin(grace_series.values)
                    grace_max = np.nanmax(grace_series.values)
                    ylim_min = grace_min - 2  # Add 2 cm
                    ylim_max = grace_max + 5  # Add 2 cm
                    
                    # Plot GRACE line (without label since axis labels are colored)
                    ax1.plot(grace_series.index, grace_series.values, 
                            color=color, lw=1.5, alpha=0.8)
                    ax1.set_ylabel(ylabel, fontsize=14, color=color)  # Smaller font size
                    ax1.set_ylim(ylim_min, ylim_max)
                    ax1.tick_params(axis='y', labelcolor=color,labelsize=12)
                    # ax1.set_title(f"{aquifer_name} ({boundary_name}): GRACE {grace_type}", 
                    #             fontsize=10, pad=4)  # Smaller title size, pad=0.1
                    ax1.grid(True, linestyle='--', alpha=0.6)
                    
                    # Format x-axis
                    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
                    ax1.xaxis.set_minor_locator(mdates.YearLocator(1))
                    ax1.xaxis.set_major_locator(mdates.YearLocator(2))
                    plt.xticks(rotation=45, fontsize=12)
                    
                    # Create secondary axis for precipitation
                    ax2 = ax1.twinx()
                    ax2.plot(precip_series.index, precip_series.values, 
                            color='red', lw=1.5, alpha=0.8)  # No label since axis is colored
                    ax2.set_ylabel("Cum. Prec. Anomaly (mm)", fontsize=14, color='red')  # Smaller font size
                    ax2.tick_params(axis='y', labelcolor='red',labelsize=12)
                    
                    precip_min = np.nanmin(precip_series.values)
                    precip_max = np.nanmax(precip_series.values)
                    ylim_min = precip_min - 20  # Add 2 cm
                    ylim_max = precip_max + 40  # Add 2 cm
                    ax2.set_ylim(ylim_min, ylim_max)
                    
                    # Add correlation annotation based on annotate_max flag
                    if annotate_max:
                        annotation_text = f'Lag 0: {corr_lag0:.2f} (p={format_pvalue(p_value_lag0)})'
                        if best_lag != 0:
                            annotation_text += f'\nMax: {best_corr:.2f} at lag {best_lag} (p={format_pvalue(best_p)})'
                        else:
                            annotation_text += f'\nMax: {best_corr:.2f} (at lag 0)'
                    else:
                        annotation_text = f'r= {corr_lag0:.2f} (p={format_pvalue(p_value_lag0)})'
                    
                    ax1.text(0.02, 0.95, 
                            annotation_text, 
                            transform=ax1.transAxes, 
                            verticalalignment='top',
                            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
                            fontsize=14)
                    
                    # No legend needed since axis labels are color-coded
                    
                    plt.tight_layout()
                    
                    # Save if path specified
                    if save_dir:
                        save_path = Path(save_dir)
                        save_path.mkdir(parents=True, exist_ok=True)
                        figure_name = f"grace_pcp_anomaly_{boundary_name}_aquifer{aquifer_id}_{grace_type.lower()}.jpeg"
                        full_path = save_path / figure_name
                        plt.savefig(full_path, dpi=500, bbox_inches='tight')
                    
                    plt.show()
                    
                    print(f"  {aquifer_name} ({boundary_name}) - {grace_type}: lag0={corr_lag0:.3f}, max={best_corr:.3f} (lag={best_lag})")
            
            except Exception as e:
                print(f"[Aquifer {aquifer_id} ({boundary_name})] skipped due to error: {e}")
                continue
    
    # Create DataFrame with correlation results
    correlation_records = []
    for key, stats in correlation_stats.items():
        correlation_records.append({
            'boundary': stats['boundary'],
            'aquifer_id': stats['aquifer_id'],
            'aquifer_name': stats['aquifer_name'],
            'grace_type': stats['grace_type'],
            'correlation_lag0': stats['correlation_lag0'],
            'p_value_lag0': stats['p_value_lag0'],
            'correlation_max': stats['correlation_max'],
            'p_value_max': stats['p_value_max'],
            'lag_max_corr': stats['lag_max_corr'],
            'n_observations': stats['n_observations']
        })
    
    correlation_df = pd.DataFrame(correlation_records)
    
    # Sort by boundary, aquifer_id, and grace_type
    if len(correlation_df) > 0:
        correlation_df = correlation_df.sort_values(by=['boundary', 'aquifer_id', 'grace_type']).reset_index(drop=True)
    
    # Save DataFrame to Excel if save_dir is provided
    if save_dir and len(correlation_df) > 0:
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        excel_path = save_path / "correlation_results.xlsx"
        correlation_df.to_excel(excel_path, index=False)
        print(f"\nCorrelation results saved to: {excel_path}")
    
    return {
        'correlation_stats': correlation_stats,
        'correlation_df': correlation_df
    }


# Backward-compatible alias (legacy notebooks may still import the old name).
