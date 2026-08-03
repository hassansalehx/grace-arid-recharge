"""Reusable EDA plotting helpers for gridded climate/LSM datasets."""

from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes as _MplAxes
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import xarray as xr


def _import_cartopy():
    """Lazy-import cartopy so pure helpers remain importable without it."""
    try:
        import cartopy.crs as ccrs
        from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "Cartopy is required for map plotting but is not installed. "
            "Install it with `mamba install -c conda-forge cartopy` "
            "or `conda install -c conda-forge cartopy`."
        ) from e
    return ccrs, LONGITUDE_FORMATTER, LATITUDE_FORMATTER


def _infer_lon_lat_names(da: xr.DataArray) -> Tuple[str, str]:
    lon_candidates = ("lon", "longitude", "x", "X")
    lat_candidates = ("lat", "latitude", "y", "Y")
    lon_name = next((c for c in lon_candidates if c in da.coords), None)
    lat_name = next((c for c in lat_candidates if c in da.coords), None)
    if lon_name is None or lat_name is None:
        raise ValueError("Could not infer lon/lat coordinate names from DataArray.")
    return lon_name, lat_name


def _resolve_units(da: xr.DataArray) -> str:
    for key in ("units", "unit"):
        value = da.attrs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return "unknown"


def _normalize_units_label(units: str) -> str:
    return units.strip().lower().replace(" ", "")


def get_display_units_and_values(
    da: xr.DataArray,
    target_units: Optional[str] = None,
) -> Tuple[xr.DataArray, str, str]:
    """Return display DataArray, units label, and conversion description.

    Keeps native units by default. If target_units is provided as "mm" or "cm",
    attempts conversion only for known depth-like units.
    """
    native_units = _resolve_units(da)
    if target_units is None:
        return da, native_units, "native"

    target = target_units.lower().strip()
    if target not in {"mm", "cm"}:
        raise ValueError("target_units must be one of: None, 'mm', 'cm'.")

    normalized = _normalize_units_label(native_units)
    factor = None

    # Common depth forms
    if normalized in {"m"}:
        factor = 1000.0 if target == "mm" else 100.0
    elif normalized in {"cm"}:
        factor = 10.0 if target == "mm" else 1.0
    elif normalized in {"mm"}:
        factor = 1.0 if target == "mm" else 0.1
    elif normalized in {"kgm-2", "kg/m^2", "kg/m2"}:
        # Water-equivalent depth assumption: 1 kg m-2 = 1 mm
        factor = 1.0 if target == "mm" else 0.1
    elif "mm/day" in normalized or "mm/d" in normalized:
        if target == "mm":
            factor = 1.0
    elif "mm/hr" in normalized or "mm/h" in normalized:
        if target == "mm":
            factor = 1.0

    if factor is None:
        # Ambiguous conversion; keep native.
        return da, native_units, "native (conversion skipped)"

    converted = da * factor
    converted.attrs = dict(da.attrs)
    converted.attrs["units"] = target
    return converted, target, f"{native_units} -> {target}"


def _units_label_with_conversion_note(units_label: str, conversion_note: str) -> str:
    """Append conversion-skip reason to colorbar/axis labels when relevant."""
    if conversion_note == "native (conversion skipped)":
        return f"{units_label} ({conversion_note})"
    return units_label


def compute_plot_limits(
    da: xr.DataArray,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    qmin: Optional[float] = None,
    qmax: Optional[float] = None,
) -> Tuple[Optional[float], Optional[float]]:
    data = da.values
    values = data[np.isfinite(data)]
    if values.size == 0:
        if qmin is not None or qmax is not None:
            warnings.warn(
                "compute_plot_limits: no finite values; returning provided vmin/vmax."
            )
        return vmin, vmax
    if vmin is None and qmin is not None:
        vmin = float(np.nanpercentile(values, qmin))
    if vmax is None and qmax is not None:
        vmax = float(np.nanpercentile(values, qmax))
    return vmin, vmax


def _setup_geo_map_axis(
    ax,
    extent: Sequence[float],
    *,
    ccrs,
    LONGITUDE_FORMATTER,
    LATITUDE_FORMATTER,
    tick_fontsize: int = 14,
    fixed_tick_locators: bool = False,
    map_aspect: Optional[str] = None,
):
    """Shared Cartopy extent / coastlines / gridline setup (visual defaults preserved)."""
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    if map_aspect is not None:
        ax.set_aspect(map_aspect)
    ax.coastlines(resolution="110m", linewidth=1)

    gl = ax.gridlines(
        crs=ccrs.PlateCarree(),
        draw_labels=True,
        linewidth=1,
        color="black",
        linestyle="--",
    )
    gl.top_labels = False
    gl.right_labels = False
    gl.left_labels = True
    gl.bottom_labels = True
    if fixed_tick_locators:
        gl.xlines = True
        lon_ticks = np.linspace(extent[0], extent[1], 5)
        lat_ticks = np.linspace(extent[2], extent[3], 5)
        gl.xlocator = mticker.FixedLocator(lon_ticks)
        gl.ylocator = mticker.FixedLocator(lat_ticks)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": tick_fontsize, "color": "black"}
    gl.ylabel_style = {"size": tick_fontsize, "color": "black"}
    return gl


def _add_map_colorbar(
    mesh,
    ax,
    *,
    label: str,
    colorbar_extend: str = "neither",
    colorbar_pad: float = 0.02,
    label_fontsize: int = 16,
    tick_fontsize: int = 14,
    colorbar_fn=None,
):
    """Shared colorbar append (plain Axes; same size/pad defaults as before)."""
    divider = make_axes_locatable(ax)
    # Plain Axes only: append_axes would otherwise clone GeoAxes without projection.
    cax = divider.append_axes("right", size="3%", pad=colorbar_pad, axes_class=_MplAxes)
    cb_fn = colorbar_fn or plt.colorbar
    cb = cb_fn(mesh, cax=cax, extend=colorbar_extend)
    cb.set_label(label, size=label_fontsize)
    cb.ax.tick_params(labelsize=tick_fontsize)
    return cb


def plot_geo_map(
    da: xr.DataArray,
    *,
    title: Optional[str] = None,
    target_units: Optional[str] = None,
    cmap: str = "rainbow",
    extent: Sequence[float] = (-150, 150, -60, 60),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    qmin: Optional[float] = None,
    qmax: Optional[float] = None,
    colorbar_extend: str = "neither",
    colorbar_pad: float = 0.02,
    figsize: Tuple[float, float] = (10, 4),
    tick_fontsize: int = 14,
    label_fontsize: int = 16,
    title_fontsize: int = 20,
):
    ccrs, LONGITUDE_FORMATTER, LATITUDE_FORMATTER = _import_cartopy()
    display_da, units_label, conversion_note = get_display_units_and_values(
        da, target_units=target_units
    )
    units_label = _units_label_with_conversion_note(units_label, conversion_note)
    vmin, vmax = compute_plot_limits(display_da, vmin=vmin, vmax=vmax, qmin=qmin, qmax=qmax)
    lon_name, lat_name = _infer_lon_lat_names(display_da)
    lon = display_da[lon_name].values
    lat = display_da[lat_name].values
    arr = display_da.values

    fig = plt.figure(figsize=figsize)
    ax = plt.axes(projection=ccrs.PlateCarree())
    _setup_geo_map_axis(
        ax,
        extent,
        ccrs=ccrs,
        LONGITUDE_FORMATTER=LONGITUDE_FORMATTER,
        LATITUDE_FORMATTER=LATITUDE_FORMATTER,
        tick_fontsize=tick_fontsize,
        fixed_tick_locators=True,
    )

    mesh = ax.pcolormesh(
        lon,
        lat,
        arr,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    _add_map_colorbar(
        mesh,
        ax,
        label=units_label,
        colorbar_extend=colorbar_extend,
        colorbar_pad=colorbar_pad,
        label_fontsize=label_fontsize,
        tick_fontsize=tick_fontsize,
    )

    map_title = title
    if conversion_note == "native (conversion skipped)":
        map_title = f"{title} [{conversion_note}]" if title else conversion_note
    if map_title:
        plt.title(map_title, size=title_fontsize)
    return fig, ax


def plot_value_distribution(
    da: xr.DataArray,
    *,
    target_units: Optional[str] = None,
    bins: int = 80,
    qmin: Optional[float] = None,
    qmax: Optional[float] = None,
    log_x: bool = False,
    title: Optional[str] = None,
    xlabel: Optional[str] = None,
    figsize: Tuple[float, float] = (10, 4),
):
    display_da, units_label, conversion_note = get_display_units_and_values(
        da, target_units=target_units
    )
    units_label = _units_label_with_conversion_note(units_label, conversion_note)
    data = display_da.values
    values = data[np.isfinite(data)]

    fig, ax = plt.subplots(figsize=figsize)
    if values.size == 0:
        warnings.warn("plot_value_distribution: no finite values to plot.")
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel(xlabel or units_label, fontsize=12)
        ax.tick_params(labelsize=11)
        dist_title = title
        if conversion_note == "native (conversion skipped)":
            dist_title = f"{title} [{conversion_note}]" if title else conversion_note
        if dist_title:
            ax.set_title(dist_title, fontsize=13)
        return fig, ax

    if qmin is not None:
        lo = np.nanpercentile(values, qmin)
        values = values[values >= lo]
    if qmax is not None:
        hi = np.nanpercentile(values, qmax)
        values = values[values <= hi]

    if values.size == 0:
        warnings.warn("plot_value_distribution: no values remain after quantile filtering.")
        ax.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax.transAxes)
        ax.set_ylabel("Count", fontsize=12)
        ax.set_xlabel(xlabel or units_label, fontsize=12)
        ax.tick_params(labelsize=11)
        if title:
            ax.set_title(title, fontsize=13)
        return fig, ax

    ax.hist(values, bins=bins, alpha=0.85, edgecolor="black")
    ax.grid(True, alpha=0.35)
    ax.set_ylabel("Count", fontsize=12)
    ax.set_xlabel(xlabel or units_label, fontsize=12)
    ax.tick_params(labelsize=11)
    if log_x:
        ax.set_xscale("log")

    stats_text = (
        f"n={values.size}\n"
        f"mean={np.nanmean(values):.3g}\n"
        f"median={np.nanmedian(values):.3g}\n"
        f"p95={np.nanpercentile(values, 95):.3g}"
    )
    ax.text(0.99, 0.98, stats_text, transform=ax.transAxes, va="top", ha="right")
    dist_title = title
    if conversion_note == "native (conversion skipped)":
        dist_title = f"{title} [{conversion_note}]" if title else conversion_note
    if dist_title:
        ax.set_title(dist_title, fontsize=13)
    return fig, ax


def plot_map_with_distribution(
    da: xr.DataArray,
    *,
    map_title: Optional[str] = None,
    dist_title: Optional[str] = None,
    target_units: Optional[str] = None,
    cmap: str = "rainbow",
    extent: Sequence[float] = (-150, 150, -60, 60),
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    qmin: Optional[float] = None,
    qmax: Optional[float] = None,
    colorbar_extend: str = "both",
    colorbar_pad: float = 0.01,
    map_tick_fontsize: int = 12,
    layout: str = "horizontal",
    figsize: Optional[Tuple[float, float]] = None,
    hist_bin_width: Optional[float] = 1.0,
    hist_bins: Optional[int] = None,
    map_aspect: Optional[str] = "auto",
):
    ccrs, LONGITUDE_FORMATTER, LATITUDE_FORMATTER = _import_cartopy()
    display_da, units_label, conversion_note = get_display_units_and_values(
        da, target_units=target_units
    )
    units_label = _units_label_with_conversion_note(units_label, conversion_note)
    vmin, vmax = compute_plot_limits(display_da, vmin=vmin, vmax=vmax, qmin=qmin, qmax=qmax)

    if layout not in {"horizontal", "vertical"}:
        raise ValueError("layout must be either 'horizontal' or 'vertical'.")

    # constrained_layout + tight_layout both fight cartopy aspect; use neither.
    if layout == "horizontal":
        nrows, ncols = 1, 2
        fig = plt.figure(figsize=figsize or (10, 4), constrained_layout=False)
        ax_map = fig.add_subplot(nrows, ncols, 1, projection=ccrs.PlateCarree())
        ax_dist = fig.add_subplot(nrows, ncols, 2)
    else:
        nrows, ncols = 2, 1
        fig = plt.figure(figsize=figsize or (10, 4), constrained_layout=False)
        ax_map = fig.add_subplot(nrows, ncols, 1, projection=ccrs.PlateCarree())
        ax_dist = fig.add_subplot(nrows, ncols, 2)

    _setup_geo_map_axis(
        ax_map,
        extent,
        ccrs=ccrs,
        LONGITUDE_FORMATTER=LONGITUDE_FORMATTER,
        LATITUDE_FORMATTER=LATITUDE_FORMATTER,
        tick_fontsize=map_tick_fontsize,
        fixed_tick_locators=False,
        map_aspect=map_aspect,
    )

    lon_name, lat_name = _infer_lon_lat_names(display_da)
    mesh = ax_map.pcolormesh(
        display_da[lon_name].values,
        display_da[lat_name].values,
        display_da.values,
        cmap=cmap,
        shading="auto",
        transform=ccrs.PlateCarree(),
        vmin=vmin,
        vmax=vmax,
    )
    _add_map_colorbar(
        mesh,
        ax_map,
        label=units_label,
        colorbar_extend=colorbar_extend,
        colorbar_pad=colorbar_pad,
        label_fontsize=13,
        tick_fontsize=11,
        colorbar_fn=fig.colorbar,
    )
    resolved_map_title = map_title or "Spatial map"
    if conversion_note == "native (conversion skipped)":
        resolved_map_title = f"{resolved_map_title} [{conversion_note}]"
    ax_map.set_title(resolved_map_title, fontsize=14)

    vals = display_da.values
    vals = vals[np.isfinite(vals)]
    if vmin is not None:
        vals = vals[vals >= vmin]
    if vmax is not None:
        vals = vals[vals <= vmax]

    if hist_bins is not None:
        bins_arg = hist_bins
    elif hist_bin_width is not None and vmin is not None and vmax is not None:
        lo = float(vmin)
        hi = float(vmax)
        edges = np.arange(lo, hi + hist_bin_width * 1.0001, hist_bin_width)
        if edges.size < 2:
            edges = np.array([lo, hi])
        bins_arg = edges
    else:
        bins_arg = 80

    if vals.size == 0:
        warnings.warn("plot_map_with_distribution: no finite values for histogram.")
        ax_dist.text(0.5, 0.5, "No finite data", ha="center", va="center", transform=ax_dist.transAxes)
    else:
        ax_dist.hist(vals, bins=bins_arg, alpha=0.85, edgecolor="black")
    ax_dist.grid(True, alpha=0.35)
    ax_dist.set_xlabel(units_label, fontsize=12)
    ax_dist.set_ylabel("Count", fontsize=12)
    ax_dist.tick_params(labelsize=11)
    if vmin is not None and vmax is not None:
        ax_dist.set_xlim(vmin, vmax)
    resolved_dist_title = dist_title or "Value distribution"
    if conversion_note == "native (conversion skipped)":
        resolved_dist_title = f"{resolved_dist_title} [{conversion_note}]"
    ax_dist.set_title(resolved_dist_title, fontsize=14)

    fig.subplots_adjust(left=0.06, right=0.98, top=0.94, bottom=0.08, hspace=0.28, wspace=0.22)
    return fig, (ax_map, ax_dist)
