"""
Download and preprocess GPM IMERG, GLDAS, and GRACE/GRACE-FO mascons.

Study window (default)
    2002-01-01 to 2025-09-30 (covers notebook 03 analysis).

Products
    - GPM IMERG Final daily (GPM_3IMERGDF, V07) -> monthly precip sums (mm/month)
    - GLDAS-2.1 monthly CLSM / NOAH / VIC -> SM, total_runoff, SWE (mm)
    - CSR / JPL CRI / GSFC RL06 mascons + CSR land mask

Auth
    Earthdata Login (earthaccess / ~/.netrc) for IMERG, GLDAS, and JPL.
    Plain HTTPS for CSR and GSFC (no Earthdata key). SSL verify=False is used
    only as a last-resort fallback when certificate validation fails.

Time convention
    Analysis stores use month-end timestamps. IMERG is resampled with
    ``time="ME"``. GLDAS native month labels are normalized to month-end
    after open so notebooks 02/03 can pair series cleanly.

FORCE / resume
    ``force=True`` rebuilds Zarrs and re-queries catalogs. Earthdata granules
    already on disk are still reused by earthaccess. Delete
    ``data/raw/gpm/...`` or ``data/raw/gldas/<model>/`` for a full re-fetch.
    Incomplete Zarr directories are detected and rebuilt automatically.

Resources
    ``get_resource_config()`` sizes download threads and dask workers from
    CPU count (env: DOWNLOAD_THREADS, DASK_WORKERS). GPU may be detected but
    is not used for this I/O-bound NetCDF -> Zarr ETL.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import urljoin

import dask
import earthaccess
import pandas as pd
import requests
import xarray as xr
from dask.diagnostics import ProgressBar

logger = logging.getLogger(__name__)

# Silence earthaccess per-file "already downloaded" INFO chatter.
logging.getLogger("earthaccess").setLevel(logging.WARNING)

__all__ = [
    "DEFAULT_START",
    "DEFAULT_END",
    "GLOBAL_BBOX",
    "GLDAS_MODELS",
    "GLDAS_PRODUCTS",
    "period_tag",
    "set_project_root",
    "ensure_earthdata_login",
    "get_resource_config",
    "run_imerg_pipeline",
    "run_gldas_all",
    "download_grace_mascons",
    "resolve_grace_paths",
    "gldas_zarr_path",
    "summarize_zarr",
    "remove_imerg_daily_granules",
    "remove_gldas_raw_granules",
]

# ---------------------------------------------------------------------------
# Product constants
# ---------------------------------------------------------------------------

DEFAULT_START = "2002-01-01"
DEFAULT_END = "2025-09-30"
GLOBAL_BBOX: Tuple[float, float, float, float] = (-180.0, -90.0, 180.0, 90.0)

IMERG_SHORT_NAME = "GPM_3IMERGDF"
IMERG_VERSION = "07"
GLDAS_VERSION = "2.1"

GLDAS_MODELS: Tuple[str, ...] = (
    "GLDAS_CLSM10_M",
    "GLDAS_NOAH10_M",
    "GLDAS_VIC10_M",
)
GLDAS_PRODUCTS: Tuple[str, ...] = ("SM", "Q", "SWE")

# Notebook 03 expects:
#   CLSM SM -> SoilMoist_P_inst
#   NOAH/VIC SM -> sm_total
#   all Q -> total_runoff
#   all SWE -> SWE_inst
GLDAS_SPECS: Dict[str, Dict[str, Dict]] = {
    "GLDAS_CLSM10_M": {
        "SM": {
            "variables": ["SoilMoist_P_inst"],
            "keep": ["SoilMoist_P_inst"],
        },
        "Q": {
            "variables": ["Qs_acc", "Qsb_acc"],
            "sum_as": "total_runoff",
            "keep": ["total_runoff"],
        },
        "SWE": {
            "variables": ["SWE_inst"],
            "keep": ["SWE_inst"],
        },
    },
    "GLDAS_NOAH10_M": {
        "SM": {
            "variables": [
                "SoilMoi0_10cm_inst",
                "SoilMoi10_40cm_inst",
                "SoilMoi40_100cm_inst",
                "SoilMoi100_200cm_inst",
            ],
            "sum_as": "sm_total",
            "keep": ["sm_total"],
        },
        "Q": {
            "variables": ["Qs_acc", "Qsb_acc"],
            "sum_as": "total_runoff",
            "keep": ["total_runoff"],
        },
        "SWE": {
            "variables": ["SWE_inst"],
            "keep": ["SWE_inst"],
        },
    },
    "GLDAS_VIC10_M": {
        "SM": {
            "variables": [
                "SoilMoi0_30cm_inst",
                "SoilMoi_depth2_inst",
                "SoilMoi_depth3_inst",
            ],
            "sum_as": "sm_total",
            "keep": ["sm_total"],
        },
        "Q": {
            "variables": ["Qs_acc", "Qsb_acc"],
            "sum_as": "total_runoff",
            "keep": ["total_runoff"],
        },
        "SWE": {
            "variables": ["SWE_inst"],
            "keep": ["SWE_inst"],
        },
    },
}

# Stable filename tokens for GRACE products (case-insensitive substrings).
GRACE_TOKENS: Dict[str, Tuple[str, ...]] = {
    "csr": ("all-corrections",),
    "csr_mask": ("landmask",),
    "jpl": ("mscnv04cri",),
    "gsfc": ("halfdegree", "obp"),
}

CSR_MASCON_PAGE = "https://www2.csr.utexas.edu/grace/RL06_mascons.html"
GSFC_MASCON_PAGE = "https://earth.gsfc.nasa.gov/geo/data/grace-mascons"
JPL_MASCON_SHORT_NAME = "TELLUS_GRAC-GRFO_MASCON_CRI_GRID_RL06.3_V4"

_MONTH_ABBR = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)

PathLike = Union[str, Path]

_REPO_ROOT: Optional[Path] = None
_EARTHDATA_OK = False
_RESOURCES_ANNOUNCED = False


# ---------------------------------------------------------------------------
# Resource detection (CPU auto; GPU detected but unused for this ETL)
# ---------------------------------------------------------------------------

def get_resource_config() -> Dict[str, Any]:
    """
    Return download / dask worker defaults from local resources.

    Environment overrides:
      DOWNLOAD_THREADS, DASK_WORKERS

    GPU may be reported as ``detected_unused``; this ETL is I/O-bound
    (NetCDF open + resample + Zarr write) and does not use the GPU.
    """
    cpus = os.cpu_count() or 4

    def _env_int(name: str, default: int) -> int:
        raw = os.environ.get(name)
        if raw is None or str(raw).strip() == "":
            return default
        try:
            return int(raw)
        except ValueError as exc:
            _raise_ctx(
                ValueError,
                f"Invalid {name}={raw!r}; expected an integer "
                f"(e.g. export {name}={default})",
                cause=exc,
            )
            raise  # pragma: no cover

    download_threads = _env_int(
        "DOWNLOAD_THREADS", min(16, max(2, cpus - 1))
    )
    dask_workers = _env_int(
        "DASK_WORKERS", min(8, max(2, cpus - 1))
    )

    gpu = "unavailable"
    gpu_note = "I/O-bound NetCDF->Zarr ETL; GPU not used"
    try:
        import cupy  # noqa: F401

        gpu = "detected_unused"
    except Exception:
        try:
            import torch

            if torch.cuda.is_available():
                gpu = "detected_unused"
        except Exception:
            pass

    return {
        "cpus": cpus,
        "download_threads": download_threads,
        "dask_workers": dask_workers,
        "dask_scheduler": "threads",
        "gpu": gpu,
        "gpu_note": gpu_note,
    }


def _announce_resources(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _RESOURCES_ANNOUNCED
    cfg = cfg or get_resource_config()
    if not _RESOURCES_ANNOUNCED:
        _announce(
            f"resources: download_threads={cfg['download_threads']}  "
            f"dask_workers={cfg['dask_workers']}  "
            f"gpu={cfg['gpu']} ({cfg['gpu_note']})"
        )
        _RESOURCES_ANNOUNCED = True
    return cfg


def _resolve_threads(threads: Optional[int]) -> int:
    cfg = _announce_resources()
    return int(threads) if threads is not None else int(cfg["download_threads"])


def _dask_workers() -> int:
    return int(_announce_resources()["dask_workers"])


# ---------------------------------------------------------------------------
# Paths and compact status printing
# ---------------------------------------------------------------------------

def _as_path(path: PathLike) -> Path:
    return Path(path)


from status_io import (  # noqa: E402  — after PathLike / logging setup
    announce as _announce,
    detect_repo_root as _detect_repo_root,
    item as _item,
    note as _note,
    raise_ctx as _raise_ctx,
    rel as _rel,
    section as _section,
    set_project_root as _set_project_root_shared,
    summarize_skipped as _summarize_skipped,
)


def set_project_root(root: PathLike) -> None:
    """Set the repo root used for relative status paths."""
    global _REPO_ROOT
    _REPO_ROOT = _as_path(root).resolve()
    _set_project_root_shared(_REPO_ROOT)


def period_tag(start: str = DEFAULT_START, end: str = DEFAULT_END) -> str:
    """Return a locale-stable period label, e.g. ``Jan2002_Sep2025``."""
    s = pd.Timestamp(start)
    e = pd.Timestamp(end)
    return (
        f"{_MONTH_ABBR[s.month - 1]}{s.year}_"
        f"{_MONTH_ABBR[e.month - 1]}{e.year}"
    )


# ---------------------------------------------------------------------------
# Earthdata auth
# ---------------------------------------------------------------------------

def ensure_earthdata_login():
    """
    Authenticate with NASA Earthdata Login via ``earthaccess``.

    Prerequisites (one-time):
      1. Create a free account at https://urs.earthdata.nasa.gov
      2. Configure credentials for earthaccess, typically via ``~/.netrc``
         (and ``~/.dodsrc`` if using OPeNDAP).
    """
    global _EARTHDATA_OK
    try:
        auth = earthaccess.login()
    except Exception as exc:
        _raise_ctx(
            RuntimeError,
            "Earthdata login failed. Create an account at "
            "https://urs.earthdata.nasa.gov and configure credentials "
            "(typically ~/.netrc for earthaccess). "
            f"Original error: {exc}",
            cause=exc,
        )
        raise  # pragma: no cover
    # earthaccess may return None / falsy Auth on failure without raising
    if auth is False or auth is None:
        _raise_ctx(
            RuntimeError,
            "Earthdata login returned no session. Check ~/.netrc "
            "(machine urs.earthdata.nasa.gov) or run earthaccess.login() interactively.",
        )
        raise  # pragma: no cover
    if not _EARTHDATA_OK:
        _announce("Earthdata login: ok")
        _EARTHDATA_OK = True
    return auth


# ---------------------------------------------------------------------------
# Granule helpers and Zarr I/O
# ---------------------------------------------------------------------------

def _expected_granule_names(results) -> Set[str]:
    """Best-effort set of remote filenames from earthaccess search results."""
    names: Set[str] = set()
    for r in results:
        name = None
        try:
            urls = r.data_links()
            if urls:
                name = urls[0].split("/")[-1].split("?")[0]
        except Exception:  # noqa: BLE001
            pass
        if not name:
            for token in str(r).replace('"', " ").replace("'", " ").split():
                if token.endswith((".nc4", ".nc")):
                    name = token.split("/")[-1]
                    break
        if name:
            names.add(name)
    return names


def _local_nc4_paths(out_dir: Path) -> List[str]:
    return sorted(glob.glob(str(out_dir / "*.nc4")))


def _local_basenames(out_dir: Path) -> Set[str]:
    return {Path(p).name for p in _local_nc4_paths(out_dir)}


def _filter_expected_files(out_dir: Path, expected: Set[str]) -> List[str]:
    files = _local_nc4_paths(out_dir)
    if not expected:
        return files
    return [p for p in files if Path(p).name in expected]


def _zarr_status(
    path: Path,
    required_vars: Optional[Sequence[str]] = None,
) -> str:
    """Return ``ok`` or a short reason the Zarr is not ready (for messages only)."""
    if not path.exists() or not path.is_dir():
        return "missing"
    try:
        if not any(path.iterdir()):
            return "empty"
    except OSError as exc:
        return f"unreadable ({exc})"
    try:
        ds = xr.open_zarr(str(path), consolidated=True)
    except Exception as exc:  # noqa: BLE001
        return f"corrupt/unopenable ({exc})"
    try:
        if ds.sizes.get("time", 0) < 1:
            return "no time steps"
        if required_vars:
            missing = [v for v in required_vars if v not in ds.data_vars]
            if missing:
                return f"missing variables {missing}"
        return "ok"
    finally:
        ds.close()


def _zarr_is_complete(
    path: Path,
    required_vars: Optional[Sequence[str]] = None,
) -> bool:
    """True if path is an openable consolidated Zarr with time and required vars."""
    return _zarr_status(path, required_vars) == "ok"


def _remove_incomplete_zarr(path: Path) -> None:
    if path.exists():
        _note(f"incomplete Zarr; removing {path.name}")
        shutil.rmtree(path, ignore_errors=True)


def _validate_gldas_model(model: str) -> None:
    if model not in GLDAS_SPECS:
        _raise_ctx(
            ValueError,
            f"Unknown GLDAS model {model!r}; expected one of {list(GLDAS_SPECS)}",
        )


def _download_earthaccess_granules(
    *,
    short_name: str,
    version: str,
    out_dir: Path,
    start: str,
    end: str,
    bbox: Tuple[float, float, float, float],
    threads: int,
    force: bool,
    label: str,
) -> List[str]:
    """
    Search Earthdata, reuse local ``*.nc4`` when possible, download missing.

    Shared by IMERG and GLDAS. Raises with product/date/path context on failure.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    local = _local_basenames(out_dir)
    if force and local:
        _note(f"{label}: FORCE=True; {len(local)} local files reused if present")

    _note(f"{label}: searching {start} to {end} ...")
    ensure_earthdata_login()
    try:
        results = earthaccess.search_data(
            short_name=short_name,
            version=version,
            temporal=(start, end),
            bounding_box=bbox,
            count=-1,
        )
    except Exception as exc:
        _raise_ctx(
            RuntimeError,
            f"{label}: Earthdata search failed for {short_name} v{version} "
            f"({start} to {end}). Check network/auth (~/.netrc) and product name. "
            f"Original error: {exc}",
            cause=exc,
        )
        raise  # pragma: no cover

    n_search = len(results)
    if n_search == 0:
        _raise_ctx(
            FileNotFoundError,
            f"{label}: 0 granules found for {short_name} v{version} "
            f"({start} to {end}, bbox={bbox}). Check the date window and "
            "Earthdata catalog availability.",
        )
        raise  # pragma: no cover

    expected = _expected_granule_names(results)
    n_expected = len(expected) if expected else n_search
    n_have = len(expected & local) if expected else len(local)
    n_missing = max(n_expected - n_have, 0)
    _note(f"{label}: found {n_search}; have {n_have} of {n_expected}; missing {n_missing}")

    if not force and expected and expected.issubset(local):
        files = _filter_expected_files(out_dir, expected)
        _item(f"{label}: {len(files)} granules", "ok")
        return files
    if not force and not expected and n_search and len(local) >= n_search:
        files = _local_nc4_paths(out_dir)
        _note(f"{label}: could not parse remote names; count match")
        _item(f"{label}: {len(files)} granules", "ok")
        return files

    _note(f"{label}: downloading missing ({threads} threads) ...")
    try:
        earthaccess.download(results, local_path=str(out_dir), threads=threads)
    except Exception as exc:
        _raise_ctx(
            RuntimeError,
            f"{label}: download failed into {_rel(out_dir)}. "
            "Check disk space, network, and Earthdata auth. "
            f"Original error: {exc}",
            cause=exc,
        )
        raise  # pragma: no cover

    files = _filter_expected_files(out_dir, expected)
    if not files:
        _raise_ctx(
            FileNotFoundError,
            f"{label}: no granules downloaded to {_rel(out_dir)} "
            f"({short_name} {start}–{end}).",
        )
        raise  # pragma: no cover
    if expected and len(files) < n_expected:
        _raise_ctx(
            FileNotFoundError,
            f"{label}: incomplete download — have {len(files)} of {n_expected} "
            f"expected granules in {_rel(out_dir)}. Re-run with FORCE=True or "
            "check disk space.",
        )
        raise  # pragma: no cover
    _item(f"{label}: {len(files)} granules", "ok")
    return files


def _patch_time_attrs(ds: xr.Dataset) -> xr.Dataset:
    """Set StartDate/EndDate and drop bulky InputPointer attrs if present."""
    if "time" in ds.coords and len(ds.time) > 0:
        ds = ds.copy()
        ds.attrs["StartDate"] = str(ds.time.values[0])[:10]
        ds.attrs["EndDate"] = str(ds.time.values[-1])[:10]
    if "InputPointer" in ds.attrs:
        ds.attrs.pop("InputPointer", None)
    return ds


def _to_month_end(ds: xr.Dataset) -> xr.Dataset:
    """Normalize time coordinate to month-end timestamps."""
    if "time" not in ds.coords:
        return ds
    times = pd.to_datetime(ds["time"].values)
    month_end = times.to_period("M").to_timestamp("M")
    return ds.assign_coords(time=month_end)


def _zarr_encoding(ds: xr.Dataset) -> Dict[str, Dict[str, Any]]:
    """float32 + Blosc/zstd when numcodecs is available."""
    compressor = None
    try:
        from numcodecs import Blosc

        compressor = Blosc(cname="zstd", clevel=3, shuffle=Blosc.BITSHUFFLE)
    except Exception:
        pass
    encoding: Dict[str, Dict[str, Any]] = {}
    for name in ds.data_vars:
        enc: Dict[str, Any] = {"dtype": "float32"}
        if compressor is not None:
            enc["compressor"] = compressor
        encoding[name] = enc
    return encoding


def _write_zarr(ds: xr.Dataset, zarr_path: Path, time_chunk: int) -> None:
    zarr_path.parent.mkdir(parents=True, exist_ok=True)
    chunked = ds.chunk({"time": time_chunk})
    workers = _dask_workers()
    with ProgressBar():
        with dask.config.set(scheduler="threads", num_workers=workers):
            chunked.to_zarr(
                str(zarr_path),
                mode="w",
                consolidated=True,
                encoding=_zarr_encoding(chunked),
            )


# ---------------------------------------------------------------------------
# GPM IMERG Final daily -> monthly Zarr
# ---------------------------------------------------------------------------

def download_imerg_daily(
    raw_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bbox: Tuple[float, float, float, float] = GLOBAL_BBOX,
    threads: Optional[int] = None,
    force: bool = False,
) -> List[str]:
    """
    Search and download GPM IMERG Final daily (``GPM_3IMERGDF``) granules.

    Returns sorted local ``*.nc4`` paths under ``raw_dir/gpm/GPM_3IMERGDF``
    that match the search result set. Raises if the expected set is incomplete
    after download.
    """
    threads = _resolve_threads(threads)
    out_dir = _as_path(raw_dir) / "gpm" / IMERG_SHORT_NAME
    set_project_root(_detect_repo_root(out_dir))
    _section("IMERG granules", out_dir)
    return _download_earthaccess_granules(
        short_name=IMERG_SHORT_NAME,
        version=IMERG_VERSION,
        out_dir=out_dir,
        start=start,
        end=end,
        bbox=bbox,
        threads=threads,
        force=force,
        label="IMERG",
    )

def imerg_daily_zarr_path(
    interim_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Path:
    """Path of the obsolete intermediate daily IMERG Zarr (removed if found)."""
    tag = period_tag(start, end)
    return _as_path(interim_dir) / "gpm" / f"GPM_3IMERGDF_{tag}.zarr"


def build_imerg_monthly_zarr(
    granule_files: Sequence[str],
    interim_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
    time_chunk: int = 200,
) -> Path:
    """
    Open daily IMERG granules, resample to month-end sums, write monthly Zarr.

    Precipitation units: mm/month (sum of daily rates over the month).
    """
    interim = _as_path(interim_dir) / "gpm"
    interim.mkdir(parents=True, exist_ok=True)
    tag = period_tag(start, end)
    zarr_path = interim / f"GPM_3IMERGDF_{tag}_resToM.zarr"
    _section("IMERG monthly Zarr", interim)

    if _zarr_is_complete(zarr_path, required_vars=["precipitation"]) and not force:
        _item(zarr_path.name, "ok")
        return zarr_path
    if zarr_path.exists() and force:
        _remove_incomplete_zarr(zarr_path)
    elif zarr_path.exists() and not _zarr_is_complete(zarr_path, ["precipitation"]):
        _remove_incomplete_zarr(zarr_path)

    _note(f"opening {len(list(granule_files))} daily granules (lazy) ...")
    try:
        ds = xr.open_mfdataset(
            list(granule_files),
            engine="h5netcdf",
            combine="by_coords",
            parallel=True,
            drop_variables=["time_bnds"],
            data_vars="minimal",
            coords="minimal",
            compat="override",
        )
        ds = ds[["precipitation"]]
    except Exception as exc:
        n = len(list(granule_files))
        sample = Path(list(granule_files)[0]).parent if n else Path(".")
        _raise_ctx(
            RuntimeError,
            f"Failed to open {n} IMERG daily granules under {_rel(sample)}. "
            "Check that files are complete NetCDF4 (.nc4). "
            f"Original error: {exc}",
            cause=exc,
        )
        raise  # pragma: no cover

    _note("resampling daily to monthly sums (month-end) ...")
    monthly = ds.resample(time="ME").sum()
    monthly["precipitation"].attrs.setdefault("units", "mm/month")
    monthly["precipitation"].attrs.setdefault(
        "long_name", "IMERG Final monthly precipitation sum"
    )
    monthly = _patch_time_attrs(monthly)

    _item(zarr_path.name, "writing")
    _write_zarr(monthly, zarr_path, time_chunk=time_chunk)
    ds.close()
    _item(zarr_path.name, "ok")
    return zarr_path


def _remove_stale_daily_zarr(
    interim_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> None:
    daily = imerg_daily_zarr_path(interim_dir, start=start, end=end)
    if daily.exists():
        _note(f"removing obsolete daily Zarr: {daily.name}")
        shutil.rmtree(daily, ignore_errors=True)


def remove_imerg_daily_granules(
    raw_dir: PathLike,
    interim_dir: PathLike,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Dict[str, Any]:
    """
    Delete raw daily IMERG ``*.nc4`` granules after the monthly Zarr is verified.

    Opt-in disk cleanup: the monthly Zarr under ``interim_dir/gpm/`` is the
    analysis product. Raw daily files (~250 GB) are not needed again unless
    you rebuild with ``FORCE=True``. Raises if the monthly Zarr is incomplete.
    """
    tag = period_tag(start, end)
    monthly_path = _as_path(interim_dir) / "gpm" / f"GPM_3IMERGDF_{tag}_resToM.zarr"
    status = _zarr_status(monthly_path, required_vars=["precipitation"])
    if status != "ok":
        _raise_ctx(
            FileNotFoundError,
            f"Monthly IMERG Zarr not ready ({status}): {_rel(monthly_path)}. "
            "Build it with run_imerg_pipeline() before removing daily granules.",
        )

    out_dir = _as_path(raw_dir) / "gpm" / IMERG_SHORT_NAME
    files = list(out_dir.glob("*.nc4")) if out_dir.is_dir() else []
    freed = 0
    n_failed = 0
    for p in files:
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError as exc:
            n_failed += 1
            logger.warning("Could not remove %s: %s", p, exc)
    n_removed = len(files) - n_failed
    freed_gb = freed / 1e9
    status_tag = "partial" if n_failed else "ok"
    _item(
        f"removed {n_removed}/{len(files)} IMERG daily granules "
        f"({freed_gb:.1f} GB); kept {_rel(monthly_path)}"
        + (f"; {n_failed} failed" if n_failed else ""),
        status_tag,
    )
    return {"n_removed": n_removed, "freed_gb": float(freed_gb), "n_failed": n_failed}


def remove_gldas_raw_granules(
    model: str,
    raw_dir: PathLike,
    interim_dir: PathLike,
    *,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Dict[str, Any]:
    """
    Delete raw GLDAS ``*.nc4`` granules for one model after its Zarr is verified.

    Opt-in disk cleanup: the model Zarr is the analysis product. Raises if the
    Zarr is incomplete. Call once per model (CLSM / NOAH / VIC).
    """
    _validate_gldas_model(model)

    keep = gldas_model_variables(model)
    zarr_path = gldas_zarr_path(interim_dir, model, start=start, end=end)
    status = _zarr_status(zarr_path, required_vars=keep)
    if status != "ok":
        _raise_ctx(
            FileNotFoundError,
            f"GLDAS Zarr not ready ({status}): {_rel(zarr_path)}. "
            "Build it with run_gldas_all() before removing raw granules.",
        )

    out_dir = _as_path(raw_dir) / "gldas" / model
    files = list(out_dir.glob("*.nc4")) if out_dir.is_dir() else []
    freed = 0
    n_failed = 0
    for p in files:
        try:
            freed += p.stat().st_size
            p.unlink()
        except OSError as exc:
            n_failed += 1
            logger.warning("Could not remove %s: %s", p, exc)
    n_removed = len(files) - n_failed
    freed_gb = freed / 1e9
    short = model.replace("GLDAS_", "").replace("10_M", "")
    status_tag = "partial" if n_failed else "ok"
    _item(
        f"removed {n_removed}/{len(files)} {short} granules ({freed_gb:.1f} GB); "
        f"kept {_rel(zarr_path)}"
        + (f"; {n_failed} failed" if n_failed else ""),
        status_tag,
    )
    return {
        "n_removed": n_removed,
        "freed_gb": float(freed_gb),
        "model": model,
        "n_failed": n_failed,
    }


def run_imerg_pipeline(
    raw_dir: PathLike,
    interim_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bbox: Tuple[float, float, float, float] = GLOBAL_BBOX,
    threads: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download IMERG daily granules, resample, write monthly Zarr (no daily Zarr)."""
    _announce_resources()
    tag = period_tag(start, end)
    monthly_path = _as_path(interim_dir) / "gpm" / f"GPM_3IMERGDF_{tag}_resToM.zarr"
    set_project_root(_detect_repo_root(monthly_path))

    if _zarr_is_complete(monthly_path, ["precipitation"]) and not force:
        _section("IMERG monthly Zarr", monthly_path.parent)
        _remove_stale_daily_zarr(interim_dir, start=start, end=end)
        n_local = len(_local_nc4_paths(_as_path(raw_dir) / "gpm" / IMERG_SHORT_NAME))
        _item(monthly_path.name, "ok")
        return {"monthly_zarr": monthly_path, "n_granules": n_local}

    files = download_imerg_daily(
        raw_dir, start=start, end=end, bbox=bbox, threads=threads, force=force
    )
    monthly = build_imerg_monthly_zarr(
        files, interim_dir, start=start, end=end, force=force
    )
    _remove_stale_daily_zarr(interim_dir, start=start, end=end)
    return {"monthly_zarr": monthly, "n_granules": len(files)}


# ---------------------------------------------------------------------------
# GLDAS (one Zarr per model: SM + runoff + SWE)
# ---------------------------------------------------------------------------

def gldas_zarr_path(
    interim_dir: PathLike,
    model: str,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Path:
    """One Zarr per model holding all analysis variables (SM, runoff, SWE)."""
    tag = period_tag(start, end)
    return _as_path(interim_dir) / "gldas" / model / f"{model}_{tag}.zarr"


def gldas_model_variables(model: str) -> List[str]:
    """Analysis variables written for a model (deduped, SM then Q then SWE)."""
    _validate_gldas_model(model)
    keep: List[str] = []
    for product in GLDAS_PRODUCTS:
        spec = GLDAS_SPECS[model].get(product)
        if spec:
            keep.extend(spec["keep"])
    return list(dict.fromkeys(keep))


def download_gldas_model(
    model: str,
    raw_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bbox: Tuple[float, float, float, float] = GLOBAL_BBOX,
    threads: Optional[int] = None,
    force: bool = False,
) -> List[str]:
    """
    Search and download one GLDAS monthly model (all variables in the granules).

    Returns only local ``*.nc4`` paths that match the search result set.
    Raises if the expected set is incomplete after download.
    """
    _validate_gldas_model(model)
    threads = _resolve_threads(threads)
    out_dir = _as_path(raw_dir) / "gldas" / model
    short = model.replace("GLDAS_", "").replace("10_M", "")
    return _download_earthaccess_granules(
        short_name=model,
        version=GLDAS_VERSION,
        out_dir=out_dir,
        start=start,
        end=end,
        bbox=bbox,
        threads=threads,
        force=force,
        label=short,
    )


def preprocess_gldas_to_zarr(
    model: str,
    granule_files: Sequence[str],
    interim_dir: PathLike,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    force: bool = False,
    time_chunk: int = 100,
) -> Path:
    """
    Build one Zarr per model: soil moisture, ``total_runoff``, and ``SWE_inst``.

    Time is normalized to month-end. Native units are mm (notebook 03 divides by 10 for cm).
    """
    _validate_gldas_model(model)

    keep = gldas_model_variables(model)
    zarr_path = gldas_zarr_path(interim_dir, model, start=start, end=end)
    rel_name = f"{model}/{zarr_path.name}"

    if _zarr_is_complete(zarr_path, required_vars=keep) and not force:
        _item(rel_name, "ok")
        return zarr_path
    if zarr_path.exists():
        _remove_incomplete_zarr(zarr_path)

    specs = GLDAS_SPECS[model]
    native_vars = list(
        dict.fromkeys(v for spec in specs.values() for v in spec["variables"])
    )
    _note(f"{model}: preprocessing {', '.join(native_vars)} ...")
    ds = xr.open_mfdataset(
        list(granule_files),
        engine="h5netcdf",
        combine="by_coords",
        parallel=True,
    )[native_vars]

    for spec in specs.values():
        if "sum_as" in spec:
            out_name = spec["sum_as"]
            ds[out_name] = sum(ds[v] for v in spec["variables"])
            ds[out_name].attrs.setdefault("units", "mm")
            ds[out_name].attrs.setdefault("long_name", out_name)

    ds = ds[keep]
    for name in keep:
        ds[name].attrs.setdefault("units", "mm")
    ds = _to_month_end(ds)
    ds = _patch_time_attrs(ds)

    _item(rel_name, "writing")
    _write_zarr(ds, zarr_path, time_chunk=time_chunk)
    ds.close()
    _item(rel_name, "ok")
    return zarr_path


def run_gldas_all(
    raw_dir: PathLike,
    interim_dir: PathLike,
    models: Optional[Sequence[str]] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
    bbox: Tuple[float, float, float, float] = GLOBAL_BBOX,
    threads: Optional[int] = None,
    force: bool = False,
) -> Dict[str, Path]:
    """
    Download each GLDAS model once, then write one Zarr per model
    (SM + ``total_runoff`` + ``SWE_inst``).

    Returns ``model -> zarr Path``.
    """
    _announce_resources()
    models = list(models or GLDAS_MODELS)
    outputs: Dict[str, Path] = {}
    gldas_root = _as_path(interim_dir) / "gldas"
    set_project_root(_detect_repo_root(gldas_root))
    _section(f"GLDAS ({len(models)} models, SM+Q+SWE)", gldas_root)

    for model in models:
        zarr_path = gldas_zarr_path(interim_dir, model, start=start, end=end)
        keep = gldas_model_variables(model)
        if _zarr_is_complete(zarr_path, required_vars=keep) and not force:
            _item(f"{model}/{zarr_path.name}", "ok")
            outputs[model] = zarr_path
            continue
        files = download_gldas_model(
            model,
            raw_dir,
            start=start,
            end=end,
            bbox=bbox,
            threads=threads,
            force=force,
        )
        outputs[model] = preprocess_gldas_to_zarr(
            model,
            files,
            interim_dir,
            start=start,
            end=end,
            force=force,
        )
    return outputs


def summarize_zarr(path: PathLike) -> str:
    """Return a one-line summary of a Zarr store (vars, time span, dims)."""
    path = _as_path(path)
    try:
        ds = xr.open_zarr(str(path), consolidated=True)
    except Exception as exc:
        _raise_ctx(
            FileNotFoundError,
            f"Cannot open Zarr {_rel(path)}. Build it with run_imerg_pipeline() "
            f"or run_gldas_all() first. Original error: {exc}",
            cause=exc,
        )
        raise  # pragma: no cover
    t0 = t1 = "n/a"
    if "time" in ds.coords and ds.sizes.get("time", 0) > 0:
        t0 = str(ds.time.values[0])[:10]
        t1 = str(ds.time.values[-1])[:10]
    dims = dict(ds.sizes)
    vars_ = list(ds.data_vars)
    ds.close()
    return f"{path.name}: vars={vars_} dims={dims} time={t0} to {t1}"


# ---------------------------------------------------------------------------
# GRACE / GRACE-FO mascons (CSR, JPL, GSFC)
# ---------------------------------------------------------------------------

def find_grace_file(
    directory: PathLike,
    *tokens: str,
    suffix: str = ".nc",
) -> Path:
    """
    Return the newest file in ``directory`` whose name contains all ``tokens``.

    Tokens are matched case-insensitively so date spans in filenames can change
    without breaking notebooks.
    """
    directory = _as_path(directory)
    if not directory.is_dir():
        raise FileNotFoundError(f"GRACE directory not found: {directory}")

    tokens_l = [t.lower() for t in tokens]
    matches = [
        p
        for p in directory.glob(f"*{suffix}")
        if all(tok in p.name.lower() for tok in tokens_l)
    ]
    if not matches:
        raise FileNotFoundError(
            f"No *{suffix} in {directory} matching tokens {tokens}. "
            "Run download_grace_mascons() first."
        )
    return max(matches, key=lambda p: p.stat().st_mtime)


def resolve_grace_paths(raw_dir: PathLike) -> Dict[str, Path]:
    """Resolve CSR / JPL / GSFC mascon + CSR land-mask paths under ``raw_dir/grace``."""
    root = _as_path(raw_dir) / "grace"
    return {
        "csr": find_grace_file(root / "csr", *GRACE_TOKENS["csr"]),
        "csr_mask": find_grace_file(root / "csr", *GRACE_TOKENS["csr_mask"]),
        "jpl": find_grace_file(root / "jpl", *GRACE_TOKENS["jpl"]),
        "gsfc": find_grace_file(root / "gsfc", *GRACE_TOKENS["gsfc"]),
    }


def _http_get_text(url: str, timeout: int = 60) -> str:
    last_err: Optional[Exception] = None
    for verify in (True, False):
        try:
            resp = requests.get(url, timeout=timeout, verify=verify)
            resp.raise_for_status()
            return resp.text
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("GET %s failed (verify=%s): %s", url, verify, exc)
    raise RuntimeError(f"Failed to fetch {url}") from last_err


def _http_download(url: str, dest: Path, force: bool = False, timeout: int = 600) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 1_000_000 and not force:
        return dest

    _note(f"downloading {dest.name} ...")
    last_err: Optional[Exception] = None
    for verify in (True, False):
        try:
            if not verify:
                _note(f"retrying {dest.name} with verify=False (SSL fallback)")
            with requests.get(url, stream=True, timeout=timeout, verify=verify) as resp:
                resp.raise_for_status()
                tmp = dest.with_suffix(dest.suffix + ".part")
                with open(tmp, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        if chunk:
                            fh.write(chunk)
                tmp.replace(dest)
            if dest.stat().st_size < 1_000_000:
                raise RuntimeError(f"Downloaded file suspiciously small: {dest}")
            return dest
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            logger.warning("Download failed (verify=%s): %s", verify, exc)
    raise RuntimeError(f"Failed to download {url}") from last_err


def _first_href_matching(html: str, page_url: str, *needles: str) -> str:
    """Return absolute URL of the first href whose path contains all needles."""
    needles_l = [n.lower() for n in needles]
    hrefs = re.findall(r'href=["\']([^"\']+)["\']', html, flags=re.I)
    for href in hrefs:
        path = href.split("?")[0].lower()
        if all(n in path for n in needles_l):
            return urljoin(page_url, href)
    raise FileNotFoundError(
        f"No link on {page_url} matching {needles}. Page layout may have changed."
    )


def download_csr_mascons(raw_dir: PathLike, force: bool = False) -> Dict[str, Path]:
    """Download CSR RL06 all-corrections mascon + land mask from UTCSR."""
    csr_dir = _as_path(raw_dir) / "grace" / "csr"
    csr_dir.mkdir(parents=True, exist_ok=True)

    out: Dict[str, Path] = {}
    try:
        if not force:
            out["csr"] = find_grace_file(csr_dir, *GRACE_TOKENS["csr"])
            out["csr_mask"] = find_grace_file(csr_dir, *GRACE_TOKENS["csr_mask"])
            return out
    except FileNotFoundError:
        pass

    _note("CSR: resolving UTCSR links ...")
    html = _http_get_text(CSR_MASCON_PAGE)
    mascon_url = _first_href_matching(html, CSR_MASCON_PAGE, "all-corrections", ".nc")
    mask_url = _first_href_matching(html, CSR_MASCON_PAGE, "landmask", ".nc")

    out["csr"] = _http_download(mascon_url, csr_dir / Path(mascon_url).name, force=force)
    out["csr_mask"] = _http_download(mask_url, csr_dir / Path(mask_url).name, force=force)
    return out


def download_gsfc_mascons(raw_dir: PathLike, force: bool = False) -> Path:
    """Download GSFC half-degree OBP mascon NetCDF from the GSFC mascon page."""
    gsfc_dir = _as_path(raw_dir) / "grace" / "gsfc"
    gsfc_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not force:
            return find_grace_file(gsfc_dir, *GRACE_TOKENS["gsfc"])
    except FileNotFoundError:
        pass

    _note("GSFC: resolving download link ...")
    html = _http_get_text(GSFC_MASCON_PAGE)
    url = _first_href_matching(html, GSFC_MASCON_PAGE, *GRACE_TOKENS["gsfc"], ".nc")
    return _http_download(url, gsfc_dir / Path(url).name, force=force)


def download_jpl_mascons(
    raw_dir: PathLike,
    force: bool = False,
    threads: Optional[int] = None,
    start: str = DEFAULT_START,
    end: str = DEFAULT_END,
) -> Path:
    """Download JPL CRI mascon grid via earthaccess / PO.DAAC (latest match)."""
    threads = _resolve_threads(threads)
    jpl_dir = _as_path(raw_dir) / "grace" / "jpl"
    jpl_dir.mkdir(parents=True, exist_ok=True)

    try:
        if not force:
            return find_grace_file(jpl_dir, *GRACE_TOKENS["jpl"])
    except FileNotFoundError:
        pass

    _note("JPL: searching PO.DAAC ...")
    ensure_earthdata_login()
    results = earthaccess.search_data(
        short_name=JPL_MASCON_SHORT_NAME,
        temporal=(start, end),
        count=-1,
    )
    if not results:
        results = earthaccess.search_data(short_name=JPL_MASCON_SHORT_NAME, count=-1)
    if not results:
        raise FileNotFoundError(
            f"No granules found for earthaccess short_name={JPL_MASCON_SHORT_NAME}"
        )
    if len(results) > 1:
        _note(f"JPL: {len(results)} granules found; downloading latest only")
        results = [results[-1]]
    _note(f"JPL: downloading {len(results)} granule(s) ...")
    earthaccess.download(results, local_path=str(jpl_dir), threads=threads)
    return find_grace_file(jpl_dir, *GRACE_TOKENS["jpl"])


def download_grace_mascons(
    raw_dir: PathLike,
    force: bool = False,
    jpl_threads: Optional[int] = None,
) -> Dict[str, Path]:
    """
    Download CSR, JPL, and GSFC mascons needed by notebooks 02 / 03.

    Returns paths resolved by stable filename tokens (see ``GRACE_TOKENS``).
    """
    _announce_resources()
    grace_root = _as_path(raw_dir) / "grace"
    set_project_root(_detect_repo_root(grace_root))
    _section("GRACE mascons", grace_root)

    csr = download_csr_mascons(raw_dir, force=force)
    jpl = download_jpl_mascons(raw_dir, force=force, threads=jpl_threads)
    gsfc = download_gsfc_mascons(raw_dir, force=force)
    paths = {
        "csr": csr["csr"],
        "csr_mask": csr["csr_mask"],
        "jpl": jpl,
        "gsfc": gsfc,
    }
    for key, path in paths.items():
        try:
            rel = path.resolve().relative_to(grace_root.resolve())
            _item(str(rel).replace("\\", "/"), "ok")
        except ValueError:
            _item(path.name, "ok")
    return paths
