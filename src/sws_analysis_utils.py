"""
Surface Water Storage (SWS) analysis utilities for arid regions.

Products:
  HydroLAKES polygons + GloLakes absolute ICESat-2 storage (v1.0 paper path).
  Compare lake volume anomalies with GRACE TWSA (CSR/JPL/GSFC mean).

Windows / units:
  Analysis 2004-04-01 to 2025-09-30; baseline 2004-2009 mean removed.
  Lake volume anomalies in km3; GRACE-window comparison in cm water equivalent.
  Time index is month-end (freq="ME").

FORCE vs cache:
  force=True on downloads deletes/replaces complete local HydroLAKES/GloLakes files.
  Incomplete files (size < remote Content-Length) are unlinked and re-fetched.
  FORCE_REBUILD / force_rebuild rebuilds the SWSA batch cache under processed/.

Resources:
  CPU thread pools for download and lake-batch work (DOWNLOAD_THREADS,
  N_PROCESS_WORKERS). GPU is unused (pandas/xarray/matplotlib batch).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote

import geopandas as gpd
import matplotlib.colors as mcolors
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import xarray as xr
from joblib import Parallel, delayed, parallel_backend
from scipy.interpolate import interp1d
from scipy.stats import linregress
from shapely.geometry import Point
from shapely.ops import unary_union

try:
    from dotenv import load_dotenv

    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False

    def load_dotenv(*_args, **_kwargs):
        return False

logger = logging.getLogger(__name__)

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    _tqdm = None

__all__ = [
    "SWSConfig",
    "get_resource_config",
    "load_sws_config",
    "resolve_precip_path",
    "run_download_all",
    "build_glolakes_arid_catalog",
    "build_glolakes_swsa_batch",
    "build_grace_time_range",
    "process_grace_mean",
    "process_precip_on_grace_grid",
    "analyze_lake_grace_comparisons",
    "save_lake_grace_comparison_figures",
    "load_arid_domains",
    "plot_lake_std_ratio_map",
    "export_lake_std_ratio_shapefile",
    "clean_lake_std_ratio_table",
    "export_lake_std_ratio_table",
    "remove_hydrolakes_raw",
]

_RESOURCES_ANNOUNCED = False

from status_io import (  # noqa: E402
    announce as _announce,
    detect_repo_root as _detect_repo_root,
    format_batch_failures as _format_batch_failures,
    item as _item,
    note as _note,
    raise_ctx as _raise_ctx,
    rel as _rel,
    summarize_skipped as _summarize_skipped,
)


def _dl_note(msg: str) -> None:
    """Backward-compatible status print; prefer ``_announce`` / ``_note`` / ``_item``."""
    _announce(msg)


def _parse_env_file(path: Union[str, Path]) -> None:
    """Load KEY=VALUE pairs from a .env file when python-dotenv is unavailable."""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip().strip('"').strip("'")
        if key:
            os.environ[key] = value

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

BASELINE_START = "2004-01-01"
BASELINE_END = "2009-12-31"
ANALYSIS_START = "2004-04-01"
ANALYSIS_END = "2025-09-30"
PLOT_DPI = 300
PLOT_FIGSIZE = (10.5, 4.2)
PLOT_FONTS = {
    "title": 13,
    "label": 12,
    "tick": 11,
    "legend": 11,
    "annotation": 12,
}
PLOT_COLORS = {
    "lake": "#2166ac",
    "grace": "#1b7837",
    "precip": "#d95f02",
}
PLOT_PRECIP_ALPHA = 0.9
GRACE_COMP_YLABEL = "Storage Anomaly Lakes/GRACE (cm)"
_MAP_CBAR_PAD = 0.01
_MAP_CBAR_FRACTION = 0.05
_MAP_CBAR_SHRINK_V = 0.95
_MAP_CBAR_SHRINK_SINGLE = 0.91  # single-panel maps (analyze_lsm_outputs one_map mode)
_MAP_SUBPLOT_WSPACE = 0.0
_MAP_SUBPLOT_HSPACE = 0.02
_MAP_TIGHT_LAYOUT_PAD = 0.2
_MAP_GRID_LABEL_FONTSIZE = 9
_MAP_AOI_LINEWIDTH = 0.5

def _repo_root() -> Path:
    """Repository root (parent of ``src/``), with cwd fallback for notebooks."""
    return _detect_repo_root()


_REPO_ROOT = _repo_root()
_DATA = _REPO_ROOT / "data"
_RAW = _DATA / "raw"
_INTERIM = _DATA / "interim"
_PROCESSED = _DATA / "processed"

DEFAULT_ARID_AREAS_PATH = str(_PROCESSED / "boundaries" / "ai_v3_yr_mask_02_pol.shp")
DEFAULT_PRECIP_PATH = str(
    _INTERIM / "gpm" / "GPM_3IMERGDF_Jan2002_Sep2025_resToM.zarr"
)


_GRACE_PLACEHOLDER_NOTED = False


def _default_grace_path(key: str) -> Path:
    """Resolve GRACE paths by stable filename tokens when files exist."""
    try:
        from download_data import resolve_grace_paths

        return resolve_grace_paths(_RAW)[key]
    except Exception:
        # Placeholders until notebook 01 has downloaded mascons
        global _GRACE_PLACEHOLDER_NOTED
        if not _GRACE_PLACEHOLDER_NOTED:
            _note(
                "GRACE mascons missing; run notebook 01 or download_grace_mascons() "
                "(using placeholder paths)"
            )
            _GRACE_PLACEHOLDER_NOTED = True
        placeholders = {
            "csr": _RAW / "grace" / "csr" / "CSR_GRACE_GRACE-FO_RL0603_Mascons_all-corrections.nc",
            "jpl": _RAW / "grace" / "jpl" / "GRCTellus.JPL.MSCNv04CRI.nc",
            "gsfc": _RAW / "grace" / "gsfc" / "gsfc_obp_halfdegree.nc",
            "csr_mask": _RAW / "grace" / "csr" / "CSR_GRACE_GRACE-FO_RL06_Mascons_v02_LandMask.nc",
        }
        return placeholders[key]


DEFAULT_CSR_GRACE_PATH = str(_default_grace_path("csr"))
DEFAULT_JPL_GRACE_PATH = str(_default_grace_path("jpl"))
DEFAULT_GSFC_GRACE_PATH = str(_default_grace_path("gsfc"))
DEFAULT_CSR_MASK_PATH = str(_default_grace_path("csr_mask"))
DEFAULT_GRACE_TIME_START = "2002-08-01"
# SWS workspace under data/raw/sws/ (raw/, processed/, catalog/, shapefiles/, figures/)
DEFAULT_DATA_ROOT = _RAW / "sws"
DEFAULT_REFERENCE_MD = _REPO_ROOT / "docs" / "SWS_REFERENCE.md"

GLOLAKES_FILE_SERVER = (
    "https://thredds.nci.org.au/thredds/fileServer/ub8/global/GloLakes"
)
# Paper pipeline uses GloLakes absolute ICESat-2 storage only.
GLOLAKES_V10_FILES = {
    "absolute_icesat2": "GloLakes_v1.0/Global_Lake_Absolute_Storage_LandsatPlusICESat2 (1984-present).nc",
}
GLOLAKES_V11_FILES = {
    "absolute_icesat2": "GloLakes_v1.1/Global_Lake_Absolute_Storage_LandsatPlusICESat2_(1984-present).nc",
}

HYDROLAKES_URL = "https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip"


@dataclass
class SWSConfig:
    """Runtime configuration for SWS analysis."""

    data_root: Path = field(default_factory=lambda: DEFAULT_DATA_ROOT)
    arid_areas_path: Path = field(default_factory=lambda: Path(DEFAULT_ARID_AREAS_PATH))
    precip_path: Path = field(default_factory=lambda: Path(DEFAULT_PRECIP_PATH))
    csr_grace_path: Path = field(default_factory=lambda: Path(DEFAULT_CSR_GRACE_PATH))
    jpl_grace_path: Path = field(default_factory=lambda: Path(DEFAULT_JPL_GRACE_PATH))
    gsfc_grace_path: Path = field(default_factory=lambda: Path(DEFAULT_GSFC_GRACE_PATH))
    csr_mask_path: Path = field(default_factory=lambda: Path(DEFAULT_CSR_MASK_PATH))
    grace_time_start: str = DEFAULT_GRACE_TIME_START
    grace_time_end: str = ANALYSIS_END
    reference_md: Path = field(default_factory=lambda: DEFAULT_REFERENCE_MD)
    baseline_start: str = BASELINE_START
    baseline_end: str = BASELINE_END
    analysis_start: str = ANALYSIS_START
    analysis_end: str = ANALYSIS_END
    n_download_workers: int = 8
    n_process_workers: int = 16  # thread workers for lake batch / summaries; 1 = serial debug

    @property
    def raw_dir(self) -> Path:
        return self.data_root / "raw"

    @property
    def processed_dir(self) -> Path:
        return self.data_root / "processed"

    @property
    def figures_dir(self) -> Path:
        return self.data_root / "figures"

    @property
    def catalog_dir(self) -> Path:
        return self.data_root / "catalog"

    @property
    def shapefiles_dir(self) -> Path:
        return self.data_root / "shapefiles"

    def ensure_dirs(self) -> None:
        for d in (self.raw_dir, self.processed_dir, self.figures_dir, self.catalog_dir, self.shapefiles_dir):
            d.mkdir(parents=True, exist_ok=True)


def get_resource_config() -> Dict[str, Any]:
    """
    Return download / process worker defaults from local CPU resources.

    Environment overrides:
      DOWNLOAD_THREADS, N_PROCESS_WORKERS

    GPU may be reported as ``detected_unused``; the SWS lake batch is
    pandas/xarray/matplotlib and does not use the GPU.
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

    download_workers = _env_int(
        "DOWNLOAD_THREADS", min(16, max(2, cpus - 1))
    )
    process_workers = _env_int(
        "N_PROCESS_WORKERS", min(16, max(2, cpus - 1))
    )

    gpu = "unavailable"
    gpu_note = "lake batch is pandas/xarray/matplotlib; CPU threads only"
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
        "download_workers": download_workers,
        "download_threads": download_workers,  # alias
        "process_workers": process_workers,
        "n_process_workers": process_workers,  # alias
        "gpu": gpu,
        "gpu_note": gpu_note,
    }


def _announce_resources(cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    global _RESOURCES_ANNOUNCED
    cfg = cfg or get_resource_config()
    if not _RESOURCES_ANNOUNCED:
        _announce(
            f"resources: download_workers={cfg['download_workers']}  "
            f"process_workers={cfg['process_workers']}  "
            f"gpu={cfg['gpu']} ({cfg['gpu_note']})"
        )
        _RESOURCES_ANNOUNCED = True
    return cfg


def load_sws_config(env_path: Optional[Union[str, Path]] = None) -> SWSConfig:
    """Load configuration and API keys from environment / .env file."""
    module_dir = Path(__file__).resolve().parent
    for candidate in [env_path, module_dir / ".env", Path.cwd() / ".env"]:
        if candidate and Path(candidate).exists():
            if _DOTENV_AVAILABLE:
                load_dotenv(candidate, override=True)
            else:
                _parse_env_file(candidate)
            break
    else:
        if _DOTENV_AVAILABLE:
            load_dotenv(override=True)

    try:
        precip = resolve_precip_path()
    except FileNotFoundError:
        # Allow config load before notebook 01 has produced the Zarr
        precip = DEFAULT_PRECIP_PATH
        logger.warning("Precipitation Zarr not found yet; using default path: %s", precip)
    res = get_resource_config()
    cfg = SWSConfig(
        precip_path=Path(precip),
        n_download_workers=int(res["download_workers"]),
        n_process_workers=int(res["process_workers"]),
    )
    cfg.ensure_dirs()
    return cfg


def resolve_precip_path(preferred: Optional[str] = None, fallback: Optional[str] = None) -> str:
    """Return an existing monthly GPM IMERG Zarr path (preferred, else fallback or scan)."""
    preferred = preferred or DEFAULT_PRECIP_PATH
    if Path(preferred).exists():
        logger.info("Using precipitation Zarr: %s", preferred)
        return preferred

    candidates: List[Path] = []
    if fallback:
        candidates.append(Path(fallback))
    gpm_dir = _INTERIM / "gpm"
    if gpm_dir.is_dir():
        candidates.extend(sorted(gpm_dir.glob("*_resToM.zarr"), key=lambda p: p.stat().st_mtime, reverse=True))

    for cand in candidates:
        if cand.exists() and cand.resolve() != Path(preferred).resolve():
            logger.warning(
                "Preferred precip path not found (%s). Using fallback: %s",
                preferred,
                cand,
            )
            update_reference_md(
                "Known issues log",
                f"- {datetime.utcnow().isoformat()}Z: preferred GPM Zarr missing; using fallback {cand}.",
            )
            return str(cand)
    raise FileNotFoundError(
        f"No precipitation Zarr found at {preferred}"
        + (f" or {fallback}" if fallback else f" (also scanned {_rel(gpm_dir)}/*_resToM.zarr)")
    )


def update_reference_md(section: str, content: str, reference_path: Optional[Path] = None) -> None:
    """Append a timestamped note under a markdown section heading."""
    reference_path = reference_path or DEFAULT_REFERENCE_MD
    text = reference_path.read_text(encoding="utf-8") if reference_path.exists() else ""
    stamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
    entry = f"\n- **{stamp}**: {content.strip()}\n"
    marker = f"## {section}"
    if marker in text:
        parts = text.split(marker, 1)
        text = parts[0] + marker + parts[1] + entry
    else:
        text += f"\n{marker}\n{entry}"
    reference_path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------

def _download_file_url(
    url: str,
    dest: Path,
    timeout: int = 600,
    retries: int = 3,
    chunk_size: int = 1024 * 1024,
    desc: Optional[str] = None,
    expected_size: Optional[int] = None,
    force: bool = False,
) -> Path:
    """
    Download a file with retries.

    Resume policy:
      - Complete local file (size matches remote Content-Length when known) is
        reused unless ``force=True``.
      - Incomplete local file (size < expected) is unlinked, then fully re-fetched.
      - ``force=True`` deletes a complete local file and re-downloads.
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if expected_size is None:
        expected_size = _remote_content_length(url, timeout=min(timeout, 120))

    if dest.exists() and dest.stat().st_size > 0:
        local_size = dest.stat().st_size
        complete = (not expected_size) or local_size >= expected_size
        if force:
            _note(f"force re-download {desc or dest.name}")
            dest.unlink()
        elif expected_size and local_size < expected_size:
            _note(
                f"incomplete {desc or dest.name} "
                f"({local_size / 1e6:.1f} / {expected_size / 1e6:.1f} MB); re-downloading"
            )
            dest.unlink()
        elif complete:
            _item(desc or dest.name, "ok")
            return dest

    if desc:
        _note(f"downloading {desc}")

    last_err: Optional[Exception] = None
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=timeout) as resp:
                resp.raise_for_status()
                total = int(resp.headers.get("content-length", 0) or 0)
                pbar = None
                if desc and _tqdm is not None and total > 0:
                    pbar = _tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        desc=f"    {desc}",
                        leave=False,
                    )
                with open(dest, "wb") as fh:
                    for chunk in resp.iter_content(chunk_size=chunk_size):
                        if chunk:
                            fh.write(chunk)
                            if pbar is not None:
                                pbar.update(len(chunk))
                if pbar is not None:
                    pbar.close()
            written = dest.stat().st_size
            if expected_size and expected_size > 0 and written < expected_size:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download incomplete for {dest.name}: {written} of {expected_size} bytes"
                )
            if total > 0 and written < total:
                dest.unlink(missing_ok=True)
                raise RuntimeError(
                    f"Download incomplete for {dest.name}: {written} of {total} bytes"
                )
            _item(desc or dest.name, "ok")
            return dest
        except Exception as exc:
            last_err = exc
            logger.warning("Download attempt %s failed for %s: %s", attempt, url, exc)
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Failed to download {url}") from last_err


def _remote_content_length(url: str, timeout: int = 120) -> Optional[int]:
    """Return Content-Length from HTTP HEAD, if available."""
    try:
        resp = requests.head(url, timeout=timeout, allow_redirects=True)
        resp.raise_for_status()
        length = resp.headers.get("content-length")
        return int(length) if length else None
    except Exception as exc:
        logger.debug("HEAD failed for %s: %s", url, exc)
        return None


def _parallel_map(
    func,
    items: Sequence,
    max_workers: int = 8,
    desc: str = "tasks",
) -> List[Any]:
    """Run func(item) in a thread pool; raise if any item fails."""
    if not items:
        return []
    results: List[Any] = []
    errors: List[Tuple[Any, Exception]] = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(func, item): item for item in items}
        iterator = as_completed(futures)
        if _tqdm is not None and len(items) > 1:
            iterator = _tqdm(iterator, total=len(items), desc=f"  {desc}", unit="file")
        for fut in iterator:
            item = futures[fut]
            try:
                results.append(fut.result())
            except Exception as exc:
                logger.error("%s failed for %s: %s", desc, item, exc)
                errors.append((item, exc))
    if errors:
        raise RuntimeError(_format_batch_failures(desc, errors, len(items)))
    return results


def _parallel_thread_map(
    func,
    items: Sequence,
    n_jobs: int = -1,
    desc: str = "tasks",
    unit: str = "task",
) -> List[Any]:
    """Run *func(item)* in a thread pool with optional tqdm progress.

    Uses joblib's threading backend so callers can share read-only objects
    (e.g. an open xarray Dataset) without pickling them to worker processes.
    """
    if not items:
        return []
    n = len(items)
    if n_jobs == 1:
        iterator = items
        if _tqdm is not None and n > 1:
            iterator = _tqdm(items, total=n, desc=f"  {desc}", unit=unit)
        return [func(item) for item in iterator]

    with parallel_backend("threading"):
        if _tqdm is not None and n > 1:
            with Parallel(n_jobs=n_jobs, return_as="generator") as parallel:
                gen = parallel(delayed(func)(item) for item in items)
                return list(_tqdm(gen, total=n, desc=f"  {desc}", unit=unit))
        return Parallel(n_jobs=n_jobs)(delayed(func)(item) for item in items)


# ---------------------------------------------------------------------------
# Spatial infrastructure
# ---------------------------------------------------------------------------

def _hydrolakes_attrs_cache_path(cfg: SWSConfig) -> Path:
    return cfg.processed_dir / "hydrolakes_attrs.parquet"


def _ensure_hydrolakes_attrs_cache(cfg: SWSConfig) -> Path:
    """Build a parquet attribute table (no geometry) for fast area lookups."""
    cache = _hydrolakes_attrs_cache_path(cfg)
    if cache.exists() and cache.stat().st_size > 0:
        return cache

    shp = download_hydrolakes(cfg)
    dbf = shp.with_suffix(".dbf")
    if not dbf.exists():
        raise FileNotFoundError(f"HydroLAKES DBF not found next to {shp}")

    read_dbf = dbf
    if str(dbf).startswith("/mnt/"):
        tmp_dbf = Path(os.environ.get("TMPDIR", "/tmp")) / "HydroLAKES_polys_v10.dbf"
        if not tmp_dbf.exists() or tmp_dbf.stat().st_size != dbf.stat().st_size:
            logger.info("Copying HydroLAKES DBF to %s for faster read", tmp_dbf)
            shutil.copy2(dbf, tmp_dbf)
        read_dbf = tmp_dbf

    try:
        from dbfread import DBF
    except ImportError as exc:
        raise ImportError(
            "Install dbfread for HydroLAKES attribute cache: pip install dbfread"
        ) from exc

    logger.info("Building HydroLAKES attribute cache from %s ...", read_dbf)
    df = pd.DataFrame(iter(DBF(str(read_dbf), load=True)))
    rename = {
        "Hylak_id": "lake_id",
        "Lake_name": "lake_name",
        "Country": "country",
        "Lake_area": "area_km2",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    keep = [c for c in ("lake_id", "lake_name", "country", "area_km2") if c in df.columns]
    df = df[keep]
    cache.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(cache, index=False)
    update_reference_md("Known issues log", f"HydroLAKES attribute cache built at {cache}")
    return cache


def load_hydrolakes_attrs(cfg: Optional[SWSConfig] = None) -> pd.DataFrame:
    """Load HydroLAKES lake attributes (lake_id, name, country, area) without geometry."""
    cfg = cfg or load_sws_config()
    return pd.read_parquet(_ensure_hydrolakes_attrs_cache(cfg))


def _projected_centroid_coords(gdf: gpd.GeoDataFrame) -> Tuple[pd.Series, pd.Series]:
    """Compute centroids in a projected CRS to avoid geographic-CRS warnings."""
    projected = gdf.to_crs(3857)
    cents = projected.geometry.centroid.to_crs(4326)
    return cents.x, cents.y


def _hydrolakes_polygon_cache_path(cfg: SWSConfig) -> Path:
    return cfg.processed_dir / "hydrolakes_polygons.parquet"


def load_hydrolakes_polygons(
    cfg: Optional[SWSConfig] = None,
    lake_ids: Optional[Sequence[int]] = None,
) -> gpd.GeoDataFrame:
    """Load HydroLAKES polygons (cached parquet) optionally filtered to lake IDs."""
    cfg = cfg or load_sws_config()
    cache = _hydrolakes_polygon_cache_path(cfg)
    if not cache.exists() or cache.stat().st_size == 0:
        shp = download_hydrolakes(cfg)
        _note("building HydroLAKES polygon cache (one-time, several minutes)")
        gdf = gpd.read_file(shp)
        rename = {
            "Hylak_id": "lake_id",
            "Lake_name": "lake_name",
            "Country": "country",
            "Lake_area": "area_km2",
        }
        gdf = gdf.rename(columns={k: v for k, v in rename.items() if k in gdf.columns})
        keep = [c for c in ("lake_id", "lake_name", "country", "area_km2", "geometry") if c in gdf.columns]
        gdf = gdf[keep]
        if gdf.crs is None:
            gdf = gdf.set_crs(4326)
        cache.parent.mkdir(parents=True, exist_ok=True)
        gdf.to_parquet(cache)
        update_reference_md("Known issues log", f"HydroLAKES polygon cache: {cache}")
    gdf = gpd.read_parquet(cache)
    if lake_ids is not None:
        ids = {int(x) for x in lake_ids}
        gdf = gdf[gdf["lake_id"].isin(ids)].copy()
    return gdf


def download_hydrolakes(cfg: Optional[SWSConfig] = None, force: bool = False) -> Path:
    """Download and extract HydroLAKES polygon shapefile."""
    cfg = cfg or load_sws_config()
    zip_path = cfg.raw_dir / "hydrolakes" / "HydroLAKES_polys_v10_shp.zip"
    extract_dir = cfg.raw_dir / "hydrolakes" / "HydroLAKES_polys_v10_shp"
    remote_size = _remote_content_length(HYDROLAKES_URL)

    if not force and extract_dir.exists():
        existing = list(extract_dir.rglob("HydroLAKES_polys_v10.shp"))
        if existing and zip_path.exists():
            if not remote_size or zip_path.stat().st_size >= remote_size:
                _item(_rel(existing[0]), "ok")
                return existing[0]
            _note("incomplete HydroLAKES zip; re-downloading")
            zip_path.unlink(missing_ok=True)

    if force and zip_path.exists():
        zip_path.unlink(missing_ok=True)
        if extract_dir.exists():
            shutil.rmtree(extract_dir, ignore_errors=True)

    _download_file_url(
        HYDROLAKES_URL,
        zip_path,
        desc="HydroLAKES zip",
        expected_size=remote_size,
        force=force,
    )
    _note("extracting HydroLAKES archive")
    extract_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    shp_candidates = list(extract_dir.rglob("HydroLAKES_polys_v10.shp"))
    if not shp_candidates:
        shp_candidates = list(extract_dir.rglob("*.shp"))
    if not shp_candidates:
        raise FileNotFoundError(f"No HydroLAKES shapefile found under {extract_dir}")
    shp_path = shp_candidates[0]
    _item(_rel(shp_path), "ok")
    update_reference_md("Known issues log", f"HydroLAKES downloaded to {shp_path.parent}")
    return shp_path


def _dir_size_bytes(path: Path) -> int:
    """Total size of files under ``path`` (0 if missing)."""
    if not path.exists():
        return 0
    total = 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += (Path(root) / name).stat().st_size
            except OSError:
                pass
    return total


def remove_hydrolakes_raw(cfg: Optional[SWSConfig] = None) -> Dict[str, Any]:
    """
    Delete raw HydroLAKES zip/shapefile after the polygon parquet cache is verified.

    Opt-in disk cleanup: ``hydrolakes_polygons.parquet`` is the analysis product.
    Raises if the cache is missing, empty, or unreadable. Re-run
    ``download_hydrolakes(force=True)`` later if you need the shapefile again.
    """
    cfg = cfg or load_sws_config()
    cache = _hydrolakes_polygon_cache_path(cfg)
    if not cache.exists() or cache.stat().st_size <= 0:
        raise FileNotFoundError(
            f"HydroLAKES polygon cache missing or empty: {_rel(cache)}. "
            "Build it via load_hydrolakes_polygons() / catalog steps before cleanup."
        )
    try:
        probe = gpd.read_parquet(cache)
    except Exception as exc:
        raise FileNotFoundError(
            f"HydroLAKES polygon cache unreadable: {_rel(cache)}. "
            f"Rebuild before cleanup. Original error: {exc}"
        ) from exc
    if probe.empty:
        raise FileNotFoundError(
            f"HydroLAKES polygon cache is empty: {_rel(cache)}. "
            "Rebuild before cleanup."
        )

    zip_path = cfg.raw_dir / "hydrolakes" / "HydroLAKES_polys_v10_shp.zip"
    extract_dir = cfg.raw_dir / "hydrolakes" / "HydroLAKES_polys_v10_shp"
    freed = 0
    n_removed = 0
    if zip_path.exists():
        try:
            freed += zip_path.stat().st_size
            zip_path.unlink()
            n_removed += 1
        except OSError as exc:
            logging.getLogger(__name__).warning("Could not remove %s: %s", zip_path, exc)
    if extract_dir.exists():
        freed += _dir_size_bytes(extract_dir)
        shutil.rmtree(extract_dir, ignore_errors=True)
        n_removed += 1
    freed_gb = freed / 1e9
    _item(
        f"removed HydroLAKES raw ({n_removed} path(s), {freed_gb:.1f} GB); "
        f"kept {_rel(cache)}",
        "ok",
    )
    return {"n_removed": n_removed, "freed_gb": float(freed_gb)}


def _read_shapefile(path: Union[str, Path]) -> gpd.GeoDataFrame:
    """Read a shapefile; restore a missing ``.shx`` index when GDAL can rebuild it."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Shapefile not found: {_rel(path)}. "
            "Place arid-mask files under data/processed/boundaries/ (see data/README.md)."
        )
    # Incomplete copies often omit .shx; GDAL can rebuild it from .shp
    prev = os.environ.get("SHAPE_RESTORE_SHX")
    os.environ["SHAPE_RESTORE_SHX"] = "YES"
    try:
        return gpd.read_file(path)
    except Exception as exc:
        shx = path.with_suffix(".shx")
        raise RuntimeError(
            f"Unable to open {_rel(path)} "
            f"(sidecar .shx {'missing' if not shx.exists() else 'present'}). "
            f"Ensure .shp/.shx/.dbf/.prj are together. Original error: {exc}"
        ) from exc
    finally:
        if prev is None:
            os.environ.pop("SHAPE_RESTORE_SHX", None)
        else:
            os.environ["SHAPE_RESTORE_SHX"] = prev


def load_arid_mask(path: Optional[Union[str, Path]] = None) -> gpd.GeoDataFrame:
    """Load and dissolve arid-region polygons to EPSG:4326."""
    path = Path(path or DEFAULT_ARID_AREAS_PATH)
    gdf = _read_shapefile(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    dissolved = gdf.dissolve()
    dissolved["geometry"] = dissolved.geometry.apply(
        lambda g: unary_union(g) if hasattr(g, "geoms") else g
    )
    return dissolved.reset_index(drop=True)


def load_arid_domains(
    path: Optional[Union[str, Path]] = None,
    cfg: Optional[SWSConfig] = None,
    domain_col: str = "Domain",
) -> gpd.GeoDataFrame:
    """Load arid-region domain polygons (one row per domain, not dissolved)."""
    cfg = cfg or load_sws_config()
    path = Path(path or cfg.arid_areas_path)
    gdf = _read_shapefile(path)
    if gdf.crs is None:
        gdf = gdf.set_crs(4326)
    elif gdf.crs.to_epsg() != 4326:
        gdf = gdf.to_crs(4326)
    if domain_col not in gdf.columns:
        gdf[domain_col] = [f"Domain_{i}" for i in range(len(gdf))]
    return gdf.reset_index(drop=True)


def filter_lakes_to_arid(
    lake_gdf: gpd.GeoDataFrame,
    arid_gdf: Optional[gpd.GeoDataFrame] = None,
    method: str = "centroid",
    arid_path: Optional[Union[str, Path]] = None,
) -> gpd.GeoDataFrame:
    """
    Keep lakes whose centroid (default) or polygon intersects the arid mask.

    Expects lake_gdf with geometry or lat/lon columns.
    """
    arid_gdf = arid_gdf if arid_gdf is not None else load_arid_mask(arid_path)
    arid_union = unary_union(arid_gdf.geometry)

    gdf = lake_gdf.copy()
    if "geometry" not in gdf.columns:
        if {"lon", "lat"}.issubset(gdf.columns):
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=[Point(xy) for xy in zip(gdf["lon"], gdf["lat"])],
                crs="EPSG:4326",
            )
        elif {"longitude", "latitude"}.issubset(gdf.columns):
            gdf = gpd.GeoDataFrame(
                gdf,
                geometry=[Point(xy) for xy in zip(gdf["longitude"], gdf["latitude"])],
                crs="EPSG:4326",
            )
        else:
            raise ValueError("lake_gdf needs geometry or lat/lon columns")

    if gdf.crs is None:
        gdf = gdf.set_crs(4326)

    if method == "centroid":
        if gdf.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).any():
            cent_lon, cent_lat = _projected_centroid_coords(gdf)
            centroids = gpd.GeoDataFrame(
                gdf.drop(columns="geometry", errors="ignore"),
                geometry=[Point(x, y) for x, y in zip(cent_lon, cent_lat)],
                crs="EPSG:4326",
            )
            mask = centroids.intersects(arid_union)
        else:
            mask = gdf.intersects(arid_union)
    elif method == "intersects":
        mask = gdf.intersects(arid_union)
    else:
        raise ValueError("method must be 'centroid' or 'intersects'")

    out = gdf.loc[mask].copy()
    out["in_arid"] = True
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# GloLakes
# ---------------------------------------------------------------------------

def _glolakes_url(rel_path: str) -> str:
    return f"{GLOLAKES_FILE_SERVER}/{quote(rel_path, safe='/()')}"


def download_glolakes(
    cfg: Optional[SWSConfig] = None,
    products: Optional[Sequence[str]] = None,
    version: str = "v1.0",
    force: bool = False,
) -> Dict[str, Path]:
    """Download GloLakes NetCDF products from NCI THREDDS."""
    cfg = cfg or load_sws_config()
    file_map = GLOLAKES_V10_FILES if version == "v1.0" else GLOLAKES_V11_FILES
    products = list(products or file_map.keys())
    missing = [p for p in products if p not in file_map]
    if missing:
        raise ValueError(f"Unknown GloLakes products for {version}: {missing}")

    out_dir = cfg.raw_dir / "glolakes" / version
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded: Dict[str, Path] = {}
    _note(f"{len(products)} GloLakes NetCDF product(s)")
    _note(f"dir: {_rel(out_dir)}/")

    def _fetch(key: str) -> Tuple[str, Path]:
        rel = file_map[key]
        dest = out_dir / Path(rel).name
        url = _glolakes_url(rel)
        remote_size = _remote_content_length(url)
        path = _download_file_url(
            url,
            dest,
            desc=key,
            expected_size=remote_size,
            force=force,
        )
        if remote_size and path.stat().st_size < remote_size:
            raise RuntimeError(
                f"GloLakes {key} incomplete after download "
                f"({path.stat().st_size} of {remote_size} bytes)"
            )
        if path.stat().st_size < 1_000_000:
            raise RuntimeError(
                f"GloLakes {key} download suspiciously small ({path.stat().st_size} bytes)"
            )
        return key, path

    results = _parallel_map(_fetch, products, max_workers=cfg.n_download_workers, desc="GloLakes")
    for key, path in results:
        downloaded[key] = path

    missing_keys = [p for p in products if p not in downloaded]
    if missing_keys:
        raise RuntimeError(f"GloLakes download incomplete; missing: {missing_keys}")

    update_reference_md(
        "Known issues log",
        f"GloLakes {version} downloaded products: {list(downloaded.keys())}",
    )
    return downloaded


def load_glolakes_dataset(
    product: str = "absolute_icesat2",
    version: str = "v1.0",
    cfg: Optional[SWSConfig] = None,
) -> xr.Dataset:
    """Open a local or remote GloLakes NetCDF product."""
    cfg = cfg or load_sws_config()
    file_map = GLOLAKES_V10_FILES if version == "v1.0" else GLOLAKES_V11_FILES
    if product not in file_map:
        raise KeyError(f"Unknown GloLakes product '{product}'. Options: {list(file_map)}")

    local = cfg.raw_dir / "glolakes" / version / Path(file_map[product]).name
    if local.exists():
        return xr.open_dataset(local, engine="netcdf4")
    _note(
        f"local GloLakes missing ({_rel(local)}); opening remote. "
        "Run run_download_all() or download_glolakes() to cache locally."
    )
    url = _glolakes_url(file_map[product])
    return xr.open_dataset(url, engine="netcdf4")


def glolakes_to_catalog(ds: xr.Dataset, dataset_source: str = "GloLakes") -> pd.DataFrame:
    """Build a lake metadata catalog from a GloLakes NetCDF dataset."""
    ids = ds["ID"].values.astype(int)
    names = [_decode_str(v) for v in ds["lake_name"].values]
    if "country_name" in ds:
        countries = [_decode_str(v) for v in ds["country_name"].values]
    else:
        countries = [""] * len(ids)
    return pd.DataFrame(
        {
            "lake_id": ids,
            "lake_name": [_display_lake_name(n) for n in names],
            "country": [c or "Unknown" for c in countries],
            "lat": ds["latitude"].values.astype(float),
            "lon": ds["longitude"].values.astype(float),
            "dataset_source": dataset_source,
        }
    )


def _decode_str(val: Any) -> str:
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return ""
    if isinstance(val, bytes):
        return val.decode("utf-8", errors="ignore").strip()
    return str(val).strip()


def _display_lake_name(name: Any) -> str:
    """Return a display-safe lake name; missing/empty values become ``No Name``."""
    text = _decode_str(name)
    if not text or text.lower() == "none":
        return "No Name"
    return text


def extract_glolakes_timeseries(
    ds: xr.Dataset,
    lake_id: int,
    area_km2: Optional[float] = None,
    *,
    id_index: Optional[Dict[int, int]] = None,
    times: Optional[pd.DatetimeIndex] = None,
) -> pd.DataFrame:
    """Extract storage time series for one HydroLAKES / GloLakes ID."""
    if id_index is not None:
        if lake_id not in id_index:
            raise KeyError(
                f"lake_id {lake_id} not found in GloLakes ID index"
            )
        idx = id_index[lake_id]
    else:
        ids = ds["ID"].values.astype(int)
        matches = np.where(ids == lake_id)[0]
        if len(matches) == 0:
            raise KeyError(
                f"lake_id {lake_id} not found in GloLakes dataset"
            )
        idx = int(matches[0])
    storage = ds["lake_storage"].isel(ID=idx).values
    if times is None:
        times = pd.to_datetime(ds["time"].values)
    df = pd.DataFrame({"date": times, "storage_mcm": storage})
    df = df[np.isfinite(df["storage_mcm"])]
    if area_km2 and area_km2 > 0:
        df["height_cm"] = mcm_to_height_cm(df["storage_mcm"].values, area_km2)
    return df.reset_index(drop=True)


def build_glolakes_arid_catalog(
    product: str = "absolute_icesat2",
    version: str = "v1.0",
    cfg: Optional[SWSConfig] = None,
    arid_path: Optional[Union[str, Path]] = None,
    filter_method: str = "intersects",
    sort_by: str = "area",
) -> gpd.GeoDataFrame:
    """Catalog of GloLakes lakes whose HydroLAKES polygon intersects arid regions."""
    cfg = cfg or load_sws_config()
    ds = load_glolakes_dataset(product=product, version=version, cfg=cfg)
    catalog = glolakes_to_catalog(ds, dataset_source=f"GloLakes_{product}")
    ds.close()

    lake_ids = catalog["lake_id"].astype(int).tolist()
    poly_gdf = load_hydrolakes_polygons(cfg, lake_ids=lake_ids)
    meta = catalog.set_index("lake_id")
    poly_gdf["lat"] = poly_gdf["lake_id"].map(meta["lat"])
    poly_gdf["lon"] = poly_gdf["lake_id"].map(meta["lon"])
    poly_gdf["dataset_source"] = f"GloLakes_{product}"
    if "area_km2" not in poly_gdf.columns or poly_gdf["area_km2"].isna().any():
        hydro = load_hydrolakes_attrs(cfg)
        area_map = hydro.set_index("lake_id")["area_km2"].to_dict()
        poly_gdf["area_km2"] = poly_gdf["lake_id"].map(area_map)

    cent_lon, cent_lat = _projected_centroid_coords(poly_gdf)
    poly_gdf["lat_centroid"] = cent_lat.values
    poly_gdf["lon_centroid"] = cent_lon.values

    arid = filter_lakes_to_arid(poly_gdf, arid_path=arid_path, method=filter_method)
    if "lake_name" in arid.columns:
        arid["lake_name"] = arid["lake_name"].map(_display_lake_name)
    sort_col = normalize_sort_by(sort_by)
    if sort_col == "completeness_pct":
        logger.info("sort_by='completeness_pct' not available yet; sorting arid catalog by area")
        sort_col = "area_km2"
    arid = arid.sort_values(sort_col, ascending=False).reset_index(drop=True)
    out_path = cfg.catalog_dir / f"glolakes_arid_{product}.csv"
    arid.drop(columns="geometry").to_csv(out_path, index=False)
    _note(f"arid catalog ({filter_method}): {len(arid)} lakes")
    _item(_rel(out_path), "ok")
    return arid


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def volume_to_height_cm(volume_km3: Union[float, np.ndarray], area_km2: Union[float, np.ndarray]) -> np.ndarray:
    """Convert volumetric storage (km³) and area (km²) to equivalent water height (cm)."""
    vol = np.asarray(volume_km3, dtype=float)
    area = np.asarray(area_km2, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (vol / area) * 1e5


def grace_window_area_km2(lat: float, window_deg: float = 1.0) -> float:
    """Area (km²) of a square lat/lon window at *lat* (1° or 3° side for window_deg 1 or 3)."""
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * np.cos(np.radians(lat))
    return float(window_deg * km_per_deg_lat * window_deg * km_per_deg_lon)


def lake_volume_to_grace_cm(
    volume_anomaly_km3: Union[pd.Series, float, np.ndarray],
    lat: float,
    window_deg: float = 1.0,
) -> Union[pd.Series, np.ndarray]:
    """Spread lake volume anomaly (km³) over the assigned GRACE footprint → cm water equivalent.

    Pass the haversine-assigned ``grace_lat`` (not the lake centroid) and ``window_deg``
    of 1 (one cell) or 3 (3×3 cells centred on the assigned pixel).
    """
    area_km2 = grace_window_area_km2(lat, window_deg)
    if isinstance(volume_anomaly_km3, pd.Series):
        out = volume_to_height_cm(volume_anomaly_km3.values, area_km2)
        return pd.Series(out, index=volume_anomaly_km3.index, name=f"lake_grace_cm_win{window_deg}")
    return volume_to_height_cm(volume_anomaly_km3, area_km2)


def mcm_to_height_cm(storage_mcm: Union[float, np.ndarray], area_km2: Union[float, np.ndarray]) -> np.ndarray:
    """Convert GloLakes storage (MCM) to equivalent water height (cm)."""
    mcm = np.asarray(storage_mcm, dtype=float)
    return volume_to_height_cm(mcm * 0.001, area_km2)


def compute_swsa(
    series: pd.Series,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
) -> Tuple[pd.Series, float]:
    """Remove baseline-period mean to obtain storage anomaly."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    mask = (s.index >= pd.Timestamp(baseline_start)) & (s.index <= pd.Timestamp(baseline_end))
    baseline = s.loc[mask].mean() if mask.sum() > 0 else s.mean()
    if pd.isna(baseline):
        baseline = 0.0
    return s - baseline, float(baseline)


def clip_to_analysis_window(
    series: pd.Series,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> pd.Series:
    """Reindex a monthly series to the analysis window (month-end index)."""
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    idx = pd.date_range(start, end, freq="ME")
    out = s.reindex(idx)
    out.name = series.name
    return out


def normalize_interp_method(method: str) -> str:
    """Validate interpolation method name."""
    key = str(method).strip().lower()
    if key not in {"linear", "spline"}:
        raise ValueError("interp_method must be 'linear' or 'spline'")
    return key


def _gap_fill_mask(
    values: np.ndarray,
    max_gap: Optional[int] = 6,
    extrapolate_edges: bool = False,
) -> np.ndarray:
    """Boolean mask of NaN positions that are allowed to be filled."""
    n = len(values)
    fill = np.zeros(n, dtype=bool)
    i = 0
    while i < n:
        if np.isfinite(values[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isfinite(values[j]):
            j += 1
        run_len = j - i
        has_before = i > 0 and np.isfinite(values[i - 1])
        has_after = j < n and np.isfinite(values[j])
        is_interior = has_before and has_after
        gap_ok = max_gap is None or run_len <= max_gap
        if gap_ok and (is_interior or (extrapolate_edges and (has_before or has_after))):
            fill[i:j] = True
        i = j
    return fill


def interpolate_series_gaps(
    series: pd.Series,
    method: str = "linear",
    max_gap: Optional[int] = 6,
    extrapolate_edges: bool = False,
) -> pd.Series:
    """
    Fill missing months in a regular monthly series.

    Parameters
    ----------
    method : {'linear', 'spline'}
        ``linear`` connects valid months with straight lines.
        ``spline`` uses a monotonic PCHIP curve (interior gaps only by default).
    max_gap : int, optional
        Maximum consecutive NaN months to fill. ``None`` = no limit (still subject
        to ``extrapolate_edges``). Short gaps (1–3 months) are the intended use.
    extrapolate_edges : bool
        If False (default), leading/trailing NaNs are **not** filled. This avoids
        end-of-series artifacts from extrapolation beyond the last observation.

    Notes
    -----
    Prefer ``linear`` as the default. Use ``spline`` only for short **interior**
    gaps in relatively complete series. GloLakes often has long trailing gaps
    (no recent observations); cubic/spline extrapolation there can produce sharp
    false dips or peaks at the series end.
    """
    method = normalize_interp_method(method)
    s = series.copy().astype(np.float64)
    if not s.isna().any():
        return s

    y = s.values
    valid = np.isfinite(y)
    fill_mask = _gap_fill_mask(y, max_gap=max_gap, extrapolate_edges=extrapolate_edges)
    if not fill_mask.any():
        return s

    valid_idx = np.where(valid)[0]
    if len(valid_idx) < 2:
        return s

    nan_idx = np.where(fill_mask)[0]
    if method == "linear":
        f = interp1d(
            valid_idx, y[valid_idx], kind="linear",
            bounds_error=False, fill_value="extrapolate",
        )
        filled_vals = f(nan_idx)
    else:
        if len(valid_idx) < 4:
            f = interp1d(
                valid_idx, y[valid_idx], kind="linear",
                bounds_error=False, fill_value="extrapolate",
            )
            filled_vals = f(nan_idx)
        else:
            from scipy.interpolate import PchipInterpolator

            pchip = PchipInterpolator(valid_idx, y[valid_idx])
            filled_vals = pchip(nan_idx)

    out = s.copy()
    out.iloc[nan_idx] = np.asarray(filled_vals, dtype=np.float64)
    out.name = series.name
    return out


def resample_monthly(df: pd.DataFrame, value_col: str = "height_cm", method: str = "mean") -> pd.Series:
    """Resample to month-end frequency."""
    s = df.set_index("date")[value_col]
    if method == "mean":
        return s.resample("ME").mean()
    if method == "last":
        return s.resample("ME").last()
    raise ValueError("method must be 'mean' or 'last'")


def compute_trend_stats(series: pd.Series) -> Dict[str, float]:
    """Compute std and linear trend on monthly anomalies (cm or km³ — same units as input)."""
    s = series.dropna()
    if len(s) < 6:
        return {"std": np.nan, "trend_yr": np.nan, "std_cm": np.nan, "trend_cm_yr": np.nan}
    std = float(s.std())
    t = np.arange(len(s), dtype=float)
    slope, *_ = linregress(t, s.values)
    trend_yr = float(slope * 12.0)
    return {"std": std, "trend_yr": trend_yr, "std_cm": std, "trend_cm_yr": trend_yr}


def _harmonic_decompose_residual(series: pd.Series, min_points: int = 12) -> pd.Series:
    """
    Residual after removing a calendar-phase-locked linear trend + annual (12mo)
    + semi-annual (6mo) harmonic fit.

    Same math as ``gw_preprocess._decompose_series_full`` /
    ``grace_analysis_utils._decompose_grace_calendar``: time axis is elapsed
    months from the first timestamp using real calendar spacing (not sequential
    integer position), so annual/semi-annual harmonics stay phase-locked even
    with missing months. NaN where the input was NaN, or where fewer than
    ``min_points`` valid samples exist / the fit fails.
    """
    s = series.copy()
    s.index = pd.to_datetime(s.index)
    y = s.to_numpy(dtype=float)
    valid = ~np.isnan(y)
    if valid.sum() < min_points:
        return pd.Series(np.nan, index=s.index, name=s.name)
    t0 = s.index.min()
    t = (s.index - t0).total_seconds().values / (365.25 / 12 * 24 * 3600)
    X = np.column_stack([
        t, np.ones_like(t),
        np.cos(2 * np.pi * t / 12), np.sin(2 * np.pi * t / 12),
        np.cos(2 * np.pi * t / 6), np.sin(2 * np.pi * t / 6),
    ])
    try:
        coeffs, *_ = np.linalg.lstsq(X[valid], y[valid], rcond=None)
    except Exception:
        return pd.Series(np.nan, index=s.index, name=s.name)
    residual = y - X @ coeffs
    residual[~valid] = np.nan
    return pd.Series(residual, index=s.index, name=s.name)


def compute_volume_trend_stats(series: pd.Series) -> Dict[str, float]:
    """Trend stats for volume anomalies (km³, km³ yr⁻¹)."""
    stats = compute_trend_stats(series)
    return {"std_km3": stats["std"], "trend_km3_yr": stats["trend_yr"]}


def completeness_fraction(
    series: pd.Series,
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
) -> float:
    """Fraction of months with valid data in the analysis window."""
    idx = pd.date_range(start, end, freq="ME")
    monthly = series.copy()
    monthly.index = pd.to_datetime(monthly.index)
    monthly = monthly.reindex(idx)
    return float(monthly.notna().mean())


def normalize_sort_by(sort_by: str) -> str:
    """Map ``sort_by`` aliases to catalog column names."""
    key = str(sort_by).strip().lower().replace("-", "_")
    aliases = {
        "area": "area_km2",
        "area_km2": "area_km2",
        "completeness": "completeness_pct",
        "completeness_pct": "completeness_pct",
    }
    if key not in aliases:
        raise ValueError("sort_by must be 'area' or 'completeness_pct'")
    return aliases[key]


def sort_arid_catalog(
    catalog: pd.DataFrame,
    sort_by: str = "area_km2",
    ascending: bool = False,
) -> pd.DataFrame:
    """Sort an arid-lake catalog by area or completeness."""
    col = normalize_sort_by(sort_by)
    if col not in catalog.columns:
        raise ValueError(
            f"Cannot sort by {col!r}: column missing. "
            "Use sort_by='area' before completeness is computed."
        )
    return catalog.sort_values(col, ascending=ascending).reset_index(drop=True)


def rank_datasets_by_completeness(
    catalog: pd.DataFrame,
    series_map: Dict[Any, pd.Series],
    start: str = ANALYSIS_START,
    end: str = ANALYSIS_END,
    sort_by: str = "completeness_pct",
) -> pd.DataFrame:
    """Add completeness % per lake and sort by area or completeness."""
    out = catalog.copy()

    def _pct(lid):
        series = series_map.get(lid)
        if series is None:
            return np.nan
        return 100.0 * completeness_fraction(series, start, end)

    out["completeness_pct"] = out["lake_id"].map(_pct)
    return sort_arid_catalog(out, sort_by=sort_by, ascending=False)


def process_lake_to_volume_anomaly(
    ts_df: pd.DataFrame,
    dataset: str,
    baseline_start: str = BASELINE_START,
    baseline_end: str = BASELINE_END,
    analysis_start: str = ANALYSIS_START,
    analysis_end: str = ANALYSIS_END,
) -> pd.Series:
    """Preprocess one lake → monthly volume storage anomaly (km³) within analysis window."""
    df = ts_df.copy()
    if "date" not in df.columns:
        df = df.reset_index().rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"])

    if "volume_km3" not in df.columns and "storage_mcm" in df.columns:
        df["volume_km3"] = df["storage_mcm"] * 0.001
    elif "volume_km3" not in df.columns:
        raise ValueError(f"Could not derive volume_km3 for dataset {dataset}")

    monthly = resample_monthly(df, "volume_km3")
    anomaly, _ = compute_swsa(monthly, baseline_start, baseline_end)
    anomaly.name = "delta_v_km3"
    return clip_to_analysis_window(anomaly, analysis_start, analysis_end)


def glolakes_batch_cache_stem(
    product: str,
    version: str,
    max_lakes: Optional[int],
    interpolate: bool,
    interp_method: str,
    interp_max_gap: Optional[int],
    interp_extrapolate_edges: bool,
) -> str:
    """Build a short filesystem-safe cache stem: product, version, n, interpolation."""
    n_label = f"n{max_lakes}" if max_lakes is not None else "nall"
    if interpolate:
        gap_label = "gapall" if interp_max_gap is None else f"gap{interp_max_gap}"
        edge_label = "edges" if interp_extrapolate_edges else "noedges"
        interp_label = f"interp-{normalize_interp_method(interp_method)}-{gap_label}-{edge_label}"
    else:
        interp_label = "interp-none"
    stem = f"{product}_{version}_{n_label}_{interp_label}"
    return re.sub(r"[^\w\-.]+", "_", stem)


def _batch_manifest_matches_request(
    manifest: Dict[str, Any],
    *,
    product: str,
    version: str,
    max_lakes: Optional[int],
    sort_by: str,
    interpolate: bool,
    interp_method: str,
    interp_max_gap: Optional[int],
    interp_extrapolate_edges: bool,
    cfg: SWSConfig,
) -> bool:
    """True when cached manifest matches the current batch request."""
    if interpolate:
        interp_method = normalize_interp_method(interp_method)
    expected = {
        "product": product,
        "version": version,
        "max_lakes": max_lakes,
        "sort_by": normalize_sort_by(sort_by),
        "interpolate": interpolate,
        "interp_method": interp_method if interpolate else None,
        "interp_max_gap": interp_max_gap if interpolate else None,
        "interp_extrapolate_edges": interp_extrapolate_edges if interpolate else None,
        "baseline_start": cfg.baseline_start,
        "baseline_end": cfg.baseline_end,
        "analysis_start": cfg.analysis_start,
        "analysis_end": cfg.analysis_end,
    }
    for key, val in expected.items():
        if manifest.get(key) != val:
            return False
    return True


def glolakes_batch_cache_paths(cfg: SWSConfig, stem: str) -> Dict[str, Path]:
    """Paths for catalog/volume cache files tied to a batch criteria stem."""
    cache_dir = cfg.processed_dir / "glolakes_batch"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cache_dir": cache_dir,
        "stem": Path(stem),
        "catalog_parquet": cache_dir / f"{stem}_catalog.parquet",
        "catalog_csv": cache_dir / f"{stem}_catalog.csv",
        "volume_parquet": cache_dir / f"{stem}_volume.parquet",
        "manifest": cache_dir / f"{stem}_manifest.json",
    }


def _prepare_batch_catalog(catalog: pd.DataFrame) -> pd.DataFrame:
    """Catalog table for save/load (no geometry / redundant flags)."""
    out = catalog.copy()
    return out.drop(columns=["geometry", "in_arid"], errors="ignore")


def _volume_series_to_long(series_map: Dict[Any, pd.Series]) -> pd.DataFrame:
    parts = []
    for lake_id, series in series_map.items():
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        df = s.rename("delta_v_km3").reset_index()
        if df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date"})
        df["lake_id"] = int(lake_id)
        parts.append(df[["lake_id", "date", "delta_v_km3"]])
    if not parts:
        return pd.DataFrame(columns=["lake_id", "date", "delta_v_km3"])
    return pd.concat(parts, ignore_index=True)


def _volume_series_from_long(vol_df: pd.DataFrame) -> Dict[int, pd.Series]:
    series_map: Dict[int, pd.Series] = {}
    if vol_df.empty:
        return series_map
    vol_df = vol_df.copy()
    vol_df["date"] = pd.to_datetime(vol_df["date"])
    for lake_id, grp in vol_df.groupby("lake_id"):
        s = grp.set_index("date")["delta_v_km3"].sort_index()
        s.name = "delta_v_km3"
        series_map[int(lake_id)] = s.astype(np.float64)
    return series_map


def _batch_cache_manifest(
    *,
    stem: str,
    product: str,
    version: str,
    max_lakes: Optional[int],
    sort_by: str,
    interpolate: bool,
    interp_method: str,
    interp_max_gap: Optional[int],
    interp_extrapolate_edges: bool,
    cfg: SWSConfig,
    n_catalog: int,
    n_volume: int,
    paths: Dict[str, Path],
) -> Dict[str, Any]:
    return {
        "stem": stem,
        "created_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "product": product,
        "version": version,
        "max_lakes": max_lakes,
        "sort_by": normalize_sort_by(sort_by),
        "interpolate": interpolate,
        "interp_method": interp_method if interpolate else None,
        "interp_max_gap": interp_max_gap if interpolate else None,
        "interp_extrapolate_edges": interp_extrapolate_edges if interpolate else None,
        "baseline_start": cfg.baseline_start,
        "baseline_end": cfg.baseline_end,
        "analysis_start": cfg.analysis_start,
        "analysis_end": cfg.analysis_end,
        "n_catalog_rows": n_catalog,
        "n_volume_series": n_volume,
        "catalog_parquet": str(paths["catalog_parquet"]),
        "catalog_csv": str(paths["catalog_csv"]),
        "volume_parquet": str(paths["volume_parquet"]),
    }


def save_glolakes_batch_cache(
    ranked_catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    paths: Dict[str, Path],
    manifest: Dict[str, Any],
) -> None:
    """Write catalog + volume series cache files."""
    catalog = _prepare_batch_catalog(ranked_catalog)
    catalog.to_parquet(paths["catalog_parquet"], index=False)
    catalog.to_csv(paths["catalog_csv"], index=False)
    _volume_series_to_long(volume_series).to_parquet(paths["volume_parquet"], index=False)
    paths["manifest"].write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def load_glolakes_batch_cache(paths: Dict[str, Path]) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    """Load catalog + volume series from cache files."""
    if not paths["catalog_parquet"].exists() or not paths["volume_parquet"].exists():
        raise FileNotFoundError(f"Batch cache incomplete for stem {paths['stem']}")
    catalog = pd.read_parquet(paths["catalog_parquet"])
    vol_df = pd.read_parquet(paths["volume_parquet"])
    return catalog, _volume_series_from_long(vol_df)


def build_glolakes_swsa_batch(
    product: str = "absolute_icesat2",
    version: str = "v1.0",
    cfg: Optional[SWSConfig] = None,
    max_lakes: Optional[int] = None,
    sort_by: str = "completeness_pct",
    interpolate: bool = False,
    interp_method: str = "linear",
    interp_max_gap: Optional[int] = 6,
    interp_extrapolate_edges: bool = False,
    use_cache: bool = True,
    force_rebuild: bool = False,
) -> Tuple[pd.DataFrame, Dict[int, pd.Series]]:
    """Process all arid GloLakes lakes to monthly volume storage anomalies (km³).

    ``completeness_pct`` in the returned catalog is always computed from the
    raw (pre-interpolation) series. ``volume_series`` may be interpolated when
    ``interpolate=True``.

    When ``use_cache=True`` (default), results are saved under
    ``cfg.processed_dir / glolakes_batch /`` using a short filename stem
    (product, version, lake count, interpolation). Full criteria including
    sort order and analysis windows are stored in the companion manifest JSON.
    Re-running with identical criteria loads the cache instead of reprocessing.
    """
    cfg = cfg or load_sws_config()
    if interpolate:
        interp_method = normalize_interp_method(interp_method)

    stem = glolakes_batch_cache_stem(
        product,
        version,
        max_lakes,
        interpolate,
        interp_method,
        interp_max_gap,
        interp_extrapolate_edges,
    )
    paths = glolakes_batch_cache_paths(cfg, stem)

    if use_cache and not force_rebuild and paths["manifest"].exists():
        try:
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            if not _batch_manifest_matches_request(
                manifest,
                product=product,
                version=version,
                max_lakes=max_lakes,
                sort_by=sort_by,
                interpolate=interpolate,
                interp_method=interp_method,
                interp_max_gap=interp_max_gap,
                interp_extrapolate_edges=interp_extrapolate_edges,
                cfg=cfg,
            ):
                logger.warning(
                    "Batch cache %s found but criteria differ (see manifest); rebuilding.",
                    stem,
                )
            else:
                ranked, series_map = load_glolakes_batch_cache(paths)
                _note(f"loaded batch cache ({len(series_map)} lakes)")
                _item(f"{_rel(paths['cache_dir'])}/{stem}_*", "ok")
                return _prepare_batch_catalog(ranked), series_map
        except Exception as exc:
            logger.warning("Batch cache load failed (%s); rebuilding.", exc)

    arid_catalog = build_glolakes_arid_catalog(
        product=product, version=version, cfg=cfg, sort_by="area",
    )
    if max_lakes:
        arid_catalog = arid_catalog.head(max_lakes)

    ds = load_glolakes_dataset(product=product, version=version, cfg=cfg)
    id_index = {int(lid): i for i, lid in enumerate(ds["ID"].values.astype(int))}
    gl_times = pd.to_datetime(ds["time"].values)

    def _one(row):
        lake_id = int(row.lake_id)
        try:
            ts = extract_glolakes_timeseries(
                ds, lake_id, id_index=id_index, times=gl_times,
            )
            vol = process_lake_to_volume_anomaly(
                ts,
                f"GloLakes_{product}",
                cfg.baseline_start,
                cfg.baseline_end,
                cfg.analysis_start,
                cfg.analysis_end,
            )
            return lake_id, vol
        except Exception as exc:
            logger.warning("GloLakes volume anomaly failed for %s: %s", lake_id, exc)
            return lake_id, None

    rows = list(arid_catalog.itertuples())
    n_workers = cfg.n_process_workers if cfg.n_process_workers > 0 else (os.cpu_count() or 1)
    _announce_resources()
    _note(f"processing {len(rows)} lakes ({n_workers} thread workers)")
    results = _parallel_thread_map(
        _one,
        rows,
        n_jobs=cfg.n_process_workers,
        desc="GloLakes volumes",
        unit="lake",
    )

    n_total = len(results)
    failed_ids = [lake_id for lake_id, vol_raw in results if vol_raw is None]
    _summarize_skipped("lakes", len(failed_ids), n_total, examples=failed_ids)

    raw_series_map: Dict[int, pd.Series] = {}
    series_map: Dict[int, pd.Series] = {}
    for lake_id, vol_raw in results:
        if vol_raw is None:
            continue
        raw_series_map[lake_id] = vol_raw
        series_map[lake_id] = (
            interpolate_series_gaps(
                vol_raw,
                interp_method,
                max_gap=interp_max_gap,
                extrapolate_edges=interp_extrapolate_edges,
            )
            if interpolate
            else vol_raw
        )

    ranked = rank_datasets_by_completeness(
        arid_catalog,
        raw_series_map,
        cfg.analysis_start,
        cfg.analysis_end,
        sort_by=sort_by,
    )
    ranked = _prepare_batch_catalog(ranked)

    if use_cache:
        manifest = _batch_cache_manifest(
            stem=stem,
            product=product,
            version=version,
            max_lakes=max_lakes,
            sort_by=sort_by,
            interpolate=interpolate,
            interp_method=interp_method,
            interp_max_gap=interp_max_gap,
            interp_extrapolate_edges=interp_extrapolate_edges,
            cfg=cfg,
            n_catalog=len(ranked),
            n_volume=len(series_map),
            paths=paths,
        )
        save_glolakes_batch_cache(ranked, series_map, paths, manifest)
        _note(f"saved batch cache ({len(series_map)} lakes)")
        _item(f"{_rel(paths['cache_dir'])}/{stem}_*", "ok")

    return ranked, series_map


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _style_publication_axes(ax: plt.Axes, *, ylabel: str, title: str) -> None:
    ax.set_ylabel(ylabel, fontsize=PLOT_FONTS["label"])
    ax.set_title(title, fontsize=PLOT_FONTS["title"], fontweight="medium", pad=10)
    ax.tick_params(axis="y", labelsize=PLOT_FONTS["tick"], direction="out", length=4)
    ax.grid(True, linestyle=":", alpha=0.45, linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.set_major_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_minor_locator(mdates.MonthLocator())
    ax.tick_params(axis="x", which="major", labelsize=PLOT_FONTS["tick"], rotation=45)
    ax.tick_params(axis="x", which="minor", length=2.5, width=0.5)


def _add_stats_box(ax: plt.Axes, text: str, loc: str = "upper left") -> None:
    positions = {
        "upper left": (0.02, 0.98, "top", "left"),
        "lower right": (0.98, 0.02, "bottom", "right"),
    }
    x, y, va, ha = positions.get(loc, positions["upper left"])
    ax.text(
        x, y, text,
        transform=ax.transAxes, va=va, ha=ha,
        bbox=dict(boxstyle="round,pad=0.45", facecolor="white", edgecolor="0.75", alpha=0.92),
        fontsize=PLOT_FONTS["annotation"],
        linespacing=1.35,
    )


def _std_cm(series: pd.Series) -> float:
    s = series.dropna()
    return float(s.std(ddof=1)) if len(s) > 1 else np.nan


def _lake_grace_stats_annotation(
    lake_cm: pd.Series,
    grace_cm: pd.Series,
    *,
    lakes_label: str = "Lakes",
) -> str:
    """Std-dev annotation (cm) — same units as the time series."""
    return (
        f"{lakes_label}: σ = {_std_cm(lake_cm):.2f} cm\n"
        f"GRACE: σ = {_std_cm(grace_cm):.2f} cm"
    )


def _series_ylim_with_headroom(
    *series: pd.Series,
    top_headroom_frac: float = 0.22,
    bottom_headroom_frac: float = 0.08,
) -> Tuple[float, float]:
    """Y-limits with 100% padding on extrema plus extra room for the stats box."""
    arrays = [s.dropna().values for s in series if s is not None and len(s.dropna())]
    if not arrays:
        return -1.0, 1.0
    vals = np.concatenate(arrays)
    ylo, yhi = float(np.nanmin(vals)), float(np.nanmax(vals))
    ymax = yhi + abs(yhi) if yhi != 0 else yhi + max(abs(ylo), 1e-3)
    ymin = ylo - abs(ylo) if ylo != 0 else ylo - max(abs(yhi), 1e-3)
    if ymin >= ymax:
        pad = max(abs(yhi - ylo), 1e-3)
        ymin, ymax = ylo - pad, yhi + pad
    span = ymax - ymin
    ymax += span * top_headroom_frac
    ymin -= span * bottom_headroom_frac
    return ymin, ymax


def _save_figure(fig: plt.Figure, save_path: Optional[Union[str, Path]], dpi: int, show: bool) -> None:
    if save_path:
        fig.savefig(save_path, dpi=dpi, format="jpeg", bbox_inches="tight", facecolor="white")
    if show:
        plt.show()
    else:
        plt.close(fig)


def plot_lake_volume_anomaly(
    lake_meta: Dict[str, Any],
    delta_v_km3: pd.Series,
    precip_mm: Optional[pd.Series] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = PLOT_DPI,
    precip_ylim_factor: float = 4.0,
    show: bool = True,
) -> plt.Figure:
    """Plot lake volume storage anomaly (km³) with optional precipitation."""
    name = _display_lake_name(lake_meta.get("lake_name"))
    country = lake_meta.get("country") or "Unknown"
    stats = compute_volume_trend_stats(delta_v_km3)

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE, layout="constrained")
    ax1.plot(delta_v_km3.index, delta_v_km3.values, color="#2166ac", linewidth=2.0, label="ΔV")
    ax1.axhline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    _style_publication_axes(ax1, ylabel="Storage anomaly (km³)", title=f"{name} — {country}")

    _add_stats_box(
        ax1,
        f"σ = {stats['std_km3']:.3f} km³\n"
        f"trend = {stats['trend_km3_yr']:.3f} km³ yr$^{{-1}}$",
    )

    lines = ax1.get_lines()
    labels = ["ΔV"]

    if precip_mm is not None and len(precip_mm.dropna()) > 0:
        ax2 = ax1.twinx()
        precip = precip_mm.dropna()
        ax2.bar(
            precip.index, precip.values,
            width=22, color="#b35806", alpha=0.45, label="Precipitation", zorder=0,
        )
        pmax = float(np.nanmax(precip.values))
        ax2.set_ylim(pmax * precip_ylim_factor, 0)
        ax2.set_ylabel("Precipitation (mm)", fontsize=PLOT_FONTS["label"])
        ax2.tick_params(axis="y", labelsize=PLOT_FONTS["tick"])
        ax2.spines["top"].set_visible(False)
        lines = lines + [ax2.patches[0]] if ax2.patches else lines
        labels.append("Precipitation")

    ylo, yhi = np.nanmin(delta_v_km3.values), np.nanmax(delta_v_km3.values)
    ymin, ymax = _series_ylim_with_headroom(delta_v_km3)
    ax1.set_ylim(ymin, ymax)

    ax1.legend(lines[: len(labels)], labels, loc="lower left", fontsize=PLOT_FONTS["legend"], framealpha=0.9)
    _save_figure(fig, save_path, dpi, show)
    return fig


# ---------------------------------------------------------------------------
# GRACE & precipitation (via grace_analysis_utils)
# ---------------------------------------------------------------------------

def _import_grace_analysis_utils():
    """Import shared GRACE/predictor helpers from ``src/`` (same as notebooks)."""
    import sys

    src_dir = Path(__file__).resolve().parent
    if str(src_dir) not in sys.path:
        sys.path.insert(0, str(src_dir))
    from grace_analysis_utils import process_grace_data, process_predictor_fine

    return process_grace_data, process_predictor_fine


def build_grace_time_range(cfg: Optional[SWSConfig] = None) -> pd.DatetimeIndex:
    """Month-end time index for GRACE / predictor processing."""
    cfg = cfg or load_sws_config()
    return pd.date_range(cfg.grace_time_start, cfg.grace_time_end, freq="ME")


def load_aoi_geometry(cfg: Optional[SWSConfig] = None):
    """Arid-region geometry for clipping GRACE and precipitation."""
    cfg = cfg or load_sws_config()
    return load_arid_mask(cfg.arid_areas_path).geometry


def process_grace_mean(
    cfg: Optional[SWSConfig] = None,
    aoi_geometry=None,
    time_range: Optional[pd.DatetimeIndex] = None,
) -> xr.DataArray:
    """
    Process CSR, JPL, and GSFC GRACE solutions and return their mean (1° TWS, cm).

    Uses the same ``process_grace_data`` workflow as the main arid subbasin notebook.
    """
    process_grace_data, _ = _import_grace_analysis_utils()
    cfg = cfg or load_sws_config()
    for label, path in (
        ("CSR GRACE", cfg.csr_grace_path),
        ("JPL GRACE", cfg.jpl_grace_path),
        ("GSFC GRACE", cfg.gsfc_grace_path),
        ("CSR land mask", cfg.csr_mask_path),
    ):
        if not Path(path).exists():
            raise FileNotFoundError(
                f"{label} not found: {_rel(path)}. "
                "Run notebook 01 or download_grace_mascons() first."
            )
    aoi_geometry = aoi_geometry if aoi_geometry is not None else load_aoi_geometry(cfg)
    time_range = time_range if time_range is not None else build_grace_time_range(cfg)

    grace_csr = process_grace_data(
        str(cfg.csr_grace_path),
        aoi_geometry,
        time_range,
        variable_name="lwe_thickness",
        land_mask_file=str(cfg.csr_mask_path),
    )
    grace_csr.name = "GRACE_CSR"

    grace_jpl = process_grace_data(
        str(cfg.jpl_grace_path),
        aoi_geometry,
        time_range,
        variable_name="lwe_thickness",
        apply_scaling_factor=False,
    )
    grace_jpl.name = "GRACE_JPL"

    grace_gsfc = process_grace_data(
        str(cfg.gsfc_grace_path),
        aoi_geometry,
        time_range,
        variable_name="lwe_thickness",
    )
    grace_gsfc.name = "GRACE_GSFC"

    grace_mean = (grace_csr + grace_jpl + grace_gsfc) / 3
    grace_mean.name = "GRACE_Mean"
    grace_mean.attrs["description"] = "Mean of CSR, JPL and GSFC GRACE TWS solutions"
    return grace_mean


def process_precip_on_grace_grid(
    grace_mean: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    precip_path: Optional[Union[str, Path]] = None,
    aoi_geometry=None,
    time_range: Optional[pd.DatetimeIndex] = None,
    precip_variable: str = "precipitation",
) -> xr.DataArray:
    """
    Process GPM precipitation onto the GRACE 1° grid (mm month⁻¹).

    Run once, then extract lake windows with ``extract_precip_at_lake``.
    """
    _, process_predictor_fine = _import_grace_analysis_utils()
    cfg = cfg or load_sws_config()
    precip_path = Path(precip_path or cfg.precip_path)
    if not precip_path.exists():
        raise FileNotFoundError(
            f"Precipitation Zarr not found: {_rel(precip_path)}. "
            "Use resolve_precip_path() or run notebook 01 first."
        )
    aoi_geometry = aoi_geometry if aoi_geometry is not None else load_aoi_geometry(cfg)
    time_range = time_range if time_range is not None else build_grace_time_range(cfg)

    precip_da = process_predictor_fine(
        grace_mean,
        str(precip_path),
        precip_variable,
        aoi_geometry,
        time_range,
    )
    precip_da.name = "precip_mm"
    return precip_da


def _normalize_lon_for_grid(lon: float, da: xr.DataArray) -> float:
    if lon < 0 and float(da.lon.max()) > 180:
        return lon % 360
    return lon


def _normalize_grid_da(da: xr.DataArray) -> xr.DataArray:
    if "latitude" in da.dims:
        return da.rename({"latitude": "lat", "longitude": "lon"})
    return da


def _validate_grace_window_deg(window_deg: float) -> int:
    w = int(window_deg)
    if w not in (1, 3):
        raise ValueError(f"window_deg must be 1 or 3 (GRACE cells), got {window_deg}")
    return w


def _haversine_distance_deg(
    lat: float,
    lon: float,
    grid_lats: np.ndarray,
    grid_lons: np.ndarray,
) -> np.ndarray:
    """Great-circle distance (degrees) from (lat, lon) to each grid node."""
    lat_diff_rad = np.radians(grid_lats[:, np.newaxis] - lat)
    lon_diff_rad = np.radians(grid_lons[np.newaxis, :] - lon)
    a = (
        np.sin(lat_diff_rad / 2) ** 2
        + np.cos(np.radians(grid_lats[:, np.newaxis]))
        * np.cos(np.radians(lat))
        * np.sin(lon_diff_rad / 2) ** 2
    )
    distances_rad = 2 * np.arcsin(np.sqrt(np.clip(a, 0, 1)))
    return np.degrees(distances_rad)


def assign_grace_grid_pixel(
    grid_da: xr.DataArray,
    lake_lat: float,
    lake_lon: float,
) -> Dict[str, Any]:
    """Assign a lake centroid to the nearest GRACE grid cell centre (haversine)."""
    data = _normalize_grid_da(grid_da)
    lon = _normalize_lon_for_grid(lake_lon, data)
    grid_lats = data.lat.values.astype(float)
    grid_lons = data.lon.values.astype(float)
    distances = _haversine_distance_deg(lake_lat, lon, grid_lats, grid_lons)
    min_idx = np.unravel_index(int(np.argmin(distances)), distances.shape)
    grace_lat = float(grid_lats[min_idx[0]])
    grace_lon = float(grid_lons[min_idx[1]])
    return {
        "assignment_method": "haversine_nearest_cell",
        "lake_lat": float(lake_lat),
        "lake_lon": float(lon),
        "grace_lat": grace_lat,
        "grace_lon": grace_lon,
        "haversine_distance_deg": float(distances[min_idx]),
    }


def _grid_index_for_coord(values: np.ndarray, target: float) -> int:
    return int(np.nanargmin(np.abs(values.astype(float) - float(target))))


def _select_grace_cells(
    grid_da: xr.DataArray,
    grace_lat: float,
    grace_lon: float,
    window_deg: float,
) -> xr.DataArray:
    """Select GRACE/precip cells: one assigned pixel (window=1) or 3×3 block (window=3)."""
    w = _validate_grace_window_deg(window_deg)
    data = _normalize_grid_da(grid_da)
    lat_vals = data.lat.values.astype(float)
    lon_vals = data.lon.values.astype(float)

    if w == 1:
        return data.sel(lat=grace_lat, lon=grace_lon)

    i = _grid_index_for_coord(lat_vals, grace_lat)
    j = _grid_index_for_coord(lon_vals, grace_lon)
    i0, i1 = max(0, i - 1), min(len(lat_vals), i + 2)
    j0, j1 = max(0, j - 1), min(len(lon_vals), j + 2)
    return data.isel(lat=slice(i0, i1), lon=slice(j0, j1))


def _extract_grid_timeseries(
    grid_da: xr.DataArray,
    grace_lat: float,
    grace_lon: float,
    window_deg: float,
) -> pd.Series:
    """Spatial mean over selected GRACE/precip cells (no baseline removal)."""
    subset = _select_grace_cells(grid_da, grace_lat, grace_lon, window_deg)
    spatial_dims = [d for d in ("lat", "lon") if d in subset.dims]
    ts = subset.mean(dim=spatial_dims, skipna=True) if spatial_dims else subset
    series = ts.to_series()
    series.index = pd.to_datetime(series.index)
    return series.sort_index()


def extract_precip_at_lake(
    precip_da: xr.DataArray,
    lat: float,
    lon: float,
    window_deg: float = 1.0,
    analysis_start: str = ANALYSIS_START,
    analysis_end: str = ANALYSIS_END,
) -> pd.Series:
    """Extract mean monthly precipitation (mm) at the haversine-assigned GRACE pixel(s)."""
    assignment = assign_grace_grid_pixel(precip_da, lat, lon)
    series = _extract_grid_timeseries(
        precip_da,
        assignment["grace_lat"],
        assignment["grace_lon"],
        window_deg,
    )
    series.name = "precip_mm"
    return clip_to_analysis_window(series, analysis_start, analysis_end)


def _lake_lat_lon(lake_meta: Dict[str, Any]) -> Tuple[float, float]:
    lat = lake_meta.get("lat_centroid", lake_meta.get("lat", lake_meta.get("latitude")))
    lon = lake_meta.get("lon_centroid", lake_meta.get("lon", lake_meta.get("longitude")))
    if lat is None or lon is None:
        raise ValueError(
            "lake_meta missing lat/lon; expected one of "
            "lat_centroid/lat/latitude and lon_centroid/lon/longitude"
        )
    return float(lat), float(lon)


def _cm_water_ylim(*series: pd.Series) -> Tuple[float, float]:
    """Left-axis limits for cm water-equivalent series (includes stats-box headroom)."""
    return _series_ylim_with_headroom(*series)


def _precip_ylim(pmax: float, factor: float = 2.5) -> Tuple[float, float]:
    """Precip axis: inverted bar chart with ymax = 150% above max (2.5× pmax)."""
    pmax = max(float(pmax), 1e-6)
    return pmax * factor, 0.0


def plot_example_lake_volume(
    lake_id: int,
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    save_path: Optional[Union[str, Path]] = None,
    show: bool = True,
) -> plt.Figure:
    """Quick single-lake volume anomaly plot (km³) for testing."""
    if lake_id not in volume_series:
        raise KeyError(f"lake_id {lake_id} not in volume_series")
    row = catalog.loc[catalog["lake_id"] == lake_id]
    if row.empty:
        raise KeyError(f"lake_id {lake_id} not in catalog")
    return plot_lake_volume_anomaly(
        row.iloc[0].to_dict(),
        volume_series[lake_id],
        save_path=save_path,
        show=show,
    )


def plot_example_lake_volumes(
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    n_lakes: int = 5,
    show: bool = True,
    save_dir: Optional[Union[str, Path]] = None,
) -> List[int]:
    """Plot volume anomalies for the first *n_lakes* catalog entries with data."""
    lake_ids = [
        int(lid) for lid in catalog["lake_id"].head(n_lakes)
        if int(lid) in volume_series
    ]
    for lake_id in lake_ids:
        save_path = None
        if save_dir is not None:
            save_path = Path(save_dir) / f"example_{lake_id}_volume.jpeg"
        plot_example_lake_volume(
            lake_id, catalog, volume_series, save_path=save_path, show=show,
        )
    return lake_ids


def plot_lake_grace_precip_comparison(
    lake_meta: Dict[str, Any],
    volume_anomaly_km3: pd.Series,
    grace_da: xr.DataArray,
    precip_da: Optional[xr.DataArray] = None,
    precip_mm: Optional[pd.Series] = None,
    window_deg: float = 1.0,
    cfg: Optional[SWSConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = PLOT_DPI,
    show: bool = True,
) -> plt.Figure:
    """
    Lake ΔV and GRACE TWSA on left axis (cm water equiv. over GRACE window);
    precipitation on right axis (mm).
    """
    cfg = cfg or load_sws_config()
    name = _display_lake_name(lake_meta.get("lake_name"))
    country = lake_meta.get("country") or "Unknown"
    lat, lon = _lake_lat_lon(lake_meta)
    assignment = assign_grace_grid_pixel(grace_da, lat, lon)
    grace_lat, grace_lon = assignment["grace_lat"], assignment["grace_lon"]
    cell_area = grace_window_area_km2(grace_lat, window_deg)

    lake_grace_cm = lake_volume_to_grace_cm(volume_anomaly_km3, grace_lat, window_deg)
    twsa_cm = extract_grace_twsa_at_lake(
        grace_da,
        lat,
        lon,
        window_deg=window_deg,
        analysis_start=cfg.analysis_start,
        analysis_end=cfg.analysis_end,
    )

    if precip_mm is None and precip_da is not None:
        precip_mm = extract_precip_at_lake(
            precip_da,
            lat,
            lon,
            window_deg=window_deg,
            analysis_start=cfg.analysis_start,
            analysis_end=cfg.analysis_end,
        )

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE, layout="constrained")
    ax1.plot(
        lake_grace_cm.index,
        lake_grace_cm.values,
        label=f"Lake ΔV / {window_deg}° ({cell_area:.0f} km²)",
        color=PLOT_COLORS["lake"],
        linewidth=2.2,
        zorder=3,
    )
    ax1.plot(
        twsa_cm.index,
        twsa_cm.values,
        label=f"GRACE TWSA / {window_deg}°",
        color=PLOT_COLORS["grace"],
        linewidth=2.0,
        zorder=3,
    )
    ax1.axhline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    _style_publication_axes(
        ax1,
        ylabel=GRACE_COMP_YLABEL,
        title=f"{name} — {country}",
    )

    ymin, ymax = _cm_water_ylim(lake_grace_cm, twsa_cm)
    ax1.set_ylim(ymin, ymax)

    _add_stats_box(
        ax1,
        _lake_grace_stats_annotation(lake_grace_cm, twsa_cm, lakes_label="Lakes"),
        loc="lower right",
    )

    lines, labels = ax1.get_legend_handles_labels()
    if precip_mm is not None and len(precip_mm.dropna()) > 0:
        ax2 = ax1.twinx()
        precip = precip_mm.dropna()
        ax2.bar(
            precip.index,
            precip.values,
            width=22,
            color=PLOT_COLORS["precip"],
            alpha=PLOT_PRECIP_ALPHA,
            label="Precipitation",
            zorder=1,
        )
        pmax = float(np.nanmax(precip.values))
        ax2.set_ylim(*_precip_ylim(pmax))
        ax2.set_ylabel("Precipitation (mm)", fontsize=PLOT_FONTS["label"])
        ax2.tick_params(axis="y", labelsize=PLOT_FONTS["tick"])
        ax2.spines["top"].set_visible(False)
        if ax2.patches:
            lines += [ax2.patches[0]]
            labels.append("Precipitation")

    ax1.legend(lines, labels, loc="lower left", fontsize=PLOT_FONTS["legend"], framealpha=0.92)
    _save_figure(fig, save_path, dpi, show)
    return fig


def plot_grace_pixel_lake_grace_comparison(
    grace_lat: float,
    grace_lon: float,
    lake_ids: Sequence[int],
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    precip_da: Optional[xr.DataArray] = None,
    window_deg: float = 1.0,
    cfg: Optional[SWSConfig] = None,
    save_path: Optional[Union[str, Path]] = None,
    dpi: int = PLOT_DPI,
    show: bool = True,
) -> plt.Figure:
    """Plot summed lake ΔV vs GRACE TWSA for all lakes in one assigned pixel."""
    cfg = cfg or load_sws_config()
    ids = [int(lid) for lid in lake_ids if int(lid) in volume_series]
    combined_km3 = _sum_volume_anomalies([volume_series[lid] for lid in ids])
    lake_cm, twsa_cm, _ = _pixel_comparison_series(
        grace_lat, grace_lon, combined_km3, grace_da, cfg, window_deg,
    )
    cell_area = grace_window_area_km2(grace_lat, window_deg)
    lake_rows = catalog.loc[catalog["lake_id"].isin(ids)]
    names = [_display_lake_name(n) for n in lake_rows.get("lake_name", pd.Series(dtype=object))]
    title = names[0] if len(names) == 1 else f"{len(ids)} lakes @ GRACE ({grace_lat:.1f}°, {grace_lon:.1f}°)"

    precip_mm = None
    if precip_da is not None:
        precip_mm = clip_to_analysis_window(
            _extract_grid_timeseries(precip_da, grace_lat, grace_lon, window_deg),
            cfg.analysis_start,
            cfg.analysis_end,
        )
        precip_mm.name = "precip_mm"

    fig, ax1 = plt.subplots(figsize=PLOT_FIGSIZE, layout="constrained")
    ax1.plot(
        lake_cm.index, lake_cm.values,
        label=f"Σ Lake ΔV / {window_deg}° ({cell_area:.0f} km²)",
        color=PLOT_COLORS["lake"], linewidth=2.2, zorder=3,
    )
    ax1.plot(
        twsa_cm.index, twsa_cm.values,
        label=f"GRACE TWSA / {window_deg}°",
        color=PLOT_COLORS["grace"], linewidth=2.0, zorder=3,
    )
    ax1.axhline(0, color="0.35", linewidth=0.9, linestyle="--", zorder=0)
    _style_publication_axes(ax1, ylabel=GRACE_COMP_YLABEL, title=title)
    ax1.set_ylim(*_cm_water_ylim(lake_cm, twsa_cm))
    lakes_label = "Σ Lakes" if len(ids) > 1 else "Lakes"
    _add_stats_box(
        ax1,
        _lake_grace_stats_annotation(lake_cm, twsa_cm, lakes_label=lakes_label),
        loc="lower right",
    )

    lines, labels = ax1.get_legend_handles_labels()
    if precip_mm is not None and len(precip_mm.dropna()) > 0:
        ax2 = ax1.twinx()
        precip = precip_mm.dropna()
        ax2.bar(
            precip.index, precip.values, width=22,
            color=PLOT_COLORS["precip"], alpha=PLOT_PRECIP_ALPHA, zorder=1,
        )
        ax2.set_ylim(*_precip_ylim(float(np.nanmax(precip.values))))
        ax2.set_ylabel("Precipitation (mm)", fontsize=PLOT_FONTS["label"])
        ax2.tick_params(axis="y", labelsize=PLOT_FONTS["tick"])
        ax2.spines["top"].set_visible(False)
        if ax2.patches:
            lines += [ax2.patches[0]]
            labels.append("Precipitation")

    ax1.legend(lines, labels, loc="lower left", fontsize=PLOT_FONTS["legend"], framealpha=0.92)
    _save_figure(fig, save_path, dpi, show)
    return fig


def describe_grace_window_assignment(
    grid_da: xr.DataArray,
    lat: float,
    lon: float,
    window_deg: float = 1.0,
) -> Dict[str, Any]:
    """
    Describe how a lake location is mapped to GRACE / precip grid pixels.

    1. Haversine scan over all grid centres → nearest cell ``(grace_lat, grace_lon)``.
    2. ``window_deg=1``: that single cell.
    3. ``window_deg=3``: unweighted mean over the 3×3 block centred on that cell
       (fewer than 9 at grid edges).
    4. GRACE values are used as-is (solution anomalies); no extra baseline removal.
    """
    assignment = assign_grace_grid_pixel(grid_da, lat, lon)
    subset = _select_grace_cells(
        grid_da,
        assignment["grace_lat"],
        assignment["grace_lon"],
        window_deg,
    )
    spatial = subset.isel(time=0, drop=True) if "time" in subset.dims else subset
    n_lat = int(spatial.lat.size) if "lat" in spatial.dims else 1
    n_lon = int(spatial.lon.size) if "lon" in spatial.dims else 1
    return {
        **assignment,
        "window_deg": float(_validate_grace_window_deg(window_deg)),
        "n_grid_lat": n_lat,
        "n_grid_lon": n_lon,
        "n_grace_pixels": n_lat * n_lon,
        "grid_lat_values": (
            spatial.lat.values.astype(float).tolist() if "lat" in spatial.dims else [assignment["grace_lat"]]
        ),
        "grid_lon_values": (
            spatial.lon.values.astype(float).tolist() if "lon" in spatial.dims else [assignment["grace_lon"]]
        ),
    }


def _lake_comparison_series(
    lake_meta: Dict[str, Any],
    volume_anomaly_km3: pd.Series,
    grace_da: xr.DataArray,
    cfg: SWSConfig,
    window_deg: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Lake and GRACE cm anomalies plus grid-assignment metadata."""
    lat, lon = _lake_lat_lon(lake_meta)
    grid_info = describe_grace_window_assignment(grace_da, lat, lon, window_deg)
    grace_lat = grid_info["grace_lat"]
    lake_cm = lake_volume_to_grace_cm(volume_anomaly_km3, grace_lat, window_deg)
    grace_cm = extract_grace_twsa_at_lake(
        grace_da,
        lat,
        lon,
        window_deg=window_deg,
        analysis_start=cfg.analysis_start,
        analysis_end=cfg.analysis_end,
    )
    return lake_cm, grace_cm, grid_info


def _sum_volume_anomalies(series_list: Sequence[pd.Series]) -> pd.Series:
    """Sum lake volume anomalies (km³); keep month if any lake has data."""
    if not series_list:
        return pd.Series(dtype=float, name="delta_v_km3")
    if len(series_list) == 1:
        out = series_list[0].copy()
        out.name = "delta_v_km3"
        return out
    combined = pd.concat(series_list, axis=1)
    out = combined.sum(axis=1, min_count=1)
    out.name = "delta_v_km3"
    return out


def group_lake_ids_by_grace_pixel(
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
) -> Dict[Tuple[float, float], List[int]]:
    """Group lake IDs by haversine-assigned ``(grace_lat, grace_lon)``."""
    groups: Dict[Tuple[float, float], List[int]] = {}
    for _, row in catalog.iterrows():
        lake_id = int(row.get("lake_id", row.name))
        if lake_id not in volume_series:
            continue
        lat, lon = _lake_lat_lon(row.to_dict())
        assignment = assign_grace_grid_pixel(grace_da, lat, lon)
        key = (assignment["grace_lat"], assignment["grace_lon"])
        groups.setdefault(key, []).append(lake_id)
    return groups


def _pixel_comparison_series(
    grace_lat: float,
    grace_lon: float,
    volume_anomaly_km3: pd.Series,
    grace_da: xr.DataArray,
    cfg: SWSConfig,
    window_deg: float,
) -> Tuple[pd.Series, pd.Series, Dict[str, Any]]:
    """Combined lake cm and GRACE cm at an assigned pixel centre."""
    grid_info = describe_grace_window_assignment(grace_da, grace_lat, grace_lon, window_deg)
    lake_cm = lake_volume_to_grace_cm(volume_anomaly_km3, grace_lat, window_deg)
    grace_cm = clip_to_analysis_window(
        _extract_grid_timeseries(grace_da, grace_lat, grace_lon, window_deg),
        cfg.analysis_start,
        cfg.analysis_end,
    )
    grace_cm.name = f"grace_twsa_cm_win{window_deg}"
    return lake_cm, grace_cm, grid_info


def _lake_grace_stats_row(
    lake_cm: pd.Series,
    grace_cm: pd.Series,
    *,
    record_type: str,
    window_deg: float,
    grid_info: Dict[str, Any],
    extra: Optional[Dict[str, Any]] = None,
    calculate_residual: bool = False,
) -> Dict[str, Any]:
    """Shared std / trend fields for lake-level or pixel-aggregated rows."""
    aligned = pd.concat([lake_cm, grace_cm], axis=1, keys=["lake", "grace"]).dropna()
    n_months = int(len(aligned))
    if n_months > 1:
        std_lake = float(aligned["lake"].std(ddof=1))
        std_grace = float(aligned["grace"].std(ddof=1))
    else:
        std_lake = std_grace = np.nan

    std_ratio_pct = (
        100.0 * std_lake / std_grace
        if np.isfinite(std_grace) and std_grace > 0
        else np.nan
    )
    lake_stats = compute_trend_stats(lake_cm)
    grace_stats = compute_trend_stats(grace_cm)

    row = {
        "record_type": record_type,
        "window_deg": window_deg,
        "grace_lat": grid_info["grace_lat"],
        "grace_lon": grid_info["grace_lon"],
        "haversine_distance_deg": grid_info.get("haversine_distance_deg"),
        "grace_window_area_km2": grace_window_area_km2(grid_info["grace_lat"], window_deg),
        "grid_assignment": grid_info["assignment_method"],
        "n_grace_pixels": grid_info["n_grace_pixels"],
        "lake_std_pct_of_grace": std_ratio_pct,
        "lake_std_cm": std_lake,
        "grace_std_cm": std_grace,
        "lake_trend_cm_yr": lake_stats["trend_yr"],
        "grace_trend_cm_yr": grace_stats["trend_yr"],
        "n_overlap_months": n_months,
    }

    if calculate_residual:
        lake_resid = _harmonic_decompose_residual(lake_cm)
        grace_resid = _harmonic_decompose_residual(grace_cm)
        aligned_resid = pd.concat(
            [lake_resid, grace_resid], axis=1, keys=["lake", "grace"],
        ).dropna()
        if len(aligned_resid) > 1:
            std_lake_r = float(aligned_resid["lake"].std(ddof=1))
            std_grace_r = float(aligned_resid["grace"].std(ddof=1))
        else:
            std_lake_r = std_grace_r = np.nan
        std_ratio_pct_r = (
            100.0 * std_lake_r / std_grace_r
            if np.isfinite(std_grace_r) and std_grace_r > 0
            else np.nan
        )
        row["lake_std_cm_residual"] = std_lake_r
        row["grace_std_cm_residual"] = std_grace_r
        row["lake_std_pct_of_grace_residual"] = std_ratio_pct_r

    if extra:
        row.update(extra)
    return row


def summarize_grace_pixel_row(
    grace_lat: float,
    grace_lon: float,
    lake_ids: Sequence[int],
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    calculate_residual: bool = False,
) -> Dict[str, Any]:
    """
    Compare summed lake ΔV (km³) in one GRACE pixel to GRACE TWSA.

    Lakes sharing ``(grace_lat, grace_lon)`` have their volume anomalies **summed
    in km³ first**, then converted once to cm over the GRACE footprint. Statistics
    (std, trends) are computed on that combined cm series — **not** by combining
    per-lake std values (which would ignore covariance and double-count footprint).

    calculate_residual : bool, default False
        If True, also decomposes both series (calendar-phase-locked linear trend +
        annual + semi-annual harmonic fit) and adds residual-based std/pct columns.
    """
    cfg = cfg or load_sws_config()
    ids = [int(lid) for lid in lake_ids if int(lid) in volume_series]
    if not ids:
        raise ValueError("No volume series for lake_ids in grace pixel group")

    combined_km3 = _sum_volume_anomalies([volume_series[lid] for lid in ids])
    lake_cm, grace_cm, grid_info = _pixel_comparison_series(
        grace_lat, grace_lon, combined_km3, grace_da, cfg, window_deg,
    )

    lake_rows = catalog.loc[catalog["lake_id"].isin(ids)]
    names = [_display_lake_name(n) for n in lake_rows.get("lake_name", pd.Series(dtype=object))]
    countries = lake_rows["country"].dropna().unique().tolist() if "country" in lake_rows else []
    areas = lake_rows["area_km2"].astype(float) if "area_km2" in lake_rows else pd.Series(dtype=float)
    completeness = (
        lake_rows["completeness_pct"].astype(float)
        if "completeness_pct" in lake_rows
        else pd.Series(dtype=float)
    )

    extra = {
        "lake_id": ids[0] if len(ids) == 1 else None,
        "lake_ids": ",".join(str(i) for i in ids),
        "n_lakes": len(ids),
        "lake_name": names[0] if len(names) == 1 else f"{len(ids)} lakes",
        "lake_names": "; ".join(names[:8]) + ("; …" if len(names) > 8 else ""),
        "country": countries[0] if len(countries) == 1 else "Multiple",
        "lat": grace_lat,
        "lon": grace_lon,
        "area_km2": float(areas.sum()) if len(areas) else np.nan,
        "completeness_pct": float(completeness.mean()) if len(completeness) else np.nan,
    }
    return _lake_grace_stats_row(
        lake_cm, grace_cm,
        record_type="grace_pixel",
        window_deg=window_deg,
        grid_info=grid_info,
        extra=extra,
        calculate_residual=calculate_residual,
    )


def summarize_lake_grace_row(
    lake_meta: Dict[str, Any],
    volume_anomaly_km3: pd.Series,
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    calculate_residual: bool = False,
) -> Dict[str, Any]:
    """
    Per-lake summary statistics for lake vs GRACE comparison (cm units).

    calculate_residual : bool, default False
        If True, also decomposes both series (calendar-phase-locked linear trend +
        annual + semi-annual harmonic fit) and adds residual-based std/pct columns.
    """
    cfg = cfg or load_sws_config()
    lat, lon = _lake_lat_lon(lake_meta)
    lake_cm, grace_cm, grid_info = _lake_comparison_series(
        lake_meta, volume_anomaly_km3, grace_da, cfg, window_deg,
    )
    extra = {
        "lake_id": lake_meta.get("lake_id"),
        "lake_ids": str(lake_meta.get("lake_id")),
        "n_lakes": 1,
        "lake_name": _display_lake_name(lake_meta.get("lake_name")),
        "lake_names": _display_lake_name(lake_meta.get("lake_name")),
        "country": lake_meta.get("country") or "Unknown",
        "lat": lat,
        "lon": lon,
        "area_km2": lake_meta.get("area_km2"),
        "completeness_pct": lake_meta.get("completeness_pct"),
    }
    return _lake_grace_stats_row(
        lake_cm, grace_cm,
        record_type="lake",
        window_deg=window_deg,
        grid_info=grid_info,
        extra=extra,
        calculate_residual=calculate_residual,
    )


def format_grace_summary_export(
    summary_df: pd.DataFrame,
    *,
    by_grace_pixel: bool = False,
) -> pd.DataFrame:
    """
    Drop redundant columns and round numerics for analysis-ready CSV export.

    Pixel mode removes duplicate coords/metadata (``lat``/``lon``, ``lake_id``,
    ``lake_name``, constant assignment fields). Per-lake mode keeps lake centroid
    and haversine distance; drops pixel-only duplicates.
    """
    if summary_df.empty:
        return summary_df.copy()

    df = summary_df.copy()

    drop_common = {"record_type", "grid_assignment"}
    float_cols = {
        "grace_lat", "grace_lon", "lat", "lon",
        "haversine_distance_deg", "grace_window_area_km2", "area_km2",
        "completeness_pct", "lake_std_cm", "grace_std_cm",
        "lake_trend_cm_yr", "grace_trend_cm_yr",
        "lake_std_cm_residual", "grace_std_cm_residual",
        "lake_std_pct_of_grace_residual",
    }
    int_cols = {"window_deg", "n_lakes", "n_grace_pixels", "n_overlap_months", "lake_id"}

    if by_grace_pixel:
        drop = drop_common | {
            "lat", "lon", "lake_id", "lake_name",
            "haversine_distance_deg", "n_grace_pixels",
        }
        col_order = [
            "grace_lat", "grace_lon", "n_lakes", "lake_ids", "lake_names",
            "country", "area_km2", "completeness_pct", "window_deg",
            "grace_window_area_km2",
        "lake_std_cm", "grace_std_cm",
        "lake_std_pct_of_grace",
        "lake_std_cm_residual", "grace_std_cm_residual",
        "lake_std_pct_of_grace_residual",
        "lake_trend_cm_yr", "grace_trend_cm_yr",
            "n_overlap_months",
        ]
    else:
        drop = drop_common | {"lake_ids", "lake_names", "n_lakes", "n_grace_pixels"}
        col_order = [
            "lake_id", "lake_name", "country", "lat", "lon",
            "grace_lat", "grace_lon", "haversine_distance_deg",
            "area_km2", "completeness_pct", "window_deg", "grace_window_area_km2",
        "lake_std_cm", "grace_std_cm",
        "lake_std_pct_of_grace",
        "lake_std_cm_residual", "grace_std_cm_residual",
        "lake_std_pct_of_grace_residual",
        "lake_trend_cm_yr", "grace_trend_cm_yr",
            "n_overlap_months",
        ]

    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(2)
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(0).astype("Int64")

    ordered = [c for c in col_order if c in df.columns]
    rest = [c for c in df.columns if c not in ordered]
    return df[ordered + rest]


def compute_grace_pixel_summaries(
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    parallel: bool = True,
    calculate_residual: bool = False,
) -> pd.DataFrame:
    """One summary row per GRACE pixel: summed lake ΔV (km³) vs GRACE TWSA."""
    cfg = cfg or load_sws_config()
    groups = group_lake_ids_by_grace_pixel(catalog, volume_series, grace_da)
    items = sorted(groups.items())
    n_lakes = sum(len(ids) for _, ids in items)
    _note(f"GRACE pixel summaries ({len(items)} pixels, {n_lakes} lakes)")

    def _one(item: Tuple[Tuple[float, float], List[int]]) -> Optional[Dict[str, Any]]:
        (grace_lat, grace_lon), lake_ids = item
        try:
            return summarize_grace_pixel_row(
                grace_lat,
                grace_lon,
                lake_ids,
                catalog,
                volume_series,
                grace_da,
                cfg=cfg,
                window_deg=window_deg,
                calculate_residual=calculate_residual,
            )
        except Exception as exc:
            logger.warning(
                "Pixel summary failed for (%s, %s): %s", grace_lat, grace_lon, exc,
            )
            return None

    n_jobs = cfg.n_process_workers if parallel else 1
    if len(items) > 1:
        results = _parallel_thread_map(
            _one, items, n_jobs=n_jobs, desc="GRACE pixels", unit="pixel",
        )
    else:
        results = [_one(item) for item in items]
    records = [r for r in results if r is not None]
    if not records:
        return pd.DataFrame()
    summary = format_grace_summary_export(pd.DataFrame(records), by_grace_pixel=True)
    out_path = cfg.processed_dir / f"grace_pixel_summary_win{int(window_deg)}.csv"
    summary.to_csv(out_path, index=False)
    logger.info(
        "GRACE pixel summary (%s pixels, %s lakes grouped) → %s",
        len(summary),
        int(summary["n_lakes"].sum()),
        out_path,
    )
    return summary


def compute_lake_grace_summaries(
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    parallel: bool = True,
    aggregate_by_grace_pixel: bool = False,
    calculate_residual: bool = False,
) -> pd.DataFrame:
    """Build lake vs GRACE summary statistics (per lake or per GRACE pixel)."""
    cfg = cfg or load_sws_config()
    if aggregate_by_grace_pixel:
        return compute_grace_pixel_summaries(
            catalog,
            volume_series,
            grace_da,
            cfg=cfg,
            window_deg=window_deg,
            parallel=parallel,
            calculate_residual=calculate_residual,
        )

    def _one(row):
        lake_id = row.get("lake_id", row.name)
        if lake_id not in volume_series:
            return None
        meta = row.to_dict()
        try:
            return summarize_lake_grace_row(
                meta, volume_series[lake_id], grace_da, cfg=cfg, window_deg=window_deg,
                calculate_residual=calculate_residual,
            )
        except Exception as exc:
            logger.warning("Summary failed for lake %s: %s", lake_id, exc)
            return None

    rows = [row for _, row in catalog.iterrows() if row.get("lake_id", row.name) in volume_series]
    _note(f"lake/GRACE summaries ({len(rows)} lakes)")
    n_jobs = cfg.n_process_workers if parallel else 1
    if len(rows) > 1:
        results = _parallel_thread_map(
            _one, rows, n_jobs=n_jobs, desc="Lake/GRACE", unit="lake",
        )
    else:
        results = [_one(r) for r in rows]
    records = [r for r in results if r is not None]
    if not records:
        return pd.DataFrame()
    summary = format_grace_summary_export(pd.DataFrame(records), by_grace_pixel=False)
    out_path = cfg.processed_dir / f"lake_grace_summary_win{window_deg}.csv"
    summary.to_csv(out_path, index=False)
    logger.info("Lake/GRACE summary table (%s lakes) → %s", len(summary), out_path)
    return summary


_ARIDITY_DOMAIN_COLORS = [
    "#0077BB", "#33BBEE", "#009988", "#EE7733", "#CC3311",
    "#EE3377", "#BBBBBB", "#000000", "#44AA99",
]


def _is_pixel_grace_summary(summary_df: pd.DataFrame) -> bool:
    if "n_lakes" in summary_df.columns and (summary_df["n_lakes"] > 1).any():
        return True
    if "lake_ids" in summary_df.columns:
        return summary_df["lake_ids"].astype(str).str.contains(",", regex=False).any()
    return False


def _normalize_summary_mode(summary_mode: str) -> str:
    mode = str(summary_mode).strip().lower()
    if mode not in ("per_lake", "grace_pixel"):
        raise ValueError("summary_mode must be 'per_lake' or 'grace_pixel'")
    return mode


def _has_grace_pixel_coords(summary_df: pd.DataFrame) -> bool:
    lat_ok = "grace_lat" in summary_df.columns or "lat" in summary_df.columns
    lon_ok = "grace_lon" in summary_df.columns or "lon" in summary_df.columns
    return lat_ok and lon_ok


def _pixel_row_lake_ids(row: pd.Series) -> List[int]:
    if "lake_id" in row.index and pd.notna(row.get("lake_id")):
        try:
            return [int(row["lake_id"])]
        except (TypeError, ValueError):
            pass
    if "lake_ids" in row.index and pd.notna(row.get("lake_ids")):
        return [int(x.strip()) for x in str(row["lake_ids"]).split(",") if x.strip().isdigit()]
    return []


def ensure_per_lake_grace_summary(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    parallel: bool = True,
    calculate_residual: bool = False,
) -> pd.DataFrame:
    """Return per-lake summary rows (recompute if *summary_df* is pixel-aggregated)."""
    if (
        not _is_pixel_grace_summary(summary_df)
        and "lake_std_pct_of_grace" in summary_df.columns
        and "lake_id" in summary_df.columns
        and (not calculate_residual or "lake_std_pct_of_grace_residual" in summary_df.columns)
    ):
        return summary_df.copy()
    logger.info("Building per-lake summary for lake-level maps …")
    return compute_lake_grace_summaries(
        catalog,
        volume_series,
        grace_da,
        cfg=cfg,
        window_deg=window_deg,
        parallel=parallel,
        aggregate_by_grace_pixel=False,
        calculate_residual=calculate_residual,
    )


def _resolve_std_ratio_summary(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    volume_series: Optional[Dict[Any, pd.Series]],
    grace_da: Optional[xr.DataArray],
    *,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    summary_mode: str = "per_lake",
    parallel: bool = True,
    calculate_residual: bool = False,
) -> pd.DataFrame:
    """Return summary rows for σ-ratio plots (per-lake or GRACE-pixel, no cross-mode recompute)."""
    mode = _normalize_summary_mode(summary_mode)
    if mode == "grace_pixel":
        if not _has_grace_pixel_coords(summary_df):
            raise ValueError(
                "summary_mode='grace_pixel' requires grace_lat/grace_lon (or lat/lon) in lake_summary"
            )
        return summary_df.copy()
    if volume_series is not None and grace_da is not None:
        return ensure_per_lake_grace_summary(
            summary_df, catalog, volume_series, grace_da,
            cfg=cfg, window_deg=window_deg, parallel=parallel,
            calculate_residual=calculate_residual,
        )
    return summary_df.copy()


def _summary_std_ratio_values(
    summary_df: pd.DataFrame,
    value_col: str = "lake_std_pct_of_grace",
) -> pd.Series:
    if value_col in summary_df.columns:
        return pd.to_numeric(summary_df[value_col], errors="coerce")
    # lake_std_cm/grace_std_cm reconstruct the *raw anomaly* ratio only — never a
    # residual ratio — so restrict this fallback to the anomaly column.
    if value_col == "lake_std_pct_of_grace" and {"lake_std_cm", "grace_std_cm"}.issubset(summary_df.columns):
        lake = pd.to_numeric(summary_df["lake_std_cm"], errors="coerce")
        grace = pd.to_numeric(summary_df["grace_std_cm"], errors="coerce")
        return 100.0 * lake / grace.replace(0, np.nan)
    if str(value_col).endswith("_residual"):
        raise KeyError(
            f"Column {value_col!r} not found in summary_df. For residual-based "
            "filtering, run analyze_lake_grace_comparisons(..., calculate_residual=True) first."
        )
    raise KeyError(f"Need {value_col!r} or lake_std_cm/grace_std_cm in summary_df")


def _summary_color_norm(
    values: np.ndarray,
    vmin: Optional[float] = 0.0,
    vmax: Optional[float] = None,
    pct_cap: float = 95.0,
) -> Tuple[mcolors.Normalize, float, float]:
    vals = np.asarray(values, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vmax is None:
        vmax = float(np.nanpercentile(vals, pct_cap)) if len(vals) else 1.0
    if vmin is None:
        vmin = 0.0
    if vmax <= vmin:
        vmax = vmin + 1.0
    return mcolors.Normalize(vmin=vmin, vmax=vmax), float(vmin), float(vmax)


def _lake_summary_geodataframe(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    cfg: Optional[SWSConfig] = None,
) -> gpd.GeoDataFrame:
    """Merge per-lake summary with HydroLAKES polygon boundaries and centroid coords."""
    cfg = cfg or load_sws_config()
    df = summary_df.copy()
    if "lake_id" not in df.columns:
        raise ValueError("Per-lake summary required (column 'lake_id' missing)")

    lake_ids = df["lake_id"].astype(int).tolist()
    poly_gdf = load_hydrolakes_polygons(cfg, lake_ids=lake_ids)
    if poly_gdf.empty:
        raise ValueError(
            "No HydroLAKES polygons found for requested lake_id(s). "
            f"Run download_hydrolakes() or build_glolakes_arid_catalog() first. "
            f"Missing IDs (sample): {lake_ids[:5]}"
        )

    cat = catalog.copy()
    if isinstance(cat, gpd.GeoDataFrame):
        cat = pd.DataFrame(cat.drop(columns="geometry", errors="ignore"))
    attr_cols = [
        c for c in ("lake_id", "lake_name", "country", "lat", "lon", "lat_centroid", "lon_centroid", "area_km2")
        if c in cat.columns and c != "lake_id"
    ]
    if attr_cols:
        df = df.merge(
            cat[["lake_id"] + attr_cols].drop_duplicates("lake_id"),
            on="lake_id",
            how="left",
            suffixes=("", "_cat"),
        )

    hydro_cols = [c for c in ("lake_id", "geometry", "lake_name", "area_km2") if c in poly_gdf.columns]
    merged = df.merge(poly_gdf[hydro_cols], on="lake_id", how="inner", suffixes=("", "_hydro"))
    if merged.empty:
        raise ValueError("No lakes matched between summary and HydroLAKES polygons")

    gdf = gpd.GeoDataFrame(merged, geometry="geometry", crs="EPSG:4326")
    cent_lon, cent_lat = _projected_centroid_coords(gdf)
    gdf["plot_lat"] = cent_lat.values
    gdf["plot_lon"] = cent_lon.values
    return gdf


def _setup_arid_map_axes(
    ax,
    extent: Sequence[float],
    *,
    domains_gdf: Optional[gpd.GeoDataFrame] = None,
    domain_col: str = "Domain",
    draw_domain_boundaries: bool = False,
):
    """Cartopy basemap with optional per-domain outlines."""
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    ax.add_feature(cfeature.OCEAN, facecolor="lightblue")
    ax.add_feature(cfeature.LAND, facecolor="lightgray")
    ax.coastlines()
    ax.add_feature(cfeature.BORDERS, linewidth=0.5)

    gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", alpha=0.5, linestyle="--")
    gl.top_labels = False
    gl.right_labels = False
    gl.xlabel_style = {"rotation": 0, "size": _MAP_GRID_LABEL_FONTSIZE}
    gl.ylabel_style = {"rotation": 90, "size": _MAP_GRID_LABEL_FONTSIZE}

    if draw_domain_boundaries and domains_gdf is not None and not domains_gdf.empty:
        dom = domains_gdf.to_crs("EPSG:4326")
        labels = dom[domain_col].astype(str).tolist() if domain_col in dom.columns else [f"D{i}" for i in range(len(dom))]
        unique = sorted(set(labels))
        color_map = {lab: _ARIDITY_DOMAIN_COLORS[i % len(_ARIDITY_DOMAIN_COLORS)] for i, lab in enumerate(unique)}
        for i, (_, row) in enumerate(dom.iterrows()):
            lab = str(row[domain_col]) if domain_col in dom.columns else f"D{i}"
            gpd.GeoSeries([row.geometry], crs=dom.crs).plot(
                ax=ax,
                facecolor="none",
                edgecolor=color_map[lab],
                linewidth=1.0,
                transform=ccrs.PlateCarree(),
                zorder=3,
            )


def _prepare_lake_std_ratio_plot_gdf(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    cfg: Optional[SWSConfig] = None,
    volume_series: Optional[Dict[Any, pd.Series]] = None,
    grace_da: Optional[xr.DataArray] = None,
    window_deg: float = 1.0,
    value_col: str = "lake_std_pct_of_grace",
    min_pct: Optional[float] = 10.0,
    summary_mode: str = "per_lake",
    calculate_residual: bool = False,
) -> gpd.GeoDataFrame:
    """Build GeoDataFrame for σ-ratio maps (per-lake polygons or GRACE-pixel points)."""
    cfg = cfg or load_sws_config()
    mode = _normalize_summary_mode(summary_mode)
    if mode == "grace_pixel":
        return _prepare_grace_pixel_std_ratio_plot_gdf(
            summary_df, value_col=value_col, min_pct=min_pct,
        )

    if volume_series is not None and grace_da is not None:
        summary_df = _resolve_std_ratio_summary(
            summary_df, catalog, volume_series, grace_da,
            cfg=cfg, window_deg=window_deg, summary_mode="per_lake",
            calculate_residual=calculate_residual,
        )

    gdf = _lake_summary_geodataframe(summary_df, catalog, cfg=cfg)
    gdf["_ratio"] = _summary_std_ratio_values(gdf, value_col).values
    plot_gdf = gdf.dropna(subset=["_ratio", "plot_lat", "plot_lon"]).copy()
    if min_pct is not None and float(min_pct) > 0:
        plot_gdf = plot_gdf[plot_gdf["_ratio"] > float(min_pct)].copy()
    return plot_gdf


def _prepare_grace_pixel_std_ratio_plot_gdf(
    summary_df: pd.DataFrame,
    *,
    value_col: str = "lake_std_pct_of_grace",
    min_pct: Optional[float] = 10.0,
) -> gpd.GeoDataFrame:
    """One map point per GRACE pixel using combined-lake σ ratio at grace_lat/grace_lon."""
    df = summary_df.copy()
    lat_col = "grace_lat" if "grace_lat" in df.columns else "lat"
    lon_col = "grace_lon" if "grace_lon" in df.columns else "lon"
    df["plot_lat"] = pd.to_numeric(df[lat_col], errors="coerce")
    df["plot_lon"] = pd.to_numeric(df[lon_col], errors="coerce")
    df["_ratio"] = _summary_std_ratio_values(df, value_col).values
    plot_df = df.dropna(subset=["plot_lat", "plot_lon", "_ratio"]).copy()
    if min_pct is not None and float(min_pct) > 0:
        plot_df = plot_df[plot_df["_ratio"] > float(min_pct)].copy()
    if plot_df.empty:
        return gpd.GeoDataFrame(
            columns=list(plot_df.columns) + ["geometry"],
            geometry=[],
            crs="EPSG:4326",
        )
    return gpd.GeoDataFrame(
        plot_df,
        geometry=gpd.points_from_xy(plot_df["plot_lon"], plot_df["plot_lat"]),
        crs="EPSG:4326",
    )


def _lake_display_name(row: pd.Series) -> str:
    if "lake_name" in row.index and pd.notna(row["lake_name"]) and str(row["lake_name"]).strip():
        return _display_lake_name(row["lake_name"])
    return f"lake_id={int(row['lake_id'])}"


def _format_plotted_lake_labels(plot_gdf: gpd.GeoDataFrame) -> List[str]:
    labels: List[str] = []
    for _, row in plot_gdf.sort_values("_ratio", ascending=False).iterrows():
        labels.append(f"{_lake_display_name(row)} ({float(row['_ratio']):.1f}%)")
    return labels


def _grace_pixel_display_name(row: pd.Series) -> str:
    n_lakes = int(row["n_lakes"]) if "n_lakes" in row.index and pd.notna(row.get("n_lakes")) else 1
    ids = _pixel_row_lake_ids(row)
    if n_lakes <= 1:
        if "lake_name" in row.index and pd.notna(row.get("lake_name")):
            name = _display_lake_name(row["lake_name"])
            if name != "No Name":
                return name
        return f"lake_id={ids[0]}" if ids else "GRACE pixel"
    if "lake_names" in row.index and pd.notna(row.get("lake_names")):
        raw_names = [n.strip() for n in str(row["lake_names"]).split(";")]
        display_names = [_display_lake_name(n) for n in raw_names if n]
        if any(n != "No Name" for n in display_names):
            joined = "; ".join(display_names[:8])
            return joined + ("; …" if len(display_names) > 8 else "")
    if ids:
        id_str = ", ".join(str(i) for i in ids[:8])
        return f"lake_ids={id_str}" + (", …" if len(ids) > 8 else "")
    return f"{n_lakes} lakes @ ({float(row['plot_lat']):.1f}°, {float(row['plot_lon']):.1f}°)"


def _format_grace_pixel_labels(plot_gdf: gpd.GeoDataFrame) -> List[str]:
    labels: List[str] = []
    for _, row in plot_gdf.sort_values("_ratio", ascending=False).iterrows():
        labels.append(f"{_grace_pixel_display_name(row)} ({float(row['_ratio']):.1f}%)")
    return labels


def _format_std_ratio_labels(plot_gdf: gpd.GeoDataFrame, summary_mode: str) -> List[str]:
    if _normalize_summary_mode(summary_mode) == "grace_pixel":
        return _format_grace_pixel_labels(plot_gdf)
    return _format_plotted_lake_labels(plot_gdf)


def _padded_map_extent(
    minx: float,
    miny: float,
    maxx: float,
    maxy: float,
    *,
    pad_frac: float = 0.14,
    min_pad: float = 1.2,
    min_span: float = 2.0,
) -> List[float]:
    """Build a PlateCarree extent with padding and a minimum span (degrees)."""
    cx = 0.5 * (minx + maxx)
    cy = 0.5 * (miny + maxy)
    lon_span = max(maxx - minx, min_span)
    lat_span = max(maxy - miny, min_span)
    pad_lon = max(min_pad, lon_span * pad_frac)
    pad_lat = max(min_pad, lat_span * pad_frac)
    half_lon = 0.5 * lon_span + pad_lon
    half_lat = 0.5 * lat_span + pad_lat
    return [cx - half_lon, cx + half_lon, cy - half_lat, cy + half_lat]


def _extent_from_grace_da(
    grace_da: xr.DataArray,
    *,
    pad_deg: float = 1.0,
) -> List[float]:
    """Map extent from GRACE grid bounds (matches ``plot_grace_correlation_map`` + AOI padding)."""
    if "lon" not in grace_da.coords or "lat" not in grace_da.coords:
        raise ValueError(
            f"grace_da must have lon/lat coordinates; got {list(grace_da.coords)}"
        )
    lon_coords = np.asarray(grace_da.lon.values, dtype=float)
    lat_coords = np.asarray(grace_da.lat.values, dtype=float)
    if lon_coords.size == 0 or lat_coords.size == 0:
        raise ValueError("grace_da has empty lon/lat coordinates")

    lon_res = float(np.diff(lon_coords).mean()) if lon_coords.size > 1 else 1.0
    lat_res = float(np.diff(lat_coords).mean()) if lat_coords.size > 1 else 1.0
    minx = float(np.min(lon_coords)) - lon_res / 2.0 - pad_deg
    maxx = float(np.max(lon_coords)) + lon_res / 2.0 + pad_deg
    miny = float(np.min(lat_coords)) - lat_res / 2.0 - pad_deg
    maxy = float(np.max(lat_coords)) + lat_res / 2.0 + pad_deg
    return [minx, maxx, miny, maxy]


def _lake_std_ratio_map_title(
    *,
    summary_mode: str,
    min_pct: Optional[float],
    window_deg: float,
) -> str:
    """Dynamic map title: filter threshold and GRACE extraction window."""
    win_label = (
        f"{float(window_deg):g}° GRACE/precip window"
        if float(window_deg) != 1.0
        else "1° GRACE/precip window"
    )
    if summary_mode == "grace_pixel":
        subject = "Combined lake σ at GRACE pixels"
    else:
        subject = "Lake storage variability"
    if min_pct is not None and float(min_pct) > 0:
        return f"{subject} > {float(min_pct):g}% of GRACE σ ({win_label})"
    return f"{subject} (% of GRACE σ; {win_label})"


def _lake_std_ratio_domain_title(
    domain_name: str,
    *,
    summary_mode: str,
    min_pct: Optional[float],
    window_deg: float,
) -> str:
    """Compact per-domain map title."""
    win = f"{float(window_deg):g}°"
    mode = _normalize_summary_mode(summary_mode)
    subject = "Combined lake σ" if mode == "grace_pixel" else "Lake σ"
    if min_pct is not None and float(min_pct) > 0:
        return f"{domain_name} - {subject} > {float(min_pct):g}% of GRACE ({win})"
    return f"{domain_name} - {subject} (% of GRACE; {win})"


def _resolve_std_ratio_vmin(
    vmin: Optional[float],
    min_pct: Optional[float],
) -> float:
    """Color scale floor: start at display threshold when lakes are filtered."""
    if vmin is not None:
        return float(vmin)
    if min_pct is not None and float(min_pct) > 0:
        return float(min_pct)
    return 0.0


def _filtered_std_ratio_summary_table(
    plot_gdf: gpd.GeoDataFrame,
    *,
    summary_mode: str,
    value_col: str = "lake_std_pct_of_grace",
) -> pd.DataFrame:
    """Tabular export of map features, sorted by σ % of GRACE (highest first)."""
    mode = _normalize_summary_mode(summary_mode)
    df = pd.DataFrame(plot_gdf.drop(columns="geometry", errors="ignore"))
    sort_col = value_col if value_col in df.columns else "_ratio"
    if sort_col not in df.columns:
        raise KeyError(f"Cannot sort filtered table: missing {value_col!r} and '_ratio'")
    df = df.sort_values(sort_col, ascending=False, na_position="last")
    df = df.drop(columns=[c for c in ("_ratio", "plot_lat", "plot_lon") if c in df.columns])
    return format_grace_summary_export(df, by_grace_pixel=(mode == "grace_pixel"))


def _extent_from_lake_ids(
    lake_ids: Sequence[int],
    cfg: SWSConfig,
    *,
    pad_frac: float = 0.18,
    min_pad: float = 0.8,
    min_span: float = 1.5,
) -> List[float]:
    """Map extent from HydroLAKES polygon bounds for the given lakes."""
    hydro = load_hydrolakes_polygons(cfg, lake_ids=[int(x) for x in lake_ids])
    if hydro.empty:
        raise ValueError(f"No HydroLAKES polygons for extent (lake_ids={list(lake_ids)[:5]})")
    minx, miny, maxx, maxy = hydro.total_bounds
    return _padded_map_extent(minx, miny, maxx, maxy, pad_frac=pad_frac, min_pad=min_pad, min_span=min_span)


def _extent_from_plot_coords(
    plot_gdf: gpd.GeoDataFrame,
    *,
    pad_frac: float = 0.18,
    min_pad: float = 0.8,
    min_span: float = 1.5,
) -> List[float]:
    """Map extent from plot_lon/plot_lat (lake centroids or GRACE pixel centres)."""
    lons = plot_gdf["plot_lon"].astype(float)
    lats = plot_gdf["plot_lat"].astype(float)
    return _padded_map_extent(
        float(lons.min()), float(lats.min()), float(lons.max()), float(lats.max()),
        pad_frac=pad_frac, min_pad=min_pad, min_span=min_span,
    )


def _balanced_domain_extent_and_figsize(
    extent: Sequence[float],
    *,
    max_aspect: float = 3.0,
    base_height: float = 4.0,
    min_width: float = 4.0,
    max_width: float = 14.0,
    min_height: float = 3.0,
    max_height: float = 12.0,
    cbar_scale: float = 1.10,
) -> Tuple[List[float], Tuple[float, float]]:
    """
    Balance map extent so display aspect (width/height) is within [1/max_aspect, max_aspect],
    then return extent and a matching matplotlib figsize.
    """
    lon_min, lon_max, lat_min, lat_max = map(float, extent)
    lon_span = max(lon_max - lon_min, 0.25)
    lat_span = max(lat_max - lat_min, 0.25)
    cx = 0.5 * (lon_min + lon_max)
    cy = 0.5 * (lat_min + lat_max)
    cos_lat = max(np.cos(np.deg2rad(cy)), 0.20)

    display_aspect = (lon_span * cos_lat) / lat_span
    min_aspect = 1.0 / float(max_aspect)

    if display_aspect > max_aspect:
        lat_span = (lon_span * cos_lat) / max_aspect
    elif display_aspect < min_aspect:
        lon_span = (lat_span * min_aspect) / cos_lat

    balanced = [cx - 0.5 * lon_span, cx + 0.5 * lon_span, cy - 0.5 * lat_span, cy + 0.5 * lat_span]
    display_aspect = (lon_span * cos_lat) / lat_span
    display_aspect = float(np.clip(display_aspect, min_aspect, max_aspect))

    height = float(np.clip(base_height, min_height, max_height))
    width = float(np.clip(height * display_aspect * cbar_scale, min_width, max_width))
    if width >= max_width - 1e-9:
        width = max_width
        height = float(np.clip(width / (display_aspect * cbar_scale), min_height, max_height))
    elif width <= min_width + 1e-9:
        width = min_width
        height = float(np.clip(width / (display_aspect * cbar_scale), min_height, max_height))
    return balanced, (round(width, 1), round(height, 1))


def _assign_lakes_to_domains(
    lake_gdf: gpd.GeoDataFrame,
    domains_gdf: gpd.GeoDataFrame,
    domain_col: str = "Domain",
) -> Dict[str, gpd.GeoDataFrame]:
    """
    Assign lakes to arid domains by polygon intersection (not centroid-in-polygon).

    When a lake intersects multiple domains, keep the domain with the largest
    overlap area (equal-area projection).
    """
    if lake_gdf.empty:
        return {}

    lakes = lake_gdf.to_crs("EPSG:4326").copy()
    dom = domains_gdf.to_crs("EPSG:4326")[[domain_col, "geometry"]].copy()
    joined = gpd.sjoin(lakes, dom, how="inner", predicate="intersects")
    if joined.empty:
        return {}

    if joined["lake_id"].duplicated().any():
        lakes_ea = lakes.to_crs("EPSG:6933")
        dom_ea = dom.to_crs("EPSG:6933")
        dom_geom_map = dom_ea.set_index(domain_col)["geometry"]
        keep_rows = []
        for lake_id, grp in joined.groupby("lake_id"):
            if len(grp) == 1:
                keep_rows.append(grp.iloc[0])
                continue
            lake_geom = lakes_ea.loc[lakes_ea["lake_id"] == lake_id, "geometry"].iloc[0]
            best_row = grp.iloc[0]
            best_area = -1.0
            for _, row in grp.iterrows():
                dom_name = row[domain_col]
                dom_geom = dom_geom_map.get(dom_name)
                if dom_geom is None:
                    continue
                try:
                    overlap = lake_geom.intersection(dom_geom).area
                except Exception:
                    overlap = 0.0
                if overlap > best_area:
                    best_area = overlap
                    best_row = row
            keep_rows.append(best_row)
        joined = gpd.GeoDataFrame(keep_rows, crs=joined.crs)

    out: Dict[str, gpd.GeoDataFrame] = {}
    for domain_name, grp in joined.groupby(joined[domain_col].astype(str)):
        out[str(domain_name)] = grp.copy()
    return out


def _assign_grace_pixels_to_domains(
    pixel_gdf: gpd.GeoDataFrame,
    domains_gdf: gpd.GeoDataFrame,
    domain_col: str = "Domain",
) -> Dict[str, gpd.GeoDataFrame]:
    """Assign GRACE-pixel points to domains by grace_lat/grace_lon location."""
    if pixel_gdf.empty:
        return {}

    pts = pixel_gdf.to_crs("EPSG:4326").copy()
    if "geometry" not in pts.columns or pts.geometry.isna().all():
        pts["geometry"] = gpd.points_from_xy(pts["plot_lon"], pts["plot_lat"])
    dom = domains_gdf.to_crs("EPSG:4326")[[domain_col, "geometry"]].copy()
    joined = gpd.sjoin(pts, dom, how="inner", predicate="within")
    if joined.empty:
        joined = gpd.sjoin(pts, dom, how="inner", predicate="intersects")
    if joined.empty:
        return {}

    out: Dict[str, gpd.GeoDataFrame] = {}
    for domain_name, grp in joined.groupby(joined[domain_col].astype(str)):
        out[str(domain_name)] = grp.copy()
    return out


def _finalize_std_ratio_map_figure(fig: plt.Figure) -> None:
    """Layout for Cartopy map + colorbar (matches GRACE correlation maps)."""
    fig.tight_layout(pad=_MAP_TIGHT_LAYOUT_PAD)


def _draw_lake_std_ratio_layers(
    ax,
    plot_gdf: gpd.GeoDataFrame,
    *,
    cfg: SWSConfig,
    norm: mcolors.Normalize,
    cmap_obj,
    point_size: float,
    polygon_edgecolor: str = "black",
    polygon_linewidth: float = 0.65,
    draw_lake_polygons: bool = True,
) -> Any:
    """HydroLAKES polygon outlines (black) plus coloured centroid markers."""
    import cartopy.crs as ccrs

    lakes = plot_gdf.dropna(subset=["plot_lat", "plot_lon", "_ratio"]).copy()
    if lakes.empty:
        raise ValueError("No lakes to draw")

    lake_ids = lakes["lake_id"].astype(int).tolist()
    if draw_lake_polygons:
        hydro_polys = load_hydrolakes_polygons(cfg, lake_ids=lake_ids)
        missing = set(lake_ids) - set(hydro_polys["lake_id"].astype(int).tolist())
        if missing:
            logger.warning(
                "HydroLAKES polygon missing for %s lake(s): %s",
                len(missing), sorted(missing)[:8],
            )
        if not hydro_polys.empty:
            hydro_polys = hydro_polys[hydro_polys.geometry.notna() & ~hydro_polys.geometry.is_empty]
            hydro_polys.plot(
                ax=ax,
                facecolor="none",
                edgecolor=polygon_edgecolor,
                linewidth=polygon_linewidth,
                linestyle="solid",
                transform=ccrs.PlateCarree(),
                zorder=4,
            )

    sc = ax.scatter(
        lakes["plot_lon"].values,
        lakes["plot_lat"].values,
        c=lakes["_ratio"].values,
        cmap=cmap_obj,
        norm=norm,
        s=point_size,
        edgecolors="0.15",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    return sc


def _draw_grace_pixel_std_ratio_layers(
    ax,
    plot_gdf: gpd.GeoDataFrame,
    *,
    cfg: SWSConfig,
    norm: mcolors.Normalize,
    cmap_obj,
    point_size: float,
    polygon_edgecolor: str = "black",
    polygon_linewidth: float = 0.65,
    draw_lake_polygons: bool = True,
) -> Any:
    """GRACE-pixel centre markers; HydroLAKES outline when a pixel has exactly one lake."""
    import cartopy.crs as ccrs

    pixels = plot_gdf.dropna(subset=["plot_lat", "plot_lon", "_ratio"]).copy()
    if pixels.empty:
        raise ValueError("No GRACE pixels to draw")

    if draw_lake_polygons:
        single_lake_ids: List[int] = []
        for _, row in pixels.iterrows():
            n_lakes = int(row["n_lakes"]) if "n_lakes" in row.index and pd.notna(row.get("n_lakes")) else 1
            if n_lakes == 1:
                single_lake_ids.extend(_pixel_row_lake_ids(row))
        if single_lake_ids:
            hydro_polys = load_hydrolakes_polygons(cfg, lake_ids=sorted(set(single_lake_ids)))
            if not hydro_polys.empty:
                hydro_polys = hydro_polys[hydro_polys.geometry.notna() & ~hydro_polys.geometry.is_empty]
                hydro_polys.plot(
                    ax=ax,
                    facecolor="none",
                    edgecolor=polygon_edgecolor,
                    linewidth=polygon_linewidth,
                    linestyle="solid",
                    transform=ccrs.PlateCarree(),
                    zorder=4,
                )

    sc = ax.scatter(
        pixels["plot_lon"].values,
        pixels["plot_lat"].values,
        c=pixels["_ratio"].values,
        cmap=cmap_obj,
        norm=norm,
        s=point_size,
        edgecolors="0.15",
        linewidths=0.35,
        transform=ccrs.PlateCarree(),
        zorder=6,
    )
    return sc


def _add_lake_std_ratio_colorbar(
    fig: plt.Figure,
    ax,
    mappable,
    *,
    label: str = "Lake σ (% of GRACE σ)",
) -> Any:
    """Vertical colorbar matched to ``plot_grace_correlation_map`` layout."""
    cbar = fig.colorbar(
        mappable,
        ax=ax,
        orientation="vertical",
        pad=_MAP_CBAR_PAD,
        fraction=_MAP_CBAR_FRACTION,
        extend="max",
        shrink=0.88,
    )
    cbar.set_label(label, fontsize=11)
    cbar.ax.tick_params(labelsize=9)
    return cbar


_FILTER_MODE_VALUE_COL = {
    "anomaly": "lake_std_pct_of_grace",
    "residual": "lake_std_pct_of_grace_residual",
}


def _resolve_filter_mode_value_col(
    filter_mode: Optional[str],
    value_col: str,
) -> str:
    """Map ``filter_mode`` ('anomaly'/'residual') to the σ-ratio column it selects.

    ``filter_mode`` takes precedence over ``value_col`` when set; ``None`` keeps the
    caller-supplied ``value_col`` for backward compatibility.
    """
    if filter_mode is None:
        return value_col
    key = str(filter_mode).strip().lower()
    if key not in _FILTER_MODE_VALUE_COL:
        raise ValueError(
            f"filter_mode must be one of {sorted(_FILTER_MODE_VALUE_COL)!r} "
            f"('anomaly' → lake_std_pct_of_grace, 'residual' → "
            f"lake_std_pct_of_grace_residual); got {filter_mode!r}"
        )
    return _FILTER_MODE_VALUE_COL[key]


def plot_lake_std_ratio_map(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    volume_series: Optional[Dict[Any, pd.Series]] = None,
    grace_da: Optional[xr.DataArray] = None,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    value_col: str = "lake_std_pct_of_grace",
    filter_mode: Optional[str] = None,
    cmap: str = "YlOrRd",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    figsize: Tuple[float, float] = (10, 4),
    dpi: int = 300,
    save_path: Optional[Union[str, Path]] = None,
    point_size: float = 36.0,
    polygon_edgecolor: str = "black",
    polygon_linewidth: float = 0.45,
    domains_gdf: Optional[gpd.GeoDataFrame] = None,
    domain_col: str = "Domain",
    draw_domain_boundaries: bool = False,
    extent: Optional[Sequence[float]] = None,
    extent_pad_deg: float = 1.0,
    title: Optional[str] = None,
    min_pct: Optional[float] = 10.0,
    plot_gdf: Optional[gpd.GeoDataFrame] = None,
    draw_lake_polygons: bool = True,
    summary_mode: str = "per_lake",
    show: bool = False,
    return_table: bool = False,
) -> Union[plt.Figure, pd.DataFrame]:
    """
    Map lake σ as % of GRACE pixel σ.

    ``summary_mode='per_lake'`` (default): HydroLAKES polygons + lake-centroid markers.
    Recomputes per-lake stats when *lake_summary* is pixel-aggregated.

    ``summary_mode='grace_pixel'``: one marker per GRACE pixel at ``grace_lat/grace_lon``
    using the combined-lake σ ratio already in *lake_summary* (no per-lake recompute).
    Draws a lake outline only when the pixel contains exactly one lake.

    Parameters
    ----------
    value_col : str, default 'lake_std_pct_of_grace'
        Column used for filtering, coloring and table sorting. Overridden by
        ``filter_mode`` when that is set.
    filter_mode : {'anomaly', 'residual'} or None, default None
        Convenience selector for ``value_col``: ``'anomaly'`` uses
        ``lake_std_pct_of_grace`` (σ of the raw monthly anomaly), ``'residual'`` uses
        ``lake_std_pct_of_grace_residual`` (σ of the harmonic residual — requires the
        summary to have been built with ``calculate_residual=True``). ``None`` keeps
        the explicit ``value_col``.
    min_pct : float or None, default 10.0
        Show only features with the selected column strictly greater than this
        percentage. When set and ``vmin`` is None, the color scale starts at this
        threshold.
    extent_pad_deg : float, default 1.0
        Padding (degrees) when deriving extent from ``grace_da`` (same as
        ``plot_grace_correlation_map`` with AOI).
    draw_lake_polygons : bool, default True
        Draw HydroLAKES outlines (all lakes in per_lake mode; single-lake pixels only
        in grace_pixel mode).
    summary_mode : {'per_lake', 'grace_pixel'}, default 'per_lake'
    return_table : bool, default False
        If True, still draw the map but return a :class:`pandas.DataFrame` of the
        filtered features (same columns as ``lake_summary``), sorted by the selected
        column descending, and print the row count.
    """
    try:
        import cartopy.crs as ccrs
    except ImportError as exc:
        raise ImportError("plot_lake_std_ratio_map requires cartopy") from exc

    cfg = cfg or load_sws_config()
    mode = _normalize_summary_mode(summary_mode)
    value_col = _resolve_filter_mode_value_col(filter_mode, value_col)
    needs_residual = str(value_col).endswith("_residual")
    if plot_gdf is not None:
        plot_gdf = plot_gdf.copy()
        if "_ratio" not in plot_gdf.columns:
            plot_gdf["_ratio"] = _summary_std_ratio_values(plot_gdf, value_col).values
        plot_gdf = plot_gdf.dropna(subset=["_ratio", "plot_lat", "plot_lon"]).copy()
        if plot_gdf.empty:
            raise ValueError(f"No valid features in plot_gdf for {value_col}")
    else:
        base_gdf = _prepare_lake_std_ratio_plot_gdf(
            summary_df,
            catalog,
            cfg=cfg,
            volume_series=volume_series,
            grace_da=grace_da,
            window_deg=window_deg,
            value_col=value_col,
            min_pct=None,
            summary_mode=mode,
            calculate_residual=needs_residual,
        )
        n_before = len(base_gdf)
        if min_pct is not None and float(min_pct) > 0:
            plot_gdf = base_gdf[base_gdf["_ratio"] > float(min_pct)].copy()
        else:
            plot_gdf = base_gdf.copy()
        if plot_gdf.empty:
            label = "pixels" if mode == "grace_pixel" else "lakes"
            if min_pct is not None and float(min_pct) > 0:
                raise ValueError(
                    f"No {label} with {value_col} > {float(min_pct):g}% after filtering "
                    f"(started with {n_before} valid rows)"
                )
            raise ValueError(f"No valid {label} with {value_col}")

        if min_pct is not None and float(min_pct) > 0 and not (mode == "grace_pixel"):
            logger.info(
                "Lake σ-ratio map: %s / %s lakes shown (> %.1f%% of GRACE σ)",
                len(plot_gdf), n_before, float(min_pct),
            )

    norm, vmin, vmax = _summary_color_norm(
        plot_gdf["_ratio"].values,
        vmin=_resolve_std_ratio_vmin(vmin, min_pct),
        vmax=vmax,
    )
    cmap_obj = plt.get_cmap(cmap)

    if extent is None:
        if grace_da is not None:
            try:
                extent = _extent_from_grace_da(grace_da, pad_deg=extent_pad_deg)
            except (ValueError, KeyError):
                extent = None
        if extent is None:
            if mode == "per_lake" and "lake_id" in plot_gdf.columns:
                try:
                    extent = _extent_from_lake_ids(plot_gdf["lake_id"].astype(int).tolist(), cfg)
                except ValueError:
                    extent = _extent_from_plot_coords(plot_gdf)
            else:
                extent = _extent_from_plot_coords(plot_gdf)

    fig, ax = plt.subplots(
        figsize=figsize,
        subplot_kw={"projection": ccrs.PlateCarree()},
    )
    _setup_arid_map_axes(
        ax, extent,
        domains_gdf=domains_gdf,
        domain_col=domain_col,
        draw_domain_boundaries=draw_domain_boundaries,
    )

    if mode == "grace_pixel":
        sc = _draw_grace_pixel_std_ratio_layers(
            ax, plot_gdf, cfg=cfg, norm=norm, cmap_obj=cmap_obj,
            point_size=point_size, polygon_edgecolor=polygon_edgecolor,
            polygon_linewidth=polygon_linewidth, draw_lake_polygons=draw_lake_polygons,
        )
    else:
        sc = _draw_lake_std_ratio_layers(
            ax, plot_gdf, cfg=cfg, norm=norm, cmap_obj=cmap_obj,
            point_size=point_size, polygon_edgecolor=polygon_edgecolor,
            polygon_linewidth=polygon_linewidth, draw_lake_polygons=draw_lake_polygons,
        )
    default_title = _lake_std_ratio_map_title(
        summary_mode=mode, min_pct=min_pct, window_deg=window_deg,
    )

    _add_lake_std_ratio_colorbar(fig, ax, sc)
    ax.set_title(
        title or default_title,
        fontsize=12, fontweight="medium", pad=8,
    )
    _finalize_std_ratio_map_figure(fig)
    _save_figure(fig, save_path, dpi, show)

    if return_table:
        table_df = _filtered_std_ratio_summary_table(
            plot_gdf, summary_mode=mode, value_col=value_col,
        )
        label = "GRACE pixels" if mode == "grace_pixel" else "lakes"
        if min_pct is not None and float(min_pct) > 0:
            _note(
                f"{len(table_df)} {label} with lake std > {float(min_pct):g}% of GRACE "
                "(sorted high to low)"
            )
        else:
            _note(f"{len(table_df)} {label} (sorted high to low)")
        return table_df
    return fig


_SHAPEFILE_FIELD_RENAME = {
    "lake_std_pct_of_grace": "std_pct_gr",
    "haversine_distance_deg": "dist_deg",
    "grace_window_area_km2": "grace_km2",
    "completeness_pct": "complete",
    "lake_trend_cm_yr": "ltrend_cm",
    "grace_trend_cm_yr": "gtrend_cm",
    "n_overlap_months": "n_months",
    "lake_std_cm_residual": "std_cm_res",
    "grace_std_cm_residual": "gstd_cm_re",
    "lake_std_pct_of_grace_residual": "std_pct_re",
}


def export_lake_std_ratio_shapefile(
    filtered_df: pd.DataFrame,
    *,
    summary_mode: str = "per_lake",
    save_path: Union[str, Path],
    crs: str = "EPSG:4326",
) -> gpd.GeoDataFrame:
    """
    Convert a ``plot_lake_std_ratio_map(..., return_table=True)`` DataFrame into a
    point GeoDataFrame/file for GIS comparison (e.g. against recharge efficiency).

    ``summary_mode='per_lake'``: one point per lake at its ``(lat, lon)``.
    ``summary_mode='grace_pixel'``: one point per GRACE pixel at
    ``(grace_lat, grace_lon)``, carrying the combined ``lake_ids``/``n_lakes``.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Output of ``plot_lake_std_ratio_map(..., return_table=True)`` (or
        ``_filtered_std_ratio_summary_table``); must retain its coordinate columns.
    summary_mode : {'per_lake', 'grace_pixel'}, default 'per_lake'
    save_path : str or Path
        Output path. ``.shp`` writes ESRI Shapefile (field names capped at 10
        characters via a fixed rename map); any other extension (e.g. ``.gpkg``)
        keeps full column names.
    crs : str, default 'EPSG:4326'

    Returns
    -------
    gpd.GeoDataFrame
        The exported GeoDataFrame (also written to ``save_path``).
    """
    mode = _normalize_summary_mode(summary_mode)
    lat_col, lon_col = ("grace_lat", "grace_lon") if mode == "grace_pixel" else ("lat", "lon")
    missing = [c for c in (lat_col, lon_col) if c not in filtered_df.columns]
    if missing:
        raise KeyError(
            f"filtered_df is missing coordinate column(s) {missing} for summary_mode={mode!r}"
        )

    df = filtered_df.copy()
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lon_col]).reset_index(drop=True)

    for col in df.columns:
        if pd.api.types.is_extension_array_dtype(df[col]):
            df[col] = df[col].astype("float64") if df[col].isna().any() else df[col].astype("int64")

    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[lon_col], df[lat_col]),
        crs=crs,
    )

    save_path = Path(save_path)
    if save_path.suffix.lower() == ".shp":
        rename = {k: v for k, v in _SHAPEFILE_FIELD_RENAME.items() if k in gdf.columns}
        gdf = gdf.rename(columns=rename)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_file(save_path)
    _item(_rel(save_path), "ok")
    return gdf


# Reader-facing column headers for the Fig S11 / supplement table.
_STD_RATIO_TABLE_RENAME_PIXEL = {
    "grace_lat": "GRACE pixel latitude (deg)",
    "grace_lon": "GRACE pixel longitude (deg)",
    "n_lakes": "Number of lakes in pixel",
    "lake_ids": "HydroLAKES IDs",
    "lake_names": "Lake names",
    "country": "Country",
    "area_km2": "Total lake area (km2)",
    "completeness_pct": "Lake record completeness (%)",
    "lake_std_pct_of_grace": "Lake std as % of GRACE std",
    "lake_std_cm": "Lake storage std (cm WE)",
    "grace_std_cm": "GRACE TWSA std (cm WE)",
    "lake_std_pct_of_grace_residual": "Residual lake std as % of GRACE",
    "lake_std_cm_residual": "Residual lake storage std (cm WE)",
    "grace_std_cm_residual": "Residual GRACE TWSA std (cm WE)",
    "lake_trend_cm_yr": "Lake trend (cm/yr)",
    "grace_trend_cm_yr": "GRACE trend (cm/yr)",
    "n_overlap_months": "Overlap months",
    "window_deg": "GRACE window (deg)",
    "grace_window_area_km2": "GRACE window area (km2)",
    "haversine_distance_deg": "Distance to GRACE cell (deg)",
}

_STD_RATIO_TABLE_RENAME_LAKE = {
    "lake_id": "HydroLAKES ID",
    "lake_name": "Lake name",
    "country": "Country",
    "lat": "Lake latitude (deg)",
    "lon": "Lake longitude (deg)",
    "grace_lat": "GRACE pixel latitude (deg)",
    "grace_lon": "GRACE pixel longitude (deg)",
    "area_km2": "Lake area (km2)",
    "completeness_pct": "Lake record completeness (%)",
    "lake_std_pct_of_grace": "Lake std as % of GRACE std",
    "lake_std_cm": "Lake storage std (cm WE)",
    "grace_std_cm": "GRACE TWSA std (cm WE)",
    "lake_std_pct_of_grace_residual": "Residual lake std as % of GRACE",
    "lake_std_cm_residual": "Residual lake storage std (cm WE)",
    "grace_std_cm_residual": "Residual GRACE TWSA std (cm WE)",
    "lake_trend_cm_yr": "Lake trend (cm/yr)",
    "grace_trend_cm_yr": "GRACE trend (cm/yr)",
    "n_overlap_months": "Overlap months",
    "window_deg": "GRACE window (deg)",
    "grace_window_area_km2": "GRACE window area (km2)",
    "haversine_distance_deg": "Distance to GRACE cell (deg)",
}

_STD_RATIO_TABLE_DROP = {
    "record_type",
    "grid_assignment",
    "n_grace_pixels",
    "lake_id",  # prefer lake_ids in pixel mode
    "lake_name",  # prefer lake_names in pixel mode
    "_ratio",
    "plot_lat",
    "plot_lon",
    "geometry",
}


def clean_lake_std_ratio_table(
    filtered_df: pd.DataFrame,
    *,
    summary_mode: str = "grace_pixel",
) -> pd.DataFrame:
    """
    Rename and trim a filtered Fig S11 summary for a reader-facing supplement table.

    Drops internal columns, renames remaining fields with clear units, and rounds
    numeric values. Does not write to disk.
    """
    mode = _normalize_summary_mode(summary_mode)
    rename_map = (
        _STD_RATIO_TABLE_RENAME_PIXEL if mode == "grace_pixel" else _STD_RATIO_TABLE_RENAME_LAKE
    )
    drop = set(_STD_RATIO_TABLE_DROP)
    if mode == "grace_pixel":
        drop |= {"lat", "lon"}  # duplicates of grace_lat / grace_lon
    else:
        drop -= {"lake_id", "lake_name"}

    df = filtered_df.copy()
    df = df.drop(columns=[c for c in drop if c in df.columns], errors="ignore")

    # Stable column order: known rename keys first, then any leftovers
    ordered = [c for c in rename_map if c in df.columns]
    leftovers = [c for c in df.columns if c not in ordered]
    df = df[ordered + leftovers]
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    lat_lon_cols = [c for c in df.columns if "latitude" in c.lower() or "longitude" in c.lower()]
    area_cols = [c for c in df.columns if "area (km2)" in c.lower()]
    pct_std_trend = [
        c for c in df.columns
        if any(tok in c.lower() for tok in ("std", "%", "trend", "completeness", "distance", "window (deg)"))
        and c not in area_cols
    ]
    for col in lat_lon_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(3)
    for col in area_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce").round(1)
    for col in pct_std_trend:
        if pd.api.types.is_numeric_dtype(df[col]) or df[col].dtype == object:
            num = pd.to_numeric(df[col], errors="coerce")
            if num.notna().any():
                df[col] = num.round(2)

    return df.reset_index(drop=True)


def export_lake_std_ratio_table(
    filtered_df: pd.DataFrame,
    *,
    summary_mode: str = "grace_pixel",
    save_path: Union[str, Path],
    min_pct: Optional[float] = None,
) -> pd.DataFrame:
    """
    Clean and save the filtered Fig S11 lake/pixel table as UTF-8 CSV for the supplement.

    Parameters
    ----------
    filtered_df : pd.DataFrame
        Output of ``plot_lake_std_ratio_map(..., return_table=True)``.
    summary_mode : {'per_lake', 'grace_pixel'}
    save_path : str or Path
        Destination CSV path (parent dirs created as needed).
    min_pct : float or None
        Optional threshold used only for the status line.

    Returns
    -------
    pd.DataFrame
        The cleaned table written to ``save_path``.
    """
    mode = _normalize_summary_mode(summary_mode)
    cleaned = clean_lake_std_ratio_table(filtered_df, summary_mode=mode)
    # Excel-safe HydroLAKES ID text (comma-separated lists otherwise truncate).
    for col in cleaned.columns:
        if "hydrolakes id" in str(col).lower():
            cleaned[col] = cleaned[col].map(
                lambda v: (
                    ""
                    if v is None or (isinstance(v, float) and pd.isna(v))
                    else (
                        str(v)
                        if str(v).startswith('="')
                        else f'="{str(v).strip().replace(chr(34), chr(34)+chr(34))}"'
                    )
                )
            )
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    cleaned.to_csv(save_path, index=False, encoding="utf-8-sig")

    label = "GRACE pixels" if mode == "grace_pixel" else "lakes"
    if min_pct is not None and float(min_pct) > 0:
        _announce(f"Fig S11: {len(cleaned)} {label} (lake std > {float(min_pct):g}% of GRACE)")
    else:
        _announce(f"Fig S11: {len(cleaned)} {label}")
    _item(_rel(save_path), "ok")
    return cleaned


def plot_lake_std_ratio_maps_by_domain(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    domain_col: str = "Domain",
    arid_domains_path: Optional[Union[str, Path]] = None,
    save_dir: Optional[Union[str, Path]] = None,
    include_global: bool = False,
    min_pct: Optional[float] = 10.0,
    print_lake_names: bool = True,
    adaptive_figsize: bool = True,
    summary_mode: str = "per_lake",
    show: bool = False,
    cbar_max: float = 100.0,
    **map_kwargs,
) -> Dict[str, plt.Figure]:
    """
    One zoomed σ-ratio map per arid ``Domain`` polygon.

    ``summary_mode='per_lake'``: HydroLAKES polygons + lake centroids (may recompute
    per-lake stats when *lake_summary* is pixel-aggregated).

    ``summary_mode='grace_pixel'``: GRACE-pixel centres with combined-lake σ ratios
    from *lake_summary* (consistent with ``aggregate_by_grace_pixel=True``).

    Parameters
    ----------
    summary_mode : {'per_lake', 'grace_pixel'}, default 'per_lake'
    cbar_max : float, default 100.0
        Upper colorbar limit for domain maps (``extend='max'`` for higher values).
    """
    cfg = cfg or load_sws_config()
    mode = _normalize_summary_mode(summary_mode)
    map_kwargs = dict(map_kwargs)
    map_kwargs.setdefault("min_pct", min_pct)
    map_kwargs.setdefault("summary_mode", mode)
    map_kwargs.setdefault("vmax", float(cbar_max))
    map_kwargs.setdefault("draw_lake_polygons", True)
    map_kwargs.setdefault("polygon_edgecolor", "black")
    map_kwargs.setdefault("polygon_linewidth", 0.65)
    user_figsize = map_kwargs.get("figsize")
    base_height = float(user_figsize[1]) if user_figsize else 4.0

    domains = load_arid_domains(arid_domains_path, cfg=cfg, domain_col=domain_col)
    filter_pct = map_kwargs.get("min_pct", min_pct)

    if mode == "grace_pixel":
        global_plot_gdf = _prepare_lake_std_ratio_plot_gdf(
            summary_df, catalog, cfg=cfg, min_pct=filter_pct, summary_mode="grace_pixel",
        )
        domain_groups = _assign_grace_pixels_to_domains(global_plot_gdf, domains, domain_col=domain_col)
        per_lake = summary_df
        id_col = None
    else:
        per_lake = _resolve_std_ratio_summary(
            summary_df, catalog, volume_series, grace_da,
            cfg=cfg, window_deg=window_deg, summary_mode="per_lake",
        )
        global_plot_gdf = _prepare_lake_std_ratio_plot_gdf(
            per_lake, catalog, cfg=cfg, window_deg=window_deg,
            min_pct=filter_pct, summary_mode="per_lake",
        )
        domain_groups = _assign_lakes_to_domains(global_plot_gdf, domains, domain_col=domain_col)
        id_col = "lake_id"

    n_global = len(global_plot_gdf)
    feature_label = "pixel(s)" if mode == "grace_pixel" else "lake(s)"

    assigned_keys: set = set()
    if id_col and id_col in global_plot_gdf.columns:
        for df in domain_groups.values():
            assigned_keys.update(df[id_col].astype(int).tolist())
        unassigned = global_plot_gdf[~global_plot_gdf[id_col].astype(int).isin(assigned_keys)]
        if print_lake_names and not unassigned.empty:
            print(
                f"\n{len(unassigned)} {feature_label} > {float(filter_pct):g}% "
                f"not assigned to any Domain:"
            )
            for label in _format_std_ratio_labels(unassigned, mode):
                print(f"  • {label}")
    elif mode == "grace_pixel" and print_lake_names:
        assigned_count = sum(len(df) for df in domain_groups.values())
        if assigned_count < n_global:
            print(
                f"\n{n_global - assigned_count} GRACE pixel(s) > {float(filter_pct):g}% "
                f"outside all Domain polygons"
            )

    figures: Dict[str, plt.Figure] = {}
    save_dir = Path(save_dir) if save_dir else None
    if save_dir:
        save_dir.mkdir(parents=True, exist_ok=True)

    if include_global:
        global_path = save_dir / f"lake_std_ratio_global_win{int(window_deg)}.jpeg" if save_dir else None
        global_kwargs = {k: v for k, v in map_kwargs.items() if k not in ("min_pct", "summary_mode")}
        figures["global"] = plot_lake_std_ratio_map(
            per_lake if mode == "per_lake" else summary_df,
            catalog,
            volume_series=volume_series,
            grace_da=grace_da,
            cfg=cfg,
            window_deg=window_deg,
            domains_gdf=domains,
            domain_col=domain_col,
            draw_domain_boundaries=True,
            save_path=global_path,
            show=show,
            plot_gdf=global_plot_gdf,
            min_pct=filter_pct,
            summary_mode=mode,
            **global_kwargs,
        )

    all_domain_names = sorted(domains[domain_col].astype(str).unique())

    for domain_name in all_domain_names:
        plot_gdf = domain_groups.get(domain_name)
        if plot_gdf is None or plot_gdf.empty:
            if print_lake_names:
                pct_msg = f" > {float(filter_pct):g}%" if filter_pct is not None and float(filter_pct) > 0 else ""
                print(f"{domain_name}: no {feature_label}{pct_msg} — skipped")
            continue

        if print_lake_names:
            labels = _format_std_ratio_labels(plot_gdf, mode)
            pct_note = (
                f" (> {float(filter_pct):g}% of GRACE σ)"
                if filter_pct is not None and float(filter_pct) > 0
                else ""
            )
            print(f"\n{domain_name}{pct_note} — {len(labels)} {'GRACE pixel' if mode == 'grace_pixel' else 'lake'}(s):")
            for label in labels:
                print(f"  • {label}")

        dom_geom = domains.loc[domains[domain_col].astype(str) == domain_name]
        raw_extent = _extent_from_plot_coords(plot_gdf)
        if adaptive_figsize and user_figsize is None:
            extent, domain_figsize = _balanced_domain_extent_and_figsize(
                raw_extent, base_height=base_height, max_aspect=3.0,
            )
        else:
            extent = raw_extent
            domain_figsize = map_kwargs.get("figsize", (10.0, base_height))

        if mode == "per_lake" and id_col:
            filtered_ids = plot_gdf[id_col].astype(int).tolist()
            sub_summary_f = per_lake[per_lake[id_col].isin(filtered_ids)]
            sub_cat_f = catalog[catalog[id_col].isin(filtered_ids)]
        else:
            sub_summary_f = summary_df
            sub_cat_f = catalog

        safe = re.sub(r"[^\w\-]+", "_", domain_name)[:40]
        dom_path = save_dir / f"lake_std_ratio_{safe}_win{int(window_deg)}.jpeg" if save_dir else None
        dom_kwargs = {
            k: v for k, v in map_kwargs.items()
            if k not in ("figsize", "min_pct", "summary_mode")
        }
        dom_title = _lake_std_ratio_domain_title(
            domain_name,
            summary_mode=mode,
            min_pct=filter_pct,
            window_deg=window_deg,
        )
        figures[domain_name] = plot_lake_std_ratio_map(
            sub_summary_f,
            sub_cat_f,
            volume_series=volume_series,
            grace_da=grace_da,
            cfg=cfg,
            window_deg=window_deg,
            domains_gdf=dom_geom,
            domain_col=domain_col,
            draw_domain_boundaries=True,
            extent=extent,
            figsize=domain_figsize,
            title=dom_title,
            save_path=dom_path,
            show=show,
            plot_gdf=plot_gdf,
            min_pct=filter_pct,
            summary_mode=mode,
            **dom_kwargs,
        )

    n_domain = len(figures) - (1 if "global" in figures else 0)
    n_domain_features = sum(len(df) for df in domain_groups.values())
    if print_lake_names and n_domain_features != n_global:
        print(
            f"\nNote: {n_global} {feature_label} pass the global filter but "
            f"{n_domain_features} appear on domain maps."
        )
    return figures


def save_lake_grace_comparison_figures(
    summary_df: pd.DataFrame,
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    precip_da: Optional[xr.DataArray] = None,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    parallel: bool = True,
    aggregate_by_grace_pixel: bool = False,
    show: bool = False,
) -> List[Path]:
    """
    Save lake/GRACE/precip comparison JPEGs for each row in ``summary_df``.

    Pass a threshold-filtered table (e.g. ``plot_lake_std_ratio_map(..., return_table=True)``)
    to write only Fig S11-qualifying figures instead of every summary row.
    """
    cfg = cfg or load_sws_config()
    if summary_df is None or summary_df.empty:
        _note("no comparison figures to save (empty table)")
        return []

    cfg.figures_dir.mkdir(parents=True, exist_ok=True)
    show = bool(show)
    n_figs = len(summary_df)
    _note(f"saving {n_figs} comparison figure(s)")
    _note(f"dir: {_rel(cfg.figures_dir)}/")

    def _plot_lake_row(row_dict):
        lake_id = row_dict.get("lake_id")
        if lake_id is None or lake_id not in volume_series:
            return None
        meta = catalog.loc[catalog["lake_id"] == lake_id]
        meta_dict = meta.iloc[0].to_dict() if not meta.empty else row_dict
        safe_name = re.sub(
            r"[^\w\-]+", "_",
            _display_lake_name(meta_dict.get("lake_name")),
        )[:60]
        out = cfg.figures_dir / f"{lake_id}_{safe_name}_lake_grace_precip_win{window_deg}.jpeg"
        plot_lake_grace_precip_comparison(
            meta_dict,
            volume_series[lake_id],
            grace_da,
            precip_da=precip_da,
            window_deg=window_deg,
            cfg=cfg,
            save_path=out,
            show=show,
        )
        return out

    def _plot_pixel_row(row_dict):
        ids_str = row_dict.get("lake_ids", "")
        lake_ids = [int(x) for x in str(ids_str).split(",") if str(x).strip()]
        if not lake_ids:
            return None
        glat = float(row_dict["grace_lat"])
        glon = float(row_dict["grace_lon"])
        out = cfg.figures_dir / (
            f"pixel_{glat:.1f}_{glon:.1f}_n{len(lake_ids)}_lake_grace_precip_win{window_deg}.jpeg"
        )
        plot_grace_pixel_lake_grace_comparison(
            glat,
            glon,
            lake_ids,
            catalog,
            volume_series,
            grace_da,
            precip_da=precip_da,
            window_deg=window_deg,
            cfg=cfg,
            save_path=out,
            show=show,
        )
        return out

    records = summary_df.to_dict("records")
    plot_fn = _plot_pixel_row if aggregate_by_grace_pixel else _plot_lake_row
    n_jobs = cfg.n_process_workers if parallel else 1
    if len(records) > 1:
        results = _parallel_thread_map(
            plot_fn, records, n_jobs=n_jobs, desc="Saving figures", unit="fig",
        )
    else:
        results = [plot_fn(r) for r in records]
    saved = [p for p in results if p is not None]
    _item(f"{len(saved)} figures in {_rel(cfg.figures_dir)}/", "ok")
    logger.info("Saved %s lake/GRACE/precip plots to %s", len(saved), cfg.figures_dir)
    return saved


def analyze_lake_grace_comparisons(
    catalog: pd.DataFrame,
    volume_series: Dict[Any, pd.Series],
    grace_da: xr.DataArray,
    precip_da: Optional[xr.DataArray] = None,
    cfg: Optional[SWSConfig] = None,
    window_deg: float = 1.0,
    parallel: bool = True,
    aggregate_by_grace_pixel: bool = False,
    save_figures: bool = False,
    show: bool = False,
    calculate_residual: bool = False,
) -> Tuple[pd.DataFrame, List[Path]]:
    """
    Compute lake vs GRACE summary statistics and optionally save comparison plots.

    Prefer ``save_figures=False`` here, then call
    ``save_lake_grace_comparison_figures`` on the threshold-filtered Fig S11 table
    so only qualifying pixels/lakes get JPEGs.

    Parameters
    ----------
    aggregate_by_grace_pixel : bool
        If True, lakes sharing the same ``(grace_lat, grace_lon)`` are combined:
        volume anomalies (km3) are summed, converted once to cm, and compared
        to one GRACE series per pixel. Std/trends are computed on that combined
        cm series (not by combining per-lake std values).
    calculate_residual : bool
        If True, also decomposes both series (calendar-phase-locked linear trend +
        annual + semi-annual harmonic fit) and adds residual-based std/pct columns
        to ``summary_df``. Does not affect figure saving.
    save_figures : bool
        If True, saves a JPEG for every summary row (can be hundreds). Prefer
        filtering first, then ``save_lake_grace_comparison_figures``.

    Returns
    -------
    summary_df : pd.DataFrame
        Per-lake rows, or per-pixel rows when ``aggregate_by_grace_pixel=True``.
    saved_plots : list of Path
        Empty when ``save_figures=False``.
    """
    cfg = cfg or load_sws_config()
    summary_df = compute_lake_grace_summaries(
        catalog,
        volume_series,
        grace_da,
        cfg=cfg,
        window_deg=window_deg,
        parallel=parallel,
        aggregate_by_grace_pixel=aggregate_by_grace_pixel,
        calculate_residual=calculate_residual,
    )
    if not save_figures:
        return summary_df, []

    saved = save_lake_grace_comparison_figures(
        summary_df,
        catalog,
        volume_series,
        grace_da,
        precip_da=precip_da,
        cfg=cfg,
        window_deg=window_deg,
        parallel=parallel,
        aggregate_by_grace_pixel=aggregate_by_grace_pixel,
        show=show,
    )
    return summary_df, saved


# ---------------------------------------------------------------------------
# GRACE comparison (legacy helpers)
# ---------------------------------------------------------------------------

def extract_grace_twsa_at_lake(
    grace_da: xr.DataArray,
    lat: float,
    lon: float,
    window_deg: float = 1.0,
    analysis_start: str = ANALYSIS_START,
    analysis_end: str = ANALYSIS_END,
    **_: Any,
) -> pd.Series:
    """
    Extract GRACE TWSA at a lake location (cm, native solution anomaly).

    Parameters
    ----------
    window_deg : int
        ``1`` = single haversine-nearest GRACE cell; ``3`` = mean over 3×3 cells
        centred on that assigned pixel.
    """
    assignment = assign_grace_grid_pixel(grace_da, lat, lon)
    series = _extract_grid_timeseries(
        grace_da,
        assignment["grace_lat"],
        assignment["grace_lon"],
        window_deg,
    )
    series.name = f"grace_twsa_cm_win{window_deg}"
    return clip_to_analysis_window(series, analysis_start, analysis_end)


# ---------------------------------------------------------------------------
# High-level pipeline entry points
# ---------------------------------------------------------------------------

def run_download_all(
    cfg: Optional[SWSConfig] = None,
    glolakes_version: str = "v1.0",
    glolakes_products: Optional[Sequence[str]] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Download HydroLAKES polygons + GloLakes absolute ICESat-2 storage (paper pipeline)."""
    cfg = cfg or load_sws_config()
    products = list(glolakes_products or ["absolute_icesat2"])
    summary: Dict[str, Any] = {}
    _announce_resources()

    _announce("SWS data download (2 steps)")
    _note(f"dir: {_rel(cfg.raw_dir)}/")
    if force:
        _note("FORCE=True (re-download raw files)")

    _announce("[1/2] HydroLAKES (global lake polygons)")
    summary["hydrolakes"] = str(download_hydrolakes(cfg, force=force))

    _announce(f"[2/2] GloLakes {glolakes_version} ({', '.join(products)})")
    glolakes_paths = download_glolakes(
        cfg=cfg, products=products, version=glolakes_version, force=force
    )
    summary["glolakes"] = {k: str(v) for k, v in glolakes_paths.items()}

    _announce("download complete")
    return summary


