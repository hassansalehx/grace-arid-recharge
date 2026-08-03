# Data layout

Paths are relative to the **repository root**.

**Study download window (notebook 01):** 2002-01-01 → 2025-09-30.

## What is shipped in this repository

| Path | Contents |
|------|----------|
| `data/processed/groundwater/` | Preprocessed IGRAC well locations and monthly time series (CSV). Used by notebook `03` by default. |
| `data/processed/boundaries/` | Arid-mask polygons (`ai_v3_yr_mask_02_pol.shp` and related files) used by notebooks `02`–`03`. |
| `outputs/figures/`, `outputs/tables/`, `outputs/rasters/` | Manuscript figures, Table S3, and derived GeoTIFF products (also regenerable by the notebooks). |

## What is downloaded/generated locally (not in git)

| Path | Filled by | Contents |
|------|-----------|----------|
| `data/raw/gpm/` | Notebook `01` | GPM IMERG Final daily granules |
| `data/raw/gldas/` | Notebook `01` | GLDAS CLSM / NOAH / VIC monthly NetCDFs |
| `data/raw/grace/` | Notebook `01` | CSR / JPL / GSFC mascons + CSR land mask (auto-downloaded) |
| `data/raw/aridity/` | Manual place | Optional aridity raster source files |
| `data/raw/sws/` | Notebook `02` | HydroLAKES + GloLakes downloads; SWS catalogs / shapefiles |
| `data/raw/groundwater/` | Optional | Empty placeholder for raw IGRAC re-download |
| `data/interim/gpm/` | Notebook `01` | `GPM_3IMERGDF_Jan2002_Sep2025_resToM.zarr` (monthly; daily is not written) |
| `data/interim/gldas/<model>/` | Notebook `01` | `{model}_Jan2002_Sep2025.zarr` (one file per model: SM + `total_runoff` + `SWE_inst`) |
| `outputs/` | Notebooks `02`–`03` | Figures, tables, GeoTIFFs |

### GRACE filename tokens

Centers update date spans in filenames. Notebooks resolve the newest match under `data/raw/grace/`:

| Key | Directory | Tokens (case-insensitive) |
|-----|-----------|---------------------------|
| CSR mascon | `grace/csr/` | `all-corrections` |
| CSR land mask | `grace/csr/` | `LandMask` |
| JPL CRI | `grace/jpl/` | `MSCNv04CRI` |
| GSFC half-degree | `grace/gsfc/` | `halfdegree` and `obp` |

## Groundwater note

Default paper analysis loads:

- `data/processed/groundwater/all_countries_well_locations.csv`
- `data/processed/groundwater/all_countries_time_series.csv`

Re-running the full IGRAC preprocess from raw country files is optional (commented cell in notebook `03`).

## Disk usage and cleanup

Approximate footprint after a full notebook `01`–`02` download (gitignored under `data/raw/**` and `data/interim/**`):

| Path | Size (order of magnitude) | Downstream artifact | Needed after artifact exists? |
|------|---------------------------|----------------------|-------------------------------|
| `data/raw/gpm/GPM_3IMERGDF/` (daily `.nc4`) | ~250 GB | `data/interim/gpm/*_resToM.zarr` (~5 GB) | No |
| `data/raw/gldas/<model>/` | ~2 GB | `data/interim/gldas/<model>/*.zarr` (~0.3 GB total) | No |
| `data/raw/sws/raw/hydrolakes/` (zip + shapefile) | ~2 GB | `data/raw/sws/processed/hydrolakes_polygons.parquet` (~1 GB cache) | No |
| `data/raw/sws/raw/glolakes/` | ~0.2 GB | Consumed directly per lake each run | Yes (kept) |
| `data/raw/grace/{csr,jpl,gsfc}/` | ~0.7 GB | **None** — notebooks `02`/`03` read these NetCDFs directly every run | Yes (kept; no interim by design) |
| `data/raw/aridity/ai_v3_yr.tif` (manually placed) | ~0.5 GB | `data/interim/aridity/ai_v3_yr_display.tif` (display cache) | Only for cache rebuild |
| `data/raw/groundwater/<country>/...` (manually placed) | ~2 GB | `data/processed/groundwater/all_countries_*.csv` (shipped) | Only for optional re-preprocessing |

Total raw is typically ~260 GB; interim ~5 GB; shipped processed ~27 MB. **Daily IMERG is the dominant cost.**

### Safe to delete after build (opt-in)

Cleanup helpers **verify** the downstream Zarr/cache first, then delete. Notebook cells are **commented by default** (same pattern as groundwater reprocessing in notebook `03`):

| Raw product | Helper | Notebook cell |
|-------------|--------|---------------|
| IMERG daily granules (~250 GB) | `remove_imerg_daily_granules()` in `src/download_data.py` | Optional cell after the IMERG build in notebook `01` |
| GLDAS raw granules (~2 GB) | `remove_gldas_raw_granules()` per model | Optional cell after the GLDAS build in notebook `01` |
| HydroLAKES zip/shapefile (~2 GB) | `remove_hydrolakes_raw()` in `src/sws_analysis_utils.py` | Optional cell after the download cell in notebook `02` |

Uncomment those cells only when you accept re-downloading if you later rebuild with `FORCE=True`.

### Keep — do not delete

- **GRACE mascons** under `data/raw/grace/` — there is no interim store; notebooks re-read these files every run (intentional).
- **`data/processed/**`** — shipped analysis inputs (groundwater CSVs, boundaries you place).
- **GloLakes raw** under `data/raw/sws/raw/glolakes/` — reused directly each run.

### User-supplied inputs (no automated cleanup)

Aridity raw TIF and groundwater raw country folders are **manually placed** (not downloaded by scripts in this repo). Archiving or deleting them after the display cache / shipped CSVs exist is your call; this repo does not ship deletion helpers for them because it cannot verify you have a backup.

## For reviewers

- Notebook `03` already uses the **fast groundwater path**: shipped CSVs under `data/processed/groundwater/`; raw IGRAC reprocessing stays commented off.
- To avoid downloading and processing ~256 GB of IMERG/GLDAS raw granules, prefer depositing the small interim products (~5.4 GB total) on a data repository (e.g. Zenodo/OSF) and placing them at:
  - `data/interim/gpm/*.zarr`
  - `data/interim/gldas/<model>/*.zarr`
- GRACE raw mascons remain a required, comparatively small (~0.7 GB) direct download regardless of path.
