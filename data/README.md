# Data layout

Paths are relative to the **repository root**.

## What is shipped in this repository

| Path | Contents |
|------|----------|
| `data/processed/groundwater/` | Preprocessed IGRAC well locations and monthly time series (CSV). Used by notebook `02` by default. |
| `data/processed/boundaries/` | Optional place for arid-mask / aquifer / Great Basin shapefiles (not shipped by default — add locally). |

## What is downloaded / generated locally (not in git)

| Path | Filled by | Contents |
|------|-----------|----------|
| `data/raw/gpm/` | Notebook `01` | IMERG precipitation downloads |
| `data/raw/gldas/` | Notebook `01` | GLDAS SM / runoff / SWE inputs |
| `data/raw/grace/` | Notebook `01` or manual place | CSR / JPL / GSFC mascon files |
| `data/raw/aridity/` | Notebook `01` or manual place | Aridity index raster / polygons |
| `data/raw/sws/` | Notebook `03` | Lake / reservoir source downloads |
| `data/raw/groundwater/` | Optional | Empty placeholder if you re-download raw IGRAC country files |
| `data/interim/` | Notebook `01` | Coarsened / analysis-ready NetCDF or Zarr products |
| `outputs/` | Notebooks `02`–`03` | Figures, tables, GeoTIFFs |

## Groundwater note

Default paper analysis loads:

- `data/processed/groundwater/all_countries_well_locations.csv`
- `data/processed/groundwater/all_countries_time_series.csv`

Re-running the full IGRAC preprocess from raw country files is optional (commented cell in notebook `02`) and can take a long time. Use `data/raw/groundwater/` only if you choose that path.
