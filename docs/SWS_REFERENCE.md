# SWS Analysis Reference

Last updated: 2026-07-17

## Purpose

Surface Water Storage Anomaly (SWSA) analysis for arid regions, comparable to GRACE TWSA, and export of the lake/reservoir std-ratio shapefile used in manuscript **Figure S11**.

- Baseline: **2004-01-01 to 2009-12-31** mean removed
- Analysis window (notebook 02): **2004-04-01 to 2025-09-30**
- Units: **cm water equivalent**
- Arid mask: `data/processed/boundaries/ai_v3_yr_mask_02_pol.shp` (Zomer AI ≤ 0.2)

## Datasets used (paper pipeline)

| Dataset | Role |
|---------|------|
| GloLakes absolute ICESat-2 (`v1.0`) | Primary monthly lake storage |
| HydroLAKES polygons | Lake geometry / arid filter / maps |
| GRACE CSR/JPL/GSFC mean | TWSA comparison |
| GPM IMERG monthly (from notebook 01) | Precipitation on GRACE grid |

Download entry point: `run_download_all(cfg)` → HydroLAKES + GloLakes `absolute_icesat2` only.

## Unit conversions

```text
# Volume (km³) and area (km²):
swsa_cm = (delta_V_km3 / area_km2) * 1e5

# GloLakes storage in MCM (million m³):
V_km3 = MCM * 0.001
swsa_cm = (MCM * 0.001 / area_km2) * 1e5
```

## Download endpoints

| Dataset | URL / method |
|---------|--------------|
| GloLakes | https://thredds.nci.org.au/thredds/catalog/catalogs/ub8/global/GloLakes/GloLakes.html |
| GloLakes ICESat-2 v1.0 | `fileServer/ub8/global/GloLakes/GloLakes_v1.0/Global_Lake_Absolute_Storage_LandsatPlusICESat2 (1984-present).nc` |
| HydroLAKES | https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip |

## Fig S11 shapefile

Notebook 02 exports:

`data/raw/sws/shapefiles/lake_std_ratio_grace_pixel_win1_thr20.shp`

Notebook 03 reads that path for the RE map with lake overlays (σ ≥ 20% of GRACE σ).

## Dependencies

- `geopandas`, `xarray`, `zarr`, `numcodecs` (GPM Zarr), `dbfread` (HydroLAKES attribute cache)
- Optional: `python-dotenv`

## HydroLAKES performance

The polygon shapefile is large. `load_hydrolakes_attrs()` builds/reads
`processed/hydrolakes_attrs.parquet` for fast area lookups.

_(Appended automatically by `update_reference_md()` during runs.)_

## Known issues log

- **2026-07-21 00:22 UTC**: HydroLAKES downloaded to /mnt/d/codes/python/grace_ds/github/data/raw/sws/raw/hydrolakes/HydroLAKES_polys_v10_shp/HydroLAKES_polys_v10_shp

- **2026-07-21 00:23 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-07-21 00:33 UTC**: HydroLAKES polygon cache: /mnt/d/codes/python/grace_ds/github/data/raw/sws/processed/hydrolakes_polygons.parquet

- **2026-07-21 23:44 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-07-21 23:59 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-07-22 01:08 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-07-22 01:10 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-08-02 23:16 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']
