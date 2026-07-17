# SWS Analysis Reference (living document)

Last updated: 2026-06-24

## Purpose

Surface Water Storage Anomaly (SWSA) analysis for arid regions (2002–2025), comparable to GRACE TWSA:
- Baseline: **2004-01-01 to 2009-12-31** mean removed
- Units: **cm water equivalent**
- Arid mask: `ai_v3_yr_mask_02_pol.shp` (Zomer AI ≤ 0.2)

## Dataset ranking (most → least useful for arid SWSA)

| Rank | Dataset | Coverage | Notes |
|------|---------|----------|-------|
| 1 | GloLakes | 27k lakes, 1984–present | Primary; NCI THREDDS; MCM/GL units |
| 2 | DAHITI | ~5,257 volume series | Relative km³; API key required |
| 3 | Hydroweb.next | Large lakes, research QC | API key; py-hydroweb |
| 4 | G-REALM | Large lakes/reservoirs | Levels only; LakePy/scrape |
| 5 | RECOG | GRACE leakage correction | Grid/SH; not per-lake primary |
| 6–7 | WaterGAP / PCR-GLOBWB | Models | Not used in phase 1 |

## Unit conversions

```text
# Volume (km³) and area (km²):
swsa_cm = (delta_V_km3 / area_km2) * 1e5

# GloLakes storage in MCM (million m³):
V_km3 = MCM * 0.001
swsa_cm = (MCM * 0.001 / area_km2) * 1e5  # equivalent to MCM/area_km2 * 100

# Water surface elevation (m):
swsa_cm = delta_h_m * 100
```

## Download endpoints

| Dataset | URL / method |
|---------|----------------|
| GloLakes | https://thredds.nci.org.au/thredds/catalog/catalogs/ub8/global/GloLakes/GloLakes.html |
| GloLakes files (v1.0) | `fileServer/ub8/global/GloLakes/GloLakes_v1.0/` |
| HydroLAKES | https://data.hydrosheds.org/file/hydrolakes/HydroLAKES_polys_v10_shp.zip |
| DAHITI API | https://dahiti.dgfi.tum.de/api/v2/ |
| Hydroweb | https://hydroweb.next.theia-land.fr/ (py-hydroweb) |
| G-REALM | https://ipad.fas.usda.gov/cropexplorer/global_reservoir/ |
| RECOG-LR | https://doi.org/10.1594/PANGAEA.921851 |

## Dependencies

- `geopandas`, `xarray`, `zarr`, `numcodecs` (GPM Zarr), `dbfread` (HydroLAKES attribute cache)
- Optional: `python-dotenv`, `py-hydroweb`, `lakepy`

## HydroLAKES performance

The 1.5 GB polygon shapefile is slow on `/mnt/` mounts. `load_hydrolakes_attrs()` builds/reads
`processed/hydrolakes_attrs.parquet` for fast area lookups. Full polygons: `load_hydrolakes(geometry=True)`.


_(Appended automatically by `update_reference_md()` during runs.)_

- **2026-06-24 19:46 UTC**: GloLakes v1.0 downloaded products: ['relative_grealm']

- **2026-06-24 19:47 UTC**: HydroLAKES downloaded to /mnt/d/grace_ds/data/SWS/raw/hydrolakes/HydroLAKES_polys_v10_shp

## Known issues log

- **2026-06-24 20:29 UTC**: RECOG-LR: Pangaea zip unavailable; saved tab-delimited matrix. Grid/SH NetCDF files may require manual download from ESSD supplement.

- **2026-06-24 20:48 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'absolute_icesat2', 'absolute_s2', 'relative_s2', 'relative_icesat2', 'relative_grealm']

- **2026-06-24 20:51 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'absolute_icesat2', 'absolute_s2', 'relative_grealm', 'relative_s2', 'relative_icesat2']

- **2026-06-24 20:52 UTC**: GloLakes v1.0 downloaded products: ['relative_icesat2', 'relative_grealm', 'absolute_s2', 'relative_s2', 'absolute_grealm', 'absolute_icesat2']

- **2026-06-24 20:53 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 20:53 UTC**: GloLakes v1.0 downloaded products: ['absolute_s2', 'absolute_grealm', 'relative_s2', 'relative_grealm', 'absolute_icesat2', 'relative_icesat2']

- **2026-06-24 20:53 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 20:54 UTC**: GloLakes v1.0 downloaded products: ['absolute_s2', 'absolute_grealm', 'relative_icesat2', 'relative_grealm', 'relative_s2', 'absolute_icesat2']

- **2026-06-24 20:54 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 20:55 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 20:55 UTC**: lakepy not installed; G-REALM download skipped. pip install lakepy

- **2026-06-24 20:55 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'relative_grealm', 'absolute_s2', 'relative_s2', 'relative_icesat2', 'absolute_icesat2']

- **2026-06-24 20:55 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 20:56 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 20:56 UTC**: G-REALM: saved 0 lake series

- **2026-06-24 20:59 UTC**: GloLakes v1.0 downloaded products: ['relative_s2', 'relative_icesat2', 'absolute_icesat2', 'absolute_grealm', 'relative_grealm', 'absolute_s2']

- **2026-06-24 20:59 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 21:00 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 21:00 UTC**: G-REALM: saved 0 lake series

- **2026-06-24 21:06 UTC**: GloLakes v1.0 downloaded products: ['absolute_s2', 'relative_icesat2', 'absolute_grealm', 'absolute_icesat2', 'relative_grealm', 'relative_s2']

- **2026-06-24 21:14 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2', 'absolute_grealm', 'relative_grealm', 'relative_s2', 'absolute_s2', 'relative_icesat2']

- **2026-06-24 21:14 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 21:15 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 21:15 UTC**: G-REALM: saved 0 lake series

- **2026-06-24 21:17 UTC**: GloLakes v1.0 downloaded products: ['relative_icesat2', 'absolute_s2', 'relative_grealm', 'absolute_grealm', 'absolute_icesat2', 'relative_s2']

- **2026-06-24 21:17 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 21:17 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 21:17 UTC**: G-REALM: saved 0 lake series

- **2026-06-24 21:25 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-06-24 21:31 UTC**: GloLakes v1.0 downloaded products: ['absolute_icesat2']

- **2026-06-24 22:07 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'absolute_s2', 'absolute_icesat2', 'relative_s2', 'relative_icesat2', 'relative_grealm']

- **2026-06-24 22:09 UTC**: RECOG-LR: Pangaea metadata saved; grid NetCDF may require ESSD supplement.

- **2026-06-24 22:11 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'relative_grealm', 'absolute_icesat2', 'absolute_s2', 'relative_s2', 'relative_icesat2']

- **2026-06-24 22:11 UTC**: DAHITI: requested 58 volume series

- **2026-06-24 22:11 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-24 22:27 UTC**: HydroLAKES polygon cache: /mnt/d/grace_ds/data/SWS/processed/hydrolakes_polygons.parquet

- **2026-06-25 17:14 UTC**: GloLakes v1.0 downloaded products: ['relative_s2', 'relative_icesat2', 'absolute_s2', 'relative_grealm', 'absolute_icesat2', 'absolute_grealm']

- **2026-06-25 17:14 UTC**: DAHITI: requested 58 volume series

- **2026-06-25 17:15 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]

- **2026-06-25 22:17 UTC**: GloLakes v1.0 downloaded products: ['absolute_grealm', 'absolute_icesat2', 'relative_icesat2', 'relative_grealm', 'relative_s2', 'absolute_s2']

- **2026-06-25 22:18 UTC**: DAHITI: requested 58 volume series

- **2026-06-25 22:18 UTC**: Hydroweb download to /mnt/d/grace_ds/data/SWS/raw/hydroweb bbox=[-121.85393229384027, -51.53194309929455, 148.17848127818854, 51.41193212566725]
