# A global, event-based assessment of aquifer response to extreme precipitation in arid regions

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.21778709.svg)](https://doi.org/10.5281/zenodo.21778709)

This repository contains the analysis code, notebooks, and derived products that accompany the study.

The workflow uses GRACE/GRACE-FO terrestrial water storage, GPM IMERG precipitation, GLDAS land-surface models, HydroLAKES/GloLakes surface-water storage, and in-situ groundwater wells.

**Author:** Hassan Saleh (Western Michigan University)

## Repository layout

```text
notebooks/   # 01 download → 02 SWS / Fig S11 shapefile → 03 paper analysis
src/         # Python modules imported by the notebooks
data/        # raw (download), interim (Zarr), processed (shipped tables / boundaries)
outputs/     # figures, tables, rasters (included here; also recreated by notebooks)
```

## Setup

### 1. Create the environment

Requires [Mamba](https://mamba.readthedocs.io/) or Conda. From the repository root:

```bash
cd /path/to/this/repo
mamba env create -f environment.yml
mamba activate grace-arid
```

Register the kernel for Jupyter if needed:

```bash
python -m ipykernel install --user --name grace-arid --display-name "Python (grace-arid)"
```

### 2. Optional extras (only for specific cells)

These are **not** required for the default analysis path. Install only if you run the cells that need them:

```bash
# Interactive TWSA–CPA pixel map in notebook 03 (QC only; not a manuscript figure)
pip install plotly ipywidgets

# Optional raw IGRAC .ods reprocessing in notebook 03
# (default path uses shipped CSVs under data/processed/groundwater/ and does not need this)
pip install odfpy
```

### 3. Credentials (downloads)

- **NASA Earthdata Login** for IMERG / GLDAS in notebook `01`: see the [`earthaccess` authentication guide](https://earthaccess.readthedocs.io/en/latest/user/explanation/authenticate/) (interactive prompt, `~/.netrc`, or environment variables). Details are also in the notebook `01` auth section.
- Notebook `02` (GloLakes + HydroLAKES) needs **no API keys**.

### 4. Packages

**Core** (installed by `environment.yml`; required for the default notebook path):

| Package | Role |
|---------|------|
| python (>=3.10,<3.13) | Runtime |
| numpy, scipy, pandas | Arrays, stats, tables |
| xarray, dask, distributed | Labeled arrays and parallel compute |
| netcdf4, h5netcdf, zarr, numcodecs | NetCDF / Zarr I/O |
| rioxarray, rasterio | Raster georeferencing |
| geopandas, shapely, pyproj | Vector GIS |
| matplotlib, cartopy, seaborn | Maps and figures |
| tqdm, joblib | Progress bars and parallel loops |
| statsmodels, scikit-learn | Regression / models |
| requests | HTTP downloads |
| jupyter, ipykernel | Notebooks |
| earthaccess | NASA Earthdata search/download |
| dbfread | HydroLAKES attribute tables |
| pympler (pip) | Optional memory diagnostics |

**Optional extras** (pip; only for opt-in cells):

| Package | When needed |
|---------|-------------|
| plotly, ipywidgets | Notebook `03` interactive TWSA–CPA map (QC) |
| odfpy | Optional raw groundwater `.ods` reprocessing in notebook `03` |

## Run order

Open notebooks with the **repository root as the working directory**.

| Step | Notebook / action | Purpose |
|------|-------------------|---------|
| 1 | [`notebooks/01_download_preprocess_data.ipynb`](notebooks/01_download_preprocess_data.ipynb) | Download GPM IMERG + GLDAS + GRACE mascons (2002-01 to 2025-09) into `data/` |
| 2 | Arid boundaries under `data/processed/boundaries/` | Shipped in this release (`ai_v3_yr_mask_02_pol.shp`). See [`data/README.md`](data/README.md) |
| 3 | [`notebooks/02_sws_arid_analysis.ipynb`](notebooks/02_sws_arid_analysis.ipynb) | Surface-water (GloLakes) analysis; exports the Fig S11 lake-overlay shapefile used by notebook `03` |
| 4 | [`notebooks/03_grace_arid_analysis.ipynb`](notebooks/03_grace_arid_analysis.ipynb) | Main paper analysis and figures |

Notebook `03` loads **shipped** groundwater CSVs from `data/processed/groundwater/` by default.

### Disk use and re-running

Manuscript figures and tables are already under `outputs/`, and notebook `03` uses the shipped groundwater CSVs, so you do not need to reprocess raw well files to inspect results.

Re-running the full download path is heavy: daily IMERG is about 250 GB. After notebook `01` builds the monthly Zarr stores, optional commented cleanup cells can delete those raw granules (see [`data/README.md`](data/README.md#disk-usage-and-cleanup)). The much smaller interim products (`data/interim/gpm/` ~5 GB and `data/interim/gldas/` ~0.3 GB) are enough to re-run most of the analysis without re-downloading raw IMERG/GLDAS. GRACE mascons (~0.7 GB) are still downloaded directly in notebook `01`.

## Outputs

Figures and tables are written under `outputs/` and are included in this release. The Fig S11 lake-point shapefile used by notebook `03` is produced by notebook `02` under `data/raw/sws/shapefiles/`; it is not shipped in git, so run notebook `02` if you need that file locally.

<p style="color:#c1121f; border-left:4px solid #c1121f; padding:0.6em 0.8em; background:#fff5f5;">
<strong>GRACE data note:</strong> Notebook <code>01</code> retrieves the <em>most recent</em> CSR / JPL / GSFC mascon NetCDF releases.
Providers (especially JPL) update a single file that re-estimates the full mission series when new months are added: because of time correlation in the mascon solution, previous months can change slightly (largest updates near the end of the record).
Re-running with a newer download may therefore shift some grid-cell counts slightly.
<strong>This does not change the scientific results or conclusions of the analysis.</strong>
</p>

## Citation

Please cite this software as:

> Saleh, H. (2026). *A global, event-based assessment of aquifer response to extreme precipitation in arid regions* (Version 0.1.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.21778709

Also see [`CITATION.cff`](CITATION.cff).

## License

MIT — see [`LICENSE`](LICENSE).
