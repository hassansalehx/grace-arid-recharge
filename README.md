# GRACE arid-region extreme precipitation recharge

Open analysis workflow for a global, event-based assessment of aquifer response to extreme precipitation in arid and hyper-arid regions (GRACE/GRACE-FO, IMERG, GLDAS, and in-situ wells).

**Authors:** Hassan Saleh, Mohamed Sultan (Western Michigan University)

## Repository layout

```text
notebooks/   # 01 download → 02 paper analysis → 03 optional SWS
src/         # Python modules imported by the notebooks
data/        # raw (download), interim (processed cubes), processed (shipped tables)
outputs/     # figures, tables, rasters (created locally)
docs/        # developer notes and manuscript–code map
```

## Quick start

### 1. Environment

```bash
cd /path/to/this/repo
mamba env create -f environment.yml
mamba activate grace-arid
```

### 2. Credentials (only for downloads)

- NASA Earthdata Login for IMERG / some NASA products: configure [`earthaccess`](https://earthaccess.readthedocs.io/) (typically `~/.netrc` or interactive login in notebook `01`).
- Optional SWS APIs: copy `.env.example` to `.env` and fill keys (never commit `.env`).

### 3. Run order

Open notebooks with the **repository root as the working directory** (Cursor/VS Code: set notebook working directory to the workspace root, or `cd` there before launching Jupyter).

| Step | Notebook | Purpose |
|------|----------|---------|
| 1 | [`notebooks/01_download_preprocess_predictors.ipynb`](notebooks/01_download_preprocess_predictors.ipynb) | Download / preprocess precipitation and LSM predictors into `data/` |
| 2 | [`notebooks/02_grace_arid_analysis.ipynb`](notebooks/02_grace_arid_analysis.ipynb) | Main paper analysis and figures |
| 3 (optional) | [`notebooks/03_sws_arid_analysis.ipynb`](notebooks/03_sws_arid_analysis.ipynb) | Surface-water / lake analysis |

Notebook `02` loads **shipped** groundwater CSVs from `data/processed/groundwater/` by default. The long IGRAC reprocessing cell remains commented for optional use with `data/raw/groundwater/`.

### 4. Outputs

Figures and tables are written under `outputs/` (created when you run the notebooks).

## Citation

See [`CITATION.cff`](CITATION.cff). Please also cite the associated manuscript when available.

## License

MIT — see [`LICENSE`](LICENSE).

## Notes for developers

- Internal handoff notes: [`docs/`](docs/).
- Large rasters and Zarr stores are **not** stored in git; they are produced by notebook `01` / `03`.
