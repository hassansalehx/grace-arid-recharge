# Context Summary — Handoff for New Chats

**Purpose:** Capture objectives, code locations, and decisions so a fresh session can continue without long chat history.  
**Primary research track:** *Groundwater recharge from extreme precipitation (EPE) in arid regions* — manuscript aimed at **Water Resources Research (WRR)**.  
**Folder:** `/mnt/d/codes/python/grace_ds/github` — curated publication copy (notebooks + modules pruned for GitHub / Zenodo).  
**Secondary tracks (same folder):** GRACE vs in-situ wells (`gw_preprocess.py`); surface-water storage / lakes (`SWS/`).

---

## 1. Research objectives (WRR paper)

- **Science question:** Link **extreme precipitation** in **arid regions** to **terrestrial water / groundwater-related storage** signals from **GRACE/GRACE-FO**, using a **pixel-level, multi-solution** framework.
- **Products:** Maps and statistics of cumulative EPE forcing, cumulative GRACE-based groundwater storage (GWS) response to valid events, uncertainty / relative uncertainty, and recharge-efficiency-style metrics where appropriate.
- **Geography:** Analysis is **geometry-driven** (polygons for arid regions or aquifer masks). The same code paths work for **arid boundaries** as for named aquifers; **clipping must be explicit** (see §4).
- **Writing status:** Paper drafting and **code review** are happening in parallel; methodology text should stay aligned with `grace_analysis_pixel.py` and `src/grace_analysis_utils.py`.

---

## 2. Publication folder layout

| Path | Role |
|------|------|
| `notebooks/02_grace_arid_analysis.ipynb` | **Authoritative** main notebook for the paper (≈45 cells / 35 code). Curated subset of the older working notebook; do **not** assume it still matches `Cursor/GRACE_ds_arid_subbasins-analysis_Optimized.ipynb`. |
| `grace_analysis_pixel.py` | Pixel EPE → GRACE response and paper map/diagnostic plots. |
| `src/grace_analysis_utils.py` | Shared utilities: GRACE/precip processing, aridity maps, correlation maps, subbasin summaries, layout constants (`_MAP_*`). |
| `grace_timeseries_clustering.py` | Slimmed clustering module; notebook uses `advanced_time_series_clustering` only. |
| `gw_preprocess.py` | Well preprocessing and GRACE–GWL correlation / maps (see §10). |
| `notebooks/03_sws_arid_analysis.ipynb` | Surface-water / lake vs GRACE analysis notebook. |
| `src/sws_analysis_utils.py` | SWS download, preprocess, and plotting helpers used by that notebook. |
| `SWS/SWS_REFERENCE.md` | SWS path / data-layout notes. |
| `CONTEXT_SUMMARY.md` | This handoff file. |

**Removed from this folder (not part of the publication package):**

- `daily_rainfall_intensity.ipynb` (out of scope)
- `clustering_example.py`, `notebook_cell_example.py` (broken / unused demos)
- `plot_aquifer_pixel_scatter.py` (scratch duplicate)
- Dead code pruned from the five modules above (~9k+ lines of unused functions), including CRH significance-test helpers, LSM/soil-moisture drivers, legacy clustering/PCA helpers, MAD/depth-class well plots, and unused SWS download/plot helpers.

**Working copy vs publication copy:** Development may still live under `grace_ds/Cursor/` (full notebook + unpruned modules). Treat **`github/`** as the curated source of truth for what ships.

---

## 3. Notebook-facing APIs (keep list)

Functions imported by the two publication notebooks (direct imports). Helpers called only inside these modules are kept transitively and are not listed here.

### `notebooks/02_grace_arid_analysis.ipynb`

- **`grace_analysis_pixel`:** `analyze_grace_response_by_pixel`, `analyze_grace_response_by_aquifer_pixel`, `plot_pixel_analysis_maps`, `plot_pixel_epe_grace_relationship`, `plot_pixel_results_distribution_diagnostics`, `plot_event_cluster_distribution_and_relationship`
- **`grace_analysis_utils`:** `process_grace_data`, `process_predictor_fine`, `plot_aridity_raster`, `plot_arid_watersheds`, `plot_multiple_maps_with_balanced_colorbar`, `calculate_grace_precip_correlation_per_pixel`, `plot_grace_correlation_map`, `summarize_grace_correlation_outputs`, `plot_grace_precip_correlation_interactive_map`, `plot_grace_precip_correlation_temporal`, `plot_grace_precip_extremes`, `plot_timeseries_with_precip`, `plot_subbasin_time_series_all`, `subbasin_trend_analysis`, `add_average_annual_precipitation`, `sort_and_index_by_area`, `analyze_results_comprehensive`, `summarize_results_dict`
- **`grace_timeseries_clustering`:** `advanced_time_series_clustering`
- **`gw_preprocess`:** `preprocess_all_countries`, `print_well_info`, `save_groundwater_data`, `plot_all_well_locations`, `correlate_wells_with_grace`, `plot_grace_well_timeseries_comparison`, `plot_correlation_distributions`, `plot_correlation_distributions_by_country`, `plot_grace_gwl_correlation_lag_maps`, `plot_optimal_lag_histograms`

### `notebooks/03_sws_arid_analysis.ipynb`

- **`sws_analysis_utils`:** `load_sws_config`, `load_hydrolakes_polygons`, `resolve_precip_path`, `run_download_all`, `build_glolakes_arid_catalog`, `build_glolakes_swsa_batch`, `build_grace_time_range`, `process_grace_mean`, `process_precip_on_grace_grid`, `plot_example_lake_volumes`, `analyze_lake_grace_comparisons`, `plot_lake_grace_precip_comparison`, `load_arid_domains`, `plot_lake_std_ratio_map`, `plot_lake_std_ratio_maps_by_domain`, `export_lake_std_ratio_shapefile`, `plot_filtered_lake_std_comparisons`

---

## 4. Core method: `analyze_grace_response_by_pixel`

**Intent:** For each grid cell, (1) define pixel-wise extreme precipitation, (2) cluster events in time, (3) quantify multi-GRACE-solution **before/after** GWS change per cluster, (4) keep events that pass validity rules, (5) aggregate to maps and an event table.

**Typical inputs (notebook):**

- `grace_solutions_list`: e.g. `grace_csr_gw`, `grace_jpl_gw`, `grace_gsfc_gw` where  
  `grace_*_gw = grace_* - sm_ensemble_mean - q_ensemble_mean` (GLDAS CLSM/Noah/VIC ensemble means).
- `precip_data`: monthly precipitation (e.g. GPM), aligned in time after processing.
- `threshold_percentile` (e.g. 0.95), `precip_floor`, `exclude_years` (e.g. `[2017, 2018]` GRACE gap), `decompose_grace` (optional OLS trend + annual + semi-annual → **residual**), `grace_threshold` (minimum response, e.g. cm EWT), `n_jobs` (joblib parallel over pixels).

**Steps (code order):**

1. Per-solution GRACE: CRS, dims, drop excluded years; optional clip to AOI.
2. **Clipping:** If `aquifer_gdf` **and** `aquifer_ids` are set → AOI from filtered geometries. Else **`aoi_geometry`** must be passed for clip. Passing only `aquifer_gdf` without IDs does **not** use that GDF for clipping unless `aoi_geometry` is set.
3. Align solutions in space/time; mean stack for coarsening reference.
4. **Precipitation → GRACE grid:** `_coarsen_precipitation_to_grace(..., method='coarsen')` — **block average** to match GRACE step, then **nearest** to exact GRACE lon/lat (not bilinear `interp`).
5. **Extreme mask (vectorized):** `quantile` along `time` + floor `max(q, precip_floor)`; extremes where precip exceeds both threshold and floor.
6. **Per pixel (parallel with joblib):** build time series, optional decomposition, cluster extreme dates (`window_days=365`), `_calculate_pixel_response`.

**`_calculate_pixel_response` (per cluster):**

- Windows: 12 months before cluster start through start; end through 12 months after end; min 6 valid months each side.
- Per solution \(k\): \(\delta_k = \bar{G}_{\text{after},k} - \bar{G}_{\text{before},k}\).
- `diff_mean` = mean(\(\delta_k\)), `diff_std` = sample std of \(\delta_k\) (`ddof=1`), or 0 if only one solution.
- **Valid** if `diff_mean > grace_threshold` **and** `diff_mean > diff_std`.
- **Pixel totals:** sum of `diff_mean` over valid clusters; **uncertainty field** `valid_std_sum` = \(\sqrt{\sum \texttt{diff\_std}^2}\) over valid clusters (quadrature; interprets inter-solution spread as uncertainty-like).

**Returns:** `total_precip`, `valid_response_sum`, `valid_std_sum` (2D), `grace_mean`, `grace_std` (per-time std across solutions — **different** from `valid_std_sum`), `precip_coarsened`, `events_dataframe`.

**Decomposition:** `decompose_grace_sin_cosin` in `src/grace_analysis_utils.py` — OLS with linear trend + annual + semi-annual harmonics on monthly index; residual used when `decompose_grace=True`.

---

## 5. GRACE grid and preprocessing

- `process_grace_data` in `src/grace_analysis_utils.py` **coarsens native mascon grids to ~1°×1°** (land-mask-aware) before analysis in this workflow.
- Precipitation is harmonized to that **1°** grid via **coarsen**, not bilinear interpolation to GRACE nodes.

---

## 6. Plotting: `plot_pixel_analysis_maps`

**File:** `grace_analysis_pixel.py`.

**Panels:** Default four-panel layout — (1) cumulative EPEs, (2) cumulative GRACE GWS response, (3) recharge efficiency–style ratio, (4) uncertainty. Optional `layout` for a single efficiency map or a three-panel strip.

**Notable options (publication-facing):**

- `ncols`, `cbar_orientation` (`vertical` / `horizontal`).
- **Uncertainty:** `uncertainty_display='absolute'` (cm; **shared color scale with panel 2**) vs `'relative_pct'` → \((\texttt{valid\_std\_sum} / \texttt{valid\_response\_sum}) \times 100\) **only where both pixel values > 0**; colorbar capped by `uncertainty_pct_max` (default 100).
- Efficiency colorbar: class boundaries (e.g. 0, 5, 10, 15) with extend for values above the last bound.
- **Lake / reservoir overlay:** `lake_points` (GeoDataFrame), `lake_value_col` (e.g. `std_pct_gr`), `lake_min_pct` filter, auto marker size from grid cell size, legend title **Reservoir/Lake (%)**.
- **GeoTIFF export:** `save_raster=True` with `saved_rasters_path` and `raster_tags`; NaNs preserved as NaN (`nodata=np.nan`), orientation corrected for north-up GeoTIFF.
- `return_pixel_catalog` to optionally return the per-pixel cumulative metrics table (default avoids dumping a huge DataFrame in the notebook).

**Layout:** Shared `_MAP_*` constants with `grace_analysis_utils` maps — tight colorbars (`pad`, `fraction`, `shrink`), `wspace=0`, small `hspace`, `tight_layout(pad=0.2)`.

---

## 7. Optimizations implemented (pixel pipeline)

| Item | Detail |
|------|--------|
| Precip to GRACE | `method='coarsen'` (block mean + nearest snap), not `interp` linear. |
| Extreme thresholds | Full-field `xr.quantile` + mask; no per-pixel `nanpercentile` in loop. |
| Progress | tqdm over **valid** pixel indices only (`np.argwhere(valid_pixel_mask)`). |
| Parallelism | `joblib.Parallel` + `delayed(_process_single_pixel)`; `n_jobs=-1` default parameter. **`xr.apply_ufunc` does not parallelize** unless paired with Dask; not used for the irregular cluster/response logic. |

---

## 8. Methodology ↔ code (paper checklist)

When describing WRR methods, tie claims to:

- GWS proxy: GRACE TWS minus GLDAS SM and runoff ensemble means; three official solutions (CSR RL06.3, JPL RL06.3, GSFC).
- EPE: pixel-wise quantile + floor; clustering 365-day window; validity uses inter-solution mean vs spread and `grace_threshold`.
- Uncertainty in maps: inter-solution **spread** aggregated in quadrature — **not** formal RL06 error bars unless you add them separately.
- Relative uncertainty map: **optional** in plotting only; interpret as heuristic signal-to-spread index.

---

## 9. Pitfalls

- **AOI:** Use `aoi_geometry=` for arid polygons, or `aquifer_gdf` + `aquifer_ids` together. Don’t assume `aquifer_gdf` alone clips.
- **Notebook naming:** Parameters named `aquifer_*` are historical; geometry can be any region.
- **`grace_std` in outputs:** Time-varying std across solutions ≠ `valid_std_sum` (event-aggregated quadrature).
- **Publication vs Cursor notebook:** Features present only in the older Cursor notebook (e.g. CRH spatially corrected Spearman test, LSM/soil-moisture cells) were **removed** from this folder because `notebooks/02_grace_arid_analysis.ipynb` does not call them. Do not reintroduce them unless the publication notebook is updated first.

---

## 10. Secondary context: GRACE–well analysis (`gw_preprocess.py`)

Still used by `notebooks/02_grace_arid_analysis.ipynb` for in-situ groundwater correlation:

- **Key kept functions:** `preprocess_groundwater_data`, `preprocess_all_countries`, `classify_well_depths`, `correlate_wells_with_grace`, `plot_grace_well_timeseries_comparison`, `plot_correlation_distributions`, `plot_correlation_distributions_by_country`, `plot_all_well_locations`, `plot_grace_gwl_correlation_lag_maps`, `plot_optimal_lag_histograms`, `print_well_info`, `save_groundwater_data`.
- **Depth:** Shallow ≤50 m, Deep >50 m from `avg_depth_m` (where classification is used).
- **Correlations:** Best lag = **maximum positive** `corr_lag(t)`; lag-0 and best-lag both stored.
- **Removed from this package:** pumping correction, depth-vs-correlation/lag plots, depth-class timeseries comparison, MAD variance suite, well time-series clustering, etc.

*(Open `gw_preprocess.py` for line-accurate signatures.)*

---

## 11. Secondary context: SWS / lakes (`SWS/`)

- Notebook: `notebooks/03_sws_arid_analysis.ipynb`.
- Module: `src/sws_analysis_utils.py` (pruned; unused alternate download/parse/plot helpers removed).
- Typical flow: config → HydroLAKES / GloLakes catalog → SWSA batch → GRACE mean & precip on GRACE grid → lake–GRACE comparisons → std-ratio maps / shapefile export.
- See `SWS/SWS_REFERENCE.md` for data roots and external sources.

---

## 12. File change log (this workstream)

- **Pixel pipeline (`grace_analysis_pixel.py`):** coarsen precip; vectorized thresholds; valid-pixel tqdm; `joblib`; `plot_pixel_analysis_maps` layouts, uncertainty modes, lake overlays, GeoTIFF export with NaN nodata / orientation fix; diagnostics and EPE–GRACE relationship plots. CRH significance-test block and unused aquifer scatter / recharge-efficiency map **removed** for publication.
- **Utils (`src/grace_analysis_utils.py`):** `_MAP_*` layout contract; aridity / correlation / multi-map helpers used by the publication notebook. Trend/LSM/soil-moisture and unused interactive/pixel-timeseries helpers **removed**.
- **Clustering (`grace_timeseries_clustering.py`):** kept only the `advanced_time_series_clustering` closure (8 functions); legacy kmeans/hierarchical/kshape/DBA/PCA-detail helpers **removed**.
- **Wells (`gw_preprocess.py`):** kept notebook-facing preprocess/correlate/plot suite; MAD/depth/pumping/clustering extras **removed**.
- **SWS (`src/sws_analysis_utils.py`):** kept notebook-facing download/catalog/compare/map suite; unused alternate parsers/plots/pipeline summaries **removed**.
- **Repo hygiene:** deleted out-of-scope notebook and demo scripts; cleared `__pycache__` bytecode from the publication folder.

---

## 13. Suggested next steps (publication)

- Replace absolute data paths in notebooks with relative / config-driven paths.
- Add `README.md`, `environment.yml` (or `requirements.txt`), and clear figure/data reproduction notes.
- Initialize git, push to GitHub, then cut a release for Zenodo DOI.
- Clear large notebook outputs before upload if size matters.
- Keep `__pycache__/` and `.env` out of version control (SWS already has a small `.gitignore`).

---

*Last updated: 2026-07-17 — publication folder prune complete; `notebooks/02_grace_arid_analysis.ipynb` + `SWS_arid_analysis.ipynb` are the keep-alive roots. Re-read the modules and notebooks for exact parameter defaults before citing numbers in the paper.*
