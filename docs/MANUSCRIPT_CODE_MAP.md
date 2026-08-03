# Manuscript ↔ Code Map

**Purpose:** Relate *Hassan & Sultan (2026 draft)* manuscript + supporting information to the publication code in this folder, so each paper figure can be regenerated from `notebooks/03_grace_arid_analysis.ipynb` + the three Python modules.

**Sources reviewed:**

- `2026_Manuscript_v2_07_17_26HS.pdf` (27 pages)
- `2026_Supp_v2.pdf` (16 pages)

**Code roots:**

| Path | Role |
|------|------|
| [`notebooks/03_grace_arid_analysis.ipynb`](notebooks/03_grace_arid_analysis.ipynb) | End-to-end analysis notebook |
| [`src/grace_analysis_utils.py`](src/grace_analysis_utils.py) | GRACE/precip processing, aridity & correlation maps |
| [`grace_analysis_pixel.py`](grace_analysis_pixel.py) | Event-based EPE → storage response + RE maps/scatters |
| [`gw_preprocess.py`](gw_preprocess.py) | Well QC, GRACE–GWL correlation, well maps |
| [`SWS/`](SWS/) | Lake/reservoir std-ratio overlays (Figure S11 inputs) |

---

## 1. Paper in one page

**Title:** A global, event-based assessment of aquifer response to extreme precipitation in arid regions  

**Three questions:**

1. Does GRACE **TWSA** agree better with shallow-well **GWLA** than **GWSA** obtained by subtracting GLDAS SM + runoff + SWE?
2. Do TWSA **residuals** (after harmonic decomposition) track precipitation-driven storage at grid scale?
3. Where, and with what **recharge efficiency (RE)**, do extreme precipitation events (EPEs) produce GRACE-detectable storage responses?

**Key results (as written):**

- LSM subtraction **degrades** well agreement (median ρ 0.56 → 0.23 for residual optimal-lag TWSA vs GWSA).
- EPEs produce a detectable response in **~75%** of arid grid cells.
- Mean RE ≈ **6.7%** (Table S2; also abstract).
- Event framework applied on **TWSA** (CSR-M, JPL-M, GSFC-M) **without** LSM subtraction; GWS language in the paper means *storage response attributed to recharge*, not GLDAS-derived GWSA.

**Study domain:** AI ≤ 0.2 (Zomer v3), nine named arid domains (Figure 1), analysis period **2002–2025**, GRACE gap years **2017–2018** excluded from GRACE pairing.

---

## 2. Methods ↔ code (must-match parameters)

### 2.1 Data (Section 2.2)

| Paper | Code |
|-------|------|
| CSR / JPL / GSFC mascons → coarsened to **1°** | `process_grace_data(...)` (notebook cell 8) |
| GLDAS CLSM/Noah/VIC SM, runoff, SWE → ensemble → GWSA | `process_predictor_fine` + ensemble in cells 10–11 → `grace_*_gw` |
| IMERG-Final monthly precip | `xr.open_zarr(precip_path)` + `process_predictor_fine` (cell 12) |
| Aridity AI ≤ 0.2 polygons + Domain labels | `arid_areas_path`, `plot_aridity_raster` (cells 5–7) |
| IGRAC shallow wells (depth < 50 m) | `gw_preprocess` preprocess / load CSVs (cells 29–31) |

### 2.2 Well–GRACE correlation (Section 2.3.1; Text S1–S2)

| Paper | Code |
|-------|------|
| Monthly GWLA; baseline 2004–2009 | QC / anomaly steps in `gw_preprocess` |
| Harmonic model Eq. 3 → residuals | Decomposition inside `correlate_wells_with_grace` |
| Spearman ρ; lags 0–36 mo; max ρ = optimal lag | `method='spearman'`, `max_lag_months=36` (cell 33) |
| Haversine nearest 1° centroid | Well–grid assignment in `correlate_wells_with_grace` |
| Min overlap ~85–120 months | Notebook: `min_common_dates=120` |
| Pumping / abrupt-drop correction (Text S1 IQR fence) | `correct_pumping=True` (cell 33) |
| Stuck sensor: 60-mo rolling std **< 0.1 m** | **See flags:** code constant is `0.05` |

### 2.3 TWSA residual vs cumulative precip anomaly (Section 2.3.2; Eqs. 4–5)

| Paper | Code |
|-------|------|
| Residual R(t) vs cumulative monthly precip anomaly C(t) | `calculate_grace_precip_correlation_per_pixel(..., precip_mode='cumsum', use_residual=True, corr_method='spearman')` (cell 14) |
| Gap months advance C(t) but drop from ρ pairing | Implemented in correlation helper; notebook `exclude_years` / gap handling |
| Parallel non-residual comparison → Figure S6 | Same function with `use_residual=False` (not currently a dedicated notebook cell) |

### 2.4 Event-based EPE → storage response (Section 2.3.3; Figure S3; Eqs. 6–10)

| Paper | Code (`analyze_grace_response_by_pixel`) |
|-------|------|
| Input = **three TWSA solutions**, no LSM subtraction | Cell 19: `grace_solutions_list = [grace_csr, grace_jpl, grace_gsfc]` |
| Precip coarsened to GRACE 1° | Internal `_coarsen_precipitation_to_grace` |
| Local **95th** percentile extremes | `threshold_percentile=0.95` |
| Floor **20 mm** | `precip_floor=2` **cm** after `rainfall_all/10` |
| Cluster if Δt ≤ **12 months** | Default clustering window |
| Cluster precip = sum of **all months** in [t_start, t_end] | `precip_ts.loc[start:end].sum()` |
| Pre/post means over up to 12 months; ≥ **6** valid months each side | `_calculate_pixel_response` |
| ΔS = G_post − G_pre per solution; mean & std across solutions | Eqs. 8–9 |
| Valid if ΔS > **1.5 cm** and ΔS > σ_k | `grace_threshold=1.5` |
| Cumulative maps: sum valid cluster precip / ΔS; σ_total = √Σσ² | `total_precip`, `valid_response_sum`, `valid_std_sum` |
| Decompose GRACE (Eq. 3) | `decompose_grace=True` |
| Exclude 2017–2018 | `exclude_years=[2017, 2018]` |
| RE = 100 × cumulative ΔS / cumulative EPE | Efficiency panel in `plot_pixel_analysis_maps` |

---

## 3. Figure-by-figure map

### Main text

| Figure | What the paper shows | Notebook / function | Match notes |
|------|----------------------|---------------------|-------------|
| **Fig 1** | AI map, AI≤0.2, nine numbered domains | Cell 7 → `plot_aridity_raster(..., vmax=0.2, aoi_geometry=aoi_geom, aoi_boundary_col="Domain")` | Good. Domain rename “North America”→“SW North America” is cell 25 (used later). |
| **Fig 2** | TWSA vs GWSA Spearman ρ vs wells: anomaly/residual × zero/optimal lag | Cell 42 → `plot_correlation_distributions(...)` | Good conceptual match. Confirm exported panel order/labels match final art. |
| **Fig 3** | Residual TWSA–CPA Spearman map + ρ histogram (median 0.36) | Cells 14–16 → `calculate_grace_precip_correlation_per_pixel` + `plot_grace_correlation_map` + `summarize_grace_correlation_outputs` | Good. Cell 14 uses `grace_mean` (TWSA), `use_residual=True`, `precip_mode='cumsum'`. |
| **Fig 4a–c** | Cumulative EPE; cumulative GRACE response; relative uncertainty % | Cell 21 → `plot_pixel_analysis_maps(..., layout='precip_grace_uncertainty', uncertainty_display='relative_pct')` | Good. Paper single 3-panel figure; notebook one call. |
| **Fig 5** | RE map (%) | Cell 23 → `layout='efficiency'`, classified 0–15% | Good for base RE map. Lake overlays belong to **S11** (commented). |
| **Fig 6** | Global cumulative EPE vs response, unc ≤50%, n=2,046, linear fit | Needs `plot_pixel_epe_grace_relationship(..., aggregation='pixel', domain='all', vmax_uncertainty_pct=50, grace_threshold=1.5, epe_input_unit='cm')` | **Gap:** no dedicated cell with `domain='all'` + `aggregation='pixel'`. |
| **Fig 7** | Nine-domain pixel-scale scatters, unc ≤50% | Same function with `domain='regions'`, `aggregation='pixel'`, `domain_gdf=arid_areas_sorted` | **Gap:** cell 26 is **`aggregation='event'`** (that is S9), not pixel. |

### Supporting information

| Figure / Table | Paper content | Notebook / function | Match notes |
|---------------|---------------|---------------------|-------------|
| **Fig S1** | Filtered shallow well locations on arid mask | Cell 31 → `plot_all_well_locations(...)` | Good. |
| **Fig S2** | SA well–grid Haversine links | Cell 36 → `plot_all_well_locations(..., grace_mean=..., country='ZAF')` | Good. |
| **Fig S3** | Methods flowchart (event framework) | Not generated by code | Manual / graphic asset. |
| **Fig S4** | Optimal-lag ρ by country (TWSA vs GWLA), anomaly & residual | Cell 39 → `plot_correlation_distributions_by_country(...)` | Good if `variable` covers TWS panels as in paper. |
| **Fig S5** | SA maps of **TWSA–GWLA** median ρ and lag | Cell 40 → `plot_grace_gwl_correlation_lag_maps(...)` | **Flag:** function default `variable='GWS'`; cell comment says GWS. Paper is **TWSA**. Set `variable='TWS'`. |
| **Fig S6** | Same as Fig 3 but **without** residual | Correlation stack with `use_residual=False` | **Gap:** no notebook cell. |
| **Fig S7** | Histograms of EPE, response, unc%, RE (n=2,347); P90/95/99 | Cell 24 → `plot_pixel_results_distribution_diagnostics(...)` | Good. Cell 22 cites 2347/3123. |
| **Fig S8** | Robustness: pixel vs event; unc ≤50% and ≤20% | `plot_pixel_epe_grace_relationship` / event diagnostics at two caps | Partial: pieces exist; not one four-panel cell. |
| **Fig S9** | Event-scale domain scatters, unc ≤50% | Cell 26 → `aggregation='event'`, `domain='regions'`, `vmax_uncertainty_pct=50` | Good. |
| **Fig S10** | All clusters before validity filter | `plot_event_cluster_distribution_and_relationship` (cell 20) and/or events DF without `is_valid` filter | Confirm cell 20 includes invalid/negative clusters as in S10. |
| **Fig S11** | RE map + lake/reservoir std-ratio points (≥20% of GRACE std) | Cell 23 + `lake_points=...`, `lake_min_pct=20` | **Flag:** `lake_points` path is **commented out**. Overlay comes from SWS shapefile export. |
| **Table S1** | ρ class counts residual vs TWSA | `summarize_grace_correlation_outputs` / class stats from cell 15–16 | Good if thresholds [0.5, 0.2] match. |
| **Table S2** | Summary stats n=2347 | Diagnostics summary from cell 24 / pixel catalog | Good. |

---

## 4. Notebook reading order (paper narrative)

Suggested section markers for a future “figure-ready” notebook reorganization:

1. **Setup & paths** → cells 2–5  
2. **Fig 1** aridity → cells 6–7  
3. **GRACE TWSA + precip** → cells 8, 12  
4. **GLDAS / GWSA construction** (for well comparison only) → cells 10–11  
5. **Figs 2, S1–S2, S4–S5** wells → cells 29–43  
6. **Figs 3, S6, Table S1** precip coupling → cells 14–17 (+ missing S6 cell)  
7. **Event framework + Figs 4–7, S7–S11** → cells 19–27 (+ missing Fig 6/7 pixel cells; enable S11 lakes)

---

## 5. Review flags (code vs manuscript)

Priority order for fixing before claiming figure reproducibility:

### High (wrong product or missing paper figure)

1. **Fig S5 uses GWS by default, paper shows TWSA**  
   Cell 40 calls `plot_grace_gwl_correlation_lag_maps` without `variable='TWS'` (default is `'GWS'`). Fix: pass `variable='TWS'` (and keep residual GDF if paper panel is residual TWSA).

2. **Fig 6 not wired as a notebook cell**  
   Need pixel-scale global scatter with `vmax_uncertainty_pct=50` (filter is real—not only a colorbar).

3. **Fig 7 not wired; cell 26 is Fig S9**  
   Cell 26 `aggregation='event'` → S9. Fig 7 needs `aggregation='pixel'` + nine domains.

4. **Fig S11 lake overlay disabled**  
   `lake_points=...` commented in cell 23; without it you only get Fig 5, not S11.

5. **Fig S6 (non-residual CPA map) missing**  
   No cell with `use_residual=False`.

### Medium (method text vs implementation)

6. **Stuck-sensor threshold inconsistency**  
   Paper / Text S1: rolling std **< 0.1 m**.  
   Code: `QC_STUCK_STD_THRESHOLD = 0.05`, docstring says 0.10, exclusion reason string mentions 0.10, print mentions 0.05. Align constant + docs + manuscript.

7. **Terminology risk for reviewers**  
   Paper event framework uses **TWSA** but labels maps “GRACE GWS response.” Notebook correctly feeds TWSA into `analyze_grace_response_by_pixel`. Keep comments explicit: *response attributed to recharge, not GLDAS GWSA*.

8. **Fig S10 validity**  
   Confirm cell 20 plots the **pre-validity** population (n≈18,463 in caption). If it only plots `is_valid=True`, it is not S10.

### Low (reproducibility / packaging)

9. **Absolute paths** throughout notebook (`/mnt/d/grace_ds/...`) — fine locally; bad for GitHub/Zenodo.  
10. **Figure export paths commented** — readers cannot one-click regenerate final JPEG/TIFF names.  
11. **Hard-coded 2347/3123** print (cell 22) — should come from `pixel_counts` after the run.  
12. **Interactive Folium map (cell 17)** — useful QC, not a paper figure.  
13. **Fig S3 flowchart** — not code-generated (OK if stated).  
14. **SWS lake shapefile** lives under `SWS/` pipeline — document that Fig S11 depends on that preprocess, not only the main notebook.

---

## 6. Alignment that already looks solid

- Event thresholds: 95th, 20 mm floor, 1.5 cm validity, 12-month cluster/windows, ≥6 months, 2017–2018 gap, residual decomposition.  
- Three-mascon inter-solution uncertainty + quadrature accumulation.  
- RE definition and classified RE colorbar (0, 5, 10, 15, extend).  
- Well lag convention (positive = well lags GRACE), Spearman, shallow ≤50 m.  
- Fig 1 / 3 / 4 / 5 / S1 / S2 / S7 / S9 parameter intent largely matches current cells.  
- Main scientific choice: event analysis on **TWSA**, LSM subtraction only for the well-comparison argument — matches notebook structure.

---

## 7. Recommended next step (notebook strategy for reviewers)

Without changing science, add a thin “Paper figures” section (or markdown tags) that:

1. Names each cell `# Figure X / Figure SX` in the first comment.  
2. Adds missing cells for **Fig 6**, **Fig 7**, **Fig S6**, and an explicit **Fig S5** with `variable='TWS'`.  
3. Uncomments / parameterizes **Fig S11** `lake_points`.  
4. Writes outputs to a fixed `paper_figures/` tree with manuscript filenames.  
5. Fixes stuck-sensor threshold to **0.1 m** (or updates the manuscript if 0.05 was intentional).

---

*Generated 2026-07-17 from manuscript PDFs + `notebooks/03_grace_arid_analysis.ipynb` / module review. Re-check numbers (n=2046, 2347, medians) against a fresh run before camera-ready.*
