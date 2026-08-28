# Daily rainfall inequality, runoff, and river nutrients (CONUS)

Code and selected regression tables for a paired chemistry–discharge analysis of the continental United States.

Holding annual precipitation fixed, the daily Gini index of precipitation (GP) is entered as a year-scale timing term. The estimator is site and year two-way fixed effects with standard errors clustered by site. Approximate load is median concentration times runoff (\(C \times R\)), not a flux reconstruction.

## What is in this repository

- `code/` — panel build, two-way FE regressions, robustness, regional contrasts, and figure scripts
- `data/*.csv` — selected coefficient tables (point estimate, SE, 95% CI)
- `data/us_states_20m.geojson` — state polygons used to map NOAA climate-region classes

Site-year panels (Parquet) and raw daily precipitation / discharge archives are not included. Rebuild those locally, then re-run the regression scripts.

## Setup

```bash
python -m pip install -r requirements.txt
```

Optional for maps: `cartopy`, `shapely`.

Point raw archives with an environment variable if you rebuild panels from source:

```bash
set PRECIP_WQ_RAW=D:\path\to\raw
```

Default local raw root (if unset): `O:\PrecipConcentrate_WQ`.

## Reproduce the tables

From `code/`:

```bash
python 04_regress_hydro.py
python 05_regress_conflict.py
python 06_regress_robust.py
python 13_deepen_regime.py
python 16_explain_why.py
```

`01_cache_panel.py` and `03_build_panel.py` expect already-cached site-year files on the local machine.

## Climate-region classes

Gauges are assigned to NOAA NCEI climate regions (Karl and Koss 1984), then aggregated to Northeast, South, Interior, and West. The split is by state, not a latitude–longitude box.

## Citation

Use this repository only as code and tables. Do not treat the CSVs as a journal data product.
