"""Cache CONUS paired tables from 0824/0823 without copying their narrative."""
from __future__ import annotations

import json
import shutil

import numpy as np
import pandas as pd

from config import (
    AREA_MAX,
    AREA_MIN,
    CACHE23,
    CACHE24,
    DATA,
    MIN_OBS_YEAR,
    MIN_YEARS,
    NOTES,
    SITES_CSV,
)


def main() -> None:
    src_panel = CACHE24 / "panel_us_site_year.parquet"
    if not src_panel.exists():
        raise FileNotFoundError(f"missing 0824 cache: {src_panel}")

    panel = pd.read_parquet(src_panel)
    panel["area"] = np.where(panel["area"] > 1e6, panel["area"] / 1e6, panel["area"])
    paired = panel.loc[panel["paired"] == True].copy()
    paired = paired.loc[(paired["area"].isna()) | ((paired["area"] >= AREA_MIN) & (paired["area"] <= AREA_MAX))]
    keep = (
        paired.loc[paired["n"] >= MIN_OBS_YEAR]
        .groupby(["site_country", "site_id", "source", "param"], as_index=False)
        .agg(n_years_chk=("year", "nunique"))
    )
    keep = keep.loc[keep["n_years_chk"] >= MIN_YEARS]
    paired = paired.merge(keep, on=["site_country", "site_id", "source", "param"], how="inner")
    paired.to_parquet(DATA / "panel_us_paired_raw.parquet", index=False)

    usgs_ids = sorted({str(x) for x in paired["usgs_site_no"].dropna().unique()})
    (DATA / "paired_usgs_ids.txt").write_text("\n".join(usgs_ids), encoding="utf-8")

    copies = {
        "usgs_flow_year.parquet": CACHE24 / "usgs_flow_year.parquet",
        "usgs_q_seasons.parquet": CACHE24 / "usgs_q_seasons.parquet",
        "gpcc_cell_year.parquet": CACHE24 / "gpcc_cell_year.parquet",
        "sites_master.parquet": CACHE24 / "sites_master.parquet",
        "wq_us_month.parquet": CACHE24 / "wq_us_month.parquet",
        "wq_us_year.parquet": CACHE24 / "wq_us_year.parquet",
        "panel_eu_si.parquet": CACHE24 / "panel_eu_si.parquet",
        "gpcc_cell_drymonths.parquet": CACHE23 / "gpcc_cell_drymonths.parquet",
    }
    for name, src in copies.items():
        dst = DATA / name
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    if SITES_CSV.exists():
        shutil.copy2(SITES_CSV, DATA / "grqa_sites_nutrients.csv")

    summary = {
        "source_panel": str(src_panel),
        "n_paired_rows": int(len(paired)),
        "n_paired_series": int(paired.groupby(["site_country", "site_id", "source", "param"]).ngroups),
        "n_usgs": int(len(usgs_ids)),
        "params": paired["param"].value_counts().to_dict(),
        "regions": paired["region"].value_counts().to_dict(),
        "year0": int(paired["year"].min()),
        "year1": int(paired["year"].max()),
        "note": "0824/0823 tables cached as inputs only; 0825 re-screens and re-estimates.",
    }
    (NOTES / "cache_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
