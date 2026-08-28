"""Build 0825 analysis panels: hydro (all Q years at paired gauges) + chemistry/load."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA, NOTES
from regions import annotate_panel
from utils import site_key


def main() -> None:
    raw = pd.read_parquet(DATA / "panel_us_paired_raw.parquet")
    decomp = pd.read_parquet(DATA / "usgs_flow_decomp.parquet")
    gpcc = pd.read_parquet(DATA / "gpcc_cell_year.parquet")
    sites = pd.read_parquet(DATA / "sites_master.parquet")
    seasons = pd.read_parquet(DATA / "usgs_q_seasons.parquet")
    monthly_c = pd.read_parquet(DATA / "wq_us_month.parquet")
    monthly_q = pd.read_parquet(DATA / "usgs_monthly_q.parquet")

    for df in (raw, sites, monthly_c):
        if "site_id" in df.columns:
            df["site_id"] = df["site_id"].astype(str).str.strip()
            df["site_country"] = df["site_country"].astype(str).str.strip()
            df["source"] = df["source"].astype(str).str.strip()

    decomp["usgs_site_no"] = decomp["usgs_site_no"].astype(str)
    monthly_q["usgs_site_no"] = monthly_q["usgs_site_no"].astype(str)
    seasons["usgs_site_no"] = seasons["usgs_site_no"].astype(str)
    raw["usgs_site_no"] = raw["usgs_site_no"].astype(str)

    # Hydro sample: unique gauges in the paired WQ set, all complete Q years
    gauges = (
        raw.groupby("usgs_site_no", as_index=False)
        .agg(
            lat=("lat", "median"),
            lon=("lon", "median"),
            area=("area", "median"),
            site_country=("site_country", "first"),
            site_id=("site_id", "first"),
            source=("source", "first"),
            region=("region", "first"),
        )
    )
    hydro = decomp.merge(gauges, on="usgs_site_no", how="inner")
    hydro["lat1"] = hydro["lat"].round(0)
    hydro["lon1"] = ((hydro["lon"] + 180) % 360 - 180).round(0)
    hydro = hydro.merge(gpcc, on=["lat1", "lon1", "year"], how="left")
    hydro = hydro.merge(
        sites.drop(columns=["lat", "lon", "area", "region"], errors="ignore"),
        on=["site_country", "site_id", "source"],
        how="left",
    )
    hydro["site_key"] = hydro["usgs_site_no"].astype(str)
    hydro["RP"] = hydro["R"] / hydro["P"].where(hydro["P"] > 0)
    hydro["logR"] = np.log(hydro["R"].where(hydro["R"] > 0))
    hydro["logP"] = np.log(hydro["P"].clip(lower=1))
    hydro["logRP"] = np.log(hydro["RP"].where(hydro["RP"] > 0))
    hydro["logR_dry"] = np.log(hydro["R_dry"].where(hydro["R_dry"] > 0))
    hydro["logR_wet"] = np.log(hydro["R_wet"].where(hydro["R_wet"] > 0))
    hydro["nwet_frac"] = hydro["nwet"] / hydro["n_days"].where(hydro["n_days"] > 0)
    hydro["log_nwet"] = np.log(hydro["nwet"].where(hydro["nwet"] > 0))
    hydro = annotate_panel(hydro)
    hydro.to_parquet(DATA / "panel_hydro.parquet", index=False)

    # Chemistry / load: paired site-years with C and decomposed R
    chem = raw.copy()
    drop_old = [c for c in ["R", "n_q", "q_mean"] if c in chem.columns]
    chem = chem.drop(columns=drop_old)
    chem = chem.merge(
        decomp[["usgs_site_no", "year", "n_q", "R", "q_mean", "R_dry", "R_wet", "n_dry", "n_wet"]],
        on=["usgs_site_no", "year"],
        how="left",
    )
    chem["site_key"] = site_key(chem, with_param=True)
    chem["site_key_noparam"] = site_key(chem, with_param=False)
    chem["RP"] = chem["R"] / chem["P"].where(chem["P"] > 0)
    chem["logC"] = np.log(chem["c"].clip(lower=1e-6))
    chem["logC_low"] = np.log(chem["c_low"].clip(lower=1e-6))
    chem["logC_high"] = np.log(chem["c_high"].clip(lower=1e-6))
    chem["logR"] = np.log(chem["R"].where(chem["R"] > 0))
    chem["logP"] = np.log(chem["P"].clip(lower=1))
    chem["logRP"] = np.log(chem["RP"].where(chem["RP"] > 0))
    chem["logR_dry"] = np.log(chem["R_dry"].where(chem["R_dry"] > 0))
    chem["logR_wet"] = np.log(chem["R_wet"].where(chem["R_wet"] > 0))
    chem["load"] = chem["c"] * chem["R"]
    chem["load_dry"] = chem["c_low"] * chem["R_dry"]
    chem["load_wet"] = chem["c_high"] * chem["R_wet"]
    chem["logLoad"] = np.log(chem["load"].where(chem["load"] > 0))
    chem["logLoad_dry"] = np.log(chem["load_dry"].where(chem["load_dry"] > 0))
    chem["logLoad_wet"] = np.log(chem["load_wet"].where(chem["load_wet"] > 0))
    tot_season = chem["load_dry"].fillna(0) + chem["load_wet"].fillna(0)
    chem["share_wet_load"] = np.where(tot_season > 0, chem["load_wet"] / tot_season, np.nan)
    chem["share_dry_load"] = np.where(tot_season > 0, chem["load_dry"] / tot_season, np.nan)
    chem["nwet_frac"] = chem["nwet"] / chem["n_days"].where(chem["n_days"] > 0)
    chem["log_nwet"] = np.log(chem["nwet"].where(chem["nwet"] > 0))
    if "crop_pc" in chem.columns:
        med = chem.drop_duplicates("site_key_noparam")["crop_pc"].median()
        chem["crop_hi"] = chem["crop_pc"] >= med
    chem = annotate_panel(chem)
    chem.to_parquet(DATA / "panel_us_paired.parquet", index=False)

    # Monthly C-Q for dry-season phosphorus (and nitrogen)
    for df in (monthly_c,):
        df["lat1"] = df["lat"].round(0)
        df["lon1"] = ((df["lon"] + 180) % 360 - 180).round(0)
    keys = raw[["site_country", "site_id", "source", "usgs_site_no", "region", "crop_pc", "urb_pc", "has_dam", "area"]].drop_duplicates(
        ["site_country", "site_id", "source"]
    )
    mq = monthly_c.merge(keys, on=["site_country", "site_id", "source"], how="inner")
    mq = mq.merge(seasons, on="usgs_site_no", how="left")
    mq = mq.merge(monthly_q, on=["usgs_site_no", "year", "month"], how="left")
    dry_ok = mq["dry_m1"].notna()
    mo = mq["month"].to_numpy()
    dry_set = mq[["dry_m1", "dry_m2", "dry_m3"]].to_numpy()
    wet_set = mq[["wet_m1", "wet_m2", "wet_m3"]].to_numpy()
    mq["is_dry"] = dry_ok & ((mo == dry_set[:, 0]) | (mo == dry_set[:, 1]) | (mo == dry_set[:, 2]))
    mq["is_wet"] = dry_ok & ((mo == wet_set[:, 0]) | (mo == wet_set[:, 1]) | (mo == wet_set[:, 2]))
    mq["logC"] = np.log(mq["c"].clip(lower=1e-6))
    mq["logQ"] = np.log(mq["q_mean"].where(mq["q_mean"] > 0))
    mq["site_key"] = site_key(mq, with_param=True)
    mq = mq.loc[mq["q_mean"].notna() & mq["c"].notna()]
    mq = annotate_panel(mq)
    mq.to_parquet(DATA / "panel_cq_month.parquet", index=False)

    summary = {
        "hydro_years": int(len(hydro)),
        "hydro_sites": int(hydro["usgs_site_no"].nunique()),
        "chem_rows": int(len(chem)),
        "chem_series": int(chem["site_key"].nunique()),
        "cq_months": int(len(mq)),
        "R_dry_cov": float(chem["R_dry"].notna().mean()),
        "c_low_cov": float(chem["c_low"].notna().mean()),
    }
    (NOTES / "panel_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
