"""Saturation-excess bucket on the same GPCC daily P.

Question: does a one-store bucket recover the observed regional Q10 split
(South lengthens the low-flow tail with GP; Northeast interval includes zero)?

If the regional signs do not match, the bucket does not enter the main figure.
"""
from __future__ import annotations

import gzip
import json

import numpy as np
import pandas as pd

from config import DATA, GPCC_DIR, NOTES, YEAR0, YEAR1
from utils import fe_ols


def load_precip(year: int):
    p = GPCC_DIR / f"full_data_daily_v2022_10_{year}.nc.gz"
    with gzip.open(p, "rb") as gz:
        import xarray as xr

        ds = xr.open_dataset(gz)
        arr = np.asarray(ds["precip"].values)
        lat = np.asarray(ds["lat"].values)
        lon = np.asarray(ds["lon"].values)
        ds.close()
    return arr, lat, lon


def bucket_series(p: np.ndarray, pet_d: float, cap: float) -> np.ndarray:
    p = np.where(np.isfinite(p), np.clip(p, 0, None), 0.0)
    s = 0.0
    r = np.empty(p.size, dtype=float)
    for t, pt in enumerate(p):
        avail = s + pt
        et = pet_d if pet_d < avail else avail
        leftover = avail - et
        if leftover > cap:
            r[t] = leftover - cap
            s = cap
        else:
            r[t] = 0.0
            s = leftover
    return r


def pick_cap(p_years: list[np.ndarray], pet_d: float, r_obs: float) -> float:
    caps = np.array([15.0, 40.0, 80.0, 140.0, 220.0])
    best, best_err = 80.0, 1e18
    for cap in caps:
        r_mean = float(np.mean([bucket_series(p, pet_d, cap).sum() for p in p_years]))
        err = abs(r_mean - r_obs)
        if err < best_err:
            best, best_err = float(cap), err
    return best


def main() -> None:
    out_p = DATA / "panel_bucket.parquet"
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    hydro["usgs_site_no"] = hydro["usgs_site_no"].astype(str).str.zfill(8)
    paired = pd.read_parquet(DATA / "panel_us_paired.parquet", columns=["usgs_site_no", "year", "PET"])
    paired["usgs_site_no"] = paired["usgs_site_no"].astype(str).str.zfill(8)
    pet = paired.groupby(["usgs_site_no", "year"], as_index=False)["PET"].median()
    pet["pet"] = (-pet["PET"]).clip(lower=50.0)
    pet_map = pet.set_index(["usgs_site_no", "year"])["pet"].to_dict()
    site_pet = pet.groupby("usgs_site_no")["pet"].median().to_dict()

    uniq = hydro[["lat1", "lon1"]].dropna().drop_duplicates().reset_index(drop=True)
    arr0, lat_grid, lon_grid = load_precip(YEAR0)
    ilat = np.abs(uniq["lat1"].to_numpy()[:, None] - lat_grid[None, :]).argmin(1)
    ilon = np.abs(uniq["lon1"].to_numpy()[:, None] - lon_grid[None, :]).argmin(1)
    del arr0
    cell_ts: dict[int, np.ndarray] = {}
    for year in range(YEAR0, YEAR1 + 1):
        arr, _, _ = load_precip(year)
        cell_ts[year] = arr[:, ilat, ilon].astype(float)
        print("  bucket P", year, flush=True)

    key_to_j = {(float(a), float(b)): i for i, (a, b) in enumerate(zip(uniq["lat1"], uniq["lon1"]))}
    meta = hydro.groupby("usgs_site_no", as_index=False).agg(
        lat1=("lat1", "first"),
        lon1=("lon1", "first"),
        region=("region", "first"),
        site_key=("site_key", "first"),
        R=("R", "median"),
    )
    rows = []
    for i, rec in enumerate(meta.itertuples(index=False), 1):
        j = key_to_j[(float(rec.lat1), float(rec.lon1))]
        years = list(range(YEAR0, YEAR1 + 1))
        series = [cell_ts[y][:, j] for y in years]
        pet_ann = float(site_pet.get(rec.usgs_site_no, 800.0))
        pet_d = pet_ann / 365.0
        r_obs = float(rec.R) if np.isfinite(rec.R) else 200.0
        cap = pick_cap(series[::4], pet_d, r_obs)  # every 4th year for speed
        # climatology Q10 from all modeled days
        all_r = np.concatenate([bucket_series(p, pet_d, cap) for p in series])
        q10 = float(np.nanpercentile(all_r, 10))
        for year, p in zip(years, series):
            r = bucket_series(p, pet_d, cap)
            n = int(np.isfinite(r).sum())
            if n < 300:
                continue
            rows.append(
                {
                    "usgs_site_no": rec.usgs_site_no,
                    "year": year,
                    "region": rec.region,
                    "site_key": rec.site_key,
                    "cap": cap,
                    "R_mod": float(r.sum()),
                    "frac_q10_mod": float((r <= q10).mean()),
                }
            )
        if i % 200 == 0:
            print(f"  bucket sites {i}/{len(meta)}", flush=True)

    out = pd.DataFrame(rows)
    h = hydro[["usgs_site_no", "year", "GP", "logP", "logR", "region", "site_key"]].drop_duplicates()
    out = out.merge(h, on=["usgs_site_no", "year"], how="left", suffixes=("", "_h"))
    if "region_h" in out.columns:
        out["region"] = out["region"].fillna(out["region_h"])
    if "site_key_h" in out.columns:
        out["site_key"] = out["site_key"].fillna(out["site_key_h"])
    out["logR_mod"] = np.log(out["R_mod"].where(out["R_mod"] > 0))
    out.to_parquet(out_p, index=False)

    rows_fe = []
    def add(r, **meta):
        r = dict(r)
        r.update(meta)
        rows_fe.append(r)

    add(fe_ols(out, "logR_mod"), sample="all", family="bucket_R")
    add(fe_ols(out, "frac_q10_mod"), sample="all", family="bucket_q10")
    add(fe_ols(out, "logR"), sample="bucket_overlap", family="obs_R")
    for region in ("Northeast", "South", "Interior", "West"):
        d = out.loc[out["region"] == region]
        add(fe_ols(d, "frac_q10_mod"), sample=region, family="bucket_q10")
        add(fe_ols(d, "logR_mod"), sample=region, family="bucket_R")

    tab = pd.DataFrame(rows_fe)
    tab.to_csv(DATA / "regression_bucket.csv", index=False)
    show = [c for c in ["family", "sample", "y", "ok", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP"] if c in tab]
    print(tab[show].to_string(index=False))
    (NOTES / "bucket_summary.json").write_text(
        json.dumps({"n": int(len(out)), "n_ok": int((tab["ok"] == True).sum())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
