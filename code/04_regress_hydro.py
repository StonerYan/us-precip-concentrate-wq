"""Hydrology FE: runoff, R/P, dry/wet Q, wet-day vs GP horse race, site slopes."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA, NOTES
from utils import binned_partial, fe_ols, site_slopes


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def main() -> None:
    h = pd.read_parquet(DATA / "panel_hydro.parquet")
    rows, bins = [], []

    outcomes = ["logR", "logRP", "logR_dry", "logR_wet"]
    samples = {
        "all": h,
        "West": h.loc[h["region"] == "West"],
        "Interior": h.loc[h["region"] == "Interior"],
        "South": h.loc[h["region"] == "South"],
        "Northeast": h.loc[h["region"] == "Northeast"],
        "CornBelt": h.loc[h["corn_belt"] == True] if "corn_belt" in h.columns else h.iloc[0:0],
        "small": h.loc[h["area"] < 500],
        "mid": h.loc[h["area"].between(500, 5000)],
        "large": h.loc[h["area"] > 5000],
        "dam": h.loc[h["has_dam"] == True] if "has_dam" in h else h.iloc[0:0],
        "no_dam": h.loc[h["has_dam"] == False] if "has_dam" in h else h.iloc[0:0],
        "area_complete": h.loc[h["area"].notna()],
    }
    for sample, d in samples.items():
        for y in outcomes:
            if y not in d.columns:
                continue
            add(rows, fe_ols(d, y), sample=sample, family="hydro")

    # horse race: is GP just fewer wet days / p95?
    horse = [
        ("GP_only", ("GP", "logP")),
        ("nwet", ("nwet_frac", "logP")),
        ("p95", ("p95share", "logP")),
        ("GP_nwet", ("GP", "nwet_frac", "logP")),
        ("GP_p95", ("GP", "p95share", "logP")),
        ("all3", ("GP", "nwet_frac", "p95share", "logP")),
    ]
    for sample_name, xvars in horse:
        add(rows, fe_ols(h, "logR", xvars=xvars), sample=sample_name, family="horse")
        add(rows, fe_ols(h, "logRP", xvars=xvars), sample=sample_name, family="horse")

    for y in ["logR", "logRP", "logR_dry", "logR_wet"]:
        b = binned_partial(h, y)
        if len(b):
            b["outcome"] = y
            b["sample"] = "all"
            bins.append(b)

    sl = site_slopes(h, "logR")
    sl.to_parquet(DATA / "site_slopes_runoff.parquet", index=False)
    if len(sl) and "beta_GP" in sl:
        share_neg = float((sl["beta_GP"] < 0).mean())
        share_neg_sig = float(((sl["beta_GP"] < 0) & (sl["beta_GP"].abs() > 1.96 * sl["se_GP"])).mean())
    else:
        share_neg = share_neg_sig = np.nan

    # within-site high vs low GP year delta for maps
    d = h.dropna(subset=["GP", "R", "P"]).copy()
    d["gp_hi"] = d["GP"] >= d.groupby("site_key")["GP"].transform("median")
    delta = (
        d.groupby(["site_key", "gp_hi"], as_index=False)
        .agg(R=("R", "median"), GP=("GP", "median"), lat=("lat", "first"), lon=("lon", "first"), region=("region", "first"))
    )
    wide = delta.pivot(index=["site_key", "lat", "lon", "region"], columns="gp_hi", values="R")
    if True in wide.columns and False in wide.columns:
        wide = wide.reset_index()
        wide["dlogR"] = np.log(wide[True].where(wide[True] > 0)) - np.log(wide[False].where(wide[False] > 0))
        wide.to_parquet(DATA / "site_gp_delta_runoff.parquet", index=False)

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_hydro.csv", index=False)
    if bins:
        pd.concat(bins, ignore_index=True).to_csv(DATA / "partial_bins_hydro.csv", index=False)

    show = tab.loc[tab["ok"] == True, [c for c in ["family", "sample", "y", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", "beta_nwet_frac", "beta_p95share"] if c in tab]]
    print(show.to_string(index=False))
    (NOTES / "hydro_summary.json").write_text(
        json.dumps(
            {
                "n_hydro": int(len(h)),
                "n_sites": int(h["site_key"].nunique()),
                "share_neg_site_slope": share_neg,
                "share_neg_site_slope_1p96se": share_neg_sig,
                "n_site_slopes": int(len(sl)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
