"""Load split + NE/South dry TP + C-Q + crop/Corn Belt dry TN."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import DATA, NOTES
from utils import binned_partial, fe_ols, site_slopes


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def cq_fe(df: pd.DataFrame) -> dict:
    d = df.dropna(subset=["logC", "logQ", "site_key", "year"]).copy()
    if d["site_key"].nunique() < 20 or len(d) < 120:
        return {"y": "logC", "n": int(len(d)), "ok": False, "reason": "small"}
    for col in ["logC", "logQ"]:
        d[col] = d[col] - d.groupby("site_key")[col].transform("mean")
    try:
        res = smf.ols("logC ~ logQ + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d["site_key"]}
        )
    except Exception as exc:
        return {"y": "logC", "n": int(len(d)), "ok": False, "reason": str(exc)}
    ci = res.conf_int().loc["logQ"]
    return {
        "y": "logC",
        "xvars": "logQ",
        "n": int(res.nobs),
        "n_sites": int(d["site_key"].nunique()),
        "r2_within": float(res.rsquared),
        "ok": True,
        "beta_logQ": float(res.params["logQ"]),
        "se_logQ": float(res.bse["logQ"]),
        "p_logQ": float(res.pvalues["logQ"]),
        "ci95_lo_logQ": float(ci[0]),
        "ci95_hi_logQ": float(ci[1]),
    }


def main() -> None:
    p = pd.read_parquet(DATA / "panel_us_paired.parquet")
    cq = pd.read_parquet(DATA / "panel_cq_month.parquet")
    rows, bins = [], []

    load_y = ["logLoad", "logLoad_dry", "logLoad_wet", "logR", "logC", "logC_low", "logC_high", "share_wet_load"]
    for param in ["TN", "NO3N", "TP"]:
        d = p.loc[p["param"] == param]
        for y in load_y:
            add(rows, fe_ols(d, y), param=param, sample="paired", family="load")
        for region in ["West", "Interior", "South", "Northeast"]:
            sub = d.loc[d["region"] == region]
            for y in ["logLoad", "logLoad_dry", "logC_low", "logC_high"]:
                add(rows, fe_ols(sub, y), param=param, sample=region, family="load_region")

    # Conflict 1: dry TP Northeast vs South, then urban/dam/area slices
    tp = p.loc[p["param"] == "TP"]
    tp_slices = {
        "Northeast": tp.loc[tp["region"] == "Northeast"],
        "South": tp.loc[tp["region"] == "South"],
        "NE_urb_hi": tp.loc[(tp["region"] == "Northeast") & (tp["urb_hi"] == True)] if "urb_hi" in tp else tp.iloc[0:0],
        "NE_urb_lo": tp.loc[(tp["region"] == "Northeast") & (tp["urb_hi"] == False)] if "urb_hi" in tp else tp.iloc[0:0],
        "South_urb_hi": tp.loc[(tp["region"] == "South") & (tp["urb_hi"] == True)] if "urb_hi" in tp else tp.iloc[0:0],
        "South_urb_lo": tp.loc[(tp["region"] == "South") & (tp["urb_hi"] == False)] if "urb_hi" in tp else tp.iloc[0:0],
        "NE_dam": tp.loc[(tp["region"] == "Northeast") & (tp["has_dam"] == True)],
        "NE_nodam": tp.loc[(tp["region"] == "Northeast") & (tp["has_dam"] == False)],
        "South_dam": tp.loc[(tp["region"] == "South") & (tp["has_dam"] == True)],
        "South_nodam": tp.loc[(tp["region"] == "South") & (tp["has_dam"] == False)],
        "NE_small": tp.loc[(tp["region"] == "Northeast") & (tp["area"] < 500)],
        "NE_large": tp.loc[(tp["region"] == "Northeast") & (tp["area"] > 5000)],
        "South_small": tp.loc[(tp["region"] == "South") & (tp["area"] < 500)],
        "South_large": tp.loc[(tp["region"] == "South") & (tp["area"] > 5000)],
    }
    for sample, d in tp_slices.items():
        add(rows, fe_ols(d, "logC_low"), param="TP", sample=sample, family="conflict_tp")
        add(rows, fe_ols(d, "logLoad_dry"), param="TP", sample=sample, family="conflict_tp")

    # C-Q in dry months
    cq_dry = cq.loc[cq["is_dry"] == True]
    for param in ["TN", "NO3N", "TP"]:
        d = cq_dry.loc[cq_dry["param"] == param]
        add(rows, cq_fe(d), param=param, sample="paired_dry", family="cq")
        for region in ["Northeast", "South", "West", "Interior"]:
            add(rows, cq_fe(d.loc[d["region"] == region]), param=param, sample=region, family="cq")

    sl_cq = site_slopes(cq_dry.loc[cq_dry["param"] == "TP"], "logC", xvars=("logQ",), min_n=8)
    sl_cq.to_parquet(DATA / "site_slopes_cq_tp.parquet", index=False)
    for param, y, name in (
        ("TP", "logC_low", "site_slopes_tp_dry.parquet"),
        ("NO3N", "logC_low", "site_slopes_no3_dry.parquet"),
        ("TN", "logC_low", "site_slopes_tn_dry.parquet"),
    ):
        sl = site_slopes(p.loc[p["param"] == param], y)
        sl.to_parquet(DATA / name, index=False)

    # Conflict 2: crop split dry TN; Corn Belt separately; wet NO3 support
    for param in ["TN", "NO3N"]:
        d = p.loc[p["param"] == param]
        crop_slices = {
            "crop_hi": d.loc[d["crop_hi"] == True] if "crop_hi" in d else d.iloc[0:0],
            "crop_lo": d.loc[d["crop_hi"] == False] if "crop_hi" in d else d.iloc[0:0],
            "CornBelt": d.loc[d["corn_belt"] == True] if "corn_belt" in d.columns else d.iloc[0:0],
            "CornBelt_crop_hi": d.loc[(d.get("corn_belt") == True) & (d["crop_hi"] == True)] if "crop_hi" in d and "corn_belt" in d.columns else d.iloc[0:0],
            "Interior": d.loc[d["region"] == "Interior"],
        }
        for sample, sub in crop_slices.items():
            add(rows, fe_ols(sub, "logC_low"), param=param, sample=sample, family="conflict_n")
            add(rows, fe_ols(sub, "logC_high"), param=param, sample=sample, family="conflict_n")
            add(rows, fe_ols(sub, "logLoad"), param=param, sample=sample, family="conflict_n")

    for y, subset, tag in [
        ("logLoad", p, "load"),
        ("logC_low", p.loc[p["param"] == "TP"], "tp_dry"),
        ("logC_low", p.loc[p["param"] == "TN"], "tn_dry"),
        ("logR", p.drop_duplicates(["usgs_site_no", "year"]), "runoff_on_chem"),
    ]:
        if "param" in subset.columns and subset["param"].nunique() == 1:
            b = binned_partial(subset, y)
        elif y == "logR":
            b = binned_partial(subset, y)
        else:
            b = pd.DataFrame()
            for param, g in subset.groupby("param"):
                bb = binned_partial(g, y)
                if len(bb):
                    bb["param"] = param
                    bins.append(bb.assign(outcome=y, sample=tag))
            continue
        if len(b):
            b["outcome"] = y
            b["sample"] = tag
            if "param" in subset.columns and subset["param"].nunique() == 1:
                b["param"] = subset["param"].iloc[0]
            bins.append(b)

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_conflict.csv", index=False)
    if bins:
        pd.concat(bins, ignore_index=True).to_csv(DATA / "partial_bins_conflict.csv", index=False)
    keep = [c for c in ["family", "param", "sample", "y", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", "beta_logQ", "ci95_lo_logQ", "ci95_hi_logQ"] if c in tab]
    print(tab.loc[tab["ok"] == True, keep].to_string(index=False))
    (NOTES / "conflict_summary.json").write_text(
        json.dumps({"n_conflict_rows": int(tab["ok"].sum()) if len(tab) else 0}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
