"""Why does South Q10 lengthen with GP while the Northeast interval includes 0?

Observed tests only (no new GPCC loop). Kill or keep a mechanism by sign/CI.

  1. Is rain-day thinning shared? nwet_frac ~ GP + logP by region.
  2. Does wet-day thinning absorb Q10? frac_q10 ~ GP + nwet_frac + logP.
  3. Is the Q10 split a PET-climatology split? site-mean PET terciles;
     GP x year-PET on frac_q10; site-slope of Q10 on GP ~ site PET.
  4. Is Q10 just dry-season R restated? frac_q10 ~ GP + logR_dry + logP
     (mediation, not a control).

Do not treat a result as confirmed unless the regional Q10 conflict
is sharpened and the CI excludes the opposite story.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import DATA, NOTES
from utils import fe_ols, site_slopes


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def fe_interact(df, y, x1, x2, ctrls=("logP",), cluster="site_key") -> dict:
    need = [y, x1, x2, *ctrls, "site_key", "year"]
    d = df.dropna(subset=need).copy()
    if d["site_key"].nunique() < 20 or len(d) < 120:
        return {"y": y, "ok": False, "reason": "small", "n": int(len(d))}
    for col in [y, x1, x2, *ctrls]:
        d[col] = d[col] - d.groupby("site_key")[col].transform("mean")
    inter = f"{x1}_x_{x2}"
    d[inter] = d[x1] * d[x2]
    xvars = [x1, x2, inter, *ctrls]
    try:
        res = smf.ols(f"{y} ~ " + " + ".join(xvars) + " + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster]}
        )
    except Exception as exc:
        return {"y": y, "ok": False, "reason": str(exc), "n": int(len(d))}

    def grab(name: str) -> dict:
        if name not in res.params:
            return {}
        ci = res.conf_int().loc[name]
        return {
            f"beta_{name}": float(res.params[name]),
            f"se_{name}": float(res.bse[name]),
            f"p_{name}": float(res.pvalues[name]),
            f"ci95_lo_{name}": float(ci[0]),
            f"ci95_hi_{name}": float(ci[1]),
        }

    out = {
        "y": y,
        "n": int(res.nobs),
        "n_sites": int(d["site_key"].nunique()),
        "r2_within": float(res.rsquared),
        "ok": True,
        "xvars": "+".join(xvars),
    }
    for xv in xvars:
        out.update(grab(xv))
    return out


def between_ols(df, y, xvars, cluster=None) -> dict:
    d = df.dropna(subset=[y, *xvars]).copy()
    if len(d) < 40:
        return {"y": y, "ok": False, "reason": "small", "n": int(len(d))}
    try:
        if cluster and cluster in d:
            res = smf.ols(f"{y} ~ " + " + ".join(xvars), data=d).fit(
                cov_type="cluster", cov_kwds={"groups": d[cluster]}
            )
        else:
            res = smf.ols(f"{y} ~ " + " + ".join(xvars), data=d).fit(cov_type="HC1")
    except Exception as exc:
        return {"y": y, "ok": False, "reason": str(exc), "n": int(len(d))}

    def grab(name: str) -> dict:
        if name not in res.params:
            return {}
        ci = res.conf_int().loc[name]
        return {
            f"beta_{name}": float(res.params[name]),
            f"se_{name}": float(res.bse[name]),
            f"p_{name}": float(res.pvalues[name]),
            f"ci95_lo_{name}": float(ci[0]),
            f"ci95_hi_{name}": float(ci[1]),
        }

    out = {
        "y": y,
        "n": int(res.nobs),
        "n_sites": int(len(d)),
        "r2_within": float(res.rsquared),
        "ok": True,
        "xvars": "+".join(xvars),
    }
    for xv in xvars:
        out.update(grab(xv))
    return out


def main() -> None:
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    lf = pd.read_parquet(DATA / "panel_lowflow.parquet")
    paired = pd.read_parquet(DATA / "panel_us_paired.parquet", columns=["usgs_site_no", "year", "PET"])
    hydro["usgs_site_no"] = hydro["usgs_site_no"].astype(str).str.zfill(8)
    lf["usgs_site_no"] = lf["usgs_site_no"].astype(str).str.zfill(8)
    paired["usgs_site_no"] = paired["usgs_site_no"].astype(str).str.zfill(8)

    h = hydro.merge(lf, on=["usgs_site_no", "year"], how="left")
    pet = paired.groupby(["usgs_site_no", "year"], as_index=False)["PET"].median()
    pet["pet"] = (-pet["PET"]).clip(lower=1.0)
    pet["logPET"] = np.log(pet["pet"])
    h = h.merge(pet[["usgs_site_no", "year", "pet", "logPET"]], on=["usgs_site_no", "year"], how="left")

    site_pet = h.groupby("usgs_site_no", as_index=False).agg(
        pet_site=("pet", "median"),
        nwet_site=("nwet_frac", "median"),
        GP_site=("GP", "median"),
        region=("region", "first"),
        lat=("lat", "first"),
        lon=("lon", "first"),
    )
    cuts = site_pet["pet_site"].quantile([1 / 3, 2 / 3])
    site_pet["pet_tercile"] = pd.cut(
        site_pet["pet_site"],
        bins=[-np.inf, cuts.iloc[0], cuts.iloc[1], np.inf],
        labels=["low_PET", "mid_PET", "high_PET"],
    )
    h = h.merge(site_pet[["usgs_site_no", "pet_site", "pet_tercile"]], on="usgs_site_no", how="left")

    rows = []

    # 1. rain-day thinning shared?
    add(rows, fe_ols(h, "nwet_frac"), sample="all", family="nwet")
    for region in ("Northeast", "South", "Interior", "West"):
        add(rows, fe_ols(h.loc[h["region"] == region], "nwet_frac"), sample=region, family="nwet")

    # 2. does nwet absorb Q10 / dry R?
    add(rows, fe_ols(h, "frac_q10", xvars=("GP", "nwet_frac", "logP")), sample="all", family="q10_ctrl_nwet")
    add(rows, fe_ols(h, "logR_dry", xvars=("GP", "nwet_frac", "logP")), sample="all", family="Rdry_ctrl_nwet")
    for region in ("Northeast", "South", "Interior", "West"):
        d = h.loc[h["region"] == region]
        add(rows, fe_ols(d, "frac_q10", xvars=("GP", "nwet_frac", "logP")), sample=region, family="q10_ctrl_nwet")
        add(rows, fe_ols(d, "logR_dry", xvars=("GP", "nwet_frac", "logP")), sample=region, family="Rdry_ctrl_nwet")

    # 3. PET climatology terciles (between-site energy)
    for terc in ("low_PET", "mid_PET", "high_PET"):
        d = h.loc[h["pet_tercile"] == terc]
        add(rows, fe_ols(d, "frac_q10"), sample=terc, family="q10_pet_tercile")
        add(rows, fe_ols(d, "logR_dry"), sample=terc, family="Rdry_pet_tercile")
        add(rows, fe_ols(d, "nwet_frac"), sample=terc, family="nwet_pet_tercile")

    for region in ("Northeast", "South"):
        d = h.loc[h["region"] == region].copy()
        med = d.drop_duplicates("usgs_site_no")["pet_site"].median()
        add(rows, fe_ols(d.loc[d["pet_site"] >= med], "frac_q10"), sample=f"{region}_highPET", family="q10_pet_within")
        add(rows, fe_ols(d.loc[d["pet_site"] < med], "frac_q10"), sample=f"{region}_lowPET", family="q10_pet_within")

    # year-to-year PET interaction (same object as the killed logR x PET)
    add(rows, fe_interact(h, "frac_q10", "GP", "logPET"), sample="all", family="q10_x_pet")
    add(rows, fe_interact(h, "logR_dry", "GP", "logPET"), sample="all", family="Rdry_x_pet")
    for region in ("Northeast", "South"):
        add(rows, fe_interact(h.loc[h["region"] == region], "frac_q10", "GP", "logPET"), sample=region, family="q10_x_pet")

    # 4. mediation: Q10 after dry-season R
    add(rows, fe_ols(h, "frac_q10", xvars=("GP", "logR_dry", "logP")), sample="all", family="q10_ctrl_Rdry")
    for region in ("Northeast", "South"):
        add(rows, fe_ols(h.loc[h["region"] == region], "frac_q10", xvars=("GP", "logR_dry", "logP")), sample=region, family="q10_ctrl_Rdry")

    # 5. between-site: Q10 site-slopes on PET / nwet / region
    sl = site_slopes(h, "frac_q10")
    sl = sl.merge(site_pet, on="usgs_site_no", how="left") if "usgs_site_no" in sl.columns else sl
    if "site_key" in sl.columns and "usgs_site_no" not in sl.columns:
        sl = sl.merge(
            h.groupby("site_key", as_index=False).agg(usgs_site_no=("usgs_site_no", "first")),
            on="site_key",
            how="left",
        )
        sl = sl.merge(site_pet, on="usgs_site_no", how="left")
    sl["log_pet_site"] = np.log(sl["pet_site"].where(sl["pet_site"] > 0))
    sl["south"] = (sl["region"] == "South").astype(float)
    sl["northeast"] = (sl["region"] == "Northeast").astype(float)
    add(rows, between_ols(sl, "beta_GP", ["log_pet_site"]), sample="all", family="slope_on_PET")
    add(rows, between_ols(sl, "beta_GP", ["nwet_site"]), sample="all", family="slope_on_nwet")
    add(rows, between_ols(sl, "beta_GP", ["log_pet_site", "south", "northeast"]), sample="all", family="slope_on_PET_region")

    # PET climatology by region (descriptive, for the note)
    desc = (
        site_pet.groupby("region", as_index=False)
        .agg(n=("usgs_site_no", "nunique"), pet_med=("pet_site", "median"), nwet_med=("nwet_site", "median"), GP_med=("GP_site", "median"))
    )

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_why.csv", index=False)
    desc.to_csv(DATA / "why_region_pet.csv", index=False)
    sl.to_parquet(DATA / "site_slopes_q10_why.parquet", index=False)

    keep = [
        c
        for c in tab.columns
        if c
        in (
            "family",
            "sample",
            "y",
            "ok",
            "n",
            "n_sites",
            "beta_GP",
            "ci95_lo_GP",
            "ci95_hi_GP",
            "beta_nwet_frac",
            "ci95_lo_nwet_frac",
            "ci95_hi_nwet_frac",
            "beta_GP_x_logPET",
            "ci95_lo_GP_x_logPET",
            "ci95_hi_GP_x_logPET",
            "beta_log_pet_site",
            "ci95_lo_log_pet_site",
            "ci95_hi_log_pet_site",
            "beta_south",
            "ci95_lo_south",
            "ci95_hi_south",
            "beta_northeast",
            "ci95_lo_northeast",
            "ci95_hi_northeast",
            "beta_nwet_site",
            "ci95_lo_nwet_site",
            "ci95_hi_nwet_site",
        )
    ]
    print(tab.loc[tab.get("ok", False) == True, keep].to_string(index=False))
    print("\nregion PET/nwet")
    print(desc.to_string(index=False))
    (NOTES / "why_summary.json").write_text(
        json.dumps({"n_rows": int(len(tab)), "n_ok": int((tab["ok"] == True).sum())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
