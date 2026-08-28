"""Aim at the paper's actual gaps: NE dry TP vs South; low-crop dry NO3; why GP lowers R.

Do not promote a null interval to a confirmed result. Keep signed conflicts only.

  1. Dry-month C–Q slope in high- vs low-GP years (regime shift)
  2. logC ~ logQ + GP + logQ×GP  (within-site interaction)
  3. Low-flow day fraction (Q < site Q10) and whether it absorbs NE dry TP
  4. GP × PET and GP × lag P on log R (evaporative / antecedent split)
  5. Dry-season vs wet-season GP from the same GPCC daily cell

Outputs: data/regression_regime.csv, data/panel_lowflow.parquet,
         data/gpcc_season_gp.parquet, notes/regime_summary.json
"""
from __future__ import annotations

import gzip
import importlib
import json

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import DATA, GPCC_DIR, NOTES, YEAR0, YEAR1
from utils import fe_ols

usgs_decomp = importlib.import_module("02_usgs_decomp")
load_flow = usgs_decomp.load_flow


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def gini_daily(x: np.ndarray, min_valid: int = 80) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), np.clip(x, 0, None), np.nan)
    n_valid = int(np.isfinite(x).sum())
    tot = float(np.nansum(x))
    if n_valid < min_valid or tot <= 0:
        return np.nan
    xs = np.sort(np.where(np.isfinite(x), x, 0.0))
    n = xs.size
    i = np.arange(1, n + 1, dtype=float)
    return float(2.0 * np.sum(i * xs) / (n * tot) - (n + 1.0) / n)


def fe_interact(df, y, x1, x2, ctrls=("logP",), cluster="site_key") -> dict:
    """Site-demean, then y ~ x1 + x2 + x1:x2 + ctrls + year FE."""
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
        "ci95_lo_logQ": float(ci[0]),
        "ci95_hi_logQ": float(ci[1]),
    }


def _max_spell(below: np.ndarray) -> int:
    best = cur = 0
    for v in below:
        if v:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return int(best)


def build_lowflow(hydro: pd.DataFrame) -> pd.DataFrame:
    out_p = DATA / "panel_lowflow.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)
    seasons = pd.read_parquet(DATA / "usgs_q_seasons.parquet")
    seasons["usgs_site_no"] = seasons["usgs_site_no"].astype(str).str.zfill(8)
    sea = seasons.set_index("usgs_site_no").to_dict("index")
    ids = sorted({str(x).zfill(8) if str(x).isdigit() else str(x) for x in hydro["usgs_site_no"].dropna()})
    print("low-flow sites", len(ids), flush=True)
    rows = []
    for i, sid in enumerate(ids, 1):
        df = load_flow(sid)
        if df is None or df.empty:
            continue
        df = df.loc[df["date"].dt.year.between(YEAR0, YEAR1)].copy()
        if df.empty:
            continue
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        q10 = float(np.nanpercentile(df["q"].to_numpy(), 10))
        if not np.isfinite(q10) or q10 < 0:
            continue
        sm = sea.get(sid, {})
        dry = {sm.get("dry_m1"), sm.get("dry_m2"), sm.get("dry_m3")}
        dry.discard(None)
        for year, g in df.groupby("year"):
            q = g["q"].to_numpy()
            n = int(np.isfinite(q).sum())
            if n < 300:
                continue
            below = np.isfinite(q) & (q <= q10)
            g_dry = g.loc[g["month"].isin(dry)] if dry else g.iloc[0:0]
            q_dry = g_dry["q"].to_numpy() if len(g_dry) else np.array([])
            n_dry = int(np.isfinite(q_dry).sum())
            below_dry = (np.isfinite(q_dry) & (q_dry <= q10)) if n_dry else np.array([], dtype=bool)
            rows.append(
                {
                    "usgs_site_no": sid,
                    "year": int(year),
                    "q10": q10,
                    "frac_q10": float(below.mean()) if n else np.nan,
                    "n_q10": int(below.sum()),
                    "spell_q10": _max_spell(below),
                    "frac_q10_dry": float(below_dry.mean()) if n_dry >= 60 else np.nan,
                    "n_q10_dry": int(below_dry.sum()) if n_dry else 0,
                }
            )
        if i % 200 == 0:
            print(f"  lowflow {i}/{len(ids)}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(out_p, index=False)
    print("wrote", out_p, len(out), flush=True)
    return out


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


def build_season_gp(hydro: pd.DataFrame) -> pd.DataFrame:
    out_p = DATA / "gpcc_season_gp.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)
    seasons = pd.read_parquet(DATA / "usgs_q_seasons.parquet")
    seasons["usgs_site_no"] = seasons["usgs_site_no"].astype(str).str.zfill(8)
    meta = hydro.groupby("usgs_site_no", as_index=False).agg(
        lat1=("lat1", "first"), lon1=("lon1", "first")
    )
    meta["usgs_site_no"] = meta["usgs_site_no"].astype(str).str.zfill(8)
    meta = meta.merge(seasons, on="usgs_site_no", how="left")
    uniq = meta[["lat1", "lon1"]].dropna().drop_duplicates().reset_index(drop=True)
    print("season-GP cells", len(uniq), "sites", len(meta), flush=True)
    arr0, lat_grid, lon_grid = load_precip(YEAR0)
    ilat = np.abs(uniq["lat1"].to_numpy()[:, None] - lat_grid[None, :]).argmin(1)
    ilon = np.abs(uniq["lon1"].to_numpy()[:, None] - lon_grid[None, :]).argmin(1)
    del arr0
    cell_ts = {}
    for year in range(YEAR0, YEAR1 + 1):
        arr, _, _ = load_precip(year)
        ts = arr[:, ilat, ilon].astype(float)
        cell_ts[year] = ts
        print("  season-GP year", year, flush=True)
    key_to_j = {(float(a), float(b)): i for i, (a, b) in enumerate(zip(uniq["lat1"], uniq["lon1"]))}
    rows = []
    for rec in meta.itertuples(index=False):
        dry = {rec.dry_m1, rec.dry_m2, rec.dry_m3}
        wet = {rec.wet_m1, rec.wet_m2, rec.wet_m3}
        dry.discard(None)
        wet.discard(None)
        j = key_to_j[(float(rec.lat1), float(rec.lon1))]
        for year, ts in cell_ts.items():
            n = ts.shape[0]
            # GPCC daily is calendar year; month from day-of-year via pandas
            months = pd.date_range(f"{year}-01-01", periods=n, freq="D").month.to_numpy()
            p = ts[:, j]
            dry_m = np.isin(months, list(dry)) if dry else np.zeros(n, dtype=bool)
            wet_m = np.isin(months, list(wet)) if wet else np.zeros(n, dtype=bool)
            p_dry = p[dry_m]
            p_wet = p[wet_m]
            rows.append(
                {
                    "usgs_site_no": rec.usgs_site_no,
                    "year": year,
                    "GP_dry": gini_daily(p_dry, min_valid=70),
                    "GP_wet": gini_daily(p_wet, min_valid=70),
                    "P_dry": float(np.nansum(np.clip(np.where(np.isfinite(p_dry), p_dry, 0.0), 0, None))),
                    "P_wet": float(np.nansum(np.clip(np.where(np.isfinite(p_wet), p_wet, 0.0), 0, None))),
                }
            )
    out = pd.DataFrame(rows)
    out.to_parquet(out_p, index=False)
    print("wrote", out_p, len(out), flush=True)
    return out


def main() -> None:
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    hydro["usgs_site_no"] = hydro["usgs_site_no"].astype(str).str.zfill(8)
    paired = pd.read_parquet(DATA / "panel_us_paired.parquet")
    cq = pd.read_parquet(DATA / "panel_cq_month.parquet")
    rows: list[dict] = []

    # --- attach annual GP to monthly C–Q ---
    ann = hydro[["usgs_site_no", "year", "GP", "logP"]].drop_duplicates()
    cq = cq.copy()
    cq["usgs_site_no"] = cq["usgs_site_no"].astype(str).str.zfill(8)
    cq = cq.merge(ann, on=["usgs_site_no", "year"], how="left")
    crop_cut = float(hydro.drop_duplicates("usgs_site_no")["crop_pc"].median())
    cq["crop_hi"] = pd.to_numeric(cq["crop_pc"], errors="coerce") >= crop_cut
    cq["gp_hi"] = cq["GP"] >= cq.groupby("site_key")["GP"].transform("median")
    print("cq crop_hi", cq["crop_hi"].mean(), "GP ok", cq["GP"].notna().mean(), flush=True)

    # 1–2. C–Q regime: high vs low GP year; interaction
    slices = {
        "TP_dry_NE": cq.loc[(cq["param"] == "TP") & (cq["is_dry"] == True) & (cq["region"] == "Northeast")],
        "TP_dry_South": cq.loc[(cq["param"] == "TP") & (cq["is_dry"] == True) & (cq["region"] == "South")],
        "TP_dry_all": cq.loc[(cq["param"] == "TP") & (cq["is_dry"] == True)],
        "NO3_dry_lowcrop": cq.loc[(cq["param"] == "NO3N") & (cq["is_dry"] == True) & (cq["crop_hi"] == False)],
        "NO3_dry_highcrop": cq.loc[(cq["param"] == "NO3N") & (cq["is_dry"] == True) & (cq["crop_hi"] == True)],
        "NO3_wet_South": cq.loc[(cq["param"] == "NO3N") & (cq["is_wet"] == True) & (cq["region"] == "South")],
        "TN_dry_all": cq.loc[(cq["param"] == "TN") & (cq["is_dry"] == True)],
    }
    for name, d in slices.items():
        add(rows, cq_fe(d), sample=name, family="cq_all")
        add(rows, cq_fe(d.loc[d["gp_hi"] == True]), sample=name, family="cq_gp_hi")
        add(rows, cq_fe(d.loc[d["gp_hi"] == False]), sample=name, family="cq_gp_lo")
        add(rows, fe_interact(d, "logC", "logQ", "GP"), sample=name, family="cq_x_gp")

    # 3. low-flow days
    lf = build_lowflow(hydro)
    lf["usgs_site_no"] = lf["usgs_site_no"].astype(str)
    h = hydro.merge(lf, on=["usgs_site_no", "year"], how="left")
    h["log_frac_q10"] = np.log(h["frac_q10"].where(h["frac_q10"] > 0))
    h["log_spell"] = np.log(h["spell_q10"].where(h["spell_q10"] > 0))
    add(rows, fe_ols(h, "frac_q10"), sample="all", family="lowflow")
    add(rows, fe_ols(h, "spell_q10"), sample="all", family="lowflow")
    add(rows, fe_ols(h, "frac_q10_dry"), sample="all", family="lowflow")
    for region in ("Northeast", "South", "Interior", "West"):
        add(rows, fe_ols(h.loc[h["region"] == region], "frac_q10"), sample=region, family="lowflow")
        add(rows, fe_ols(h.loc[h["region"] == region], "frac_q10_dry"), sample=region, family="lowflow")

    # does low-flow absorb dry-season concentration?
    p2 = paired.copy()
    p2["usgs_site_no"] = p2["usgs_site_no"].astype(str)
    p2 = p2.merge(lf, on=["usgs_site_no", "year"], how="left")
    for sample, d in {
        "TP_NE": p2.loc[(p2["param"] == "TP") & (p2["region"] == "Northeast")],
        "TP_South": p2.loc[(p2["param"] == "TP") & (p2["region"] == "South")],
        "NO3_lowcrop": p2.loc[(p2["param"] == "NO3N") & (p2["crop_hi"] == False)],
        "NO3_highcrop": p2.loc[(p2["param"] == "NO3N") & (p2["crop_hi"] == True)],
    }.items():
        add(rows, fe_ols(d, "logC_low"), sample=sample, family="c_dry")
        add(rows, fe_ols(d, "logC_low", xvars=("GP", "logP", "frac_q10_dry")), sample=sample, family="c_dry_ctrl_q10")
        add(rows, fe_ols(d, "logLoad_dry"), sample=sample, family="load_dry")
        add(rows, fe_ols(d, "logLoad_dry", xvars=("GP", "logP", "frac_q10_dry")), sample=sample, family="load_dry_ctrl_q10")

    # 4. PET × GP and lag P × GP on runoff
    pet = paired.groupby(["usgs_site_no", "year"], as_index=False)["PET"].median()
    pet["usgs_site_no"] = pet["usgs_site_no"].astype(str)
    pet["pet"] = (-pet["PET"]).clip(lower=1.0)
    pet["logPET"] = np.log(pet["pet"])
    hp = h.merge(pet[["usgs_site_no", "year", "logPET"]], on=["usgs_site_no", "year"], how="left")
    hp = hp.sort_values(["site_key", "year"])
    hp["logP_lag"] = hp.groupby("site_key")["logP"].shift(1)
    add(rows, fe_interact(hp, "logR", "GP", "logPET"), sample="all", family="gp_x_pet")
    add(rows, fe_interact(hp, "logR_dry", "GP", "logPET"), sample="all", family="gp_x_pet")
    add(rows, fe_interact(hp, "logR", "GP", "logP_lag"), sample="all", family="gp_x_lagP")
    add(rows, fe_interact(hp, "logR_dry", "GP", "logP_lag"), sample="all", family="gp_x_lagP")
    pet_med = hp["logPET"].median()
    add(rows, fe_ols(hp.loc[hp["logPET"] >= pet_med], "logR"), sample="high_PET", family="pet_split")
    add(rows, fe_ols(hp.loc[hp["logPET"] < pet_med], "logR"), sample="low_PET", family="pet_split")
    for region in ("Northeast", "South", "Interior", "West"):
        add(rows, fe_ols(hp.loc[hp["region"] == region], "logR"), sample=region, family="pet_check")
        add(rows, fe_interact(hp.loc[hp["region"] == region], "logR", "GP", "logPET"), sample=region, family="gp_x_pet")

    # 5. seasonal GP
    sgp = build_season_gp(hydro)
    sgp["usgs_site_no"] = sgp["usgs_site_no"].astype(str)
    hs = h.merge(sgp, on=["usgs_site_no", "year"], how="left")
    hs["logP_dry"] = np.log(hs["P_dry"].where(hs["P_dry"] > 0))
    hs["logP_wet"] = np.log(hs["P_wet"].where(hs["P_wet"] > 0))
    add(rows, fe_ols(hs, "logR_dry", xvars=("GP_dry", "logP_dry")), sample="all", family="gp_season")
    add(rows, fe_ols(hs, "logR_wet", xvars=("GP_wet", "logP_wet")), sample="all", family="gp_season")
    add(rows, fe_ols(hs, "logR_dry", xvars=("GP_dry", "GP", "logP")), sample="all", family="gp_season_horse")
    add(rows, fe_ols(hs, "logR_wet", xvars=("GP_wet", "GP", "logP")), sample="all", family="gp_season_horse")
    for region in ("Northeast", "South", "Interior", "West"):
        d = hs.loc[hs["region"] == region]
        add(rows, fe_ols(d, "logR_dry", xvars=("GP_dry", "logP_dry")), sample=region, family="gp_season")
        add(rows, fe_ols(d, "logR_wet", xvars=("GP_wet", "logP_wet")), sample=region, family="gp_season")

    ps = paired.copy()
    ps["usgs_site_no"] = ps["usgs_site_no"].astype(str)
    ps = ps.merge(sgp, on=["usgs_site_no", "year"], how="left")
    ps["logP_dry"] = np.log(ps["P_dry"].where(ps.get("P_dry", 0) > 0)) if "P_dry" in ps else np.nan
    for sample, d in {
        "TP_NE": ps.loc[(ps["param"] == "TP") & (ps["region"] == "Northeast")],
        "TP_South": ps.loc[(ps["param"] == "TP") & (ps["region"] == "South")],
        "NO3_lowcrop": ps.loc[(ps["param"] == "NO3N") & (ps["crop_hi"] == False)],
    }.items():
        add(rows, fe_ols(d, "logC_low", xvars=("GP_dry", "logP")), sample=sample, family="c_dry_gpdry")
        add(rows, fe_ols(d, "logC_low", xvars=("GP_dry", "GP", "logP")), sample=sample, family="c_dry_horse")

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_regime.csv", index=False)
    show = [c for c in tab.columns if c in ("family", "sample", "y", "ok", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", "beta_logQ", "ci95_lo_logQ", "ci95_hi_logQ", "beta_logQ_x_GP", "ci95_lo_logQ_x_GP", "ci95_hi_logQ_x_GP", "beta_GP_x_logPET", "ci95_lo_GP_x_logPET", "ci95_hi_GP_x_logPET", "beta_GP_dry", "ci95_lo_GP_dry", "ci95_hi_GP_dry", "beta_frac_q10_dry", "ci95_lo_frac_q10_dry")]
    print(tab.loc[tab.get("ok", False) == True, show].to_string(index=False))
    (NOTES / "regime_summary.json").write_text(
        json.dumps({"n_rows": int(len(tab)), "n_ok": int((tab["ok"] == True).sum())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
