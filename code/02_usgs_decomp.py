"""USGS daily flow: annual R, dry/wet-season runoff, monthly Q for C-Q."""
from __future__ import annotations

import numpy as np
import pandas as pd

from config import DATA, USGS_DIRS, YEAR0, YEAR1


def load_flow(site_no: str) -> pd.DataFrame | None:
    cands = [site_no]
    if site_no.isdigit():
        cands += [site_no.lstrip("0") or site_no, site_no.zfill(8), str(int(site_no))]
    tried = set()
    for folder in USGS_DIRS:
        for cand in cands:
            key = (str(folder), cand)
            if key in tried:
                continue
            tried.add(key)
            p = folder / f"sw_{cand}_00060.parquet"
            if p.exists() and p.stat().st_size > 500:
                df = pd.read_parquet(p)
                df["date"] = pd.to_datetime(df["date"], errors="coerce")
                valcol = "value" if "value" in df.columns else df.columns[-1]
                df["q"] = pd.to_numeric(df[valcol], errors="coerce")
                return df.dropna(subset=["date", "q"])
    return None


def cfs_to_mm(q_cfs: pd.Series, area_km2: float) -> pd.Series:
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return q_cfs * np.nan
    return q_cfs * 0.028316846592 * 86400.0 / (area_km2 * 1e6) * 1000.0


def main() -> None:
    raw = pd.read_parquet(DATA / "panel_us_paired_raw.parquet")
    seasons = pd.read_parquet(DATA / "usgs_q_seasons.parquet")
    seasons["usgs_site_no"] = seasons["usgs_site_no"].astype(str)
    area_map = raw.groupby("usgs_site_no")["area"].median().to_dict()
    ids = sorted({str(x) for x in raw["usgs_site_no"].dropna().unique()})
    season_map = seasons.set_index("usgs_site_no").to_dict("index")
    print("USGS paired ids", len(ids), flush=True)

    year_rows, month_rows = [], []
    miss = 0
    for i, sid in enumerate(ids, 1):
        df = load_flow(sid)
        if df is None or df.empty:
            miss += 1
            continue
        area = float(area_map.get(sid, np.nan))
        df["year"] = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df = df.loc[df["year"].between(YEAR0, YEAR1)]
        if df.empty:
            continue
        df["q_mm"] = cfs_to_mm(df["q"], area)
        sea = season_map.get(sid, {})
        dry = {sea.get("dry_m1"), sea.get("dry_m2"), sea.get("dry_m3")}
        wet = {sea.get("wet_m1"), sea.get("wet_m2"), sea.get("wet_m3")}
        dry.discard(None)
        wet.discard(None)
        df["is_dry"] = df["month"].isin(dry) if dry else False
        df["is_wet"] = df["month"].isin(wet) if wet else False

        g = df.groupby("year").agg(
            n_q=("q", "size"),
            R=("q_mm", "sum"),
            q_mean=("q", "mean"),
            n_dry=("is_dry", "sum"),
            n_wet=("is_wet", "sum"),
        )
        g["R_dry"] = df.loc[df["is_dry"]].groupby("year")["q_mm"].sum()
        g["R_wet"] = df.loc[df["is_wet"]].groupby("year")["q_mm"].sum()
        g = g.reset_index()
        g = g.loc[g["n_q"] >= 300]
        g["usgs_site_no"] = sid
        year_rows.append(g)

        m = df.groupby(["year", "month"], as_index=False).agg(
            n_qm=("q", "size"),
            q_mean=("q", "mean"),
            R_month=("q_mm", "sum"),
        )
        m = m.loc[m["n_qm"] >= 20]
        m["usgs_site_no"] = sid
        month_rows.append(m)

        if i % 200 == 0:
            print(f"  flow {i}/{len(ids)}", flush=True)

    flow = pd.concat(year_rows, ignore_index=True) if year_rows else pd.DataFrame()
    monthly = pd.concat(month_rows, ignore_index=True) if month_rows else pd.DataFrame()
    flow.to_parquet(DATA / "usgs_flow_decomp.parquet", index=False)
    monthly.to_parquet(DATA / "usgs_monthly_q.parquet", index=False)
    print("flow years", len(flow), "sites", flow["usgs_site_no"].nunique() if len(flow) else 0, "miss", miss)
    print("monthly rows", len(monthly))


if __name__ == "__main__":
    main()
