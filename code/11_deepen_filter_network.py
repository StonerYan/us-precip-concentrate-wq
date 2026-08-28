"""Supplement 0825 gaps with existing + public data. Do not overwrite 0825 tables.

Tracks (notes/extension_tracks.md):
  A  annual runoff Gini (GQ); NLDI nested up–downstream pairs
  B  memoryless Horton threshold from GPCC daily P (sign contrast vs observed R)
  C  monthly C×R product (closer to flux than annual median C × annual R)

New files only: data/panel_gq.parquet, data/nldi_pairs.parquet,
data/gpcc_threshold_year.parquet, data/panel_monthly_load.parquet,
data/regression_deepen.csv, notes/deepen_summary.json
"""
from __future__ import annotations

import gzip
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from config import DATA, GPCC_DIR, NOTES, YEAR0, YEAR1
from utils import fe_ols, site_slopes

import importlib

usgs_decomp = importlib.import_module("02_usgs_decomp")
load_flow = usgs_decomp.load_flow
cfs_to_mm = usgs_decomp.cfs_to_mm

NLDI_BASES = (
    "https://api.water.usgs.gov/nldi/linked-data",
)
NLDI_CACHE = DATA / "nldi_cache"
NLDI_DISTANCE_KM = 600
NLDI_NAV = "UT"  # upstream tributaries = full contributing network


def gini_daily(x: np.ndarray, min_valid: int = 300) -> float:
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


def add(rows: list, r: dict, **meta) -> None:
    r = dict(r)
    r.update(meta)
    rows.append(r)


# ---------------------------------------------------------------------------
# A1. Annual runoff Gini from USGS daily Q
# ---------------------------------------------------------------------------
def build_gq(hydro: pd.DataFrame) -> pd.DataFrame:
    out_p = DATA / "panel_gq.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)

    ids = sorted({str(x) for x in hydro["usgs_site_no"].dropna().unique()})
    area_map = hydro.groupby("usgs_site_no")["area"].median().to_dict()
    print("GQ sites", len(ids), flush=True)
    rows = []
    miss = 0
    for i, sid in enumerate(ids, 1):
        df = load_flow(sid)
        if df is None or df.empty:
            miss += 1
            continue
        df = df.loc[df["date"].dt.year.between(YEAR0, YEAR1)].copy()
        if df.empty:
            continue
        df["year"] = df["date"].dt.year
        area = float(area_map.get(sid, np.nan))
        df["q_mm"] = cfs_to_mm(df["q"], area)
        for year, g in df.groupby("year"):
            q = g["q_mm"].to_numpy() if np.isfinite(area) else g["q"].to_numpy()
            n = int(np.isfinite(q).sum())
            if n < 300:
                continue
            rows.append(
                {
                    "usgs_site_no": sid,
                    "year": int(year),
                    "GQ": gini_daily(q, min_valid=300),
                    "n_q_gq": n,
                    "q_p95share": _p95_share(q),
                }
            )
        if i % 100 == 0:
            print(f"  GQ {i}/{len(ids)} miss={miss}", flush=True)
    out = pd.DataFrame(rows)
    out.to_parquet(out_p, index=False)
    print("wrote", out_p, len(out), "miss", miss, flush=True)
    return out


def _p95_share(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=float)
    x = np.where(np.isfinite(x), np.clip(x, 0, None), np.nan)
    tot = float(np.nansum(x))
    if tot <= 0:
        return np.nan
    finite = x[np.isfinite(x)]
    thr = np.nanpercentile(finite, 95)
    return float(np.nansum(finite[finite >= thr]) / tot)


# ---------------------------------------------------------------------------
# A2. Nested pairs via NLDI (public USGS network navigation)
# ---------------------------------------------------------------------------
def nwis_feat(sid: str) -> str:
    s = str(sid).strip()
    if s.upper().startswith("USGS-"):
        body = s.split("-", 1)[1]
        return f"USGS-{body.zfill(8) if body.isdigit() else body}"
    if s.isdigit():
        return f"USGS-{s.zfill(8)}"
    return f"USGS-{s}"


def _nldi_get(url: str, timeout: int = 45) -> dict | list | None:
    req = urllib.request.Request(url, headers={"Accept": "application/json", "User-Agent": "precip-concentrate-research"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, json.JSONDecodeError):
        return None


def nldi_upstream_nwis(sid: str) -> list[str]:
    NLDI_CACHE.mkdir(parents=True, exist_ok=True)
    cache = NLDI_CACHE / f"{nwis_feat(sid).replace('USGS-', '')}.json"
    if cache.exists() and cache.stat().st_size > 10:
        try:
            return json.loads(cache.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            pass
    feat = nwis_feat(sid)
    ids: list[str] = []
    ok = False
    for base in NLDI_BASES:
        url = f"{base}/nwissite/{feat}/navigation/{NLDI_NAV}/nwissite?distance={NLDI_DISTANCE_KM}"
        payload = _nldi_get(url)
        if payload is None:
            continue
        feats = payload.get("features", payload) if isinstance(payload, dict) else payload
        if not isinstance(feats, list):
            continue
        ok = True
        for f in feats:
            props = f.get("properties", f) if isinstance(f, dict) else {}
            ident = str(props.get("identifier") or props.get("comid") or "")
            if ident.upper().startswith("USGS-"):
                ids.append(ident.split("-", 1)[1])
            elif str(ident).isdigit():
                ids.append(str(ident))
        break
    self = feat.replace("USGS-", "")
    ids = sorted({i.zfill(8) if i.isdigit() else i for i in ids if i.zfill(8) != self.zfill(8)})
    if ok:
        cache.write_text(json.dumps(ids), encoding="utf-8")
    return ids


def build_nldi_pairs(hydro: pd.DataFrame) -> pd.DataFrame:
    out_p = DATA / "nldi_pairs.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)

    meta = (
        hydro.groupby("usgs_site_no", as_index=False)
        .agg(
            area=("area", "median"),
            lat=("lat", "first"),
            lon=("lon", "first"),
            site_key=("site_key", "first"),
            region=("region", "first"),
        )
    )
    meta["usgs_site_no"] = meta["usgs_site_no"].astype(str)
    ours = {s.zfill(8) if s.isdigit() else s for s in meta["usgs_site_no"]}
    area = dict(zip(meta["usgs_site_no"].map(lambda s: s.zfill(8) if str(s).isdigit() else s), meta["area"]))
    key = dict(zip(meta["usgs_site_no"].map(lambda s: s.zfill(8) if str(s).isdigit() else s), meta["site_key"]))
    lat = dict(zip(meta["usgs_site_no"].map(lambda s: s.zfill(8) if str(s).isdigit() else s), meta["lat"]))
    lon = dict(zip(meta["usgs_site_no"].map(lambda s: s.zfill(8) if str(s).isdigit() else s), meta["lon"]))
    region = dict(zip(meta["usgs_site_no"].map(lambda s: s.zfill(8) if str(s).isdigit() else s), meta["region"]))

    downs = sorted(ours)
    print("NLDI query", len(downs), "gauges", flush=True)
    found: dict[str, list[str]] = {}

    def one(sid: str) -> tuple[str, list[str]]:
        return sid, nldi_upstream_nwis(sid)

    done = 0
    with ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(one, sid) for sid in downs]
        for fut in as_completed(futs):
            sid, ups = fut.result()
            found[sid] = ups
            done += 1
            if done % 50 == 0:
                print(f"  NLDI {done}/{len(downs)}", flush=True)
            time.sleep(0.0)

    rows = []
    for down, ups in found.items():
        a_d = area.get(down)
        if not np.isfinite(a_d):
            continue
        for up in ups:
            up8 = up.zfill(8) if up.isdigit() else up
            if up8 not in ours:
                continue
            a_u = area.get(up8)
            if not np.isfinite(a_u) or a_u >= 0.95 * a_d or a_u < 10:
                continue
            rows.append(
                {
                    "up": up8,
                    "down": down,
                    "area_up": float(a_u),
                    "area_down": float(a_d),
                    "site_key_up": key.get(up8),
                    "site_key_down": key.get(down),
                    "lat_up": lat.get(up8),
                    "lon_up": lon.get(up8),
                    "lat_down": lat.get(down),
                    "lon_down": lon.get(down),
                    "region_down": region.get(down),
                    "source": "nldi_UT",
                }
            )
    pairs = pd.DataFrame(rows).drop_duplicates(["up", "down"])
    # one immediate-like pair per downstream: largest upstream still nested
    if len(pairs):
        imm = pairs.sort_values("area_up").groupby("down", as_index=False).tail(1)
        imm["immediate"] = True
        pairs = pairs.merge(imm[["up", "down", "immediate"]], on=["up", "down"], how="left")
        pairs["immediate"] = pairs["immediate"].fillna(False)
    else:
        pairs["immediate"] = pd.Series(dtype=bool)
    pairs.to_parquet(out_p, index=False)
    print("wrote", out_p, "pairs", len(pairs), "immediate", int(pairs["immediate"].sum()) if len(pairs) else 0, flush=True)
    return pairs


# ---------------------------------------------------------------------------
# B. Memoryless Horton threshold from GPCC daily P
# ---------------------------------------------------------------------------
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


def build_threshold(hydro: pd.DataFrame) -> pd.DataFrame:
    out_p = DATA / "gpcc_threshold_year.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)

    uniq = hydro[["lat1", "lon1"]].dropna().drop_duplicates().reset_index(drop=True)
    print("GPCC cells", len(uniq), flush=True)
    arr0, lat_grid, lon_grid = load_precip(YEAR0)
    ilat = np.abs(uniq["lat1"].to_numpy()[:, None] - lat_grid[None, :]).argmin(1)
    ilon = np.abs(uniq["lon1"].to_numpy()[:, None] - lon_grid[None, :]).argmin(1)
    del arr0
    rows = []
    for year in range(YEAR0, YEAR1 + 1):
        arr, _, _ = load_precip(year)
        ts = arr[:, ilat, ilon].astype(float)
        ts = np.where(np.isfinite(ts), np.clip(ts, 0, None), np.nan)
        p = np.nansum(np.where(np.isfinite(ts), ts, 0.0), axis=0)
        n_valid = np.isfinite(ts).sum(axis=0)
        gp = np.array([gini_daily(ts[:, j], min_valid=300) for j in range(ts.shape[1])])
        rec = {
            "year": year,
            "P_thr": p,
            "GP_thr": gp,
            "n_days": n_valid,
            "R_thr1": np.nansum(np.maximum(np.where(np.isfinite(ts), ts, 0.0) - 1.0, 0.0), axis=0),
            "R_thr5": np.nansum(np.maximum(np.where(np.isfinite(ts), ts, 0.0) - 5.0, 0.0), axis=0),
            "R_thr10": np.nansum(np.maximum(np.where(np.isfinite(ts), ts, 0.0) - 10.0, 0.0), axis=0),
        }
        block = uniq.copy()
        for k, v in rec.items():
            block[k] = v
        rows.append(block)
        print("  threshold year", year, flush=True)
    out = pd.concat(rows, ignore_index=True)
    out.to_parquet(out_p, index=False)
    print("wrote", out_p, len(out), flush=True)
    return out


# ---------------------------------------------------------------------------
# C. Monthly C × monthly R  (not WRTDS; closer than annual median × annual R)
# ---------------------------------------------------------------------------
def build_monthly_load() -> pd.DataFrame:
    out_p = DATA / "panel_monthly_load.parquet"
    if out_p.exists():
        print("reuse", out_p, flush=True)
        return pd.read_parquet(out_p)

    cq = pd.read_parquet(DATA / "panel_cq_month.parquet")
    need = ["site_key", "param", "year", "c", "R_month"]
    d = cq.dropna(subset=need).copy()
    d = d.loc[(d["c"] > 0) & (d["R_month"] > 0)]
    prod = (
        d.assign(prod=d["c"] * d["R_month"])
        .groupby(["site_key", "param", "year"], as_index=False)
        .agg(L_month=("prod", "sum"), n_m=("prod", "size"), R_m_sum=("R_month", "sum"), c_med=("c", "median"))
    )
    prod = prod.loc[prod["n_m"] >= 6]
    # hydro.site_key is the USGS number; chemistry site_key includes country|id|source|param
    prod["usgs_site_no"] = prod["site_key"].str.extract(r"USGS-(\d+)", expand=False).str.zfill(8)
    prod.to_parquet(out_p, index=False)
    print("wrote", out_p, len(prod), flush=True)
    return prod


# ---------------------------------------------------------------------------
# Regressions + pair contrast
# ---------------------------------------------------------------------------
def pair_slope_contrast(pairs: pd.DataFrame, slopes: pd.DataFrame) -> dict:
    if pairs.empty or slopes.empty or "beta_GP" not in slopes:
        return {"ok": False, "reason": "empty"}
    sl = slopes.set_index("site_key")["beta_GP"]
    use = pairs.loc[pairs["immediate"] == True].copy() if "immediate" in pairs else pairs.copy()
    use["beta_up"] = use["site_key_up"].map(sl)
    use["beta_down"] = use["site_key_down"].map(sl)
    use = use.dropna(subset=["beta_up", "beta_down"])
    if len(use) < 15:
        return {"ok": False, "reason": "few_pairs", "n_pairs": int(len(use))}
    use["delta"] = use["beta_down"] - use["beta_up"]
    rng = np.random.default_rng(11)
    boots = [float(use["delta"].iloc[rng.choice(len(use), size=len(use), replace=True)].mean()) for _ in range(1000)]
    return {
        "ok": True,
        "n_pairs": int(len(use)),
        "n_up": int(use["site_key_up"].nunique()),
        "n_down": int(use["site_key_down"].nunique()),
        "mean_beta_up": float(use["beta_up"].mean()),
        "mean_beta_down": float(use["beta_down"].mean()),
        "mean_delta": float(use["delta"].mean()),
        "median_delta": float(use["delta"].median()),
        "share_down_more_neg": float((use["delta"] < 0).mean()),
        "ci95_lo_delta": float(np.percentile(boots, 2.5)),
        "ci95_hi_delta": float(np.percentile(boots, 97.5)),
        "mean_area_up": float(use["area_up"].mean()),
        "mean_area_down": float(use["area_down"].mean()),
    }


def main() -> None:
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    hydro["usgs_site_no"] = hydro["usgs_site_no"].astype(str)
    rows: list[dict] = []

    # --- A1 GQ ---
    gq = build_gq(hydro)
    gq["usgs_site_no"] = gq["usgs_site_no"].astype(str)
    h = hydro.merge(gq, on=["usgs_site_no", "year"], how="left")
    h["logGQ"] = np.log(h["GQ"].where(h["GQ"] > 0))
    h["GQ_minus_GP"] = h["GQ"] - h["GP"]
    add(rows, fe_ols(h, "GQ"), sample="all", family="gq")
    add(rows, fe_ols(h, "GQ_minus_GP"), sample="all", family="gq")
    add(rows, fe_ols(h, "logR"), sample="gq_overlap", family="hydro_check")
    for sample, d in {
        "West": h.loc[h["region"] == "West"],
        "Interior": h.loc[h["region"] == "Interior"],
        "South": h.loc[h["region"] == "South"],
        "Northeast": h.loc[h["region"] == "Northeast"],
        "small": h.loc[h["area"] < 500],
        "mid": h.loc[h["area"].between(500, 5000)],
        "large": h.loc[h["area"] > 5000],
    }.items():
        add(rows, fe_ols(d, "GQ"), sample=sample, family="gq")
        add(rows, fe_ols(d, "logR"), sample=sample, family="hydro_check")

    sl_r = site_slopes(h, "logR")
    sl_g = site_slopes(h.dropna(subset=["GQ"]), "GQ")
    sl_r.to_parquet(DATA / "site_slopes_runoff_0826.parquet", index=False)
    sl_g.to_parquet(DATA / "site_slopes_gq.parquet", index=False)

    # --- B threshold (before NLDI so a slow API cannot block this contrast) ---
    thr = build_threshold(hydro)
    ht = hydro.merge(thr, on=["lat1", "lon1", "year"], how="left")
    for y, raw in [("logR_thr1", "R_thr1"), ("logR_thr5", "R_thr5"), ("logR_thr10", "R_thr10")]:
        ht[y] = np.log(ht[raw].where(ht[raw] > 0))
        add(rows, fe_ols(ht, y), sample="all", family="horton")
    add(rows, fe_ols(ht, "logR"), sample="thr_overlap", family="horton")

    # --- C monthly product ---
    ml = build_monthly_load()
    h2 = hydro.copy()
    h2["usgs_site_no"] = h2["usgs_site_no"].astype(str).str.zfill(8)
    ann = h2[["usgs_site_no", "year", "GP", "logP", "R", "logR", "region", "area"]].drop_duplicates()
    if "usgs_site_no" not in ml.columns:
        ml["usgs_site_no"] = ml["site_key"].str.extract(r"USGS-(\d+)", expand=False).str.zfill(8)
    ml = ml.merge(ann, on=["usgs_site_no", "year"], how="left")
    ml["L_fwm_annual"] = (ml["L_month"] / ml["R_m_sum"]) * ml["R"]
    ml["logL_month"] = np.log(ml["L_month"].where(ml["L_month"] > 0))
    ml["logL_fwm"] = np.log(ml["L_fwm_annual"].where(ml["L_fwm_annual"] > 0))
    # also annual-median product on the same years if panel exists
    paired = None
    pfile = DATA / "panel_us_paired.parquet"
    if pfile.exists():
        paired = pd.read_parquet(pfile)
    for param in ("TN", "NO3N", "TP"):
        d = ml.loc[ml["param"] == param]
        add(rows, fe_ols(d, "logL_month"), sample=param, family="month_load")
        add(rows, fe_ols(d, "logL_fwm"), sample=param, family="fwm_load")
        add(rows, fe_ols(d, "logR"), sample=param, family="month_load_R")
        if paired is not None and "param" in paired.columns:
            same = paired.loc[paired["param"] == param].copy()
            if "logCR" in same.columns:
                add(rows, fe_ols(same, "logCR"), sample=param, family="cxr_ref")
            elif "C" in same.columns and "R" in same.columns:
                same["logCR"] = np.log((same["C"] * same["R"]).where((same["C"] > 0) & (same["R"] > 0)))
                add(rows, fe_ols(same, "logCR"), sample=param, family="cxr_ref")

    pd.DataFrame(rows).to_csv(DATA / "regression_deepen.csv", index=False)
    print("checkpoint regressions (pre-NLDI)", len(rows), flush=True)

    # --- A2 NLDI pairs last (public API; cached) ---
    pairs = build_nldi_pairs(hydro)
    contrast = pair_slope_contrast(pairs, sl_r)
    if pairs is not None and len(pairs):
        imm = pairs.loc[pairs["immediate"] == True] if "immediate" in pairs.columns else pairs
        down_keys = set(imm["site_key_down"].dropna())
        up_keys = set(imm["site_key_up"].dropna())
        add(rows, fe_ols(h.loc[h["site_key"].isin(down_keys)], "logR"), sample="nldi_down", family="network")
        add(rows, fe_ols(h.loc[h["site_key"].isin(up_keys - down_keys)], "logR"), sample="nldi_up_only", family="network")
        add(rows, fe_ols(h.loc[h["site_key"].isin(down_keys)], "GQ"), sample="nldi_down", family="network_gq")
        add(rows, fe_ols(h.loc[h["site_key"].isin(up_keys - down_keys)], "GQ"), sample="nldi_up_only", family="network_gq")

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_deepen.csv", index=False)
    (NOTES / "deepen_summary.json").write_text(
        json.dumps(
            {
                "gq_n": int(h["GQ"].notna().sum()),
                "gq_sites": int(h.loc[h["GQ"].notna(), "site_key"].nunique()),
                "nldi_pairs": int(len(pairs)) if pairs is not None else 0,
                "nldi_immediate": int(pairs["immediate"].sum()) if pairs is not None and len(pairs) and "immediate" in pairs else 0,
                "pair_contrast": contrast,
                "month_load_n": int(len(ml)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    show_cols = [c for c in ["family", "sample", "y", "ok", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP"] if c in tab]
    print(tab.loc[tab.get("ok", True) == True, show_cols].to_string(index=False))
    print("PAIR CONTRAST", json.dumps(contrast, indent=2))


if __name__ == "__main__":
    main()
