"""Shared FE estimator and helpers. Re-estimate; do not import 0824 results."""
from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def site_key(df: pd.DataFrame, with_param: bool = True) -> pd.Series:
    k = (
        df["site_country"].astype(str)
        + "|"
        + df["site_id"].astype(str)
        + "|"
        + df["source"].astype(str)
    )
    if with_param and "param" in df.columns:
        k = k + "|" + df["param"].astype(str)
    return k


def fe_ols(df: pd.DataFrame, y: str, xvars=("GP", "logP"), cluster="site_key") -> dict:
    need = [y, *xvars, "site_key", "year"]
    d = df.dropna(subset=need).copy()
    if d["site_key"].nunique() < 20 or len(d) < 120:
        return {"y": y, "n": int(len(d)), "ok": False, "reason": "small"}
    for col in [y, *xvars]:
        d[col] = d[col] - d.groupby("site_key")[col].transform("mean")
    try:
        res = smf.ols(f"{y} ~ " + " + ".join(xvars) + " + C(year)", data=d).fit(
            cov_type="cluster", cov_kwds={"groups": d[cluster]}
        )
    except Exception as exc:
        return {"y": y, "n": int(len(d)), "ok": False, "reason": str(exc)}

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


def binned_partial(df, y, x="GP", ctrl="logP", n_bins=8, n_boot=200, seed=11):
    need = [y, x, ctrl, "site_key", "year"]
    d = df.dropna(subset=need).copy()
    if len(d) < 400:
        return pd.DataFrame()
    for col in [y, x, ctrl]:
        d[col + "_d"] = d[col] - d.groupby("site_key")[col].transform("mean")
    r_y = smf.ols(f"{y}_d ~ {ctrl}_d + C(year)", data=d).fit()
    r_x = smf.ols(f"{x}_d ~ {ctrl}_d + C(year)", data=d).fit()
    d["y_r"] = r_y.resid
    d["x_r"] = r_x.resid
    qs = np.quantile(d["x_r"], np.linspace(0.05, 0.95, n_bins + 1))
    d["bin"] = pd.cut(d["x_r"], qs, include_lowest=True)
    rng = np.random.default_rng(seed)
    rows = []
    for _, sub in d.groupby("bin", observed=True):
        if len(sub) < 30:
            continue
        idx = sub.index.to_numpy()
        means = [
            float(d.loc[rng.choice(idx, size=len(idx), replace=True), "y_r"].mean())
            for _ in range(n_boot)
        ]
        rows.append(
            {
                "x": float(sub["x_r"].mean()),
                "y": float(sub["y_r"].mean()),
                "y_lo": float(np.percentile(means, 2.5)),
                "y_hi": float(np.percentile(means, 97.5)),
                "n": int(len(sub)),
            }
        )
    return pd.DataFrame(rows)


def site_slopes(df: pd.DataFrame, y: str, xvars=("GP", "logP"), min_n=8) -> pd.DataFrame:
    rows = []
    need = [y, *xvars]
    for key, g in df.dropna(subset=need).groupby("site_key"):
        if len(g) < min_n:
            continue
        try:
            res = smf.ols(f"{y} ~ " + " + ".join(xvars), data=g).fit()
        except Exception:
            continue
        row = {"site_key": key, "y": y, "n": int(res.nobs), "r2": float(res.rsquared)}
        for xv in xvars:
            if xv in res.params:
                row[f"beta_{xv}"] = float(res.params[xv])
                row[f"se_{xv}"] = float(res.bse[xv])
        rows.append(row)
    return pd.DataFrame(rows)
