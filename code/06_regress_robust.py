"""Placebos and robustness on runoff/load mains; SI-only national C and Europe sign."""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import CACHE24, DATA, NOTES
from utils import fe_ols


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def main() -> None:
    h = pd.read_parquet(DATA / "panel_hydro.parquet")
    p = pd.read_parquet(DATA / "panel_us_paired.parquet")
    rows = []
    rng = np.random.default_rng(7)

    # Within-site GP permutation on runoff and crude load (mains only)
    hp = h.dropna(subset=["GP", "logP", "logR", "site_key"]).copy()
    hp["GP"] = hp.groupby("site_key")["GP"].transform(lambda s: rng.permutation(s.to_numpy()))
    add(rows, fe_ols(hp, "logR"), sample="placebo_GPperm", family="placebo", param="R")
    add(rows, fe_ols(hp, "logRP"), sample="placebo_GPperm", family="placebo", param="RP")

    for param in ["TN", "NO3N", "TP"]:
        d = p.loc[p["param"] == param].dropna(subset=["GP", "logP", "logLoad", "site_key"]).copy()
        d["GP"] = d.groupby("site_key")["GP"].transform(lambda s: rng.permutation(s.to_numpy()))
        add(rows, fe_ols(d, "logLoad"), param=param, sample="placebo_GPperm", family="placebo")
        d2 = p.loc[p["param"] == param].copy()
        if "load" in d2:
            lo, hi = d2["load"].quantile(0.005), d2["load"].quantile(0.995)
            d2["logLoad_w"] = np.log(d2["load"].clip(lo, hi).where(d2["load"] > 0))
            add(rows, fe_ols(d2, "logLoad_w"), param=param, sample="winsor_load", family="robust")
        d3 = p.loc[(p["param"] == param) & p["area"].notna()]
        add(rows, fe_ols(d3, "logLoad"), param=param, sample="area_complete", family="robust")
        add(rows, fe_ols(d3, "logC"), param=param, sample="area_complete_C", family="si_national_c")

    # SI: national annual C and p95-on-C (one line each; not the paper spine)
    for param in ["TN", "NO3N", "TP"]:
        d = p.loc[p["param"] == param]
        add(rows, fe_ols(d, "logC"), param=param, sample="national_annual_C", family="si_national_c")
        if "p95share" in d:
            add(rows, fe_ols(d, "logC_low", xvars=("p95share", "logP")), param=param, sample="p95_on_Clow", family="si_national_c")

    # Hydro winsor / area complete
    hw = h.copy()
    lo, hi = hw["R"].quantile(0.005), hw["R"].quantile(0.995)
    hw["logR_w"] = np.log(hw["R"].clip(lo, hi).where(hw["R"] > 0))
    add(rows, fe_ols(hw, "logR_w"), sample="winsor_R", family="robust", param="R")
    add(rows, fe_ols(h.loc[h["area"].notna()], "logR"), sample="area_complete", family="robust", param="R")

    # Europe SI sign check if cached
    eu_path = DATA / "panel_eu_si.parquet"
    if not eu_path.exists() and (CACHE24 / "panel_eu_si.parquet").exists():
        eu_path = CACHE24 / "panel_eu_si.parquet"
    if eu_path.exists():
        eu = pd.read_parquet(eu_path)
        if "site_key" not in eu.columns:
            eu["site_key"] = (
                eu["site_country"].astype(str)
                + "|"
                + eu["site_id"].astype(str)
                + "|"
                + eu["source"].astype(str)
                + "|"
                + eu["param"].astype(str)
            )
        if "logC_low" not in eu.columns and "c_low" in eu.columns:
            eu["logC_low"] = np.log(eu["c_low"].clip(lower=1e-6))
        if "logP" not in eu.columns and "P" in eu.columns:
            eu["logP"] = np.log(eu["P"].clip(lower=1))
        for param in ["TN", "NO3N", "TP"]:
            d = eu.loc[eu["param"] == param] if "param" in eu.columns else eu
            if "logC_low" in d.columns:
                add(rows, fe_ols(d, "logC_low"), param=param, sample="europe_dryC", family="si_europe")

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_robust.csv", index=False)
    keep = [c for c in ["family", "param", "sample", "y", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP"] if c in tab]
    print(tab.loc[tab["ok"] == True, keep].to_string(index=False))
    (NOTES / "robust_summary.json").write_text(
        json.dumps({"n_ok": int(tab["ok"].sum()) if len(tab) else 0}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
