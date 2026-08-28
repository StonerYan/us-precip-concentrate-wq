"""Explain the original (not only Q10) signed findings.

Mediation / horse-race only. Write into the paper only if a CI
sharpens an existing conflict.

  NE dry TP     ~ GP | logR_dry, GP_dry (already in regime)
  low-crop NO3  ~ GP | logR_dry
  South wet NO3 ~ GP | logR_wet
  crop-class median C and dry-month C-Q (already in regime)
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from config import DATA, NOTES
from utils import fe_ols


def add(rows, r, **meta):
    r = dict(r)
    r.update(meta)
    rows.append(r)


def main() -> None:
    p = pd.read_parquet(DATA / "panel_us_paired.parquet")
    crop_cut = float(p.drop_duplicates("usgs_site_no")["crop_pc"].median())
    p["crop_hi"] = pd.to_numeric(p["crop_pc"], errors="coerce") >= crop_cut

    rows = []
    ne_tp = p.loc[(p["param"] == "TP") & (p["region"] == "Northeast")]
    so_no3 = p.loc[(p["param"] == "NO3N") & (p["region"] == "South")]
    lo_no3 = p.loc[(p["param"] == "NO3N") & (p["crop_hi"] == False)]
    hi_no3 = p.loc[(p["param"] == "NO3N") & (p["crop_hi"] == True)]

    add(rows, fe_ols(ne_tp, "logC_low"), sample="TP_NE", family="base")
    add(rows, fe_ols(ne_tp, "logC_low", xvars=("GP", "logP", "logR_dry")), sample="TP_NE", family="ctrl_Rdry")
    add(rows, fe_ols(lo_no3, "logC_low"), sample="NO3_lowcrop", family="base")
    add(rows, fe_ols(lo_no3, "logC_low", xvars=("GP", "logP", "logR_dry")), sample="NO3_lowcrop", family="ctrl_Rdry")
    add(rows, fe_ols(so_no3, "logC_high"), sample="NO3_South", family="base")
    add(rows, fe_ols(so_no3, "logC_high", xvars=("GP", "logP", "logR_wet")), sample="NO3_South", family="ctrl_Rwet")
    add(rows, fe_ols(hi_no3, "logC_low"), sample="NO3_highcrop", family="base")
    add(rows, fe_ols(hi_no3, "logC_low", xvars=("GP", "logP", "logR_dry")), sample="NO3_highcrop", family="ctrl_Rdry")

    desc = []
    for name, d, col in (
        ("NO3_lowcrop_dryC", lo_no3, "c_low"),
        ("NO3_highcrop_dryC", hi_no3, "c_low"),
        ("NO3_South_wetC", so_no3, "c_high"),
        ("TP_NE_dryC", ne_tp, "c_low"),
        ("TP_South_dryC", p.loc[(p["param"] == "TP") & (p["region"] == "South")], "c_low"),
    ):
        x = pd.to_numeric(d[col], errors="coerce")
        desc.append(
            {
                "sample": name,
                "n": int(x.notna().sum()),
                "n_sites": int(d.loc[x.notna(), "site_key"].nunique()) if "site_key" in d else np.nan,
                "median": float(x.median()) if x.notna().any() else np.nan,
                "p25": float(x.quantile(0.25)) if x.notna().any() else np.nan,
                "p75": float(x.quantile(0.75)) if x.notna().any() else np.nan,
            }
        )

    tab = pd.DataFrame(rows)
    tab.to_csv(DATA / "regression_why_chem.csv", index=False)
    pd.DataFrame(desc).to_csv(DATA / "why_chem_medians.csv", index=False)
    keep = [c for c in tab.columns if c in ("family", "sample", "y", "ok", "n", "n_sites", "beta_GP", "ci95_lo_GP", "ci95_hi_GP")]
    print(tab.loc[tab.get("ok", False) == True, keep].to_string(index=False))
    print(pd.DataFrame(desc).to_string(index=False))
    (NOTES / "why_chem_summary.json").write_text(
        json.dumps({"n_ok": int((tab["ok"] == True).sum())}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
