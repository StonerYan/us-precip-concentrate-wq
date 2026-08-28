"""SI figures for the 0826 supplement (GQ, Horton contrast, nested pairs, monthly product)."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA, FIG, NOTES
from fig_style import PAL, PARAM_COLOR, PARAM_LAB, letter, lollipop, save_pub

SI = FIG / "si"


def _row(tab: pd.DataFrame, family: str, sample: str, y: str) -> pd.Series | None:
    d = tab.loc[(tab["family"] == family) & (tab["sample"] == sample) & (tab["y"] == y)]
    if d.empty:
        return None
    return d.iloc[0]


def fig_s3_filter(tab: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))

    # (a) observed R vs memoryless Horton from the same daily P
    specs = [
        ("Observed runoff", "horton", "thr_overlap", "logR", PAL["flow"]),
        ("Threshold 1 mm", "horton", "all", "logR_thr1", PAL["gray"]),
        ("Threshold 5 mm", "horton", "all", "logR_thr5", "#8a4b08"),
        ("Threshold 10 mm", "horton", "all", "logR_thr10", PAL["pos"]),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, fam, samp, y, c in specs:
        r = _row(tab, fam, samp, y)
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(axes[0], labs, b, lo, hi, colors=cols)
    axes[0].set_xlabel("Coefficient on GP")
    letter(axes[0], "a")

    # (b) runoff Gini and log R by basin area
    specs2 = [
        ("GQ, <500 km$^2$", "gq", "small", "GQ", PAL["gray"]),
        ("GQ, 500–5000", "gq", "mid", "GQ", PAL["gray"]),
        ("GQ, >5000", "gq", "large", "GQ", PAL["gray"]),
        ("log R, <500", "hydro_check", "small", "logR", PAL["flow"]),
        ("log R, 500–5000", "hydro_check", "mid", "logR", PAL["flow"]),
        ("log R, >5000", "hydro_check", "large", "logR", PAL["flow"]),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, fam, samp, y, c in specs2:
        r = _row(tab, fam, samp, y)
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(axes[1], labs, b, lo, hi, colors=cols)
    axes[1].set_xlabel("Coefficient on GP")
    letter(axes[1], "b")

    fig.tight_layout()
    save_pub(fig, "si_s3_filter_horton", outdir=SI)
    plt.close(fig)


def fig_s4_pairs() -> None:
    pfile = DATA / "nldi_pairs.parquet"
    sfile = DATA / "site_slopes_runoff_0826.parquet"
    if not pfile.exists() or not sfile.exists():
        print("skip S4: pairs or slopes missing")
        return
    pairs = pd.read_parquet(pfile)
    sl = pd.read_parquet(sfile)
    if pairs.empty or "beta_GP" not in sl:
        print("skip S4: empty")
        return
    slb = sl.set_index("site_key")["beta_GP"]
    use = pairs.loc[pairs["immediate"] == True].copy() if "immediate" in pairs else pairs.copy()
    use["beta_up"] = use["site_key_up"].map(slb)
    use["beta_down"] = use["site_key_down"].map(slb)
    use = use.dropna(subset=["beta_up", "beta_down"])
    if len(use) < 10:
        print("skip S4: few pairs", len(use))
        return
    use["delta"] = use["beta_down"] - use["beta_up"]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.1))
    axes[0].axhline(0, color="0.35", lw=0.55, ls="--")
    axes[0].axvline(0, color="0.35", lw=0.55, ls="--")
    axes[0].scatter(use["beta_up"], use["beta_down"], s=14, c=PAL["flow"], alpha=0.55, edgecolors="none")
    lim = np.nanpercentile(np.r_[use["beta_up"], use["beta_down"]], [2, 98])
    axes[0].plot(lim, lim, color="0.5", lw=0.6)
    axes[0].set_xlim(lim)
    axes[0].set_ylim(lim)
    axes[0].set_xlabel("Upstream site slope (log R on GP)")
    axes[0].set_ylabel("Downstream site slope (log R on GP)")
    letter(axes[0], "a")

    axes[1].hist(use["delta"], bins=24, color=PAL["flow"], alpha=0.8, edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="0.35", lw=0.7, ls="--")
    axes[1].axvline(use["delta"].mean(), color=PAL["pos"], lw=1.1)
    axes[1].set_xlabel("Downstream − upstream slope")
    axes[1].set_ylabel("Nested pairs")
    letter(axes[1], "b")

    fig.tight_layout()
    save_pub(fig, "si_s4_nested_pairs", outdir=SI)
    plt.close(fig)


def fig_s5_month_load(tab: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    labels, b, lo, hi, cols = [], [], [], [], []
    for param in ("TN", "NO3N", "TP"):
        for fam, suffix, c in (
            ("cxr_ref", "annual C×R", PARAM_COLOR[param]),
            ("month_load", "monthly Σ C$_m$R$_m$", PAL["gray"]),
        ):
            y = "logLoad" if fam == "cxr_ref" else "logL_month"
            r = _row(tab, fam, param, y)
            if r is None or not bool(r.get("ok", False)):
                continue
            labels.append(f"{PARAM_LAB[param]}, {suffix}")
            b.append(float(r["beta_GP"]))
            lo.append(float(r["ci95_lo_GP"]))
            hi.append(float(r["ci95_hi_GP"]))
            cols.append(c)
    if not labels:
        print("skip S5")
        return
    lollipop(ax, labels, b, lo, hi, colors=cols)
    ax.set_xlabel("Coefficient on GP")
    letter(ax, "a")
    fig.tight_layout()
    save_pub(fig, "si_s5_month_load", outdir=SI)
    plt.close(fig)


def main() -> None:
    tab = pd.read_csv(DATA / "regression_deepen.csv")
    fig_s3_filter(tab)
    fig_s4_pairs()
    fig_s5_month_load(tab)
    print("done SI deepen figures")


if __name__ == "__main__":
    main()
