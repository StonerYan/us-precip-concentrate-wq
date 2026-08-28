"""Figure 6 — main-text.

Conclusion: after the same wet-day thinning, GP lengthens the South
Q10 tail and that tail carries South dry-season phosphorus load; the
Northeast tail does not lengthen and Northeast dry-season TP is a
year-scale concentration sign.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec
import pandas as pd

from config import DATA, FIG
from fig_style import (
    HAS_CARTOPY,
    PAL,
    letter,
    lollipop,
    map_points,
    map_proj,
    save_pub,
)
from utils import site_slopes


def _row(tab, family, sample, y):
    d = tab.loc[(tab["family"] == family) & (tab["sample"] == sample) & (tab["y"] == y)]
    return None if d.empty else d.iloc[0]


def _proj():
    return map_proj() if HAS_CARTOPY else None


def main() -> None:
    tr = pd.read_csv(DATA / "regression_regime.csv")
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    lf = pd.read_parquet(DATA / "panel_lowflow.parquet")
    hydro["usgs_site_no"] = hydro["usgs_site_no"].astype(str).str.zfill(8)
    lf["usgs_site_no"] = lf["usgs_site_no"].astype(str).str.zfill(8)
    h = hydro.merge(lf, on=["usgs_site_no", "year"], how="left")
    sl = site_slopes(h, "frac_q10")
    meta = hydro.groupby("site_key", as_index=False).agg(lat=("lat", "first"), lon=("lon", "first"), region=("region", "first"))
    sl = sl.merge(meta, on="site_key", how="left")

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.15, 1.0], hspace=0.34)

    ax0 = fig.add_subplot(gs[0], projection=_proj()) if HAS_CARTOPY else fig.add_subplot(gs[0])
    d = sl.dropna(subset=["lat", "lon", "beta_GP"]).copy()
    t = d["beta_GP"].abs() / d["se_GP"].replace(0, np.nan)
    d["_sz"] = 8 + 14 * np.clip(t.fillna(0) / 3.0, 0, 1)
    d = d.sort_values("_sz", kind="mergesort")
    nrm = TwoSlopeNorm(vmin=-1.2, vcenter=0, vmax=1.2)
    sc = map_points(
        ax0, d["lon"], d["lat"], c=d["beta_GP"], s=d["_sz"],
        cmap="RdBu_r", norm=nrm, alpha=0.38, edgecolors="none", linewidths=0,
    )
    fig.canvas.draw()
    pos = ax0.get_position()
    scale = 1.21
    new_w, new_h = pos.width * scale, pos.height * scale
    new_x = max(0.005, pos.x0 - 0.5 * (new_w - pos.width) - 0.10 * new_w)
    ax0.set_position([
        new_x,
        pos.y0 - 0.25 * (new_h - pos.height),
        new_w, new_h,
    ])
    if hasattr(ax0, "patch"):
        ax0.patch.set_alpha(0.0)
    cb = fig.colorbar(sc, ax=ax0, shrink=0.82, pad=0.015, fraction=0.025)
    cb.set_label(r"Site slope of $Q_{10}$-day fraction on GP")
    letter(ax0, "a", x=-0.02, y=1.03)

    fig.canvas.draw()
    pos_a = ax0.get_position()
    pos_cb = cb.ax.get_position()
    left = pos_a.x0
    right = pos_cb.x1
    ax_slot = fig.add_subplot(gs[1])
    pos_bot = ax_slot.get_position()
    ax_slot.remove()
    label_pad = 0.115
    gutter = 0.018
    each_w = (right - left - label_pad - gutter) / 2.0
    ax1 = fig.add_axes([left, pos_bot.y0, each_w, pos_bot.height])
    specs = [
        ("Northeast", "Northeast", PAL["ne"]),
        ("South", "South", PAL["south"]),
        ("Interior", "Interior", "#3d6b3d"),
        ("West", "West", "#7a5a3a"),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, samp, c in specs:
        r = _row(tr, "lowflow", samp, "frac_q10")
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(ax1, labs, b, lo, hi, colors=cols, xlabel="Coefficient on GP")
    ax1.set_title(r"Days at or below $Q_{10}$", fontsize=8, pad=2)
    letter(ax1, "b")

    ax2 = fig.add_axes([left + each_w + label_pad + gutter, pos_bot.y0, each_w, pos_bot.height])
    specs2 = [
        ("Northeast", "load_dry", "TP_NE", PAL["ne"]),
        (r"Northeast, $Q_{10}$ in", "load_dry_ctrl_q10", "TP_NE", PAL["gray"]),
        ("South", "load_dry", "TP_South", PAL["south"]),
        (r"South, $Q_{10}$ in", "load_dry_ctrl_q10", "TP_South", PAL["gray"]),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, fam, samp, c in specs2:
        r = _row(tr, fam, samp, "logLoad_dry")
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(ax2, labs, b, lo, hi, colors=cols, xlabel="Coefficient on GP")
    ax2.set_title(r"Dry-season TP $C \times R$", fontsize=8, pad=2)
    letter(ax2, "c")

    save_pub(fig, "fig6_tail")
    plt.close(fig)


if __name__ == "__main__":
    main()
