"""CONUS maps, density contours, rainclouds. Native GridSpec."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from config import DATA, FIG
from fig_style import (
    HAS_CARTOPY,
    PAL,
    PARAM_COLOR,
    PARAM_LAB,
    REG_COL,
    REG_LAB,
    dumbbell,
    kde_contour,
    letter,
    lollipop,
    lowess_xy,
    map_points,
    map_proj,
    raincloud,
    save_pub,
    style_conus,
    style_conus_locator,
)
from regions import REG_FILL


def load():
    hydro = pd.read_parquet(DATA / "panel_hydro.parquet")
    chem = pd.read_parquet(DATA / "panel_us_paired.parquet")
    cq = pd.read_parquet(DATA / "panel_cq_month.parquet")
    rh = pd.read_csv(DATA / "regression_hydro.csv")
    rc = pd.read_csv(DATA / "regression_conflict.csv")
    rr = pd.read_csv(DATA / "regression_robust.csv")
    bins_h = pd.read_csv(DATA / "partial_bins_hydro.csv") if (DATA / "partial_bins_hydro.csv").exists() else pd.DataFrame()
    bins_c = pd.read_csv(DATA / "partial_bins_conflict.csv") if (DATA / "partial_bins_conflict.csv").exists() else pd.DataFrame()
    slopes = pd.read_parquet(DATA / "site_slopes_runoff.parquet")
    seasons = pd.read_parquet(DATA / "usgs_q_seasons.parquet")
    sites = hydro.groupby("usgs_site_no", as_index=False).agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        GP=("GP", "mean"),
        region=("region", "first"),
        climate_region=("climate_region", "first") if "climate_region" in hydro.columns else ("region", "first"),
        area=("area", "median"),
        corn_belt=("corn_belt", "first") if "corn_belt" in hydro.columns else ("region", "first"),
    )
    loc = hydro.groupby("site_key").agg(lat=("lat", "median"), lon=("lon", "median"), region=("region", "first"))
    slopes = slopes.merge(loc, on="site_key", how="left")
    sl_tp = pd.read_parquet(DATA / "site_slopes_tp_dry.parquet") if (DATA / "site_slopes_tp_dry.parquet").exists() else pd.DataFrame()
    sl_no3 = pd.read_parquet(DATA / "site_slopes_no3_dry.parquet") if (DATA / "site_slopes_no3_dry.parquet").exists() else pd.DataFrame()
    cloc = chem.groupby("site_key").agg(
        lat=("lat", "median"),
        lon=("lon", "median"),
        region=("region", "first"),
        crop_hi=("crop_hi", "first") if "crop_hi" in chem.columns else ("region", "first"),
        corn_belt=("corn_belt", "first") if "corn_belt" in chem.columns else ("region", "first"),
    )
    if len(sl_tp):
        sl_tp = sl_tp.merge(cloc, on="site_key", how="left")
    if len(sl_no3):
        sl_no3 = sl_no3.merge(cloc, on="site_key", how="left")
    return {
        "hydro": hydro, "chem": chem, "cq": cq, "rh": rh, "rc": rc, "rr": rr,
        "bins_h": bins_h, "bins_c": bins_c, "slopes": slopes, "seasons": seasons,
        "sites": sites, "sl_tp": sl_tp, "sl_no3": sl_no3,
    }


def _row(df, **kw):
    d = df.copy()
    for k, v in kw.items():
        d = d.loc[d[k] == v]
    if d.empty:
        return None
    return d.iloc[0]


def residualize(df, y, x="GP"):
    d = df.dropna(subset=[y, x, "logP", "site_key"]).copy()
    for col in (y, x, "logP"):
        d[col + "_d"] = d[col] - d.groupby("site_key")[col].transform("mean")
    X = np.c_[np.ones(len(d)), d["logP_d"].to_numpy()]
    by, *_ = np.linalg.lstsq(X, d[y + "_d"].to_numpy(), rcond=None)
    bg, *_ = np.linalg.lstsq(X, d[x + "_d"].to_numpy(), rcond=None)
    d["y_r"] = d[y + "_d"].to_numpy() - X @ by
    d["g_r"] = d[x + "_d"].to_numpy() - X @ bg
    return d


def _proj():
    return map_proj()


def _map_cell(fig, spec):
    """Map + horizontal colorbar inside one 2×2 cell; map keeps the full cell width."""
    inner = GridSpecFromSubplotSpec(
        2, 1, subplot_spec=spec, height_ratios=[1.0, 0.078], hspace=0.08,
    )
    ax = fig.add_subplot(inner[0], projection=_proj()) if HAS_CARTOPY else fig.add_subplot(inner[0])
    cax = fig.add_subplot(inner[1])
    return ax, cax


def _hcolorbar(fig, sc, label, ax=None, cax=None):
    if cax is not None:
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    else:
        cb = fig.colorbar(
            sc, ax=ax, orientation="horizontal",
            fraction=0.055, pad=0.08, aspect=28, shrink=0.90,
        )
    cb.set_label(label, fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5, pad=1)
    return cb


def _enlarge_conus(ax, scale=1.24):
    """Map vs the 2×2 cell; then shrink 1/5 and shift left by 1/10 of map width."""
    fig = ax.figure
    fig.canvas.draw()
    pos = ax.get_position()
    new_w = pos.width * scale
    new_h = pos.height * scale
    new_x = max(0.005, pos.x0 - 0.15 * (new_w - pos.width) - 0.10 * new_w)
    new_y = pos.y0 - 0.10 * (new_h - pos.height)
    ax.set_position([new_x, new_y, new_w, new_h])
    ax.set_zorder(5)
    if hasattr(ax, "patch"):
        ax.patch.set_alpha(0.0)
    fig.canvas.draw()
    return ax.get_position()


def _slope_points(ax, sl, s, vmin, vmax):
    """Translucent overlay: small |t| first so later points blend instead of hiding."""
    sl = sl.copy()
    t = sl["beta_GP"].abs() / sl["se_GP"].replace(0, np.nan)
    sl["_sz"] = s[0] + s[1] * np.clip(t.fillna(0) / 4.0, 0, 1)
    sl = sl.sort_values("_sz", kind="mergesort")
    nrm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
    return map_points(
        ax, sl["lon"], sl["lat"], c=sl["beta_GP"], s=sl["_sz"],
        cmap="RdBu_r", norm=nrm, alpha=0.38, edgecolors="none", linewidths=0,
    )


def _place_map_cbar(fig, ax, sc, label, cax=None):
    """Horizontal colorbar under the map."""
    if cax == "below":
        pos = _enlarge_conus(ax, scale=1.24)
        cax = fig.add_axes([pos.x0 + 0.08 * pos.width, pos.y0 - 0.026, pos.width * 0.84, 0.014])
        cax.set_zorder(6)
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    elif cax is not None:
        cb = fig.colorbar(sc, cax=cax, orientation="horizontal")
    else:
        cb = fig.colorbar(
            sc, ax=ax, orientation="horizontal",
            fraction=0.055, pad=0.08, aspect=28, shrink=0.90,
        )
    cb.set_label(label, fontsize=6.2)
    cb.ax.tick_params(labelsize=5.5, pad=1)
    return cb


# ---------- Fig 1 ----------
def fig1a_concept(d=None, ax=None):
    """Schematic: amount–timing chain plus CONUS locator for NE / South."""
    close = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7.1, 3.85))
        pos = type("P", (), {"x0": 0.02, "y0": 0.04, "width": 0.96, "height": 0.93})()
    else:
        fig = ax.figure
        fig.canvas.draw()
        pos = ax.get_position()
        ax.remove()

    ax_path = fig.add_axes([pos.x0, pos.y0 + 0.62 * pos.height, pos.width, 0.36 * pos.height])
    ax_map = (
        fig.add_axes([pos.x0, pos.y0, 0.64 * pos.width, 0.58 * pos.height], projection=_proj())
        if HAS_CARTOPY
        else fig.add_axes([pos.x0, pos.y0, 0.64 * pos.width, 0.58 * pos.height])
    )
    ax_n = fig.add_axes([pos.x0 + 0.66 * pos.width, pos.y0 + 0.04 * pos.height, 0.33 * pos.width, 0.52 * pos.height])

    ax_path.set_xlim(0, 12.2)
    ax_path.set_ylim(0, 2.15)
    ax_path.axis("off")

    def box(a, x, y, w, h, fc, text, fs=6.0, ec="#2c3e50"):
        a.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.03,rounding_size=0.07",
                facecolor=fc, edgecolor=ec, lw=0.55, zorder=2,
            )
        )
        a.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=3, linespacing=1.2)

    def arrow(a, x1, y1, x2, y2, col="#2c3e50"):
        a.annotate(
            "", xy=(x2, y2), xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color=col, lw=1.05, mutation_scale=8),
            zorder=1,
        )

    ax_path.text(0.15, 1.92, "Amount–timing pathway", fontsize=7.5, fontweight="bold", ha="left")
    box(ax_path, 0.15, 0.72, 2.05, 0.95, "#eef2f5", "Same annual $P$\n(amount held)")
    arrow(ax_path, 2.25, 1.20, 2.55, 1.20)
    box(ax_path, 2.60, 0.72, 2.85, 0.95, "#f3d6d0", "Higher GP\nyear-scale daily inequality")
    arrow(ax_path, 5.50, 1.20, 5.80, 1.20, PAL["tp"])
    box(ax_path, 5.85, 0.72, 1.85, 0.95, "#e8b4ae", "Runoff $R$ down")
    arrow(ax_path, 7.75, 1.20, 8.05, 1.20, PAL["tp"])
    box(ax_path, 8.10, 0.72, 3.90, 0.95, "#f7efe4", "National $C\\times R$ down\nunless a named object reverses it")
    ax_path.text(2.60, 0.42, "not a wet-day count", fontsize=5.4, color="#5d6d7e", ha="left")
    ax_path.text(0.15, 0.08, "Named exceptions sit on the climate-region map", fontsize=6.2, color="#5d6d7e", style="italic")
    letter(ax_path, "a", x=-0.01, y=1.02)

    style_conus_locator(ax_map)
    geo = {}
    if HAS_CARTOPY:
        import cartopy.crs as ccrs
        geo = {"transform": ccrs.PlateCarree()}
    ax_map.text(-74.2, 42.6, "Northeast", fontsize=6.3, color=PAL["ne"], fontweight="medium", ha="center", zorder=6, **geo)
    ax_map.text(-74.2, 39.7, "year-scale $C$\ndry TP load up\n$Q_{10}$ tail does not lengthen", fontsize=5.2, color="#2c3e50", ha="center", zorder=6, **geo)
    ax_map.text(-90.5, 32.4, "South", fontsize=6.3, color=PAL["south"], fontweight="medium", ha="center", zorder=6, **geo)
    ax_map.text(-90.5, 28.8, "$Q_{10}$ tail lengthens\ndry P load follows tail", fontsize=5.2, color="#2c3e50", ha="center", zorder=6, **geo)

    ax_n.set_xlim(0, 1)
    ax_n.set_ylim(0, 1)
    ax_n.axis("off")
    box(ax_n, 0.04, 0.08, 0.92, 0.84, "#eef3ea", "", ec=PAL["crop"])
    ax_n.text(0.50, 0.82, "Dry-season nitrate", ha="center", fontsize=6.4, color=PAL["crop"], fontweight="medium")
    ax_n.text(0.50, 0.64, "not a climate region", ha="center", fontsize=5.2, color="#5d6d7e", style="italic")
    ax_n.text(0.50, 0.48, "low-crop: $C$ rises", ha="center", fontsize=6.0)
    ax_n.text(0.50, 0.34, "high-crop: $C$ already with $Q$", ha="center", fontsize=6.0)
    ax_n.text(0.50, 0.20, "load still follows $R$", ha="center", fontsize=6.1, color=PAL["crop"])
    ax_n.text(0.50, 0.08, "South wet NO$_3$: wet-season $C$", ha="center", fontsize=5.2, color="#5d6d7e")

    if close:
        save_pub(fig, "fig1a_concept", FIG / "panels")
        save_pub(fig, "concept_amount_timing")
        plt.close(fig)
    return ax_map


def fig1a_sites(d, ax=None, panel="a"):
    close = ax is None
    if ax is None:
        fig = plt.figure(figsize=(4.4, 2.8))
        ax = fig.add_subplot(111, projection=_proj()) if HAS_CARTOPY else fig.add_subplot(111)
    else:
        fig = ax.figure
    u = d["sites"].dropna(subset=["lat", "lon"])
    for reg, col in REG_COL.items():
        sub = u.loc[u["region"] == reg]
        if len(sub):
            map_points(ax, sub["lon"], sub["lat"], c=col, s=8, alpha=0.88)
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=c, markersize=4.5, label=REG_LAB[k])
        for k, c in REG_COL.items()
    ]
    ax.legend(handles=handles, loc="lower left", fontsize=5.6, ncol=2, borderpad=0.2)
    letter(ax, panel, x=-0.02, y=1.02)
    if close:
        save_pub(fig, "fig1a_sites", FIG / "panels")
        plt.close(fig)
    return ax


def fig1b_gp(d, ax=None):
    close = ax is None
    if ax is None:
        fig = plt.figure(figsize=(4.4, 2.8))
        ax = fig.add_subplot(111, projection=_proj()) if HAS_CARTOPY else fig.add_subplot(111)
    else:
        fig = ax.figure
    u = d["sites"].dropna(subset=["lat", "lon", "GP"])
    sc = map_points(ax, u["lon"], u["lat"], c=u["GP"], s=9, cmap="YlOrBr", vmin=0.78, vmax=0.92)
    cb = fig.colorbar(sc, ax=ax, shrink=0.72, pad=0.02, fraction=0.035)
    cb.set_label("Mean annual GP")
    letter(ax, "b")
    if close:
        save_pub(fig, "fig1b_gp", FIG / "panels")
        plt.close(fig)
    return ax


def fig1c_drymonths(d, ax=None, panel="c"):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.2, 2.8), subplot_kw={"projection": "polar"})
    seasons = d["seasons"]
    months = pd.concat([seasons["dry_m1"], seasons["dry_m2"], seasons["dry_m3"]])
    counts = months.value_counts().reindex(range(1, 13), fill_value=0).to_numpy().astype(float)
    theta = np.linspace(0, 2 * np.pi, 12, endpoint=False)
    width = 2 * np.pi / 12 * 0.86
    cols = ["#1a365d" if m in (7, 8, 9, 10) else "#8aa0b3" for m in range(1, 13)]
    if ax.name != "polar":
        # compose may have passed a cartesian axis; draw a compact polar inset
        fig = ax.figure
        pos = ax.get_position()
        ax.axis("off")
        ax = fig.add_axes(pos, projection="polar")
    ax.set_theta_zero_location("N")
    ax.set_theta_direction(-1)
    ax.bar(theta, counts, width=width, color=cols, edgecolor="white", linewidth=0.4, align="edge")
    ax.set_xticks(theta + width / 2)
    ax.set_xticklabels(list("JFMAMJJASOND"), fontsize=6)
    ax.set_yticks([])
    ax.spines["polar"].set_linewidth(0.4)
    ax.set_title("Low-flow months (site $Q$)", fontsize=7, pad=8)
    letter(ax, panel, x=-0.12, y=1.08)
    if close:
        save_pub(fig, "fig1c_drymonths", FIG / "panels")
        plt.close(fig)
    return ax


def fig1d_gpsd(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.4))
    sd = d["hydro"].groupby("site_key")["GP"].std().dropna()
    ax.hist(sd, bins=22, color="#4a6741", alpha=0.88, edgecolor="white", linewidth=0.3, density=True)
    xs = np.linspace(sd.min(), sd.max(), 120)
    from scipy.stats import gaussian_kde
    ax.plot(xs, gaussian_kde(sd)(xs), color="#1a365d", lw=1.1)
    ax.axvline(sd.median(), color=PAL["tp"], lw=1.1)
    ax.text(sd.median() + 0.002, ax.get_ylim()[1] * 0.88, f"median {sd.median():.3f}", color=PAL["tp"], fontsize=6.5)
    ax.set_xlabel("Within-site SD of annual GP")
    ax.set_ylabel("Density")
    letter(ax, "d")
    if close:
        save_pub(fig, "fig1d_gpsd", FIG / "panels")
        plt.close(fig)
    return ax


def compose_fig1(d):
    fig = plt.figure(figsize=(7.1, 3.35))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.45, 0.85], wspace=0.16)
    axa = fig.add_subplot(gs[0], projection=_proj()) if HAS_CARTOPY else fig.add_subplot(gs[0])
    axb = fig.add_subplot(gs[1], projection="polar")
    fig1a_sites(d, axa, panel="a")
    fig1c_drymonths(d, axb, panel="b")
    save_pub(fig, "fig1_coverage")
    plt.close(fig)


# ---------- Fig 2 ----------
def fig2a_slope_map(d, ax=None, cax=None):
    close = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7.1, 3.15))
        ax = fig.add_subplot(111, projection=_proj()) if HAS_CARTOPY else fig.add_subplot(111)
    else:
        fig = ax.figure
    sl = d["slopes"].dropna(subset=["lat", "lon", "beta_GP"]).copy()
    sc = _slope_points(ax, sl, s=(8, 18), vmin=-4.0, vmax=2.0)
    _place_map_cbar(fig, ax, sc, r"Site slope of $\log R$ on GP", cax)
    letter(ax, "a", x=0.01, y=0.98)
    if close:
        save_pub(fig, "fig2a_slope_map", FIG / "panels")
        plt.close(fig)
    return ax


def fig2b_resid_hex(d, ax=None):
    """2D density + LOWESS. Not a mean±SD hex overlay."""
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    res = residualize(d["hydro"], "logR")
    x, y = res["g_r"].to_numpy(), res["y_r"].to_numpy()
    kde_contour(ax, x, y, PAL["flow"], levels=7, alpha=0.22)
    rng = np.random.default_rng(3)
    if len(x) > 2500:
        idx = rng.choice(len(x), 2500, replace=False)
        ax.scatter(x[idx], y[idx], s=2.5, alpha=0.08, color=PAL["flow"], linewidths=0, zorder=2)
    xs, ys = lowess_xy(x, y, frac=0.4)
    ax.plot(xs, ys, color="#0b1f33", lw=1.45, zorder=4)
    ax.axhline(0, color="0.5", lw=0.45, ls="--")
    ax.axvline(0, color="0.5", lw=0.45, ls="--")
    ax.set_xlim(-0.08, 0.08)
    ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel("GP residual (hold $P$, site FE)")
    ax.set_ylabel(r"$\log R$ residual")
    letter(ax, "b")
    if close:
        save_pub(fig, "fig2b_resid_hex", FIG / "panels")
        plt.close(fig)
    return ax


def fig2c_bins(d, ax=None):
    """Raincloud of site runoff slopes by climate-region class."""
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sl = d["slopes"].dropna(subset=["beta_GP", "region"])
    order = ["Northeast", "South", "Interior", "West"]
    series = [sl.loc[sl["region"] == r, "beta_GP"].to_numpy() for r in order]
    raincloud(ax, series, [REG_LAB[r] for r in order], [REG_COL[r] for r in order],
              xlabel=r"Site slope of $\log R$ on GP")
    letter(ax, "c")
    if close:
        save_pub(fig, "fig2c_bins", FIG / "panels")
        plt.close(fig)
    return ax


def fig2d_decomp(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 2.3))
    specs = [
        ("Annual $R$", "all", "logR", PAL["flow"]),
        ("Dry-season $R$", "all", "logR_dry", "#6b3a2a"),
        ("Wet-season $R$", "all", "logR_wet", "#2a5f6b"),
        ("Small basins", "small", "logR", "#6d7a86"),
        ("Large basins", "large", "logR", "#1a365d"),
        ("Northeast", "Northeast", "logR", REG_COL["Northeast"]),
        ("South", "South", "logR", REG_COL["South"]),
        ("Interior", "Interior", "logR", REG_COL["Interior"]),
        ("West", "West", "logR", REG_COL["West"]),
    ]
    labels, beta, lo, hi, cols = [], [], [], [], []
    for lab, sample, y, col in specs:
        r = _row(d["rh"], family="hydro", **{"sample": sample}, y=y)
        if r is None or not bool(r["ok"]):
            continue
        labels.append(lab)
        beta.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(col)
    lollipop(ax, labels, beta, lo, hi, colors=cols)
    letter(ax, "d")
    if close:
        save_pub(fig, "fig2d_decomp", FIG / "panels")
        plt.close(fig)
    return ax


def compose_fig2(d):
    fig = plt.figure(figsize=(7.1, 6.20))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.40)
    axa = fig.add_subplot(gs[0, 0], projection=_proj()) if HAS_CARTOPY else fig.add_subplot(gs[0, 0])
    axb = fig.add_subplot(gs[0, 1])
    axc = fig.add_subplot(gs[1, 0])
    axd = fig.add_subplot(gs[1, 1])
    fig2b_resid_hex(d, axb)
    fig2c_bins(d, axc)
    fig2d_decomp(d, axd)
    fig2a_slope_map(d, axa, cax="below")
    save_pub(fig, "fig2_runoff")
    plt.close(fig)


# ---------- Fig 3 ----------
def fig3a_cr_plane(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    p = d["chem"].dropna(subset=["logC", "logR", "logLoad", "GP", "logP"])
    param_lw = {"TN": 0.95, "NO3N": 1.45, "TP": 2.10}
    param_z = {"TN": 2, "NO3N": 3, "TP": 4}
    for param, col in [("TN", PAL["tn"]), ("NO3N", PAL["no3"]), ("TP", PAL["tp"])]:
        sub = residualize(p.loc[p["param"] == param], "logC")
        rr = residualize(p.loc[p["param"] == param], "logR")
        m = sub[["site_key", "year", "y_r"]].merge(rr[["site_key", "year", "y_r"]], on=["site_key", "year"], suffixes=("_c", "_r"))
        kde_contour(
            ax, m["y_r_r"], m["y_r_c"], col,
            levels=4, alpha=0.20, linewidths=param_lw[param], zorder=param_z[param],
        )
    ax.axhline(0, color="0.5", lw=0.45, ls="--")
    ax.axvline(0, color="0.5", lw=0.45, ls="--")
    ax.set_xlabel(r"$\log R$ residual")
    ax.set_ylabel(r"$\log C$ residual")
    ax.legend(
        handles=[
            Line2D([0], [0], color=PARAM_COLOR[k], lw=param_lw[k], label=PARAM_LAB[k])
            for k in ("TN", "NO3N", "TP")
        ],
        loc="upper right", fontsize=6,
    )
    letter(ax, "a")
    if close:
        save_pub(fig, "fig3a_cr_plane", FIG / "panels")
        plt.close(fig)
    return ax


def fig3b_load_hex(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    p = d["chem"].loc[d["chem"]["param"] == "TN"]
    res = residualize(p, "logLoad")
    x, y = res["g_r"].to_numpy(), res["y_r"].to_numpy()
    kde_contour(ax, x, y, "#5b4b8a", levels=7, alpha=0.22)
    rng = np.random.default_rng(5)
    if len(x) > 2000:
        idx = rng.choice(len(x), 2000, replace=False)
        ax.scatter(x[idx], y[idx], s=2.4, alpha=0.08, color="#5b4b8a", linewidths=0)
    xs, ys = lowess_xy(x, y, frac=0.4)
    ax.plot(xs, ys, color="#2a1f4a", lw=1.35)
    ax.axhline(0, color="0.5", lw=0.45, ls="--")
    ax.axvline(0, color="0.5", lw=0.45, ls="--")
    ax.set_xlabel("GP residual")
    ax.set_ylabel(r"$\log(C \times R)$ residual (TN)")
    letter(ax, "b")
    if close:
        save_pub(fig, "fig3b_load_hex", FIG / "panels")
        plt.close(fig)
    return ax


def fig3c_channels(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    xs = np.array([0, 1, 2])
    outcomes = [("logR", "Runoff"), ("logC", "C"), ("logLoad", r"$C\times R$")]
    for param in ["TN", "NO3N", "TP"]:
        betas, lo, hi = [], [], []
        for y, _ in outcomes:
            r = _row(d["rc"], family="load", param=param, **{"sample": "paired"}, y=y)
            betas.append(float(r["beta_GP"]) if r is not None else np.nan)
            lo.append(float(r["ci95_lo_GP"]) if r is not None else np.nan)
            hi.append(float(r["ci95_hi_GP"]) if r is not None else np.nan)
        col = PARAM_COLOR[param]
        lw = {"TN": 1.10, "NO3N": 1.50, "TP": 2.05}[param]
        ax.plot(xs, betas, color=col, lw=lw, zorder=2)
        for i, (b, a, c) in enumerate(zip(betas, lo, hi)):
            ax.plot([i, i], [a, c], color=col, lw=1.5, solid_capstyle="round", alpha=0.85)
            ax.scatter([i], [b], s=28, color=col, edgecolors="white", linewidths=0.45, zorder=3)
    ax.axhline(0, color="0.35", lw=0.55, ls="--")
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab in outcomes])
    ax.set_ylabel("Coefficient on GP")
    ax.legend(
        handles=[
            Line2D([0], [0], color=PARAM_COLOR[k], marker="o", lw={"TN": 1.10, "NO3N": 1.50, "TP": 2.05}[k], label=PARAM_LAB[k])
            for k in ("TN", "NO3N", "TP")
        ],
        fontsize=6, loc="upper right",
    )
    letter(ax, "c")
    if close:
        save_pub(fig, "fig3c_channels", FIG / "panels")
        plt.close(fig)
    return ax


def fig3d_season_load(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    y = np.arange(3)[::-1]
    for i, param in enumerate(["TN", "NO3N", "TP"]):
        rd = _row(d["rc"], family="load", param=param, **{"sample": "paired"}, y="logLoad_dry")
        rw = _row(d["rc"], family="load", param=param, **{"sample": "paired"}, y="logLoad_wet")
        if rd is None or rw is None:
            continue
        yi = y[i]
        ax.annotate(
            "",
            xy=(float(rw["beta_GP"]), yi),
            xytext=(float(rd["beta_GP"]), yi),
            arrowprops=dict(arrowstyle="-|>", color=PARAM_COLOR[param], lw=1.3, mutation_scale=8),
        )
        ax.scatter([float(rd["beta_GP"])], [yi], s=30, color=PARAM_COLOR[param], zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter([float(rw["beta_GP"])], [yi], s=30, facecolors="white", edgecolors=PARAM_COLOR[param], linewidths=1.1, zorder=3)
        ax.plot([float(rd["ci95_lo_GP"]), float(rd["ci95_hi_GP"])], [yi - 0.12, yi - 0.12], color=PARAM_COLOR[param], lw=1.1, alpha=0.45)
        ax.plot([float(rw["ci95_lo_GP"]), float(rw["ci95_hi_GP"])], [yi + 0.12, yi + 0.12], color=PARAM_COLOR[param], lw=1.1, alpha=0.45)
    ax.axvline(0, color="0.35", lw=0.55, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels([PARAM_LAB[k] for k in ("TN", "NO3N", "TP")])
    ax.set_xlabel("GP coefficient on seasonal $C \\times R$")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor="#444", markersize=5, label="Dry"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor="white", markeredgecolor="#444", markersize=5, label="Wet"),
        ],
        fontsize=6, loc="upper right", bbox_to_anchor=(0.90, 1.02),
    )
    letter(ax, "d")
    if close:
        save_pub(fig, "fig3d_season_load", FIG / "panels")
        plt.close(fig)
    return ax


def compose_fig3(d):
    fig = plt.figure(figsize=(7.1, 5.7))
    gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.36)
    fig3a_cr_plane(d, fig.add_subplot(gs[0, 0]))
    fig3b_load_hex(d, fig.add_subplot(gs[0, 1]))
    fig3c_channels(d, fig.add_subplot(gs[1, 0]))
    fig3d_season_load(d, fig.add_subplot(gs[1, 1]))
    save_pub(fig, "fig3_load")
    plt.close(fig)


# ---------- Fig 4 ----------
def fig4a_ne_south(d, ax=None, cax=None):
    """Map of site-level dry-season TP slopes."""
    close = ax is None
    if ax is None:
        fig = plt.figure(figsize=(7.1, 3.1))
        ax = fig.add_subplot(111, projection=_proj()) if HAS_CARTOPY else fig.add_subplot(111)
    else:
        fig = ax.figure
    sl = d["sl_tp"].dropna(subset=["lat", "lon", "beta_GP"]) if len(d["sl_tp"]) else pd.DataFrame()
    if sl.empty:
        letter(ax, "a")
        return ax
    sc = _slope_points(ax, sl, s=(9, 16), vmin=-4.0, vmax=4.0)
    _place_map_cbar(fig, ax, sc, "Site slope, dry-season log TP on GP", cax)
    letter(ax, "a", x=0.01, y=0.98)
    if close:
        save_pub(fig, "fig4a_ne_south", FIG / "panels")
        plt.close(fig)
    return ax


def fig4b_heatmap(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sl = d["sl_tp"].dropna(subset=["beta_GP", "region"]) if len(d["sl_tp"]) else pd.DataFrame()
    order = ["Northeast", "South", "Interior", "West"]
    series = [sl.loc[sl["region"] == r, "beta_GP"].to_numpy() for r in order] if len(sl) else [np.array([])] * 4
    raincloud(ax, series, [REG_LAB[r] for r in order], [REG_COL[r] for r in order],
              xlabel="Site slope, dry log TP on GP")
    letter(ax, "b")
    if close:
        save_pub(fig, "fig4b_heatmap", FIG / "panels")
        plt.close(fig)
    return ax


def fig4c_cq(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    cq = d["cq"]
    cq = cq.loc[(cq["param"] == "TP") & (cq["is_dry"] == True) & cq["region"].isin(["Northeast", "South"])]
    for reg, col in [("Northeast", PAL["ne"]), ("South", PAL["south"])]:
        sub = cq.loc[cq["region"] == reg].dropna(subset=["logC", "logQ", "site_key"]).copy()
        for coln in ["logC", "logQ"]:
            sub[coln] = sub[coln] - sub.groupby("site_key")[coln].transform("mean")
        kde_contour(ax, sub["logQ"], sub["logC"], col, levels=5, alpha=0.22)
        if len(sub) > 80:
            xs, ys = lowess_xy(sub["logQ"], sub["logC"], frac=0.5)
            ax.plot(xs, ys, color=col, lw=1.4, label=REG_LAB[reg])
    ax.axhline(0, color="0.5", lw=0.45, ls="--")
    ax.axvline(0, color="0.5", lw=0.45, ls="--")
    ax.set_xlabel(r"$\log Q$ residual (monthly)")
    ax.set_ylabel("log dry-month TP residual")
    ax.legend(fontsize=6)
    letter(ax, "c")
    if close:
        save_pub(fig, "fig4c_cq", FIG / "panels")
        plt.close(fig)
    return ax


def fig4d_slices(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    labels, left, right, llo, lhi, rlo, rhi = [], [], [], [], [], [], []
    pairs = [
        ("Northeast", "South", "logC_low", "Dry TP"),
        ("Northeast", "South", "logLoad_dry", "Dry $C\\times R$"),
        ("NE_urb_lo", "NE_urb_hi", "logC_low", "NE urban lo / hi"),
        ("NE_small", "South_small", "logC_low", "Small basins"),
    ]
    for a, b, y, lab in pairs:
        ra = _row(d["rc"], family="conflict_tp", param="TP", **{"sample": a}, y=y)
        rb = _row(d["rc"], family="conflict_tp", param="TP", **{"sample": b}, y=y)
        if ra is None:
            ra = _row(d["rc"], family="load_region", param="TP", **{"sample": a}, y=y)
        if rb is None:
            rb = _row(d["rc"], family="load_region", param="TP", **{"sample": b}, y=y)
        if ra is None or rb is None or not bool(ra.get("ok", True)) or not bool(rb.get("ok", True)):
            continue
        if pd.isna(ra.get("beta_GP")) or pd.isna(rb.get("beta_GP")):
            continue
        labels.append(lab)
        left.append(float(ra["beta_GP"]))
        right.append(float(rb["beta_GP"]))
        llo.append(float(ra["ci95_lo_GP"]))
        lhi.append(float(ra["ci95_hi_GP"]))
        rlo.append(float(rb["ci95_lo_GP"]))
        rhi.append(float(rb["ci95_hi_GP"]))
    dumbbell(ax, labels, left, right, llo, lhi, rlo, rhi,
             c_left=PAL["ne"], c_right=PAL["south"], lab_left="Left of pair", lab_right="Right of pair")
    letter(ax, "d")
    if close:
        save_pub(fig, "fig4d_slices", FIG / "panels")
        plt.close(fig)
    return ax


def compose_fig4(d):
    fig = plt.figure(figsize=(7.1, 6.20))
    gs = GridSpec(2, 2, figure=fig, wspace=0.28, hspace=0.40)
    axa = fig.add_subplot(gs[0, 0], projection=_proj()) if HAS_CARTOPY else fig.add_subplot(gs[0, 0])
    fig4b_heatmap(d, fig.add_subplot(gs[0, 1]))
    fig4c_cq(d, fig.add_subplot(gs[1, 0]))
    fig4d_slices(d, fig.add_subplot(gs[1, 1]))
    fig4a_ne_south(d, axa, cax="below")
    save_pub(fig, "fig4_phosphorus")
    plt.close(fig)


# ---------- Fig 5 ----------
def fig5a_crop_no3(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    sl = d["sl_no3"].dropna(subset=["beta_GP"]) if len(d["sl_no3"]) else pd.DataFrame()
    if sl.empty:
        letter(ax, "a")
        return ax
    groups = [
        sl.loc[sl["crop_hi"] == False, "beta_GP"].to_numpy() if "crop_hi" in sl else np.array([]),
        sl.loc[sl["crop_hi"] == True, "beta_GP"].to_numpy() if "crop_hi" in sl else np.array([]),
    ]
    raincloud(ax, groups, ["Low crop", "High crop"],
              [PAL["pos"], PAL["crop"]], xlabel=r"Site slope, dry log NO$_3$-N on GP")
    letter(ax, "a")
    if close:
        save_pub(fig, "fig5a_crop_no3", FIG / "panels")
        plt.close(fig)
    return ax


def fig5b_horse(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    rows = [
        ("GP", "GP_only", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", PAL["flow"]),
        ("Wet-day fraction", "nwet", "beta_nwet_frac", "ci95_lo_nwet_frac", "ci95_hi_nwet_frac", PAL["gray"]),
        ("p95 rain share", "p95", "beta_p95share", "ci95_lo_p95share", "ci95_hi_p95share", PAL["gray"]),
        ("GP | wet days", "GP_nwet", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", PAL["flow"]),
        ("GP | p95 share", "GP_p95", "beta_GP", "ci95_lo_GP", "ci95_hi_GP", PAL["flow"]),
    ]
    labels, beta, lo, hi, cols = [], [], [], [], []
    for lab, sample, b, a, c, col in rows:
        r = _row(d["rh"], family="horse", **{"sample": sample}, y="logR")
        if r is None or pd.isna(r.get(b)):
            continue
        labels.append(lab)
        beta.append(float(r[b]))
        lo.append(float(r[a]))
        hi.append(float(r[c]))
        cols.append(col)
    lollipop(ax, labels, beta, lo, hi, colors=cols)
    letter(ax, "b")
    if close:
        save_pub(fig, "fig5b_horse", FIG / "panels")
        plt.close(fig)
    return ax


def fig5c_placebo(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    from utils import fe_ols
    from scipy.stats import gaussian_kde

    h = d["hydro"].dropna(subset=["GP", "logP", "logR", "site_key"]).copy()
    rng = np.random.default_rng(2026)
    betas = []
    for _ in range(40):
        hp = h.copy()
        hp["GP"] = hp.groupby("site_key")["GP"].transform(lambda s: rng.permutation(s.to_numpy()))
        r = fe_ols(hp, "logR")
        if r.get("ok"):
            betas.append(r["beta_GP"])
    betas = np.asarray(betas)
    xs = np.linspace(betas.min() - 0.15, betas.max() + 0.15, 160)
    ax.fill_between(xs, 0, gaussian_kde(betas)(xs), color="#b8c4ce", alpha=0.7, lw=0)
    ax.plot(xs, gaussian_kde(betas)(xs), color="#5d6d7e", lw=1.0)
    ax.plot(betas, np.full_like(betas, -0.02 * gaussian_kde(betas)(xs).max()), "|", color="#5d6d7e", ms=6)
    r0 = _row(d["rh"], family="hydro", **{"sample": "all"}, y="logR")
    if r0 is not None:
        ax.axvline(float(r0["beta_GP"]), color=PAL["flow"], lw=1.35, label="Observed")
        ax.axvspan(float(r0["ci95_lo_GP"]), float(r0["ci95_hi_GP"]), color=PAL["flow"], alpha=0.12)
    ax.axvline(0, color="0.35", lw=0.6, ls="--")
    ax.set_xlabel(r"Placebo GP coefficient on $\log R$")
    ax.set_ylabel("Density")
    ax.legend(fontsize=6)
    letter(ax, "c")
    if close:
        save_pub(fig, "fig5c_placebo", FIG / "panels")
        plt.close(fig)
    return ax


def fig5d_south_flush(d, ax=None):
    close = ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(3.4, 2.7))
    specs = [
        ("Low crop, dry NO$_3$", "NO3N", "crop_lo", "logC_low", "conflict_n", PAL["pos"]),
        ("High crop, dry NO$_3$", "NO3N", "crop_hi", "logC_low", "conflict_n", PAL["crop"]),
        ("South wet NO$_3$", "NO3N", "South", "logC_high", "load_region", PAL["no3"]),
        ("NE dry TP", "TP", "Northeast", "logC_low", "conflict_tp", PAL["ne"]),
        ("South dry TP", "TP", "South", "logC_low", "conflict_tp", PAL["south"]),
    ]
    labels, beta, lo, hi, cols = [], [], [], [], []
    for lab, param, sample, y, fam, col in specs:
        r = _row(d["rc"], family=fam, param=param, **{"sample": sample}, y=y)
        if r is None or not bool(r.get("ok", True)):
            continue
        labels.append(lab)
        beta.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(col)
    lollipop(ax, labels, beta, lo, hi, colors=cols)
    letter(ax, "d")
    if close:
        save_pub(fig, "fig5d_support", FIG / "panels")
        plt.close(fig)
    return ax


def compose_fig5(d):
    fig = plt.figure(figsize=(7.1, 5.7))
    gs = GridSpec(2, 2, figure=fig, wspace=0.30, hspace=0.36)
    fig5a_crop_no3(d, fig.add_subplot(gs[0, 0]))
    fig5b_horse(d, fig.add_subplot(gs[0, 1]))
    fig5c_placebo(d, fig.add_subplot(gs[1, 0]))
    fig5d_south_flush(d, fig.add_subplot(gs[1, 1]))
    save_pub(fig, "fig5_nitrogen_robust")
    plt.close(fig)


def compose_si(d):
    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    sl = d["slopes"]["beta_GP"].dropna()
    from scipy.stats import gaussian_kde
    xs = np.linspace(sl.quantile(0.01), sl.quantile(0.99), 160)
    ax.fill_between(xs, 0, gaussian_kde(sl)(xs), color=PAL["flow"], alpha=0.25, lw=0)
    ax.plot(xs, gaussian_kde(sl)(xs), color=PAL["flow"], lw=1.15)
    ax.axvline(0, color="0.35", lw=0.7, ls="--")
    ax.axvline(sl.median(), color=PAL["tp"], lw=1.1)
    ax.text(0.04, 0.92, f"{(sl < 0).mean():.0%} of sites negative", transform=ax.transAxes, fontsize=6.5)
    ax.set_xlabel(r"Site-level GP coefficient on $\log R$")
    ax.set_ylabel("Density")
    save_pub(fig, "si_site_slopes", FIG / "si")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(3.8, 2.7))
    r = d["rr"]
    r = r.loc[(r["family"] == "si_europe") & (r["ok"] == True)]
    labels, beta, lo, hi, cols = [], [], [], [], []
    for _, row in r.iterrows():
        labels.append(f"Europe dry {PARAM_LAB.get(row['param'], row['param'])}")
        beta.append(float(row["beta_GP"]))
        lo.append(float(row["ci95_lo_GP"]))
        hi.append(float(row["ci95_hi_GP"]))
        cols.append(PARAM_COLOR.get(row["param"], PAL["gray"]))
    if labels:
        lollipop(ax, labels, beta, lo, hi, colors=cols)
    save_pub(fig, "si_europe", FIG / "si")
    plt.close(fig)


def toc_graphic():
    """ES&T TOC art: amount vs timing arrows + CONUS locator. Code-composed, not AI."""
    fig = plt.figure(figsize=(3.25, 1.75))
    ax_l = fig.add_axes([0.02, 0.08, 0.34, 0.86])
    ax_l.set_xlim(0, 1)
    ax_l.set_ylim(0, 1)
    ax_l.axis("off")
    ax_l.annotate(
        "", xy=(0.96, 0.78), xytext=(0.04, 0.78),
        arrowprops=dict(arrowstyle="-|>", color=PAL["flow"], lw=1.15, mutation_scale=7),
    )
    ax_l.text(0.04, 0.92, r"More annual $P$", fontsize=5.6, color=PAL["flow"], va="center")
    ax_l.text(0.04, 0.64, r"$R$ and $C\times R$ up", fontsize=5.2, color=PAL["flow"], va="center")
    ax_l.annotate(
        "", xy=(0.96, 0.28), xytext=(0.04, 0.28),
        arrowprops=dict(arrowstyle="-|>", color=PAL["tp"], lw=1.15, mutation_scale=7),
    )
    ax_l.text(0.04, 0.46, r"Same annual $P$,", fontsize=5.6, color=PAL["tp"], va="center")
    ax_l.text(0.04, 0.36, "higher GP", fontsize=5.6, color=PAL["tp"], va="center")
    ax_l.text(0.04, 0.12, r"$R$ and $C\times R$ down", fontsize=5.2, color=PAL["tp"], va="center")

    ax_m = (
        fig.add_axes([0.36, 0.26, 0.42, 0.68], projection=_proj())
        if HAS_CARTOPY
        else fig.add_axes([0.36, 0.26, 0.42, 0.68])
    )
    style_conus_locator(ax_m)
    geo = {}
    if HAS_CARTOPY:
        import cartopy.crs as ccrs
        geo = {"transform": ccrs.PlateCarree()}
    ax_m.text(-73.8, 43.4, "NE", fontsize=5.4, color=PAL["ne"], fontweight="medium", ha="center", zorder=6, **geo)
    ax_m.text(-73.8, 40.2, r"year-scale $C$", fontsize=4.6, color="#2c3e50", ha="center", zorder=6, **geo)
    ax_m.text(-90.2, 32.8, "South", fontsize=5.4, color=PAL["south"], fontweight="medium", ha="center", zorder=6, **geo)
    ax_m.text(-90.2, 29.6, r"$Q_{10}$ tail", fontsize=4.6, color="#2c3e50", ha="center", zorder=6, **geo)

    ax_r = fig.add_axes([0.78, 0.10, 0.21, 0.82])
    ax_r.set_xlim(0, 1)
    ax_r.set_ylim(0, 1)
    ax_r.axis("off")
    ax_r.text(0.02, 0.88, r"NE: year-scale $C$", fontsize=5.1, color=PAL["ne"], va="top")
    ax_r.text(0.02, 0.74, r"dry TP $C\times R$ up", fontsize=4.8, color=PAL["ne"], va="top")
    ax_r.text(0.02, 0.52, r"South: $Q_{10}$ tail", fontsize=5.1, color=PAL["south"], va="top")
    ax_r.text(0.02, 0.38, r"dry P $C\times R$", fontsize=4.8, color=PAL["south"], va="top")
    ax_r.text(0.02, 0.28, "follows tail", fontsize=4.8, color=PAL["south"], va="top")
    ax_r.text(0.02, 0.08, r"Low-crop: dry NO$_3$ $C$ up", fontsize=4.8, color=PAL["crop"], va="center")

    FIG.mkdir(parents=True, exist_ok=True)
    stem = FIG / "toc_graphic"
    fig.savefig(f"{stem}.svg")
    fig.savefig(f"{stem}.pdf")
    fig.savefig(f"{stem}.png", dpi=600)
    fig.savefig(f"{stem}.tiff", dpi=600)
    print("wrote", stem)
    plt.close(fig)


def save_panels(d):
    fig1a_sites(d)
    fig1b_gp(d)
    fig1c_drymonths(d)
    fig1d_gpsd(d)
    fig2a_slope_map(d)
    fig2b_resid_hex(d)
    fig2c_bins(d)
    fig2d_decomp(d)
    fig3a_cr_plane(d)
    fig3b_load_hex(d)
    fig3c_channels(d)
    fig3d_season_load(d)
    fig4a_ne_south(d)
    fig4b_heatmap(d)
    fig4c_cq(d)
    fig4d_slices(d)
    fig5a_crop_no3(d)
    fig5b_horse(d)
    fig5c_placebo(d)
    fig5d_south_flush(d)


def main():
    d = load()
    save_panels(d)
    compose_fig1(d)
    compose_fig2(d)
    compose_fig3(d)
    compose_fig4(d)
    compose_fig5(d)
    from figures.fig6_tail import main as compose_fig6
    compose_fig6()
    compose_si(d)
    toc_graphic()


if __name__ == "__main__":
    main()
