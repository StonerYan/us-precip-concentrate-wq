"""Shared ES&T matplotlib style and richer panel helpers."""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

from config import FIG
from regions import REG_COL, REG_FILL, REG_LAB, STATES_GEOJSON, ncei_of, region_of, NAME_TO_POSTAL

PAL = {
    "tn": "#1b4f72",
    "no3": "#1a7a72",
    "tp": "#9b2c2c",
    "flow": "#1a365d",
    "pos": "#9b2c2c",
    "neg": "#1a365d",
    "gray": "#5d6d7e",
    "crop": "#3d5a3d",
    "ne": "#c0841a",
    "south": "#1a7a72",
    "ci": "#85929e",
}

PARAM_COLOR = {"TN": PAL["tn"], "NO3N": PAL["no3"], "TP": PAL["tp"]}
PARAM_LAB = {"TN": "TN", "NO3N": "NO$_3$-N", "TP": "TP"}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 7,
        "axes.labelsize": 7,
        "axes.titlesize": 8,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "axes.linewidth": 0.7,
        "legend.frameon": False,
        "axes.axisbelow": True,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.facecolor": "white",
    }
)

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except Exception:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None


def save_pub(fig: plt.Figure, name: str, outdir: Path | None = None) -> None:
    outdir = outdir or FIG
    outdir.mkdir(parents=True, exist_ok=True)
    stem = outdir / name
    fig.savefig(f"{stem}.svg", bbox_inches="tight")
    fig.savefig(f"{stem}.pdf", bbox_inches="tight")
    fig.savefig(f"{stem}.png", dpi=400, bbox_inches="tight")
    fig.savefig(f"{stem}.tiff", dpi=600, bbox_inches="tight")
    print("wrote", stem)


def letter(ax, s, x=-0.06, y=1.04) -> None:
    ax.text(x, y, s, transform=ax.transAxes, fontsize=9, fontweight="bold", va="bottom")


def map_proj():
    if not HAS_CARTOPY:
        return None
    return ccrs.AlbersEqualArea(
        central_longitude=-96.0,
        central_latitude=37.5,
        standard_parallels=(29.5, 45.5),
    )


def _pc():
    return ccrs.PlateCarree() if HAS_CARTOPY else None


def style_conus(ax, highlight=None) -> bool:
    if not HAS_CARTOPY:
        ax.set_xlim(-125, -66.5)
        ax.set_ylim(24.2, 49.8)
        ax.set_aspect("equal")
        return False
    ax.set_extent([-125.5, -66.2, 24.4, 49.6], crs=ccrs.PlateCarree())
    ax.set_facecolor("white")
    try:
        ax.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#f4f6f8", edgecolor="none", zorder=0)
        ax.add_feature(cfeature.LAKES.with_scale("50m"), facecolor="#eef2f5", edgecolor="#c5ccd3", linewidth=0.2, zorder=1)
        _fill_climate_regions(ax, highlight=highlight)
        ax.add_feature(cfeature.STATES.with_scale("50m"), linewidth=0.25, edgecolor="#b7bec4", facecolor="none", zorder=2)
        ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor="#6d767d", zorder=3)
        ax.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.35, edgecolor="#6d767d", zorder=3)
    except Exception:
        ax.add_feature(cfeature.STATES, linewidth=0.25, edgecolor="#c5ccd1")
        ax.coastlines(linewidth=0.35)
    if hasattr(ax, "spines"):
        for sp in ax.spines.values():
            sp.set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    return True


def style_conus_locator(ax, highlight=("Northeast", "South")) -> bool:
    """CONUS climate-region silhouette; only named exception regions are colored."""
    return style_conus(ax, highlight=highlight)


def _fill_climate_regions(ax, highlight=None) -> None:
    if not STATES_GEOJSON.exists():
        return
    import json
    from shapely.geometry import shape
    from shapely.ops import unary_union

    gj = json.loads(STATES_GEOJSON.read_text(encoding="utf-8"))
    buckets: dict[str, list] = {k: [] for k in REG_FILL}
    for feat in gj["features"]:
        name = feat.get("properties", {}).get("name")
        postal = NAME_TO_POSTAL.get(name)
        if not postal or postal in {"AK", "HI"}:
            continue
        reg = region_of(postal)
        if reg in buckets:
            buckets[reg].append(shape(feat["geometry"]))
    for reg, geoms in buckets.items():
        if not geoms:
            continue
        merged = unary_union(geoms)
        if highlight is None:
            fc, al = REG_FILL[reg], 0.95
        elif reg in highlight:
            fc, al = REG_FILL[reg], 0.95
        else:
            fc, al = "#e4e7ea", 0.9
        ax.add_geometries(
            [merged],
            crs=ccrs.PlateCarree(),
            facecolor=fc,
            edgecolor="none",
            zorder=1,
            alpha=al,
        )


def map_points(ax, lon, lat, c, s=9, prepare=True, **kw):
    if prepare and not getattr(ax, "_conus_ready", False):
        style_conus(ax)
        ax._conus_ready = True
    extra = {"transform": ccrs.PlateCarree()} if HAS_CARTOPY else {}
    extra.setdefault("linewidths", 0.18)
    extra.setdefault("edgecolors", "white")
    extra.setdefault("zorder", 5)
    extra.update(kw)
    return ax.scatter(lon, lat, c=c, s=s, **extra)


def lowess_xy(x, y, frac=0.45):
    from statsmodels.nonparametric.smoothers_lowess import lowess

    z = lowess(np.asarray(y, float), np.asarray(x, float), frac=frac, return_sorted=True)
    return z[:, 0], z[:, 1]


def kde_contour(ax, x, y, color, levels=6, alpha=0.35, linewidths=0.7, zorder=2):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x, y = x[m], y[m]
    if len(x) < 80:
        ax.scatter(x, y, s=4, alpha=0.25, color=color, linewidths=0)
        return
    # subsample for KDE speed
    rng = np.random.default_rng(7)
    if len(x) > 4000:
        idx = rng.choice(len(x), 4000, replace=False)
        xs, ys = x[idx], y[idx]
    else:
        xs, ys = x, y
    from scipy.stats import gaussian_kde

    try:
        kde = gaussian_kde(np.vstack([xs, ys]))
    except Exception:
        ax.scatter(x, y, s=3, alpha=0.15, color=color, linewidths=0)
        return
    xmin, xmax = np.quantile(x, [0.02, 0.98])
    ymin, ymax = np.quantile(y, [0.02, 0.98])
    xx, yy = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
    zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
    ax.contour(xx, yy, zz, levels=levels, colors=[color], linewidths=linewidths, alpha=0.9, zorder=zorder)


def raincloud(ax, series, labels, colors, xlabel="Site-level GP coefficient", ylim=None):
    """Violin + jitter + mean 95% CI. Extreme tails are clipped for display only."""
    cleaned = []
    for vals in series:
        v = np.asarray(vals, float)
        v = v[np.isfinite(v)]
        cleaned.append(v)
    if ylim is None:
        pool = np.concatenate([v for v in cleaned if len(v)]) if any(len(v) for v in cleaned) else np.array([-2.0, 2.0])
        lo, hi = np.quantile(pool, [0.02, 0.98])
        pad = 0.15 * (hi - lo + 1e-6)
        ylim = (lo - pad, hi + pad)
    for i, (v, lab, col) in enumerate(zip(cleaned, labels, colors)):
        if len(v) < 8:
            continue
        v = v[(v >= ylim[0]) & (v <= ylim[1])]
        parts = ax.violinplot(v, positions=[i], widths=0.72, showextrema=False, showmedians=False, vert=True)
        for pc in parts["bodies"]:
            pc.set_facecolor(col)
            pc.set_edgecolor(col)
            pc.set_alpha(0.22)
            pc.set_linewidth(0.6)
        rng = np.random.default_rng(11 + i)
        jit = i + 0.16 * (rng.random(len(v)) - 0.5)
        ax.scatter(jit, v, s=6, color=col, alpha=0.38, linewidths=0, zorder=3)
        mu = float(np.mean(v))
        se = float(np.std(v, ddof=1) / np.sqrt(len(v)))
        ax.plot([i, i], [mu - 1.96 * se, mu + 1.96 * se], color="#1a1a1a", lw=1.25, zorder=4)
        ax.scatter([i], [mu], s=26, color=col, edgecolors="white", linewidths=0.6, zorder=5)
    ax.axhline(0, color="0.45", lw=0.5, ls="--")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_ylabel(xlabel)
    ax.set_ylim(*ylim)


def lollipop(ax, labels, beta, lo, hi, colors=None, xlabel="Coefficient on GP (log units)"):
    y = np.arange(len(labels))[::-1]
    if colors is None:
        colors = [PAL["flow"]] * len(labels)
    for yi, b, a, c, col in zip(y, beta, lo, hi, colors):
        ax.plot([a, c], [yi, yi], color=col, lw=2.0, solid_capstyle="round", zorder=2, alpha=0.9)
        ax.scatter([b], [yi], s=28, color=col, zorder=3, edgecolors="white", linewidths=0.5)
    ax.axvline(0, color="0.35", lw=0.55, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(xlabel)


def dumbbell(ax, labels, left, right, left_lo=None, left_hi=None, right_lo=None, right_hi=None,
             c_left=PAL["flow"], c_right=PAL["tp"], lab_left="Runoff", lab_right="Concentration"):
    y = np.arange(len(labels))[::-1]
    for i, yi in enumerate(y):
        ax.plot([left[i], right[i]], [yi, yi], color="#c5ccd1", lw=1.1, zorder=1)
        if left_lo is not None:
            ax.plot([left_lo[i], left_hi[i]], [yi, yi], color=c_left, lw=1.6, alpha=0.35, zorder=2)
        if right_lo is not None:
            ax.plot([right_lo[i], right_hi[i]], [yi, yi], color=c_right, lw=1.6, alpha=0.35, zorder=2)
        ax.scatter([left[i]], [yi], s=26, color=c_left, zorder=3, edgecolors="white", linewidths=0.4)
        ax.scatter([right[i]], [yi], s=26, color=c_right, zorder=3, edgecolors="white", linewidths=0.4)
    ax.axvline(0, color="0.35", lw=0.55, ls="--")
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Coefficient on GP")
    ax.legend(
        handles=[
            Line2D([0], [0], marker="o", color="none", markerfacecolor=c_left, markersize=5, label=lab_left),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=c_right, markersize=5, label=lab_right),
        ],
        loc="best",
        fontsize=6,
    )


# back-compat name used by older panel scripts
coef_dots = lollipop
