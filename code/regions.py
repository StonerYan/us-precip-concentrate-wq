"""NOAA NCEI climate regions + USDA Corn Belt. Not lat–lon boxes.

Climate regions follow Karl and Koss (1984) / NOAA NCEI. The four analysis
classes are aggregations of those nine climate regions. Corn Belt is a
USDA NASS production-region flag, not a climate class.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import DATA

# Karl and Koss 1984 / NOAA NCEI U.S. climate regions (state postal codes)
NCEI = {
    "Northeast": {"CT", "DE", "ME", "MD", "MA", "NH", "NJ", "NY", "PA", "RI", "VT"},
    "Southeast": {"AL", "FL", "GA", "NC", "SC", "VA"},
    "South": {"AR", "KS", "LA", "MS", "OK", "TX"},
    "Central": {"IL", "IN", "KY", "MO", "OH", "TN", "WV"},
    "EastNorthCentral": {"IA", "MI", "MN", "WI"},
    "WestNorthCentral": {"MT", "NE", "ND", "SD", "WY"},
    "Southwest": {"AZ", "CO", "NM", "UT"},
    "Northwest": {"ID", "OR", "WA"},
    "West": {"CA", "NV"},
}

# Four analysis regions: climate-region aggregates (not lat–lon boxes)
CLIMATE_TO_REGION = {
    "Northeast": "Northeast",
    "Southeast": "South",
    "South": "South",
    "Central": "Interior",
    "EastNorthCentral": "Interior",
    "WestNorthCentral": "West",
    "Southwest": "West",
    "Northwest": "West",
    "West": "West",
}

# USDA NASS core Corn Belt states (agricultural production region)
CORN_BELT_STATES = {"IA", "IL", "IN", "OH", "NE", "MN", "MO", "WI", "SD", "MI"}

POSTAL_TO_NCEI = {st: name for name, states in NCEI.items() for st in states}

REG_COL = {
    "Northeast": "#c0841a",
    "South": "#1a7a72",
    "Interior": "#3d6b3d",
    "West": "#7a5a3a",
}
REG_LAB = {
    "Northeast": "Northeast",
    "South": "South",
    "Interior": "Interior",
    "West": "West",
}
REG_FILL = {
    "Northeast": "#f6d48a",
    "South": "#9fd4ce",
    "Interior": "#b5d0a8",
    "West": "#e2c49a",
}

STATES_GEOJSON = DATA / "us_states_20m.geojson"
STATES_URL = "https://raw.githubusercontent.com/PublicaMundi/MappingAPI/master/data/geojson/us-states.json"

# PublicaMundi name → postal
NAME_TO_POSTAL = {
    "Alabama": "AL", "Alaska": "AK", "Arizona": "AZ", "Arkansas": "AR",
    "California": "CA", "Colorado": "CO", "Connecticut": "CT", "Delaware": "DE",
    "Florida": "FL", "Georgia": "GA", "Hawaii": "HI", "Idaho": "ID",
    "Illinois": "IL", "Indiana": "IN", "Iowa": "IA", "Kansas": "KS",
    "Kentucky": "KY", "Louisiana": "LA", "Maine": "ME", "Maryland": "MD",
    "Massachusetts": "MA", "Michigan": "MI", "Minnesota": "MN", "Mississippi": "MS",
    "Missouri": "MO", "Montana": "MT", "Nebraska": "NE", "Nevada": "NV",
    "New Hampshire": "NH", "New Jersey": "NJ", "New Mexico": "NM", "New York": "NY",
    "North Carolina": "NC", "North Dakota": "ND", "Ohio": "OH", "Oklahoma": "OK",
    "Oregon": "OR", "Pennsylvania": "PA", "Rhode Island": "RI", "South Carolina": "SC",
    "South Dakota": "SD", "Tennessee": "TN", "Texas": "TX", "Utah": "UT",
    "Vermont": "VT", "Virginia": "VA", "Washington": "WA", "West Virginia": "WV",
    "Wisconsin": "WI", "Wyoming": "WY", "District of Columbia": "DC",
}


def ncei_of(postal: str) -> str | None:
    if postal == "DC":
        return "Northeast"
    return POSTAL_TO_NCEI.get(postal)


def region_of(postal: str) -> str | None:
    climate = ncei_of(postal)
    return CLIMATE_TO_REGION.get(climate) if climate else None


def is_corn_belt(postal: str) -> bool:
    return postal in CORN_BELT_STATES


def _ensure_states_geojson() -> Path:
    if STATES_GEOJSON.exists() and STATES_GEOJSON.stat().st_size > 1000:
        return STATES_GEOJSON
    import urllib.request

    STATES_GEOJSON.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(STATES_URL, STATES_GEOJSON)
    return STATES_GEOJSON


@lru_cache(maxsize=1)
def _state_tree():
    from shapely.geometry import shape
    from shapely.strtree import STRtree

    path = _ensure_states_geojson()
    gj = json.loads(path.read_text(encoding="utf-8"))
    geoms, postals = [], []
    for feat in gj["features"]:
        name = feat.get("properties", {}).get("name") or feat.get("properties", {}).get("NAME")
        postal = NAME_TO_POSTAL.get(name)
        if not postal or postal in {"AK", "HI"}:
            continue
        geoms.append(shape(feat["geometry"]))
        postals.append(postal)
    tree = STRtree(geoms)
    return tree, geoms, postals


def assign_postal(lon: float, lat: float) -> str | None:
    if not np.isfinite(lon) or not np.isfinite(lat):
        return None
    from shapely.geometry import Point

    tree, geoms, postals = _state_tree()
    pt = Point(float(lon), float(lat))
    hits = tree.query(pt)
    if len(hits) and not isinstance(hits[0], (int, np.integer)):
        for geom in hits:
            if geom.contains(pt) or geom.intersects(pt):
                return postals[geoms.index(geom)]
        hits = []
    for idx in np.atleast_1d(hits):
        geom = geoms[int(idx)]
        if geom.contains(pt) or geom.intersects(pt):
            return postals[int(idx)]
    # nearest state if on a river boundary
    dmin, best = 1e9, None
    for geom, postal in zip(geoms, postals):
        d = geom.distance(pt)
        if d < dmin:
            dmin, best = d, postal
    return best if dmin < 1.5 else None


def classify_sites(df: pd.DataFrame, lon="lon", lat="lat") -> pd.DataFrame:
    """Add state, climate_region, region, corn_belt. Unique lat/lon only."""
    out = df.copy()
    key = out[[lon, lat]].drop_duplicates()
    posts = [assign_postal(r[lon], r[lat]) for r in key.to_dict("records")]
    key = key.assign(state=posts)
    key["climate_region"] = key["state"].map(ncei_of)
    key["region"] = key["state"].map(region_of)
    key["corn_belt"] = key["state"].map(lambda s: bool(s) and is_corn_belt(s))
    return out.merge(key, on=[lon, lat], how="left")


def annotate_panel(df: pd.DataFrame) -> pd.DataFrame:
    """Replace box-based region columns on a site-year panel that already has lat/lon."""
    drop = [c for c in ("state", "climate_region", "corn_belt") if c in df.columns]
    base = df.drop(columns=drop + (["region"] if "region" in df.columns else []), errors="ignore")
    return classify_sites(base)
