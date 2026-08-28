"""Replace lat–lon box regions with NCEI climate-region classes."""
from __future__ import annotations

import json

import pandas as pd

from config import DATA, NOTES
from regions import annotate_panel


def _rewrite(name: str) -> dict:
    path = DATA / name
    df = pd.read_parquet(path)
    if "lat" not in df.columns or "lon" not in df.columns:
        return {name: "skipped (no lat/lon)"}
    out = annotate_panel(df)
    out.to_parquet(path, index=False)
    counts = out.drop_duplicates(
        [c for c in ("usgs_site_no", "site_key") if c in out.columns][:1] or None
    )
    key = "usgs_site_no" if "usgs_site_no" in out.columns else ("site_key" if "site_key" in out.columns else None)
    if key:
        u = out.drop_duplicates(key)
        vc = u["region"].value_counts(dropna=False).to_dict()
        cb = int(u["corn_belt"].sum()) if "corn_belt" in u else None
    else:
        vc, cb = out["region"].value_counts(dropna=False).to_dict(), None
    return {name: {"region": {str(k): int(v) for k, v in vc.items()}, "corn_belt_sites": cb, "n": int(len(out))}}


def main() -> None:
    summary = {}
    for name in (
        "panel_us_paired_raw.parquet",
        "panel_hydro.parquet",
        "panel_us_paired.parquet",
        "panel_cq_month.parquet",
        "sites_master.parquet",
    ):
        if (DATA / name).exists():
            summary.update(_rewrite(name))
    (NOTES / "region_reassign.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
