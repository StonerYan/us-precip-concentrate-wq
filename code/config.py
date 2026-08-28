import os
from pathlib import Path

PROJECT = Path(__file__).resolve().parents[1]
RAW = Path(os.environ.get("PRECIP_WQ_RAW", r"O:\PrecipConcentrate_WQ"))
CACHE24 = Path(os.environ.get("PRECIP_WQ_CACHE24", PROJECT.parent / "PrecipConcentrate_WQ_0824" / "data"))
CACHE23 = Path(os.environ.get("PRECIP_WQ_CACHE23", PROJECT.parent / "PrecipConcentrate_WQ_0823" / "data"))

DATA = PROJECT / "data"
FIG = PROJECT / "figures"
NOTES = PROJECT / "notes"
PAPER = PROJECT / "paper"
SI = PROJECT / "si"

GRQA_ZIP = RAW / "Zenodo_15335450_Global River Water Quality Archive GRQA" / "GRQA_data_v1.4.zip"
SITES_CSV = RAW / "derived" / "grqa_sites_nutrients.csv"
GPCC_DIR = RAW / "gpcc_daily"
USGS_DIRS = [RAW / "usgs_dv", RAW / "usgs_dv_supplement"]

PARAMS = ("TN", "NO3N", "TP")
YEAR0, YEAR1 = 1982, 2020
MIN_YEARS = 8
MIN_OBS_YEAR = 4
AREA_MIN, AREA_MAX = 10.0, 50_000.0

US_COUNTRIES = {
    "United States",
    "USA",
    "US",
    "United States of America (the)",
}

for p in (DATA, FIG, NOTES, PAPER, SI, FIG / "si", FIG / "panels"):
    p.mkdir(parents=True, exist_ok=True)
