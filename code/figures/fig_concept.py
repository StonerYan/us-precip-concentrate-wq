"""Fig. 1a pathway schematic. Single source: make_all.fig1a_concept."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from figures.make_all import fig1a_concept

if __name__ == "__main__":
    fig1a_concept()
