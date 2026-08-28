"""Figure S3: low-flow tail by region; South dry TP load vs Q10 control."""
from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd

from config import DATA, FIG
from fig_style import PAL, letter, lollipop, save_pub

SI = FIG / "si"


def _row(tab, family, sample, y):
    d = tab.loc[(tab["family"] == family) & (tab["sample"] == sample) & (tab["y"] == y)]
    return None if d.empty else d.iloc[0]


def main() -> None:
    tab = pd.read_csv(DATA / "regression_regime.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))

    specs = [
        ("Northeast", "Northeast", PAL["ne"]),
        ("South", "South", PAL["south"]),
        ("Interior", "Interior", "#3d6b3d"),
        ("West", "West", "#7a5a3a"),
        ("CONUS", "all", PAL["flow"]),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, samp, c in specs:
        r = _row(tab, "lowflow", samp, "frac_q10")
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(axes[0], labs, b, lo, hi, colors=cols, xlabel=r"Coefficient on GP ($Q_{10}$-day fraction)")
    letter(axes[0], "a")

    specs2 = [
        ("NE dry TP $C\\times R$", "load_dry", "TP_NE", PAL["pos"]),
        (r"NE, $Q_{10}$ controlled", "load_dry_ctrl_q10", "TP_NE", PAL["gray"]),
        ("South dry TP $C\\times R$", "load_dry", "TP_South", PAL["south"]),
        (r"South, $Q_{10}$ controlled", "load_dry_ctrl_q10", "TP_South", PAL["gray"]),
    ]
    labs, b, lo, hi, cols = [], [], [], [], []
    for lab, fam, samp, c in specs2:
        r = _row(tab, fam, samp, "logLoad_dry")
        if r is None or not bool(r.get("ok", False)):
            continue
        labs.append(lab)
        b.append(float(r["beta_GP"]))
        lo.append(float(r["ci95_lo_GP"]))
        hi.append(float(r["ci95_hi_GP"]))
        cols.append(c)
    lollipop(axes[1], labs, b, lo, hi, colors=cols, xlabel="Coefficient on GP")
    letter(axes[1], "b")

    fig.tight_layout()
    save_pub(fig, "si_s2_lowflow_tp", outdir=SI)
    plt.close(fig)


if __name__ == "__main__":
    main()
