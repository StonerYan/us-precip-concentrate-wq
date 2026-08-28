from pathlib import Path

HERE = Path(__file__).parent
PANELS = [
    "fig1a_sites",
    "fig1b_gp",
    "fig1c_drymonths",
    "fig1d_gpsd",
    "fig2a_slope_map",
    "fig2b_resid_hex",
    "fig2c_bins",
    "fig2d_decomp",
    "fig3a_cr_plane",
    "fig3b_load_hex",
    "fig3c_channels",
    "fig3d_season_load",
    "fig4a_ne_south",
    "fig4b_heatmap",
    "fig4c_cq",
    "fig4d_slices",
    "fig5a_crop_no3",
    "fig5b_horse",
    "fig5c_placebo",
    "fig5d_south_flush",
]
for name in PANELS:
    (HERE / f"{name}.py").write_text(
        f"from figures.make_all import load, {name}\n\nif __name__ == '__main__':\n    {name}(load())\n",
        encoding="utf-8",
    )
for i in range(1, 6):
    (HERE / f"compose_fig{i}.py").write_text(
        f"from figures.make_all import compose_fig{i}, load\n\nif __name__ == '__main__':\n    compose_fig{i}(load())\n",
        encoding="utf-8",
    )
(HERE / "toc_graphic.py").write_text(
    "from figures.make_all import toc_graphic\n\nif __name__ == '__main__':\n    toc_graphic()\n",
    encoding="utf-8",
)
print("wrappers written")
