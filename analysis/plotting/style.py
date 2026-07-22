"""
analysis/plotting/style.py

Global plotting configuration for the Earth Observation Satellite
analysis package.
"""

from __future__ import annotations

# ==========================================================
# Figure Configuration
# ==========================================================

FIGURE_SIZE = (11, 6.5)

DPI = 300

EXPORT_FORMAT = "png"

FIGURE_FACE_COLOR = "white"

# ==========================================================
# Fonts
# ==========================================================

TITLE_FONT_SIZE = 17

LABEL_FONT_SIZE = 13

TICK_FONT_SIZE = 11

LEGEND_FONT_SIZE = 10

TITLE_WEIGHT = "bold"

# ==========================================================
# Lines
# ==========================================================

LINE_WIDTH = 2.2

REFERENCE_LINE_WIDTH = 1.5

REFERENCE_LINE_STYLE = "--"

REFERENCE_ALPHA = 0.65

GRID_STYLE = "--"

GRID_ALPHA = 0.35

MARKER_SIZE = 70

# ==========================================================
# Orbit Plot
# ==========================================================

EARTH_RADIUS = 6378.137e3

EARTH_COLOR = "#4F81BD"

ORBIT_COLOR = "#E67E22"

START_COLOR = "#2ECC71"

END_COLOR = "#E74C3C"

# ==========================================================
# Reaction Wheels
# ==========================================================

SATURATION_LINE_STYLE = "--"

SATURATION_LINE_WIDTH = 1.5

SATURATION_COLOR = "red"

# ==========================================================
# Default Colours
# ==========================================================

COLORS = [

    "#1f77b4",      # Blue

    "#ff7f0e",      # Orange

    "#2ca02c",      # Green

    "#d62728",      # Red

    "#9467bd",      # Purple

    "#8c564b",

    "#e377c2",

]