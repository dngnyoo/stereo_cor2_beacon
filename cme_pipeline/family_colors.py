"""Stable high-contrast family colors for signed RD figures.

Golden-angle HSV avoids the grey/pastel entries and 20-ID wrapping in tab20.
Colors represent identity only, never a detection confidence value.
"""
from __future__ import annotations

import colorsys

def track_color(track_id: int) -> str:
    tid = int(track_id)
    if tid < 0:
        raise ValueError("Track ID must be nonnegative")
    hue = (tid * 0.618033988749895) % 1.0
    r, g, b = colorsys.hsv_to_rgb(hue, 0.88, 0.72)
    return "#{:02x}{:02x}{:02x}".format(round(r * 255), round(g * 255), round(b * 255))

def family_color(family_id: str) -> str:
    if not family_id.startswith("FAM_T") or not family_id[5:].isdigit():
        raise ValueError(f"Unexpected identity Family ID: {family_id!r}")
    return track_color(int(family_id[5:]))
