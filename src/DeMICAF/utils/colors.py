"""Figure colour palette, loaded from the bundled ``data/colors.json``.

The JSON file maps palette names (e.g. ``"Steel Blue"``) to design-token entries with a
hex value. :data:`COLOR_DICT` maps plot-level concepts (datasets, losses, causes, ...)
to RGB tuples; when the JSON file is unavailable a neutral fallback color is used so
imports never fail.
"""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

from DeMICAF.utils.paths import get_repo_root

if TYPE_CHECKING:
    from pathlib import Path

RGB = tuple[float, float, float]

_FALLBACK: RGB = (0.5, 0.5, 0.5)


def load_colors(json_path: Path) -> dict[str, RGB]:
    """Load palette-name → RGB mappings from a design-token JSON file."""
    colors: dict[str, RGB] = {}
    with json_path.open() as f:
        data = json.load(f)
    for color_name, color_info in data.items():
        if not color_name.startswith("$") and isinstance(color_info, dict):
            hex_value = color_info.get("$value", {}).get("hex")
            if hex_value:
                hex_clean = hex_value.lstrip("#")
                r = int(hex_clean[0:2], 16) / 255.0
                g = int(hex_clean[2:4], 16) / 255.0
                b = int(hex_clean[4:6], 16) / 255.0
                colors[color_name] = (r, g, b)
    return colors


_json_path = get_repo_root() / "data" / "colors.json"
COLORS: dict[str, RGB] = load_colors(_json_path) if _json_path.exists() else {}

COLOR_MAP = {
    "CheXpert": "Steel Blue",
    "ChestX-ray8": "Blue",
    "MIMIC-CXR": "Lilac",
    "PadChest": "Red",
    "All": "Black",
    "SimCLR": "Blue",
    "CSI": "Green",
    "UniCon": "Lilac",
    "UniConSA": "Grey",
    "SupCon": "Steel Blue",
    "Compliant": "Green",
    "Non-compliant": "Red",
    "Frontal (AP)": "Deep Blue",
    "Frontal (PA)": "Steel Blue",
    "Lateral": "Red",
    "Unknown": "Grey",
    "Female": "Deep Blue",
    "Male": "Steel Blue",
    "Age": "Lilac",
    "Pixel": "Blue",
    "Corrupted Images": "Blue",
    "Not a Thorax": "Deep Blue",
    "Incomplete Thorax": "Orange",
    "Image Quality Problems": "Yellow",
    "Area Not Valid": "Red",
    "Overlaying Objects": "Lilac",
    "Non-Canonical Positions": "Steel Blue",
    "Repeated": "Steel Blue",
    "None": "Green",
    "Mean": "Blue",
    "Cardiomegaly": "Red",
    "Atelectasis": "Yellow",
    "Nodule": "Steel Blue",
    "Alveolar Pattern": "Orange",
    "Pleural Effusion": "Deep Blue",
    "Pneumothorax": "Lilac",
    "Best": "Blue",
    "Control": "Lilac",
    "Worst": "Red",
    "Centralised": "Steel Blue",
    "Federated": "Lilac",
}

COLOR_DICT: dict[str, RGB] = {key: COLORS.get(color_name, _FALLBACK) for key, color_name in COLOR_MAP.items()}

if __name__ == "__main__":
    for name, rgb in COLOR_DICT.items():
        print(f"COLOR_DICT['{name}'] = {rgb}")
