"""Matplotlib helpers shared by the result-generation scripts and notebooks."""

from __future__ import annotations

from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

if TYPE_CHECKING:
    from matplotlib.axes import Axes

#: Default figure size used by the thesis figures.
FIG_SIZE = (4, 3)


def apply_thesis_style() -> None:
    """Apply the serif font setup used by every thesis figure."""
    plt.rcParams["font.family"] = "serif"
    plt.rcParams["font.serif"] = ["Times New Roman", "Times", "Nimbus Roman", "DejaVu Serif"]


def format_thousands(value: float) -> str:
    """Format a count compactly, e.g. ``1500 -> '1.5k'`` and ``2000 -> '2k'``."""
    if value >= 1000:
        return f"{value / 1000:.1f}k" if value % 1000 != 0 else f"{int(value / 1000)}k"
    return f"{int(value)}"


def thousands_formatter(x: float, pos: int) -> str:
    """Tick formatter wrapper around :func:`format_thousands` (use with ``FuncFormatter``)."""
    return format_thousands(x)


def use_thousands_axis(ax: Axes) -> None:
    """Format the y-axis of ``ax`` with compact thousands labels."""
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_formatter))


def annotate_bars(ax: Axes, bars: object) -> None:
    """Write the (compactly formatted) height above each bar of a bar plot."""
    for bar in bars:  # type: ignore[attr-defined]
        height = bar.get_height()
        ax.annotate(
            format_thousands(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
        )
