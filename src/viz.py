"""Shared plotting style: the validated reference palette, light surface."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"

# categorical slots, fixed order, never cycled
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
       "#e87ba4", "#008300", "#4a3aa7", "#e34948"]

# diverging: blue <-> red, neutral gray midpoint
DIVERGING = LinearSegmentedColormap.from_list(
    "bl_gy_rd", ["#0d366b", "#256abf", "#86b6ef", "#f0efec",
                 "#f0a3a2", "#e34948", "#8f2020"])
# sequential: one hue, light -> dark
SEQ = LinearSegmentedColormap.from_list(
    "blues", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#184f95", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "DejaVu Sans", "Arial"],
    "font.size": 9,
    "axes.edgecolor": BASELINE, "axes.linewidth": 0.8,
    "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 10, "axes.titleweight": "semibold",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "xtick.labelcolor": INK2, "ytick.labelcolor": INK2,
    "legend.frameon": False, "legend.fontsize": 8,
    "lines.linewidth": 2.0, "lines.markersize": 5,
    "figure.dpi": 150, "savefig.dpi": 200, "savefig.bbox": "tight",
})


def despine(ax, keep=("left", "bottom")):
    for s in ("top", "right", "left", "bottom"):
        ax.spines[s].set_visible(s in keep)


def zeroline(ax, horizontal=True):
    (ax.axhline if horizontal else ax.axvline)(
        0, color=BASELINE, lw=1.0, zorder=1)
