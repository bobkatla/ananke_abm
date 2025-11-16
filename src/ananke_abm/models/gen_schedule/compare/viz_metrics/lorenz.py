# lorenz.py
import os
from typing import Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt


def lorenz_curve_from_counts(counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    counts: 1D non-negative array (e.g. frequencies per schedule or per activity).

    Returns
    -------
    x : np.ndarray, shape (K+1,)
        Cumulative share of categories (from 0 to 1).
    y : np.ndarray, shape (K+1,)
        Cumulative share of mass (from 0 to 1).
    gini : float
        Gini coefficient in [0,1].
    """
    counts = np.asarray(counts, dtype=np.float64)
    if counts.ndim != 1:
        raise ValueError(f"counts must be 1D, got shape {counts.shape}")

    if np.any(counts < 0):
        raise ValueError("counts must be non-negative")

    total = counts.sum()
    if total <= 0:
        # degenerate: all zero, Lorenz is diagonal, Gini = 0
        x = np.linspace(0.0, 1.0, len(counts) + 1)
        y = x.copy()
        return x, y, 0.0

    # sort from smallest to largest
    sorted_counts = np.sort(counts)
    # cumulative share of mass
    cum_mass = np.cumsum(sorted_counts) / total  # (K,)
    # x-axis: cumulative share of categories
    K = len(counts)
    x = np.linspace(0.0, 1.0, K + 1)
    y = np.concatenate([[0.0], cum_mass])  # prepend 0

    # Gini = 1 - 2 * area_under_lorenz
    area = np.trapezoid(y, x)
    gini = 1.0 - 2.0 * area

    return x, y, float(gini)


def plot_lorenz_for_models(
    model_counts: Dict[str, np.ndarray],
    title: str = "Lorenz curves and Gini coefficients",
    output_dir: Optional[str] = None,
    show: bool = False,
    prefix: str = "",
    colors: Optional[Dict[str, str]] = None,
):
    """
    Plot Lorenz curves and a companion bar chart of Gini coefficients.

    The function produces a single figure with two subplots:
      - Left: Lorenz curves for all models + equality line.
      - Right: Bar chart of Gini coefficients per model (with value annotations).

    Parameters
    ----------
    model_counts : dict[str, 1D array-like]
        Mapping model name -> counts (e.g. schedule frequencies).
    title : str, default "Lorenz curves and Gini coefficients"
        Overall figure title.
    output_dir : str or None, default None
        If not None, save figure into this directory.
    show : bool, default False
        If True, call plt.show() at the end.
    prefix : str, default ""
        String to prepend to saved filename.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The created figure.
    gini_dict : dict[str, float]
        Mapping model name -> Gini coefficient.
    """
    if not model_counts:
        raise ValueError("model_counts must be a non-empty dict")

    # Ensure deterministic ordering (sorted by model name)
    model_names = list(model_counts.keys())

    # Compute Lorenz curves and Ginis
    curves = {}
    gini_dict: Dict[str, float] = {}
    for name in model_names:
        counts = np.asarray(model_counts[name], dtype=np.float64)
        x, y, gini = lorenz_curve_from_counts(counts)
        curves[name] = (x, y)
        gini_dict[name] = gini

    # Set up figure with two subplots side by side
    fig, (ax_lorenz, ax_bar) = plt.subplots(1, 2, figsize=(10, 4))

    # ----- Left subplot: Lorenz curves -----
    # Grayscale-friendly: vary markers/linestyles; default color black
    marker_cycle = ["o", "s", "^", "D", "v", "x", "+", ">", "<", "p"]
    linestyle_cycle = ["-", "--", "-.", ":"]
    n_markers = len(marker_cycle)
    n_linestyles = len(linestyle_cycle)

    for idx, name in enumerate(model_names):
        x, y = curves[name]

        marker = marker_cycle[idx % n_markers]
        linestyle = linestyle_cycle[(idx // n_markers) % n_linestyles]

        ax_lorenz.plot(
            x,
            y,
            label=name,
            color=colors.get(name, "black") if colors is not None else "black",
            marker=marker,
            linestyle=linestyle,
            markevery=max(len(x) // 5, 1),
        )

    # equality line
    ax_lorenz.plot([0, 1], [0, 1], linestyle=":", linewidth=1.0, color="gray", label="equality")

    ax_lorenz.set_xlabel("Cumulative share of schedules (sorted)")
    ax_lorenz.set_ylabel("Cumulative share of probability/mass")
    ax_lorenz.set_title("Lorenz curves")
    ax_lorenz.legend(fontsize="small")
    ax_lorenz.grid(True, linestyle=":", linewidth=0.5)

    # ----- Right subplot: Gini bar chart -----
    ginis = np.array([gini_dict[name] for name in model_names], dtype=float)
    x_pos = np.arange(len(model_names))

    bars = ax_bar.bar(x_pos, ginis, width=0.6)

    # Annotate values on top of bars
    for xpos, g, bar in zip(x_pos, ginis, bars):
        height = bar.get_height()
        ax_bar.text(
            xpos,
            height,
            f"{g:.3f}",
            ha="center",
            va="bottom",
            fontsize="small",
        )

    ax_bar.set_xticks(x_pos)
    ax_bar.set_xticklabels(model_names, rotation=30, ha="right")
    ax_bar.set_ylabel("Gini coefficient")
    ax_bar.set_ylim(0.0, min(1.0, ginis.max() * 1.1))
    ax_bar.set_title("Gini by model")
    ax_bar.grid(axis="y", linestyle=":", linewidth=0.5)

    # Overall title
    fig.suptitle(title)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    # Save if requested
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        fname = f"{prefix}lorenz_gini.png"
        fig_path = os.path.join(output_dir, fname)
        fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()

