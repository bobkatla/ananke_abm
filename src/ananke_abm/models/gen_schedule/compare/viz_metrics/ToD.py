import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Optional


def _compute_tod_marginals(Y: np.ndarray, P: int) -> np.ndarray:
    """
    Compute time-of-day marginals for each purpose.

    Parameters
    ----------
    Y : np.ndarray
        Shape (N, T), integer purpose indices.
    P : int
        Number of distinct purposes.

    Returns
    -------
    m : np.ndarray
        Shape (T, P), where m[t, p] = mean(Y[:, t] == p).
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, T), got shape {Y.shape}")
    N, T = Y.shape
    m = np.zeros((T, P), dtype=np.float64)

    # m[t, p] = mean(Y[:, t] == p)
    for p in range(P):
        m[:, p] = (Y == p).mean(axis=0)

    return m


def plot_tod_by_purpose(
    Y_list: List[np.ndarray],
    purpose_maps: List[Dict[str, int]],
    dataset_names: List[str],
    colors: Optional[List[str]] = None,
    time_grid: int = 5,
    start_time_min: int = 0,
    outdir: Optional[str] = None,
    show: bool = False,
    prefix: str = ""
):
    """
    Plot time-of-day probability curves for each purpose.

    One figure per purpose. Within each figure, one line per dataset.
    Grayscale-friendly: all lines are designed to be distinguishable by
    marker shape and line style, not just color.

    Parameters
    ----------
    Y_list : list of np.ndarray
        Each array has shape (N_i, T) with integer purpose indices.
    purpose_maps : list of dict
        Each dict is {purpose_name: idx}. All purpose_maps must be identical
        across datasets and indices must be 0..P-1.
    dataset_names : list of str
        Names for each dataset, used in legends. Same length as Y_list.
    colors : list of str, optional
        Optional list of Matplotlib colors for each dataset. If provided,
        they are used, but markers/linestyles still ensure grayscale
        distinguishability. If None, all lines are black.
    time_grid : int, default 5
        Bin size in minutes. Used to map bin index to time in minutes:
        t_min = start_time_min + bin_idx * time_grid.
    start_time_min : int, default 0
        Start time of bin 0 in minutes (e.g., 0 = midnight, -180 = 21:00
        previous day). Used only for the x-axis mapping.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        One figure per purpose, in index order.
    purpose_names : list of str
        Names of purposes in the same order as the figures.
    """
    # Basic sanity checks
    if not (len(Y_list) == len(purpose_maps) == len(dataset_names)):
        raise ValueError("Y_list, purpose_maps, and dataset_names must have same length")

    if colors is not None and len(colors) != len(Y_list):
        raise ValueError("If provided, colors must have same length as Y_list")

    # Reference purpose_map
    ref_pm = purpose_maps[0]

    # Check all purpose_maps identical
    for i, pm in enumerate(purpose_maps[1:], start=1):
        if pm != ref_pm:
            raise ValueError(f"purpose_map mismatch between dataset 0 and {i}")

    # Build index -> name and ordered purpose_names
    inv_ref = {idx: name for name, idx in ref_pm.items()}
    P = len(ref_pm)
    if set(inv_ref.keys()) != set(range(P)):
        raise ValueError(
            f"purpose indices must be contiguous 0..P-1; got indices {sorted(inv_ref.keys())}"
        )
    purpose_names = [inv_ref[i] for i in range(P)]

    # Check shapes and consistent T across datasets
    T = Y_list[0].shape[1]
    for i, Y in enumerate(Y_list):
        if Y.ndim != 2:
            raise ValueError(f"Y_list[{i}] must be 2D (N, T), got {Y.shape}")
        if Y.shape[1] != T:
            raise ValueError(
                f"All Y arrays must have same T; Y_list[0].shape[1]={T}, "
                f"Y_list[{i}].shape[1]={Y.shape[1]}"
            )

    # Build time axis in hours
    t_bins = np.arange(T)
    t_min = start_time_min + t_bins * time_grid
    t_hours = t_min / 60.0

    # Compute ToD marginals for each dataset
    tod_list = []
    for i, Y in enumerate(Y_list):
        tod = _compute_tod_marginals(Y, P)
        if tod.shape != (T, P):
            raise ValueError(
                f"ToD marginals shape mismatch for dataset {i}: "
                f"expected {(T, P)}, got {tod.shape}"
            )
        tod_list.append(tod)

    # Grayscale-friendly styles: markers + linestyles
    marker_cycle = ["o", "s", "^", "D", "v", "x", "+", ">", "<", "p"]
    linestyle_cycle = ["-", "--", "-.", ":"]
    n_markers = len(marker_cycle)
    n_linestyles = len(linestyle_cycle)

    figs: List[plt.Figure] = []

    # Plot one figure per purpose (index order)
    for p_idx, p_name in enumerate(purpose_names):
        fig, ax = plt.subplots()

        # Mark only some points to avoid clutter
        mark_interval = max(T // 12, 1)  # roughly one marker per hour if grid_min=5
        markevery = slice(0, None, mark_interval)

        for d_idx, (tod_m, ds_name) in enumerate(zip(tod_list, dataset_names)):
            y = tod_m[:, p_idx]

            # Style selection
            marker = marker_cycle[d_idx % n_markers]
            linestyle = linestyle_cycle[(d_idx // n_markers) % n_linestyles]

            plot_kwargs = {
                "marker": marker,
                "linestyle": linestyle,
                "markevery": markevery,
            }

            # Color handling: default black for all if no colors specified
            if colors is not None:
                plot_kwargs["color"] = colors[d_idx]
            else:
                plot_kwargs["color"] = "black"

            ax.plot(
                t_hours,
                y,
                label=ds_name,
                **plot_kwargs,
            )

        # ax.set_title(f"Time-of-day probability – {p_name}")
        ax.set_xlabel("Time of day (hours)")
        ax.set_ylabel(f"P(activity = {p_name})")

        ax.set_xlim(t_hours[0], t_hours[-1])

        # Y-limits: ensure non-negative; upper bound slightly above max
        y_min = 0.0
        y_max = max(1e-8, max(tod_m[:, p_idx].max() for tod_m in tod_list))
        ax.set_ylim(y_min, min(1.0, y_max * 1.05))

        ax.grid(True, alpha=0.3)
        ax.legend()

        figs.append(fig)

    if outdir is not None:
        import os
        os.makedirs(outdir, exist_ok=True)
        for p_idx, fig in enumerate(figs):
            p_name = purpose_names[p_idx]
            safe_p_name = p_name.replace(" ", "_").replace("/", "_")
            outpath = os.path.join(outdir, f"{prefix}_tod_prob_{safe_p_name}.png")
            fig.savefig(outpath)
            plt.close(fig)
    if show:
        for fig in figs:
            fig.show()