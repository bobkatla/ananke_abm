import os
from typing import List, Dict, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def _extract_durations_per_purpose(
    Y: np.ndarray,
    P: int,
    time_grid: int,
) -> List[np.ndarray]:
    """
    Extract contiguous activity durations per purpose (in minutes).

    For each row in Y, every maximal contiguous run of the same purpose index
    contributes one duration = run_length * time_grid.

    Parameters
    ----------
    Y : np.ndarray
        Shape (N, T), integer purpose indices.
    P : int
        Number of distinct purposes (assumed 0..P-1).
    time_grid : int
        Bin size in minutes.

    Returns
    -------
    durations_per_purpose : list of np.ndarray
        List of length P; durations_per_purpose[p] is a 1D array of durations
        in minutes for purpose index p.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N, T), got shape {Y.shape}")

    N, T = Y.shape
    durations_per_purpose: List[List[float]] = [[] for _ in range(P)]

    for i in range(N):
        row = Y[i]
        curr = int(row[0])
        run_len = 1
        for t in range(1, T):
            val = int(row[t])
            if val == curr:
                run_len += 1
            else:
                durations_per_purpose[curr].append(run_len * time_grid)
                curr = val
                run_len = 1
        # last run
        durations_per_purpose[curr].append(run_len * time_grid)

    # Convert to arrays
    return [np.asarray(d, dtype=float) if len(d) > 0 else np.asarray([], dtype=float)
            for d in durations_per_purpose]


def _build_purpose_names(purpose_map: Dict[str, int]) -> Tuple[List[str], int]:
    """
    From {purpose_name: idx} build ordered purpose_names list and P.

    Assumes indices are 0..P-1.
    """
    inv = {idx: name for name, idx in purpose_map.items()}
    P = len(purpose_map)
    if set(inv.keys()) != set(range(P)):
        raise ValueError(
            f"purpose indices must be contiguous 0..P-1; got indices {sorted(inv.keys())}"
        )
    purpose_names = [inv[i] for i in range(P)]
    return purpose_names, P


def plot_duration_boxplots(
    Y_list: List[np.ndarray],
    purpose_maps: List[Dict[str, int]],
    dataset_names: List[str],
    time_grid: int = 5,
    layout: str = "compressed",  # "compressed" or "separate"
    output_dir: Optional[str] = None,
    show: bool = False,
    prefix: str = "",
    colors: Optional[List[str]] = None,
):
    """
    Plot boxplots of contiguous activity durations per purpose.

    Each contiguous run of a given purpose in a person's schedule is one
    "activity episode"; its duration is (run_length * time_grid) minutes.

    Comparisons: multiple datasets plotted side-by-side for each activity.

    Parameters
    ----------
    Y_list : list of np.ndarray
        Each array has shape (N_i, T) with integer purpose indices.
    purpose_maps : list of dict
        Each dict is {purpose_name: idx}. All purpose_maps must be identical
        across datasets and indices must be 0..P-1.
    dataset_names : list of str
        Names for each dataset, used in legends. Same length as Y_list.
    time_grid : int, default 5
        Bin size in minutes (used to convert run lengths to minutes).
    layout : {"compressed", "separate"}, default "compressed"
        "compressed": one big figure with grouped boxplots for all activities.
        "separate" : one figure per activity.
    output_dir : str or None, default None
        If not None, save figures into this directory (created if needed).
    show : bool, default False
        If True, call plt.show() at the end.
    prefix : str, default ""
        String to prepend to saved filenames.
    colors : list of str, optional
        Optional list of Matplotlib colors for each dataset. If provided,
        must have same length as Y_list.

    Returns
    -------
    figs : list of matplotlib.figure.Figure
        List of created figures.
    purpose_names : list of str
        Purpose names in consistent index order.
    """
    # Basic sanity checks
    if not (len(Y_list) == len(purpose_maps) == len(dataset_names)):
        raise ValueError("Y_list, purpose_maps, and dataset_names must have same length")

    if colors is not None and len(colors) != len(Y_list):
        raise ValueError("If provided, colors must have same length as Y_list")

    layout = layout.lower()
    if layout not in ("compressed", "separate"):
        raise ValueError("layout must be 'compressed' or 'separate'")

    # Check purpose_maps identical
    ref_pm = purpose_maps[0]
    for i, pm in enumerate(purpose_maps[1:], start=1):
        if pm != ref_pm:
            raise ValueError(f"purpose_map mismatch between dataset 0 and {i}")

    purpose_names, P = _build_purpose_names(ref_pm)

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

    # Extract durations per purpose per dataset
    # durations_by_dataset[d][p] -> np.ndarray of durations (minutes)
    durations_by_dataset: List[List[np.ndarray]] = []
    for Y in Y_list:
        durations_by_dataset.append(_extract_durations_per_purpose(Y, P, time_grid))

    # Prepare output directory
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)

    figs = []

    # ---------- COMPRESSED LAYOUT: one big grouped boxplot ----------
    if layout == "compressed":
        num_datasets = len(Y_list)
        fig, ax = plt.subplots(figsize=(max(6, P * 1.2), 6))

        group_width = 0.8
        box_width = group_width / max(num_datasets, 1)

        # For legend handles
        legend_handles = []

        for d_idx in range(num_datasets):
            color = colors[d_idx] if colors is not None else None

            for p_idx in range(P):
                durations = durations_by_dataset[d_idx][p_idx]
                # If no data, feed NaN to avoid errors but keep position
                data = durations if durations.size > 0 else np.array([np.nan])

                center = p_idx
                pos = center - group_width / 2 + box_width / 2 + d_idx * box_width

                bp = ax.boxplot(
                    data,
                    positions=[pos],
                    widths=box_width,
                    patch_artist=True,
                    manage_ticks=False,
                )

                # Style
                if color is not None:
                    for patch in bp["boxes"]:
                        patch.set_facecolor(color)
                        patch.set_alpha(0.5)
                else:
                    for patch in bp["boxes"]:
                        patch.set_facecolor("lightgray")
                        patch.set_alpha(0.7)

                for whisker in bp["whiskers"]:
                    whisker.set_color("black")
                for cap in bp["caps"]:
                    cap.set_color("black")
                for median in bp["medians"]:
                    median.set_color("black")

                # For legend: capture first box of each dataset
                if p_idx == 0:
                    legend_handles.append(bp["boxes"][0])

        ax.set_xticks(range(P))
        ax.set_xticklabels(purpose_names, rotation=45, ha="right")

        ax.set_ylabel("Duration (minutes)")
        # ax.set_title("Activity episode durations by purpose")

        ax.grid(axis="y", alpha=0.3)

        # Legend
        ax.legend(legend_handles, dataset_names, title="Dataset")

        figs.append(fig)

        # Save if requested
        if output_dir is not None:
            fname = f"{prefix}duration_boxplots_compressed.png"
            fig_path = os.path.join(output_dir, fname)
            fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    # ---------- SEPARATE LAYOUT: one figure per purpose ----------
    else:  # layout == "separate"
        num_datasets = len(Y_list)

        for p_idx, p_name in enumerate(purpose_names):
            fig, ax = plt.subplots(figsize=(max(4, num_datasets * 1.2), 6))

            data = []
            for d_idx in range(num_datasets):
                durations = durations_by_dataset[d_idx][p_idx]
                if durations.size == 0:
                    data.append(np.array([np.nan]))
                else:
                    data.append(durations)

            positions = np.arange(num_datasets)

            bp = ax.boxplot(
                data,
                positions=positions,
                widths=0.6,
                patch_artist=True,
                manage_ticks=False,
            )

            # Style boxes by dataset
            for d_idx in range(num_datasets):
                color = colors[d_idx] if colors is not None else None
                box = bp["boxes"][d_idx]
                if color is not None:
                    box.set_facecolor(color)
                    box.set_alpha(0.5)
                else:
                    box.set_facecolor("lightgray")
                    box.set_alpha(0.7)

            for whisker in bp["whiskers"]:
                whisker.set_color("black")
            for cap in bp["caps"]:
                cap.set_color("black")
            for median in bp["medians"]:
                median.set_color("black")

            ax.set_xticks(positions)
            ax.set_xticklabels(dataset_names, rotation=30, ha="right")

            ax.set_ylabel("Duration (minutes)")
            # ax.set_title(f"Activity episode durations – {p_name}")

            ax.grid(axis="y", alpha=0.3)

            figs.append(fig)

            # Save if requested
            if output_dir is not None:
                safe_name = p_name.replace(" ", "_")
                fname = f"{prefix}duration_boxplots_{safe_name}.png"
                fig_path = os.path.join(output_dir, fname)
                fig.savefig(fig_path, bbox_inches="tight", dpi=300)

    if show:
        plt.show()
