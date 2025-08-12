"""
Visualization utilities for mode_sep model.
"""
from __future__ import annotations

from typing import List

import matplotlib.pyplot as plt
import numpy as np


def plot_person_trajectory(
    times_dense: np.ndarray,
    pred_ids_dense: np.ndarray,
    gt_times: np.ndarray,
    gt_ids: np.ndarray,
    zone_names: List[str],
    out_path: str,
    v_abs_dense: np.ndarray | None = None,
    stay_intervals: List[tuple] | None = None,
    thresholds: dict | None = None,
):
    # Determine layout: with velocity subplot if provided
    if v_abs_dense is not None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(14, 6))
        ax2 = None

    # Trajectory subplot
    ax1.plot(times_dense, pred_ids_dense, '-', label='Predicted', alpha=0.85)
    if gt_times.size > 0:
        ax1.plot(gt_times, gt_ids, 'o', label='GT snaps', markersize=6, color='black')
    ax1.set_yticks(np.arange(len(zone_names)))
    ax1.set_yticklabels(zone_names)
    ax1.set_ylabel('Location')
    ax1.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    ax1.legend(loc='upper right')
    ax1.set_title('Predicted trajectory vs GT')

    # Velocity subplot
    if ax2 is not None and v_abs_dense is not None:
        ax2.plot(times_dense, v_abs_dense, '-', color='tab:blue', label='|v|(dense)')
        # thresholds
        if thresholds:
            if 'epsilon_v' in thresholds and thresholds['epsilon_v'] is not None:
                ax2.axhline(thresholds['epsilon_v'], color='tab:green', linestyle='--', alpha=0.8, label=f"epsilon_v={thresholds['epsilon_v']}")
            if 'v_min_move' in thresholds and thresholds['v_min_move'] is not None:
                ax2.axhline(thresholds['v_min_move'], color='tab:orange', linestyle='--', alpha=0.8, label=f"v_min_move={thresholds['v_min_move']}")
            if 'v_max_move' in thresholds and thresholds['v_max_move'] is not None:
                ax2.axhline(thresholds['v_max_move'], color='tab:red', linestyle='--', alpha=0.8, label=f"v_max_move={thresholds['v_max_move']}")
        # stays shading
        if stay_intervals:
            for (a, b) in stay_intervals:
                ax2.axvspan(a, b, color='tab:green', alpha=0.1)
        ax2.set_ylabel('|v|')
        ax2.set_xlabel('Time (hours)')
        ax2.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        ax2.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(out_path)
    plt.close(fig)


