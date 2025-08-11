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
):
    plt.figure(figsize=(14, 6))
    plt.plot(times_dense, pred_ids_dense, '-', label='Predicted', alpha=0.8)
    if gt_times.size > 0:
        plt.plot(gt_times, gt_ids, 'o', label='GT snaps', markersize=6, color='black')

    plt.yticks(ticks=np.arange(len(zone_names)), labels=zone_names)
    plt.xlabel('Time (hours)')
    plt.ylabel('Location index')
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


