# ananke_abm/models/gen_schedule/compare/extract_metrics/diversity.py

import os
from typing import Dict, List, Tuple

import numpy as np

from ananke_abm.models.gen_schedule.compare.utils import (
    ensure_dir,
    schedule_counts,
    ngram_counts,
)


# ---------------------------------------------------------------------
# Core helpers: entropy & Gini on count dictionaries
# ---------------------------------------------------------------------

def _entropy_from_counts(counts: Dict[Tuple[int, ...], int], eps: float = 1e-12) -> float:
    """
    Shannon entropy H = -sum p_i log p_i computed from raw counts.

    Args:
        counts: dict cell -> count
        eps:    small constant to avoid log(0)

    Returns:
        entropy in nats (float)
    """
    if not counts:
        return 0.0

    vals = np.array(list(counts.values()), dtype=np.float64)
    total = float(vals.sum())
    if total <= 0.0:
        return 0.0

    p = vals / total
    # guard against log(0)
    p = np.clip(p, eps, 1.0)
    H = -float(np.sum(p * np.log(p)))
    return H


def _gini_from_counts(counts: Dict[Tuple[int, ...], int], eps: float = 1e-12) -> float:
    """
    Gini coefficient computed from raw counts.

    Standard discrete formula over non-negative values x_i:

        Sort x_i ascending -> x_(i), i=1..n
        G = (2 * sum_i i * x_(i) / (n * sum_i x_(i))) - (n + 1) / n

    Args:
        counts: dict cell -> count
        eps:    small constant to guard against zero total

    Returns:
        Gini in [0, 1] (float)
    """
    if not counts:
        return 0.0

    vals = np.array(list(counts.values()), dtype=np.float64)
    vals = vals[vals >= 0.0]
    if vals.size == 0:
        return 0.0

    vals_sorted = np.sort(vals)
    n = vals_sorted.size
    cum = np.cumsum(vals_sorted)
    total = float(cum[-1])
    if total <= eps:
        return 0.0

    idx = np.arange(1, n + 1, dtype=np.float64)
    gini = (2.0 * np.sum(idx * vals_sorted) / (n * total)) - (n + 1.0) / n
    # Numerical guards
    gini = float(max(0.0, min(1.0, gini)))
    return gini


def _diversity_from_counts_pair(
    counts_ref: Dict[Tuple[int, ...], int],
    counts_syn: Dict[Tuple[int, ...], int],
) -> Dict[str, float]:
    """
    Given reference and synthetic count dicts over the same *universe*
    (not enforced here), compute:

      - entropy_overall  : entropy over all synthetic cells
      - entropy_confirmed: entropy over synthetic mass restricted to cells
                           that are present in reference
      - gini_overall     : Gini over all synthetic counts
      - gini_confirmed   : Gini over confirmed synthetic counts
    """
    # overall
    H_overall = _entropy_from_counts(counts_syn)
    G_overall = _gini_from_counts(counts_syn)

    # confirmed = restrict syn counts to keys present in ref
    if counts_ref:
        confirmed_counts_syn = {
            k: v for k, v in counts_syn.items() if k in counts_ref
        }
    else:
        confirmed_counts_syn = {}

    H_conf = _entropy_from_counts(confirmed_counts_syn)
    G_conf = _gini_from_counts(confirmed_counts_syn)

    return {
        "entropy_overall": H_overall,
        "entropy_confirmed": H_conf,
        "gini_overall": G_overall,
        "gini_confirmed": G_conf,
    }


# ---------------------------------------------------------------------
# Schedule-level diversity metrics
# ---------------------------------------------------------------------

def metric_diversity_schedules(ref: Dict, models: List[Dict], outdir: str):
    """
    Diversity of full schedules (each row is one cell).

    Writes CSV:
        diversity_schedules.csv with columns:
          model,
          entropy_overall, entropy_confirmed,
          gini_overall,    gini_confirmed
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]  # (N_ref, T)
    counts_ref = schedule_counts(Y_ref)

    rows = []

    # reference row: entropy/gini from its own counts, confirmed == overall
    H_ref = _entropy_from_counts(counts_ref)
    G_ref = _gini_from_counts(counts_ref)
    rows.append({
        "model": "ref",
        "entropy_overall": H_ref,
        "entropy_confirmed": H_ref,
        "gini_overall": G_ref,
        "gini_confirmed": G_ref,
    })

    # synthetic models
    for m in models:
        name = m["name"]
        Y_syn = m["Y"]
        counts_syn = schedule_counts(Y_syn)

        stats = _diversity_from_counts_pair(counts_ref, counts_syn)
        row = {"model": name}
        row.update(stats)
        rows.append(row)

    import csv
    csv_path = os.path.join(outdir, "diversity_schedules.csv")
    fieldnames = [
        "model",
        "entropy_overall",
        "entropy_confirmed",
        "gini_overall",
        "gini_confirmed",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------------------------------------------------------------------
# N-gram-level diversity metrics (n = 1..4)
# ---------------------------------------------------------------------

def metric_diversity_ngram(ref: Dict, models: List[Dict], outdir: str):
    """
    Diversity of n-grams for n = 1..4.

    For each n, writes:
        diversity_ngram_n{n}.csv with columns:
          model,
          entropy_overall, entropy_confirmed,
          gini_overall,    gini_confirmed

    Cells are n-grams (tuples of length n) extracted with a sliding window.
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]  # (N_ref, T)

    for n in (1, 2, 3, 4):
        counts_ref = ngram_counts(Y_ref, n=n, as_schedule=False)

        rows = []
        # reference row
        H_ref = _entropy_from_counts(counts_ref)
        G_ref = _gini_from_counts(counts_ref)
        rows.append({
            "model": "ref",
            "entropy_overall": H_ref,
            "entropy_confirmed": H_ref,
            "gini_overall": G_ref,
            "gini_confirmed": G_ref,
        })

        # synthetic models
        for m in models:
            name = m["name"]
            Y_syn = m["Y"]
            counts_syn = ngram_counts(Y_syn, n=n, as_schedule=False)

            stats = _diversity_from_counts_pair(counts_ref, counts_syn)
            row = {"model": name}
            row.update(stats)
            rows.append(row)

        import csv
        csv_path = os.path.join(outdir, f"diversity_ngram_n{n}.csv")
        fieldnames = [
            "model",
            "entropy_overall",
            "entropy_confirmed",
            "gini_overall",
            "gini_confirmed",
        ]
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for r in rows:
                writer.writerow(r)


# ---------------------------------------------------------------------
# Registry for compare.metric_tables
# ---------------------------------------------------------------------

from ananke_abm.models.gen_schedule.compare.viz_metrics.lorenz import plot_lorenz_for_models

def plot_diversity_lorenz_schedules(ref: Dict, models: List[Dict], outdir: str):
    """
    Plot Lorenz curves for schedule-level diversity.
    """
    ensure_dir(outdir)

    Y_ref = ref["Y"]
    counts_ref = np.array(list(schedule_counts(Y_ref).values()), dtype=np.float64)

    to_plot_models = {"ref": counts_ref}

    for m in models:
        name = m["name"]
        Y_syn = m["Y"]
        counts_syn = np.array(list(schedule_counts(Y_syn).values()), dtype=np.float64)
        to_plot_models[name] = counts_syn

    plot_lorenz_for_models(
        model_counts=to_plot_models,
        title="Schedule-level Diversity Lorenz Curves",
    )

DIVERSITY_FUNCS = {
    "diversity_schedules": metric_diversity_schedules,
    "diversity_ngram":     metric_diversity_ngram,
    # "diversity_lorenz_schedules": plot_diversity_lorenz_schedules,
}
