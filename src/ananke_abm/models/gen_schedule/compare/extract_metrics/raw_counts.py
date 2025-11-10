# raw_counts.py

import os
from typing import Dict, List

import numpy as np

from ananke_abm.models.gen_schedule.compare.utils import ensure_dir


# ---------- helpers ----------

def _segment_activities(row: np.ndarray) -> List[int]:
    """
    Given a single schedule row of length T, return the sequence of activity
    labels with consecutive duplicates collapsed.

    Example:
        [2,2,2,1,1,3] -> [2,1,3]
    """
    T = row.shape[0]
    if T == 0:
        return []
    seq = [int(row[0])]
    for t in range(1, T):
        if row[t] != row[t - 1]:
            seq.append(int(row[t]))
    return seq


def _build_schedule_counts(Y: np.ndarray) -> Dict[bytes, int]:
    """
    Build a dict from schedule bytes -> count in Y.
    Each row Y[i, :] is converted to bytes via .tobytes().
    """
    counts: Dict[bytes, int] = {}
    for row in Y:
        key = row.tobytes()
        counts[key] = counts.get(key, 0) + 1
    return counts


# ---------- metric 1: raw activity / n-gram counts ----------

def metric_raw_cells(ref: Dict, models: List[Dict], outdir: str):
    """
    Count activities, bigrams, trigrams, and quadgrams over the population,
    and derive per-person averages.

    "Activities" are segments of constant purpose in a day schedule.
    Bigrams/trigrams/quadgrams are overlapping windows over the activity
    sequence per person.

    Writes:
        raw_cells_counts.csv with columns:
            model, N_persons,
            total_activities, total_bigrams, total_trigrams, total_quadgrams,
            avg_activities_per_person, avg_bigrams_per_person,
            avg_trigrams_per_person, avg_quadgrams_per_person
    """
    import csv

    ensure_dir(outdir)

    rows = []

    def _compute_for_dataset(name: str, Y: np.ndarray):
        N, T = Y.shape
        total_acts = 0
        total_bi = 0
        total_tri = 0
        total_quad = 0

        for i in range(N):
            seq = _segment_activities(Y[i])
            K = len(seq)
            total_acts += K
            if K >= 2:
                total_bi += (K - 1)
            if K >= 3:
                total_tri += (K - 2)
            if K >= 4:
                total_quad += (K - 3)

        if N > 0:
            avg_acts = total_acts / float(N)
            avg_bi = total_bi / float(N)
            avg_tri = total_tri / float(N)
            avg_quad = total_quad / float(N)
        else:
            avg_acts = avg_bi = avg_tri = avg_quad = 0.0

        rows.append(
            {
                "model": name,
                "N_persons": int(N),
                "total_activities": int(total_acts),
                "total_bigrams": int(total_bi),
                "total_trigrams": int(total_tri),
                "total_quadgrams": int(total_quad),
                "avg_activities_per_person": float(avg_acts),
                "avg_bigrams_per_person": float(avg_bi),
                "avg_trigrams_per_person": float(avg_tri),
                "avg_quadgrams_per_person": float(avg_quad),
            }
        )

    # reference first
    _compute_for_dataset("ref", ref["Y"])

    # then each model
    for m in models:
        _compute_for_dataset(m["name"], m["Y"])

    csv_path = os.path.join(outdir, "raw_cells_counts.csv")
    fieldnames = list(rows[0].keys()) if rows else [
        "model",
        "N_persons",
        "total_activities",
        "total_bigrams",
        "total_trigrams",
        "total_quadgrams",
        "avg_activities_per_person",
        "avg_bigrams_per_person",
        "avg_trigrams_per_person",
        "avg_quadgrams_per_person",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------- metric 2: schedule confirmation / coverage ----------

def metric_raw_schedules(ref: Dict, models: List[Dict], outdir: str):
    """
    Count confirmed vs non-confirmed schedules (individual- and unique-level),
    and coverage of the reference population.

    Definitions (using your terminology):
      - H-population = ref["Y"] (never exposed to models in training).
      - Confirmed schedule  = schedule that exists in H-population.
      - Non-confirmed       = does not exist in H-population.

    For each dataset (ref and each model), we compute:
      - confirmed_individual_count / pct
      - non_confirmed_individual_count / pct
      - unique_confirmed_count
      - unique_non_confirmed_count
      - unique_confirmed_pct_of_ref_unique
          = (# unique confirmed schedules in this dataset) /
            (# unique confirmed schedules in ref)
      - ref_coverage_by_confirmed_unique
          = fraction of *individuals* in H-population whose schedule is in
            the set of confirmed unique schedules in this dataset.

    Writes:
        raw_counts_schedule_confirmation.csv
    """
    import csv

    ensure_dir(outdir)

    Y_ref: np.ndarray = ref["Y"]
    N_ref = Y_ref.shape[0]

    # Build reference schedule universe
    ref_sched_counts = _build_schedule_counts(Y_ref)
    unique_confirmed_ref = set(ref_sched_counts.keys())
    num_unique_confirmed_ref = len(unique_confirmed_ref)

    def _compute_for_dataset(name: str, Y: np.ndarray) -> Dict[str, float]:
        N = Y.shape[0]

        # individual-level confirmed / non-confirmed
        confirmed_individual_count = 0
        for row in Y:
            key = row.tobytes()
            if key in ref_sched_counts:
                confirmed_individual_count += 1
        non_confirmed_individual_count = N - confirmed_individual_count

        confirmed_individual_pct = (
            confirmed_individual_count / float(N) if N > 0 else 0.0
        )
        non_confirmed_individual_pct = (
            non_confirmed_individual_count / float(N) if N > 0 else 0.0
        )

        # unique-level sets
        model_sched_counts = _build_schedule_counts(Y)
        unique_in_model = set(model_sched_counts.keys())

        unique_confirmed_in_model = unique_in_model.intersection(unique_confirmed_ref)
        unique_non_confirmed_in_model = unique_in_model.difference(unique_confirmed_ref)

        unique_confirmed_count = len(unique_confirmed_in_model)
        unique_non_confirmed_count = len(unique_non_confirmed_in_model)

        if num_unique_confirmed_ref > 0:
            unique_confirmed_pct_of_ref_unique = (
                unique_confirmed_count / float(num_unique_confirmed_ref)
            )
        else:
            unique_confirmed_pct_of_ref_unique = 0.0

        # coverage of ref population by these confirmed unique schedules
        if N_ref > 0 and unique_confirmed_in_model:
            covered_ref_individuals = 0
            for row in Y_ref:
                key = row.tobytes()
                if key in unique_confirmed_in_model:
                    covered_ref_individuals += 1
            ref_coverage_by_confirmed_unique = covered_ref_individuals / float(N_ref)
        else:
            ref_coverage_by_confirmed_unique = 0.0

        return {
            "model": name,
            "N_persons": int(N),
            "confirmed_individual_count": int(confirmed_individual_count),
            "confirmed_individual_pct": float(confirmed_individual_pct),
            "non_confirmed_individual_count": int(non_confirmed_individual_count),
            "non_confirmed_individual_pct": float(non_confirmed_individual_pct),
            "unique_confirmed_count": int(unique_confirmed_count),
            "unique_non_confirmed_count": int(unique_non_confirmed_count),
            "unique_confirmed_pct_of_ref_unique": float(
                unique_confirmed_pct_of_ref_unique
            ),
            "ref_coverage_by_confirmed_unique": float(
                ref_coverage_by_confirmed_unique
            ),
        }

    rows = []
    # reference row (it is the H-population)
    rows.append(_compute_for_dataset("ref", Y_ref))

    # models
    for m in models:
        rows.append(_compute_for_dataset(m["name"], m["Y"]))

    csv_path = os.path.join(outdir, "raw_counts_schedule_confirmation.csv")
    fieldnames = list(rows[0].keys()) if rows else [
        "model",
        "N_persons",
        "confirmed_individual_count",
        "confirmed_individual_pct",
        "non_confirmed_individual_count",
        "non_confirmed_individual_pct",
        "unique_confirmed_count",
        "unique_non_confirmed_count",
        "unique_confirmed_pct_of_ref_unique",
        "ref_coverage_by_confirmed_unique",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------- metric 3: home patterns (home-all-day, home-bound) ----------

def metric_raw_home_patterns(ref: Dict, models: List[Dict], outdir: str):
    """
    Count home-pattern statistics:

      - start_home_pct
      - end_home_pct
      - home_bound_pct (start AND end at home)
      - non_home_bound_pct (violations of the above)
      - home_all_day_pct (fully home-bound at every time bin)

    "Home" index is determined from ref["purpose_map"]["Home"] if available,
    otherwise we fall back to the most common first label in ref.
    The same home_idx is used for all models.

    Writes:
        raw_counts_home_patterns.csv
    """
    import csv

    ensure_dir(outdir)

    Y_ref: np.ndarray = ref["Y"]
    purpose_map: Dict[str, int] = ref["purpose_map"]
    N_ref, T = Y_ref.shape

    # Determine home_idx from ref only, then reuse
    if "Home" in purpose_map:
        home_idx = int(purpose_map["Home"])
    else:
        first_col = Y_ref[:, 0]
        vals, counts = np.unique(first_col, return_counts=True)
        home_idx = int(vals[np.argmax(counts)])

    def _compute_for_dataset(name: str, Y: np.ndarray) -> Dict[str, float]:
        N, T_local = Y.shape
        if N == 0 or T_local == 0:
            return {
                "model": name,
                "N_persons": int(N),
                "start_home_count": 0,
                "start_home_pct": 0.0,
                "end_home_count": 0,
                "end_home_pct": 0.0,
                "home_bound_count": 0,
                "home_bound_pct": 0.0,
                "non_home_bound_count": 0,
                "non_home_bound_pct": 0.0,
                "home_all_day_count": 0,
                "home_all_day_pct": 0.0,
            }

        start_home = (Y[:, 0] == home_idx)
        end_home = (Y[:, -1] == home_idx)
        home_bound = start_home & end_home
        non_home_bound = ~home_bound
        home_all_day = (Y == home_idx).all(axis=1)

        start_home_count = int(start_home.sum())
        end_home_count = int(end_home.sum())
        home_bound_count = int(home_bound.sum())
        non_home_bound_count = int(non_home_bound.sum())
        home_all_day_count = int(home_all_day.sum())

        start_home_pct = start_home_count / float(N)
        end_home_pct = end_home_count / float(N)
        home_bound_pct = home_bound_count / float(N)
        non_home_bound_pct = non_home_bound_count / float(N)
        home_all_day_pct = home_all_day_count / float(N)

        return {
            "model": name,
            "N_persons": int(N),
            "start_home_count": start_home_count,
            "start_home_pct": float(start_home_pct),
            "end_home_count": end_home_count,
            "end_home_pct": float(end_home_pct),
            "home_bound_count": home_bound_count,
            "home_bound_pct": float(home_bound_pct),
            "non_home_bound_count": non_home_bound_count,
            "non_home_bound_pct": float(non_home_bound_pct),
            "home_all_day_count": home_all_day_count,
            "home_all_day_pct": float(home_all_day_pct),
        }

    rows = []
    # reference first
    rows.append(_compute_for_dataset("ref", Y_ref))
    # then each model
    for m in models:
        rows.append(_compute_for_dataset(m["name"], m["Y"]))

    csv_path = os.path.join(outdir, "raw_counts_home_patterns.csv")
    fieldnames = list(rows[0].keys()) if rows else [
        "model",
        "N_persons",
        "start_home_count",
        "start_home_pct",
        "end_home_count",
        "end_home_pct",
        "home_bound_count",
        "home_bound_pct",
        "non_home_bound_count",
        "non_home_bound_pct",
        "home_all_day_count",
        "home_all_day_pct",
    ]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


# ---------- registry ----------

RAW_COUNTS_FUNCS = {
    "raw_cells":      metric_raw_cells,
    "raw_schedules":  metric_raw_schedules,
    "raw_home":       metric_raw_home_patterns,
}
