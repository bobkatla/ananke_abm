# utils.py  (replace the relevant parts)

import numpy as np
import os
import json
from typing import Dict, List, Tuple, Optional, Hashable


# --------------------------------------------------------------------------
# Basic helpers: loading
# --------------------------------------------------------------------------

def _load_one_npz_with_meta(npz_path: str, meta_path: str, name: str) -> Dict:
    arr = np.load(npz_path)
    # support Y_generated (samples) and Y (raw grid)
    if "Y_generated" in arr:
        Y = arr["Y_generated"].astype(np.int64)
    elif "Y" in arr:
        Y = arr["Y"].astype(np.int64)
    else:
        raise KeyError(f"{npz_path} must contain 'Y_generated' or 'Y'")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    purpose_map = meta["purpose_map"]
    grid_min = meta.get("grid_min", None)
    horizon_min = meta.get("horizon_min", None)
    # Prefer explicit L from meta, otherwise infer from Y
    T_meta = meta.get("L", None)
    T = int(T_meta) if T_meta is not None else int(Y.shape[1])

    return {
        "name": name,
        "Y": Y,                         # (N, T)
        "purpose_map": purpose_map,     # {name: idx}
        "grid_min": grid_min,           # may be None
        "horizon_min": horizon_min,     # may be None
        "T": T,
    }


def load_reference(ref_npz: str, ref_meta: str) -> Dict:
    return _load_one_npz_with_meta(ref_npz, ref_meta, name="ref")


def load_comparison_models(compare_dir: str) -> List[Dict]:
    """
    Expects in compare_dir:
      - <model>.npz
      - matching meta as either <model>_meta.json or <model>.json
    """
    models = []
    for fname in sorted(os.listdir(compare_dir)):
        if not fname.endswith(".npz"):
            continue

        stem = os.path.splitext(fname)[0]
        npz_path = os.path.join(compare_dir, fname)

        # try <stem>_meta.json then <stem>.json
        meta_candidates = [
            os.path.join(compare_dir, f"{stem}_meta.json"),
            os.path.join(compare_dir, f"{stem}.json"),
        ]
        meta_path = None
        for cand in meta_candidates:
            if os.path.exists(cand):
                meta_path = cand
                break
        if meta_path is None:
            raise FileNotFoundError(
                f"No meta json found for {npz_path}. "
                f"Tried {meta_candidates}"
            )

        models.append(_load_one_npz_with_meta(npz_path, meta_path, name=stem))

    if not models:
        raise ValueError(f"No .npz models found in {compare_dir}")

    # basic shape consistency: all Y must share same (N,T)
    N0, T0 = models[0]["Y"].shape
    for m in models[1:]:
        N, T = m["Y"].shape
        if T != T0:
            raise AssertionError(
                f"Time bins mismatch among models. "
                f"{models[0]['name']} has T={T0}, {m['name']} has T={T}"
            )
        if N != N0:
            raise AssertionError(
                f"All synthetic models must have same N for fair comparison. "
                f"{models[0]['name']} has N={N0}, {m['name']} has N={N}"
            )

    return models


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


# --------------------------------------------------------------------------
# NEW: enforce same temporal grid between reference and all models
# --------------------------------------------------------------------------

def assert_same_temporal_grid(ref: Dict, models: List[Dict]) -> None:
    """
    Enforce that reference and all synthetic models share the same temporal grid.
    Checks:
      - T (number of bins) must match ref["T"]
      - If grid_min is present for both, they must match
      - If horizon_min is present for both, they must match
    Raises AssertionError with a precise message if any mismatch is found.
    """
    T_ref = ref["T"]
    grid_ref = ref.get("grid_min", None)
    horizon_ref = ref.get("horizon_min", None)

    for m in models:
        # T must match exactly
        if m["T"] != T_ref:
            raise AssertionError(
                f"Temporal mismatch: ref T={T_ref}, model '{m['name']}' T={m['T']}."
            )

        # grid_min must match when both are known
        if (grid_ref is not None) and (m.get("grid_min", None) is not None):
            if m["grid_min"] != grid_ref:
                raise AssertionError(
                    f"grid_min mismatch: ref={grid_ref}, model '{m['name']}'={m['grid_min']}."
                )

        # horizon_min must match when both are known
        if (horizon_ref is not None) and (m.get("horizon_min", None) is not None):
            if m["horizon_min"] != horizon_ref:
                raise AssertionError(
                    f"horizon_min mismatch: ref={horizon_ref}, model '{m['name']}'={m['horizon_min']}."
                )


# ---------------------------------------------------------------------
# Helpers for distributions over "cells" (activities / n-grams / schedules)
# ---------------------------------------------------------------------

def counts_to_probs(counts: Dict[Hashable, float],
                    eps: float = 1e-12) -> Dict[Hashable, float]:
    """
    Normalize a dict of non-negative counts into probabilities.
    If total == 0, returns all zeros.
    """
    total = float(sum(counts.values()))
    if total < eps:
        return {k: 0.0 for k in counts.keys()}
    return {k: float(v) / total for k, v in counts.items()}


def align_distributions(
    probs_ref: Dict[Hashable, float],
    probs_syn: Dict[Hashable, float],
) -> Tuple[List[Hashable], np.ndarray, np.ndarray]:
    """
    Align two probability dicts over the union of keys.

    Returns:
        keys: list of keys in a deterministic order
        p_ref: np.array of shape (K,)
        p_syn: np.array of shape (K,)
    """
    keys = sorted(set(probs_ref.keys()) | set(probs_syn.keys()))
    if not keys:
        return [], np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    p_ref = np.array([probs_ref.get(k, 0.0) for k in keys], dtype=np.float64)
    p_syn = np.array([probs_syn.get(k, 0.0) for k in keys], dtype=np.float64)
    return keys, p_ref, p_syn


def compute_srmse_from_probs(
    p_ref: np.ndarray,
    p_syn: np.ndarray,
    eps: float = 1e-12,
) -> float:
    """
    Compute SRMSE between two aligned probability vectors.

    We use the common scaled RMSE form from population synthesis:

        SRMSE = sqrt( sum_i (p_syn[i] - p_ref[i])^2
                      / max( sum_i p_ref[i]^2, eps ) )

    where p_ref and p_syn are probability vectors over the same cells.

    Args:
        p_ref: (K,) reference probabilities (non-negative, usually sum to 1)
        p_syn: (K,) synthetic probabilities
        eps:   tiny constant to avoid division by zero

    Returns:
        scalar SRMSE (float)
    """
    if p_ref.size == 0:
        return 0.0

    diff = p_syn - p_ref
    num = float(np.sum(diff * diff))
    den = float(np.sum(p_ref * p_ref))
    if den < eps:
        return 0.0
    return float(np.sqrt(num / den))


def compute_srmse_from_counts(
    counts_ref: Dict[Hashable, float],
    counts_syn: Dict[Hashable, float],
    eps: float = 1e-12,
) -> float:
    """
    Convenience wrapper: take raw counts, normalize to probabilities,
    align, then compute SRMSE.

    Args:
        counts_ref: dict cell -> count (reference)
        counts_syn: dict cell -> count (synthetic)
        eps:        tiny constant to avoid division by zero

    Returns:
        SRMSE scalar
    """
    probs_ref = counts_to_probs(counts_ref, eps=eps)
    probs_syn = counts_to_probs(counts_syn, eps=eps)
    _, p_ref, p_syn = align_distributions(probs_ref, probs_syn)
    return compute_srmse_from_probs(p_ref, p_syn, eps=eps)


# ---------------------------------------------------------------------
# N-gram / schedule counting utilities
# ---------------------------------------------------------------------

def ngram_counts(
    Y: np.ndarray,
    n: Optional[int],
    as_schedule: bool = False,
) -> Dict[Tuple[int, ...], int]:
    """
    Count n-grams (or full schedules) in a (N, T) label grid.

    Args:
        Y:          np.ndarray of shape (N, T), int labels.
        n:          length of n-gram (1,2,3,4,...) if as_schedule=False.
                    Ignored if as_schedule=True.
        as_schedule:
            - If True: each full row (schedule) is a single cell (tuple of length T).
            - If False: sliding window of length n along each row.

    Returns:
        counts: dict where key is a tuple of ints, value is count.
    """
    if Y.ndim != 2:
        raise ValueError(f"Y must be 2D (N,T), got shape {Y.shape}")

    N, T = Y.shape
    counts: Dict[Tuple[int, ...], int] = {}

    if as_schedule:
        # Treat each full row as one "cell"
        for i in range(N):
            seq = tuple(int(x) for x in Y[i])
            counts[seq] = counts.get(seq, 0) + 1
        return counts

    if n is None or n <= 0:
        raise ValueError("n must be a positive integer when as_schedule=False")

    if n > T:
        # No n-grams possible if n > T
        return counts

    for i in range(N):
        seq = Y[i]
        # sliding window of length n
        for t in range(T - n + 1):
            gram = tuple(int(x) for x in seq[t : t + n])
            counts[gram] = counts.get(gram, 0) + 1

    return counts


def schedule_counts(Y: np.ndarray) -> Dict[Tuple[int, ...], int]:
    """
    Shortcut: full-schedule counts (each row as one cell).
    """
    return ngram_counts(Y, n=None, as_schedule=True)
