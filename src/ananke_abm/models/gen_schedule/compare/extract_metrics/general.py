import os
import numpy as np
from typing import Dict, List
from ananke_abm.models.gen_schedule.losses.jsd import jsd
from ananke_abm.models.gen_schedule.compare.utils import ensure_dir


def _minutes_share(Y: np.ndarray, P: int) -> np.ndarray:
    """
    Y: (N, T) int
    returns: (P,) float, overall share of minutes per purpose
    """
    flat = Y.ravel()
    counts = np.bincount(flat, minlength=P).astype(np.float64)
    total = float(flat.size)
    if total > 0:
        counts /= total
    return counts  # (P,)


def _tod_marginals(Y: np.ndarray, P: int) -> np.ndarray:
    """
    Y: (N, T)
    returns m[T, P] with m[t, p] = P(Y_t = p)
    """
    N, T = Y.shape
    m = np.zeros((T, P), dtype=np.float64)
    for t in range(T):
        col = Y[:, t]
        counts = np.bincount(col, minlength=P).astype(np.float64)
        if N > 0:
            counts /= float(N)
        m[t] = counts
    return m


def _bigram_matrix(Y: np.ndarray, P: int) -> np.ndarray:
    """
    Y: (N, T)
    returns M[P, P] with M[u, v] = P(y_t=u, y_{t+1}=v) over all t, normalized.
    """
    N, T = Y.shape
    M = np.zeros((P, P), dtype=np.float64)
    Z = 0.0
    for i in range(N):
        seq = Y[i]
        a = seq[:-1]
        b = seq[1:]
        for u, v in zip(a, b):
            M[u, v] += 1.0
            Z += 1.0
    if Z > 0:
        M /= Z
    return M


# ---------- METRICS ----------

def metric_minutes_share_stub(ref: Dict, models: List[Dict], outdir: str):
    """
    Compute per-purpose minutes-share for reference and each model.

    Writes:
        minutes_share_levels.csv with columns:
            model, share_<purpose1>, ..., share_<purposeP>

        minutes_share_abs_error.csv with columns:
            model, abs_error_<purpose1>, ..., abs_error_<purposeP>, mean_abs_error
    """
    ensure_dir(outdir)

    Y_ref: np.ndarray = ref["Y"]           # (N_ref, T)
    purpose_map: Dict[str, int] = ref["purpose_map"]
    P = len(purpose_map)

    # canonical ordering of purposes by index
    inv_purpose_map = {v: k for k, v in purpose_map.items()}
    purpose_indices = sorted(inv_purpose_map.keys())
    purpose_names = [inv_purpose_map[i] for i in purpose_indices]

    # reference shares
    share_ref_all = _minutes_share(Y_ref, P)  # (P,)

    # ---- levels CSV ----
    rows_levels = []
    # reference row
    ref_row_levels = {"model": "ref"}
    for idx, pname in zip(purpose_indices, purpose_names):
        ref_row_levels[f"share_{pname}"] = float(share_ref_all[idx])
    rows_levels.append(ref_row_levels)

    # model rows
    for m in models:
        Y_model: np.ndarray = m["Y"]
        name: str = m["name"]
        share_model_all = _minutes_share(Y_model, P)

        row = {"model": name}
        for idx, pname in zip(purpose_indices, purpose_names):
            row[f"share_{pname}"] = float(share_model_all[idx])
        rows_levels.append(row)

    import csv
    levels_csv_path = os.path.join(outdir, "minutes_share_levels.csv")
    fieldnames_levels = list(rows_levels[0].keys())
    with open(levels_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_levels)
        writer.writeheader()
        for r in rows_levels:
            writer.writerow(r)

    # ---- abs error CSV ----
    rows_err = []
    # reference row: zero error by definition
    ref_row_err = {"model": "ref"}
    for pname in purpose_names:
        ref_row_err[f"abs_error_{pname}"] = 0.0
    ref_row_err["mean_abs_error"] = 0.0
    rows_err.append(ref_row_err)

    for m in models:
        Y_model: np.ndarray = m["Y"]
        name: str = m["name"]
        share_model_all = _minutes_share(Y_model, P)

        row = {"model": name}
        abs_errors = []
        for idx, pname in zip(purpose_indices, purpose_names):
            sr = float(share_ref_all[idx])
            sm = float(share_model_all[idx])
            ae = abs(sm - sr)
            row[f"abs_error_{pname}"] = ae
            abs_errors.append(ae)
        row["mean_abs_error"] = float(np.mean(abs_errors)) if abs_errors else 0.0
        rows_err.append(row)

    err_csv_path = os.path.join(outdir, "minutes_share_abs_error.csv")
    fieldnames_err = list(rows_err[0].keys())
    with open(err_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames_err)
        writer.writeheader()
        for r in rows_err:
            writer.writerow(r)


def metric_tod_jsd_stub(ref: Dict, models: List[Dict], outdir: str):
    """
    Compute macro-averaged JSD between reference and each model
    over time-of-day marginals.

    Writes:
        tod_jsd.csv with columns: model, tod_jsd_macro
    """
    ensure_dir(outdir)

    Y_ref: np.ndarray = ref["Y"]
    purpose_map: Dict[str, int] = ref["purpose_map"]
    P = len(purpose_map)

    m_ref = _tod_marginals(Y_ref, P)   # (T, P)

    rows = []

    # reference row
    rows.append({"model": "ref", "tod_jsd_macro": 0.0})

    for m in models:
        Y_model: np.ndarray = m["Y"]
        name: str = m["name"]
        m_syn = _tod_marginals(Y_model, P)  # (T, P)

        # macro-average JSD over time bins
        T = m_ref.shape[0]
        jsds = []
        for t in range(T):
            p = m_ref[t]
            q = m_syn[t]
            # guard tiny numerical issues
            p = p.astype(np.float64)
            q = q.astype(np.float64)
            # jsd expects valid distributions; small epsilon handling is inside jsd
            jsds.append(jsd(p, q))
        tod_jsd_macro = float(np.mean(jsds)) if jsds else 0.0

        rows.append({"model": name, "tod_jsd_macro": tod_jsd_macro})

    # write CSV
    import csv
    csv_path = os.path.join(outdir, "tod_jsd.csv")
    fieldnames = ["model", "tod_jsd_macro"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def metric_bigram_L1_stub(ref: Dict, models: List[Dict], outdir: str):
    """
    Compute bigram matrices for ref and each model, and store L1 distance.

    Writes:
        bigram_L1.csv with columns: model, bigram_L1
    """
    ensure_dir(outdir)

    Y_ref: np.ndarray = ref["Y"]
    purpose_map: Dict[str, int] = ref["purpose_map"]
    P = len(purpose_map)

    B_ref = _bigram_matrix(Y_ref, P)

    rows = []

    # reference row
    rows.append({"model": "ref", "bigram_L1": 0.0})

    for m in models:
        Y_model: np.ndarray = m["Y"]
        name: str = m["name"]
        B_model = _bigram_matrix(Y_model, P)
        bigram_L1 = float(np.abs(B_model - B_ref).sum())
        rows.append({"model": name, "bigram_L1": bigram_L1})

    # write CSV
    import csv
    csv_path = os.path.join(outdir, "bigram_L1.csv")
    fieldnames = ["model", "bigram_L1"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


GENERAL_FUNCS = {
    "minutes_share":        metric_minutes_share_stub,
    "tod_jsd":              metric_tod_jsd_stub,
    "bigram_L1":            metric_bigram_L1_stub,
}