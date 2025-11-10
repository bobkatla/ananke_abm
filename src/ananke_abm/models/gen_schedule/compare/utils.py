import numpy as np
import os
import json
from typing import Dict, List


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

    return {
        "name": name,
        "Y": Y,                       # (N,T)
        "purpose_map": purpose_map,   # {name: idx}
        "grid_min": grid_min,
        "horizon_min": horizon_min,
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