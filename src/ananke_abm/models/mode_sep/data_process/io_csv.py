"""
CSV IO and strict validation for mode_sep model.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from ananke_abm.models.mode_sep.data_process.data_paths import DataPaths
from ananke_abm.models.mode_sep.data_process.id_maps import IdMaps


SNAPS_COLS = {
    "person_id": int,
    "timestamp": float,
    "location": str,
    "purpose": str,
    "anchor": int,
}

PERIODS_COLS = {
    "person_id": int,
    "start_time": float,
    "end_time": float,
    "type": str,  # {stay, travel}
    "location": str,
    "purpose": str,
    "mode": str,
}

ZONES_COLS = {
    "zone_id": int,
    "name": str,  # unique
    "type": str,
    "x_coord": float,
    "y_coord": float,
    "population": float,
    "job_opportunities": float,
    "retail_accessibility": float,
    "transit_accessibility": float,
    "attractiveness": float,
}

PERSONS_COLS = {
    "person_id": int,
    "name": str,
    "age": float,
    "income": float,
    "home_zone_id": int,
    "work_zone_id": int,
}


@dataclass
class LoadedCSVs:
    snaps: pd.DataFrame
    periods: pd.DataFrame
    zones: pd.DataFrame
    dist_mat: torch.Tensor
    zone_names: List[str]
    persons: pd.DataFrame
    id_maps: IdMaps


def _validate_columns(df: pd.DataFrame, required: Dict[str, type], name: str) -> None:
    missing = [c for c in required.keys() if c not in df.columns]
    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}. Expected columns: {list(required.keys())}."
        )


def _coerce_types(df: pd.DataFrame, schema: Dict[str, type]) -> pd.DataFrame:
    # Coerce numeric columns and strip strings
    for col, typ in schema.items():
        if typ in (int, float):
            df[col] = pd.to_numeric(df[col], errors="raise")
            if typ is int:
                # Ensure integer by casting after numeric
                df[col] = df[col].astype(int)
        elif typ is str:
            df[col] = df[col].astype(str)
    return df


def _load_and_validate_dist_mat(dist_path: str, zone_names: List[str]) -> torch.Tensor:
    # Read with pandas to preserve headers
    raw = pd.read_csv(dist_path)
    if raw.columns[0].lower() not in {"loc_id", "location", "name"}:
        raise ValueError(
            "dist_mat.csv: First column must be a location identifier header named 'loc_id' or 'location' or 'name'."
        )

    header_names = list(raw.columns[1:])
    if header_names != zone_names:
        raise ValueError(
            "dist_mat.csv header does not match zone order from zones.csv. "
            f"Expected: {zone_names} but got: {header_names}. "
            "Ensure zones.csv is sorted by zone_id and dist_mat columns use zone names in exactly that order."
        )

    # Validate row labels also match order
    row_names = raw.iloc[:, 0].tolist()
    if row_names != zone_names:
        raise ValueError(
            "dist_mat.csv row labels do not match zone order from zones.csv. "
            f"Expected first column values: {zone_names} but got: {row_names}."
        )

    mat = raw.iloc[:, 1:].to_numpy(dtype=np.float32)

    # Validate square, symmetry, and near-zero diagonal
    if mat.shape[0] != mat.shape[1]:
        raise ValueError(f"dist_mat.csv must be a square matrix; got shape {mat.shape}.")
    if not np.allclose(mat, mat.T, atol=1e-6):
        raise ValueError("dist_mat.csv must be symmetric (within 1e-6).")
    diag = np.diag(mat)
    if not np.all(np.abs(diag) <= 1e-6):
        raise ValueError("dist_mat.csv diagonal must be approximately 0 (|diag| <= 1e-6). Units must be km.")

    return torch.tensor(mat, dtype=torch.float32)


def load_csvs(paths: DataPaths) -> LoadedCSVs:
    # zones first (provides ordering and name mapping)
    zones = pd.read_csv(paths.zones_csv)
    _validate_columns(zones, ZONES_COLS, "zones.csv")
    zones = _coerce_types(zones, ZONES_COLS)
    zones = zones.sort_values("zone_id").reset_index(drop=True)

    # Build zone-related mappings
    zone_names: List[str] = zones["name"].tolist()
    zone_id_to_index = {int(zid): idx for idx, zid in enumerate(zones["zone_id"].tolist())}
    loc_id_to_index = {name: idx for idx, name in enumerate(zone_names)}
    index_to_loc_id = {idx: name for name, idx in loc_id_to_index.items()}

    # dist matrix
    dist_mat = _load_and_validate_dist_mat(str(paths.dist_mat_csv), zone_names)

    # persons
    persons = pd.read_csv(paths.persons_csv)
    _validate_columns(persons, PERSONS_COLS, "persons.csv")
    persons = _coerce_types(persons, PERSONS_COLS)

    # snaps
    snaps = pd.read_csv(paths.snaps_csv)
    _validate_columns(snaps, SNAPS_COLS, "snaps.csv")
    snaps = _coerce_types(snaps, SNAPS_COLS)

    # periods
    periods = pd.read_csv(paths.periods_csv)
    _validate_columns(periods, PERIODS_COLS, "periods.csv")
    periods = _coerce_types(periods, PERIODS_COLS)

    # Validate zone names used in snaps/periods
    unknown_snaps = sorted(set(snaps["location"]) - set(zone_names))
    if unknown_snaps:
        raise ValueError(
            "snaps.csv contains unknown location names not present in zones.csv: " + ", ".join(unknown_snaps)
        )
    # For periods: allow 'travel' rows to have a special 'location' placeholder like 'travel'
    ptype_lower = periods["type"].str.lower()
    periods_non_travel = periods[ptype_lower != "travel"]
    unknown_periods = sorted(set(periods_non_travel["location"]) - set(zone_names))
    if unknown_periods:
        raise ValueError(
            "periods.csv contains unknown location names not present in zones.csv: " + ", ".join(unknown_periods)
        )

    # Map names to indices
    snaps["loc_idx"] = snaps["location"].map(loc_id_to_index)
    periods["loc_idx"] = periods["location"].map(loc_id_to_index)
    # Set travel rows to -1 explicitly
    periods.loc[ptype_lower == "travel", "loc_idx"] = -1
    periods["loc_idx"] = periods["loc_idx"].fillna(-1).astype(int)

    # Ensure persons home/work zones exist and map to indices
    unknown_zone_ids = [
        int(z) for z in (
            set(persons["home_zone_id"]) | set(persons["work_zone_id"])  # union
        ) if int(z) not in zone_id_to_index
    ]
    if unknown_zone_ids:
        raise ValueError(
            "persons.csv references zone_id values not present in zones.csv: " + ", ".join(map(str, unknown_zone_ids))
        )

    # Final id_maps container
    id_maps = IdMaps(
        Z=len(zone_names),
        zone_names=zone_names,
        loc_id_to_index=loc_id_to_index,
        index_to_loc_id=index_to_loc_id,
        zone_id_to_index=zone_id_to_index,
    )

    return LoadedCSVs(
        snaps=snaps,
        periods=periods,
        zones=zones,
        dist_mat=dist_mat,
        zone_names=zone_names,
        persons=persons,
        id_maps=id_maps,
    )


