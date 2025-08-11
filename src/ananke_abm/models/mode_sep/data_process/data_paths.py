"""
Utilities for loading and validating data paths from a YAML file.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import yaml


@dataclass(frozen=True)
class DataPaths:
    snaps_csv: Path
    periods_csv: Path
    zones_csv: Path
    dist_mat_csv: Path
    persons_csv: Path
    id_maps: Path


REQUIRED_KEYS = [
    "snaps_csv",
    "periods_csv",
    "zones_csv",
    "dist_mat_csv",
    "persons_csv",
    "id_maps",
]


def _normalize_and_validate_paths(raw: Dict[str, str], base_dir: Path) -> DataPaths:
    missing = [k for k in REQUIRED_KEYS if k not in raw]
    if missing:
        raise ValueError(
            f"data_paths.yml is missing required keys: {missing}. "
            f"Expected keys: {REQUIRED_KEYS}."
        )

    def norm(p: str) -> Path:
        # Expand user and make absolute relative to YAML directory
        path = Path(p).expanduser()
        if not path.is_absolute():
            path = (base_dir / path).resolve()
        return path

    dp = DataPaths(
        snaps_csv=norm(raw["snaps_csv"]),
        periods_csv=norm(raw["periods_csv"]),
        zones_csv=norm(raw["zones_csv"]),
        dist_mat_csv=norm(raw["dist_mat_csv"]),
        persons_csv=norm(raw["persons_csv"]),
        id_maps=norm(raw["id_maps"]),
    )

    # Verify all files exist
    errors = [str(p) for p in [
        dp.snaps_csv, dp.periods_csv, dp.zones_csv,
        dp.dist_mat_csv, dp.persons_csv, dp.id_maps
    ] if not p.exists()]
    if errors:
        raise FileNotFoundError(
            "The following paths from data_paths.yml do not exist: " + ", ".join(errors)
        )

    return dp


def load_data_paths(yaml_path: str | Path) -> DataPaths:
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        raise FileNotFoundError(
            f"data_paths.yml not found at {yaml_path}. Create it with the required keys: {REQUIRED_KEYS}."
        )
    with open(yaml_path, "r", encoding="utf-8") as f:
        try:
            raw = yaml.safe_load(f) or {}
        except Exception as e:
            raise ValueError(f"Failed to parse YAML at {yaml_path}: {e}") from e

    return _normalize_and_validate_paths(raw, yaml_path.parent)


