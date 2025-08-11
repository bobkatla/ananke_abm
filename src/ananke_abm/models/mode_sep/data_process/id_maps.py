"""
Small helpers for exposing ID mappings.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List


@dataclass(frozen=True)
class IdMaps:
    Z: int
    zone_names: List[str]
    loc_id_to_index: Dict[str, int]
    index_to_loc_id: Dict[int, str]
    zone_id_to_index: Dict[int, int]


