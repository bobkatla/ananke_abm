"""Generate mock CSV outputs for two mock people.

This script creates periods.csv and snaps.csv using the mock people and
schedule data defined in mock_2p.py together with zone information from
mock_locations.py. The resulting files are written to the repository's
``data/`` directory and can be regenerated at any time by running this
script.
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple

import csv
import sys
import types

# Allow running as a script by ensuring the package root is on sys.path
if __package__ is None:  # pragma: no cover - runtime path fix
    sys.path.append(str(Path(__file__).resolve().parents[2]))

# Stub heavy optional dependencies to keep this script lightweight
sys.modules.setdefault("torch", types.ModuleType("torch"))
sys.modules.setdefault("numpy", types.ModuleType("numpy"))

class _Graph:
    def __init__(self):
        self._nodes = {}
        self._edges = []

    def add_node(self, node_id, **attrs):
        self._nodes[node_id] = attrs

    def add_edges_from(self, edges):
        self._edges.extend(edges)

    def nodes(self):  # pragma: no cover - minimal implementation
        return self._nodes.keys()

    def edges(self):  # pragma: no cover - minimal implementation
        return [(u, v) for u, v, _ in self._edges]

sys.modules.setdefault("networkx", types.SimpleNamespace(Graph=_Graph))

from ananke_abm.data_generator import mock_2p, mock_locations

# Replace distance matrix calculation with no-op to avoid numpy requirements
mock_locations.create_distance_matrix = lambda zones: None

create_sarah = mock_2p.create_sarah
create_marcus = mock_2p.create_marcus
create_sarah_daily_pattern = mock_2p.create_sarah_daily_pattern
create_marcus_daily_pattern = mock_2p.create_marcus_daily_pattern
create_mock_zone_graph = mock_locations.create_mock_zone_graph


def build_person_periods(person, schedule, zone_data) -> List[Dict]:
    """Convert a daily schedule into periods for one person."""
    periods: List[Dict] = []

    n = len(schedule)
    i = 0
    while i < n - 1:
        event = schedule[i]
        mode_type = "stay" if event.get("travel_mode", "Stay") == "Stay" else "travel"
        start_time = event["time"]
        j = i + 1
        # extend j while mode stays the same (merging contiguous events)
        while j < n and (("stay" if schedule[j].get("travel_mode", "Stay") == "Stay" else "travel") == mode_type):
            j += 1
        end_time = schedule[j]["time"] if j < n else schedule[-1]["time"]

        if mode_type == "stay":
            location = zone_data[event["zone"]]["name"]
            purpose = event["activity"]
            periods.append(
                {
                    "person_id": person.person_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "type": "stay",
                    "location": location,
                    "purpose": purpose,
                    "mode": "stay",
                }
            )
        else:  # travel
            periods.append(
                {
                    "person_id": person.person_id,
                    "start_time": start_time,
                    "end_time": end_time,
                    "type": "travel",
                    "location": "travel",
                    "purpose": "travel",
                    "mode": event["travel_mode"].lower(),
                }
            )
        i = j
    return periods


def build_snaps_from_periods(periods: List[Dict]) -> List[Dict]:
    """Create snaps from a list of period data."""
    snaps: List[Dict] = []

    activity_to_group = {
        # Home activities
        "sleep": "home", "morning_routine": "home", "evening": "home",
        "dinner": "home", "arrive_home": "home",
        # Work/Education
        "work": "work", "arrive_work": "work", "end_work": "work",
        # Subsistence
        "lunch": "shopping", "lunch_start": "shopping", "lunch_end": "shopping",
        # Leisure & Recreation
        "gym": "social", "gym_end": "social",
        "exercise": "social", "leaving_park": "social",
        # Social
        "social": "social", "leaving_social": "social", "dinner_social": "social",
        # Travel/Transit
        "prepare_commute": "travel", "start_commute": "travel",
        "transit": "travel", "leaving_home": "travel",
        "break": "travel",
    }
    
    person_periods = {}
    for p in periods:
        person_periods.setdefault(p['person_id'], []).append(p)

    all_snaps = []
    for person_id, single_person_periods in person_periods.items():
        person_snaps = []
        for period in single_person_periods:
            if period["type"] == "stay":
                grouped_purpose = activity_to_group.get(period["purpose"], period["purpose"])
                
                person_snaps.append({
                    "person_id": person_id,
                    "timestamp": period["start_time"],
                    "location": period["location"],
                    "purpose": grouped_purpose,
                    "anchor": 0,
                })
                person_snaps.append({
                    "person_id": person_id,
                    "timestamp": period["end_time"],
                    "location": period["location"],
                    "purpose": grouped_purpose,
                    "anchor": 0,
                })
        
        if person_snaps:
            person_snaps.sort(key=lambda x: x['timestamp'])
            person_snaps[0]["anchor"] = 1
            person_snaps[-1]["anchor"] = 1
        
        all_snaps.extend(person_snaps)
        
    all_snaps.sort(key=lambda x: (x['person_id'], x['timestamp']))

    return all_snaps


def main() -> None:
    """Generate CSV files for the mock schedules."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    _, zone_data, _ = create_mock_zone_graph()

    sarah = create_sarah()
    sarah_schedule = create_sarah_daily_pattern()
    sarah_periods = build_person_periods(sarah, sarah_schedule, zone_data)

    marcus = create_marcus()
    marcus_schedule = create_marcus_daily_pattern()
    marcus_periods = build_person_periods(marcus, marcus_schedule, zone_data)

    all_periods = sarah_periods + marcus_periods
    all_snaps = build_snaps_from_periods(all_periods)

    periods_path = data_dir / "periods.csv"
    snaps_path = data_dir / "snaps.csv"

    period_fields = [
        "person_id",
        "start_time",
        "end_time",
        "type",
        "location",
        "purpose",
        "mode",
    ]
    with periods_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=period_fields)
        writer.writeheader()
        writer.writerows(all_periods)

    snap_fields = ["person_id", "timestamp", "location", "purpose", "anchor"]
    with snaps_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=snap_fields)
        writer.writeheader()
        writer.writerows(all_snaps)


if __name__ == "__main__":
    main()
