"""Generate mock CSV outputs for two mock people.

This script creates:
  - periods.csv (unchanged)
  - snaps.csv   (unchanged)
  - zones.csv
  - dist_mat.csv
  - persons.csv
  - segments.csv
  - id_maps.json

using the mock people and schedules from mock_2p.py and zone information
from mock_locations.py. Files are written to the repository's `data/` directory.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple
import csv
import json
import sys

# Ensure package root is on path when run as a script
if __package__ is None:  # pragma: no cover - runtime path fix
    sys.path.append(str(Path(__file__).resolve().parents[2]))

# IMPORTANT: We *do not* stub numpy/torch/networkx anymore; we need real distance matrix
from ananke_abm.data_generator import mock_2p, mock_locations


create_sarah = mock_2p.create_sarah
create_marcus = mock_2p.create_marcus
create_sarah_daily_pattern = mock_2p.create_sarah_daily_pattern
create_marcus_daily_pattern = mock_2p.create_marcus_daily_pattern
create_mock_zone_graph = mock_locations.create_mock_zone_graph
create_distance_matrix = mock_locations.create_distance_matrix


def build_person_periods(person, schedule, zone_data) -> List[Dict]:
    """Convert a daily schedule into periods for one person.

    Keeps legacy behavior for periods.csv (do not change schema/semantics).
    """
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
    """Create snaps from a list of period data. (Legacy behavior kept)"""
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
        person_periods.setdefault(p["person_id"], []).append(p)

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
            person_snaps.sort(key=lambda x: x["timestamp"])
            person_snaps[0]["anchor"] = 1
            person_snaps[-1]["anchor"] = 1

        all_snaps.extend(person_snaps)

    all_snaps.sort(key=lambda x: (x["person_id"], x["timestamp"]))
    return all_snaps


def build_segments_from_periods(periods: List[Dict]) -> List[Dict]:
    """Emit segments.csv rows: one per travel period with origin/destination."""
    segments: List[Dict] = []
    # group by person
    by_person: Dict[int, List[Dict]] = {}
    for p in periods:
        by_person.setdefault(p["person_id"], []).append(p)

    for pid, plist in by_person.items():
        # ensure chronological
        plist = sorted(plist, key=lambda r: r["start_time"])
        for idx, row in enumerate(plist):
            if row["type"] != "travel":
                continue
            # find previous stay and next stay
            origin = None
            destination = None
            # previous
            for j in range(idx - 1, -1, -1):
                if plist[j]["type"] == "stay":
                    origin = plist[j]["location"]
                    break
            # next
            for j in range(idx + 1, len(plist)):
                if plist[j]["type"] == "stay":
                    destination = plist[j]["location"]
                    break
            if origin is None or destination is None:
                # skip malformed triples
                continue

            segments.append({
                "person_id": pid,
                "start_time": row["start_time"],
                "end_time": row["end_time"],
                "origin": origin,
                "destination": destination,
                "mode": row["mode"],
            })
    return segments


def write_zones_csv(zones_path: Path, zones_data: Dict[int, Dict]) -> None:
    fields = [
        "zone_id", "name", "type", "x_coord", "y_coord",
        "population", "job_opportunities", "retail_accessibility",
        "transit_accessibility", "attractiveness",
    ]
    with zones_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for zid in sorted(zones_data.keys()):
            z = zones_data[zid]
            w.writerow({
                "zone_id": zid,
                "name": z["name"],
                "type": z["type"],
                "x_coord": z["coordinates"][0],
                "y_coord": z["coordinates"][1],
                "population": z["population"],
                "job_opportunities": z["job_opportunities"],
                "retail_accessibility": z["retail_accessibility"],
                "transit_accessibility": z["transit_accessibility"],
                "attractiveness": z["attractiveness"],
            })


def write_dist_mat_csv(dist_path: Path, zones_data: Dict[int, Dict]) -> None:
    # compute using real helper (no stub)
    D = create_distance_matrix(zones_data)  # torch tensor (Z,Z)
    import torch
    D = D.detach().cpu().numpy()
    # header order by increasing zone_id
    names = [zones_data[zid]["name"] for zid in sorted(zones_data.keys())]
    with dist_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["loc_id"] + names)
        for i, name in enumerate(names):
            row = [name] + [f"{D[i, j]:.6f}" for j in range(len(names))]
            w.writerow(row)
    # quick validation
    import numpy as np
    assert D.shape[0] == D.shape[1], "dist_mat must be square"
    assert np.allclose(D, D.T, atol=1e-6), "dist_mat must be symmetric"
    assert np.allclose(np.diag(D), 0.0, atol=1e-6), "dist_mat diagonal must be 0"


def write_persons_csv(persons_path: Path, zone_name_to_id: Dict[str, int]) -> None:
    sarah = create_sarah()
    marcus = create_marcus()
    rows = []
    for p in (sarah, marcus):
        rows.append({
            "person_id": p.person_id,
            "name": p.name,
            "age": p.age,
            "income": p.income,
            "home_zone_id": zone_name_to_id[p.home_zone],
            "work_zone_id": zone_name_to_id[p.work_zone],
        })
    fields = ["person_id", "name", "age", "income", "home_zone_id", "work_zone_id"]
    with persons_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def write_id_maps_json(idmaps_path: Path, zones_data: Dict[int, Dict], periods: List[Dict]) -> None:
    # loc_id_to_index: order by zone_id (1..Z)
    loc_names = [zones_data[zid]["name"] for zid in sorted(zones_data.keys())]
    loc_id_to_index = {name: i for i, name in enumerate(loc_names)}
    # mode_to_index: derive from periods, stable sort
    modes = sorted({row["mode"].lower() for row in periods if row["type"] in ("stay", "travel")})
    mode_to_index = {m: i for i, m in enumerate(modes)}
    with idmaps_path.open("w") as f:
        json.dump(
            {
                "loc_id_to_index": loc_id_to_index,
                "mode_to_index": mode_to_index,
            },
            f,
            indent=2,
        )


def main() -> None:
    """Generate CSV files for the mock schedules (backward compatible)."""
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)

    # Real graph, zones, and distance matrix (no stub)
    _, zone_data, _ = create_mock_zone_graph()

    # Build people and periods
    sarah = create_sarah()
    sarah_schedule = create_sarah_daily_pattern()
    sarah_periods = build_person_periods(sarah, sarah_schedule, zone_data)

    marcus = create_marcus()
    marcus_schedule = create_marcus_daily_pattern()
    marcus_periods = build_person_periods(marcus, marcus_schedule, zone_data)

    all_periods = sarah_periods + marcus_periods
    all_snaps = build_snaps_from_periods(all_periods)
    all_segments = build_segments_from_periods(all_periods)

    # ---- Write legacy files (unchanged schema) ----
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

    # ---- Write new supporting artefacts (additive, non-breaking) ----
    zones_path = data_dir / "zones.csv"
    dist_path = data_dir / "dist_mat.csv"
    persons_path = data_dir / "persons.csv"
    segments_path = data_dir / "segments.csv"
    idmaps_path = data_dir / "id_maps.json"

    write_zones_csv(zones_path, zone_data)
    write_dist_mat_csv(dist_path, zone_data)

    # zone name → id map for persons.csv
    zone_name_to_id = {zone_data[zid]["name"]: zid for zid in sorted(zone_data.keys())}
    write_persons_csv(persons_path, zone_name_to_id)

    # segments.csv
    seg_fields = ["person_id", "start_time", "end_time", "origin", "destination", "mode"]
    with segments_path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=seg_fields)
        w.writeheader()
        w.writerows(all_segments)

    # id_maps.json
    write_id_maps_json(idmaps_path, zone_data, all_periods)

    print(f"✅ Wrote {periods_path}")
    print(f"✅ Wrote {snaps_path}")
    print(f"✅ Wrote {zones_path}")
    print(f"✅ Wrote {dist_path}")
    print(f"✅ Wrote {persons_path}")
    print(f"✅ Wrote {segments_path}")
    print(f"✅ Wrote {idmaps_path}")


if __name__ == "__main__":
    main()
