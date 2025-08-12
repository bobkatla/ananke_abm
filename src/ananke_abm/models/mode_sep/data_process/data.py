"""
Builds per-person dictionaries and shared objects from loaded CSVs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import pandas as pd

from ananke_abm.models.mode_sep.data_process.io_csv import LoadedCSVs
from ananke_abm.models.mode_sep.data_process.id_maps import IdMaps


@dataclass
class PersonData:
    person_id: int
    person_name: str
    times_snap: torch.Tensor          # (S,) float32
    loc_ids: torch.Tensor             # (S,) long
    stay_intervals: List[Tuple[float, float]]
    stay_segments: List[Tuple[float, float, int]]
    home_zone_idx: int
    work_zone_idx: int
    person_traits_raw: torch.Tensor   # (2,) [age_norm, income_norm]


@dataclass
class SharedData:
    dist_mat: torch.Tensor            # (Z, Z)
    zone_names: List[str]
    id_maps: IdMaps


def build_person_and_shared(loaded: LoadedCSVs, device: torch.device) -> Tuple[List[PersonData], SharedData]:
    snaps = loaded.snaps.copy()
    periods = loaded.periods.copy()
    persons = loaded.persons.copy()
    id_maps = loaded.id_maps

    # Prepare per-person data
    people: List[PersonData] = []
    for _, prow in persons.iterrows():
        pid = int(prow["person_id"])  # unique
        pname = str(prow["name"]) if "name" in prow and not pd.isna(prow["name"]) else str(pid)
        # person snaps
        s_df = snaps[snaps["person_id"] == pid].sort_values("timestamp")
        times = torch.tensor(s_df["timestamp"].to_numpy(dtype=np.float32), dtype=torch.float32, device=device)
        locs = torch.tensor(s_df["loc_idx"].to_numpy(dtype=np.int64), dtype=torch.long, device=device)

        # stay intervals
        p_df = periods[(periods["person_id"] == pid) & (periods["type"].str.lower() == "stay")]
        stays = [(float(r["start_time"]), float(r["end_time"])) for _, r in p_df.iterrows()]
        stay_segments = [(float(r["start_time"]), float(r["end_time"]), int(r["loc_idx"])) for _, r in p_df.iterrows()]

        # home/work indices
        home_zone_id = int(prow["home_zone_id"])
        work_zone_id = int(prow["work_zone_id"])
        if home_zone_id not in id_maps.zone_id_to_index or work_zone_id not in id_maps.zone_id_to_index:
            raise ValueError(
                f"Person {pid} refers to home_zone_id/work_zone_id not present in zones.csv."
            )
        home_idx = id_maps.zone_id_to_index[home_zone_id]
        work_idx = id_maps.zone_id_to_index[work_zone_id]

        # person traits (normalized)
        age_norm = float(prow["age"]) / 100.0
        income_norm = float(prow["income"]) / 1e5
        traits = torch.tensor([age_norm, income_norm], dtype=torch.float32, device=device)

        people.append(PersonData(
            person_id=pid,
            person_name=pname,
            times_snap=times,
            loc_ids=locs,
            stay_intervals=stays,
            stay_segments=stay_segments,
            home_zone_idx=home_idx,
            work_zone_idx=work_idx,
            person_traits_raw=traits,
        ))

    shared = SharedData(
        dist_mat=loaded.dist_mat.to(device),
        zone_names=loaded.zone_names,
        id_maps=id_maps,
    )
    return people, shared


