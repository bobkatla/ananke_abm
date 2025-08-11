"""
Inference utilities for mode_sep model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import pandas as pd
import torch

from ananke_abm.models.mode_sep.config import ModeSepConfig
from ananke_abm.models.mode_sep.data_process.data_paths import load_data_paths
from ananke_abm.models.mode_sep.data_process.io_csv import load_csvs
from ananke_abm.models.mode_sep.data_process.data import build_person_and_shared
from ananke_abm.models.mode_sep.data_process.batching import build_union_batch
from ananke_abm.models.mode_sep.architecture.model import ModeSepModel
from ananke_abm.models.mode_sep.inference.viz import plot_person_trajectory


def load_best_model(config: ModeSepConfig, Z: int) -> ModeSepModel:
    ckpt_path = Path(config.checkpoints_dir) / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Best checkpoint not found at {ckpt_path}. Train the model first.")
    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    model = ModeSepModel(Z=Z, config=config)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model


def run_inference(yaml_path: str = "src/ananke_abm/models/mode_sep/data_paths.yml"):
    config = ModeSepConfig()
    device = torch.device(config.device if (config.device == "cpu" or torch.cuda.is_available()) else "cpu")
    # Load CSVs and model
    dpaths = load_data_paths(yaml_path)
    loaded = load_csvs(dpaths)
    persons, shared = build_person_and_shared(loaded, device)

    model = load_best_model(config, Z=shared.id_maps.Z).to(device)

    out_csv_path = Path(config.runs_dir) / "model_predictions.csv"
    if out_csv_path.exists():
        out_csv_path.unlink()
    out_csv_path.parent.mkdir(parents=True, exist_ok=True)

    for p in persons:
        # Union grid for this person only
        union = build_union_batch([p], config, device)
        times_u = union.times_union
        home_idx = torch.tensor([p.home_zone_idx], dtype=torch.long, device=device)
        work_idx = torch.tensor([p.work_zone_idx], dtype=torch.long, device=device)
        traits = p.person_traits_raw.unsqueeze(0)

        with torch.no_grad():
            _, logits_u, v_u = model(times_union=times_u, home_idx=home_idx, work_idx=work_idx, person_traits_raw=traits)
            pred_idx_u = logits_u.argmax(dim=-1)[0].cpu().numpy()  # [T]

        # Dense grid [0, 24]
        t_dense = torch.linspace(0.0, 24.0, config.dense_resolution, device=device)
        with torch.no_grad():
            _, logits_d, _ = model(times_union=t_dense, home_idx=home_idx, work_idx=work_idx, person_traits_raw=traits)
            pred_idx_d = logits_d.argmax(dim=-1)[0].cpu().numpy()

        # Write predictions at GT snaps
        df_rows = []
        zone_names = shared.zone_names
        for j, is_gt in enumerate(union.is_gt_union[0].cpu().numpy()):
            if not bool(is_gt):
                continue
            t = float(times_u[j].cpu().item())
            snap_idx = int(union.snap_indices[0, j].cpu().item())
            gt_index = int(p.loc_ids[snap_idx].cpu().item())
            pred_index = int(pred_idx_u[j])
            distance_km = float(shared.dist_mat[gt_index, pred_index].cpu().item())
            df_rows.append({
                "person_id": p.person_id,
                "person_name": p.person_name,
                "timestamp": t,
                "gt_loc_id": zone_names[gt_index],
                "pred_loc_id": zone_names[pred_index],
                "gt_index": gt_index,
                "pred_index": pred_index,
                "distance_km": distance_km,
                "match": "yes" if gt_index == pred_index else "no",
            })

        df = pd.DataFrame(df_rows)
        if not df.empty:
            df.to_csv(out_csv_path, mode="a", header=not out_csv_path.exists(), index=False)

        # Save plot
        fig_path = Path(config.figures_dir) / f"trajectory_{p.person_id}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_person_trajectory(
            times_dense=t_dense.cpu().numpy(),
            pred_ids_dense=pred_idx_d,
            gt_times=p.times_snap.cpu().numpy(),
            gt_ids=p.loc_ids.cpu().numpy(),
            zone_names=zone_names,
            out_path=str(fig_path),
        )

    print(f"Predictions written to {out_csv_path}")


if __name__ == "__main__":
    run_inference()


