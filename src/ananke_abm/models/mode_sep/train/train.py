"""
Training script for mode_sep model.
"""
from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from ananke_abm.models.mode_sep.config import ModeSepConfig
from ananke_abm.models.mode_sep.data_process.data_paths import load_data_paths
from ananke_abm.models.mode_sep.data_process.io_csv import load_csvs
from ananke_abm.models.mode_sep.data_process.data import build_person_and_shared, PersonData
from ananke_abm.models.mode_sep.data_process.batching import build_union_batch
from ananke_abm.models.mode_sep.architecture.model import ModeSepModel
from ananke_abm.models.mode_sep.architecture.losses import total_loss


class PersonsDataset(Dataset):
    def __init__(self, persons: List[PersonData]):
        self.persons = persons

    def __len__(self):
        return len(self.persons)

    def __getitem__(self, idx: int) -> PersonData:
        return self.persons[idx]


def seed_everything(seed: int):
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def train(yaml_path: str = "src/ananke_abm/models/mode_sep/data_paths.yml"):
    config = ModeSepConfig()
    seed_everything(config.seed)

    # Device
    device = torch.device(config.device if (config.device == "cpu" or torch.cuda.is_available()) else "cpu")

    # IO setup
    Path(config.checkpoints_dir).mkdir(parents=True, exist_ok=True)
    Path(config.figures_dir).mkdir(parents=True, exist_ok=True)
    Path(config.runs_dir).mkdir(parents=True, exist_ok=True)

    # Load CSVs
    dpaths = load_data_paths(yaml_path)
    loaded = load_csvs(dpaths)
    persons, shared = build_person_and_shared(loaded, device)

    # Model
    model = ModeSepModel(Z=shared.id_maps.Z, config=config).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    # Simple batching: batch size 1 or 2
    dataset = PersonsDataset(persons)
    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=lambda xs: xs)

    best_loss = np.inf
    curves_path = Path(config.runs_dir) / "curves.csv"
    if not curves_path.exists():
        with open(curves_path, "w", encoding="utf-8") as f:
            f.write("epoch,loss,ce,mse,dist,stay_vel,move_vel,acc\n")

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running = {"loss": 0.0, "ce": 0.0, "mse": 0.0, "dist": 0.0, "stay_vel": 0.0, "move_vel": 0.0, "acc": 0.0}
        batches = 0

        for batch_persons in loader:
            union = build_union_batch(batch_persons, config, device)
            times_u = union.times_union

            # Build batched inputs for model
            home_idx = torch.tensor([p.home_zone_idx for p in batch_persons], dtype=torch.long, device=device)
            work_idx = torch.tensor([p.work_zone_idx for p in batch_persons], dtype=torch.long, device=device)
            traits = torch.stack([p.person_traits_raw for p in batch_persons], dim=0)  # [B,2]

            pred_emb, logits, v = model(
                times_union=times_u,
                home_idx=home_idx,
                work_idx=work_idx,
                person_traits_raw=traits,
            )

            # Ground truth indices at union times
            B, T, _ = logits.shape
            y_union = torch.full((B, T), -1, dtype=torch.long, device=device)
            for i, p in enumerate(batch_persons):
                sidx = union.snap_indices[i]
                mask = sidx >= 0
                if mask.any():
                    y_union[i, mask] = p.loc_ids[sidx[mask]]

            # Loss components
            ce_mse_dist_loss, parts = total_loss(
                config=config,
                logits=logits,
                pred_emb=pred_emb,
                y_union=y_union,
                is_gt_mask=union.is_gt_union,
                dist_mat=shared.dist_mat,
                class_table=model.class_table,
            )

            # Velocity regularization
            v_abs = v.norm(dim=-1)  # [B, T]
            # 1) stay_v: inside stays but not at snaps -> want |v|≈0
            m_stay_core = union.stay_non_gt_mask   # [B, T] bool
            if m_stay_core.any():
                stay_vel_pen = (v_abs[m_stay_core] ** 2).mean()
            else:
                stay_vel_pen = torch.zeros((), device=v.device)
            # 2) move_v: at interior GT snaps (exclude first/last) -> want |v| >= v_min_move
            m_gt_move = union.gt_interior_mask
            if m_gt_move.any():
                v_m = v_abs[m_gt_move]
                low  = (config.v_min_move - v_m).clamp(min=0.0)
                high = (v_m - config.v_max_move).clamp(min=0.0)
                move_vel_pen = (low**2 + high**2).mean()
            else:
                move_vel_pen = torch.zeros((), device=v.device)

            # total loss
            total = ce_mse_dist_loss \
                    + config.w_stay_vel_core * stay_vel_pen \
                    + config.w_move_vel_hinge * move_vel_pen

            optim.zero_grad(set_to_none=True)
            total.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.grad_clip)
            optim.step()

            # Accuracy at GT snaps
            with torch.no_grad():
                pred_idx = logits.argmax(dim=-1)  # [B,T]
                gt_mask = union.is_gt_union
                correct = ((pred_idx == y_union) & gt_mask)
                acc = (correct.sum().float() / gt_mask.sum().clamp(min=1)).item()
                # n_gt = union.is_gt_union.sum().item()
                # n_gt_move = union.gt_interior_mask.sum().item()
                # n_stay_core = union.stay_non_gt_mask.sum().item()
                # # mean |v| at interior GT snaps
                # mean_v_gt = v_abs[union.gt_interior_mask].mean().item() if n_gt_move > 0 else float('nan')
                # # mean |v| inside stays (non-snap)
                # mean_v_stay = v_abs[union.stay_non_gt_mask].mean().item() if n_stay_core > 0 else float('nan')
                # print(f"n_gt: {n_gt}, n_gt_move: {n_gt_move}, n_stay_core: {n_stay_core}")
                # print(f"mean_v_gt: {mean_v_gt}, mean_v_stay: {mean_v_stay}")

            # Track
            running["loss"] += float(total.item())
            running["ce"] += parts["ce"]
            running["mse"] += parts["mse"]
            running["dist"] += parts["dist"]
            running["stay_vel"] += float(stay_vel_pen.detach().item())
            running["move_vel"] += float(move_vel_pen.detach().item())
            running["acc"] += acc
            batches += 1

        # Averages
        for k in running:
            running[k] /= max(batches, 1)

        # Save curves
        with open(curves_path, "a", encoding="utf-8") as f:
            f.write(
                f"{epoch},{running['loss']:.6f},{running['ce']:.6f},{running['mse']:.6f},{running['dist']:.6f},{running['stay_vel']:.6f},{running['move_vel']:.6f},{running['acc']:.6f}\n"
            )

        # Checkpoint on best accuracy
        if running["loss"] < best_loss:
            best_loss = running["loss"]
            ckpt = {
                "model_state": model.state_dict(),
                "config": vars(config),
                "Z": shared.id_maps.Z,
            }
            torch.save(ckpt, str(Path(config.checkpoints_dir) / "best.pt"))

        if epoch % 20 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:4d} | loss={running['loss']:.4f} ce={running['ce']:.4f} mse={running['mse']:.4f} "
                f"dist={running['dist']:.4f} stay_vel={running['stay_vel']:.4f} move_vel={running['move_vel']:.4f} acc={running['acc']:.3f}",
                flush=True,
            )

    print("Training complete.")


if __name__ == "__main__":
    train()
