"""
Evaluation utilities for mode_sep model.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
import matplotlib.pyplot as plt

from ananke_abm.models.mode_sep.config import ModeSepConfig
from ananke_abm.models.mode_sep.data_process.data_paths import load_data_paths
from ananke_abm.models.mode_sep.data_process.io_csv import load_csvs
from ananke_abm.models.mode_sep.data_process.data import build_person_and_shared
from ananke_abm.models.mode_sep.data_process.batching import build_union_batch
from ananke_abm.models.mode_sep.architecture.model import ModeSepModel
from ananke_abm.models.mode_sep.inference.viz import plot_person_trajectory


def _roc_auc_binary(scores: np.ndarray, labels: np.ndarray) -> float:
    # Compute ROC AUC using Mann–Whitney U relationship
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    # Rank-based AUC: P(score_pos > score_neg)
    # Use broadcasting; may be memory-heavy for very large arrays but fine here
    comp = (pos.reshape(-1, 1) > neg.reshape(1, -1)).mean()
    return float(comp)


def evaluate(yaml_path: str = "src/ananke_abm/models/mode_sep/data_paths.yml"):
    config = ModeSepConfig()
    device = torch.device(config.device if (config.device == "cpu" or torch.cuda.is_available()) else "cpu")

    dpaths = load_data_paths(yaml_path)
    loaded = load_csvs(dpaths)
    persons, shared = build_person_and_shared(loaded, device)

    # Load model
    ckpt_path = Path(config.checkpoints_dir) / "best.pt"
    if not ckpt_path.exists():
        print(f"Best checkpoint not found at {ckpt_path}. Train first.")
        return
    ckpt = torch.load(str(ckpt_path), map_location=device)
    model = ModeSepModel(Z=ckpt["Z"], config=config).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    total_correct = 0
    total_snaps = 0
    dist_vals: List[float] = []

    # Velocity diagnostics accumulators
    stay_velocities: List[float] = []
    travel_velocities: List[float] = []
    all_abs_v: List[float] = []
    all_labels: List[int] = []

    for p in persons:
        union = build_union_batch([p], config, device)
        times_u = union.times_union
        home_idx = torch.tensor([p.home_zone_idx], dtype=torch.long, device=device)
        work_idx = torch.tensor([p.work_zone_idx], dtype=torch.long, device=device)
        traits = p.person_traits_raw.unsqueeze(0)

        with torch.no_grad():
            pred_emb, logits, v = model(times_union=times_u, home_idx=home_idx, work_idx=work_idx, person_traits_raw=traits)
            pred_idx = logits.argmax(dim=-1)[0]  # [T]
            # Evaluate at snaps
            mask = union.is_gt_union[0]
            if mask.sum() > 0:
                gt_idx = p.loc_ids[union.snap_indices[0, mask]]
                pr_idx = pred_idx[mask]
                total_correct += int((gt_idx == pr_idx).sum().item())
                total_snaps += int(mask.sum().item())
                # Expected distance at snaps (unweighted mean)
                for gi, pi in zip(gt_idx.tolist(), pr_idx.tolist()):
                    dist_vals.append(float(shared.dist_mat[gi, pi].item()))

            # Velocity diagnostics across union
            v_abs = v.norm(dim=-1)[0].cpu().numpy()  # [T]
            mask_stay = union.stay_mask[0].cpu().numpy().astype(bool)
            all_abs_v.extend(v_abs.tolist())
            all_labels.extend(mask_stay.astype(int).tolist())
            stay_velocities.extend(v_abs[mask_stay].tolist())
            travel_velocities.extend(v_abs[~mask_stay].tolist())

        # Per-person trajectory plot (dense grid prediction vs GT snaps)
        t_dense = torch.linspace(0.0, 24.0, config.dense_resolution, device=device)
        with torch.no_grad():
            _, logits_d, v_d = model(times_union=t_dense, home_idx=home_idx, work_idx=work_idx, person_traits_raw=traits)
            pred_ids_dense = logits_d.argmax(dim=-1)[0].cpu().numpy()
            v_abs_dense = v_d.norm(dim=-1)[0].cpu().numpy()

        # Build stay intervals from union mask for shading
        tu = times_u.cpu().numpy()
        stay_mask_np = union.stay_mask[0].cpu().numpy().astype(bool)
        intervals = []
        if tu.size == stay_mask_np.size and tu.size > 0:
            start = None
            for idx in range(len(tu)):
                if stay_mask_np[idx] and start is None:
                    start = tu[idx]
                if (not stay_mask_np[idx] and start is not None) or (idx == len(tu) - 1 and start is not None):
                    end = tu[idx] if not stay_mask_np[idx] else tu[idx]
                    if end < start:
                        end = start
                    intervals.append((start, end))
                    start = None

        fig_path = Path(config.figures_dir) / f"evaluation_trajectory_{p.person_id}.png"
        fig_path.parent.mkdir(parents=True, exist_ok=True)
        plot_person_trajectory(
            times_dense=t_dense.cpu().numpy(),
            pred_ids_dense=pred_ids_dense,
            gt_times=p.times_snap.cpu().numpy(),
            gt_ids=p.loc_ids.cpu().numpy(),
            zone_names=shared.zone_names,
            out_path=str(fig_path),
            v_abs_dense=v_abs_dense,
            stay_intervals=intervals,
            thresholds={
                'epsilon_v': getattr(config, 'epsilon_v', None),
                'v_min_move': getattr(config, 'v_min_move', None),
                'v_max_move': getattr(config, 'v_max_move', None),
            }
        )

    snap_acc = (total_correct / total_snaps) if total_snaps > 0 else float("nan")
    mean_expected_dist = float(np.mean(dist_vals)) if len(dist_vals) > 0 else float("nan")

    # ROC AUC for |v| distinguishing stay(1) vs travel(0)
    scores = np.array(all_abs_v)
    labels = np.array(all_labels)
    auc = _roc_auc_binary(-scores, labels)  # Lower |v| => stay, so negate scores to rank stays higher

    # Stay compliance
    eps = config.epsilon_v
    stay_comp = float(np.mean((scores[labels == 1] <= eps))) if np.any(labels == 1) else float("nan")

    # Transition sharpness: approximate using finite diffs around transitions
    # Here we compute mean delta |v| where labels change; within +/- window we take diffs
    transition_deltas: List[float] = []
    # This is a coarse estimate aggregating all persons; a more precise per-person windowing could be added
    # For simplicity, approximate dt using median min_dt across batches isn't tracked here; assume uniform resolution near transitions.
    for i in range(1, len(labels)):
        if labels[i] != labels[i - 1]:
            transition_deltas.append(abs(scores[i] - scores[i - 1]))
    transition_sharpness = float(np.mean(transition_deltas)) if transition_deltas else float("nan")

    metrics = {
        "snap_accuracy": snap_acc,
        "mean_expected_distance_km": mean_expected_dist,
        "roc_auc_abs_v_stay_vs_travel": auc,
        "stay_compliance_fraction": stay_comp,
        "transition_sharpness_mean_delta_abs_v": transition_sharpness,
        "stay_vel_mean": float(np.mean(stay_velocities)) if stay_velocities else float("nan"),
        "stay_vel_median": float(np.median(stay_velocities)) if stay_velocities else float("nan"),
        "travel_vel_mean": float(np.mean(travel_velocities)) if travel_velocities else float("nan"),
        "travel_vel_median": float(np.median(travel_velocities)) if travel_velocities else float("nan"),
    }

    out_path = Path(config.figures_dir) / "metrics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

    # -------- Summary plots --------
    figs_dir = Path(config.figures_dir)
    figs_dir.mkdir(parents=True, exist_ok=True)

    # 1) Velocity distributions for stay vs travel
    if stay_velocities or travel_velocities:
        plt.figure(figsize=(10, 6))
        if stay_velocities:
            plt.hist(stay_velocities, bins=40, alpha=0.6, label='Stay |v|', density=True)
        if travel_velocities:
            plt.hist(travel_velocities, bins=40, alpha=0.6, label='Travel |v|', density=True)
        plt.axvline(config.epsilon_v, color='k', linestyle='--', alpha=0.8, label=f"epsilon_v={config.epsilon_v}")
        plt.xlabel('|v|')
        plt.ylabel('Density')
        plt.title('Velocity magnitude distributions: stay vs travel')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / 'velocity_distributions.png')
        plt.close()

    # 2) ROC curve for classifying stay vs travel via |v|
    if len(all_abs_v) > 1 and (np.any(labels == 1) and np.any(labels == 0)):
        # Compute ROC by thresholding |v| (lower => stay)
        scores_arr = np.array(all_abs_v)
        labels_arr = np.array(all_labels)
        thresholds = np.linspace(scores_arr.min(), scores_arr.max(), 200)
        tpr_list = []
        fpr_list = []
        for th in thresholds:
            # Predict stay if |v| <= th
            preds_stay = scores_arr <= th
            tp = np.sum((preds_stay == 1) & (labels_arr == 1))
            fn = np.sum((preds_stay == 0) & (labels_arr == 1))
            fp = np.sum((preds_stay == 1) & (labels_arr == 0))
            tn = np.sum((preds_stay == 0) & (labels_arr == 0))
            tpr = tp / (tp + fn + 1e-12)
            fpr = fp / (fp + tn + 1e-12)
            tpr_list.append(tpr)
            fpr_list.append(fpr)
        plt.figure(figsize=(7, 7))
        plt.plot(fpr_list, tpr_list, label=f'ROC (AUC={auc:.3f})')
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC curve for stay vs travel using |v|')
        plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.savefig(figs_dir / 'roc_curve.png')
        plt.close()


if __name__ == "__main__":
    evaluate()


