"""
Script for evaluating the SDE model with segment-based mode prediction.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.inference.inference import InferenceEngine, get_location_mappings
from ananke_abm.data_generator.feature_engineering import MODE_ID_MAP, get_purpose_features, PURPOSE_ID_MAP
from matplotlib.patches import Patch


def _get_zone_names_and_embeds(location_to_embedding: dict):
    zone_names = list(location_to_embedding.keys())
    zone_feats = torch.stack([location_to_embedding[name] for name in zone_names])
    return zone_names, zone_feats


def _encode_zone_features(model, zone_feats: torch.Tensor, device: torch.device):
    with torch.no_grad():
        return model.zone_feature_encoder(zone_feats.to(device))  # (Z, D)


def _nearest_location_name(model, location_to_embedding, loc_embed_np, device):
    zone_names, zone_feats = _get_zone_names_and_embeds(location_to_embedding)
    zone_embeds = _encode_zone_features(model, zone_feats, device)  # (Z, D)
    loc_embed = torch.tensor(loc_embed_np, dtype=torch.float32, device=device)
    loc_embed = loc_embed / (loc_embed.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8))
    zone_norm = zone_embeds / (zone_embeds.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8))
    sims = torch.matmul(zone_norm, loc_embed)
    idx = int(torch.argmax(sims).item())
    return zone_names[idx], idx


def _nearest_purpose_name(purpose_embed_np):
    purpose_names = list(PURPOSE_ID_MAP.keys())
    purpose_feats = torch.stack([get_purpose_features(name) for name in purpose_names])
    pe = torch.tensor(purpose_embed_np, dtype=torch.float32)
    pe = pe / (pe.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8))
    pf = purpose_feats / (purpose_feats.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8))
    sims = torch.matmul(pf, pe)
    idx = int(torch.argmax(sims).item())
    return purpose_names[idx]


def _build_gt_timeline(person_id: int, periods_df: pd.DataFrame, snaps_df: pd.DataFrame):
    """Build GT timeline using periods for timing and snaps.csv for grouped purpose labels."""
    rows = periods_df[periods_df["person_id"] == person_id].copy().sort_values(["start_time"])  # hours
    timeline = []
    # Pre-index snaps for quick lookup by (person_id, timestamp)
    snaps_person = snaps_df[snaps_df["person_id"] == person_id].set_index("timestamp")
    for _, r in rows.iterrows():
        start = float(r["start_time"]) ; end = float(r["end_time"]) ; loc = r["location"]
        if r["type"] == "stay":
            # Use grouped purpose from snaps.csv at the stay start timestamp
            purpose = snaps_person.loc[start]["purpose"] if start in snaps_person.index else r["purpose"]
            timeline.append({
                "type": "stay",
                "start": start,
                "end": end,
                "location": loc,
                "purpose": str(purpose),
            })
        else:
            timeline.append({
                "type": "travel",
                "start": start,
                "end": end,
                "location": loc,
                "purpose": "travel",
                "mode": r.get("mode", "unknown"),
            })
    return timeline


def _build_pred_timeline(itinerary: list, model, location_to_embedding, device):
    # First pass: map stays to labels
    labeled = []
    for seg in itinerary:
        if seg["type"] == "stay":
            loc_name, _ = _nearest_location_name(model, location_to_embedding, seg["location_embedding"], device)
            purpose_name = _nearest_purpose_name(seg["purpose_embedding"])
            labeled.append({
                "type": "stay",
                "start": float(seg["start_time"]),
                "end": float(seg["end_time"]),
                "location": loc_name,
                "purpose": purpose_name,
            })
        else:
            labeled.append({
                "type": "travel",
                "start": float(seg["start_time"]),
                "end": float(seg["end_time"]),
                "mode": list(MODE_ID_MAP.keys())[seg.get("mode_id", -1)] if seg.get("mode_id", -1) >= 0 else "unknown",
            })
    # Second pass: ensure we know start/end locations for travel to position mode label
    # We derive start/end location IDs from neighboring stays when available
    return labeled


def _plot_person(ax_top, ax_bot, gt_timeline, pred_timeline, zone_names_order, title_suffix=""):
    zone_to_id = {name: i for i, name in enumerate(zone_names_order)}

    # Build purpose color map from union of GT and Pred purposes
    gt_purposes = {seg["purpose"] for seg in gt_timeline if seg["type"] == "stay"}
    pred_purposes = {seg["purpose"] for seg in pred_timeline if seg["type"] == "stay"}
    purposes = sorted(gt_purposes.union(pred_purposes))
    if not purposes:
        purposes = ["unknown"]
    purpose_colors = {p: plt.cm.tab10(i % 10) for i, p in enumerate(purposes)}

    # Helper to draw timeline on a given axis
    def draw(ax, timeline, label_modes=True):
        for seg in timeline:
            if seg["type"] == "stay":
                y = zone_to_id.get(seg["location"], -1)
                if y >= 0:
                    ax.fill_betweenx([y - 0.35, y + 0.35], seg["start"], seg["end"],
                                     color=purpose_colors.get(seg["purpose"], "#cccccc"), alpha=0.85)
                    ax.plot([seg["start"], seg["end"]], [y, y], color="#333333", linewidth=1)
            else:
                if not label_modes:
                    continue
                mid_t = (seg["start"] + seg["end"]) / 2.0
                ax.text(mid_t, -0.8, seg.get("mode", ""),
                        ha="center", va="center", fontsize=10,
                        bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"))

        ax.set_yticks(list(range(len(zone_names_order))))
        ax.set_yticklabels(zone_names_order)
        ax.set_xlim(0, 24)
        ax.set_xlabel("Time (hours)")
        ax.grid(True, axis="y", alpha=0.2)

    draw(ax_top, gt_timeline)
    ax_top.set_title(f"Ground Truth {title_suffix}")
    draw(ax_bot, pred_timeline)
    ax_bot.set_title(f"Generated {title_suffix}")

    # Purpose legend (shared, shown on top axis)
    legend_patches = [Patch(facecolor=purpose_colors[p], edgecolor='none', label=p) for p in purposes]
    ax_top.legend(handles=legend_patches, title='Stay Purposes', loc='upper left', bbox_to_anchor=(1.02, 1.0))


def evaluate():
    """Loads a trained model and evaluates its generated itineraries."""
    config = GenerativeODEConfig()
    save_folder = Path("saved_models/mode_separated")
    save_folder.mkdir(parents=True, exist_ok=True)
    model_path = "saved_models/best_model.pth"
    
    inference_engine = InferenceEngine(model_path, config)
    processor = inference_engine.processor
    device = inference_engine.device
    
    # Create the mappings we need for evaluation
    location_to_embedding, _ = get_location_mappings()
    zone_names_order = list(location_to_embedding.keys())
    
    person_ids = [1, 2]
    generated_itineraries = inference_engine.predict_trajectories(person_ids)

    periods = pd.read_csv("data/periods.csv")
    snaps = pd.read_csv("data/snaps.csv")

    for result in generated_itineraries:
        person_id = result['person_id']
        itinerary = result['itinerary']
        
        # GT timeline in real labels, using grouped purposes from snaps.csv
        gt_timeline = _build_gt_timeline(person_id, periods, snaps)
        
        # Predicted timeline (map embeddings to nearest labels)
        pred_timeline = _build_pred_timeline(itinerary, inference_engine.model, location_to_embedding, device)

        # --- Plotting: two stacked panels per person ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
        _plot_person(ax1, ax2, gt_timeline, pred_timeline, zone_names_order, title_suffix=f"Itinerary for Person {person_id}")
        plt.tight_layout()
        plt.savefig(save_folder / f"evaluation_itinerary_person_{person_id}.png")
        plt.close()
        print(f"Generated evaluation plot for person {person_id}.")


if __name__ == "__main__":
    evaluate()
