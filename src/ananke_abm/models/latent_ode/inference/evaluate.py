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


def plot_itinerary(ax, itinerary, person_data, title):
    """Plots a generated or ground-truth itinerary."""
    ax.set_title(title)
    ax.set_xlabel("Time (hours)")
    ax.set_ylabel("Location (Embedding Similarity)") # Simplified y-axis
    
    # Plot stays
    for seg in itinerary:
        if seg['type'] == 'stay':
            # Simplified: Use y-coordinate as a proxy for location
            y_val = np.mean(seg['location_embedding'][:2])
            ax.plot([seg['start_time']/60, seg['end_time']/60], [y_val, y_val], linewidth=4, label=f"Stay")

    # Plot GT snaps for reference
    ax.plot(person_data['gt_times'].cpu()/60, 
            [np.mean(e.cpu().numpy()[:2]) for e in person_data['gt_loc_emb']], 
            'o', color='black', markersize=8, label="GT Snaps")
    ax.legend()


def evaluate():
    """Loads a trained model and evaluates its generated itineraries."""
    config = GenerativeODEConfig()
    save_folder = Path("saved_models/mode_separated")
    save_folder.mkdir(parents=True, exist_ok=True)
    model_path = "saved_models/best_model.pth"
    
    inference_engine = InferenceEngine(model_path, config)
    processor = inference_engine.processor
    
    # Create the mappings we need for evaluation
    location_to_embedding, _ = get_location_mappings()
    purpose_to_embedding = {name: get_purpose_features(pid) for name, pid in PURPOSE_ID_MAP.items()}
    
    person_ids = [1, 2]
    generated_itineraries = inference_engine.predict_trajectories(person_ids)

    for result in generated_itineraries:
        person_id = result['person_id']
        itinerary = result['itinerary']
        person_data = processor.get_data(person_id)

        # --- Create Ground Truth Itinerary for Comparison ---
        gt_itinerary = []
        periods = pd.read_csv("data/periods.csv")
        person_periods = periods[periods["person_id"] == person_id]
        
        for _, row in person_periods.iterrows():
            gt_itinerary.append({
                "type": row['type'],
                "start_time": row['start_time'] * 60,
                "end_time": row['end_time'] * 60,
                "location_embedding": location_to_embedding.get(row['location'], torch.zeros(config.zone_embed_dim)).numpy(),
                "purpose_embedding": purpose_to_embedding.get(row['purpose'], torch.zeros(config.purpose_feature_dim)).numpy(),
                "mode_id": MODE_ID_MAP.get(row['mode'], -1)
            })

        # --- Plotting ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), sharex=True, sharey=True)
        plot_itinerary(ax1, gt_itinerary, person_data, f"Ground Truth Itinerary for Person {person_id}")
        plot_itinerary(ax2, itinerary, person_data, f"Generated Itinerary for Person {person_id}")
        
        plt.tight_layout()
        plt.savefig(save_folder / f"evaluation_itinerary_person_{person_id}.png")
        plt.close()
        
        print(f"Generated evaluation plot for person {person_id}.")

if __name__ == "__main__":
    evaluate()
