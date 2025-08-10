"""
Script for evaluating a trained Generative Latent ODE model.
Enhanced with batched inference for scalable evaluation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.cm as cm

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor
from ananke_abm.models.latent_ode.inference.inference import BatchedInferenceEngine
from ananke_abm.data_generator.feature_engineering import ID_TO_MODE_MAP, ID_TO_PURPOSE_MAP

def evaluate():
    """Loads a trained model and generates evaluation plots."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}")

    # --- Load Trained Model ---
    folder_path = Path("saved_models/mode_generative_ode_batched")
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    print(f"📈 Evaluating model from '{model_path}'...")
    
    try:
        inference_engine = BatchedInferenceEngine(str(model_path), config, device=str(device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return
        
    # --- Plot Training Loss ---
    training_stats_path = folder_path / "latent_ode_training_stats_batched.npz"
    try:
        stats = np.load(training_stats_path)
        plt.figure(figsize=(16, 8))
        loss_keys = {k: k.replace('_', ' ').title() for k in stats.files}
        
        for key, label in loss_keys.items():
            plt.plot(stats[key], label=label, alpha=0.9)

        plt.title("All Training Loss Components")
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss (Log Scale)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        loss_plot_path = folder_path / "all_training_loss_curves_batched.png"
        plt.savefig(loss_plot_path)
        print(f"   📉 All training loss plots saved to '{loss_plot_path}'")
        plt.close()
            
    except FileNotFoundError:
        print(f"WARNING: Training stats file not found at {training_stats_path}. Skipping loss plot.")

    # --- Batched Inference ---
    person_ids = [1, 2]
    time_resolution = 500
    num_samples_to_plot = 3 # Generate multiple trajectories to see stochasticity
    
    print(f"🚀 Performing batched inference for {len(person_ids)} people ({num_samples_to_plot} samples each)...")
    
    predictions = inference_engine.predict_trajectories(
        person_ids=person_ids,
        time_resolution=time_resolution,
        batch_size=len(person_ids),
        num_samples=num_samples_to_plot
    )
    
    plot_times = predictions['times']
    pred_locations = predictions['locations'] # Shape: [num_people, num_samples, num_times]
    pred_purposes = predictions['purposes'] 
    pred_modes = predictions['modes']

    # --- Dynamic Labels from Feature Engineering ---
    purpose_names = [ID_TO_PURPOSE_MAP[i] for i in sorted(ID_TO_PURPOSE_MAP.keys())]
    mode_names = [ID_TO_MODE_MAP[i] for i in sorted(ID_TO_MODE_MAP.keys())]

    # --- Individual Visualizations ---
    for i, person_id in enumerate(person_ids):
        data = processor.get_data(person_id=person_id)
        person_name = data['person_name']
        print(f"   -> Generating visualization for {person_name}...")

        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
        colors = cm.viridis(np.linspace(0, 1, num_samples_to_plot))

        # --- Plot Ground Truth ---
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', color='black', label='Ground Truth Location', markersize=8)
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', color='black', label='Ground Truth Purpose', markersize=8)
        ax3.plot(data["times"].cpu().numpy(), data["target_mode_ids"].cpu().numpy(), 'o', color='black', label='Ground Truth Mode', markersize=8)

        # --- Plot Generated Samples ---
        for s_idx in range(num_samples_to_plot):
            label = f'Generated Sample {s_idx+1}'
            ax1.plot(plot_times, pred_locations[i, s_idx, :], '-', color=colors[s_idx], label=label, alpha=0.8)
            ax2.plot(plot_times, pred_purposes[i, s_idx, :], '-', color=colors[s_idx], label=label, alpha=0.8)
            ax3.plot(plot_times, pred_modes[i, s_idx, :], '-', color=colors[s_idx], label=label, alpha=0.8)

        # Formatting
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name}")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(purpose_names)))
        ax2.set_yticklabels(purpose_names, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Mode ID")
        ax3.set_yticks(np.arange(len(mode_names)))
        ax3.set_yticklabels(mode_names, rotation=0, ha='right')
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
        
        # Create a single legend for the figure
        handles, labels = ax1.get_legend_handles_labels()
        unique_labels = dict(zip(labels, handles))
        fig.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

        plt.tight_layout(rect=[0, 0, 0.9, 1]) # Adjust layout to make space for legend
        
        save_path = folder_path / f"evaluation_trajectory_{person_name.replace(' ', '_')}.png"
        plt.savefig(save_path)
        print(f"   📄 Plot saved to '{save_path}'")
        plt.close()

if __name__ == "__main__":
    evaluate()
