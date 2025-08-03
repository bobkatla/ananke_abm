"""
Script for evaluating a trained Generative Latent ODE model.
Enhanced with batched inference for scalable evaluation.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor
from ananke_abm.models.latent_ode.inference.inference import BatchedInferenceEngine

def score_trajectories(purposes: np.ndarray, modes: np.ndarray, config: GenerativeODEConfig) -> np.ndarray:
    """
    Scores trajectories based on a penalty for logical contradictions.

    Args:
        purposes: Predicted purposes of shape [num_samples, num_times].
        modes: Predicted modes of shape [num_samples, num_times].
        config: The model configuration.

    Returns:
        An array of scores (lower is better) for each sample trajectory.
    """
    scores = np.zeros(purposes.shape[0])
    travel_purpose_idx = len(config.purpose_groups) - 1
    stay_mode_idx = 0

    for i in range(purposes.shape[0]):  # Iterate over each sample
        # Find where the contradiction "Travel" purpose AND "Stay" mode occurs
        contradiction = (purposes[i] == travel_purpose_idx) & (modes[i] == stay_mode_idx)
        # The score is the number of time steps with this contradiction
        scores[i] = np.sum(contradiction)
    return scores


def evaluate():
    """Loads a trained model and generates evaluation plots using batched inference."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}")

    # --- Load Trained Model ---
    folder_path = Path("saved_models/mode_generative_ode_batched")
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    print(f"📈 Evaluating model from '{model_path}'...")
    
    # Initialize batched inference engine
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
        
        loss_keys = {
            'total_loss': 'Total Loss',
            'classification_loss': 'Location Classification',
            'embedding_loss': 'Location Embedding',
            'distance_loss': 'Physical Distance',
            'purpose_loss': 'Purpose Classification',
            'mode_loss': 'Mode Classification',  # NEW: Mode loss component
            'kl_loss': 'KL Divergence'
        }
        
        for key, label in loss_keys.items():
            if key in stats:
                plt.plot(stats[key], label=label, alpha=0.9)

        plt.title("All Training Loss Components (Unweighted, Batched Training with Mode Choice)")
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
    mode_names = ["Stay", "Walk", "Car", "Public_Transit"]
    time_resolution = 500
    num_samples = 10  # Generate 10 different trajectories for each person
    
    print(f"🚀 Performing batched inference for {len(person_ids)} people with {num_samples} samples each...")
    
    # Use batched inference for all people at once
    predictions = inference_engine.predict_trajectories(
        person_ids=person_ids,
        time_resolution=time_resolution,
        batch_size=len(person_ids),  # Process all people in one batch
        num_samples=num_samples
    )
    
    # Extract results
    plot_times = predictions['times']
    # Shape is now [num_people, num_samples, num_times]
    pred_locations_samples = predictions['locations']
    pred_purposes_samples = predictions['purposes'] 
    pred_modes_samples = predictions['modes']
    person_names = predictions['person_names']

    # --- Individual Visualizations ---
    for i, person_id in enumerate(person_ids):
        # Get ground truth data for comparison
        data = processor.get_data(person_id=person_id)
        person_name = data['person_name']
        
        print(f"   -> Generating visualization for {person_name}...")

        # Score and find the best sample for this person
        person_purposes = pred_purposes_samples[i]  # [num_samples, num_times]
        person_modes = pred_modes_samples[i]        # [num_samples, num_times]
        scores = score_trajectories(person_purposes, person_modes, config)
        best_sample_idx = np.argmin(scores)
        print(f"   -> Best sample for {person_name} is #{best_sample_idx} with a score of {scores[best_sample_idx]}")

        # --- Enhanced Visualization with 3 subplots ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
        
        # Plot all samples with low opacity
        for s_idx in range(num_samples):
            # Don't re-plot the best sample in the loop
            if s_idx == best_sample_idx:
                continue
            ax1.plot(plot_times, pred_locations_samples[i, s_idx, :], '-', color='gray', alpha=0.3, zorder=1)
            ax2.plot(plot_times, pred_purposes_samples[i, s_idx, :], '-', color='gray', alpha=0.3, zorder=1)
            ax3.plot(plot_times, pred_modes_samples[i, s_idx, :], '-', color='gray', alpha=0.3, zorder=1)

        # Plot the best sample with high opacity
        ax1.plot(plot_times, pred_locations_samples[i, best_sample_idx, :], '-', color='blue', label=f'Best Sample (Score: {scores[best_sample_idx]:.0f})', zorder=3)
        ax2.plot(plot_times, pred_purposes_samples[i, best_sample_idx, :], '-', color='blue', label=f'Best Sample (Score: {scores[best_sample_idx]:.0f})', zorder=3)
        ax3.plot(plot_times, pred_modes_samples[i, best_sample_idx, :], '-', color='blue', label=f'Best Sample (Score: {scores[best_sample_idx]:.0f})', zorder=3)

        # Plot ground truth
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8, color='red', zorder=4)
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8, color='red', zorder=4)
        ax3.plot(data["times"].cpu().numpy(), data["target_mode_ids"].cpu().numpy(), 'o', label='Ground Truth Mode', markersize=8, color='red', zorder=4)

        # Formatting
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} ({num_samples} Samples)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Mode ID")
        ax3.set_yticks(np.arange(len(mode_names)))
        ax3.set_yticklabels(mode_names, rotation=0, ha='right')
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax3.legend()
        
        plt.tight_layout()
        
        save_path = folder_path / f"batched_ode_trajectory_{person_name.replace(' ', '_')}_with_modes.png"
        plt.savefig(save_path)
        print(f"   📄 Plot saved to '{save_path}'")
        plt.close()

if __name__ == "__main__":
    # Run standard evaluation
    evaluate()
    
    # Run benchmark
    print("\n" + "="*80)