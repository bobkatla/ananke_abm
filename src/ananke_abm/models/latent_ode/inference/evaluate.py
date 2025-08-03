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
    
    print(f"🚀 Performing batched inference for {len(person_ids)} people...")
    
    # Use batched inference for all people at once
    predictions = inference_engine.predict_trajectories(
        person_ids=person_ids,
        time_resolution=time_resolution,
        batch_size=len(person_ids)  # Process all people in one batch
    )
    
    # Extract results
    plot_times = predictions['times']
    pred_locations = predictions['locations']
    pred_purposes = predictions['purposes'] 
    pred_modes = predictions['modes']
    person_names = predictions['person_names']

    # --- Individual Visualizations (same as before) ---
    for i, person_id in enumerate(person_ids):
        # Get ground truth data for comparison
        data = processor.get_data(person_id=person_id)
        person_name = data['person_name']
        
        print(f"   -> Generating visualization for {person_name}...")

        # Predict at exact ground truth time points for this person
        print(f"   -> Predicting at exact event times for {person_name}...")
        exact_time_predictions = inference_engine.predict_trajectories(
            person_ids=[person_id],
            times_to_predict=data["times"],
            batch_size=1
        )
        pred_locations_exact = exact_time_predictions['locations'][0]
        pred_purposes_exact = exact_time_predictions['purposes'][0]
        pred_modes_exact = exact_time_predictions['modes'][0]

        # --- Enhanced Visualization with 3 subplots ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
        
        # Location plot
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8, zorder=3)
        ax1.plot(plot_times, pred_locations[i], '-', label='Generated Location (Interpolated)', zorder=2)
        ax1.plot(data["times"].cpu().numpy(), pred_locations_exact, '*', color='red', label='Generated @ Truth Times', markersize=12, alpha=0.8, linestyle='none', zorder=4)
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} (Batched Inference with Mode Choice)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Purpose plot
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8, zorder=3)
        ax2.plot(plot_times, pred_purposes[i], '-', label='Generated Purpose (Interpolated)', zorder=2)
        ax2.plot(data["times"].cpu().numpy(), pred_purposes_exact, '*', color='red', label='Generated @ Truth Times', markersize=12, alpha=0.8, linestyle='none', zorder=4)
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        # Mode plot
        ax3.plot(data["times"].cpu().numpy(), data["target_mode_ids"].cpu().numpy(), 'o', label='Ground Truth Mode', markersize=8, zorder=3)
        ax3.plot(plot_times, pred_modes[i], '-', label='Generated Mode (Interpolated)', zorder=2)
        ax3.plot(data["times"].cpu().numpy(), pred_modes_exact, '*', color='red', label='Generated @ Truth Times', markersize=12, alpha=0.8, linestyle='none', zorder=4)
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
        
        # --- Mode Choice Analysis ---
        ground_truth_modes = data["target_mode_ids"].cpu().numpy()
        print(f"   📊 Mode choice analysis for {person_name}:")
        print(f"      Ground truth mode distribution: {dict(zip(mode_names, [np.sum(ground_truth_modes == j) for j in range(len(mode_names))]))}")
        
        # Calculate mode transition statistics
        mode_transitions = []
        for j in range(len(ground_truth_modes) - 1):
            if ground_truth_modes[j] != ground_truth_modes[j+1]:
                mode_transitions.append((mode_names[ground_truth_modes[j]], mode_names[ground_truth_modes[j+1]]))
        
        print(f"      Mode transitions: {len(mode_transitions)} transitions")
        if mode_transitions:
            print(f"      Most common transitions: {mode_transitions[:5]}")

    print("✅ Evaluation complete with batched inference and mode choice analysis.")

if __name__ == "__main__":
    # Run standard evaluation
    evaluate()
    
    # Run benchmark
    print("\n" + "="*80)