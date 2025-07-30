"""
Script for evaluating a trained Generative Latent ODE model (OLD VERSION - before mode choice).
This script evaluates models trained with the old architecture that only predicts location and purpose.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.run.latent_ode.data import DataProcessor

# Import the old model architecture (we'll need to use the current one but handle outputs differently)
from ananke_abm.models.run.latent_ode.model import GenerativeODE

def evaluate_old():
    """Loads a trained model (old architecture) and generates evaluation plots."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}")
    print(f"📊 Evaluating OLD model (before mode choice integration)")

    # --- Model Initialization ---
    init_data = processor.get_data(person_id=1)
    model = GenerativeODE(
        person_feat_dim=init_data["person_features"].shape[-1],
        num_zone_features=init_data["all_zone_features"].shape[-1],
        config=config,
    ).to(device)
    
    # --- Load Trained Model ---
    folder_path = Path("saved_models/generative_ode_batched")
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    print(f"📈 Evaluating old model from '{model_path}'...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device), strict=False)
        print("⚠️  Loaded with strict=False (some parameters may be missing/extra)")
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return
        
    # --- Plot Training Loss (Old Format) ---
    training_stats_path = folder_path / "latent_ode_training_stats_batched.npz"
    try:
        stats = np.load(training_stats_path)
        
        plt.figure(figsize=(14, 8))
        
        # Old loss components (before mode loss)
        old_loss_keys = {
            'total_loss': 'Total Loss',
            'classification_loss': 'Location Classification',
            'embedding_loss': 'Location Embedding',
            'distance_loss': 'Physical Distance',
            'purpose_loss': 'Purpose Classification',
            'kl_loss': 'KL Divergence'
        }
        
        for key, label in old_loss_keys.items():
            if key in stats:
                plt.plot(stats[key], label=label, alpha=0.9)

        plt.title("Training Loss Components (OLD MODEL - Before Mode Choice)")
        plt.xlabel("Iteration")
        plt.ylabel("Average Loss (Log Scale)")
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.yscale('log')
        plt.tight_layout()
        loss_plot_path = folder_path / "old_training_loss_curves_batched.png"
        plt.savefig(loss_plot_path)
        print(f"   📉 Old training loss plots saved to '{loss_plot_path}'")
        plt.close()
            
    except FileNotFoundError:
        print(f"WARNING: Training stats file not found at {training_stats_path}. Skipping loss plot.")

    model.eval()

    person_ids = [1, 2]

    for person_id in person_ids:
        with torch.no_grad():
            data = processor.get_data(person_id=person_id)
            person_name = data['person_name']
            print(f"   -> Generating trajectory for {person_name}...")

            person_features = data["person_features"].unsqueeze(0)
            home_zone_features = data["home_zone_features"].unsqueeze(0)
            work_zone_features = data["work_zone_features"].unsqueeze(0)
            start_purpose_id = torch.tensor([data["start_purpose_id"]], device=device)
            all_zone_features = data["all_zone_features"] # No batch dim needed
            adjacency_matrix = data["adjacency_matrix"] # No batch dim needed

            plot_times = torch.linspace(0, 24, 100).to(device)
            
            # Old model outputs (before mode choice) - only extract first 5 outputs
            try:
                model_outputs = model(
                    person_features, home_zone_features, work_zone_features, 
                    start_purpose_id, plot_times, all_zone_features, adjacency_matrix
                )
                # Handle both old and new model outputs
                if len(model_outputs) == 6:  # New model with mode outputs
                    pred_y_logits, _, pred_purpose_logits, pred_mode_logits, _, _ = model_outputs
                    print("   ⚠️  Detected new model format, ignoring mode outputs")
                else:  # Old model format
                    pred_y_logits, _, pred_purpose_logits, _, _ = model_outputs
                    
            except Exception as e:
                print(f"   ❌ Error during model forward pass: {e}")
                continue
            
            pred_y = torch.argmax(pred_y_logits.squeeze(0), dim=1)
            pred_purpose = torch.argmax(pred_purpose_logits.squeeze(0), dim=1)

        # --- Original 2-Panel Visualization (before mode choice) ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Location plot
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8)
        ax1.plot(plot_times.cpu().numpy(), pred_y.cpu().numpy(), '-', label='Generated Location')
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} (OLD MODEL - Before Mode Choice)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Purpose plot  
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8)
        ax2.plot(plot_times.cpu().numpy(), pred_purpose.cpu().numpy(), '-', label='Generated Purpose')
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = folder_path / f"old_generative_ode_trajectory_{person_name.replace(' ', '_')}_batched.png"
        plt.savefig(save_path)
        print(f"   📄 Old model plot saved to '{save_path}'")
        plt.close()
        
        # --- Original Statistics (no mode analysis) ---
        ground_truth_purposes = data["target_purpose_ids"].cpu().numpy()
        print(f"   📊 Purpose analysis for {person_name}:")
        purpose_names = config.purpose_groups
        print(f"      Ground truth purpose distribution: {dict(zip(purpose_names, [np.sum(ground_truth_purposes == i) for i in range(len(purpose_names))]))}")
        
        # Calculate purpose transition statistics
        purpose_transitions = []
        for i in range(len(ground_truth_purposes) - 1):
            if ground_truth_purposes[i] != ground_truth_purposes[i+1]:
                purpose_transitions.append((purpose_names[ground_truth_purposes[i]], purpose_names[ground_truth_purposes[i+1]]))
        
        print(f"      Purpose transitions: {len(purpose_transitions)} transitions")
        if purpose_transitions:
            print(f"      Most common transitions: {purpose_transitions[:3]}")

    print("✅ Old model evaluation complete.")

if __name__ == "__main__":
    evaluate_old() 