"""
Script for evaluating a trained Generative Latent ODE model.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.run.latent_ode.data import DataProcessor
from ananke_abm.models.run.latent_ode.model import GenerativeODE

def evaluate():
    """Loads a trained model and generates evaluation plots."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}")

    # --- Model Initialization ---
    init_data = processor.get_data(person_id=1)
    model = GenerativeODE(
        person_feat_dim=init_data["person_features"].shape[-1],
        num_zone_features=init_data["all_zone_features"].shape[-1],
        config=config,
    ).to(device)
    
    # --- Load Trained Model ---
    folder_path = Path("saved_models/mode_generative_ode_batched")
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    print(f"📈 Evaluating best model from '{model_path}'...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
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

    model.eval()

    person_ids = [1, 2]
    mode_names = ["Stay", "Walk", "Car", "Public_Transit"]

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
            
            # Enhanced model outputs with mode prediction
            pred_y_logits, _, pred_purpose_logits, pred_mode_logits, _, _ = model(
                person_features, home_zone_features, work_zone_features, 
                start_purpose_id, plot_times, all_zone_features, adjacency_matrix
            )
            
            pred_y = torch.argmax(pred_y_logits.squeeze(0), dim=1)
            pred_purpose = torch.argmax(pred_purpose_logits.squeeze(0), dim=1)
            pred_mode = torch.argmax(pred_mode_logits.squeeze(0), dim=1)  # NEW: Mode predictions

        # --- Enhanced Visualization with 3 subplots ---
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(15, 14), sharex=True)
        
        # Location plot
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8)
        ax1.plot(plot_times.cpu().numpy(), pred_y.cpu().numpy(), '-', label='Generated Location')
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} (Batched Training with Mode Choice)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Purpose plot
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8)
        ax2.plot(plot_times.cpu().numpy(), pred_purpose.cpu().numpy(), '-', label='Generated Purpose')
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        # NEW: Mode plot
        ax3.plot(data["times"].cpu().numpy(), data["target_mode_ids"].cpu().numpy(), 'o', label='Ground Truth Mode', markersize=8)
        ax3.plot(plot_times.cpu().numpy(), pred_mode.cpu().numpy(), '-', label='Generated Mode')
        ax3.set_xlabel("Time (hours)")
        ax3.set_ylabel("Mode ID")
        ax3.set_yticks(np.arange(len(mode_names)))
        ax3.set_yticklabels(mode_names, rotation=0, ha='right')
        ax3.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax3.legend()
        
        plt.tight_layout()
        
        save_path = folder_path / f"generative_ode_trajectory_{person_name.replace(' ', '_')}_batched_with_modes.png"
        plt.savefig(save_path)
        print(f"   📄 Plot saved to '{save_path}'")
        plt.close()
        
        # --- Mode Choice Analysis ---
        ground_truth_modes = data["target_mode_ids"].cpu().numpy()
        print(f"   📊 Mode choice analysis for {person_name}:")
        print(f"      Ground truth mode distribution: {dict(zip(mode_names, [np.sum(ground_truth_modes == i) for i in range(len(mode_names))]))}")
        
        # Calculate mode transition statistics
        mode_transitions = []
        for i in range(len(ground_truth_modes) - 1):
            if ground_truth_modes[i] != ground_truth_modes[i+1]:
                mode_transitions.append((mode_names[ground_truth_modes[i]], mode_names[ground_truth_modes[i+1]]))
        
        print(f"      Mode transitions: {len(mode_transitions)} transitions")
        if mode_transitions:
            print(f"      Most common transitions: {mode_transitions[:5]}")

    print("✅ Evaluation complete with mode choice analysis.")

if __name__ == "__main__":
    evaluate() 