"""
Script for evaluating a trained Generative Latent ODE model.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

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
        num_zones=init_data["num_zones"],
        config=config,
    ).to(device)
    
    # --- Load Trained Model ---
    model_path = "latent_ode_best_model_composite_loss_anchor.pth"
    print(f"📈 Evaluating best model from '{model_path}'...")
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
    except FileNotFoundError:
        print(f"ERROR: Model file not found at {model_path}. Please run train.py first.")
        return
        
    model.eval()

    person_ids = [1, 2]  # Sarah and Marcus

    for person_id in person_ids:
        with torch.no_grad():
            data = processor.get_data(person_id=person_id)
            person_name = data['person_name']
            print(f"   -> Generating trajectory for {person_name}...")

            person_features = data["person_features"].unsqueeze(0)
            home_zone_id = torch.tensor([data["home_zone_id"]], device=device)
            work_zone_id = torch.tensor([data["work_zone_id"]], device=device)
            purpose_summary_features = data["purpose_summary_features"].unsqueeze(0)

            plot_times = torch.linspace(0, 24, 100).to(device)
            
            # Model returns purpose logits now as well
            pred_y_logits, _, pred_purpose_logits, _, _ = model(
                person_features, home_zone_id, work_zone_id, purpose_summary_features, plot_times
            )
            
            pred_y = torch.argmax(pred_y_logits.squeeze(0), dim=1)
            pred_purpose = torch.argmax(pred_purpose_logits.squeeze(0), dim=1)

        # --- Visualization ---
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 10), sharex=True)
        
        # Plot Location Trajectory
        ax1.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Location', markersize=8)
        ax1.plot(plot_times.cpu().numpy(), pred_y.cpu().numpy(), '-', label='Generated Location')
        ax1.set_ylabel("Zone ID")
        ax1.set_title(f"Generated vs. Ground Truth for {person_name} (Composite Loss w/ Anchor)")
        ax1.set_yticks(np.arange(data["num_zones"]))
        ax1.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax1.legend()
        
        # Plot Purpose Trajectory
        ax2.plot(data["times"].cpu().numpy(), data["target_purpose_ids"].cpu().numpy(), 'o', label='Ground Truth Purpose', markersize=8)
        ax2.plot(plot_times.cpu().numpy(), pred_purpose.cpu().numpy(), '-', label='Generated Purpose')
        ax2.set_xlabel("Time (hours)")
        ax2.set_ylabel("Purpose ID")
        ax2.set_yticks(np.arange(len(config.purpose_groups)))
        ax2.set_yticklabels(config.purpose_groups, rotation=30, ha='right')
        ax2.grid(True, which='both', linestyle='--', linewidth=0.5)
        ax2.legend()
        
        plt.tight_layout()
        
        save_path = f"generative_ode_trajectory_{person_name.replace(' ', '_')}_composite_loss_anchor.png"
        plt.savefig(save_path)
        print(f"   📄 Plot saved to '{save_path}'")
        plt.close()

    print("✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate() 