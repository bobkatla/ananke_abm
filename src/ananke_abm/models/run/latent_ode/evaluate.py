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
    model_path = "latent_ode_best_model.pth"
    print(f"📈 Evaluating best model from '{model_path}'...")
    model.load_state_dict(torch.load(model_path))
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
            purpose_features = data["purpose_features"].unsqueeze(0)

            plot_times = torch.linspace(0, 24, 100).to(device)
            pred_y_logits, _, _ = model(person_features, home_zone_id, work_zone_id, purpose_features, plot_times)
            pred_y_logits = pred_y_logits.squeeze(0)
            pred_y = torch.argmax(pred_y_logits, dim=1)

        # --- Visualization ---
        plt.figure(figsize=(15, 6))
        plt.plot(data["times"].cpu().numpy(), data["trajectory_y"].cpu().numpy(), 'o', label='Ground Truth Snaps', markersize=8)
        plt.plot(plot_times.cpu().numpy(), pred_y.cpu().numpy(), '-', label='Generated Trajectory')
        
        plt.xlabel("Time (hours)")
        plt.ylabel("Zone ID")
        plt.title(f"Generated vs. Ground Truth Trajectory for {person_name}")
        plt.yticks(np.arange(data["num_zones"]))
        plt.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.legend()
        plt.tight_layout()
        
        save_path = f"generative_ode_trajectory_{person_name.replace(' ', '_')}.png"
        plt.savefig(save_path)
        print(f"   📄 Plot saved to '{save_path}'")
        plt.close()

    print("✅ Evaluation complete.")

if __name__ == "__main__":
    evaluate() 