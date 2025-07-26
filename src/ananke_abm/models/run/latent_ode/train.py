"""
Main script for training the Generative Latent ODE model.
"""
import torch
import numpy as np

from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.run.latent_ode.data import DataProcessor
from ananke_abm.models.run.latent_ode.model import GenerativeODE
from ananke_abm.models.run.latent_ode.loss import calculate_loss

def train():
    """Orchestrates the training of the Generative ODE model."""
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
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # --- Training Loop for Multiple People ---
    print("🚀 Starting training for all people...")
    best_loss = float('inf')
    model_path = "latent_ode_best_model.pth"
    training_stats_path = "latent_ode_training_stats.npz"
    person_ids = [1, 2]  # Sarah and Marcus
    
    # Lists to store training statistics
    iteration_losses = []
    
    for i in range(config.num_iterations):
        
        total_iter_loss = 0
        # In each iteration, train on each person's data
        for person_id in person_ids:
            optimizer.zero_grad()
            
            data = processor.get_data(person_id=person_id)
            
            person_features = data["person_features"].unsqueeze(0)
            trajectory_y = data["trajectory_y"]
            times = data["times"]
            home_zone_id = torch.tensor([data["home_zone_id"]], device=device)
            work_zone_id = torch.tensor([data["work_zone_id"]], device=device)
            purpose_features = data["purpose_features"].unsqueeze(0)

            pred_y_logits, mu, log_var = model(person_features, home_zone_id, work_zone_id, purpose_features, times)
            pred_y_logits = pred_y_logits.squeeze(0)

            loss, recon_loss, kl_loss = calculate_loss(
                pred_y_logits, trajectory_y, mu, log_var, config.kl_weight
            )

            loss.backward()
            optimizer.step()
            total_iter_loss += loss.item()

        avg_loss = total_iter_loss / len(person_ids)
        iteration_losses.append(avg_loss)
        
        if (i + 1) % 500 == 0:
            print(f"   Iter {i+1}, Avg Loss: {avg_loss:.4f}")

        # Save the best model based on average loss for the iteration
        if total_iter_loss < best_loss:
            best_loss = total_iter_loss
            torch.save(model.state_dict(), model_path)
            
    print("✅ Training complete.")

    # --- Save Training Statistics ---
    np.savez(
        training_stats_path,
        iteration_losses=np.array(iteration_losses),
    )
    print(f"   💾 Training stats saved to '{training_stats_path}'")

if __name__ == "__main__":
    train() 