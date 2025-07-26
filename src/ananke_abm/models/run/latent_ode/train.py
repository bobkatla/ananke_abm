"""
Main script for training the Generative Latent ODE model.
"""
import torch
import numpy as np
import torch.nn.functional as F
from pathlib import Path

from ananke_abm.models.run.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.run.latent_ode.data import DataProcessor
from ananke_abm.models.run.latent_ode.model import GenerativeODE
from ananke_abm.models.run.latent_ode.loss import calculate_composite_loss

def train():
    """Orchestrates the training of the Generative ODE model, one person at a time."""
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
    
    # --- Training Loop (Sequential) ---
    print("🚀 Starting training (sequential, composite loss)...")
    best_loss = float('inf')
    folder_path = Path("saved_models/generative_ode")
    model_path = folder_path / "latent_ode_best_model_composite_loss_anchor.pth"
    training_stats_path = folder_path / "latent_ode_training_stats_composite_loss_anchor.npz"
    person_ids = [1, 2]

    iteration_losses = []
    
    for i in range(config.num_iterations):
        total_iter_loss = 0
        model.train()
        
        for person_id in person_ids:
            optimizer.zero_grad()
            
            data = processor.get_data(person_id=person_id)
            
            person_features = data["person_features"].unsqueeze(0)
            times = data["times"]
            home_zone_id = torch.tensor([data["home_zone_id"]], device=device)
            work_zone_id = torch.tensor([data["work_zone_id"]], device=device)
            purpose_summary_features = data["purpose_summary_features"].unsqueeze(0)

            # Forward pass now returns purpose predictions as well
            pred_y_logits, pred_y_embeds, pred_purpose_logits, mu, log_var = model(
                person_features, home_zone_id, work_zone_id, purpose_summary_features, times
            )
            
            # Calculate the new composite loss
            loss, loss_c, loss_e, loss_d, loss_p, loss_kl = calculate_composite_loss(
                pred_y_logits,
                pred_y_embeds,
                pred_purpose_logits,
                data["trajectory_y"],
                data["target_purpose_ids"],
                model,
                mu,
                log_var,
                processor.distance_matrix,
                config
            )

            loss.backward()
            optimizer.step()
            total_iter_loss += loss.item()
            
        avg_loss = total_iter_loss / len(person_ids)
        iteration_losses.append(avg_loss)

        if (i + 1) % 500 == 0:
            print(f"Iter {i+1}, Avg Loss: {avg_loss:.4f} | "
                  f"Classif: {loss_c.item():.4f}, "
                  f"Embed: {loss_e.item():.4f}, "
                  f"Dist: {loss_d.item():.4f}, "
                  f"Purp: {loss_p.item():.4f}, "
                  f"KL: {loss_kl.item():.4f}")

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), model_path)
            
    print("✅ Training complete.")

    np.savez(training_stats_path, iteration_losses=np.array(iteration_losses))
    print(f"   💾 Training stats saved to '{training_stats_path}'")

if __name__ == "__main__":
    train() 