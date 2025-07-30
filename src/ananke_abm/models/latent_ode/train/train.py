"""
Main script for training the Generative Latent ODE model.
"""
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor, LatentODEDataset
from ananke_abm.models.latent_ode.architecture.model import GenerativeODE
from ananke_abm.models.latent_ode.architecture.loss import calculate_composite_loss
from ananke_abm.models.latent_ode.data_process.batching import unify_and_interpolate_batch

def train():
    """Orchestrates the training of the Generative ODE model using batched data."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = DataProcessor(device, config)
    print(f"🔬 Using device: {device}", flush=True)
    print(f"🧠 Attention enabled: {config.enable_attention} (strength: {config.attention_strength})", flush=True)
    print(f"🚀 Mode choice enabled: {config.num_modes} modes with weight {config.loss_weight_mode}", flush=True)
    print(f"🎲 SDE enabled: {config.enable_sde} (noise strength: {config.sde_noise_strength})", flush=True)

    # --- Setup DataLoader ---
    person_ids = [1, 2] # Sarah and Marcus
    dataset = LatentODEDataset(person_ids, processor)
    data_loader = DataLoader(dataset, batch_size=len(person_ids), shuffle=True, collate_fn=unify_and_interpolate_batch)

    # --- Model Initialization ---
    # Use a sample from the dataset to get dimensions for model init
    init_batch = next(iter(data_loader))
    model = GenerativeODE(
        person_feat_dim=init_batch["person_features"].shape[-1],
        num_zone_features=init_batch["all_zone_features"].shape[-1],
        config=config,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # --- Training Loop (Batched) ---
    print("🚀 Starting training with batched data and mode choice...", flush=True)
    folder_path = Path("saved_models/mode_generative_ode_batched")
    folder_path.mkdir(exist_ok=True, parents=True)
    model_path = folder_path / "latent_ode_best_model_batched.pth"
    training_stats_path = folder_path / "latent_ode_training_stats_batched.npz"

    best_loss = float('inf')
    all_losses = []

    for i in range(config.num_iterations):
        for batch in data_loader:
            model.train()
            optimizer.zero_grad()
            
            # Forward pass using the unified timeline
            model_outputs = model(
                batch['person_features'], 
                batch['home_zone_features'], 
                batch['work_zone_features'],
                batch['start_purpose_id'],
                batch['t_unified'],
                batch['all_zone_features'],
                batch['adjacency_matrix']
            )
            
            # Calculate the composite loss on the batch (now includes mode loss)
            loss, loss_c, loss_e, loss_d, loss_p, loss_mode, loss_kl = calculate_composite_loss(
                batch, model_outputs, model, processor.distance_matrix, config
            )

            loss.backward()
            optimizer.step()
        
        # Store all loss components for the iteration
        all_losses.append([loss.item(), loss_c.item(), loss_e.item(), loss_d.item(), loss_p.item(), loss_mode.item(), loss_kl.item()])

        if (i + 1) % 500 == 0:
            print(f"Iter {i+1}, Loss: {loss.item():.4f} | Classif: {loss_c.item():.4f}, Embed: {loss_e.item():.4f}, Dist: {loss_d.item():.4f}, Purp: {loss_p.item():.4f}, Mode: {loss_mode.item():.4f}, KL: {loss_kl.item():.4f}", flush=True)

        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), model_path)
            print(f"💾 New best model saved at iteration {i+1} with loss {best_loss:.4f}", flush=True)
            
    print("✅ Training complete.", flush=True)

    # --- Save Training Statistics ---
    all_losses = np.array(all_losses)
    np.savez(
        training_stats_path, 
        total_loss=all_losses[:, 0],
        classification_loss=all_losses[:, 1],
        embedding_loss=all_losses[:, 2],
        distance_loss=all_losses[:, 3],
        purpose_loss=all_losses[:, 4],
        mode_loss=all_losses[:, 5],  # NEW: Mode loss statistics
        kl_loss=all_losses[:, 6]
    )
    print(f"   💾 Training stats saved to '{training_stats_path}'", flush=True)

if __name__ == "__main__":
    train() 