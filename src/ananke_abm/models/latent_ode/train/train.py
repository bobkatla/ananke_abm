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
    print(f"🎲 SDE enabled: {config.enable_sde} (noise strength: {config.sde_noise_strength})", flush=True)

    # --- Setup DataLoader ---
    person_ids = [1, 2] # Sarah and Marcus
    dataset = LatentODEDataset(person_ids, processor)
    data_loader = DataLoader(dataset, batch_size=len(person_ids), shuffle=True, collate_fn=unify_and_interpolate_batch)

    # --- Model Initialization ---
    init_batch = next(iter(data_loader))
    model = GenerativeODE(
        person_feat_dim=init_batch["person_features"].shape[-1],
        num_zone_features=init_batch["all_zone_features"].shape[-1],
        config=config,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # --- Training Loop (Batched) ---
    print("🚀 Starting training with hybrid MSE loss...", flush=True)
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
            
            # We need the purpose/mode features at the first time step for the encoder
            initial_purpose_features = batch['y_purp_feat_dense'][:, 0, :]
            initial_mode_features = batch['y_mode_feat_dense'][:, 0, :]
            
            model_outputs = model(
                batch['person_features'], 
                batch['home_zone_features'], 
                batch['work_zone_features'],
                initial_purpose_features,
                initial_mode_features,
                batch['t_unified'],
                batch['all_zone_features']
            )
            
            # Calculate the composite loss, now with MSE components
            losses = calculate_composite_loss(
                batch, model_outputs, model, processor.distance_matrix, config
            )
            total_loss = losses[0]

            total_loss.backward()
            optimizer.step()
        
        # Store all loss components for the iteration
        all_losses.append([l.item() for l in losses])

        if (i + 1) % 500 == 0:
            (
                _, loss_c, loss_e, loss_d, 
                loss_pc, loss_pm, loss_mc, loss_mm, loss_kl
            ) = [l.item() for l in losses]
            print(f"Iter {i+1}, Loss: {total_loss.item():.4f} | "
                  f"Loc (C/E/D): {loss_c:.2f}/{loss_e:.2f}/{loss_d:.2f} | "
                  f"Purp (C/MSE): {loss_pc:.2f}/{loss_pm:.2f} | "
                  f"Mode (C/MSE): {loss_mc:.2f}/{loss_mm:.2f} | "
                  f"KL: {loss_kl:.2f}", flush=True)


        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
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
        purpose_class_loss=all_losses[:, 4],
        purpose_mse_loss=all_losses[:, 5],
        mode_class_loss=all_losses[:, 6],
        mode_mse_loss=all_losses[:, 7],
        kl_loss=all_losses[:, 8]
    )
    print(f"   💾 Training stats saved to '{training_stats_path}'", flush=True)

if __name__ == "__main__":
    train()
