"""
Main script for training the SDE model with segment-based mode prediction.
"""
import torch
from pathlib import Path
from torch.utils.data import DataLoader

from ananke_abm.models.latent_ode.config import GenerativeODEConfig
from ananke_abm.models.latent_ode.data_process.data import DataProcessor, LatentSDEDataset
from ananke_abm.models.latent_ode.architecture.model import GenerativeODE
from ananke_abm.models.latent_ode.architecture.loss import calculate_snap_loss, calculate_segment_mode_loss
from ananke_abm.models.latent_ode.data_process.batching import sde_collate_fn
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph


def get_location_mappings():
    """Creates authoritative mappings from the mock locations."""
    _, zones_raw, _ = create_mock_zone_graph()
    location_to_embedding = {}
    location_name_to_id = {}
    for i, (zone_id, zone_data) in enumerate(sorted(zones_raw.items())):
        zone_name = zone_data["name"]
        features = [
            zone_data["population"] / 10000.0,
            zone_data["job_opportunities"] / 5000.0,
            zone_data["retail_accessibility"],
            zone_data["transit_accessibility"],
            zone_data["attractiveness"],
            zone_data["coordinates"][0] / 5.0,
            zone_data["coordinates"][1] / 5.0,
        ]
        location_to_embedding[zone_name] = torch.tensor(features, dtype=torch.float32)
        location_name_to_id[zone_name] = i
    return location_to_embedding, location_name_to_id


def train():
    """Orchestrates the training of the Generative SDE model."""
    config = GenerativeODEConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # --- Create authoritative location mappings ---
    location_to_embedding, location_name_to_id = get_location_mappings()

    # --- Setup DataProcessor and DataLoader ---
    processor = DataProcessor(device, location_to_embedding, location_name_to_id)
    dataset = LatentSDEDataset(person_ids=[1, 2], processor=processor)
    data_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=sde_collate_fn)


    # --- Model Initialization ---
    sample_batch = next(iter(data_loader))
    sample_data = processor.get_data(1)
    
    # This needs a more robust way to get these dimensions
    person_feat_dim = 8 # Hardcoded based on previous implementation
    
    model = GenerativeODE(
        person_feat_dim=person_feat_dim,
        num_zone_features=sample_data['gt_loc_emb'].shape[-1],
        config=config,
    ).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # --- Prepare for saving the best model ---
    saved_models_dir = Path("saved_models")
    saved_models_dir.mkdir(exist_ok=True)
    best_loss = float('inf')

    # --- Training Loop ---
    print("🚀 Starting training...")
    for i in range(config.num_iterations):
        for batch in data_loader:
            model.train()
            optimizer.zero_grad()
            
            # --- Forward Pass ---
            # 1. SDE forward pass to get the latent path
            model_outputs = model(
                person_features=torch.randn(2, person_feat_dim, device=device), # Placeholder
                home_zone_features=batch['loc_emb_union'][:, 0, :], # Simplification
                work_zone_features=batch['loc_emb_union'][:, 0, :], # Simplification
                initial_purpose_features=batch['purp_emb_union'][:, 0, :],
                times=batch['grid_times'],
                all_zone_features=processor.location_embeddings
            )
            pred_p, pred_v = model_outputs[4], model_outputs[5]

            # 2. Segment processing
            seg_logits, seg_h = model.predict_mode_from_segments(pred_p, pred_v, batch['grid_times'], batch['segments_batch'])
            
            # --- Loss Calculation ---
            snap_losses = calculate_snap_loss(batch, model_outputs, model, None, config)
            mode_losses = calculate_segment_mode_loss(seg_logits, seg_h, batch['segments_batch'], config)
            
            loss_loc_ce, loss_loc_mse, loss_purp_ce, loss_purp_mse, kl_loss = snap_losses
            loss_mode_ce, loss_mode_feat = mode_losses

            total_loss = (
                config.loss_weight_classification * loss_loc_ce +
                config.loss_weight_embedding * loss_loc_mse +
                config.loss_weight_purpose_class * loss_purp_ce +
                config.loss_weight_purpose_mse * loss_purp_mse +
                config.loss_weight_mode_ce_segment * loss_mode_ce +
                config.loss_weight_mode_feat_segment * loss_mode_feat +
                config.kl_weight * kl_loss
            )

            total_loss.backward()
            optimizer.step()

        if (i + 1) % 500 == 0:
            print(f"Iter {i+1}, Loss: {total_loss.item():.4f} | "
                  f"Snap (Loc/Purp): {loss_loc_ce.item():.2f}/{loss_purp_ce.item():.2f} | "
                  f"Segment (Mode CE/Feat): {loss_mode_ce.item():.2f}/{loss_mode_feat.item():.2f} | "
                  f"KL: {kl_loss.item():.2f}")

        # Save the model if it's the best so far
        if total_loss.item() < best_loss:
            best_loss = total_loss.item()
            save_path = saved_models_dir / "best_model.pth"
            torch.save(model.state_dict(), save_path)
            print(f"✨ New best model saved to {save_path} with loss: {best_loss:.4f}")

    print("✅ Training complete.")

if __name__ == "__main__":
    train()
