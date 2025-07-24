"""
Household Zone Movement Prediction using an LSTM Encoder and a CDE Decoder.
This architecture is inspired by the FEM/BEM approach in physics simulations,
decomposing the problem into two parts:
1. An LSTM Encoder finds the high-level "context" from a long history.
2. A CDE Decoder uses that context to make a precise, short-term prediction.
This should be significantly faster and more effective than a single, long CDE.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import os
import torchcde
from torch_geometric.nn import GATConv, GCNConv
from fastdtw import fastdtw

warnings.filterwarnings('ignore')

# Using the same mock data generators
from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.utils.helpers import pad_sequences


@dataclass
class HouseholdEncoderDecoderConfig:
    """Configuration for the Encoder-Decoder CDE model"""
    zone_embed_dim: int = 32
    person_embed_dim: int = 32
    
    # Encoder (LSTM)
    encoder_hidden_dim: int = 128
    encoder_num_layers: int = 2
    
    # Decoder (CDE)
    cde_hidden_dim: int = 64
    
    # Training
    learning_rate: float = 0.001
    num_epochs: int = 2000
    weight_decay: float = 1e-5
    
    # CDE specific
    cde_method: str = 'dopri5'
    cde_rtol: float = 1e-3
    cde_atol: float = 1e-3
    
    # Windowing
    history_length: int = 10  # How many past steps the LSTM sees
    prediction_length: int = 2 # CDE sees last step to predict the immediate next one


class CDEFunc(nn.Module):
    """The vector field f for the CDE."""
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # Add LayerNorm for stability
        self.norm = nn.LayerNorm(hidden_dim)
        self.linear1 = nn.Linear(hidden_dim, 128)
        self.linear2 = nn.Linear(128, input_dim * hidden_dim)

    def forward(self, t, h):
        h = self.norm(h)
        h = self.linear1(h)
        h = h.relu()
        h = self.linear2(h)
        return torch.tanh(h).view(-1, self.hidden_dim, self.input_dim)


class EncoderDecoderCDE(nn.Module):
    """
    An LSTM-CDE model that first encodes a history and then decodes the next step.
    This version includes GNNs for both social and spatial context.
    """
    def __init__(self, config: HouseholdEncoderDecoderConfig, num_zones: int, person_feat_dim: int, zone_feat_dim: int, edge_index_phys: torch.Tensor, edge_index_sem: torch.Tensor, padding_idx: int):
        super().__init__()
        self.config = config
        
        # --- Spatial GNNs for Zone Embeddings ---
        self.zone_gnn_phys = GCNConv(zone_feat_dim, config.zone_embed_dim)
        self.zone_gnn_sem = GCNConv(zone_feat_dim, config.zone_embed_dim)
        self.register_buffer('edge_index_phys', edge_index_phys)
        self.register_buffer('edge_index_sem', edge_index_sem)

        # --- Embedders ---
        self.person_feature_embedder = nn.Linear(person_feat_dim, config.person_embed_dim)

        # --- Social GNN for household interaction ---
        self.social_gnn = GATConv(config.person_embed_dim, config.person_embed_dim, heads=2, concat=False, dropout=0.2)

        # --- Encoder (LSTM) ---
        # The input dimension now includes the concatenated embeddings from the two zone GNNs
        gnn_embed_dim = config.zone_embed_dim * 2
        encoder_input_dim = gnn_embed_dim + config.person_embed_dim + gnn_embed_dim # Zone, Person, Home
        self.encoder_rnn = nn.LSTM(
            input_size=encoder_input_dim,
            hidden_size=config.encoder_hidden_dim,
            num_layers=config.encoder_num_layers,
            batch_first=True,
            dropout=0.2 # Add dropout for regularization
        )

        # --- Decoder (CDE) ---
        self.initial_state_mapper = nn.Linear(config.encoder_hidden_dim, config.cde_hidden_dim)
        
        # The CDE's control path includes time and the concatenated GNN embeddings
        control_path_dim = 2 + gnn_embed_dim # sin/cos time + phys/sem gnn
        self.cde_func = CDEFunc(control_path_dim, config.cde_hidden_dim)
        
        # The final predictor maps the CDE's state to a zone prediction
        self.predictor = nn.Linear(config.cde_hidden_dim, num_zones)

    def get_zone_gnn_embeds(self, zone_features):
        """Helper to compute zone embeddings from the spatial GNNs."""
        phys_embeds = torch.relu(self.zone_gnn_phys(zone_features, self.edge_index_phys))
        sem_embeds = torch.relu(self.zone_gnn_sem(zone_features, self.edge_index_sem))
        return torch.cat([phys_embeds, sem_embeds], dim=1)

    def forward(self, history_path, cde_coeffs, cde_times):
        """
        Processes history with LSTM, then decodes next step with CDE.
        
        Args:
            history_path: Tensor of shape (batch, history_len, features) for the LSTM.
            cde_coeffs: Pre-computed tensor of coefficients for the CDE's path.
            cde_times: The time points for the short CDE solve.
        """
        # 1. Encode History to get context
        _, (h_n, _) = self.encoder_rnn(history_path)
        context = h_n[-1] # Use the final hidden state of the last LSTM layer as context

        # 2. Initialize CDE state from the context
        h0 = torch.tanh(self.initial_state_mapper(context))

        # 3. Solve CDE for the next step prediction
        X_path_short = torchcde.CubicSpline(cde_coeffs)
        
        h = torchcde.cdeint(
            X=X_path_short,
            func=self.cde_func,
            z0=h0,
            t=cde_times,
            method=self.config.cde_method,
            rtol=self.config.cde_rtol,
            atol=self.config.cde_atol
        )
        
        # We only care about the prediction at the very end of the short interval
        pred_y = self.predictor(h[:, -1, :]) # Shape: (batch, num_zones)
        return pred_y


def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class SimpleDataProcessor:
    """Process mock data for the CDE model, now including spatial graphs."""
    def _create_semantic_adjacency_matrix(self, trajectories, num_zones, threshold_percentile=25):
        """
        Creates a semantic adjacency matrix based on the similarity of zone usage patterns.
        """
        num_people, max_len = trajectories.shape
        zone_activity = torch.zeros((num_zones, max_len))
        for t in range(max_len):
            for p in range(num_people):
                zone_id = trajectories[p, t]
                if zone_id < num_zones:
                    zone_activity[zone_id, t] = 1

        distances, zone_pairs = [], []
        for i in range(num_zones):
            for j in range(i + 1, num_zones):
                dist, _ = fastdtw(zone_activity[i].numpy(), zone_activity[j].numpy(), dist=lambda a, b: abs(a - b))
                distances.append(dist)
                zone_pairs.append((i, j))
        
        if not distances: return torch.empty((2, 0), dtype=torch.long)
        
        distance_threshold = np.percentile(distances, threshold_percentile)
        adj = torch.zeros((num_zones, num_zones))
        for (i, j), dist in zip(zone_pairs, distances):
            if dist <= distance_threshold:
                adj[i, j] = adj[j, i] = 1
        return adj.to_sparse().indices()

    def process_data(self, repeat_pattern=True):
        """
        Process mock data to return raw tensors for the simple CDE model.
        """
        print("   Loading and processing data from mock_2p...")
        sarah_data, marcus_data = create_two_person_training_data(repeat_pattern=repeat_pattern)

        person_features_raw = torch.stack([d['person_attrs'] for d in [sarah_data, marcus_data]])
        trajectories_data = [torch.tensor(d['zone_observations'], dtype=torch.long) for d in [sarah_data, marcus_data]]
        person_names = [d['person_name'] for d in [sarah_data, marcus_data]]
        num_zones = sarah_data['num_zones']
        zone_features = sarah_data['zone_features']
        
        # Physical graph from data generator
        edge_index_phys = sarah_data['edge_index']

        padding_value = num_zones 
        max_len = max(len(t) for t in trajectories_data)
        times = torch.linspace(0, max_len - 1, max_len)
        padded_trajs = pad_sequences(trajectories_data, padding_value=padding_value)

        # Create semantic graph from trajectories
        edge_index_sem = self._create_semantic_adjacency_matrix(padded_trajs, num_zones)

        # Create cyclical time features
        time_of_day = times % 24.0
        times_sin = torch.sin(2 * np.pi * time_of_day / 24.0)
        times_cos = torch.cos(2 * np.pi * time_of_day / 24.0)
        time_features = torch.stack([times_sin, times_cos], dim=1)

        # Create the people graph for household interactions
        people_edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)

        return {
            'person_features_raw': person_features_raw,
            'trajectories_y': padded_trajs,
            'times': times,
            'time_features': time_features,
            'num_people': len(person_names),
            'num_zones': num_zones,
            'padding_value': padding_value,
            'person_names': person_names,
            'zone_features': zone_features,
            'edge_index_phys': edge_index_phys,
            'edge_index_sem': edge_index_sem,
            'people_edge_index': people_edge_index
        }


def train_model():
    """
    Trains the Encoder-Decoder CDE model.
    """
    print("🏠 Training Encoder-Decoder CDE Model")
    print("=" * 60)
    
    config = HouseholdEncoderDecoderConfig()
    processor = SimpleDataProcessor()
    device = get_device()
    print(f"🔬 Using device: {device}")

    print("📊 Processing raw data...")
    # By default, use the better, repeated data for training.
    # Can be set to False for testing the single-day pattern.
    data = processor.process_data(repeat_pattern=True)
    num_zones = data['num_zones']
    
    # Move graph data to device
    zone_features = data['zone_features'].to(device)
    edge_index_phys = data['edge_index_phys'].to(device)
    edge_index_sem = data['edge_index_sem'].to(device)

    model = EncoderDecoderCDE(
        config,
        num_zones=num_zones,
        person_feat_dim=data['person_features_raw'].shape[1],
        zone_feat_dim=zone_features.shape[1],
        edge_index_phys=edge_index_phys,
        edge_index_sem=edge_index_sem,
        padding_idx=data['padding_value']
    ).to(device)

    # Add weight decay for regularization
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    # Use a more adaptive scheduler that reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5)
    
    print(f"\n🚀 Training for {config.num_epochs} epochs...")
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models', 'encoder_decoder_cde33')
    os.makedirs(save_dir, exist_ok=True)
    
    # Store losses for plotting and saving
    train_losses = []
    val_losses = []
    val_epochs = []
    best_val_loss = float('inf')

    full_times = data['times'].to(device)
    time_features = data['time_features'].to(device) # Get new features
    full_y = data['trajectories_y'].to(device)
    person_features_raw = data['person_features_raw'].to(device)
    people_edge_index = data['people_edge_index'].to(device) # Get social graph
    num_people, full_seq_len = full_y.shape
    
    # --- Train/Validation Split ---
    split_idx = int(full_seq_len * 0.8)
    train_y = full_y[:, :split_idx]
    val_y = full_y[:, split_idx:]
    train_times = full_times[:split_idx]
    val_times = full_times[split_idx:]
    train_time_features = time_features[:split_idx] # Split new features
    val_time_features = time_features[split_idx:]   # Split new features
    
    home_zone_ids = full_y[:, 0] # Assume first observation is home

    for epoch in range(config.num_epochs):
        model.train()
        
        # --- Training Step ---
        optimizer.zero_grad()
        
        # 1. Compute all embeddings for this step
        # Social embeddings
        person_embeds = model.person_feature_embedder(person_features_raw)
        social_context = model.social_gnn(person_embeds, people_edge_index)
        person_embeds = person_embeds + social_context

        # Spatial zone embeddings
        zone_gnn_embeds_real = model.get_zone_gnn_embeds(zone_features)
        
        # Add a zero-vector for the padding index to avoid out-of-bounds error
        padding_embed = torch.zeros(1, zone_gnn_embeds_real.shape[1], device=device)
        zone_gnn_embeds = torch.cat([zone_gnn_embeds_real, padding_embed], dim=0)

        home_zone_embeds = zone_gnn_embeds[home_zone_ids]
        
        # 2. Sample a window from the TRAINING set
        total_window_size = config.history_length + config.prediction_length
        start_idx = torch.randint(0, train_y.shape[1] - total_window_size, (1,)).item()
        history_end_idx = start_idx + config.history_length
        cde_end_idx = history_end_idx + config.prediction_length

        # 3. Slice the data for the window
        y_history = train_y[:, start_idx:history_end_idx]
        y_cde = train_y[:, history_end_idx - 1 : cde_end_idx]
        
        # 4. Construct LSTM history path (with home embedding)
        history_zone_embeds = zone_gnn_embeds[y_history] # Use GNN embeds
        static_embeds = torch.cat([person_embeds, home_zone_embeds], dim=1)
        expanded_static_embeds = static_embeds.unsqueeze(1).expand(-1, config.history_length, -1)
        history_path = torch.cat([history_zone_embeds, expanded_static_embeds], dim=2)
        
        # 5. Construct CDE control path and get coefficients
        cde_zone_embeds = zone_gnn_embeds[y_cde] # Use GNN embeds
        cde_times = train_times[history_end_idx - 1 : cde_end_idx]
        
        # Use new cyclical time features for the CDE path
        cde_time_feats = train_time_features[history_end_idx - 1 : cde_end_idx]
        cde_time_path = cde_time_feats.unsqueeze(0).expand(num_people, -1, -1)
        
        cde_path_values = torch.cat([cde_time_path, cde_zone_embeds], dim=2)
        cde_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(cde_path_values)
        
        pred_y = model(history_path, cde_coeffs, cde_times)
        
        target = train_y[:, cde_end_idx - 1]
        mask = (target != data['padding_value'])
        
        if not mask.any():
            continue

        loss = F.cross_entropy(pred_y[mask], target[mask])
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        train_losses.append(loss.item())

        # --- Validation Step ---
        # Run validation only periodically to avoid freezing the training process.
        if (epoch + 1) % 20 == 0:
            # Only run validation if the validation set is large enough for at least one window.
            if val_y.shape[1] < total_window_size:
                if (epoch + 1) % 100 == 0:
                     print(f"   Epoch {epoch+1:5d} | Train Loss: {loss.item():.4f} | Validation set too small, skipping.")
                continue

            model.eval()
            with torch.no_grad():
                total_val_loss = 0
                num_val_samples = 0
                
                val_start_idx = 0
                while val_start_idx < (val_y.shape[1] - total_window_size):
                    v_history_end = val_start_idx + config.history_length
                    v_cde_end = v_history_end + config.prediction_length

                    vy_history = val_y[:, val_start_idx:v_history_end]
                    vy_cde = val_y[:, v_history_end - 1 : v_cde_end]
                    
                    vtarget = val_y[:, v_cde_end - 1]
                    vmask = (vtarget != data['padding_value'])
                    
                    if vmask.any():
                        # Embeddings are pre-computed for the validation step
                        vhistory_zone_embeds = zone_gnn_embeds[vy_history]
                        vstatic_embeds = torch.cat([person_embeds, home_zone_embeds], dim=1)
                        vexpanded_static_embeds = vstatic_embeds.unsqueeze(1).expand(-1, config.history_length, -1)
                        vhistory_path = torch.cat([vhistory_zone_embeds, vexpanded_static_embeds], dim=2)
                        
                        vcde_zone_embeds = zone_gnn_embeds[vy_cde]
                        vcde_times = val_times[v_history_end - 1 : v_cde_end]

                        # Use new cyclical time features for validation CDE path
                        vcde_time_feats = val_time_features[v_history_end - 1 : v_cde_end]
                        vcde_time_path = vcde_time_feats.unsqueeze(0).expand(num_people, -1, -1)
                        
                        vcde_path_values = torch.cat([vcde_time_path, vcde_zone_embeds], dim=2)
                        vcde_coeffs = torchcde.hermite_cubic_coefficients_with_backward_differences(vcde_path_values)
                        
                        vpred_y = model(vhistory_path, vcde_coeffs, vcde_times)
                        vloss = F.cross_entropy(vpred_y[vmask], vtarget[vmask])
                        total_val_loss += vloss.item() * vmask.sum()
                        num_val_samples += vmask.sum()

                    val_start_idx += 1

                if num_val_samples > 0:
                    avg_val_loss = total_val_loss / num_val_samples
                    val_losses.append(avg_val_loss)
                    val_epochs.append(epoch + 1)

                    if avg_val_loss < best_val_loss:
                        best_val_loss = avg_val_loss
                        torch.save(model.state_dict(), os.path.join(save_dir, 'enc_dec_cde_best.pth'))
                        print(f"   Epoch {epoch+1:5d} | Train Loss: {loss.item():.4f} | ✨ Val Loss: {avg_val_loss:.4f} (new best)")
                    else:
                        print(f"   Epoch {epoch+1:5d} | Train Loss: {loss.item():.4f} | Val Loss: {avg_val_loss:.4f}")
                    
                    # Step the scheduler based on the validation loss
                    scheduler.step(avg_val_loss)
                    
                else:
                    print(f"   Epoch {epoch+1:5d} | Train Loss: {loss.item():.4f} | No validation samples in this epoch.")
        
        # If not a validation epoch, we don't step the ReduceLROnPlateau scheduler
            
    # Final plot
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label="Training Loss (per batch)")
    plt.plot(val_epochs, val_losses, label="Validation Loss", marker='o')
    plt.title("Encoder-Decoder CDE Training and Validation Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    plt.savefig(os.path.join(save_dir, "enc_dec_cde_training_validation_loss.png"))
    plt.close()

    # Save the losses to a file for later analysis
    loss_path = os.path.join(save_dir, 'training_losses.npz')
    np.savez(loss_path, train_loss=np.array(train_losses), val_loss=np.array(val_losses), val_epochs=np.array(val_epochs))
    print(f"   -> Saved losses to {loss_path}")

    print(f"\n✅ Training completed! Final validation loss: {val_losses[-1]:.4f}")


def main():
    """Main function to run model training."""
    try:
        train_model()
        print("\n✅ Training complete.")
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 