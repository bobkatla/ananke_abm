"""
Household Zone Movement Prediction using a Controlled Differential Equation (CDE)
based on a Spatio-Temporal Graph Neural Network (STGNN).
This model is autoregressive and learns to predict the next zone based on the
continuous evolution of a person's hidden state, controlled by their trajectory.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings
import os
import time
import torchcde

warnings.filterwarnings('ignore')

from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx
from ananke_abm.utils.helpers import pad_sequences

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw


@dataclass
class HouseholdConfig:
    """Configuration for household movement prediction"""
    zone_embed_dim: int = 64
    person_feat_dim: int = 8 
    person_embed_dim: int = 32
    hidden_dim: int = 128
    num_layers: int = 2
    num_zones: int = 10
    learning_rate: float = 0.005
    num_epochs: int = 5000
    dropout: float = 0.2
    # CDE specific
    cde_hidden_dim: int = 64
    cde_method: str = 'dopri5'
    cde_rtol: float = 1e-4
    cde_atol: float = 1e-4
    # Training windowing
    min_window_size: int = 4
    max_window_size: int = 10
    window_step_size: int = 2


class CDEFunc(nn.Module):
    """
    The function f that defines the dynamics of the hidden state h in the CDE.
    It's a feed-forward network that takes the hidden state h and returns the
    matrix that is multiplied with dX/dt.
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        # The function f maps the hidden state h to a matrix.
        # f: R^hidden_dim -> R^(hidden_dim x input_dim)
        self.linear = nn.Linear(hidden_dim, hidden_dim * input_dim)

    def forward(self, t, h):
        # h has shape (batch, hidden_dim)
        # The `t` argument is mandatory for the solver API, but we don't use it here
        # because time-dependence is now handled by the augmented control path.
        output = self.linear(h)
        output = torch.tanh(output)
        return output.view(-1, self.hidden_dim, self.input_dim)


class CDE_STGNN(nn.Module):
    """
    The main Spatio-Temporal Graph Neural Network model using a time-aware,
    identity-aware CDE-RNN.
    """
    def __init__(self, config: HouseholdConfig, num_zones: int, person_feat_dim: int, zone_features: torch.Tensor, edge_index_phys: torch.Tensor, edge_index_sem: torch.Tensor):
        super().__init__()
        self.config = config
        
        # --- GNN Layers for Zone Embeddings (now owned by the model) ---
        self.zone_gnn_phys = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        self.zone_gnn_sem = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        self.register_buffer('static_zone_features', zone_features)
        self.register_buffer('edge_index_phys', edge_index_phys)
        self.register_buffer('edge_index_sem', edge_index_sem)
        
        # --- Static Feature Embedders (now owned by the model) ---
        self.person_feature_embedder = nn.Linear(person_feat_dim, config.person_embed_dim)

        # --- CDE and Prediction Components ---
        gnn_embed_dim = config.zone_embed_dim * 2
        static_feature_dim = config.person_embed_dim + gnn_embed_dim # person embed + home GNN embed
        
        # The initial hidden state h0 is a function of static features and the first GNN embedding
        self.initial_state_mapper = nn.Linear(static_feature_dim + gnn_embed_dim, config.cde_hidden_dim)

        # Augmented control path dim: time (1) + static_features + current gnn_path
        control_path_dim = 1 + static_feature_dim + gnn_embed_dim
        self.cde_func = CDEFunc(control_path_dim, config.cde_hidden_dim)
        
        self.predictor = nn.Linear(config.cde_hidden_dim, num_zones)

    def get_gnn_embeds(self):
        """Helper function to compute the current zone embeddings."""
        phys_zone_embeds = torch.relu(self.zone_gnn_phys(self.static_zone_features, self.edge_index_phys))
        sem_zone_embeds = torch.relu(self.zone_gnn_sem(self.static_zone_features, self.edge_index_sem))
        return phys_zone_embeds, sem_zone_embeds

    def forward(self, times, X_path, initial_state):
        """
        The forward pass now accepts a torchcde.CubicSpline object representing
        the time-and-identity-augmented control path.
        """
        h = torchcde.cdeint(
            X=X_path,
            func=self.cde_func,
            z0=initial_state,
            t=times,
            method=self.config.cde_method,
            rtol=self.config.cde_rtol,
            atol=self.config.cde_atol
        )
        pred_y = self.predictor(h)
        return pred_y


def get_device():
    """Get the best available device (CUDA or CPU)."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model():
    """
    Trains the CDE-STGNN model.
    """
    print("🏠 Training Household Zone Movement Prediction Model (Time/ID-Aware CDE)")
    print("=" * 60)
    
    config = HouseholdConfig()
    processor = HouseholdDataProcessor()
    device = get_device()
    print(f"🔬 Using device: {device}")

    # The Data Processor now only provides raw data
    print("📊 Processing raw data...")
    data = processor.process_data()
    config.num_zones = data['num_zones']

    model = CDE_STGNN(
        config,
        data['num_zones'],
        data['person_features_raw'].shape[1],
        data['zone_features'],
        data['edge_index_phys'],
        data['edge_index_sem']
    ).to(device)

    # Add torch.compile for potential speedup - NOTE: Disabled due to incompatibility with torchcde library.
    # try:
    #     model = torch.compile(model, mode="reduce-overhead")
    #     print("✅ Model compiled successfully with torch.compile!")
    # except Exception as e:
    #     print(f"⚠️ Could not compile model with torch.compile: {e}. Continuing without it.")

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"\n🚀 Training for {config.num_epochs} epochs...")
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models')
    os.makedirs(save_dir, exist_ok=True)
    
    losses = []
    best_loss = float('inf')

    # Move all data to the selected device
    full_times = data['times'].to(device)
    full_y = data['trajectories_y'].to(device)
    person_features_raw = data['person_features_raw'].to(device)
    num_people, full_seq_len = full_y.shape

    for epoch in range(config.num_epochs):
        model.train()
        optimizer.zero_grad()
        
        window_size = torch.randint(config.min_window_size, config.max_window_size, (1,)).item()
        start_idx = torch.randint(0, full_seq_len - window_size, (1,)).item()
        end_idx = start_idx + window_size
        
        times = full_times[start_idx:end_idx]
        y = full_y[:, start_idx:end_idx]
        
        # --- Graph is now built dynamically inside the training loop ---
        
        # 1. Get current GNN embeddings from the model
        phys_zone_embeds, sem_zone_embeds = model.get_gnn_embeds()

        # 2. Create static features for the household
        person_embeds = model.person_feature_embedder(person_features_raw)
        home_zone_ids = full_y[:, 0]
        home_phys_embeds = phys_zone_embeds[home_zone_ids]
        home_sem_embeds = sem_zone_embeds[home_zone_ids]
        static_features = torch.cat([person_embeds, home_phys_embeds, home_sem_embeds], dim=1)
        
        # 3. Get initial hidden state h0 for the window
        initial_gnn_embed = torch.cat([
            phys_zone_embeds[full_y[:, start_idx]],
            sem_zone_embeds[full_y[:, start_idx]]
        ], dim=1)
        h0 = torch.tanh(model.initial_state_mapper(torch.cat([static_features, initial_gnn_embed], dim=1)))
        
        # 4. Construct the augmented control path for the window
        window_gnn_path_phys = phys_zone_embeds[y]
        window_gnn_path_sem = sem_zone_embeds[y]
        window_gnn_path = torch.cat([window_gnn_path_phys, window_gnn_path_sem], dim=2)
        
        window_time_path = times.view(1, -1, 1).expand(num_people, -1, -1)
        window_static_path = static_features.unsqueeze(1).expand(-1, window_size, -1)
        
        window_X_values = torch.cat([window_time_path, window_static_path, window_gnn_path], dim=2)
        X_path_window = torchcde.CubicSpline(torchcde.hermite_cubic_coefficients_with_backward_differences(window_X_values))
        
        # --- Forward Pass on the window ---
        pred_y = model(times, X_path_window, h0)
        
        pred_for_loss = pred_y[:, :-1, :].reshape(-1, config.num_zones)
        target_for_loss = y[:, 1:].reshape(-1)
        
        loss = F.cross_entropy(pred_for_loss, target_for_loss)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.item())
        
        if loss.item() < best_loss:
            best_loss = loss.item()
            torch.save(model.state_dict(), os.path.join(save_dir, 'cde_stgnn_best.pth'))

        if (epoch + 1) % 100 == 0:
            print(f"   Epoch {epoch+1:5d}/{config.num_epochs} | Loss: {loss.item():.4f} | Best Loss: {best_loss:.4f}")
            
    # Final plot
    plt.figure(figsize=(10, 5))
    plt.plot(losses)
    plt.title("Training Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, "cde_training_loss.png"))
    plt.close()

    print(f"\n✅ Training completed! Final loss: {losses[-1]:.4f}")
    return model, data, processor, config, data['adjacency_matrix']


class HouseholdDataProcessor:
    """Process mock data for the time-aware CDE model."""
    def __init__(self):
        self.zone_to_id = {}
        self.id_to_zone = {}
        self.person_names = []
        self.zone_names = []
        
    def _create_semantic_adjacency_matrix(self, trajectories, num_zones, threshold_percentile=25):
        """
        Creates a semantic adjacency matrix based on the similarity of zone usage
        patterns using Dynamic Time Warping (DTW).
        
        An edge is created between two zones if the DTW distance between their
        activity time series is in the bottom `threshold_percentile` of all
        pairwise distances.
        """
        print("   Calculating semantic adjacency matrix using DTW...")
        num_people, max_len = trajectories.shape

        # 1. Create a time series for each zone representing its activity (is someone present?)
        zone_activity = torch.zeros((num_zones, max_len))
        for t in range(max_len):
            for p in range(num_people):
                zone_id = trajectories[p, t]
                # Assuming padding value is a non-valid zone_id and won't be accessed.
                # If padding is a valid zone_id, this logic needs adjustment.
                if zone_id < num_zones:
                    zone_activity[zone_id, t] = 1 # Mark as active (presence)

        # 2. Compute the DTW distance for every pair of zones
        distances = []
        zone_pairs = []
        for i in range(num_zones):
            for j in range(i + 1, num_zones):
                ts_i = zone_activity[i].numpy()
                ts_j = zone_activity[j].numpy()
                # Use absolute difference for scalar points in the time series, as euclidean expects vectors.
                distance, _ = fastdtw(ts_i, ts_j, dist=lambda a, b: abs(a - b))
                distances.append(distance)
                zone_pairs.append((i, j))

        if not distances:
            # Handle case with no pairs or single zone
            return torch.empty((2, 0), dtype=torch.long)

        # 3. Determine the threshold and build the adjacency matrix
        distance_threshold = np.percentile(distances, threshold_percentile)
        
        adj = torch.zeros((num_zones, num_zones))
        for (i, j), dist in zip(zone_pairs, distances):
            if dist <= distance_threshold:
                adj[i, j] = 1
                adj[j, i] = 1 # Symmetric relationship
                
        # 4. Convert adjacency matrix to edge index format
        edge_index = adj.to_sparse().indices()
        print(f"   Semantic graph created with {edge_index.shape[1]} edges (DTW distance < {distance_threshold:.2f})")
        return edge_index

    def process_data(self):
        """
        Process mock data to return raw tensors.
        All embeddings and path creation will happen inside the training loop.
        """
        print("   Loading and processing data from mock_2p...")
        sarah_data, marcus_data = create_two_person_training_data()

        person_features_raw = torch.stack([d['person_attrs'] for d in [sarah_data, marcus_data]])
        trajectories_data = [torch.tensor(d['zone_observations'], dtype=torch.long) for d in [sarah_data, marcus_data]]
        self.person_names = [d['person_name'] for d in [sarah_data, marcus_data]]
        
        # GNN graph structure
        zone_features = sarah_data['zone_features']
        zone_graph, self.zone_names = create_mock_zone_graph()
        edge_index_phys = from_networkx(zone_graph).edge_index
        
        # Padded trajectories and time grid
        # IMPORTANT: Use a padding value that is NOT a valid zone_id. Here, num_zones is a safe choice.
        num_zones = sarah_data['num_zones']
        padding_value = num_zones
        max_len = max(len(t) for t in trajectories_data)
        times = torch.linspace(0, max_len - 1, max_len)
        padded_trajs = pad_sequences(trajectories_data, padding_value=padding_value)

        # Create the semantic graph based on movement patterns
        edge_index_sem = self._create_semantic_adjacency_matrix(padded_trajs, num_zones)

        return {
            'person_features_raw': person_features_raw,
            'trajectories_y': padded_trajs,
            'times': times,
            'num_people': len(self.person_names),
            'num_zones': num_zones,
            'zone_features': zone_features,
            'edge_index_phys': edge_index_phys,
            'edge_index_sem': edge_index_sem,
            'adjacency_matrix': torch.tensor(nx.to_numpy_array(zone_graph), dtype=torch.float32),
            'person_names': self.person_names,
            'zone_names': self.zone_names,
            'padding_value': padding_value,
        }

def main():
    """Main function to run model training."""
    print("🏠 Household Zone Movement Prediction using CDE-STGNN")
    print("Based on: Controlled Differential Equations for Time Series")
    print("=" * 60)
    
    try:
        train_model()
        print("\n✅ Training complete.")
        
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()