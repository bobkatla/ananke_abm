"""
Household Zone Movement Prediction using STG-NODE approach
Simplified version focusing on zone transitions with physics constraints
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
warnings.filterwarnings('ignore')

from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
import networkx as nx
from torch_geometric.nn import GCNConv
from torch_geometric.utils import from_networkx

from scipy.spatial.distance import euclidean
from fastdtw import fastdtw

from ananke_abm.models.run.ode_components import ODEFunc, ODEBlock
from ananke_abm.models.run.cvae_components import Encoder, LatentVariable, LatentODEFunc
from torchdiffeq import odeint_adjoint as odeint

@dataclass
class HouseholdConfig:
    """Configuration for household movement prediction"""
    zone_embed_dim: int = 64  # Much larger embedding
    temporal_embed_dim: int = 32  # Larger temporal embedding
    hidden_dim: int = 128  # Much larger hidden dimension
    num_layers: int = 3  # Multiple layers for complexity
    num_zones: int = 10
    learning_rate: float = 0.001  # Lower learning rate for stability
    num_epochs: int = 11000  # Extended training time
    physics_weight: float = 1.0  # Not needed with hard constraints
    exploration_noise: float = 0.02  # Smaller noise for stability
    dropout: float = 0.1  # Regularization
    person_feat_dim: int = 8 # UPDATED to use all features from mock_2p
    person_embed_dim: int = 32
    ode_hidden_dim: int = 64 # Hidden dim for the layers inside the ODE function
    latent_dim: int = 16 # Dimension of the CVAE latent space
    encoder_hidden_dim: int = 64 # Hidden dim for the encoder GRU
    kld_weight: float = 0.001 # Weight for the KL Divergence loss term
    latent_ode_hidden_dim: int = 32 # Hidden dim for the latent ODE function

class CombinedODEFunc(nn.Module):
    """Combines latent and physical dynamics for simultaneous solving."""
    def __init__(self, latent_ode_func, physical_dynamics_func):
        super().__init__()
        self.latent_ode_func = latent_ode_func
        self.physical_dynamics_func = physical_dynamics_func

    def forward(self, t, state):
        z, x = state # Unpack the combined state
        dz_dt = self.latent_ode_func(t, z)
        dx_dt = self.physical_dynamics_func(t, x, z)
        return (dz_dt, dx_dt)

class STG_CVAE(nn.Module):
    """
    The main Hierarchical Latent ODE CVAE model.
    """
    def __init__(self, config: HouseholdConfig, num_zones: int, zone_features: torch.Tensor, person_features: torch.Tensor, edge_index_phys: torch.Tensor, edge_index_sem: torch.Tensor):
        super().__init__()
        self.config = config
        self.encoder = Encoder(num_zones, config.encoder_hidden_dim, config.latent_dim)
        self.latent_sampler = LatentVariable()
        
        # Instantiate the two dynamic functions
        latent_dynamics = LatentODEFunc(config.latent_dim, config.latent_ode_hidden_dim)
        physical_dynamics = STGNodeDynamics(config, zone_features, person_features, edge_index_phys, edge_index_sem)
        
        # The main ODE function is a combination of both
        self.combined_ode_func = CombinedODEFunc(latent_dynamics, physical_dynamics)

    def forward(self, agent_trajectories, initial_time, eval_times):
        # 1. Encode the full trajectories to get the *initial* latent state, z0
        mu_0, log_var_0 = self.encoder(agent_trajectories)
        z_0 = self.latent_sampler(mu_0, log_var_0)
        
        # 2. Get the initial physical state, x0
        initial_zones = agent_trajectories[:, 0]
        x_0 = self.combined_ode_func.physical_dynamics_func.get_initial_state(initial_zones, z_0)
        
        # 3. Set up the combined initial state and solve the hierarchical ODE
        combined_state_0 = (z_0, x_0)
        
        # The ODE solver handles the temporal evolution of both z and x
        _, x_solution = odeint(
            self.combined_ode_func,
            combined_state_0,
            eval_times,
            method='dopri5'
        )
        
        # 4. Predict zones from the final evolved physical states
        zone_logits = self.combined_ode_func.physical_dynamics_func.zone_predictor(x_solution)

        return zone_logits, mu_0, log_var_0

class STGNodeDynamics(nn.Module):
    """
    Defines the dynamics of the agent states (dx/dt), conditioned on z(t).
    This is NOT a full model, but a component of the combined ODE.
    """
    def __init__(self, config: HouseholdConfig, zone_features: torch.Tensor, person_features: torch.Tensor, edge_index_phys: torch.Tensor, edge_index_sem: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer('static_zone_features', zone_features)
        self.register_buffer('person_features', person_features)
        self.register_buffer('edge_index_phys', edge_index_phys)
        self.register_buffer('edge_index_sem', edge_index_sem)

        # GNN Layers for spatial and semantic graphs
        self.zone_gnn_phys = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        self.zone_gnn_sem = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        
        # Person embedders to create initial hidden state
        self.person_feature_embedder = nn.Linear(config.person_feat_dim, config.person_embed_dim)
        initial_state_input_dim = config.person_embed_dim + (config.zone_embed_dim * 2) + config.latent_dim
        self.initial_state_mapper = nn.Linear(initial_state_input_dim, config.hidden_dim)
        
        # The ODE function for the physical state
        self.physical_ode_func = nn.Sequential(
            nn.Linear(config.hidden_dim + config.latent_dim, config.ode_hidden_dim),
            nn.Tanh(),
            nn.Linear(config.ode_hidden_dim, config.hidden_dim)
        )
        
        # Final output predictor
        self.zone_predictor = nn.Linear(config.hidden_dim, config.num_zones)
        
    def get_initial_state(self, initial_zones, z0):
        """Get initial person hidden states, conditioned on initial latent vector z0."""
        # Pre-compute GNN embeddings once
        phys_zone_embeds = torch.relu(self.zone_gnn_phys(self.static_zone_features, self.edge_index_phys))
        sem_zone_embeds = torch.relu(self.zone_gnn_sem(self.static_zone_features, self.edge_index_sem))
        
        initial_phys_context = phys_zone_embeds[initial_zones]
        initial_sem_context = sem_zone_embeds[initial_zones]
        
        person_embeds = self.person_feature_embedder(self.person_features)
        
        combined_context = torch.cat([person_embeds, initial_phys_context, initial_sem_context, z0], dim=-1)
        initial_state = torch.relu(self.initial_state_mapper(combined_context))
        return initial_state
        
    def forward(self, t, x, z):
        """Computes dx/dt, the derivative of the physical system state."""
        # The input x is the current hidden state from the ODE solver.
        # The input z is the current latent state from the ODE solver.
        input_for_ode = torch.cat([x, z], dim=-1)
        dx_dt = self.physical_ode_func(input_for_ode)
        return dx_dt

def compute_semantic_adjacency(trajectories, num_zones, threshold=0.5):
    """
    Computes a semantic adjacency matrix based on trajectory similarity using DTW.
    Instead of node-to-node, this computes zone-to-zone similarity based on the
    people's trajectories that visit them.
    """
    print("   Computing semantic adjacency matrix using DTW...")
    # This is a simplified example. A more complex implementation would
    # aggregate features for each zone based on who visits and when.
    # For now, we compare entire trajectories to create a person-person graph,
    # then assume this implies some zone-zone semantic connection.
    
    num_people = len(trajectories)
    person_similarity_matrix = np.ones((num_people, num_people))
    
    for i in range(num_people):
        for j in range(i + 1, num_people):
            # Reshape from (seq_len,) to (seq_len, 1) to be compatible with scipy's distance funcs
            traj_i = trajectories[i].reshape(-1, 1)
            traj_j = trajectories[j].reshape(-1, 1)
            distance, _ = fastdtw(traj_i, traj_j, dist=euclidean)
            person_similarity_matrix[i, j] = person_similarity_matrix[j, i] = distance

    # Normalize distances to get similarities (higher value = more similar)
    if num_people > 1:
        max_dist = person_similarity_matrix.max()
        if max_dist > 0:
            person_similarity = 1.0 - (person_similarity_matrix / max_dist)
        else:
            person_similarity = np.ones_like(person_similarity_matrix)
    else:
        person_similarity = np.ones_like(person_similarity_matrix)
        
    # How to map this person-person similarity to zone-zone?
    # This is a modeling choice. Let's create a simple mapping: if two people
    # are similar, the zones they frequently visit become semantically linked.
    zone_adj = np.zeros((num_zones, num_zones))
    
    # Get zone visitation counts for each person
    zone_visits = [np.bincount(traj.int(), minlength=num_zones) for traj in trajectories]
    
    for i in range(num_people):
        for j in range(i + 1, num_people):
            if person_similarity[i, j] > threshold:
                # Find common zones and link them
                zones_i = np.where(zone_visits[i] > 0)[0]
                zones_j = np.where(zone_visits[j] > 0)[0]
                
                for z1 in zones_i:
                    for z2 in zones_j:
                        if z1 != z2:
                           # Increase weight for connection, could be based on similarity
                           zone_adj[z1, z2] = zone_adj[z2, z1] = max(zone_adj[z1, z2], person_similarity[i, j])

    print("   Semantic adjacency matrix computed.")
    return torch.tensor(zone_adj, dtype=torch.float32)

class HouseholdDataProcessor:
    """Process mock data for household movement prediction from mock_2p"""
    
    def __init__(self):
        self.zone_to_id = {}
        self.id_to_zone = {}
        self.person_names = []
        
    def process_data(self):
        """Process the mock data from mock_2p"""
        print("   Loading and processing data from mock_2p...")
        sarah_data, marcus_data = create_two_person_training_data()

        # Combine Sarah and Marcus into a two-person household for the multi-agent model
        person_features = torch.stack([sarah_data['person_attrs'], marcus_data['person_attrs']])
        trajectories_data = [sarah_data['zone_observations'], marcus_data['zone_observations']]
        times_data = [sarah_data['times'], marcus_data['times']]
        person_names = [sarah_data['person_name'], marcus_data['person_name']]
        num_people = 2
        num_zones = sarah_data['num_zones']
        zone_features = sarah_data['zone_features']

        # Physical graph structure
        zone_graph, _ = create_mock_zone_graph()
        edge_index_phys = from_networkx(zone_graph).edge_index
        adj_matrix_phys = nx.to_numpy_array(zone_graph, nodelist=sorted(zone_graph.nodes()))

        # Semantic graph structure (A_se)
        full_trajectories = [torch.tensor(t) for t in trajectories_data]
        adj_matrix_sem = compute_semantic_adjacency(full_trajectories, num_zones)
        sem_graph = nx.from_numpy_array(adj_matrix_sem.numpy())
        edge_index_sem = from_networkx(sem_graph).edge_index
        
        # Create zone mappings for visualization/CSV output
        self.zone_to_id = {i + 1: i for i in range(num_zones)}
        self.id_to_zone = {i: i + 1 for i in range(num_zones)}
        self.person_names = person_names

        # Create a common time grid for the ODE solver
        all_times = torch.cat(times_data)
        min_time, max_time = all_times.min(), all_times.max()
        max_seq_len = max(len(t) for t in trajectories_data)
        common_time_grid = torch.linspace(min_time.item(), max_time.item(), max_seq_len)

        print(f"   Successfully processed data for: {person_names}")
        
        return {
            'trajectories_data': trajectories_data,
            'times_data': times_data,
            'common_time_grid': common_time_grid,
            'zone_features': zone_features,
            'edge_index_phys': edge_index_phys,
            'edge_index_sem': edge_index_sem,
            'num_zones': num_zones,
            'num_people': num_people,
            'person_names': self.person_names,
            'person_features': person_features,
            'adjacency_matrix': torch.tensor(adj_matrix_phys, dtype=torch.float32)
        }

def physics_constraint_loss(predictions, current_zones, adjacency_matrix):
    """
    Calculate physics constraint loss to prevent impossible movements
    Args:
        predictions: [batch, num_people, seq_len-1, num_zones] - predicted zone probabilities
        current_zones: [batch, num_people, seq_len] - actual current zones
        adjacency_matrix: [num_zones, num_zones] - zone connectivity
    """
    batch_size, num_people, seq_len_minus_1, num_zones = predictions.shape
    
    total_physics_loss = 0.0
    
    for t in range(seq_len_minus_1):
        current_zone_ids = current_zones[:, :, t]  # [batch, num_people]
        pred_probs = F.softmax(predictions[:, :, t, :], dim=-1)  # [batch, num_people, num_zones]
        
        physics_loss = 0.0
        for b in range(batch_size):
            for p in range(num_people):
                current_zone = current_zone_ids[b, p].item()
                
                # Get allowed next zones (connected zones + staying in same zone)
                allowed_zones = adjacency_matrix[current_zone].clone()
                allowed_zones[current_zone] = 1.0  # Can stay in same zone
                
                # Penalize probability mass on impossible transitions
                impossible_zones = 1.0 - allowed_zones
                impossible_prob = torch.sum(pred_probs[b, p] * impossible_zones)
                physics_loss += impossible_prob
        
        total_physics_loss += physics_loss
    
    return total_physics_loss / (batch_size * num_people * seq_len_minus_1)

def train_model():
    """Train the household movement prediction model"""
    print("🏠 Training Household Zone Movement Prediction Model")
    print("=" * 60)
    
    # Process data using the new processor
    print("📊 Processing mock data...")
    processor = HouseholdDataProcessor()
    data = processor.process_data()
    
    print(f"   Number of people: {data['num_people']}")
    print(f"   Zone features shape: {data['zone_features'].shape}")
    print(f"   Number of zones: {data['num_zones']}")
    print(f"   Person features shape: {data['person_features'].shape}")
    
    # Get adjacency matrix directly from the new data processor
    adjacency_matrix = data['adjacency_matrix']
    print(f"   Adjacency matrix shape: {adjacency_matrix.shape}")
    
    # Configuration
    config = HouseholdConfig()
    config.num_zones = data['num_zones']
    
    # Create the top-level CVAE model
    model = STG_CVAE(
        config, 
        data['num_zones'], 
        data['zone_features'], 
        data['person_features'], 
        data['edge_index_phys'],
        data['edge_index_sem']
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, weight_decay=1e-4)
    
    # Multiple schedulers for better training
    scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100, factor=0.7)
    scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
    
    # Prepare training data
    trajectories_data = data['trajectories_data']
    times_data = data['times_data']
    common_time_grid = data['common_time_grid']
    
    # Training loop with curriculum learning
    print(f"\n🚀 Training for {config.num_epochs} epochs...")
    save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models')
    save_dir = os.path.abspath(save_dir)
    plot_dir = os.path.join(save_dir, 'plots')
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(plot_dir, exist_ok=True)

    losses = []
    violation_rates = []
    best_loss = float('inf')
    best_violation = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kld_loss = 0.0
        total_violations = 0
        total_preds = 0
        
        # We need a consistent window for all people in the household
        # We'll use the curriculum learning window size on the shortest trajectory in the household
        shortest_traj_len = min(len(t) for t in trajectories_data)

        if epoch < 1000:
            max_window_size = 5
        elif epoch < 3000:
            max_window_size = 8
        elif epoch < 7000:
            max_window_size = 12
        else:
            max_window_size = 12
        
        window_size = min(max_window_size, shortest_traj_len - 1)
        if window_size < 2:
            continue
            
        step_size = max(1, window_size // 3)
        start_indices = list(range(0, shortest_traj_len - window_size, step_size))
        
        if epoch >= 7000:
            end_anchor_start = shortest_traj_len - window_size
            if end_anchor_start > 0 and end_anchor_start not in start_indices:
                start_indices.append(end_anchor_start)

        for start_idx in start_indices:
            end_idx = start_idx + window_size
            
            # Prepare a batch of trajectories for all agents
            agent_trajectories_batch = torch.stack([
                traj[start_idx:end_idx] for traj in trajectories_data
            ])
            
            # Prepare initial conditions
            initial_time = times_data[0][start_idx] # Assume common time grid for household
            eval_times = times_data[0][start_idx:end_idx]

            # Forward pass through the CVAE with the batch of trajectories
            raw_predictions, mu, log_var = model(agent_trajectories_batch, initial_time, eval_times)
            
            ground_truth_zones = agent_trajectories_batch
            
            # --- Physics Constraints and Violation Tracking ---
            constrained_predictions = raw_predictions.clone()
            for p in range(data['num_people']):
                for t in range(len(eval_times)): 
                    if t > 0:
                        prev_zone = ground_truth_zones[p, t-1].item()
                        valid_mask = adjacency_matrix[prev_zone].clone()
                        valid_mask[prev_zone] = 1.0 
                        invalid_mask = (valid_mask == 0)
                        constrained_predictions[t, p, invalid_mask] = -1e9
                    
                    pred_zone = torch.argmax(constrained_predictions[t, p, :]).item()
                    
                    if t > 0:
                        prev_zone = ground_truth_zones[p, t-1].item()
                        is_valid = adjacency_matrix[prev_zone, pred_zone] == 1.0 or pred_zone == prev_zone
                        if not is_valid:
                            total_violations += 1
                    total_preds += 1
            
            # --- CVAE Loss Calculation ---
            # Reshape for loss calculation: [Time, People, Zones] -> [Time * People, Zones]
            # Target shape: [People, Time] -> [Time * People]
            pred_for_loss = constrained_predictions.transpose(0, 1).reshape(-1, config.num_zones)
            target_for_loss = ground_truth_zones.reshape(-1)

            # 1. Reconstruction Loss (calculated across all agents at once)
            recon_loss = F.cross_entropy(pred_for_loss, target_for_loss)
            
            # 2. KL Divergence Loss (summed across all agents)
            kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            
            # 3. Total Loss (ELBO)
            loss = recon_loss + config.kld_weight * kld_loss
            total_loss += loss
            total_recon_loss += recon_loss.item()
            total_kld_loss += kld_loss.item()

        if total_loss > 0:
            # Backward pass
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Gradient clipping
            optimizer.step()
            
            loss_value = total_loss.item()
            recon_loss_val = total_recon_loss
            kld_loss_val = total_kld_loss
            violation_rate = (total_violations / total_preds) if total_preds > 0 else 0
            losses.append(loss_value)
            violation_rates.append(violation_rate)
            
            # Learning rate scheduling
            scheduler1.step(loss_value)
            scheduler2.step()
            
            # Early stopping check
            if (loss_value < best_loss) and (violation_rate == 0):
                best_loss = loss_value
                patience_counter = 0 # Reset patience when a better model is found
                best_violation = violation_rate
                best_model_state = {
                    'model_state_dict': model.state_dict(),
                    'config': config,
                    'processor_zone_to_id': processor.zone_to_id,
                    'processor_id_to_zone': processor.id_to_zone
                }
                # --- ROBUST ATOMIC SAVE WITH RETRY ---
                # On Windows, other processes (like antivirus) can briefly lock the destination file.
                # We will retry the replace operation a few times to handle this.
                model_path = os.path.join(save_dir, 'household_stg_node_best.pth')
                tmp_model_path = model_path + '.tmp'
                torch.save(best_model_state, tmp_model_path)
                
                retries = 5
                for i in range(retries):
                    try:
                        os.replace(tmp_model_path, model_path)
                        break # Success
                    except PermissionError:
                        if i < retries - 1:
                            time.sleep(0.1) # Wait a moment for the lock to be released
                        else:
                            # Re-raise the exception after all retries have failed.
                            raise
            else:
                patience_counter += 1
            
            if (epoch + 1) % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:4d}/{config.num_epochs} | Total Loss: {loss_value:.4f} | Recon: {recon_loss_val:.4f} | KLD: {kld_loss_val:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")
                
            # Early stopping if no improvement for too long
            if patience_counter > 1000 and epoch > 2000:
                print(f"   Early stopping at epoch {epoch+1} (no improvement for 1000 epochs)")
                break
        else:
            print(f"   Epoch {epoch+1:4d}/{config.num_epochs} | No valid training windows")
    
    # Plot and save training loss and violation rate
    plt.figure()
    plt.plot(losses, label='Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss Curve')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'training_loss_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()

    plt.figure()
    plt.plot(violation_rates, label='Physics Violation Rate')
    plt.xlabel('Epoch')
    plt.ylabel('Violation Rate')
    plt.title('Physics Violation Rate Curve')
    plt.legend()
    plt.savefig(os.path.join(plot_dir, 'violation_rate_curve.png'), dpi=200, bbox_inches='tight')
    plt.close()

    print(f"\n✅ Training completed! Final loss: {losses[-1] if losses else 'N/A'}")
    print(f"Best model saved to {os.path.join(save_dir, 'household_stg_node_best.pth')}")
    return model, data, processor, config, adjacency_matrix

def main():
    """Main function to run model training."""
    print("🏠 Household Zone Movement Prediction using STG-NODE")
    print("Based on: Spatial-temporal graph neural ODE networks")
    print("=" * 60)
    
    try:
        # Train model and save the best checkpoint
        train_model()
        
        print("\n✅ Training complete.")
        print("To evaluate the best model and see visualizations, run:")
        print("python -m src.ananke_abm.models.run.evaluate")
        
    except Exception as e:
        print(f"❌ Error occurred during training: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()