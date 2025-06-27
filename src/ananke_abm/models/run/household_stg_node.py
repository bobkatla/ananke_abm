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

class STGNodeHousehold(nn.Module):
    """Spatially-Conditioned STG-NODE model."""
    
    def __init__(self, config: HouseholdConfig, num_zones: int, zone_features: torch.Tensor, person_features: torch.Tensor, edge_index_phys: torch.Tensor, edge_index_sem: torch.Tensor):
        super().__init__()
        self.config = config
        self.num_zones = num_zones
        self.register_buffer('static_zone_features', zone_features)
        self.register_buffer('person_features', person_features)
        self.register_buffer('edge_index_phys', edge_index_phys)
        self.register_buffer('edge_index_sem', edge_index_sem)

        # GNN Layers for spatial and semantic graphs
        self.zone_gnn_phys = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        self.zone_gnn_sem = GCNConv(zone_features.shape[1], config.zone_embed_dim)
        
        # ODE function and block
        ode_func = ODEFunc(config.hidden_dim, config.ode_hidden_dim, config.temporal_embed_dim)
        self.ode_block = ODEBlock(ode_func)
        
        # Person embedders to create initial hidden state
        self.person_feature_embedder = nn.Linear(config.person_feat_dim, config.person_embed_dim)
        # The input to the initial state mapper now includes embeddings from both GNNs
        initial_state_input_dim = config.person_embed_dim + (config.zone_embed_dim * 2)
        self.initial_state_mapper = nn.Linear(initial_state_input_dim, config.hidden_dim)
        
        # Final output predictor
        self.zone_predictor = nn.Linear(config.hidden_dim, num_zones)
        
    def get_initial_state(self, initial_zones, phys_zone_embeds, sem_zone_embeds):
        """Get initial person hidden states."""
        # Get embeddings for the specific zones people start in from both graphs
        initial_phys_context = phys_zone_embeds[initial_zones]
        initial_sem_context = sem_zone_embeds[initial_zones]
        
        person_embeds = self.person_feature_embedder(self.person_features)
        
        # Combine person features with their starting spatial and semantic context
        combined_context = torch.cat([person_embeds, initial_phys_context, initial_sem_context], dim=-1)
        initial_state = torch.relu(self.initial_state_mapper(combined_context))
        return initial_state
        
    def forward(self, initial_zones, initial_time, eval_times):
        """
        The forward pass now uses two GNNs and a generic ODEBlock.
        """
        # 1. Create spatially-aware and semantically-aware zone embeddings
        phys_zone_embeds = torch.relu(self.zone_gnn_phys(self.static_zone_features, self.edge_index_phys))
        sem_zone_embeds = torch.relu(self.zone_gnn_sem(self.static_zone_features, self.edge_index_sem))

        # 2. Get initial person hidden states
        initial_person_h = self.get_initial_state(initial_zones, phys_zone_embeds, sem_zone_embeds)
        
        # 3. Solve ODE
        # The ODE solver requires the time points to be sorted and start from the initial state's time
        ode_eval_times = torch.unique(torch.cat([initial_time.reshape(1), eval_times]))
        
        # The ODEBlock handles the temporal evolution
        person_h_solution = self.ode_block(initial_person_h, ode_eval_times)
        
        # We only need the outputs at the original eval_times
        # Find the indices in the solution that correspond to our desired eval_times
        time_indices = [torch.where(ode_eval_times == t)[0] for t in eval_times]
        solution_indices = torch.tensor([idx[0] if len(idx) > 0 else -1 for idx in time_indices]).long()
        final_solution = person_h_solution[solution_indices]

        # 4. Predict zones from the final evolved person states
        zone_logits = self.zone_predictor(final_solution)
        
        return zone_logits

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
    
    # Create model, passing the new edge_indices
    model = STGNodeHousehold(
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
        total_violations = 0
        total_preds = 0
        
        # Train on each person's trajectory
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            # Curriculum learning: start with shorter sequences, gradually increase, then focus on end-of-day
            if epoch < 1000:
                max_window_size = 5  # Short sequences first
            elif epoch < 3000:
                max_window_size = 8  # Medium sequences
            elif epoch < 7000:
                max_window_size = 12 # Full sequences
            else: # Stage 4: Focus on end-of-day sequences
                max_window_size = 12
            
            window_size = min(max_window_size, len(person_trajectory) - 1)
            if window_size < 2:
                continue
                
            # More overlapping windows for better coverage, plus end-anchored windows
            step_size = max(1, window_size // 3)
            start_indices = list(range(0, len(person_trajectory) - window_size, step_size))
            
            # In the end-of-day focus stage, add windows anchored to the end
            if epoch >= 7000:
                end_anchor_start = len(person_trajectory) - window_size
                if end_anchor_start > 0 and end_anchor_start not in start_indices:
                    start_indices.append(end_anchor_start)

            for start_idx in start_indices:
                end_idx = min(start_idx + window_size, len(person_trajectory))
                if end_idx - start_idx < 2:
                    continue
                
                # Get initial conditions
                initial_zones = torch.tensor([person_trajectory[start_idx].item() for _ in range(data['num_people'])])
                initial_time = person_times[start_idx]
                eval_times = person_times[start_idx:end_idx]
                
                # Forward pass - model handles ODEs internally
                raw_predictions = model(initial_zones, initial_time, eval_times)
                
                ground_truth_zones = person_trajectory[start_idx:end_idx]
                
                # The model's output corresponds to eval_times.
                # The shape should be [len(eval_times), num_people, num_zones]
                constrained_predictions = raw_predictions.clone()

                for t in range(len(eval_times)): 
                    if t > 0:
                        # Apply constraint based on previous ground_truth state
                        prev_zone = ground_truth_zones[t-1].item()
                        valid_mask = adjacency_matrix[prev_zone].clone()
                        valid_mask[prev_zone] = 1.0 # Can stay in the same zone
                        invalid_mask = (valid_mask == 0)
                        constrained_predictions[t, person_idx, invalid_mask] = -1e9
                    
                    # Track physics violations for this window
                    pred_zone = torch.argmax(constrained_predictions[t, person_idx, :]).item()
                    
                    if t > 0:
                        prev_zone = ground_truth_zones[t-1].item()
                        is_valid = adjacency_matrix[prev_zone, pred_zone] == 1.0 or pred_zone == prev_zone
                        if not is_valid:
                            total_violations += 1
                    total_preds += 1
                
                # Target zones for this person (matches the length of predictions)
                target_zones = person_trajectory[start_idx:end_idx]
                
                # Calculate prediction loss for this person only
                person_pred = constrained_predictions[:, person_idx, :]  # [num_eval_times, num_zones]
                
                # Ensure target and prediction lengths match
                if person_pred.shape[0] != target_zones.shape[0]:
                    # This can happen if the ODE solver returns a different number of steps
                    # This indicates a bug in how we handle time steps.
                    # For now, we will skip this batch to avoid a crash.
                    print(f"Warning: Mismatch in prediction ({person_pred.shape[0]}) and target ({target_zones.shape[0]}) lengths. Skipping batch.")
                    continue

                pred_loss = F.cross_entropy(person_pred, target_zones)
                
                # NO separate physics loss needed - constraints are HARD enforced
                # Just use the prediction loss on constrained predictions
                total_loss += pred_loss
        
        if total_loss > 0:
            # Backward pass
            total_loss.backward()
            optimizer.step()
            
            loss_value = total_loss.item()
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
            
            if (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:4d}/{config.num_epochs} | Loss: {loss_value:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")
                
            # Early stopping if no improvement for too long
            if patience_counter > 1000 and epoch > 2000:
                print(f"   Early stopping at epoch {epoch+1} (no improvement for 500 epochs)")
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