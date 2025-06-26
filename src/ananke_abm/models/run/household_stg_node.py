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
warnings.filterwarnings('ignore')

from torchdiffeq import odeint
from ananke_abm.data_generator.mock_2p import create_two_person_training_data
from ananke_abm.data_generator.mock_locations import create_mock_zone_graph
import networkx as nx

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

class ZoneEmbedding(nn.Module):
    """Embedding layer for zones with spatial and temporal features"""
    
    def __init__(self, config: HouseholdConfig, num_zones: int):
        super().__init__()
        self.config = config
        self.num_zones = num_zones
        
        # Zone embeddings based on features
        self.zone_spatial = nn.Linear(2, config.zone_embed_dim // 2)  # x, y coords
        self.zone_features = nn.Linear(9, config.zone_embed_dim // 2)  # zone type + amenities
        
        # Temporal embeddings
        self.time_embed = nn.Linear(1, config.temporal_embed_dim)
        
        # Zone ID embedding for current location
        self.zone_id_embed = nn.Embedding(num_zones, config.zone_embed_dim)
        
    def forward(self, zone_features, current_zone_ids, time_step):
        """
        Args:
            zone_features: [batch, num_people, num_zones, features] - features for all zones
            current_zone_ids: [batch, num_people] - current zone ID for each person
            time_step: [batch, 1] - current time step
        """
        batch_size, num_people = current_zone_ids.shape
        
        # Get current zone features for each person
        current_zone_features = []
        for b in range(batch_size):
            person_features = []
            for p in range(num_people):
                zone_id = current_zone_ids[b, p].item()
                person_features.append(zone_features[b, p, zone_id])
            current_zone_features.append(torch.stack(person_features))
        current_zone_features = torch.stack(current_zone_features)  # [batch, num_people, features]
        
        # Embed spatial coordinates
        spatial_coords = current_zone_features[:, :, :2]  # x, y
        other_features = current_zone_features[:, :, 2:]  # zone types + amenities
        
        spatial_embed = self.zone_spatial(spatial_coords)
        feature_embed = self.zone_features(other_features)
        
        zone_embed = torch.cat([spatial_embed, feature_embed], dim=-1)
        
        # Add zone ID embedding
        zone_id_embed = self.zone_id_embed(current_zone_ids)
        zone_embed = zone_embed + zone_id_embed
        
        # Temporal embedding
        time_embed = self.time_embed(time_step.unsqueeze(-1))
        time_embed = time_embed.expand(batch_size, num_people, -1)
        
        # Combine embeddings
        combined_embed = torch.cat([zone_embed, time_embed], dim=-1)
        
        return combined_embed

class HouseholdDynamics(nn.Module):
    """Household movement dynamics with physics constraints"""
    
    def __init__(self, config: HouseholdConfig):
        super().__init__()
        self.config = config
        
        total_embed_dim = config.zone_embed_dim + config.temporal_embed_dim
        
        # Household interaction network (fully connected)
        self.household_gnn = nn.Linear(total_embed_dim, config.hidden_dim)
        self.household_interaction = nn.Linear(config.hidden_dim * 2, config.hidden_dim)
        
        # Movement dynamics
        self.movement_dynamics = nn.Linear(config.hidden_dim, config.hidden_dim)
        
    def forward(self, embeddings):
        """
        Args:
            embeddings: [batch, num_people, embed_dim]
        """
        batch_size, num_people, embed_dim = embeddings.shape
        
        # Apply household GNN
        h = torch.tanh(self.household_gnn(embeddings))
        
        # Household member interactions (fully connected)
        interactions = []
        for i in range(num_people):
            member_i = h[:, i:i+1, :]  # [batch, 1, hidden_dim]
            
            if num_people > 1:
                # Aggregate information from all other members
                other_members = torch.cat([h[:, :i, :], h[:, i+1:, :]], dim=1)
                
                if other_members.shape[1] > 0:
                    # Mean aggregation of other members
                    other_agg = torch.mean(other_members, dim=1, keepdim=True)
                    
                    # Interaction between member i and others
                    interaction = torch.cat([member_i, other_agg], dim=-1)
                    interaction = torch.tanh(self.household_interaction(interaction))
                else:
                    interaction = member_i
            else:
                interaction = member_i
                
            interactions.append(interaction)
        
        h_interaction = torch.cat(interactions, dim=1)  # [batch, num_people, hidden_dim]
        
        # Movement dynamics
        movement_features = torch.tanh(self.movement_dynamics(h_interaction))
        
        return movement_features

class HouseholdODEFunc(nn.Module):
    """Enhanced ODE function for continuous household movement dynamics"""
    
    def __init__(self, config: HouseholdConfig, zone_features: torch.Tensor):
        super().__init__()
        self.config = config
        self.register_buffer('zone_features', zone_features)
        
        # Embedding layers
        self.zone_embed = nn.Embedding(len(zone_features), config.zone_embed_dim)
        self.time_embed = nn.Linear(1, config.temporal_embed_dim)
        
        # Multi-layer household dynamics network
        total_dim = config.zone_embed_dim + config.person_embed_dim + config.temporal_embed_dim
        
        # Input layer
        self.input_layer = nn.Linear(total_dim, config.hidden_dim)
        
        # GRU cells for the hidden layers for better temporal state management
        self.hidden_layers = nn.ModuleList([
            nn.GRUCell(config.hidden_dim + config.temporal_embed_dim, config.hidden_dim) 
            for _ in range(config.num_layers)
        ])
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
        # Attention projection layers
        self.q_layers = nn.ModuleList([nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers)])
        self.k_layers = nn.ModuleList([nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers)])
        self.v_layers = nn.ModuleList([nn.Linear(config.hidden_dim, config.hidden_dim) for _ in range(config.num_layers)])
        
        # Dropout for regularization
        self.dropout = nn.Dropout(config.dropout)
        
        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, total_dim)
        
        # Layer normalization for stability
        self.layer_norms = nn.ModuleList([
            nn.LayerNorm(config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
    def forward(self, t, state):
        """
        Args:
            t: current time (scalar)
            state: [num_people, embed_dim] - current state embeddings for each person
        """
        num_people, embed_dim = state.shape
        
        # Add time information
        time_embed = self.time_embed(t.unsqueeze(0).unsqueeze(0))  # [1, 1, time_dim]
        time_embed = time_embed.expand(num_people, -1)  # [num_people, time_dim]
        
        # Re-combine dynamic part of state with static person features and new time
        zone_part, person_part, _ = torch.split(
            state, 
            [self.config.zone_embed_dim, self.config.person_embed_dim, self.config.temporal_embed_dim], 
            dim=-1
        )
        state_with_time = torch.cat([zone_part, person_part, time_embed], dim=-1)
        
        # Input layer
        h = torch.relu(self.input_layer(state_with_time))
        h = self.dropout(h)
        
        # Multi-layer processing with interactions
        for layer_idx in range(self.config.num_layers):
            # 1. Perform interactions on the current state `h`
            interactions = []
            for i in range(num_people):
                member_i = h[i:i+1, :]
                other_members = torch.cat([h[:i, :], h[i+1:, :]], dim=0)
                
                if other_members.shape[0] > 0:
                    # Attention mechanism
                    q = self.q_layers[layer_idx](member_i)
                    k = self.k_layers[layer_idx](other_members)
                    v = self.v_layers[layer_idx](other_members)
                    
                    d_k = q.size(-1)
                    scores = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)
                    attn_weights = F.softmax(scores, dim=-1)
                    context = torch.matmul(attn_weights, v)

                    interaction_input = torch.cat([member_i, context], dim=-1)
                    interaction = torch.relu(self.interaction_layers[layer_idx](interaction_input))
                else:
                    interaction = member_i
                    
                interactions.append(interaction)
            
            h_interacted = torch.cat(interactions, dim=0)

            # 2. Gated update (GRU) with re-injected time
            time_embed_re_injected = self.time_embed(t.unsqueeze(0).unsqueeze(0)).expand(num_people, -1)
            gru_input = torch.cat([h_interacted, time_embed_re_injected], dim=-1)
            
            # Update hidden state using GRU
            h = self.hidden_layers[layer_idx](gru_input, h)
            
            # Apply normalization and dropout
            h = self.layer_norms[layer_idx](h)
            h = self.dropout(h)
        
        # Compute derivatives with appropriate magnitude
        dstate_dt = 0.1 * self.output_proj(h)  # Larger magnitude for better learning
        
        # Add exploration noise during training
        if self.training:
            noise = torch.randn_like(dstate_dt) * self.config.exploration_noise
            dstate_dt = dstate_dt + noise
        
        return dstate_dt

class STGNodeHousehold(nn.Module):
    """STG-NODE model for household zone movement prediction"""
    
    def __init__(self, config: HouseholdConfig, num_zones: int, zone_features: torch.Tensor, person_features: torch.Tensor):
        super().__init__()
        self.config = config
        self.num_zones = num_zones
        self.register_buffer('zone_features', zone_features)
        self.register_buffer('person_features', person_features)
        
        # ODE function
        self.ode_func = HouseholdODEFunc(config, zone_features)
        
        # Initial state embedding
        self.zone_embed = nn.Embedding(num_zones, config.zone_embed_dim)
        self.person_feature_embedder = nn.Linear(config.person_feat_dim, config.person_embed_dim)
        self.time_embed = nn.Linear(1, config.temporal_embed_dim)
        
        # Output head for zone prediction
        total_dim = config.zone_embed_dim + config.person_embed_dim + config.temporal_embed_dim
        self.zone_predictor = nn.Linear(total_dim, num_zones)
        
        # Store adjacency matrix for physics constraints
        self.register_buffer('adjacency_matrix', torch.zeros(num_zones, num_zones))
        
    def get_initial_state(self, initial_zones, initial_time):
        """Get initial state embeddings including person features."""
        num_people = len(initial_zones)
        zone_embeds = self.zone_embed(initial_zones)
        time_embed = self.time_embed(initial_time.unsqueeze(0).unsqueeze(0)).expand(num_people, -1)
        person_embeds = self.person_feature_embedder(self.person_features)
        
        initial_state = torch.cat([zone_embeds, person_embeds, time_embed], dim=-1)
        return initial_state
        
    def forward(self, initial_zones, initial_time, eval_times):
        """
        Args:
            initial_zones: [num_people] - initial zone IDs for each person
            initial_time: scalar - initial time
            eval_times: [num_eval_times] - times to evaluate at
        """
        # Get initial state
        initial_state = self.get_initial_state(initial_zones, initial_time)
        
        # Solve ODE
        try:
            # Use simple euler method for stability
            solution = odeint(self.ode_func, initial_state, eval_times, method='euler')
            # solution shape: [num_eval_times, num_people, embed_dim]
        except Exception as e:
            print(f"ODE solver failed: {e}, using simple forward pass")
            # Fallback: just repeat initial state
            solution = initial_state.unsqueeze(0).expand(len(eval_times), -1, -1)
        
        # Check for NaN values from the ODE solver
        if torch.isnan(solution).any():
            print("⚠️ WARNING: NaN values detected in ODE solution. Clamping to 0.")
            solution = torch.nan_to_num(solution, nan=0.0)

        # Predict zones at each time step
        zone_logits = self.zone_predictor(solution)  # [num_eval_times, num_people, num_zones]
        
        return zone_logits
    
    def set_adjacency_matrix(self, adjacency_matrix):
        """Set the adjacency matrix for physics constraints"""
        self.adjacency_matrix = adjacency_matrix
        
    def apply_physics_constraints(self, logits, current_zones):
        """Apply HARD physics constraints to predictions - ZERO violations allowed"""
        # logits: [num_times, num_people, num_zones]
        # current_zones: [num_people] - starting zones
        
        num_times, num_people, num_zones = logits.shape
        constrained_logits = logits.clone()
        
        # Track actual zone assignments step by step
        current_zone_assignments = current_zones.clone()
        
        for t in range(num_times):
            for p in range(num_people):
                if t == 0:
                    # First timestep - use initial zone
                    continue
                else:
                    # Get the ACTUAL previous zone (from ground truth or previous prediction)
                    prev_zone = current_zone_assignments[p].item()
                    
                    # Get valid next zones
                    valid_mask = self.adjacency_matrix[prev_zone].clone()
                    valid_mask[prev_zone] = 1.0  # Can stay in same zone
                    
                    # HARD constraint: Set invalid transitions to very negative values
                    invalid_mask = (valid_mask == 0)
                    constrained_logits[t, p, invalid_mask] = -1e9
                    
                    # Update current zone assignment based on constrained prediction
                    current_zone_assignments[p] = torch.argmax(constrained_logits[t, p, :])
                
        return constrained_logits

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

        # Create the adjacency matrix from the shared zone graph
        zone_graph, _ = create_mock_zone_graph()
        adj_matrix = nx.to_numpy_array(zone_graph, nodelist=sorted(zone_graph.nodes()))
        
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
            'num_zones': num_zones,
            'num_people': num_people,
            'person_names': self.person_names,
            'person_features': person_features,
            'adjacency_matrix': torch.tensor(adj_matrix, dtype=torch.float32)
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
    
    # Create model
    model = STGNodeHousehold(config, data['num_zones'], data['zone_features'], data['person_features'])
    model.set_adjacency_matrix(adjacency_matrix)  # Set physics constraints
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
                
                # Forward pass
                raw_predictions = model(initial_zones, initial_time, eval_times)
                ground_truth_zones = person_trajectory[start_idx:end_idx]
                constrained_predictions = raw_predictions.clone()
                for t in range(1, len(eval_times)):
                    prev_zone = ground_truth_zones[t-1].item()
                    valid_mask = adjacency_matrix[prev_zone].clone()
                    valid_mask[prev_zone] = 1.0
                    invalid_mask = (valid_mask == 0)
                    constrained_predictions[t, person_idx, invalid_mask] = -1e9
                    # Track physics violations for this window
                    pred_zone = torch.argmax(constrained_predictions[t, person_idx, :]).item()
                    if valid_mask[pred_zone] == 0:
                        total_violations += 1
                    total_preds += 1
                
                # Target zones for this person
                target_zones = person_trajectory[start_idx:end_idx]
                
                # Calculate prediction loss for this person only
                person_pred = constrained_predictions[:, person_idx, :]  # [num_eval_times, num_zones]
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
                torch.save(best_model_state, os.path.join(save_dir, 'household_stg_node_best.pth'))
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

def evaluate_model(model, data, processor, config, adjacency_matrix):
    """Evaluate the trained model and return detailed results."""
    print("\n🔍 Evaluating Model Performance")
    print("=" * 50)
    
    model.eval()
    
    person_results = {}
    total_correct = 0
    total_predictions = 0
    total_violations = 0
    
    with torch.no_grad():
        trajectories_data = data['trajectories_data']
        times_data = data['times_data']
        
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            print(f"\n   Person {person_idx + 1} ({data['person_names'][person_idx]}):")
            print("   Time | True Zone | Pred Zone | Match | Physics OK")
            print("   -----|-----------|-----------|-------|----------")
            
            person_correct = 0
            person_preds = 0
            predicted_zones_list = []
            
            if len(person_trajectory) > 1:
                initial_zones = torch.tensor([person_trajectory[0].item() for _ in range(data['num_people'])])
                initial_time = person_times[0]
                eval_times = person_times
                
                raw_predictions = model(initial_zones, initial_time, eval_times)
                constrained_predictions = model.apply_physics_constraints(raw_predictions, initial_zones)
                predicted_zones = torch.argmax(constrained_predictions[:, person_idx, :], dim=-1)
                
                for t in range(len(person_trajectory)):
                    true_zone = person_trajectory[t].item()
                    pred_zone = predicted_zones[t].item()
                    predicted_zones_list.append(pred_zone)
                    
                    true_zone_name = str(processor.id_to_zone[true_zone])
                    pred_zone_name = str(processor.id_to_zone[pred_zone])
                    
                    match = "✓" if true_zone == pred_zone else "✗"
                    if true_zone == pred_zone:
                        person_correct += 1
                    
                    if t > 0:
                        prev_pred_zone = predicted_zones[t-1].item()
                        is_compliant = (adjacency_matrix[prev_pred_zone, pred_zone] == 1 or prev_pred_zone == pred_zone)
                        physics_ok = "✓" if is_compliant else "✗"
                        if not is_compliant:
                            total_violations += 1
                            print(f"      VIOLATION: {prev_pred_zone} -> {pred_zone}")
                    else:
                        physics_ok = "✓"
                    
                    person_preds += 1
                    time_val = person_times[t].item() if t < len(person_times) else t
                    print(f"   {time_val:4.1f} | {true_zone_name:9s} | {pred_zone_name:9s} | {match:5s} | {physics_ok:8s}")
            
            total_correct += person_correct
            total_predictions += person_preds
            person_results[person_idx] = {
                'accuracy': person_correct / person_preds if person_preds > 0 else 0,
                'predicted_traj': predicted_zones_list,
                'true_traj': person_trajectory.tolist(),
                'times': person_times.tolist()
            }

    overall_accuracy = total_correct / total_predictions if total_predictions > 0 else 0
    violation_rate = total_violations / total_predictions if total_predictions > 0 else 0
    
    print(f"\n📈 Overall Zone Prediction Accuracy: {overall_accuracy:.4f} ({overall_accuracy*100:.1f}%)")
    print(f"⚠️  Physics Violation Rate: {violation_rate:.4f} ({violation_rate*100:.1f}%) ({total_violations}/{total_predictions})")
    
    return {
        'overall_accuracy': overall_accuracy,
        'violation_rate': violation_rate,
        'person_results': person_results,
        'total_violations': total_violations,
        'total_predictions': total_predictions
    }

def visualize_results(eval_results, data, processor, config, plot_path=None):
    """Visualize the prediction results in a dashboard."""
    print("\n🎨 Visualizing Results Dashboard...")
    
    num_people = data['num_people']
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1])

    # Plot trajectories for first 2 people
    for i in range(min(2, num_people)):
        ax = fig.add_subplot(gs[0, i])
        person_data = eval_results['person_results'][i]
        acc = person_data['accuracy']
        
        ax.plot(person_data['times'], person_data['true_traj'], 'o-', label='True', linewidth=2, markersize=5)
        ax.plot(person_data['times'], person_data['predicted_traj'], 's--', label='Predicted', linewidth=2, markersize=5)
        ax.set_title(f"{data['person_names'][i]}'s Trajectory - Accuracy: {acc:.1%}")
        ax.set_xlabel('Time (hours)')
        ax.set_ylabel('Zone ID')
        ax.legend()
        ax.grid(True, alpha=0.4)
        ax.set_ylim(-0.5, config.num_zones - 0.5)

    # Plot accuracy bar chart
    ax_bar = fig.add_subplot(gs[1, 0])
    accuracies = [res['accuracy'] for res in eval_results['person_results'].values()]
    person_names = data['person_names']
    colors = plt.cm.viridis(np.linspace(0, 1, num_people))
    
    bars = ax_bar.bar(person_names, accuracies, color=colors)
    ax_bar.set_title('Accuracy by Person')
    ax_bar.set_ylabel('Accuracy')
    ax_bar.set_ylim(0, 1)
    for bar in bars:
        height = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width() / 2.0, height, f'{height:.1%}', ha='center', va='bottom')

    # Plot physics compliance pie chart
    ax_pie = fig.add_subplot(gs[1, 1])
    total_preds = eval_results['total_predictions']
    total_violations = eval_results['total_violations']
    compliant_preds = total_preds - total_violations
    
    sizes = [compliant_preds, total_violations]
    labels = [f'Compliant ({compliant_preds})', f'Violations ({total_violations})']
    colors = ['lightgreen', 'lightcoral']
    explode = (0, 0.1) if total_violations > 0 else (0, 0)

    ax_pie.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=90)
    ax_pie.axis('equal')
    ax_pie.set_title(f'Physics Compliance ({total_preds} total predictions)')

    fig.suptitle('GNN-ODE Evaluation Dashboard', fontsize=20)
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if plot_path:
        plt.savefig(plot_path, dpi=300)
        print(f"📊 Dashboard saved to '{plot_path}'")
    plt.show()

def main():
    """Main function"""
    print("🏠 Household Zone Movement Prediction using STG-NODE")
    print("Based on: Spatial-temporal graph neural ODE networks")
    print("=" * 60)
    
    try:
        # Train model
        model, data, processor, config, adjacency_matrix = train_model()
        
        save_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), '..', '..', 'saved_models')
        save_dir = os.path.abspath(save_dir)
        plot_dir = os.path.join(save_dir, 'plots')
        
        # Evaluate model to get detailed results
        eval_results = evaluate_model(model, data, processor, config, adjacency_matrix)
        
        # Save trajectory predictions for first 2 people to CSV
        csv_path = os.path.join(save_dir, 'predictions_sample.csv')
        records = []
        for i in range(min(2, data['num_people'])):
            person_res = eval_results['person_results'][i]
            for t_idx, time in enumerate(person_res['times']):
                records.append({
                    'person_id': i,
                    'person_name': data['person_names'][i],
                    'time': time,
                    'true_zone': processor.id_to_zone[person_res['true_traj'][t_idx]],
                    'predicted_zone': processor.id_to_zone[person_res['predicted_traj'][t_idx]]
                })
        pd.DataFrame(records).to_csv(csv_path, index=False)
        print(f"Predictions for 2 persons saved to {csv_path}")

        # Visualize results using the new dashboard
        plot_path = os.path.join(plot_dir, 'evaluation_dashboard.png')
        visualize_results(eval_results, data, processor, config, plot_path=plot_path)
        
        print(f"\n🎯 Final Results:")
        print(f"   Zone Prediction Accuracy: {eval_results['overall_accuracy']*100:.1f}%")
        print(f"   Physics Violation Rate: {eval_results['violation_rate']*100:.1f}%")
        print(f"\n💡 The model learned household movement patterns with physics constraints!")
        print(f"   High accuracy with low violation rate indicates successful learning.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()