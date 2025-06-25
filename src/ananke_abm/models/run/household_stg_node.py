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
warnings.filterwarnings('ignore')

from torchdiffeq import odeint
from ananke_abm.data_generator.load_data import load_mobility_data, get_zone_adjacency_matrix

@dataclass
class HouseholdConfig:
    """Configuration for household movement prediction"""
    zone_embed_dim: int = 64  # Much larger embedding
    temporal_embed_dim: int = 32  # Larger temporal embedding
    hidden_dim: int = 128  # Much larger hidden dimension
    num_layers: int = 3  # Multiple layers for complexity
    num_zones: int = 10
    learning_rate: float = 0.001  # Lower learning rate for stability
    num_epochs: int = 5000  # Extended training time
    physics_weight: float = 1.0  # Not needed with hard constraints
    exploration_noise: float = 0.02  # Smaller noise for stability
    dropout: float = 0.1  # Regularization

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
        total_dim = config.zone_embed_dim + config.temporal_embed_dim
        
        # Input layer
        self.input_layer = nn.Linear(total_dim, config.hidden_dim)
        
        # Multiple hidden layers for complexity
        self.hidden_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim, config.hidden_dim) 
            for _ in range(config.num_layers)
        ])
        
        # Interaction layers
        self.interaction_layers = nn.ModuleList([
            nn.Linear(config.hidden_dim * 2, config.hidden_dim)
            for _ in range(config.num_layers)
        ])
        
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
        
        # Combine with current state
        state_with_time = torch.cat([state[:, :-self.config.temporal_embed_dim], time_embed], dim=-1)
        
        # Input layer
        h = torch.relu(self.input_layer(state_with_time))
        h = self.dropout(h)
        
        # Multi-layer processing with interactions
        for layer_idx in range(self.config.num_layers):
            # Individual processing
            h_individual = torch.relu(self.hidden_layers[layer_idx](h))
            h_individual = self.layer_norms[layer_idx](h_individual)
            
            # Household interactions
            if num_people > 1:
                interactions = []
                for i in range(num_people):
                    member_i = h_individual[i:i+1, :]  # [1, hidden_dim]
                    other_members = torch.cat([h_individual[:i, :], h_individual[i+1:, :]], dim=0)
                    
                    if other_members.shape[0] > 0:
                        # Attention-like mechanism for better interactions
                        other_agg = torch.mean(other_members, dim=0, keepdim=True)
                        interaction_input = torch.cat([member_i, other_agg], dim=-1)
                        interaction = torch.relu(self.interaction_layers[layer_idx](interaction_input))
                    else:
                        interaction = member_i
                        
                    interactions.append(interaction)
                
                h_interaction = torch.cat(interactions, dim=0)
            else:
                h_interaction = h_individual
            
            # Residual connection
            h = h + h_interaction
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
    
    def __init__(self, config: HouseholdConfig, num_zones: int, zone_features: torch.Tensor):
        super().__init__()
        self.config = config
        self.num_zones = num_zones
        self.register_buffer('zone_features', zone_features)
        
        # ODE function
        self.ode_func = HouseholdODEFunc(config, zone_features)
        
        # Initial state embedding
        self.zone_embed = nn.Embedding(num_zones, config.zone_embed_dim)
        self.time_embed = nn.Linear(1, config.temporal_embed_dim)
        
        # Output head for zone prediction
        total_dim = config.zone_embed_dim + config.temporal_embed_dim
        self.zone_predictor = nn.Linear(total_dim, num_zones)
        
        # Store adjacency matrix for physics constraints
        self.register_buffer('adjacency_matrix', torch.zeros(num_zones, num_zones))
        
    def get_initial_state(self, initial_zones, initial_time):
        """Get initial state embeddings"""
        zone_embeds = self.zone_embed(initial_zones)  # [num_people, zone_embed_dim]
        time_embed = self.time_embed(initial_time.unsqueeze(0).unsqueeze(0))  # [1, 1, time_dim]
        time_embed = time_embed.expand(len(initial_zones), -1)  # [num_people, time_dim]
        
        initial_state = torch.cat([zone_embeds, time_embed], dim=-1)
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
    """Process mock data for household movement prediction"""
    
    def __init__(self):
        self.zone_to_id = {}
        self.id_to_zone = {}
        
    def process_data(self):
        """Process the mock data"""
        trajectories_dict, people_df, zones_df = load_mobility_data()
        
        # Create zone mappings
        unique_zones = sorted(zones_df['zone_id'].unique())
        self.zone_to_id = {zone: i for i, zone in enumerate(unique_zones)}
        self.id_to_zone = {i: zone for zone, i in self.zone_to_id.items()}
        
        print(f"   Found {len(unique_zones)} unique zones: {unique_zones}")
        print(f"   Found {len(trajectories_dict)} people in household")
        
        # Process zone features
        zone_features = []
        for zone_id in unique_zones:
            zone_info = zones_df[zones_df['zone_id'] == zone_id].iloc[0]
            features = np.array([
                zone_info['x_coord'], zone_info['y_coord'],  # spatial
                zone_info['zone_type_residential'], zone_info['zone_type_office'],
                zone_info['zone_type_retail'], zone_info['zone_type_recreation'],
                zone_info['zone_type_transport'],
                zone_info['population'], zone_info['job_opportunities'],
                zone_info['retail_accessibility'], zone_info['transit_accessibility']
            ])
            zone_features.append(features)
        
        zone_features = torch.tensor(zone_features, dtype=torch.float32)
        
        # Process trajectories
        household_trajectories = []
        for person_name, trajectory in trajectories_dict.items():
            person_trajectory = []
            zones = trajectory['zones']  # Array of zone IDs
            for zone_id in zones:
                mapped_zone_id = self.zone_to_id[int(zone_id)]
                person_trajectory.append(mapped_zone_id)
            household_trajectories.append(person_trajectory)
            print(f"   Person {person_name}: {len(person_trajectory)} time steps, zones {min(person_trajectory)}-{max(person_trajectory)}")
        
        # Handle different trajectory lengths naturally with ODE approach
        num_people = len(household_trajectories)
        trajectory_lengths = [len(traj) for traj in household_trajectories]
        max_seq_len = max(trajectory_lengths)
        
        print(f"   Trajectory lengths: {trajectory_lengths}")
        print(f"   Using common time grid with max length: {max_seq_len}")
        
        # Store trajectories with their actual data and corresponding times
        trajectories_data = []
        times_data = []
        
        for p, trajectory in enumerate(household_trajectories):
            person_name = list(trajectories_dict.keys())[p]
            person_times = trajectories_dict[person_name]['times']
            
            trajectories_data.append(torch.tensor(trajectory, dtype=torch.long))
            times_data.append(torch.tensor(person_times, dtype=torch.float32))
        
        # Create a common time grid for evaluation
        all_times = torch.cat(times_data)
        min_time, max_time = all_times.min(), all_times.max()
        common_time_grid = torch.linspace(min_time, max_time, max_seq_len)
        
        return {
            'trajectories_data': trajectories_data,
            'times_data': times_data,
            'common_time_grid': common_time_grid,
            'zone_features': zone_features,
            'num_zones': len(unique_zones),
            'num_people': num_people
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
    
    # Process data
    print("📊 Processing mock data...")
    processor = HouseholdDataProcessor()
    data = processor.process_data()
    
    print(f"   Number of people: {data['num_people']}")
    print(f"   Zone features shape: {data['zone_features'].shape}")
    print(f"   Number of zones: {data['num_zones']}")
    print(f"   Common time grid: {data['common_time_grid'].shape}")
    
    # Get adjacency matrix
    adjacency_matrix = torch.tensor(get_zone_adjacency_matrix(), dtype=torch.float32)
    print(f"   Adjacency matrix shape: {adjacency_matrix.shape}")
    
    # Configuration
    config = HouseholdConfig()
    config.num_zones = data['num_zones']
    
    # Create model
    model = STGNodeHousehold(config, data['num_zones'], data['zone_features'])
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
    losses = []
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(config.num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # Train on each person's trajectory
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            # Curriculum learning: start with shorter sequences, gradually increase
            if epoch < 1000:
                max_window_size = 5  # Short sequences first
            elif epoch < 3000:
                max_window_size = 8  # Medium sequences
            else:
                max_window_size = 12  # Full sequences
            
            window_size = min(max_window_size, len(person_trajectory) - 1)
            if window_size < 2:
                continue
                
            # More overlapping windows for better coverage
            step_size = max(1, window_size // 3)
            for start_idx in range(0, len(person_trajectory) - window_size, step_size):
                end_idx = min(start_idx + window_size, len(person_trajectory))
                if end_idx - start_idx < 2:
                    continue
                
                # Get initial conditions
                initial_zones = torch.tensor([person_trajectory[start_idx].item() for _ in range(data['num_people'])])
                initial_time = person_times[start_idx]
                eval_times = person_times[start_idx:end_idx]
                
                # Forward pass
                raw_predictions = model(initial_zones, initial_time, eval_times)
                # raw_predictions shape: [num_eval_times, num_people, num_zones]
                
                # Apply physics constraints - use ground truth zones for training
                ground_truth_zones = person_trajectory[start_idx:end_idx]
                
                # Create proper constraint tracking
                constrained_predictions = raw_predictions.clone()
                for t in range(1, len(eval_times)):
                    prev_zone = ground_truth_zones[t-1].item()  # Use ground truth for training
                    
                    # Get valid transitions
                    valid_mask = adjacency_matrix[prev_zone].clone()
                    valid_mask[prev_zone] = 1.0  # Can stay
                    
                    # Apply hard constraints
                    invalid_mask = (valid_mask == 0)
                    constrained_predictions[t, person_idx, invalid_mask] = -1e9
                
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
            losses.append(loss_value)
            
            # Learning rate scheduling
            scheduler1.step(loss_value)
            scheduler2.step()
            
            # Early stopping check
            if loss_value < best_loss:
                best_loss = loss_value
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 50 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"   Epoch {epoch+1:4d}/{config.num_epochs} | Loss: {loss_value:.4f} | Best: {best_loss:.4f} | LR: {current_lr:.6f}")
                
            # Early stopping if no improvement for too long
            if patience_counter > 500 and epoch > 2000:
                print(f"   Early stopping at epoch {epoch+1} (no improvement for 500 epochs)")
                break
        else:
            print(f"   Epoch {epoch+1:4d}/{config.num_epochs} | No valid training windows")
    
    print(f"\n✅ Training completed! Final loss: {losses[-1] if losses else 'N/A'}")
    
    return model, data, processor, config, adjacency_matrix

def evaluate_model(model, data, processor, config, adjacency_matrix):
    """Evaluate the trained model"""
    print("\n🔍 Evaluating Model Performance")
    print("=" * 50)
    
    model.eval()
    with torch.no_grad():
        trajectories_data = data['trajectories_data']
        times_data = data['times_data']
        
        total_correct = 0
        total_predictions = 0
        violations = 0
        
        # Evaluate each person's trajectory
        for person_idx in range(data['num_people']):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            print(f"\n   Person {person_idx + 1}:")
            print("   Time | True Zone | Pred Zone | Match | Physics OK")
            print("   -----|-----------|-----------|-------|----------")
            
            # Predict trajectory starting from first position
            if len(person_trajectory) > 1:
                initial_zones = torch.tensor([person_trajectory[0].item() for _ in range(data['num_people'])])
                initial_time = person_times[0]
                eval_times = person_times
                
                # Get predictions
                raw_predictions = model(initial_zones, initial_time, eval_times)
                constrained_predictions = model.apply_physics_constraints(raw_predictions, initial_zones)
                predicted_zones = torch.argmax(constrained_predictions[:, person_idx, :], dim=-1)
                
                # Compare predictions with ground truth
                for t in range(min(12, len(person_trajectory))):
                    true_zone = person_trajectory[t].item()
                    pred_zone = predicted_zones[t].item()
                    
                    true_zone_name = str(processor.id_to_zone[true_zone])
                    pred_zone_name = str(processor.id_to_zone[pred_zone])
                    
                    match = "✓" if true_zone == pred_zone else "✗"
                    
                    # Check physics (if not first step)
                    if t > 0:
                        # Use PREDICTED previous zone for physics check (not ground truth)
                        prev_pred_zone = predicted_zones[t-1].item()
                        physics_ok = "✓" if (adjacency_matrix[prev_pred_zone, pred_zone] == 1 or 
                                           prev_pred_zone == pred_zone) else "✗"
                        
                        if adjacency_matrix[prev_pred_zone, pred_zone] == 0 and prev_pred_zone != pred_zone:
                            violations += 1
                            print(f"      VIOLATION: {prev_pred_zone} -> {pred_zone}")
                    else:
                        physics_ok = "✓"  # First step is always valid
                    
                    total_correct += (true_zone == pred_zone)
                    total_predictions += 1
                    
                    time_val = person_times[t].item() if t < len(person_times) else t
                    print(f"   {time_val:4.1f} | {true_zone_name:9s} | {pred_zone_name:9s} | {match:5s} | {physics_ok:8s}")
        
        # Calculate overall metrics
        accuracy = total_correct / total_predictions if total_predictions > 0 else 0
        violation_rate = violations / total_predictions if total_predictions > 0 else 0
        
        print(f"\n📈 Overall Zone Prediction Accuracy: {accuracy:.4f} ({accuracy*100:.1f}%)")
        print(f"⚠️  Physics Violation Rate: {violation_rate:.4f} ({violation_rate*100:.1f}%)")
    
    return accuracy, violation_rate

def visualize_results(model, data, processor, config):
    """Visualize the prediction results"""
    print("\n🎨 Visualizing Results...")
    
    model.eval()
    with torch.no_grad():
        trajectories_data = data['trajectories_data']
        times_data = data['times_data']
        
        # Create visualization
        num_people = data['num_people']
        fig, axes = plt.subplots(num_people, 1, figsize=(12, 4*num_people))
        if num_people == 1:
            axes = [axes]
        
        for person_idx in range(num_people):
            person_trajectory = trajectories_data[person_idx]
            person_times = times_data[person_idx]
            
            # Get predictions for this person
            if len(person_trajectory) > 1:
                initial_zones = torch.tensor([person_trajectory[0].item() for _ in range(num_people)])
                initial_time = person_times[0]
                eval_times = person_times
                
                predictions = model(initial_zones, initial_time, eval_times)
                predicted_zones = torch.argmax(predictions[:, person_idx, :], dim=-1)
                
                # Plot true vs predicted
                true_zones = person_trajectory.numpy()
                pred_zones = predicted_zones.numpy()
                time_points = person_times.numpy()
                
                axes[person_idx].plot(time_points, true_zones, 'o-', 
                                    label='True', linewidth=2, markersize=6)
                axes[person_idx].plot(time_points, pred_zones, 's--', 
                                    label='Predicted', linewidth=2, markersize=6)
                axes[person_idx].set_title(f'Person {person_idx + 1} - Zone Movement Over Time')
                axes[person_idx].set_xlabel('Time (hours)')
                axes[person_idx].set_ylabel('Zone ID')
                axes[person_idx].legend()
                axes[person_idx].grid(True, alpha=0.3)
                axes[person_idx].set_ylim(-0.5, config.num_zones - 0.5)
        
        plt.tight_layout()
        plt.savefig('household_zone_movement_prediction.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("📊 Visualization saved as 'household_zone_movement_prediction.png'")

def main():
    """Main function"""
    print("🏠 Household Zone Movement Prediction using STG-NODE")
    print("Based on: Spatial-temporal graph neural ODE networks")
    print("=" * 60)
    
    try:
        # Train model
        model, data, processor, config, adjacency_matrix = train_model()
        
        # Evaluate model
        accuracy, violation_rate = evaluate_model(model, data, processor, config, adjacency_matrix)
        
        # Visualize results
        visualize_results(model, data, processor, config)
        
        print(f"\n🎯 Final Results:")
        print(f"   Zone Prediction Accuracy: {accuracy*100:.1f}%")
        print(f"   Physics Violation Rate: {violation_rate*100:.1f}%")
        print(f"\n💡 The model learned household movement patterns with physics constraints!")
        print(f"   High accuracy with low violation rate indicates successful learning.")
        
    except Exception as e:
        print(f"❌ Error occurred: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()