#!/usr/bin/env python3
"""
Manifold-Based GNN-ODE for Continuous Movement Modeling
Models human movement as continuous dynamics on graph manifold with hard physics constraints.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torchdiffeq import odeint
from typing import Dict, List, Tuple, Optional
import numpy as np
from pathlib import Path
import tempfile
import shutil

from HomoGraph import HomoGraph

def safe_model_save(model, filepath):
    """Safely save model to avoid corruption - Windows compatible"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=filepath.parent, suffix='.pth')
    tmp_name = tmp_file.name
    tmp_file.close()
    
    try:
        torch.save(model.state_dict(), tmp_name)
        shutil.move(tmp_name, filepath)
        print(f"Model saved to: {filepath}")
    except Exception as e:
        if Path(tmp_name).exists():
            try:
                Path(tmp_name).unlink()
            except:
                pass
        print(f"Error saving model: {e}")
        raise

class GraphManifoldState:
    """Represents state as probability distribution over zones (on simplex)"""
    
    def __init__(self, zone_weights: torch.Tensor, num_zones: int = 8):
        self.num_zones = num_zones
        
        # Ensure weights are on probability simplex
        if zone_weights.dim() == 0:  # Single zone index
            self.weights = F.one_hot(zone_weights.long(), num_zones).float()
        else:  # Already weights
            self.weights = F.softmax(zone_weights, dim=-1)
    
    @classmethod
    def from_zone_idx(cls, zone_idx: int, num_zones: int = 8):
        """Create pure state at single zone"""
        return cls(torch.tensor(zone_idx), num_zones)
    
    @classmethod
    def from_transition(cls, start_zone: int, end_zone: int, progress: float, num_zones: int = 8):
        """Create transition state between two zones"""
        weights = torch.zeros(num_zones)
        weights[start_zone] = 1 - progress
        weights[end_zone] = progress
        return cls(weights, num_zones)
    
    def to_embedding(self, zone_embeddings: torch.Tensor) -> torch.Tensor:
        """Convert to continuous embedding as weighted combination"""
        return torch.sum(self.weights.unsqueeze(1) * zone_embeddings, dim=0)
    
    def get_dominant_zone(self) -> int:
        """Get zone with highest weight"""
        return torch.argmax(self.weights).item()
    
    def get_active_zones(self, threshold: float = 0.01) -> torch.Tensor:
        """Get zones with significant weight"""
        return torch.where(self.weights > threshold)[0]
    
    def is_pure_state(self, threshold: float = 0.99) -> bool:
        """Check if state is pure (concentrated in single zone)"""
        return torch.max(self.weights) > threshold

class ManifoldGNNPhysicsODE(nn.Module):
    """Manifold-based GNN-ODE for continuous movement on graph"""
    
    def __init__(self, location_graph: HomoGraph, person_graph: HomoGraph, 
                 embedding_dim: int = 64, num_zones: int = 8):
        super().__init__()
        
        self.location_graph = location_graph
        self.person_graph = person_graph
        self.embedding_dim = embedding_dim
        self.num_zones = num_zones
        
        # Extract schemas for person features
        person_schema = person_graph.extract_schema()
        
        # Fixed zone embeddings as basis vectors (learnable parameters)
        self.zone_embeddings = nn.Parameter(torch.randn(num_zones, embedding_dim))
        nn.init.xavier_uniform_(self.zone_embeddings)
        
        # Person embedder (simplified - no complex GNN)
        self.person_embedder = nn.Sequential(
            nn.Linear(person_schema['node_feature_dim'], embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
        # Time encoding
        self.time_encoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.ReLU(),
            nn.Linear(32, embedding_dim)
        )
        
        # Flow prediction network - predicts desired movement in zone probability space
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim * 3, 128),  # current_state + person + time
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, num_zones)  # Flows in zone probability space
        )
        
        # Precompute adjacency matrix for hard physics constraints
        self.adjacency_matrix = self._build_adjacency_matrix()
        
        # Cache person embeddings
        self.person_embeddings = self._compute_person_embeddings()
        
    def _build_adjacency_matrix(self) -> torch.Tensor:
        """Build adjacency matrix for hard physics constraints"""
        adj = torch.zeros(self.num_zones, self.num_zones)
        edge_index = self.location_graph.edge_index
        
        # Add edges (bidirectional)
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i].item(), edge_index[1, i].item()
            adj[u, v] = 1.0
            adj[v, u] = 1.0
        
        # Add self-loops (can stay in same zone)
        for i in range(self.num_zones):
            adj[i, i] = 1.0
            
        return adj
    
    def _compute_person_embeddings(self) -> torch.Tensor:
        """Compute and cache person embeddings"""
        person_data = self.person_graph.get_data()
        return self.person_embedder(person_data.x)
    
    def state_to_zone_weights(self, state: torch.Tensor) -> torch.Tensor:
        """Convert embedding state back to zone weights"""
        # Find closest representation as weighted combination of zone embeddings
        # This is an approximation - in practice, state should be tracked explicitly
        distances = torch.norm(self.zone_embeddings - state.unsqueeze(0), dim=1)
        weights = F.softmax(-distances * 10, dim=0)  # Sharp softmax
        return weights
    
    def get_allowed_flows(self, zone_weights: torch.Tensor) -> torch.Tensor:
        """Get allowed flow directions based on current zone distribution and physics"""
        # For each active zone, find allowed transitions
        allowed_flows = torch.zeros(self.num_zones)
        
        for zone_idx in range(self.num_zones):
            if zone_weights[zone_idx] > 0.01:  # If significantly present in this zone
                # Can flow to any adjacent zone
                allowed_flows += zone_weights[zone_idx] * self.adjacency_matrix[zone_idx]
        
        return allowed_flows
    
    def ode_function(self, t: torch.Tensor, state: torch.Tensor, person_idx: int) -> torch.Tensor:
        """
        ODE function for manifold dynamics with hard physics constraints
        
        Args:
            t: current time
            state: current state in embedding space [embedding_dim]
            person_idx: which person this is for
        
        Returns:
            velocity: derivative dx/dt [embedding_dim] 
        """
        
        # Handle time encoding
        if isinstance(t, torch.Tensor):
            t_val = t.item() if t.dim() == 0 else t[0].item()
        else:
            t_val = float(t)
        
        t_tensor = torch.tensor([[t_val]], dtype=torch.float32)
        t_encoded = self.time_encoder(t_tensor).squeeze(0)
        
        # CRITICAL FIX: Recompute embeddings to avoid cached graph issues
        # Don't use cached self.person_embeddings
        person_data = self.person_graph.get_data()
        person_embedding = self.person_embedder(person_data.x)[person_idx]
        
        # Predict desired flow in zone probability space
        combined_input = torch.cat([state, person_embedding, t_encoded])
        desired_flows = self.flow_net(combined_input.unsqueeze(0)).squeeze(0)
        
        # Convert current state to zone weights for physics constraints
        current_zone_weights = self.state_to_zone_weights(state)
        
        # Apply HARD physics constraints
        allowed_flows = self.get_allowed_flows(current_zone_weights)
        
        # Zero out impossible flows (HARD constraint)
        physics_constrained_flows = desired_flows * allowed_flows
        
        # Convert flows back to embedding space velocity
        # Velocity = Σ flow_i * (zone_i - current_state)
        zone_directions = self.zone_embeddings - state.unsqueeze(0)
        velocity = torch.sum(physics_constrained_flows.unsqueeze(1) * zone_directions, dim=0)
        
        # Scale for stability - much smaller to prevent explosion
        return velocity * 0.01
    
    def predict_trajectory(self, person_idx: int, times: torch.Tensor, 
                         initial_zone_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict continuous trajectory on manifold
        
        Args:
            person_idx: index of person in person graph  
            times: time points to predict [num_times]
            initial_zone_idx: starting zone index
            
        Returns:
            trajectory: continuous trajectory in embedding space [num_times, embedding_dim]
            zone_predictions: discrete zone predictions [num_times]
        """
        
        # Set initial state as pure state in starting zone
        initial_state = self.zone_embeddings[initial_zone_idx]
        
        # Define ODE function for this person
        def ode_func(t, state):
            return self.ode_function(t, state, person_idx)
        
        # Solve ODE using manifold dynamics  
        try:
            # Use euler method with retain_graph to handle gradient issues
            trajectory = odeint(
                ode_func,
                initial_state,
                times,
                method='euler',
                options={'step_size': 0.001}  # Even smaller step size for stability
            )
        except Exception as e:
            print(f"ODE solver failed: {e}, using fallback")
            # Fallback: stay at initial position
            trajectory = initial_state.unsqueeze(0).repeat(len(times), 1)
        
        # Convert continuous trajectory to zone predictions
        zone_predictions = []
        for state in trajectory:
            zone_weights = self.state_to_zone_weights(state)
            dominant_zone = torch.argmax(zone_weights).item()
            zone_predictions.append(dominant_zone)
        
        zone_predictions = torch.tensor(zone_predictions, dtype=torch.long)
        
        return trajectory, zone_predictions
    
    def generate_true_manifold_trajectory(self, zone_sequence: torch.Tensor, 
                                        times: torch.Tensor) -> torch.Tensor:
        """
        Generate true continuous trajectory from discrete zone sequence
        
        Args:
            zone_sequence: sequence of zone visits [num_times]
            times: corresponding time points [num_times]
            
        Returns:
            true_trajectory: continuous trajectory in embedding space [num_times, embedding_dim]
        """
        
        true_states = []
        
        for i in range(len(times)):
            if i == 0:
                # Start in pure state
                state = GraphManifoldState.from_zone_idx(zone_sequence[i].item(), self.num_zones)
            else:
                # Check if we're transitioning
                prev_zone = zone_sequence[i-1].item()
                curr_zone = zone_sequence[i].item()
                
                if prev_zone == curr_zone:
                    # Staying in same zone - pure state
                    state = GraphManifoldState.from_zone_idx(curr_zone, self.num_zones)
                else:
                    # Transitioning between zones
                    # Estimate transition progress based on time
                    time_in_transition = (times[i] - times[i-1]).item()
                    
                    # Simple linear transition model
                    # In practice, this could be more sophisticated
                    transition_progress = min(time_in_transition / 0.5, 1.0)  # Assume 0.5 hour transitions
                    
                    state = GraphManifoldState.from_transition(
                        prev_zone, curr_zone, transition_progress, self.num_zones
                    )
            
            # Convert to embedding
            true_embedding = state.to_embedding(self.zone_embeddings)
            true_states.append(true_embedding)
        
        return torch.stack(true_states)
    
    def compute_loss(self, trajectories: Dict) -> torch.Tensor:
        """
        Compute loss against true manifold trajectories
        
        Args:
            trajectories: dict with person trajectory data
            
        Returns:
            loss: scalar loss tensor combining trajectory and physics losses
        """
        
        total_loss = torch.tensor(0.0, requires_grad=True)
        num_trajectories = 0
        physics_violations = 0
        
        for person_name, traj_data in trajectories.items():
            person_id = traj_data["person_id"]
            times = traj_data["times"]
            observed_zones = traj_data["zones"]
            
            # Find person index in graph
            person_node_ids = self.person_graph.map_idx_to_node_id
            person_idx = person_node_ids.index(person_id)
            
            initial_zone_idx = observed_zones[0].item()
            
            # Generate true manifold trajectory
            true_trajectory = self.generate_true_manifold_trajectory(observed_zones, times)
            
            # Predict trajectory - process one at a time to avoid gradient issues
            try:
                pred_trajectory, pred_zones = self.predict_trajectory(
                    person_idx, times, initial_zone_idx
                )
                
                # Trajectory loss in embedding space
                trajectory_loss = F.mse_loss(pred_trajectory, true_trajectory)
                
                # Clamp trajectory loss to prevent explosion
                trajectory_loss = torch.clamp(trajectory_loss, 0, 50.0)
                
                # Physics violation loss (HARD penalty)
                physics_loss = self._compute_physics_violation_loss(pred_zones)
                
                # Combined loss
                person_loss = trajectory_loss + 100.0 * physics_loss  # Reduced physics penalty
                total_loss = total_loss + person_loss
                
                # Count violations for tracking
                violations = self._count_physics_violations(pred_zones)
                physics_violations += violations
                
            except Exception as e:
                print(f"Error computing loss for {person_name}: {e}")
                penalty = torch.tensor(10.0, requires_grad=True)
                total_loss = total_loss + penalty
                
            num_trajectories += 1
        
        final_loss = total_loss / num_trajectories if num_trajectories > 0 else torch.tensor(0.0, requires_grad=True)
        
        # Store physics violations for tracking
        self.last_physics_violations = physics_violations
        
        return final_loss
    
    def _compute_physics_violation_loss(self, zone_sequence: torch.Tensor) -> torch.Tensor:
        """Compute loss for physics violations"""
        violations = 0
        for i in range(len(zone_sequence) - 1):
            current_zone = zone_sequence[i].item()
            next_zone = zone_sequence[i + 1].item()
            
            # Check if transition is allowed
            if self.adjacency_matrix[current_zone, next_zone] == 0:
                violations += 1
        
        return torch.tensor(violations, dtype=torch.float32)
    
    def _count_physics_violations(self, zone_sequence: torch.Tensor) -> int:
        """Count number of physics violations"""
        violations = 0
        for i in range(len(zone_sequence) - 1):
            current_zone = zone_sequence[i].item()
            next_zone = zone_sequence[i + 1].item()
            
            if self.adjacency_matrix[current_zone, next_zone] == 0:
                violations += 1
        
        return violations

class ModelTracker:
    """Enhanced model tracker with zero-violation requirement"""
    
    def __init__(self, save_dir: Path, model_name: str = "manifold_gnn_ode"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_model_path = self.save_dir / f"{model_name}_best.pth"
        self.zero_violation_model_path = self.save_dir / f"{model_name}_zero_violations.pth"
        
        self.training_losses = []
        self.learning_rates = []
        self.physics_violations = []
        
        self.best_zero_violation_accuracy = 0.0
        
    def update(self, model, loss: float, accuracy: float = None, 
              physics_violations: int = 0, lr: float = None):
        """Update tracking with zero-violation priority"""
        
        self.training_losses.append(loss)
        self.physics_violations.append(physics_violations)
        if lr is not None:
            self.learning_rates.append(lr)
        
        # Priority 1: Zero violations with best accuracy
        if physics_violations == 0:
            if accuracy is not None and accuracy > self.best_zero_violation_accuracy:
                self.best_zero_violation_accuracy = accuracy
                print(f"New best ZERO-VIOLATION model: {accuracy:.1%} accuracy (saving)")
                safe_model_save(model, self.zero_violation_model_path)
        
        # Priority 2: Overall best loss (for tracking)
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"New best loss: {loss:.6f} (violations: {physics_violations})")
            safe_model_save(model, self.best_model_path)
        
        if accuracy is not None and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
    
    def save_training_data(self):
        """Save all training curves"""
        base_path = self.save_dir / self.model_name
        
        np.save(f"{base_path}_training_losses.npy", np.array(self.training_losses))
        np.save(f"{base_path}_physics_violations.npy", np.array(self.physics_violations))
        
        if self.learning_rates:
            np.save(f"{base_path}_learning_rates.npy", np.array(self.learning_rates))
        
        print(f"Training data saved to: {self.save_dir}")

class ManifoldGNNODETrainer:
    """Trainer for manifold-based GNN-ODE model"""
    
    def __init__(self, model: ManifoldGNNPhysicsODE, lr: float = 0.001, save_dir: str = "saved_models"):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.7, patience=15
        )
        
        self.tracker = ModelTracker(Path(save_dir), "manifold_gnn_ode")
        
    def train_step(self, trajectories: Dict) -> Tuple[float, int]:
        """Single training step returning loss and physics violations"""
        self.model.train()
        
        total_loss = 0.0
        total_violations = 0
        
        # Train on each person separately to avoid gradient issues
        for person_name, traj_data in trajectories.items():
            self.optimizer.zero_grad()
            
            # Create single-person trajectory dict
            single_traj = {person_name: traj_data}
            
            try:
                loss = self.model.compute_loss(single_traj)
                loss.backward(retain_graph=True)  # CRITICAL FIX: Retain graph for multiple backward passes
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                self.optimizer.step()
                
                total_loss += loss.item()
                total_violations += getattr(self.model, 'last_physics_violations', 0)
                
            except Exception as e:
                raise Exception(f"Training error for {person_name}: {e}")

        # Average across people
        avg_loss = total_loss / len(trajectories)
        
        return avg_loss, total_violations
    
    def train(self, trajectories: Dict, num_epochs: int = 100, verbose: bool = True):
        """Train the model with zero-violation priority"""
        
        print("🎯 Training Manifold GNN-ODE (Zero-Violation Priority)")
        
        for epoch in range(num_epochs):
            loss, physics_violations = self.train_step(trajectories)
            
            # Get current learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Evaluate accuracy periodically
            accuracy = None
            if (epoch + 1) % 10 == 0:
                results = self.evaluate(trajectories)
                accuracy = np.mean([r['accuracy'] for r in results.values()])
            
            # Update tracker
            self.tracker.update(self.model, loss, accuracy, physics_violations, current_lr)
            
            if verbose and (epoch + 1) % 10 == 0:
                viol_status = f"✅ ZERO" if physics_violations == 0 else f"❌ {physics_violations}"
                acc_str = f", Acc: {accuracy:.1%}" if accuracy is not None else ""
                print(f"Epoch {epoch+1:4d}: Loss: {loss:.6f}, Violations: {viol_status}, LR: {current_lr:.6f}{acc_str}")
            
            self.scheduler.step(loss)
            
            # Early stopping for very low loss
            if loss < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break
        
        # Save training data
        self.tracker.save_training_data()
        
        print(f"\n🏆 Training Complete!")
        print(f"   Best model: {self.tracker.best_model_path}")
        if self.tracker.best_zero_violation_accuracy > 0:
            print(f"   🎯 Zero-violation model: {self.tracker.zero_violation_model_path}")
            print(f"   🎯 Best zero-violation accuracy: {self.tracker.best_zero_violation_accuracy:.1%}")
        else:
            print(f"   ⚠️  No zero-violation models found!")
    
    def load_best_model(self, zero_violations_priority: bool = True):
        """Load best model (prioritizing zero violations)"""
        
        if zero_violations_priority and self.tracker.zero_violation_model_path.exists():
            self.model.load_state_dict(torch.load(self.tracker.zero_violation_model_path))
            print(f"✅ Loaded ZERO-VIOLATION model: {self.tracker.zero_violation_model_path}")
            return True
        elif self.tracker.best_model_path.exists():
            self.model.load_state_dict(torch.load(self.tracker.best_model_path))
            print(f"📊 Loaded best loss model: {self.tracker.best_model_path}")
            return True
        else:
            print("❌ No saved model found!")
            return False
    
    def evaluate(self, trajectories: Dict) -> Dict:
        """Evaluate model performance"""
        self.model.eval()
        
        results = {}
        
        with torch.no_grad():
            for person_name, traj_data in trajectories.items():
                person_id = traj_data["person_id"]
                times = traj_data["times"]
                observed_zones = traj_data["zones"]
                
                # Find person index
                person_node_ids = self.model.person_graph.map_idx_to_node_id
                person_idx = person_node_ids.index(person_id)
                
                # Predict
                initial_zone_idx = observed_zones[0].item()
                pred_trajectory, pred_zones = self.model.predict_trajectory(
                    person_idx, times, initial_zone_idx
                )
                
                # Compute accuracy
                accuracy = (pred_zones == observed_zones).float().mean().item()
                
                results[person_name] = {
                    "predicted_zones": pred_zones,
                    "observed_zones": observed_zones,
                    "accuracy": accuracy,
                    "times": times
                }
        
        return results 