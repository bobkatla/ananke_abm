#!/usr/bin/env python3
"""
GNN-Based Physics-Informed ODE Models
Uses Graph Neural Networks to embed locations and people, then applies ODE dynamics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, global_mean_pool, global_add_pool
from torch_geometric.data import Batch
from torchdiffeq import odeint
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from pathlib import Path
import tempfile
import shutil

from .HomoGraph import HomoGraph

def safe_model_save(model, filepath):
    """Safely save model to avoid corruption - Windows compatible"""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    # Create temporary file and ensure it's closed before moving (Windows fix)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, dir=filepath.parent, suffix='.pth')
    tmp_name = tmp_file.name
    tmp_file.close()  # Close immediately to release file handle
    
    try:
        # Save model to temporary file
        torch.save(model.state_dict(), tmp_name)
        
        # Move to final location (now safe on Windows)
        shutil.move(tmp_name, filepath)
        print(f"Model saved to: {filepath}")
        
    except Exception as e:
        # Clean up temporary file if something goes wrong
        if Path(tmp_name).exists():
            try:
                Path(tmp_name).unlink()
            except:
                pass  # Best effort cleanup
        print(f"Error saving model: {e}")
        raise

class GNNEmbedder(nn.Module):
    """GNN-based embedder for homogeneous graphs"""
    
    def __init__(self, input_dim: int, embedding_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
        # Input projection
        self.input_proj = nn.Linear(input_dim, embedding_dim)
        
        # GNN layers
        self.gnn_layers = nn.ModuleList()
        for i in range(num_layers):
            self.gnn_layers.append(GCNConv(embedding_dim, embedding_dim))
        
        # Attention layer for context-aware embeddings
        self.attention = GATConv(
            embedding_dim, 
            embedding_dim, 
            heads=4, 
            concat=False,
            dropout=0.1
        )
        
        # Output projection
        self.output_proj = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [num_nodes, input_dim] node features
            edge_index: [2, num_edges] edge connectivity
        
        Returns:
            node_embeddings: [num_nodes, embedding_dim]
        """
        
        # Project input features
        h = self.input_proj(x)
        h = F.relu(h)
        
        # Apply GNN layers
        for layer in self.gnn_layers:
            h = layer(h, edge_index)
            h = F.relu(h)
            h = F.dropout(h, training=self.training)
        
        # Apply attention
        h = self.attention(h, edge_index)
        
        # Final projection
        h = self.output_proj(h)
        
        return h

class PhysicsGNNODE(nn.Module):
    """
    Batched Multi-Agent GNN-based ODE for Household Dynamics.
    
    This model processes batches of households, where each household is a
    disjoint graph of people. It models the coupled dynamics of agents
    within each household simultaneously.
    """
    def __init__(self, num_people: int, num_zones: int, location_edge_index: torch.Tensor,
                 embedding_dim: int = 32, person_feature_dim: int = 1):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_people = num_people
        self.num_zones = num_zones
        self.register_buffer('location_edge_index', location_edge_index)

        # --- Learnable Embeddings ---
        self.location_embeddings = nn.Embedding(num_zones, embedding_dim)
        self.person_initial_embedder = nn.Linear(person_feature_dim, embedding_dim)

        # --- GNNs for Context ---
        self.person_gnn = GATConv(embedding_dim, embedding_dim, heads=2, concat=False)
        self.location_gnn = GCNConv(embedding_dim, embedding_dim)
        
        # --- ODE Core Components ---
        self.time_encoder = nn.Linear(1, embedding_dim)
        self.flow_net = nn.Sequential(
            nn.Linear(embedding_dim * 4, 128), # Social_State + Location_Context + Base_Person_State + Time
            nn.ReLU(),
            nn.Linear(128, embedding_dim)
        )

    def ode_function(self, t: torch.Tensor, X: torch.Tensor, batch: Batch) -> torch.Tensor:
        """
        Calculates dX/dt for all people in the batch.
        X shape: [num_people_in_batch, embedding_dim]
        """
        # --- 1. Social Dynamics (PersonGNN) ---
        # Model interactions between people in the same household
        socially_aware_states = self.person_gnn(X, batch.edge_index)

        # --- 2. Spatial Context (LocationGNN) ---
        # Get context for all zones based on the static location graph
        contextual_zone_embeds = self.location_gnn(self.location_embeddings.weight, self.location_edge_index)
        
        # Find the current zone for each person by finding the closest zone embedding
        current_zone_indices = torch.cdist(X, self.location_embeddings.weight).argmin(dim=1)
        
        # Gather the spatial context for each person's current location
        location_contexts = contextual_zone_embeds[current_zone_indices]

        # --- 3. Time and Attribute Injection ---
        t_encoded = self.time_encoder(t.expand(X.size(0), 1))
        base_person_states = self.person_initial_embedder(batch.x) # Use initial features as base

        # --- 4. Velocity Prediction ---
        flow_input = torch.cat([socially_aware_states, location_contexts, base_person_states, t_encoded], dim=1)
        desired_velocities = self.flow_net(flow_input)
        
        # --- 5. Physics Projection ---
        # This part is vectorized for efficiency
        # Get allowed directions for each person based on their current zone
        allowed_directions = []
        for i in range(X.size(0)):
            zone_idx = current_zone_indices[i].item()
            
            # Find neighbors of the current zone
            neighbors = self.location_edge_index[1][self.location_edge_index[0] == zone_idx]
            allowed_zone_indices = torch.cat([torch.tensor([zone_idx]), neighbors]).unique()
            
            # Get direction vectors to allowed zones
            current_pos = self.location_embeddings.weight[zone_idx]
            target_pos = self.location_embeddings.weight[allowed_zone_indices]
            directions = target_pos - current_pos.unsqueeze(0)
            norms = torch.norm(directions, p=2, dim=1, keepdim=True)
            safe_norms = torch.where(norms > 1e-6, norms, torch.ones_like(norms))
            normalized_directions = directions / safe_norms
            
            # Project desired velocity onto the basis of allowed directions
            desired_v = desired_velocities[i].unsqueeze(0)
            projections = (desired_v @ normalized_directions.t()).squeeze(0)
            weights = F.softmax(projections, dim=0)
            final_v = (weights.unsqueeze(1) * normalized_directions).sum(dim=0)
            allowed_directions.append(final_v)

        constrained_velocities = torch.stack(allowed_directions)
        
        return constrained_velocities * 0.1 # Scale for stability

    def forward(self, batch: Batch, times: torch.Tensor) -> torch.Tensor:
        """
        Solves the ODE for the entire batch of households.
        """
        # Initial state X(0) is derived from the initial zone of each person
        initial_zone_indices = batch.x.long().squeeze(1)
        initial_states = self.location_embeddings(initial_zone_indices)
        
        # Define the ODE function for the solver
        def func(t, x):
            return self.ode_function(t, x, batch)
        
        # Solve the ODE for all time steps
        trajectory = odeint(func, initial_states, times, method='euler')
        
        # trajectory shape: [num_times, num_people, embedding_dim]
        return trajectory

    def compute_loss(self, pred_trajectory: torch.Tensor, true_trajectories: Dict, 
                     person_id_to_idx: Dict) -> torch.Tensor:
        """
        Computes loss between predicted trajectories and ground truth.
        """
        total_loss = 0.0
        
        # Convert trajectory dict to a tensor ordered by person_idx
        true_zones_tensor = torch.stack([
            true_trajectories[pid]["zones"] for pid in sorted(person_id_to_idx.keys())
        ]).T # Shape: [num_times, num_people]

        # Get true embeddings for the ground truth zones
        target_embeddings = self.location_embeddings(true_zones_tensor)
        
        # MSE loss on the embeddings
        loss_mse = F.mse_loss(pred_trajectory, target_embeddings)
        
        # Cross-entropy loss on the zone classification
        # Reshape for classification: [num_times * num_people, embedding_dim]
        pred_flat = pred_trajectory.view(-1, self.embedding_dim)
        true_flat = true_zones_tensor.reshape(-1)
        
        # Calculate distances to all zone embeddings to get logits
        distances = torch.cdist(pred_flat, self.location_embeddings.weight)
        logits = -distances # Closer distance = higher logit
        loss_ce = F.cross_entropy(logits, true_flat)
        
        # Combine losses
        total_loss = loss_mse + 0.5 * loss_ce
        return total_loss

class GNNODETrainer:
    """Trainer for the Batched Multi-Agent GNN-ODE"""
    
    def __init__(self, model: PhysicsGNNODE, lr: float = 0.005, save_dir: str = "saved_models"):
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=10, min_lr=1e-6
        )
        self.tracker = ModelTracker(Path(save_dir), "gnn_ode")
        
    def train_step(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, times: torch.Tensor) -> float:
        """Single training step on a batch of households."""
        self.model.train()
        self.optimizer.zero_grad()
        
        pred_trajectory = self.model(batch, times)
        loss = self.model.compute_loss(pred_trajectory, true_trajectories, person_id_to_idx)
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()

    def train(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, 
              times: torch.Tensor, num_epochs: int = 200, verbose: bool = True):
        """Train the model"""
        
        for epoch in range(num_epochs):
            loss = self.train_step(batch, true_trajectories, person_id_to_idx, times)
            
            # Evaluate on the training data itself
            self.model.eval()
            with torch.no_grad():
                pred_trajectory = self.model(batch, times)
                
                # Calculate accuracy
                distances = torch.cdist(pred_trajectory, self.model.location_embeddings.weight)
                pred_zones = torch.argmin(distances, dim=2) # Shape: [T, N]
                
                true_zones = torch.stack([
                    true_trajectories[pid]["zones"] for pid in sorted(person_id_to_idx.keys())
                ]).T # Shape: [T, N]

                correct_preds = (pred_zones == true_zones).sum()
                total_preds = true_zones.numel()
                accuracy = (correct_preds / total_preds).item() * 100

            self.scheduler.step(loss)
            lr = self.optimizer.param_groups[0]['lr']
            self.tracker.update(self.model, loss, accuracy, lr)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss:.6f}, LR: {lr:.6f}, Accuracy: {accuracy:.2f}%")
            
            # Early stopping
            if loss < 1e-6:
                print(f"Converged at epoch {epoch+1}")
                break
        
        # Save training data at the end
        self.tracker.save_training_data()
        print(f"Best model saved to: {self.tracker.best_model_path}")
        print(f"Best accuracy achieved: {self.tracker.best_accuracy:.2f}%")
    
    def load_best_model(self):
        """Load the best saved model"""
        if self.tracker.best_model_path.exists():
            self.model.load_state_dict(torch.load(self.tracker.best_model_path))
            print(f"Loaded best model from: {self.tracker.best_model_path}")
        else:
            print("No saved model found!")
    
    def evaluate(self, batch: Batch, true_trajectories: Dict, person_id_to_idx: Dict, times: torch.Tensor) -> Dict:
        """Evaluates the model on a given batch."""
        self.model.eval()
        results = {"predictions": {}, "accuracy": 0.0}

        with torch.no_grad():
            pred_trajectory_embed = self.model(batch, times)

            distances = torch.cdist(pred_trajectory_embed, self.model.location_embeddings.weight)
            pred_zones = torch.argmin(distances, dim=2) # Shape: [T, N]

            true_zones_tensor = torch.stack([
                true_trajectories[pid]["zones"] for pid in sorted(person_id_to_idx.keys())
            ]).T

            correct_preds = (pred_zones == true_zones_tensor).sum()
            total_preds = true_zones_tensor.numel()
            accuracy = (correct_preds / total_preds).item() * 100
            results["accuracy"] = accuracy

            # Store predictions per person
            person_map = {idx: pid for pid, idx in person_id_to_idx.items()}
            for i in range(pred_zones.shape[1]): # Iterate over people
                person_id = person_map[i]
                results["predictions"][person_id] = {
                    "times": times.numpy(),
                    "pred_zones": pred_zones[:, i].numpy(),
                    "true_zones": true_zones_tensor[:, i].numpy()
                }
        
        return results

class ModelTracker:
    """Track and save best models during training"""
    
    def __init__(self, save_dir: Path, model_name: str = "gnn_ode"):
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        self.model_name = model_name
        self.best_loss = float('inf')
        self.best_accuracy = 0.0
        self.best_model_path = self.save_dir / f"{model_name}_best.pth"
        self.training_losses = []
        self.learning_rates = []
        self.accuracies = []
        
    def update(self, model, loss: float, accuracy: float = None, lr: float = None):
        """Update tracking and save if best model"""
        self.training_losses.append(loss)
        if lr is not None:
            self.learning_rates.append(lr)
        if accuracy is not None:
            self.accuracies.append(accuracy)
        
        # Save if loss improved
        if loss < self.best_loss:
            self.best_loss = loss
            print(f"New best loss: {loss:.6f} (saving model)")
            safe_model_save(model, self.best_model_path)
        
        # Track best accuracy if provided
        if accuracy is not None and accuracy > self.best_accuracy:
            self.best_accuracy = accuracy
    
    def save_training_data(self):
        """Save training curves as numpy arrays"""
        losses_path = self.save_dir / f"{self.model_name}_training_losses.npy"
        lr_path = self.save_dir / f"{self.model_name}_learning_rates.npy"
        acc_path = self.save_dir / f"{self.model_name}_accuracies.npy"
        
        np.save(losses_path, np.array(self.training_losses))
        print(f"Training losses saved to: {losses_path}")
        
        if self.learning_rates:
            np.save(lr_path, np.array(self.learning_rates))
            print(f"Learning rates saved to: {lr_path}")

        if self.accuracies:
            np.save(acc_path, np.array(self.accuracies))
            print(f"Accuracies saved to: {acc_path}")
