#!/usr/bin/env python3
"""
Physics-Informed Diffusion ODE Model
Combines manifold-based physics dynamics with diffusion for trajectory diversity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .physics_ode import PhysicsInformedODE


class SimplifiedPhysicsDiffusionODE(nn.Module):
    """
    Simplified Physics-Diffusion model that focuses on working dynamics
    """
    
    def __init__(self, person_attrs_dim=8, num_zones=8, embedding_dim=64, 
                 diffusion_strength=0.2):
        super().__init__()
        
        self.num_zones = num_zones
        self.embedding_dim = embedding_dim
        self.diffusion_strength = diffusion_strength
        
        # Use well-separated zone embeddings for better dynamics
        self.zone_embeddings = nn.Parameter(self._initialize_zone_embeddings())
        
        # Enhanced trajectory predictor with more capacity
        self.trajectory_net = nn.Sequential(
            nn.Linear(person_attrs_dim + 1 + num_zones, 256),  # person + time + current_zone_onehot
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_zones)  # Direct zone logits
        )
        
        # Diversity injection network
        self.diversity_net = nn.Sequential(
            nn.Linear(person_attrs_dim + 1, 128),  # person + time
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, num_zones),
            nn.Tanh()  # Bounded diversity injection
        )
        
        # Create adjacency matrix for physics constraints
        self.adjacency_matrix = None
        
        # Gumbel temperature for differentiable sampling
        self.gumbel_temperature = nn.Parameter(torch.tensor(1.0))
        
    def _initialize_zone_embeddings(self):
        """Initialize well-separated zone embeddings in a circle"""
        embeddings = torch.zeros(self.num_zones, self.embedding_dim)
        
        # Place zones in a circle in the first 2 dimensions for interpretability
        for i in range(self.num_zones):
            angle = 2 * np.pi * i / self.num_zones
            embeddings[i, 0] = 3.0 * np.cos(angle)  # Large radius for separation
            embeddings[i, 1] = 3.0 * np.sin(angle)
            
        # Add some random variation in higher dimensions
        embeddings[:, 2:] = torch.randn(self.num_zones, self.embedding_dim - 2) * 0.5
        
        return embeddings
    
    def set_adjacency_matrix(self, edge_index):
        """Create adjacency matrix from edge index"""
        self.adjacency_matrix = torch.zeros(self.num_zones, self.num_zones)
        
        for i in range(edge_index.shape[1]):
            u, v = edge_index[0, i], edge_index[1, i]
            self.adjacency_matrix[u, v] = 1.0
            self.adjacency_matrix[v, u] = 1.0
            
        # Add self-loops
        for i in range(self.num_zones):
            self.adjacency_matrix[i, i] = 1.0
    
    def apply_physics_constraints(self, zone_logits, current_zone):
        """Apply hard physics constraints to zone predictions"""
        if self.adjacency_matrix is not None:
            # Apply large penalty to non-adjacent zones
            penalty = 1000.0
            adjacency_row = self.adjacency_matrix[current_zone]
            constrained_logits = zone_logits - penalty * (1 - adjacency_row)
            return constrained_logits
        return zone_logits
    
    def forward_differentiable_trajectory(self, person_attrs, times, edge_index):
        """Generate differentiable trajectory for training using Gumbel softmax"""
        
        # Set up adjacency matrix
        self.set_adjacency_matrix(edge_index)
        
        # Initialize with soft one-hot for zone 0 (home)
        current_zone_soft = torch.zeros(self.num_zones)
        current_zone_soft[0] = 1.0
        
        trajectory_logits = []
        trajectory_zones = []
        
        for i, time in enumerate(times):
            # Normalize time to [0, 1]
            norm_time = time / times[-1] if times[-1] > 0 else 0.0
            
            # Base trajectory prediction
            traj_input = torch.cat([
                person_attrs,
                torch.tensor([norm_time]),
                current_zone_soft  # Use soft assignment
            ])
            
            base_logits = self.trajectory_net(traj_input)
            
            # Add diversity
            diversity_input = torch.cat([
                person_attrs,
                torch.tensor([norm_time])
            ])
            diversity = self.diversity_net(diversity_input)
            combined_logits = base_logits + self.diffusion_strength * diversity
            
            # Apply physics constraints to the hard current zone
            current_zone_hard = torch.argmax(current_zone_soft).item()
            constrained_logits = self.apply_physics_constraints(combined_logits, current_zone_hard)
            
            trajectory_logits.append(constrained_logits)
            
            # Use Gumbel softmax for differentiable sampling
            zone_probs = F.gumbel_softmax(constrained_logits, tau=self.gumbel_temperature, hard=False)
            
            # For hard assignment (used in next iteration)
            zone_hard = F.gumbel_softmax(constrained_logits, tau=self.gumbel_temperature, hard=True)
            current_zone_soft = zone_hard
            
            # Store hard prediction for evaluation
            trajectory_zones.append(torch.argmax(zone_hard))
        
        return torch.stack(trajectory_logits), torch.stack(trajectory_zones)
    
    def forward_single_trajectory(self, person_attrs, times, edge_index, 
                                include_diversity=True, sample_idx=0):
        """Generate a single trajectory with optional diversity"""
        
        # Set up adjacency matrix
        self.set_adjacency_matrix(edge_index)
        
        # Initialize
        current_zone = 0  # Start at home
        trajectory = []
        all_logits = []
        
        for i, time in enumerate(times):
            # Current zone one-hot
            current_zone_onehot = torch.zeros(self.num_zones)
            current_zone_onehot[current_zone] = 1.0
            
            # Normalize time to [0, 1]
            norm_time = time / times[-1] if times[-1] > 0 else 0.0
            
            # Base trajectory prediction
            traj_input = torch.cat([
                person_attrs,
                torch.tensor([norm_time]),
                current_zone_onehot
            ])
            
            base_logits = self.trajectory_net(traj_input)
            
            # Add diversity if requested
            if include_diversity:
                diversity_input = torch.cat([
                    person_attrs,
                    torch.tensor([norm_time])
                ])
                diversity = self.diversity_net(diversity_input)
                
                # Add sample-specific noise for different trajectories
                if sample_idx > 0:
                    torch.manual_seed(sample_idx * 42 + i)  # Deterministic but different per sample
                    diversity += torch.randn_like(diversity) * self.diffusion_strength
                
                combined_logits = base_logits + self.diffusion_strength * diversity
            else:
                combined_logits = base_logits
            
            # Apply physics constraints
            constrained_logits = self.apply_physics_constraints(combined_logits, current_zone)
            
            # Sample next zone
            probs = torch.softmax(constrained_logits, dim=0)
            next_zone = torch.multinomial(probs, 1).item()
            
            trajectory.append(next_zone)
            all_logits.append(constrained_logits)
            current_zone = next_zone
        
        return torch.tensor(trajectory), torch.stack(all_logits)
    
    def forward(self, person_attrs, times, zone_features, edge_index, 
                training=True, num_samples=1):
        """Main forward method"""
        
        if training:
            # During training, use differentiable trajectory
            return self.forward_differentiable_trajectory(person_attrs, times, edge_index)
        
        elif num_samples == 1:
            # Single trajectory for evaluation
            trajectory, logits = self.forward_single_trajectory(
                person_attrs, times, edge_index, 
                include_diversity=False,  # Deterministic for single eval
                sample_idx=0
            )
            return logits, trajectory
        
        else:
            # Multiple diverse trajectories
            all_trajectories = []
            all_logits = []
            
            for sample_idx in range(num_samples):
                trajectory, logits = self.forward_single_trajectory(
                    person_attrs, times, edge_index,
                    include_diversity=True,
                    sample_idx=sample_idx
                )
                all_trajectories.append(trajectory)
                all_logits.append(logits)
            
            # Return ensemble prediction
            trajectories_tensor = torch.stack(all_trajectories)
            
            # Create ensemble logits (average)
            ensemble_logits = torch.stack(all_logits).mean(dim=0)
            best_trajectory = all_trajectories[0]  # or select based on some criteria
            
            return ensemble_logits, best_trajectory
    
    def sample_diverse_trajectories(self, person_attrs, times, zone_features, edge_index, 
                                  num_samples=10):
        """Generate diverse trajectories"""
        all_trajectories = []
        
        for sample_idx in range(num_samples):
            trajectory, _ = self.forward_single_trajectory(
                person_attrs, times, edge_index,
                include_diversity=True,
                sample_idx=sample_idx
            )
            all_trajectories.append(trajectory)
        
        return torch.stack(all_trajectories), None  # positions not needed
    

class PhysicsDiffusionODE(SimplifiedPhysicsDiffusionODE):
    """Alias for backward compatibility"""
    pass


class PhysicsDiffusionTrajectoryPredictor(nn.Module):
    """
    Higher-level wrapper that handles trajectory prediction with uncertainty
    """
    
    def __init__(self, person_attrs_dim=8, num_zones=8, embedding_dim=64,
                 diffusion_strength=0.2):
        super().__init__()
        
        self.num_zones = num_zones
        self.diffusion_ode = SimplifiedPhysicsDiffusionODE(
            person_attrs_dim=person_attrs_dim,
            num_zones=num_zones,
            embedding_dim=embedding_dim,
            diffusion_strength=diffusion_strength
        )
    
    def forward(self, person_attrs, times, zone_features, edge_index, 
                training=True, num_samples=1):
        """Main forward pass"""
        return self.diffusion_ode(
            person_attrs, times, zone_features, edge_index, 
            training=training, num_samples=num_samples
        )
    
    def predict_with_uncertainty(self, person_attrs, times, zone_features, edge_index,
                               num_samples=20):
        """Predict with uncertainty estimation"""
        
        # Generate multiple trajectories
        trajectories, _ = self.diffusion_ode.sample_diverse_trajectories(
            person_attrs, times, zone_features, edge_index, num_samples
        )
        
        # Calculate uncertainties (entropy at each time point)
        uncertainties = []
        for t in range(len(times)):
            zones_at_t = trajectories[:, t]
            unique_zones, counts = torch.unique(zones_at_t, return_counts=True)
            
            if len(unique_zones) > 1:
                probs = counts.float() / num_samples
                entropy = -torch.sum(probs * torch.log(probs + 1e-8))
                uncertainties.append(entropy.item())
            else:
                uncertainties.append(0.0)  # No uncertainty if all same
        
        # Get ensemble prediction
        ensemble_logits, ensemble_prediction = self.forward(
            person_attrs, times, zone_features, edge_index,
            training=False, num_samples=1
        )
        
        return {
            'prediction': ensemble_prediction,
            'logits': ensemble_logits,
            'uncertainties': uncertainties,
            'all_trajectories': trajectories
        } 